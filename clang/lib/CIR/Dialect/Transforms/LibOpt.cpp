//===- LibOpt.cpp - Optimize CIR raised C/C++ library idioms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

#include "StdHelpers.h"

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace mlir::cir;

namespace {

struct LibOptPass : public LibOptBase<LibOptPass> {
  LibOptPass() = default;
  void runOnOperation() override;
  void xformStdFindIntoMemchr(StdFindOp findOp);

  // Handle pass options
  struct Options {
    enum : unsigned {
      None = 0,
      RemarkTransforms = 1,
      RemarkAll = 1 << 1,
    };
    unsigned val = None;
    bool isOptionsParsed = false;

    void parseOptions(ArrayRef<StringRef> remarks) {
      if (isOptionsParsed)
        return;

      for (auto &remark : remarks) {
        val |= StringSwitch<unsigned>(remark)
                   .Case("transforms", RemarkTransforms)
                   .Case("all", RemarkAll)
                   .Default(None);
      }
      isOptionsParsed = true;
    }

    void parseOptions(LibOptPass &pass) {
      SmallVector<llvm::StringRef, 4> remarks;

      for (auto &r : pass.remarksList)
        remarks.push_back(r);

      parseOptions(remarks);
    }

    bool emitRemarkAll() { return val & RemarkAll; }
    bool emitRemarkTransforms() {
      return emitRemarkAll() || val & RemarkTransforms;
    }
  } opts;

  ///
  /// AST related
  /// -----------
  clang::ASTContext *astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace

static bool isSequentialContainer(mlir::Type t) {
  // TODO: other sequential ones, vector, dequeue, list, forward_list.
  return isStdArrayType(t);
}

static bool getIntegralNTTPAt(StructType t, size_t pos, unsigned &size) {
  auto *d =
      dyn_cast<clang::ClassTemplateSpecializationDecl>(t.getAst().getRawDecl());
  if (!d)
    return false;

  auto &templArgs = d->getTemplateArgs();
  if (pos >= templArgs.size())
    return false;

  auto arraySizeTemplateArg = templArgs[pos];
  if (arraySizeTemplateArg.getKind() != clang::TemplateArgument::Integral)
    return false;

  size = arraySizeTemplateArg.getAsIntegral().getSExtValue();
  return true;
}

static bool containerHasStaticSize(StructType t, unsigned &size) {
  // TODO: add others.
  if (!isStdArrayType(t))
    return false;

  // Get "size" from std::array<T, size>
  unsigned sizeNTTPPos = 1;
  return getIntegralNTTPAt(t, sizeNTTPPos, size);
}

void LibOptPass::xformStdFindIntoMemchr(StdFindOp findOp) {
  // template <class T>
  //  requires (sizeof(T) == 1 && is_integral_v<T>)
  // T* find(T* first, T* last, T value) {
  //   if (auto result = __builtin_memchr(first, value, last - first))
  //     return result;
  //   return last;
  // }

  auto first = findOp.getOperand(0);
  auto last = findOp.getOperand(1);
  auto value = findOp->getOperand(2);
  if (!first.getType().isa<PointerType>() || !last.getType().isa<PointerType>())
    return;

  // Transformation:
  // - 1st arg: the data pointer
  //   - Assert the Iterator is a pointer to primitive type.
  //   - Check IterBeginOp is char sized. TODO: add other types that map to
  //   char size.
  auto iterResTy = findOp.getType().dyn_cast<PointerType>();
  assert(iterResTy && "expected pointer type for iterator");
  auto underlyingDataTy = iterResTy.getPointee().dyn_cast<IntType>();
  if (!underlyingDataTy || underlyingDataTy.getWidth() != 8)
    return;

  // - 2nd arg: the pattern
  //   - Check it's a pointer type.
  //   - Load the pattern from memory
  //   - cast it to `int`.
  auto patternAddrTy = value.getType().dyn_cast<PointerType>();
  if (!patternAddrTy || patternAddrTy.getPointee() != underlyingDataTy)
    return;

  // - 3rd arg: the size
  //   - Create and pass a cir.const with NTTP value

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(findOp.getOperation());
  auto memchrOp0 =
      builder.createBitcast(first.getLoc(), first, builder.getVoidPtrTy());

  // FIXME: get datalayout based "int" instead of fixed size 4.
  auto loadPattern =
      builder.create<LoadOp>(value.getLoc(), underlyingDataTy, value);
  auto memchrOp1 = builder.createIntCast(
      loadPattern, IntType::get(builder.getContext(), 32, true));

  const auto uInt64Ty = IntType::get(builder.getContext(), 64, false);

  // Build memchr op:
  //  void *memchr(const void *s, int c, size_t n);
  auto memChr = [&] {
    if (auto iterBegin = dyn_cast<IterBeginOp>(first.getDefiningOp());
        iterBegin && isa<IterEndOp>(last.getDefiningOp())) {
      // Both operands have the same type, use iterBegin.

      // Look at this pointer to retrieve container information.
      auto thisPtr =
          iterBegin.getOperand().getType().cast<PointerType>().getPointee();
      auto containerTy = dyn_cast<StructType>(thisPtr);

      unsigned staticSize = 0;
      if (containerTy && isSequentialContainer(containerTy) &&
          containerHasStaticSize(containerTy, staticSize)) {
        return builder.create<MemChrOp>(
            findOp.getLoc(), memchrOp0, memchrOp1,
            builder.create<ConstantOp>(
                findOp.getLoc(), uInt64Ty,
                mlir::cir::IntAttr::get(uInt64Ty, staticSize)));
      }
    }
    return builder.create<MemChrOp>(
        findOp.getLoc(), memchrOp0, memchrOp1,
        builder.create<PtrDiffOp>(findOp.getLoc(), uInt64Ty, last, first));
  }();

  auto MemChrResult =
      builder.createBitcast(findOp.getLoc(), memChr.getResult(), iterResTy);

  // if (result)
  //   return result;
  // else
  // return last;
  auto NullPtr = builder.create<ConstantOp>(
      findOp.getLoc(), first.getType(), ConstPtrAttr::get(first.getType(), 0));
  auto CmpResult = builder.create<CmpOp>(
      findOp.getLoc(), BoolType::get(builder.getContext()), CmpOpKind::eq,
      NullPtr.getRes(), MemChrResult);

  auto result = builder.create<TernaryOp>(
      findOp.getLoc(), CmpResult.getResult(),
      [&](mlir::OpBuilder &ob, mlir::Location Loc) {
        ob.create<YieldOp>(Loc, last);
      },
      [&](mlir::OpBuilder &ob, mlir::Location Loc) {
        ob.create<YieldOp>(Loc, MemChrResult);
      });

  findOp.replaceAllUsesWith(result);
  findOp.erase();
}

void LibOptPass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  opts.parseOptions(*this);
  auto *op = getOperation();
  if (isa<::mlir::ModuleOp>(op))
    theModule = cast<::mlir::ModuleOp>(op);

  SmallVector<StdFindOp> stdFindToTransform;
  op->walk([&](StdFindOp findOp) { stdFindToTransform.push_back(findOp); });

  for (auto c : stdFindToTransform)
    xformStdFindIntoMemchr(c);
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}

std::unique_ptr<Pass> mlir::createLibOptPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<LibOptPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
