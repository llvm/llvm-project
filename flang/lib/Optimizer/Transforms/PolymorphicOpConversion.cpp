//===-- PolymorphicOpConversion.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/TypeCode.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Runtime/derived-api.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include <mutex>

namespace fir {
#define GEN_PASS_DEF_POLYMORPHICOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;

namespace {

/// SelectTypeOp converted to an if-then-else chain
///
/// This lowers the test conditions to calls into the runtime.
class SelectTypeConv : public OpConversionPattern<fir::SelectTypeOp> {
public:
  using OpConversionPattern<fir::SelectTypeOp>::OpConversionPattern;

  SelectTypeConv(mlir::MLIRContext *ctx, std::mutex *moduleMutex)
      : mlir::OpConversionPattern<fir::SelectTypeOp>(ctx),
        moduleMutex(moduleMutex) {}

  mlir::LogicalResult
  matchAndRewrite(fir::SelectTypeOp selectType, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;

private:
  // Generate comparison of type descriptor addresses.
  mlir::Value genTypeDescCompare(mlir::Location loc, mlir::Value selector,
                                 mlir::Type ty, mlir::ModuleOp mod,
                                 mlir::PatternRewriter &rewriter) const;

  static int getTypeCode(mlir::Type ty, fir::KindMapping &kindMap);

  mlir::LogicalResult genTypeLadderStep(mlir::Location loc,
                                        mlir::Value selector,
                                        mlir::Attribute attr, mlir::Block *dest,
                                        std::optional<mlir::ValueRange> destOps,
                                        mlir::ModuleOp mod,
                                        mlir::PatternRewriter &rewriter,
                                        fir::KindMapping &kindMap) const;

  llvm::SmallSet<llvm::StringRef, 4> collectAncestors(fir::DispatchTableOp dt,
                                                      mlir::ModuleOp mod) const;

  // Mutex used to guard insertion of mlir::func::FuncOp in the module.
  std::mutex *moduleMutex;
};

/// Convert FIR structured control flow ops to CFG ops.
class PolymorphicOpConversion
    : public fir::impl::PolymorphicOpConversionBase<PolymorphicOpConversion> {
public:
  mlir::LogicalResult initialize(mlir::MLIRContext *ctx) override {
    moduleMutex = new std::mutex();
    return mlir::success();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<SelectTypeConv>(context, moduleMutex);
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::AffineDialect, mlir::cf::ControlFlowDialect,
                           FIROpsDialect, mlir::func::FuncDialect>();

    // apply the patterns
    target.addIllegalOp<SelectTypeOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to CFG\n");
      signalPassFailure();
    }
  }

private:
  std::mutex *moduleMutex;
};
} // namespace

mlir::LogicalResult SelectTypeConv::matchAndRewrite(
    fir::SelectTypeOp selectType, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  auto typeGuards = selectType.getCases();
  unsigned typeGuardNum = typeGuards.size();
  auto selector = selectType.getSelector();
  auto loc = selectType.getLoc();
  auto mod = selectType.getOperation()->getParentOfType<mlir::ModuleOp>();
  fir::KindMapping kindMap = fir::getKindMapping(mod);

  // Order type guards so the condition and branches are done to respect the
  // Execution of SELECT TYPE construct as described in the Fortran 2018
  // standard 11.1.11.2 point 4.
  // 1. If a TYPE IS type guard statement matches the selector, the block
  //    following that statement is executed.
  // 2. Otherwise, if exactly one CLASS IS type guard statement matches the
  //    selector, the block following that statement is executed.
  // 3. Otherwise, if several CLASS IS type guard statements match the
  //    selector, one of these statements will inevitably specify a type that
  //    is an extension of all the types specified in the others; the block
  //    following that statement is executed.
  // 4. Otherwise, if there is a CLASS DEFAULT type guard statement, the block
  //    following that statement is executed.
  // 5. Otherwise, no block is executed.

  llvm::SmallVector<unsigned> orderedTypeGuards;
  llvm::SmallVector<unsigned> orderedClassIsGuards;
  unsigned defaultGuard = typeGuardNum - 1;

  // The following loop go through the type guards in the fir.select_type
  // operation and sort them into two lists.
  // - All the TYPE IS type guard are added in order to the orderedTypeGuards
  //   list. This list is used at the end to generate the if-then-else ladder.
  // - CLASS IS type guard are added in a separate list. If a CLASS IS type
  //   guard type extends a type already present, the type guard is inserted
  //   before in the list to respect point 3. above. Otherwise it is just
  //   added in order at the end.
  for (unsigned t = 0; t < typeGuardNum; ++t) {
    if (auto a = typeGuards[t].dyn_cast<fir::ExactTypeAttr>()) {
      orderedTypeGuards.push_back(t);
      continue;
    }

    if (auto a = typeGuards[t].dyn_cast<fir::SubclassAttr>()) {
      if (auto recTy = a.getType().dyn_cast<fir::RecordType>()) {
        auto dt = mod.lookupSymbol<fir::DispatchTableOp>(recTy.getName());
        assert(dt && "dispatch table not found");
        llvm::SmallSet<llvm::StringRef, 4> ancestors =
            collectAncestors(dt, mod);
        if (!ancestors.empty()) {
          auto it = orderedClassIsGuards.begin();
          while (it != orderedClassIsGuards.end()) {
            fir::SubclassAttr sAttr =
                typeGuards[*it].dyn_cast<fir::SubclassAttr>();
            if (auto ty = sAttr.getType().dyn_cast<fir::RecordType>()) {
              if (ancestors.contains(ty.getName()))
                break;
            }
            ++it;
          }
          if (it != orderedClassIsGuards.end()) {
            // Parent type is present so place it before.
            orderedClassIsGuards.insert(it, t);
            continue;
          }
        }
      }
      orderedClassIsGuards.push_back(t);
    }
  }
  orderedTypeGuards.append(orderedClassIsGuards);
  orderedTypeGuards.push_back(defaultGuard);
  assert(orderedTypeGuards.size() == typeGuardNum &&
         "ordered type guard size doesn't match number of type guards");

  for (unsigned idx : orderedTypeGuards) {
    auto *dest = selectType.getSuccessor(idx);
    std::optional<mlir::ValueRange> destOps =
        selectType.getSuccessorOperands(operands, idx);
    if (typeGuards[idx].dyn_cast<mlir::UnitAttr>())
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(selectType, dest);
    else if (mlir::failed(genTypeLadderStep(loc, selector, typeGuards[idx],
                                            dest, destOps, mod, rewriter,
                                            kindMap)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult SelectTypeConv::genTypeLadderStep(
    mlir::Location loc, mlir::Value selector, mlir::Attribute attr,
    mlir::Block *dest, std::optional<mlir::ValueRange> destOps,
    mlir::ModuleOp mod, mlir::PatternRewriter &rewriter,
    fir::KindMapping &kindMap) const {
  mlir::Value cmp;
  // TYPE IS type guard comparison are all done inlined.
  if (auto a = attr.dyn_cast<fir::ExactTypeAttr>()) {
    if (fir::isa_trivial(a.getType()) ||
        a.getType().isa<fir::CharacterType>()) {
      // For type guard statement with Intrinsic type spec the type code of
      // the descriptor is compared.
      int code = getTypeCode(a.getType(), kindMap);
      if (code == 0)
        return mlir::emitError(loc)
               << "type code unavailable for " << a.getType();
      mlir::Value typeCode = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getI8IntegerAttr(code));
      mlir::Value selectorTypeCode = rewriter.create<fir::BoxTypeCodeOp>(
          loc, rewriter.getI8Type(), selector);
      cmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, selectorTypeCode, typeCode);
    } else {
      // Flang inline the kind parameter in the type descriptor so we can
      // directly check if the type descriptor addresses are identical for
      // the TYPE IS type guard statement.
      mlir::Value res =
          genTypeDescCompare(loc, selector, a.getType(), mod, rewriter);
      if (!res)
        return mlir::failure();
      cmp = res;
    }
    // CLASS IS type guard statement is done with a runtime call.
  } else if (auto a = attr.dyn_cast<fir::SubclassAttr>()) {
    // Retrieve the type descriptor from the type guard statement record type.
    assert(a.getType().isa<fir::RecordType>() && "expect fir.record type");
    fir::RecordType recTy = a.getType().dyn_cast<fir::RecordType>();
    std::string typeDescName =
        fir::NameUniquer::getTypeDescriptorName(recTy.getName());
    auto typeDescGlobal = mod.lookupSymbol<fir::GlobalOp>(typeDescName);
    auto typeDescAddr = rewriter.create<fir::AddrOfOp>(
        loc, fir::ReferenceType::get(typeDescGlobal.getType()),
        typeDescGlobal.getSymbol());
    mlir::Type typeDescTy = ReferenceType::get(rewriter.getNoneType());
    mlir::Value typeDesc =
        rewriter.create<ConvertOp>(loc, typeDescTy, typeDescAddr);

    // Prepare the selector descriptor for the runtime call.
    mlir::Type descNoneTy = fir::BoxType::get(rewriter.getNoneType());
    mlir::Value descSelector =
        rewriter.create<ConvertOp>(loc, descNoneTy, selector);

    // Generate runtime call.
    llvm::StringRef fctName = RTNAME_STRING(ClassIs);
    mlir::func::FuncOp callee;
    {
      // Since conversion is done in parallel for each fir.select_type
      // operation, the runtime function insertion must be threadsafe.
      std::lock_guard<std::mutex> lock(*moduleMutex);
      callee =
          fir::createFuncOp(rewriter.getUnknownLoc(), mod, fctName,
                            rewriter.getFunctionType({descNoneTy, typeDescTy},
                                                     rewriter.getI1Type()));
    }
    cmp = rewriter
              .create<fir::CallOp>(loc, callee,
                                   mlir::ValueRange{descSelector, typeDesc})
              .getResult(0);
  }

  auto *thisBlock = rewriter.getInsertionBlock();
  auto *newBlock =
      rewriter.createBlock(dest->getParent(), mlir::Region::iterator(dest));
  rewriter.setInsertionPointToEnd(thisBlock);
  if (destOps.has_value())
    rewriter.create<mlir::cf::CondBranchOp>(loc, cmp, dest, destOps.value(),
                                            newBlock, std::nullopt);
  else
    rewriter.create<mlir::cf::CondBranchOp>(loc, cmp, dest, newBlock);
  rewriter.setInsertionPointToEnd(newBlock);
  return mlir::success();
}

// Generate comparison of type descriptor addresses.
mlir::Value
SelectTypeConv::genTypeDescCompare(mlir::Location loc, mlir::Value selector,
                                   mlir::Type ty, mlir::ModuleOp mod,
                                   mlir::PatternRewriter &rewriter) const {
  assert(ty.isa<fir::RecordType>() && "expect fir.record type");
  fir::RecordType recTy = ty.dyn_cast<fir::RecordType>();
  std::string typeDescName =
      fir::NameUniquer::getTypeDescriptorName(recTy.getName());
  auto typeDescGlobal = mod.lookupSymbol<fir::GlobalOp>(typeDescName);
  if (!typeDescGlobal)
    return {};
  auto typeDescAddr = rewriter.create<fir::AddrOfOp>(
      loc, fir::ReferenceType::get(typeDescGlobal.getType()),
      typeDescGlobal.getSymbol());
  auto intPtrTy = rewriter.getIndexType();
  mlir::Type tdescType =
      fir::TypeDescType::get(mlir::NoneType::get(rewriter.getContext()));
  mlir::Value selectorTdescAddr =
      rewriter.create<fir::BoxTypeDescOp>(loc, tdescType, selector);
  auto typeDescInt =
      rewriter.create<fir::ConvertOp>(loc, intPtrTy, typeDescAddr);
  auto selectorTdescInt =
      rewriter.create<fir::ConvertOp>(loc, intPtrTy, selectorTdescAddr);
  return rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, typeDescInt, selectorTdescInt);
}

int SelectTypeConv::getTypeCode(mlir::Type ty, fir::KindMapping &kindMap) {
  if (auto intTy = ty.dyn_cast<mlir::IntegerType>())
    return fir::integerBitsToTypeCode(intTy.getWidth());
  if (auto floatTy = ty.dyn_cast<mlir::FloatType>())
    return fir::realBitsToTypeCode(floatTy.getWidth());
  if (auto logicalTy = ty.dyn_cast<fir::LogicalType>())
    return fir::logicalBitsToTypeCode(
        kindMap.getLogicalBitsize(logicalTy.getFKind()));
  if (fir::isa_complex(ty)) {
    if (auto cmplxTy = ty.dyn_cast<mlir::ComplexType>())
      return fir::complexBitsToTypeCode(
          cmplxTy.getElementType().cast<mlir::FloatType>().getWidth());
    auto cmplxTy = ty.cast<fir::ComplexType>();
    return fir::complexBitsToTypeCode(
        kindMap.getRealBitsize(cmplxTy.getFKind()));
  }
  if (auto charTy = ty.dyn_cast<fir::CharacterType>())
    return fir::characterBitsToTypeCode(
        kindMap.getCharacterBitsize(charTy.getFKind()));
  return 0;
}

llvm::SmallSet<llvm::StringRef, 4>
SelectTypeConv::collectAncestors(fir::DispatchTableOp dt,
                                 mlir::ModuleOp mod) const {
  llvm::SmallSet<llvm::StringRef, 4> ancestors;
  if (!dt.getParent().has_value())
    return ancestors;
  while (dt.getParent().has_value()) {
    ancestors.insert(*dt.getParent());
    dt = mod.lookupSymbol<fir::DispatchTableOp>(*dt.getParent());
  }
  return ancestors;
}

std::unique_ptr<mlir::Pass> fir::createPolymorphicOpConversionPass() {
  return std::make_unique<PolymorphicOpConversion>();
}
