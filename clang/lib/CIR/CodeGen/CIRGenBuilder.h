//===-- CIRGenBuilder.h - CIRBuilder implementation  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H

#include "Address.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/FPEnv.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/FloatingPointMode.h"

namespace cir {

class CIRGenFunction;

class CIRGenBuilderTy : public mlir::OpBuilder {
  bool IsFPConstrained = false;
  fp::ExceptionBehavior DefaultConstrainedExcept = fp::ebStrict;
  llvm::RoundingMode DefaultConstrainedRounding = llvm::RoundingMode::Dynamic;

public:
  CIRGenBuilderTy(mlir::MLIRContext &C) : mlir::OpBuilder(&C) {}

  //
  // Floating point specific helpers
  // -------------------------------
  //

  /// Enable/Disable use of constrained floating point math. When enabled the
  /// CreateF<op>() calls instead create constrained floating point intrinsic
  /// calls. Fast math flags are unaffected by this setting.
  void setIsFPConstrained(bool IsCon) {
    if (IsCon)
      llvm_unreachable("Constrained FP NYI");
    IsFPConstrained = IsCon;
  }

  /// Query for the use of constrained floating point math
  bool getIsFPConstrained() {
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");
    return IsFPConstrained;
  }

  /// Set the exception handling to be used with constrained floating point
  void setDefaultConstrainedExcept(fp::ExceptionBehavior NewExcept) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> ExceptStr =
        convertExceptionBehaviorToStr(NewExcept);
    assert(ExceptStr && "Garbage strict exception behavior!");
#endif
    DefaultConstrainedExcept = NewExcept;
  }

  /// Set the rounding mode handling to be used with constrained floating point
  void setDefaultConstrainedRounding(llvm::RoundingMode NewRounding) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> RoundingStr =
        convertRoundingModeToStr(NewRounding);
    assert(RoundingStr && "Garbage strict rounding mode!");
#endif
    DefaultConstrainedRounding = NewRounding;
  }

  /// Get the exception handling used with constrained floating point
  fp::ExceptionBehavior getDefaultConstrainedExcept() {
    return DefaultConstrainedExcept;
  }

  /// Get the rounding mode handling used with constrained floating point
  llvm::RoundingMode getDefaultConstrainedRounding() {
    return DefaultConstrainedRounding;
  }

  //
  // Attribute helpers
  // -----------------
  //
  mlir::TypedAttr getZeroAttr(mlir::Type t) {
    return mlir::cir::ZeroAttr::get(getContext(), t);
  }

  mlir::cir::ConstArrayAttr getString(llvm::StringRef str, mlir::Type eltTy,
                                      unsigned size = 0) {
    unsigned finalSize = size ? size : str.size();
    auto arrayTy = mlir::cir::ArrayType::get(getContext(), eltTy, finalSize);
    return getConstArray(mlir::StringAttr::get(str, arrayTy), arrayTy);
  }

  mlir::cir::ConstArrayAttr getConstArray(mlir::Attribute attrs,
                                          mlir::cir::ArrayType arrayTy) {
    return mlir::cir::ConstArrayAttr::get(arrayTy, attrs);
  }

  mlir::cir::TypeInfoAttr getTypeInfo(mlir::ArrayAttr fieldsAttr) {
    llvm::SmallVector<mlir::Type, 4> members;
    for (auto &f : fieldsAttr) {
      auto gva = f.dyn_cast<mlir::cir::GlobalViewAttr>();
      assert(gva && "expected #cir.global_view attribute for element");
      members.push_back(gva.getType());
    }
    auto structType = mlir::cir::StructType::get(getContext(), members, "",
                                                 /*body=*/true);
    return mlir::cir::TypeInfoAttr::get(structType, fieldsAttr);
  }

  //
  // Type helpers
  // ------------
  //
  mlir::Type getInt8Ty() { return mlir::IntegerType::get(getContext(), 8); }
  mlir::Type getInt32Ty() { return mlir::IntegerType::get(getContext(), 32); }
  mlir::Type getInt64Ty() { return mlir::IntegerType::get(getContext(), 64); }
  mlir::cir::BoolType getBoolTy() {
    return ::mlir::cir::BoolType::get(getContext());
  }
  mlir::Type getVirtualFnPtrType([[maybe_unused]] bool isVarArg = false) {
    // FIXME: replay LLVM codegen for now, perhaps add a vtable ptr special
    // type so it's a bit more clear and C++ idiomatic.
    auto fnTy = mlir::FunctionType::get(getContext(), {}, {getInt32Ty()});
    assert(!UnimplementedFeature::isVarArg());
    return getPointerTo(getPointerTo(fnTy));
  }

  // Fetch the type representing a pointer to integer values.
  mlir::cir::PointerType getInt8PtrTy(unsigned AddrSpace = 0) {
    return mlir::cir::PointerType::get(getContext(), getInt8Ty());
  }
  mlir::cir::PointerType getInt32PtrTy(unsigned AddrSpace = 0) {
    return mlir::cir::PointerType::get(getContext(), getInt32Ty());
  }
  mlir::cir::PointerType getPointerTo(mlir::Type ty,
                                      unsigned addressSpace = 0) {
    assert(!UnimplementedFeature::addressSpace() && "NYI");
    return mlir::cir::PointerType::get(getContext(), ty);
  }

  //
  // Constant creation helpers
  // -------------------------
  //
  mlir::cir::ConstantOp getInt32(uint32_t C, mlir::Location loc) {
    auto int32Ty = getInt32Ty();
    return create<mlir::cir::ConstantOp>(loc, int32Ty,
                                         mlir::IntegerAttr::get(int32Ty, C));
  }
  mlir::Value getBool(bool state, mlir::Location loc) {
    return create<mlir::cir::ConstantOp>(
        loc, getBoolTy(), mlir::BoolAttr::get(getContext(), state));
  }

  // Creates constant nullptr for pointer type ty.
  mlir::cir::ConstantOp getNullPtr(mlir::Type ty, mlir::Location loc) {
    assert(ty.isa<mlir::cir::PointerType>() && "expected cir.ptr");
    return create<mlir::cir::ConstantOp>(
        loc, ty, mlir::cir::NullAttr::get(getContext(), ty));
  }

  // Creates constant null value for integral type ty.
  mlir::cir::ConstantOp getNullValue(mlir::Type ty, mlir::Location loc) {
    assert(ty.isa<mlir::IntegerType>() && "NYI");
    return create<mlir::cir::ConstantOp>(loc, ty,
                                         mlir::IntegerAttr::get(ty, 0));
  }

  mlir::cir::ConstantOp getZero(mlir::Location loc, mlir::Type ty) {
    // TODO: dispatch creation for primitive types.
    assert(ty.isa<mlir::cir::StructType>() && "NYI for other types");
    return create<mlir::cir::ConstantOp>(loc, ty, getZeroAttr(ty));
  }

  //
  // Block handling helpers
  // ----------------------
  //
  OpBuilder::InsertPoint getBestAllocaInsertPoint(mlir::Block *block) {
    auto lastAlloca =
        std::find_if(block->rbegin(), block->rend(), [](mlir::Operation &op) {
          return mlir::isa<mlir::cir::AllocaOp>(&op);
        });

    if (lastAlloca != block->rend())
      return OpBuilder::InsertPoint(block,
                                    ++mlir::Block::iterator(&*lastAlloca));
    return OpBuilder::InsertPoint(block, block->begin());
  };

  //
  // Operation creation helpers
  // --------------------------
  //
  mlir::Value createFPExt(mlir::Value v, mlir::Type destType) {
    if (getIsFPConstrained())
      llvm_unreachable("constrainedfp NYI");

    return create<mlir::cir::CastOp>(v.getLoc(), destType,
                                     mlir::cir::CastKind::floating, v);
  }

  /// Cast the element type of the given address to a different type,
  /// preserving information like the alignment.
  cir::Address createElementBitCast(mlir::Location loc, cir::Address addr,
                                    mlir::Type destType) {
    if (destType == addr.getElementType())
      return addr;

    auto ptrTy = getPointerTo(destType);
    return Address(createBitcast(loc, addr.getPointer(), ptrTy), destType,
                   addr.getAlignment());
  }

  mlir::Value createBitcast(mlir::Location loc, mlir::Value src,
                            mlir::Type newTy) {
    if (newTy == src.getType())
      return src;
    return create<mlir::cir::CastOp>(loc, newTy, mlir::cir::CastKind::bitcast,
                                     src);
  }

  mlir::Value createLoad(mlir::Location loc, Address addr) {
    return create<mlir::cir::LoadOp>(loc, addr.getElementType(),
                                     addr.getPointer());
  }

  mlir::cir::StoreOp createStore(mlir::Location loc, mlir::Value val,
                                 Address dst) {
    return create<mlir::cir::StoreOp>(loc, val, dst.getPointer());
  }
};

} // namespace cir

#endif
