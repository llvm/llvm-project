//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "flang/Lower/FIRBuilder.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace Fortran::lower {

mlir::Type RuntimeStaticDescription::getMLIRType(TypeCode t,
                                                 mlir::MLIRContext *context) {
  switch (t) {
  case TypeCode::i32:
    return mlir::IntegerType::get(32, context);
  case TypeCode::i64:
    return mlir::IntegerType::get(64, context);
  case TypeCode::f32:
    return mlir::FloatType::getF32(context);
  case TypeCode::f64:
    return mlir::FloatType::getF64(context);
  // TODO need to access mapping between fe/target
  case TypeCode::c32:
    return fir::CplxType::get(context, 4);
  case TypeCode::c64:
    return fir::CplxType::get(context, 8);
  case TypeCode::boolean:
    return mlir::IntegerType::get(8, context);
  case TypeCode::charPtr:
    return fir::ReferenceType::get(fir::CharacterType::get(context, 1));
  // ! IOCookie is experimental only so far
  case TypeCode::IOCookie:
    return fir::ReferenceType::get(mlir::IntegerType::get(64, context));
  }
  llvm_unreachable("bug");
  return {};
}

mlir::FunctionType RuntimeStaticDescription::getMLIRFunctionType(
    mlir::MLIRContext *context) const {
  llvm::SmallVector<mlir::Type, 2> argMLIRTypes;
  for (const TypeCode &t : argumentTypeCodes) {
    argMLIRTypes.push_back(getMLIRType(t, context));
  }
  if (resultTypeCode.has_value()) {
    mlir::Type resMLIRType{getMLIRType(*resultTypeCode, context)};
    return mlir::FunctionType::get(argMLIRTypes, resMLIRType, context);
  }
  return mlir::FunctionType::get(argMLIRTypes, {}, context);
}

mlir::FuncOp RuntimeStaticDescription::getFuncOp(
    Fortran::lower::FirOpBuilder &builder) const {
  auto module = builder.getModule();
  auto funTy = getMLIRFunctionType(module.getContext());
  auto function = builder.addNamedFunction(symbol, funTy);
  function.setAttr("fir.runtime", builder.getUnitAttr());
  if (funTy != function.getType())
    llvm_unreachable("runtime function type mismatch");
  return function;
}

class RuntimeEntryDescription : public RuntimeStaticDescription {
public:
  using Key = RuntimeEntryCode;
  constexpr RuntimeEntryDescription(Key k, const char *s, MaybeTypeCode r,
                                    TypeCodeVector a)
      : RuntimeStaticDescription{s, r, a}, key{k} {}
  Key key;
};

static constexpr RuntimeEntryDescription runtimeTable[]{
    {RuntimeEntryCode::StopStatement, "StopStatement",
     RuntimeStaticDescription::voidTy,
     RuntimeStaticDescription::TypeCodeVector::create<
         RuntimeStaticDescription::TypeCode::i32,
         RuntimeStaticDescription::TypeCode::boolean,
         RuntimeStaticDescription::TypeCode::boolean>()},
    {RuntimeEntryCode::StopStatementText, "StopStatementText",
     RuntimeStaticDescription::voidTy,
     RuntimeStaticDescription::TypeCodeVector::create<
         RuntimeStaticDescription::TypeCode::charPtr,
         RuntimeStaticDescription::TypeCode::i32,
         RuntimeStaticDescription::TypeCode::boolean,
         RuntimeStaticDescription::TypeCode::boolean>()},
    {RuntimeEntryCode::FailImageStatement, "StopStatementText",
     RuntimeStaticDescription::voidTy,
     RuntimeStaticDescription::TypeCodeVector::create<>()},
};

static constexpr StaticMultimapView<RuntimeEntryDescription> runtimeMap{
    runtimeTable};

mlir::FuncOp genRuntimeFunction(RuntimeEntryCode code,
                                Fortran::lower::FirOpBuilder &builder) {
  auto description = runtimeMap.find(code);
  assert(description != runtimeMap.end());
  return description->getFuncOp(builder);
}

} // namespace Fortran::lower
