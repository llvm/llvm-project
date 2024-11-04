//===- ConvertProcedureDesignator.cpp -- Procedure Designator ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertProcedureDesignator.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertCall.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROps.h"

static bool areAllSymbolsInExprMapped(const Fortran::evaluate::ExtentExpr &expr,
                                      Fortran::lower::SymMap &symMap) {
  for (const auto &sym : Fortran::evaluate::CollectSymbols(expr))
    if (!symMap.lookupSymbol(sym))
      return false;
  return true;
}

fir::ExtendedValue Fortran::lower::convertProcedureDesignator(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  if (const Fortran::evaluate::SpecificIntrinsic *intrinsic =
          proc.GetSpecificIntrinsic()) {
    mlir::FunctionType signature =
        Fortran::lower::translateSignature(proc, converter);
    // Intrinsic lowering is based on the generic name, so retrieve it here in
    // case it is different from the specific name. The type of the specific
    // intrinsic is retained in the signature.
    std::string genericName =
        converter.getFoldingContext().intrinsics().GetGenericIntrinsicName(
            intrinsic->name);
    mlir::SymbolRefAttr symbolRefAttr =
        fir::getUnrestrictedIntrinsicSymbolRefAttr(builder, loc, genericName,
                                                   signature);
    mlir::Value funcPtr =
        builder.create<fir::AddrOfOp>(loc, signature, symbolRefAttr);
    return funcPtr;
  }
  const Fortran::semantics::Symbol *symbol = proc.GetSymbol();
  assert(symbol && "expected symbol in ProcedureDesignator");
  mlir::Value funcPtr;
  mlir::Value funcPtrResultLength;
  if (Fortran::semantics::IsDummy(*symbol)) {
    Fortran::lower::SymbolBox val = symMap.lookupSymbol(*symbol);
    assert(val && "Dummy procedure not in symbol map");
    funcPtr = val.getAddr();
    if (fir::isCharacterProcedureTuple(funcPtr.getType(),
                                       /*acceptRawFunc=*/false))
      std::tie(funcPtr, funcPtrResultLength) =
          fir::factory::extractCharacterProcedureTuple(builder, loc, funcPtr);
  } else {
    mlir::func::FuncOp func =
        Fortran::lower::getOrDeclareFunction(proc, converter);
    mlir::SymbolRefAttr nameAttr = builder.getSymbolRefAttr(func.getSymName());
    funcPtr =
        builder.create<fir::AddrOfOp>(loc, func.getFunctionType(), nameAttr);
  }
  if (Fortran::lower::mustPassLengthWithDummyProcedure(proc, converter)) {
    // The result length, if available here, must be propagated along the
    // procedure address so that call sites where the result length is assumed
    // can retrieve the length.
    Fortran::evaluate::DynamicType resultType = proc.GetType().value();
    if (const auto &lengthExpr = resultType.GetCharLength()) {
      // The length expression may refer to dummy argument symbols that are
      // meaningless without any actual arguments. Leave the length as
      // unknown in that case, it be resolved on the call site
      // with the actual arguments.
      if (areAllSymbolsInExprMapped(*lengthExpr, symMap)) {
        mlir::Value rawLen = fir::getBase(
            converter.genExprValue(toEvExpr(*lengthExpr), stmtCtx));
        // F2018 7.4.4.2 point 5.
        funcPtrResultLength =
            fir::factory::genMaxWithZero(builder, loc, rawLen);
      }
    }
    if (!funcPtrResultLength)
      funcPtrResultLength = builder.createIntegerConstant(
          loc, builder.getCharacterLengthType(), -1);
    return fir::CharBoxValue{funcPtr, funcPtrResultLength};
  }
  return funcPtr;
}

hlfir::EntityWithAttributes Fortran::lower::convertProcedureDesignatorToHLFIR(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ProcedureDesignator &proc,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  const auto *sym = proc.GetSymbol();
  if (sym) {
    if (sym->GetUltimate().attrs().test(Fortran::semantics::Attr::INTRINSIC))
      TODO(loc, "Procedure pointer with intrinsic target.");
    if (std::optional<fir::FortranVariableOpInterface> varDef =
            symMap.lookupVariableDefinition(*sym))
      return *varDef;
  }

  fir::ExtendedValue procExv =
      convertProcedureDesignator(loc, converter, proc, symMap, stmtCtx);
  // Directly package the procedure address as a fir.boxproc or
  // tuple<fir.boxbroc, len> so that it can be returned as a single mlir::Value.
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();

  mlir::Value funcAddr = fir::getBase(procExv);
  if (!funcAddr.getType().isa<fir::BoxProcType>()) {
    mlir::Type boxTy =
        Fortran::lower::getUntypedBoxProcType(&converter.getMLIRContext());
    if (auto host = Fortran::lower::argumentHostAssocs(converter, funcAddr))
      funcAddr = builder.create<fir::EmboxProcOp>(
          loc, boxTy, llvm::ArrayRef<mlir::Value>{funcAddr, host});
    else
      funcAddr = builder.create<fir::EmboxProcOp>(loc, boxTy, funcAddr);
  }

  mlir::Value res = procExv.match(
      [&](const fir::CharBoxValue &box) -> mlir::Value {
        mlir::Type tupleTy =
            fir::factory::getCharacterProcedureTupleType(funcAddr.getType());
        return fir::factory::createCharacterProcedureTuple(
            builder, loc, tupleTy, funcAddr, box.getLen());
      },
      [funcAddr](const auto &) { return funcAddr; });
  return hlfir::EntityWithAttributes{res};
}

mlir::Value Fortran::lower::convertProcedureDesignatorInitialTarget(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::semantics::Symbol &sym) {
  Fortran::lower::SymMap globalOpSymMap;
  Fortran::lower::StatementContext stmtCtx;
  Fortran::evaluate::ProcedureDesignator proc(sym);
  auto procVal{Fortran::lower::convertProcedureDesignatorToHLFIR(
      loc, converter, proc, globalOpSymMap, stmtCtx)};
  return fir::getBase(Fortran::lower::convertToAddress(
      loc, converter, procVal, stmtCtx, procVal.getType()));
}
