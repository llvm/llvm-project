//===--- LowerTypes.cpp - Type translation to target-specific types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenTypes.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "LowerTypes.h"
#include "CIRToCIRArgMapping.h"
#include "LowerModule.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace ::mlir::cir;

using ABIArgInfo = ::cir::ABIArgInfo;

unsigned LowerTypes::clangCallConvToLLVMCallConv(clang::CallingConv CC) {
  switch (CC) {
  case clang::CC_C:
    return llvm::CallingConv::C;
  default:
    llvm_unreachable("calling convention NYI");
  }
}

LowerTypes::LowerTypes(LowerModule &LM, StringRef DLString)
    : LM(LM), context(LM.getContext()), Target(LM.getTarget()),
      CXXABI(LM.getCXXABI()),
      TheABIInfo(LM.getTargetLoweringInfo().getABIInfo()),
      mlirContext(LM.getMLIRContext()), DL(LM.getModule()) {}

/// Return the ABI-specific function type for a CIR function type.
FuncType LowerTypes::getFunctionType(const LowerFunctionInfo &FI) {

  mlir::Type resultType = {};
  const ::cir::ABIArgInfo &retAI = FI.getReturnInfo();
  switch (retAI.getKind()) {
  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    resultType = retAI.getCoerceToType();
    break;
  case ::cir::ABIArgInfo::Ignore:
    resultType = VoidType::get(getMLIRContext());
    break;
  default:
    llvm_unreachable("Missing ABIArgInfo::Kind");
  }

  CIRToCIRArgMapping IRFunctionArgs(getContext(), FI, true);
  SmallVector<Type, 8> ArgTypes(IRFunctionArgs.totalIRArgs());

  // Add type for sret argument.
  assert(!::cir::MissingFeatures::sretArgs());

  // Add type for inalloca argument.
  assert(!::cir::MissingFeatures::inallocaArgs());

  // Add in all of the required arguments.
  unsigned ArgNo = 0;
  LowerFunctionInfo::const_arg_iterator it = FI.arg_begin(),
                                        ie = it + FI.getNumRequiredArgs();
  for (; it != ie; ++it, ++ArgNo) {
    const ABIArgInfo &ArgInfo = it->info;

    assert(!::cir::MissingFeatures::argumentPadding());

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      Type argType = ArgInfo.getCoerceToType();
      StructType st = dyn_cast<StructType>(argType);
      if (st && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        assert(NumIRArgs == st.getNumElements());
        for (unsigned i = 0, e = st.getNumElements(); i != e; ++i)
          ArgTypes[FirstIRArg + i] = st.getMembers()[i];
      } else {
        assert(NumIRArgs == 1);
        ArgTypes[FirstIRArg] = argType;
      }
      break;
    }
    default:
      llvm_unreachable("Missing ABIArgInfo::Kind");
    }
  }

  return FuncType::get(getMLIRContext(), ArgTypes, resultType, FI.isVariadic());
}

/// Convert a CIR type to its ABI-specific default form.
mlir::Type LowerTypes::convertType(Type T) {
  /// NOTE(cir): It the original codegen this method is used to get the default
  /// LLVM IR representation for a given AST type. When a the ABI-specific
  /// function info sets a nullptr for a return or argument type, the default
  /// type given by this method is used. In CIR's case, its types are already
  /// supposed to be ABI-specific, so this method is not really useful here.
  /// It's kept here for codegen parity's sake.

  // Certain CIR types are already ABI-specific, so we just return them.
  if (isa<BoolType, IntType, SingleType, DoubleType>(T)) {
    return T;
  }

  llvm::outs() << "Missing default ABI-specific type for " << T << "\n";
  llvm_unreachable("NYI");
}
