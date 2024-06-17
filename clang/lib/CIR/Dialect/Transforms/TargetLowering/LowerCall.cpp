#include "LowerCall.h"
#include "LowerFunctionInfo.h"
#include "LowerTypes.h"
#include "clang/CIR/FnInfoOpts.h"

using namespace mlir;
using namespace mlir::cir;

using FnInfoOpts = ::cir::FnInfoOpts;

namespace {

/// Arrange a call as unto a free function, except possibly with an
/// additional number of formal parameters considered required.
const LowerFunctionInfo &
arrangeFreeFunctionLikeCall(LowerTypes &LT, LowerModule &LM,
                            const OperandRange &args, const FuncType fnType,
                            unsigned numExtraRequiredArgs, bool chainCall) {
  assert(args.size() >= numExtraRequiredArgs);

  assert(!::cir::MissingFeatures::extParamInfo());

  // In most cases, there are no optional arguments.
  RequiredArgs required = RequiredArgs::All;

  // If we have a variadic prototype, the required arguments are the
  // extra prefix plus the arguments in the prototype.
  // FIXME(cir): Properly check if function is no-proto.
  if (/*IsPrototypedFunction=*/true) {
    if (fnType.isVarArg())
      llvm_unreachable("NYI");

    if (::cir::MissingFeatures::extParamInfo())
      llvm_unreachable("NYI");
  }

  // TODO(cir): There's some CC stuff related to no-proto functions here, but
  // I'm skipping it since it requires CodeGen info. Maybe we can embbed this
  // information in the FuncOp during CIRGen.

  assert(!::cir::MissingFeatures::chainCall() && !chainCall && "NYI");
  FnInfoOpts opts = chainCall ? FnInfoOpts::IsChainCall : FnInfoOpts::None;
  return LT.arrangeLLVMFunctionInfo(fnType.getReturnType(), opts,
                                    fnType.getInputs(), required);
}

} // namespace

/// Figure out the rules for calling a function with the given formal
/// type using the given arguments.  The arguments are necessary
/// because the function might be unprototyped, in which case it's
/// target-dependent in crazy ways.
const LowerFunctionInfo &
LowerTypes::arrangeFreeFunctionCall(const OperandRange args,
                                    const FuncType fnType, bool chainCall) {
  return arrangeFreeFunctionLikeCall(*this, LM, args, fnType, chainCall ? 1 : 0,
                                     chainCall);
}

/// Arrange the argument and result information for an abstract value
/// of a given function type.  This is the method which all of the
/// above functions ultimately defer to.
///
/// \param resultType - ABI-agnostic CIR result type.
/// \param opts - Options to control the arrangement.
/// \param argTypes - ABI-agnostic CIR argument types.
/// \param required - Information about required/optional arguments.
const LowerFunctionInfo &
LowerTypes::arrangeLLVMFunctionInfo(Type resultType, FnInfoOpts opts,
                                    ArrayRef<Type> argTypes,
                                    RequiredArgs required) {
  assert(!::cir::MissingFeatures::qualifiedTypes());

  LowerFunctionInfo *FI = nullptr;

  // FIXME(cir): Allow user-defined CCs (e.g. __attribute__((vectorcall))).
  assert(!::cir::MissingFeatures::extParamInfo());
  unsigned CC = clangCallConvToLLVMCallConv(clang::CallingConv::CC_C);

  // Construct the function info. We co-allocate the ArgInfos.
  // NOTE(cir): This initial function info might hold incorrect data.
  FI = LowerFunctionInfo::create(
      CC, /*isInstanceMethod=*/false, /*isChainCall=*/false,
      /*isDelegateCall=*/false, resultType, argTypes, required);

  // Compute ABI information.
  if (CC == llvm::CallingConv::SPIR_KERNEL) {
    llvm_unreachable("NYI");
  } else if (::cir::MissingFeatures::extParamInfo()) {
    llvm_unreachable("NYI");
  } else {
    // NOTE(cir): This corects the initial function info data.
    getABIInfo().computeInfo(*FI); // FIXME(cir): Args should be set to null.
  }

  // Loop over all of the computed argument and return value info. If any of
  // them are direct or extend without a specified coerce type, specify the
  // default now.
  ::cir::ABIArgInfo &retInfo = FI->getReturnInfo();
  if (retInfo.canHaveCoerceToType() && retInfo.getCoerceToType() == nullptr)
    llvm_unreachable("NYI");

  for (auto &_ : FI->arguments())
    llvm_unreachable("NYI");

  return *FI;
}
