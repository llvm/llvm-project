#include "LowerCall.h"
#include "CIRToCIRArgMapping.h"
#include "LowerFunctionInfo.h"
#include "LowerModule.h"
#include "LowerTypes.h"
#include "clang/CIR/FnInfoOpts.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::cir;

using ABIArgInfo = ::cir::ABIArgInfo;
using FnInfoOpts = ::cir::FnInfoOpts;
using MissingFeatures = ::cir::MissingFeatures;

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

/// Adds the formal parameters in FPT to the given prefix. If any parameter in
/// FPT has pass_object_size attrs, then we'll add parameters for those, too.
static void appendParameterTypes(SmallVectorImpl<Type> &prefix, FuncType fnTy) {
  // Fast path: don't touch param info if we don't need to.
  if (/*!fnTy->hasExtParameterInfos()=*/true) {
    prefix.append(fnTy.getInputs().begin(), fnTy.getInputs().end());
    return;
  }

  assert(MissingFeatures::extParamInfo());
  llvm_unreachable("NYI");
}

/// Arrange the LLVM function layout for a value of the given function
/// type, on top of any implicit parameters already stored.
///
/// \param CGT - Abstraction for lowering CIR types.
/// \param instanceMethod - Whether the function is an instance method.
/// \param prefix - List of implicit parameters to be prepended (e.g. 'this').
/// \param FTP - ABI-agnostic function type.
static const LowerFunctionInfo &
arrangeCIRFunctionInfo(LowerTypes &CGT, bool instanceMethod,
                       SmallVectorImpl<mlir::Type> &prefix, FuncType fnTy) {
  assert(!MissingFeatures::extParamInfo());
  RequiredArgs Required = RequiredArgs::forPrototypePlus(fnTy, prefix.size());
  // FIXME: Kill copy.
  appendParameterTypes(prefix, fnTy);
  assert(!MissingFeatures::qualifiedTypes());
  Type resultType = fnTy.getReturnType();

  FnInfoOpts opts =
      instanceMethod ? FnInfoOpts::IsInstanceMethod : FnInfoOpts::None;
  return CGT.arrangeLLVMFunctionInfo(resultType, opts, prefix, Required);
}

} // namespace

/// Update function with ABI-specific attributes.
///
/// NOTE(cir): Partially copies CodeGenModule::ConstructAttributeList, but
/// focuses on ABI/Target-related attributes.
void LowerModule::constructAttributeList(StringRef Name,
                                         const LowerFunctionInfo &FI,
                                         FuncOp CalleeInfo, FuncOp newFn,
                                         unsigned &CallingConv,
                                         bool AttrOnCallSite, bool IsThunk) {
  // Collect function IR attributes from the CC lowering.
  // We'll collect the paramete and result attributes later.
  // FIXME(cir): Codegen differentiates between CallConv and EffectiveCallConv,
  // but I don't think we need to do this here.
  CallingConv = FI.getCallingConvention();
  // FIXME(cir): No-return should probably be set in CIRGen (ABI-agnostic).
  if (MissingFeatures::noReturn())
    llvm_unreachable("NYI");
  if (MissingFeatures::csmeCall())
    llvm_unreachable("NYI");

  // TODO(cir): Implement AddAttributesFromFunctionProtoType here.
  // TODO(cir): Implement AddAttributesFromOMPAssumes here.
  assert(!MissingFeatures::openMP());

  // TODO(cir): Skipping a bunch of AST queries here. We will need to partially
  // implement some of them as this section sets target-specific attributes
  // too.
  // if (TargetDecl) {
  //   [...]
  // }

  // NOTE(cir): The original code adds default and no-builtin attributes here as
  // well. AFAIK, these are ABI/Target-agnostic, so it would be better handled
  // in CIRGen. Regardless, I'm leaving this comment here as a heads up.

  // Override some default IR attributes based on declaration-specific
  // information.
  // NOTE(cir): Skipping another set of AST queries here.

  // Collect attributes from arguments and return values.
  CIRToCIRArgMapping IRFunctionArgs(getContext(), FI);

  const ABIArgInfo &RetAI = FI.getReturnInfo();

  // TODO(cir): No-undef attribute for return values partially depends on
  // ABI-specific information. Maybe we should include it here.

  switch (RetAI.getKind()) {
  case ABIArgInfo::Ignore:
    break;
  default:
    llvm_unreachable("Missing ABIArgInfo::Kind");
  }

  if (!IsThunk) {
    if (MissingFeatures::qualTypeIsReferenceType()) {
      llvm_unreachable("NYI");
    }
  }

  // Attach attributes to sret.
  if (MissingFeatures::sretArgs()) {
    llvm_unreachable("sret is NYI");
  }

  // Attach attributes to inalloca arguments.
  if (MissingFeatures::inallocaArgs()) {
    llvm_unreachable("inalloca is NYI");
  }

  // Apply `nonnull`, `dereferencable(N)` and `align N` to the `this` argument,
  // unless this is a thunk function.
  // FIXME: fix this properly, https://reviews.llvm.org/D100388
  if (MissingFeatures::funcDeclIsCXXMethodDecl() ||
      MissingFeatures::inallocaArgs()) {
    llvm_unreachable("`this` argument attributes are NYI");
  }

  unsigned ArgNo = 0;
  for (LowerFunctionInfo::const_arg_iterator I = FI.arg_begin(),
                                             E = FI.arg_end();
       I != E; ++I, ++ArgNo) {
    llvm_unreachable("NYI");
  }
  assert(ArgNo == FI.arg_size());
}

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const LowerFunctionInfo &LowerTypes::arrangeFunctionDeclaration(FuncOp fnOp) {
  if (MissingFeatures::funcDeclIsCXXMethodDecl())
    llvm_unreachable("NYI");

  assert(!MissingFeatures::qualifiedTypes());
  FuncType FTy = fnOp.getFunctionType();

  assert(!MissingFeatures::CUDA());

  // When declaring a function without a prototype, always use a
  // non-variadic type.
  if (fnOp.getNoProto()) {
    llvm_unreachable("NYI");
  }

  return arrangeFreeFunctionType(FTy);
}

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

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const LowerFunctionInfo &LowerTypes::arrangeFreeFunctionType(FuncType FTy) {
  SmallVector<mlir::Type, 16> argTypes;
  return ::arrangeCIRFunctionInfo(*this, /*instanceMethod=*/false, argTypes,
                                  FTy);
}

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const LowerFunctionInfo &LowerTypes::arrangeGlobalDeclaration(FuncOp fnOp) {
  if (MissingFeatures::funcDeclIsCXXConstructorDecl() ||
      MissingFeatures::funcDeclIsCXXDestructorDecl())
    llvm_unreachable("NYI");

  return arrangeFunctionDeclaration(fnOp);
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
