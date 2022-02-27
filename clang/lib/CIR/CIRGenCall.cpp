#include "CIRGenFunction.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenTypes.h"

#include "clang/AST/GlobalDecl.h"

#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

using namespace cir;
using namespace clang;

CIRGenFunctionInfo *CIRGenFunctionInfo::create(
    unsigned cirCC, bool instanceMethod, bool chainCall,
    const clang::FunctionType::ExtInfo &info,
    llvm::ArrayRef<ExtParameterInfo> paramInfos, clang::CanQualType resultType,
    llvm::ArrayRef<clang::CanQualType> argTypes, RequiredArgs required) {
  assert(paramInfos.empty() || paramInfos.size() == argTypes.size());
  assert(!required.allowsOptionalArgs() ||
         required.getNumRequiredArgs() <= argTypes.size());

  void *buffer = operator new(totalSizeToAlloc<ArgInfo, ExtParameterInfo>(
      argTypes.size() + 1, paramInfos.size()));

  CIRGenFunctionInfo *FI = new (buffer) CIRGenFunctionInfo();
  FI->CallingConvention = cirCC;
  FI->EffectiveCallingConvention = cirCC;
  FI->ASTCallingConvention = info.getCC();
  FI->InstanceMethod = instanceMethod;
  FI->ChainCall = chainCall;
  FI->CmseNSCall = info.getCmseNSCall();
  FI->NoReturn = info.getNoReturn();
  FI->ReturnsRetained = info.getProducesResult();
  FI->NoCallerSavedRegs = info.getNoCallerSavedRegs();
  FI->NoCfCheck = info.getNoCfCheck();
  FI->Required = required;
  FI->HasRegParm = info.getHasRegParm();
  FI->RegParm = info.getRegParm();
  FI->ArgStruct = nullptr;
  FI->ArgStructAlign = 0;
  FI->NumArgs = argTypes.size();
  FI->HasExtParameterInfos = !paramInfos.empty();
  FI->getArgsBuffer()[0].type = resultType;
  for (unsigned i = 0; i < argTypes.size(); ++i)
    FI->getArgsBuffer()[i + 1].type = argTypes[i];
  for (unsigned i = 0; i < paramInfos.size(); ++i)
    FI->getExtParameterInfosBuffer()[i] = paramInfos[i];

  return FI;
}

namespace {

/// Encapsulates information about hte way function arguments from
/// CIRGenFunctionInfo should be passed to actual CIR function.
class ClangToCIRArgMapping {
  static const unsigned InvalidIndex = ~0U;
  unsigned InallocaArgNo;
  unsigned SRetArgNo;
  unsigned TotalCIRArgs;

  /// Arguments of CIR function corresponding to single Clang argument.
  struct CIRArgs {
    unsigned PaddingArgIndex;
    // Argument is expanded to CIR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned FirstArgIndex;
    unsigned NumberOfArgs;

    CIRArgs()
        : PaddingArgIndex(InvalidIndex), FirstArgIndex(InvalidIndex),
          NumberOfArgs(0) {}
  };

  SmallVector<CIRArgs, 8> ArgInfo;

public:
  ClangToCIRArgMapping(const ASTContext &Context, const CIRGenFunctionInfo &FI,
                       bool OnlyRequiredArgs = false)
      : InallocaArgNo(InvalidIndex), SRetArgNo(InvalidIndex), TotalCIRArgs(0),
        ArgInfo(OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size()) {
    construct(Context, FI, OnlyRequiredArgs);
  }

  bool hasSRetArg() const { return SRetArgNo != InvalidIndex; }

  bool hasInallocaArg() const { return InallocaArgNo != InvalidIndex; }

  unsigned totalCIRArgs() const { return TotalCIRArgs; }

  bool hasPaddingArg(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return ArgInfo[ArgNo].PaddingArgIndex != InvalidIndex;
  }

  /// Returns index of first CIR argument corresponding to ArgNo, and their
  /// quantity.
  std::pair<unsigned, unsigned> getCIRArgs(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return std::make_pair(ArgInfo[ArgNo].FirstArgIndex,
                          ArgInfo[ArgNo].NumberOfArgs);
  }

private:
  void construct(const ASTContext &Context, const CIRGenFunctionInfo &FI,
                 bool OnlyRequiredArgs);
};

void ClangToCIRArgMapping::construct(const ASTContext &Context,
                                     const CIRGenFunctionInfo &FI,
                                     bool OnlyRequiredArgs) {
  unsigned CIRArgNo = 0;
  bool SwapThisWithSRet = false;
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  assert(RetAI.getKind() != ABIArgInfo::Indirect && "NYI");

  unsigned ArgNo = 0;
  unsigned NumArgs = OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size();
  for (CIRGenFunctionInfo::const_arg_iterator I = FI.arg_begin();
       ArgNo < NumArgs; ++I, ++ArgNo) {
    assert(I != FI.arg_end());
    const ABIArgInfo &AI = I->info;
    // Collect data about CIR arguments corresponding to Clang argument ArgNo.
    auto &CIRArgs = ArgInfo[ArgNo];

    assert(!AI.getPaddingType() && "NYI");

    switch (AI.getKind()) {
    default:
      assert(false && "NYI");
    case ABIArgInfo::Direct: {
      assert(!AI.getCoerceToType().dyn_cast<mlir::cir::StructType>() && "NYI");
      // FIXME: handle sseregparm someday...
      // FIXME: handle structs
      CIRArgs.NumberOfArgs = 1;
      break;
    }
    }

    if (CIRArgs.NumberOfArgs > 0) {
      CIRArgs.FirstArgIndex = CIRArgNo;
      CIRArgNo += CIRArgs.NumberOfArgs;
    }

    assert(!SwapThisWithSRet && "NYI");
  }
  assert(ArgNo == ArgInfo.size());

  assert(!FI.usesInAlloca() && "NYI");

  TotalCIRArgs = CIRArgNo;
}

} // namespace

