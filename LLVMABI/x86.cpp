#include "Type.h"
#include "ABIFunctionInfo.h"
#include "ABICall.h"
using namespace ABI;
using namespace ABIFunction;

class X86_64ABIInfo : public ABICall {
public:
  ABIArgInfo classifyReturnType(ABIQualType RetTy) const;
  ABIArgInfo classifyArgumentType(ABIQualType RetTy) const;

  void computeInfo(ABIFunctionInfo FI) const override;

};

ABIArgInfo
X86_64ABIInfo::classifyReturnType(ABIQualType Ty) const {

}

ABIArgInfo
X86_64ABIInfo::classifyArgumentType(ABIQualType Ty) const {

}

void X86_64ABIInfo::computeInfo(ABIFunctionInfo FI) const {

    // adjust the return type according to the ABI spec
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

    for (ABIFunctionInfo::arg_iterator it = FI.arg_begin(), ie = FI.arg_end();
       it != ie; ++it, ++ArgNo) {
      it->info = classifyArgumentType(it->type);
    }
}



std::unique_ptr<TargetCodeGenInfo>
CodeGen::createBPFTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<BPFTargetCodeGenInfo>(CGM.getTypes());
}