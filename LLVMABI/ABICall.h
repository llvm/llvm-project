#include "Type.h"
#include "ABIFunctionInfo.h"

using namespace ABI;
using namespace ABIFunction;

class ABICall {
public:
  virtual void computeInfo(ABIFunctionInfo FI) const = 0;
}

class DefaultABICall : public ABICall {
public:
  ABIBuiltinType classifyReturnType(QualType RetTy) const;
  ABIBuiltinType classifyArgumentType(QualType RetTy) const;

  void computeInfo(ABIFunctionInfo FI) const override;

};
