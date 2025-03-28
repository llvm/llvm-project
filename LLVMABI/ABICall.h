#ifndef ABICALL_H
#define ABICALL_H

#include "Type.h"
#include "ABIFunctionInfo.h"

using namespace ABI;
using namespace ABIFunction;

class ABICall {
public:
  virtual void computeInfo(ABIFunctionInfo &FI) const = 0;
};

class DefaultABICall : public ABICall {
public:
  ABIArgInfo classifyReturnType(ABIQualType RetTy) const;
  ABIArgInfo classifyArgumentType(ABIQualType RetTy) const;

  void computeInfo(ABIFunctionInfo &FI) const override;

};

#endif