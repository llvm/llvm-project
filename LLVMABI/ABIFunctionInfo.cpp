#include "LLVMABI/ABIFunctionInfo.h"
#include <iostream>

using namespace ABIFunction;

void ABIArgInfo::dump() const {
  std::cout << "ABIArgInfo Kind: ";
  switch (TheKind) {
  case Direct:
    std::cout << "Direct";
    break;
  case Extend:
    std::cout << "Extend";
    break;
  case Indirect:
    std::cout << "Indirect";
    break;
  case IndirectAliased:
    std::cout << "IndirectAliased";
    break;
  case Ignore:
    std::cout << "Ignore";
    break;
  case Expand:
    std::cout << "Expand";
    break;
  case CoerceAndExpand:
    std::cout << "CoerceAndExpand";
    break;
  case InAlloca:
    std::cout << "InAlloca";
    break;
  default:
    std::cout << "Unknown";
    break;
  }
  std::cout << std::endl;
}

ABIFunction::ABIFunctionInfo::ABIFunctionInfo(
    ABIFunction::CallingConv cc, std::vector<const Type *> parameters,
    const ABI::Type *ReturnType)
    : CC(cc) {
  for (const auto *Ty : parameters) {
    Parameters.push_back({Ty, ABIArgInfo()});
  }

  RetInfo = {ReturnType, ABIArgInfo()};
}

ABIFunctionInfo *ABIFunctionInfo::create(CallingConv cc,
                                         std::vector<const Type *> parameters,
                                         const Type *ReturnType) {
  return new ABIFunctionInfo(cc, parameters, ReturnType);
}

ABIArgInfo &ABIFunctionInfo::getReturnInfo() { return RetInfo.info; }

const ABIArgInfo &ABIFunctionInfo::getReturnInfo() const {
  return RetInfo.info;
}

const Type *ABIFunctionInfo::getReturnType() const { return RetInfo.type; }

ABIFunctionInfo::arg_iterator ABIFunctionInfo::arg_begin() {
  return Parameters.begin();
}

ABIFunctionInfo::arg_iterator ABIFunctionInfo::arg_end() {
  return Parameters.end();
}

ABIFunctionInfo::const_arg_iterator ABIFunctionInfo::arg_begin() const {
  return Parameters.begin();
}

ABIFunctionInfo::const_arg_iterator ABIFunctionInfo::arg_end() const {
  return Parameters.end();
}

void ABIFunctionInfo::dump() const {
  std::cout << "Calling Convention: ";
  switch (CC) {
  case CC_C:
    std::cout << "CC_C\n";
    break;
  case CC_X86StdCall:
    std::cout << "CC_X86StdCall\n";
    break;
  case Win64:
    std::cout << "Win64\n";
    break;
  case X86_RegCall:
    std::cout << "X86_RegCall\n";
    break;
  default:
    std::cout << "Unknown CC\n";
    break;
  }

  std::cout << "Return Type:\n";
  if (RetInfo.type) {
    RetInfo.type->dump();
    RetInfo.info.dump();
  } else {
    std::cout << "  (null)\n";
  }

  std::cout << "Arguments:\n";
  for (const auto &Arg : Parameters) {
    if (Arg.type)
      Arg.type->dump();
    else
      std::cout << "  (null)\n";
    Arg.info.dump();
  }
}
