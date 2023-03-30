

#ifndef LLVM_CLANG_AST_INTERP_FUNCTION_POINTER_H
#define LLVM_CLANG_AST_INTERP_FUNCTION_POINTER_H

#include "Function.h"
#include "Primitives.h"
#include "clang/AST/APValue.h"

namespace clang {
namespace interp {

class FunctionPointer final {
private:
  const Function *Func;

public:
  FunctionPointer() : Func(nullptr) {}
  FunctionPointer(const Function *Func) : Func(Func) { assert(Func); }

  const Function *getFunction() const { return Func; }

  APValue toAPValue() const {
    if (!Func)
      return APValue(static_cast<Expr *>(nullptr), CharUnits::Zero(), {},
                     /*OnePastTheEnd=*/false, /*IsNull=*/true);

    return APValue(Func->getDecl(), CharUnits::Zero(), {},
                   /*OnePastTheEnd=*/false, /*IsNull=*/false);
  }

  void print(llvm::raw_ostream &OS) const {
    OS << "FnPtr(";
    if (Func)
      OS << Func->getName();
    else
      OS << "nullptr";
    OS << ")";
  }

  ComparisonCategoryResult compare(const FunctionPointer &RHS) const {
    if (Func == RHS.Func)
      return ComparisonCategoryResult::Equal;
    return ComparisonCategoryResult::Unordered;
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     FunctionPointer FP) {
  FP.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
