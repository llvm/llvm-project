#ifndef LLVM_CLANG_SEMA_COMMON_H
#define LLVM_CLANG_SEMA_COMMON_H

#include "clang/Sema/Sema.h"

namespace clang {

using LLVMFnRef = llvm::function_ref<bool(clang::QualType PassedType)>;
using PairParam = std::pair<unsigned int, unsigned int>;
using CheckParam = std::variant<PairParam, LLVMFnRef>;

bool CheckArgTypeIsCorrect(
    Sema *S, Expr *Arg, QualType ExpectedType,
    llvm::function_ref<bool(clang::QualType PassedType)> Check);

bool CheckAllArgTypesAreCorrect(
    Sema *SemaPtr, CallExpr *TheCall,
    std::variant<QualType, std::nullopt_t> ExpectedType, CheckParam Check);

} // namespace clang

#endif
