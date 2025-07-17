#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_MLIROPBUILDERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_MLIROPBUILDERCHECK_H

#include "../utils/TransformerClangTidyCheck.h"

namespace clang::tidy::llvm_check {

/// Checks for uses of MLIR's `OpBuilder::create<T>` and suggests using
/// `T::create` instead.
class MlirOpBuilderCheck : public utils::TransformerClangTidyCheck {
public:
  MlirOpBuilderCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return getLangOpts().CPlusPlus;
  }
};

} // namespace clang::tidy::llvm_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_LLVM_MLIROPBUILDERCHECK_H
