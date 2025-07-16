#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MLIR_OPBUILDERCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MLIR_OPBUILDERCHECK_H

#include "../utils/TransformerClangTidyCheck.h"

namespace clang::tidy::mlir_check {

/// Checks for uses of `OpBuilder::create<T>` and suggests using `T::create`
/// instead.
class OpBuilderCheck : public utils::TransformerClangTidyCheck {
public:
  OpBuilderCheck(StringRef Name, ClangTidyContext *Context);

  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return getLangOpts().CPlusPlus;
  }
};

} // namespace clang::tidy::mlir_check

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MLIR_OPBUILDERCHECK_H
