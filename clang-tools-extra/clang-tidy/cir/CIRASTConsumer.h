#include "../ClangTidyDiagnosticConsumer.h"
#include "ClangTidyCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/Frontend/CompilerInstance.h"

using namespace clang;

namespace clang::tidy::cir {

constexpr const char *LifetimeCheckName = "cir-lifetime-check";

class CIRASTConsumer : public ASTConsumer {
public:
  CIRASTConsumer(CompilerInstance &CI, StringRef inputFile,
                 clang::tidy::ClangTidyContext &Context);

private:
  void Initialize(ASTContext &Context) override;
  void HandleTranslationUnit(ASTContext &C) override;
  bool HandleTopLevelDecl(DeclGroupRef D) override;
  std::unique_ptr<::cir::CIRGenerator> Gen;
  ASTContext *AstContext{nullptr};
  clang::tidy::ClangTidyContext &Context;
  clang::tidy::ClangTidyCheck::OptionsView OptsView;
};

} // namespace clang::tidy::cir
