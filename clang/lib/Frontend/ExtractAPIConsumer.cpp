#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"

using namespace clang;

namespace {
class ExtractAPIVisitor : public RecursiveASTVisitor<ExtractAPIVisitor> {
public:
  bool VisitNamedDecl(NamedDecl *Decl) {
    llvm::outs() << Decl->getName() << "\n";
    return true;
  }
};

class ExtractAPIConsumer : public ASTConsumer {
public:
  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  ExtractAPIVisitor Visitor;
};
} // namespace

std::unique_ptr<ASTConsumer>
ExtractAPIAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  return std::make_unique<ExtractAPIConsumer>();
}
