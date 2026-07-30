#include "clang/AST/DeclarationName.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace {

class LookupAction : public ASTFrontendAction {
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef /*Unused*/) override {
    return std::make_unique<clang::ASTConsumer>();
  }

  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    ASSERT_FALSE(CI.hasSema());
    CI.createSema(getTranslationUnitKind(), nullptr);
    ASSERT_TRUE(CI.hasSema());
    Sema &S = CI.getSema();
    ParseAST(S);

    ASTContext &Ctx = S.getASTContext();
    auto Name = &Ctx.Idents.get("Foo");
    LookupResult R_cpp(S, Name, SourceLocation(), Sema::LookupOrdinaryName);
    S.LookupName(R_cpp, S.TUScope, /*AllowBuiltinCreation=*/false,
                 /*ForceNoCPlusPlus=*/false);
    // By this point, parsing is done and S.TUScope is nullptr
    // CppLookupName will perform an early return with no results if the Scope
    // we pass in is nullptr. We expect to find nothing.
    ASSERT_TRUE(R_cpp.empty());

    // On the other hand, the non-C++ path doesn't care if the Scope passed in
    // is nullptr. We'll force the non-C++ path with a flag.
    LookupResult R_nocpp(S, Name, SourceLocation(), Sema::LookupOrdinaryName);
    S.LookupName(R_nocpp, S.TUScope, /*AllowBuiltinCreation=*/false,
                 /*ForceNoCPlusPlus=*/true);
    ASSERT_TRUE(!R_nocpp.empty());
  }
};

TEST(SemaLookupTest, ForceNoCPlusPlusPath) {
  const char *file_contents = R"objcxx(
@protocol Foo
@end
@interface Foo <Foo>
@end
  )objcxx";
  ASSERT_TRUE(runToolOnCodeWithArgs(std::make_unique<LookupAction>(),
                                    file_contents, {"-x", "objective-c++"},
                                    "test.mm"));
}
} // namespace
