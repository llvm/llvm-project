// REQUIRES: plugins
// RUN: clang-tidy -checks='-*,mytest*' --list-checks -load %llvmshlibdir/CTTestTidyModule%pluginext -load %llvmshlibdir/LLVMHello%pluginext | FileCheck --check-prefix=CHECK-LIST %s
// CHECK-LIST: Enabled checks:
// CHECK-LIST-NEXT:    mytest1
// CHECK-LIST-NEXT:    mytest2
// RUN: clang-tidy -checks='-*,mytest*,misc-definitions-in-headers' -load %llvmshlibdir/CTTestTidyModule%pluginext /dev/null -- -xc 2>&1 | FileCheck %s
// CHECK: 3 warnings generated.
// CHECK-NEXT: warning: mytest success [misc-definitions-in-headers,mytest1,mytest2]

#include "clang-tidy/ClangTidy.h"
#include "clang-tidy/ClangTidyCheck.h"
#include "clang-tidy/ClangTidyModule.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang;
using namespace clang::tidy;
using namespace clang::ast_matchers;

namespace {
class MyTestCheck : public ClangTidyCheck {

public:
  MyTestCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override {
    Finder->addMatcher(translationUnitDecl().bind("tu"), this);
  }

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override {
    auto S = Result.Nodes.getNodeAs<TranslationUnitDecl>("tu");
    if (S)
      diag("mytest success");
  }

private:
};

class CTTestModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<MyTestCheck>("mytest1");
    CheckFactories.registerCheck<MyTestCheck>("mytest2");
    // intentionally collide with an existing test name, overriding it
    CheckFactories.registerCheck<MyTestCheck>("misc-definitions-in-headers");
  }
};
} // namespace

namespace tidy1 {
// Register the CTTestTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<::CTTestModule>
    X("mytest-module", "Adds my checks.");
} // namespace tidy1

namespace tidy2 {
// intentionally collide with an existing test group name, merging with it
static ClangTidyModuleRegistry::Add<::CTTestModule>
    X("misc-module", "Adds miscellaneous lint checks.");
} // namespace tidy2

// This anchor is used to force the linker to link in the generated object file
// and thus register the CTTestModule.
volatile int CTTestModuleAnchorSource = 0;
