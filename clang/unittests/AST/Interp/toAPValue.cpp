#include "../../../lib/AST/Interp/Context.h"
#include "../../../lib/AST/Interp/Descriptor.h"
#include "../../../lib/AST/Interp/Program.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::interp;
using namespace clang::ast_matchers;

/// Test the various toAPValue implementations.
TEST(ToAPValue, Pointers) {
  constexpr char Code[] =
      "struct A { bool a; bool z; };\n"
      "struct S {\n"
      "  A a[3];\n"
      "};\n"
      "constexpr S d = {{{true, false}, {false, true}, {false, false}}};\n"
      "constexpr const bool *b = &d.a[1].z;\n";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-fexperimental-new-constant-interpreter"});

  auto &Ctx = AST->getASTContext().getInterpContext();
  Program &Prog = Ctx.getProgram();

  auto getDecl = [&](const char *Name) -> const ValueDecl * {
    auto Nodes =
        match(valueDecl(hasName(Name)).bind("var"), AST->getASTContext());
    assert(Nodes.size() == 1);
    const auto *D = Nodes[0].getNodeAs<ValueDecl>("var");
    assert(D);
    return D;
  };
  auto getGlobalPtr = [&](const char *Name) -> Pointer {
    const VarDecl *D = cast<VarDecl>(getDecl(Name));
    return Prog.getPtrGlobal(*Prog.getGlobal(D));
  };

  const Pointer &GP = getGlobalPtr("b");
  const Pointer &P = GP.deref<Pointer>();
  ASSERT_TRUE(P.isLive());
  APValue A = P.toAPValue();
  ASSERT_TRUE(A.isLValue());
  ASSERT_TRUE(A.hasLValuePath());
  const auto &Path = A.getLValuePath();
  ASSERT_EQ(Path.size(), 3u);
  ASSERT_EQ(A.getLValueBase(), getDecl("d"));
}

TEST(ToAPValue, FunctionPointers) {
  constexpr char Code[] = " constexpr bool foo() { return true; }\n"
                          " constexpr bool (*func)() = foo;\n";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-fexperimental-new-constant-interpreter"});

  auto &Ctx = AST->getASTContext().getInterpContext();
  Program &Prog = Ctx.getProgram();

  auto getDecl = [&](const char *Name) -> const ValueDecl * {
    auto Nodes =
        match(valueDecl(hasName(Name)).bind("var"), AST->getASTContext());
    assert(Nodes.size() == 1);
    const auto *D = Nodes[0].getNodeAs<ValueDecl>("var");
    assert(D);
    return D;
  };

  auto getGlobalPtr = [&](const char *Name) -> Pointer {
    const VarDecl *D = cast<VarDecl>(getDecl(Name));
    return Prog.getPtrGlobal(*Prog.getGlobal(D));
  };

  const Pointer &GP = getGlobalPtr("func");
  const FunctionPointer &FP = GP.deref<FunctionPointer>();
  ASSERT_FALSE(FP.isZero());
  APValue A = FP.toAPValue();
  ASSERT_TRUE(A.hasValue());
  ASSERT_TRUE(A.isLValue());
  ASSERT_TRUE(A.hasLValuePath());
  const auto &Path = A.getLValuePath();
  ASSERT_EQ(Path.size(), 0u);
  ASSERT_FALSE(A.getLValueBase().isNull());
  ASSERT_EQ(A.getLValueBase().dyn_cast<const ValueDecl *>(), getDecl("foo"));
}
