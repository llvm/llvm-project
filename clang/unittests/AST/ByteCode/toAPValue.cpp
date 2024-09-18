#include "../../../lib/AST/ByteCode/Context.h"
#include "../../../lib/AST/ByteCode/Descriptor.h"
#include "../../../lib/AST/ByteCode/Program.h"
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
      "constexpr const bool *b = &d.a[1].z;\n"
      "const void *p = (void*)12;\n"
      "const void *nullp = (void*)0;\n";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-fexperimental-new-constant-interpreter"});

  auto &ASTCtx = AST->getASTContext();
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

  {
    const Pointer &GP = getGlobalPtr("b");
    const Pointer &P = GP.deref<Pointer>();
    ASSERT_TRUE(P.isLive());
    APValue A = P.toAPValue(ASTCtx);
    ASSERT_TRUE(A.isLValue());
    ASSERT_TRUE(A.hasLValuePath());
    const auto &Path = A.getLValuePath();
    ASSERT_EQ(Path.size(), 3u);
    ASSERT_EQ(A.getLValueBase(), getDecl("d"));
    // FIXME: Also test all path elements.
  }

  {
    const ValueDecl *D = getDecl("p");
    ASSERT_NE(D, nullptr);
    const Pointer &GP = getGlobalPtr("p");
    const Pointer &P = GP.deref<Pointer>();
    ASSERT_TRUE(P.isIntegralPointer());
    APValue A = P.toAPValue(ASTCtx);
    ASSERT_TRUE(A.isLValue());
    ASSERT_TRUE(A.getLValueBase().isNull());
    APSInt I;
    bool Success = A.toIntegralConstant(I, D->getType(), AST->getASTContext());
    ASSERT_TRUE(Success);
    ASSERT_EQ(I, 12);
  }

  {
    const ValueDecl *D = getDecl("nullp");
    ASSERT_NE(D, nullptr);
    const Pointer &GP = getGlobalPtr("nullp");
    const Pointer &P = GP.deref<Pointer>();
    ASSERT_TRUE(P.isIntegralPointer());
    APValue A = P.toAPValue(ASTCtx);
    ASSERT_TRUE(A.isLValue());
    ASSERT_TRUE(A.getLValueBase().isNull());
    ASSERT_TRUE(A.isNullPointer());
    APSInt I;
    bool Success = A.toIntegralConstant(I, D->getType(), AST->getASTContext());
    ASSERT_TRUE(Success);
    ASSERT_EQ(I, 0);
  }
}

TEST(ToAPValue, FunctionPointers) {
  constexpr char Code[] = " constexpr bool foo() { return true; }\n"
                          " constexpr bool (*func)() = foo;\n"
                          " constexpr bool (*nullp)() = nullptr;\n";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-fexperimental-new-constant-interpreter"});

  auto &ASTCtx = AST->getASTContext();
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

  {
    const Pointer &GP = getGlobalPtr("func");
    const FunctionPointer &FP = GP.deref<FunctionPointer>();
    ASSERT_FALSE(FP.isZero());
    APValue A = FP.toAPValue(ASTCtx);
    ASSERT_TRUE(A.hasValue());
    ASSERT_TRUE(A.isLValue());
    ASSERT_TRUE(A.hasLValuePath());
    const auto &Path = A.getLValuePath();
    ASSERT_EQ(Path.size(), 0u);
    ASSERT_FALSE(A.getLValueBase().isNull());
    ASSERT_EQ(A.getLValueBase().dyn_cast<const ValueDecl *>(), getDecl("foo"));
  }

  {
    const ValueDecl *D = getDecl("nullp");
    ASSERT_NE(D, nullptr);
    const Pointer &GP = getGlobalPtr("nullp");
    const auto &P = GP.deref<FunctionPointer>();
    APValue A = P.toAPValue(ASTCtx);
    ASSERT_TRUE(A.isLValue());
    ASSERT_TRUE(A.getLValueBase().isNull());
    ASSERT_TRUE(A.isNullPointer());
    APSInt I;
    bool Success = A.toIntegralConstant(I, D->getType(), AST->getASTContext());
    ASSERT_TRUE(Success);
    ASSERT_EQ(I, 0);
  }
}

TEST(ToAPValue, FunctionPointersC) {
  // NB: The declaration of func2 is useless, but it makes us register a global
  // variable for func.
  constexpr char Code[] = "const int (* const func)(int *) = (void*)17;\n"
                          "const int (*func2)(int *) = func;\n";
  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-x", "c", "-fexperimental-new-constant-interpreter"});

  auto &ASTCtx = AST->getASTContext();
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

  {
    const ValueDecl *D = getDecl("func");
    const Pointer &GP = getGlobalPtr("func");
    ASSERT_TRUE(GP.isLive());
    const FunctionPointer &FP = GP.deref<FunctionPointer>();
    ASSERT_FALSE(FP.isZero());
    APValue A = FP.toAPValue(ASTCtx);
    ASSERT_TRUE(A.hasValue());
    ASSERT_TRUE(A.isLValue());
    const auto &Path = A.getLValuePath();
    ASSERT_EQ(Path.size(), 0u);
    ASSERT_TRUE(A.getLValueBase().isNull());
    APSInt I;
    bool Success = A.toIntegralConstant(I, D->getType(), AST->getASTContext());
    ASSERT_TRUE(Success);
    ASSERT_EQ(I, 17);
  }
}

TEST(ToAPValue, MemberPointers) {
  constexpr char Code[] = "struct S {\n"
                          "  int m, n;\n"
                          "};\n"
                          "constexpr int S::*pm = &S::m;\n"
                          "constexpr int S::*nn = nullptr;\n";

  auto AST = tooling::buildASTFromCodeWithArgs(
      Code, {"-fexperimental-new-constant-interpreter"});

  auto &ASTCtx = AST->getASTContext();
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

  {
    const Pointer &GP = getGlobalPtr("pm");
    ASSERT_TRUE(GP.isLive());
    const MemberPointer &FP = GP.deref<MemberPointer>();
    APValue A = FP.toAPValue(ASTCtx);
    ASSERT_EQ(A.getMemberPointerDecl(), getDecl("m"));
    ASSERT_EQ(A.getKind(), APValue::MemberPointer);
  }

  {
    const Pointer &GP = getGlobalPtr("nn");
    ASSERT_TRUE(GP.isLive());
    const MemberPointer &NP = GP.deref<MemberPointer>();
    ASSERT_TRUE(NP.isZero());
    APValue A = NP.toAPValue(ASTCtx);
    ASSERT_EQ(A.getKind(), APValue::MemberPointer);
  }
}
