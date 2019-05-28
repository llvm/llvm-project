//=== unittests/Sema/CodeCompleteTest.cpp - Code Complete tests ==============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Testing/Support/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <string>

namespace {

using namespace clang;
using namespace clang::tooling;
using ::testing::Each;
using ::testing::UnorderedElementsAre;

const char TestCCName[] = "test.cc";

struct CompletionContext {
  std::vector<std::string> VisitedNamespaces;
  std::string PreferredType;
  // String representation of std::ptrdiff_t on a given platform. This is a hack
  // to properly account for different configurations of clang.
  std::string PtrDiffType;
};

class VisitedContextFinder : public CodeCompleteConsumer {
public:
  VisitedContextFinder(CompletionContext &ResultCtx)
      : CodeCompleteConsumer(/*CodeCompleteOpts=*/{}), ResultCtx(ResultCtx),
        CCTUInfo(std::make_shared<GlobalCodeCompletionAllocator>()) {}

  void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *Results,
                                  unsigned NumResults) override {
    ResultCtx.VisitedNamespaces =
        getVisitedNamespace(Context.getVisitedContexts());
    ResultCtx.PreferredType = Context.getPreferredType().getAsString();
    ResultCtx.PtrDiffType =
        S.getASTContext().getPointerDiffType().getAsString();
  }

  CodeCompletionAllocator &getAllocator() override {
    return CCTUInfo.getAllocator();
  }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  std::vector<std::string> getVisitedNamespace(
      CodeCompletionContext::VisitedContextSet VisitedContexts) const {
    std::vector<std::string> NSNames;
    for (const auto *Context : VisitedContexts)
      if (const auto *NS = llvm::dyn_cast<NamespaceDecl>(Context))
        NSNames.push_back(NS->getQualifiedNameAsString());
    return NSNames;
  }

  CompletionContext &ResultCtx;
  CodeCompletionTUInfo CCTUInfo;
};

class CodeCompleteAction : public SyntaxOnlyAction {
public:
  CodeCompleteAction(ParsedSourceLocation P, CompletionContext &ResultCtx)
      : CompletePosition(std::move(P)), ResultCtx(ResultCtx) {}

  bool BeginInvocation(CompilerInstance &CI) override {
    CI.getFrontendOpts().CodeCompletionAt = CompletePosition;
    CI.setCodeCompletionConsumer(new VisitedContextFinder(ResultCtx));
    return true;
  }

private:
  // 1-based code complete position <Line, Col>;
  ParsedSourceLocation CompletePosition;
  CompletionContext &ResultCtx;
};

ParsedSourceLocation offsetToPosition(llvm::StringRef Code, size_t Offset) {
  Offset = std::min(Code.size(), Offset);
  StringRef Before = Code.substr(0, Offset);
  int Lines = Before.count('\n');
  size_t PrevNL = Before.rfind('\n');
  size_t StartOfLine = (PrevNL == StringRef::npos) ? 0 : (PrevNL + 1);
  return {TestCCName, static_cast<unsigned>(Lines + 1),
          static_cast<unsigned>(Offset - StartOfLine + 1)};
}

CompletionContext runCompletion(StringRef Code, size_t Offset) {
  CompletionContext ResultCtx;
  auto Action = llvm::make_unique<CodeCompleteAction>(
      offsetToPosition(Code, Offset), ResultCtx);
  clang::tooling::runToolOnCodeWithArgs(Action.release(), Code, {"-std=c++11"},
                                        TestCCName);
  return ResultCtx;
}

CompletionContext runCodeCompleteOnCode(StringRef AnnotatedCode) {
  llvm::Annotations A(AnnotatedCode);
  return runCompletion(A.code(), A.point());
}

std::vector<std::string>
collectPreferredTypes(StringRef AnnotatedCode,
                      std::string *PtrDiffType = nullptr) {
  llvm::Annotations A(AnnotatedCode);
  std::vector<std::string> Types;
  for (size_t Point : A.points()) {
    auto Results = runCompletion(A.code(), Point);
    if (PtrDiffType) {
      assert(PtrDiffType->empty() || *PtrDiffType == Results.PtrDiffType);
      *PtrDiffType = Results.PtrDiffType;
    }
    Types.push_back(Results.PreferredType);
  }
  return Types;
}

TEST(SemaCodeCompleteTest, VisitedNSForValidQualifiedId) {
  auto VisitedNS = runCodeCompleteOnCode(R"cpp(
     namespace ns1 {}
     namespace ns2 {}
     namespace ns3 {}
     namespace ns3 { namespace nns3 {} }

     namespace foo {
     using namespace ns1;
     namespace ns4 {} // not visited
     namespace { using namespace ns2; }
     inline namespace bar { using namespace ns3::nns3; }
     } // foo
     namespace ns { foo::^ }
  )cpp")
                       .VisitedNamespaces;
  EXPECT_THAT(VisitedNS, UnorderedElementsAre("foo", "ns1", "ns2", "ns3::nns3",
                                              "foo::(anonymous)"));
}

TEST(SemaCodeCompleteTest, VisitedNSForInvalidQualifiedId) {
  auto VisitedNS = runCodeCompleteOnCode(R"cpp(
     namespace na {}
     namespace ns1 {
     using namespace na;
     foo::^
     }
  )cpp")
                       .VisitedNamespaces;
  EXPECT_THAT(VisitedNS, UnorderedElementsAre("ns1", "na"));
}

TEST(SemaCodeCompleteTest, VisitedNSWithoutQualifier) {
  auto VisitedNS = runCodeCompleteOnCode(R"cpp(
    namespace n1 {
    namespace n2 {
      void f(^) {}
    }
    }
  )cpp")
                       .VisitedNamespaces;
  EXPECT_THAT(VisitedNS, UnorderedElementsAre("n1", "n1::n2"));
}

TEST(PreferredTypeTest, BinaryExpr) {
  // Check various operations for arithmetic types.
  StringRef Code = R"cpp(
    void test(int x) {
      x = ^10;
      x += ^10; x -= ^10; x *= ^10; x /= ^10; x %= ^10;
      x + ^10; x - ^10; x * ^10; x / ^10; x % ^10;
    })cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("int"));

  Code = R"cpp(
    void test(float x) {
      x = ^10;
      x += ^10; x -= ^10; x *= ^10; x /= ^10; x %= ^10;
      x + ^10; x - ^10; x * ^10; x / ^10; x % ^10;
    })cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("float"));

  // Pointer types.
  Code = R"cpp(
    void test(int *ptr) {
      ptr - ^ptr;
      ptr = ^ptr;
    })cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("int *"));

  Code = R"cpp(
    void test(int *ptr) {
      ptr + ^10;
      ptr += ^10;
      ptr -= ^10;
    })cpp";
  {
    std::string PtrDiff;
    auto Types = collectPreferredTypes(Code, &PtrDiff);
    EXPECT_THAT(Types, Each(PtrDiff));
  }

  // Comparison operators.
  Code = R"cpp(
    void test(int i) {
      i <= ^1; i < ^1; i >= ^1; i > ^1; i == ^1; i != ^1;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("int"));

  Code = R"cpp(
    void test(int *ptr) {
      ptr <= ^ptr; ptr < ^ptr; ptr >= ^ptr; ptr > ^ptr;
      ptr == ^ptr; ptr != ^ptr;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("int *"));

  // Relational operations.
  Code = R"cpp(
    void test(int i, int *ptr) {
      i && ^1; i || ^1;
      ptr && ^1; ptr || ^1;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("_Bool"));

  // Bitwise operations.
  Code = R"cpp(
    void test(long long ll) {
      ll | ^1; ll & ^1;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("long long"));

  Code = R"cpp(
    enum A {};
    void test(A a) {
      a | ^1; a & ^1;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("enum A"));

  Code = R"cpp(
    enum class A {};
    void test(A a) {
      // This is technically illegal with the 'enum class' without overloaded
      // operators, but we pretend it's fine.
      a | ^a; a & ^a;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("enum A"));

  // Binary shifts.
  Code = R"cpp(
    void test(int i, long long ll) {
      i << ^1; ll << ^1;
      i <<= ^1; i <<= ^1;
      i >> ^1; ll >> ^1;
      i >>= ^1; i >>= ^1;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("int"));

  // Comma does not provide any useful information.
  Code = R"cpp(
    class Cls {};
    void test(int i, int* ptr, Cls x) {
      (i, ^i);
      (ptr, ^ptr);
      (x, ^x);
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("NULL TYPE"));

  // User-defined types do not take operator overloading into account.
  // However, they provide heuristics for some common cases.
  Code = R"cpp(
    class Cls {};
    void test(Cls c) {
      // we assume arithmetic and comparions ops take the same type.
      c + ^c; c - ^c; c * ^c; c / ^c; c % ^c;
      c == ^c; c != ^c; c < ^c; c <= ^c; c > ^c; c >= ^c;
      // same for the assignments.
      c = ^c; c += ^c; c -= ^c; c *= ^c; c /= ^c; c %= ^c;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("class Cls"));

  Code = R"cpp(
    class Cls {};
    void test(Cls c) {
      // we assume relational ops operate on bools.
      c && ^c; c || ^c;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("_Bool"));

  Code = R"cpp(
    class Cls {};
    void test(Cls c) {
      // we make no assumptions about the following operators, since they are
      // often overloaded with a non-standard meaning.
      c << ^c; c >> ^c; c | ^c; c & ^c;
      c <<= ^c; c >>= ^c; c |= ^c; c &= ^c;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("NULL TYPE"));
}

TEST(PreferredTypeTest, Members) {
  StringRef Code = R"cpp(
    struct vector {
      int *begin();
      vector clone();
    };

    void test(int *a) {
      a = ^vector().^clone().^begin();
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("int *"));
}

TEST(PreferredTypeTest, Conditions) {
  StringRef Code = R"cpp(
    struct vector {
      bool empty();
    };

    void test() {
      if (^vector().^empty()) {}
      while (^vector().^empty()) {}
      for (; ^vector().^empty();) {}
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("_Bool"));
}

TEST(PreferredTypeTest, InitAndAssignment) {
  StringRef Code = R"cpp(
    struct vector {
      int* begin();
    };

    void test() {
      const int* x = ^vector().^begin();
      x = ^vector().^begin();

      if (const int* y = ^vector().^begin()) {}
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("const int *"));
}

TEST(PreferredTypeTest, UnaryExprs) {
  StringRef Code = R"cpp(
    void test(long long a) {
      a = +^a;
      a = -^a
      a = ++^a;
      a = --^a;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("long long"));

  Code = R"cpp(
    void test(int a, int *ptr) {
      !^a;
      !^ptr;
      !!!^a;

      a = !^a;
      a = !^ptr;
      a = !!!^a;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("_Bool"));

  Code = R"cpp(
    void test(int a) {
      const int* x = &^a;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("const int"));

  Code = R"cpp(
    void test(int *a) {
      int x = *^a;
      int &r = *^a;
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("int *"));

  Code = R"cpp(
    void test(int a) {
      *^a;
      &^a;
    }

  )cpp";
}

TEST(PreferredTypeTest, ParenExpr) {
  StringRef Code = R"cpp(
    const int *i = ^(^(^(^10)));
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("const int *"));
}

TEST(PreferredTypeTest, FunctionArguments) {
  StringRef Code = R"cpp(
    void foo(const int*);

    void bar(const int*);
    void bar(const int*, int b);

    struct vector {
      const int *data();
    };
    void test() {
      foo(^(^(^(^vec^tor^().^da^ta^()))));
      bar(^(^(^(^vec^tor^().^da^ta^()))));
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("const int *"));

  Code = R"cpp(
    void bar(int, volatile double *);
    void bar(int, volatile double *, int, int);

    struct vector {
      double *data();
    };

    struct class_members {
      void bar(int, volatile double *);
      void bar(int, volatile double *, int, int);
    };
    void test() {
      bar(10, ^(^(^(^vec^tor^().^da^ta^()))));
      class_members().bar(10, ^(^(^(^vec^tor^().^da^ta^()))));
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("volatile double *"));

  Code = R"cpp(
    namespace ns {
      struct vector {
      };
    }
    void accepts_vector(ns::vector);

    void test() {
      accepts_vector(^::^ns::^vector());
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("ns::vector"));

  Code = R"cpp(
    template <class T>
    struct vector { using self = vector; };

    void accepts_vector(vector<int>);
    int foo(int);

    void test() {
      accepts_vector(^::^vector<decltype(foo(1))>::^self);
    }
  )cpp";
  EXPECT_THAT(collectPreferredTypes(Code), Each("vector<int>"));
}
} // namespace
