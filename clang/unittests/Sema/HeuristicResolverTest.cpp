//===-- HeuristicResolverTests.cpp --------------------------*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Sema/HeuristicResolver.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace clang::ast_matchers;
using testing::ElementsAre;

namespace clang {
namespace {

// Helper for matching a sequence of elements with a variadic list of matchers.
// Usage: `ElementsAre(matchAdapter(Vs, MatchFunction)...)`, where `Vs...` is
//        a variadic list of matchers.
// For each `V` in `Vs`, this will match the corresponding element `E` if
// `MatchFunction(V, E)` is true.
MATCHER_P2(matchAdapter, MatcherForElement, MatchFunction, "matchAdapter") {
  return MatchFunction(MatcherForElement, arg);
}

template <typename InputNode>
using ResolveFnT = std::function<std::vector<const NamedDecl *>(
    const HeuristicResolver *, const InputNode *)>;

// Test heuristic resolution on `Code` using the resolution procedure
// `ResolveFn`, which takes a `HeuristicResolver` and an input AST node of type
// `InputNode` and returns a `std::vector<const NamedDecl *>`.
// `InputMatcher` should be an AST matcher that matches a single node to pass as
// input to `ResolveFn`, bound to the ID "input". `OutputMatchers` should be AST
// matchers that each match a single node, bound to the ID "output".
template <typename InputNode, typename InputMatcher, typename... OutputMatchers>
void expectResolution(llvm::StringRef Code, ResolveFnT<InputNode> ResolveFn,
                      const InputMatcher &IM, const OutputMatchers &...OMS) {
  auto TU = tooling::buildASTFromCodeWithArgs(Code, {"-std=c++20"});
  auto &Ctx = TU->getASTContext();
  auto InputMatches = match(IM, Ctx);
  ASSERT_EQ(1u, InputMatches.size());
  const auto *Input = InputMatches[0].template getNodeAs<InputNode>("input");
  ASSERT_TRUE(Input);

  auto OutputNodeMatches = [&](auto &OutputMatcher, auto &Actual) {
    auto OutputMatches = match(OutputMatcher, Ctx);
    if (OutputMatches.size() != 1u)
      return false;
    const auto *ExpectedOutput =
        OutputMatches[0].template getNodeAs<NamedDecl>("output");
    if (!ExpectedOutput)
      return false;
    return ExpectedOutput == Actual;
  };

  HeuristicResolver H(Ctx);
  auto Results = ResolveFn(&H, Input);
  EXPECT_THAT(Results, ElementsAre(matchAdapter(OMS, OutputNodeMatches)...));
}

// Wrapper for the above that accepts a HeuristicResolver member function
// pointer directly.
template <typename InputNode, typename InputMatcher, typename... OutputMatchers>
void expectResolution(llvm::StringRef Code,
                      std::vector<const NamedDecl *> (
                          HeuristicResolver::*ResolveFn)(const InputNode *)
                          const,
                      const InputMatcher &IM, const OutputMatchers &...OMS) {
  expectResolution(Code, ResolveFnT<InputNode>(std::mem_fn(ResolveFn)), IM,
                   OMS...);
}

TEST(HeuristicResolver, MemberExpr) {
  std::string Code = R"cpp(
    template <typename T>
    struct S {
      void bar() {}
    };

    template <typename T>
    void foo(S<T> arg) {
      arg.bar();
    }
  )cpp";
  // Test resolution of "bar" in "arg.bar()".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("bar")).bind("input"),
      cxxMethodDecl(hasName("bar")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_Overloads) {
  std::string Code = R"cpp(
    template <typename T>
    struct S {
      void bar(int);
      void bar(float);
    };

    template <typename T, typename U>
    void foo(S<T> arg, U u) {
      arg.bar(u);
    }
  )cpp";
  // Test resolution of "bar" in "arg.bar(u)". Both overloads should be found.
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("bar")).bind("input"),
      cxxMethodDecl(hasName("bar"), hasParameter(0, hasType(asString("int"))))
          .bind("output"),
      cxxMethodDecl(hasName("bar"), hasParameter(0, hasType(asString("float"))))
          .bind("output"));
}

TEST(HeuristicResolver, MemberExpr_SmartPointer) {
  std::string Code = R"cpp(
    template <typename> struct S { void foo() {} };
    template <typename T> struct unique_ptr {
      T* operator->();
    };
    template <typename T>
    void test(unique_ptr<S<T>>& v) {
      v->foo();
    }
  )cpp";
  // Test resolution of "foo" in "v->foo()".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("foo")).bind("input"),
      cxxMethodDecl(hasName("foo")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_SmartPointer_Qualified) {
  std::string Code = R"cpp(
    template <typename> struct Waldo {
      void find();
      void find() const;
    };
    template <typename T> struct unique_ptr {
      T* operator->();
    };
    template <typename T>
    void test(unique_ptr<const Waldo<T>>& w) {
      w->find();
    }
  )cpp";
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("find")).bind("input"),
      cxxMethodDecl(hasName("find"), isConst()).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_Static_Qualified) {
  std::string Code = R"cpp(
    template <typename T>
    struct Waldo {
      static void find();
    };
    template <typename T>
    void foo(const Waldo<T>& t) {
      t.find();
    }
  )cpp";
  // Test resolution of "find" in "t.find()".
  // The object being `const` should have no bearing on a call to a static
  // method.
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("find")).bind("input"),
      cxxMethodDecl(hasName("find")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_AutoTypeDeduction1) {
  std::string Code = R"cpp(
    template <typename T>
    struct A {
      int waldo;
    };
    template <typename T>
    void foo(A<T> a) {
      auto copy = a;
      copy.waldo;
    }
  )cpp";
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("waldo")).bind("input"),
      fieldDecl(hasName("waldo")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_AutoTypeDeduction2) {
  std::string Code = R"cpp(
    struct B {
      int waldo;
    };

    template <typename T>
    struct A {
      B b;
    };
    template <typename T>
    void foo(A<T> a) {
      auto b = a.b;
      b.waldo;
    }
  )cpp";
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("waldo")).bind("input"),
      fieldDecl(hasName("waldo")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_Chained) {
  std::string Code = R"cpp(
    struct A { void foo() {} };
    template <typename T>
    struct B {
      A func(int);
      void bar() {
        func(1).foo();
      }
    };
  )cpp";
  // Test resolution of "foo" in "func(1).foo()".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("foo")).bind("input"),
      cxxMethodDecl(hasName("foo")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_ReferenceType) {
  std::string Code = R"cpp(
    struct B {
      int waldo;
    };
    template <typename T>
    struct A {
      B &b;
    };
    template <typename T>
    void foo(A<T> &a) {
      a.b.waldo;
    }
  )cpp";
  // Test resolution of "waldo" in "a.b.waldo".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("waldo")).bind("input"),
      fieldDecl(hasName("waldo")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_PointerType) {
  std::string Code = R"cpp(
    struct B {
      int waldo;
    };
    template <typename T>
    struct A {
      B *b;
    };
    template <typename T>
    void foo(A<T> &a) {
      a.b->waldo;
    }
  )cpp";
  // Test resolution of "waldo" in "a.b->waldo".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("waldo")).bind("input"),
      fieldDecl(hasName("waldo")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_TemplateArgs) {
  std::string Code = R"cpp(
    struct Foo {
      static Foo k(int);
      template <typename T> T convert();
    };
    template <typename T>
    void test() {
      Foo::k(T()).template convert<T>();
    }
  )cpp";
  // Test resolution of "convert" in "Foo::k(T()).template convert<T>()".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("convert")).bind("input"),
      functionTemplateDecl(hasName("convert")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_TypeAlias) {
  std::string Code = R"cpp(
    template <typename T>
    struct Waldo {
      void find();
    };
    template <typename T>
    using Wally = Waldo<T>;
    template <typename T>
    void foo(Wally<T> w) {
      w.find();
    }
  )cpp";
  // Test resolution of "find" in "w.find()".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("find")).bind("input"),
      cxxMethodDecl(hasName("find")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_BaseClass_TypeAlias) {
  std::string Code = R"cpp(
    template <typename T>
    struct Waldo {
      void find();
    };
    template <typename T>
    using Wally = Waldo<T>;
    template <typename T>
    struct S : Wally<T> {
      void foo() {
        this->find();
      }
    };
  )cpp";
  // Test resolution of "find" in "this->find()".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("find")).bind("input"),
      cxxMethodDecl(hasName("find")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_Metafunction) {
  std::string Code = R"cpp(
    template <typename T>
    struct Waldo {
      void find();
    };
    template <typename T>
    struct MetaWaldo {
      using Type = Waldo<T>;
    };
    template <typename T>
    void foo(typename MetaWaldo<T>::Type w) {
      w.find();
    }
  )cpp";
  // Test resolution of "find" in "w.find()".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("find")).bind("input"),
      cxxMethodDecl(hasName("find")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_Metafunction_Enumerator) {
  std::string Code = R"cpp(
    enum class State { Hidden };
    template <typename T>
    struct Meta {
      using Type = State;
    };
    template <typename T>
    void foo(typename Meta<T>::Type t) {
      t.Hidden;
    }
  )cpp";
  // Test resolution of "Hidden" in "t.Hidden".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("Hidden")).bind("input"),
      enumConstantDecl(hasName("Hidden")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_DeducedNonTypeTemplateParameter) {
  std::string Code = R"cpp(
    template <int N>
    struct Waldo {
      const int found = N;
    };
    template <Waldo W>
    int foo() {
      return W.found;
    }
  )cpp";
  // Test resolution of "found" in "W.found".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("found")).bind("input"),
      fieldDecl(hasName("found")).bind("output"));
}

TEST(HeuristicResolver, MemberExpr_HangIssue126536) {
  std::string Code = R"cpp(
    template <class T>
    void foo() {
      T bar;
      auto baz = (bar, bar);
      baz.foo();
    }
  )cpp";
  // Test resolution of "foo" in "baz.foo()".
  // Here, we are testing that we do not get into an infinite loop.
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("foo")).bind("input"));
}

TEST(HeuristicResolver, MemberExpr_DefaultTemplateArgument) {
  std::string Code = R"cpp(
    struct Default {
      void foo();
    };
    template <typename T = Default>
    void bar(T t) {
      t.foo();
    }
  )cpp";
  // Test resolution of "foo" in "t.foo()".
  expectResolution(
      Code, &HeuristicResolver::resolveMemberExpr,
      cxxDependentScopeMemberExpr(hasMemberName("foo")).bind("input"),
      cxxMethodDecl(hasName("foo")).bind("output"));
}

TEST(HeuristicResolver, DeclRefExpr_StaticMethod) {
  std::string Code = R"cpp(
    template <typename T>
    struct S {
      static void bar() {}
    };

    template <typename T>
    void foo() {
      S<T>::bar();
    }
  )cpp";
  // Test resolution of "bar" in "S<T>::bar()".
  expectResolution(
      Code, &HeuristicResolver::resolveDeclRefExpr,
      dependentScopeDeclRefExpr(hasDependentName("bar")).bind("input"),
      cxxMethodDecl(hasName("bar")).bind("output"));
}

TEST(HeuristicResolver, DeclRefExpr_StaticOverloads) {
  std::string Code = R"cpp(
    template <typename T>
    struct S {
      static void bar(int);
      static void bar(float);
    };

    template <typename T, typename U>
    void foo(U u) {
      S<T>::bar(u);
    }
  )cpp";
  // Test resolution of "bar" in "S<T>::bar(u)". Both overloads should be found.
  expectResolution(
      Code, &HeuristicResolver::resolveDeclRefExpr,
      dependentScopeDeclRefExpr(hasDependentName("bar")).bind("input"),
      cxxMethodDecl(hasName("bar"), hasParameter(0, hasType(asString("int"))))
          .bind("output"),
      cxxMethodDecl(hasName("bar"), hasParameter(0, hasType(asString("float"))))
          .bind("output"));
}

TEST(HeuristicResolver, DeclRefExpr_Enumerator) {
  std::string Code = R"cpp(
    template <typename T>
    struct Foo {
      enum class E { A, B };
      E e = E::A;
    };
  )cpp";
  // Test resolution of "A" in "E::A".
  expectResolution(
      Code, &HeuristicResolver::resolveDeclRefExpr,
      dependentScopeDeclRefExpr(hasDependentName("A")).bind("input"),
      enumConstantDecl(hasName("A")).bind("output"));
}

TEST(HeuristicResolver, DeclRefExpr_RespectScope) {
  std::string Code = R"cpp(
    template <typename Info>
    struct PointerIntPair {
      void *getPointer() const { return Info::getPointer(); }
    };
  )cpp";
  // Test resolution of "getPointer" in "Info::getPointer()".
  // Here, we are testing that we do not incorrectly get the enclosing
  // getPointer() function as a result.
  expectResolution(
      Code, &HeuristicResolver::resolveDeclRefExpr,
      dependentScopeDeclRefExpr(hasDependentName("getPointer")).bind("input"));
}

TEST(HeuristicResolver, DeclRefExpr_Nested) {
  std::string Code = R"cpp(
    struct S {
      static int Waldo;
    };
    template <typename T>
    struct Meta {
      using Type = S;
    };
    template <typename T>
    void foo() {
      Meta<T>::Type::Waldo;
    }
  )cpp";
  // Test resolution of "Waldo" in "Meta<T>::Type::Waldo".
  expectResolution(
      Code, &HeuristicResolver::resolveDeclRefExpr,
      dependentScopeDeclRefExpr(hasDependentName("Waldo")).bind("input"),
      varDecl(hasName("Waldo")).bind("output"));
}

TEST(HeuristicResolver, DependentNameType) {
  std::string Code = R"cpp(
    template <typename>
    struct A {
      struct B {};
    };
    template <typename T>
    void foo(typename A<T>::B);
  )cpp";
  // Tests resolution of "B" in "A<T>::B".
  expectResolution(
      Code, &HeuristicResolver::resolveDependentNameType,
      functionDecl(hasParameter(0, hasType(dependentNameType().bind("input")))),
      classTemplateDecl(
          has(cxxRecordDecl(has(cxxRecordDecl(hasName("B")).bind("output"))))));
}

TEST(HeuristicResolver, DependentNameType_Nested) {
  std::string Code = R"cpp(
    template <typename>
    struct A {
      struct B {
        struct C {};
      };
    };
    template <typename T>
    void foo(typename A<T>::B::C);
  )cpp";
  // Tests resolution of "C" in "A<T>::B::C".
  expectResolution(
      Code, &HeuristicResolver::resolveDependentNameType,
      functionDecl(hasParameter(0, hasType(dependentNameType().bind("input")))),
      classTemplateDecl(has(cxxRecordDecl(has(
          cxxRecordDecl(has(cxxRecordDecl(hasName("C")).bind("output"))))))));
}

TEST(HeuristicResolver, DependentNameType_Recursion) {
  std::string Code = R"cpp(
    template <int N>
    struct Waldo {
      using Type = typename Waldo<N - 1>::Type::Next;
    };
  )cpp";
  // Test resolution of "Next" in "typename Waldo<N - 1>::Type::Next".
  // Here, we are testing that we do not get into an infinite recursion.
  expectResolution(Code, &HeuristicResolver::resolveDependentNameType,
                   typeAliasDecl(hasType(dependentNameType().bind("input"))));
}

TEST(HeuristicResolver, DependentNameType_MutualRecursion) {
  std::string Code = R"cpp(
    template <int N>
    struct Odd;
    template <int N>
    struct Even {
      using Type = typename Odd<N - 1>::Type::Next;
    };
    template <int N>
    struct Odd {
      using Type = typename Even<N - 1>::Type::Next;
    };
  )cpp";
  // Test resolution of "Next" in "typename Even<N - 1>::Type::Next".
  // Similar to the above but we have two mutually recursive templates.
  expectResolution(
      Code, &HeuristicResolver::resolveDependentNameType,
      classTemplateDecl(hasName("Odd"),
                        has(cxxRecordDecl(has(typeAliasDecl(
                            hasType(dependentNameType().bind("input"))))))));
}

TEST(HeuristicResolver, NestedNameSpecifier) {
  // Test resolution of "B" in "A<T>::B::C".
  // Unlike the "C", the "B" does not get its own DependentNameTypeLoc node,
  // so the resolution uses the NestedNameSpecifier as input.
  std::string Code = R"cpp(
    template <typename>
    struct A {
      struct B {
        struct C {};
      };
    };
    template <typename T>
    void foo(typename A<T>::B::C);
  )cpp";
  // Adapt the call to resolveNestedNameSpecifierToType() to the interface
  // expected by expectResolution() (returning a vector of decls).
  ResolveFnT<NestedNameSpecifier> ResolveFn =
      [](const HeuristicResolver *H,
         const NestedNameSpecifier *NNS) -> std::vector<const NamedDecl *> {
    return {H->resolveNestedNameSpecifierToType(NNS)->getAsCXXRecordDecl()};
  };
  expectResolution(Code, ResolveFn,
                   nestedNameSpecifier(hasPrefix(specifiesType(hasDeclaration(
                                           classTemplateDecl(hasName("A"))))))
                       .bind("input"),
                   classTemplateDecl(has(cxxRecordDecl(
                       has(cxxRecordDecl(hasName("B")).bind("output"))))));
}

TEST(HeuristicResolver, TemplateSpecializationType) {
  std::string Code = R"cpp(
    template <typename>
    struct A {
      template <typename>
      struct B {};
    };
    template <typename T>
    void foo(typename A<T>::template B<int>);
  )cpp";
  // Test resolution of "B" in "A<T>::template B<int>".
  expectResolution(Code, &HeuristicResolver::resolveTemplateSpecializationType,
                   functionDecl(hasParameter(0, hasType(type().bind("input")))),
                   classTemplateDecl(has(cxxRecordDecl(
                       has(classTemplateDecl(hasName("B")).bind("output"))))));
}

TEST(HeuristicResolver, DependentCall_NonMember) {
  std::string Code = R"cpp(
    template <typename T>
    void nonmember(T);
    template <typename T>
    void bar(T t) {
      nonmember(t);
    }
  )cpp";
  // Test resolution of "nonmember" in "nonmember(t)".
  expectResolution(Code, &HeuristicResolver::resolveCalleeOfCallExpr,
                   callExpr(callee(unresolvedLookupExpr(hasAnyDeclaration(
                                functionTemplateDecl(hasName("nonmember"))))))
                       .bind("input"),
                   functionTemplateDecl(hasName("nonmember")).bind("output"));
}

TEST(HeuristicResolver, DependentCall_Member) {
  std::string Code = R"cpp(
    template <typename T>
    struct A {
      void member(T);
    };
    template <typename T>
    void bar(A<T> a, T t) {
      a.member(t);
    }
  )cpp";
  // Test resolution of "member" in "a.member(t)".
  expectResolution(
      Code, &HeuristicResolver::resolveCalleeOfCallExpr,
      callExpr(callee(cxxDependentScopeMemberExpr(hasMemberName("member"))))
          .bind("input"),
      cxxMethodDecl(hasName("member")).bind("output"));
}

TEST(HeuristicResolver, DependentCall_StaticMember) {
  std::string Code = R"cpp(
    template <typename T>
    struct A {
      static void static_member(T);
    };
    template <typename T>
    void bar(T t) {
      A<T>::static_member(t);
    }
  )cpp";
  // Test resolution of "static_member" in "A<T>::static_member(t)".
  expectResolution(Code, &HeuristicResolver::resolveCalleeOfCallExpr,
                   callExpr(callee(dependentScopeDeclRefExpr(
                                hasDependentName("static_member"))))
                       .bind("input"),
                   cxxMethodDecl(hasName("static_member")).bind("output"));
}

TEST(HeuristicResolver, DependentCall_Overload) {
  std::string Code = R"cpp(
    void overload(int);
    void overload(double);
    template <typename T>
    void bar(T t) {
      overload(t);
    }
  )cpp";
  // Test resolution of "overload" in "overload(t)". Both overload should be
  // found.
  expectResolution(Code, &HeuristicResolver::resolveCalleeOfCallExpr,
                   callExpr(callee(unresolvedLookupExpr(hasAnyDeclaration(
                                functionDecl(hasName("overload"))))))
                       .bind("input"),
                   functionDecl(hasName("overload"),
                                hasParameter(0, hasType(asString("double"))))
                       .bind("output"),
                   functionDecl(hasName("overload"),
                                hasParameter(0, hasType(asString("int"))))
                       .bind("output"));
}

TEST(HeuristicResolver, UsingValueDecl) {
  std::string Code = R"cpp(
    template <typename T>
    struct Base {
      void waldo();
    };
    template <typename T>
    struct Derived : Base<T> {
      using Base<T>::waldo;
    };
  )cpp";
  // Test resolution of "waldo" in "Base<T>::waldo".
  expectResolution(Code, &HeuristicResolver::resolveUsingValueDecl,
                   unresolvedUsingValueDecl(hasName("waldo")).bind("input"),
                   cxxMethodDecl(hasName("waldo")).bind("output"));
}

} // namespace
} // namespace clang
