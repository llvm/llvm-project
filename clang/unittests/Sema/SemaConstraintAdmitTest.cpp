//===- unittests/Sema/SemaConstraintAdmitTest.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests Sema::constraintsAdmitNonProgramDefinedType, which decides whether a
// class template partial specialization's constraints permit a type that is
// not program-defined.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringMap.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

// Walks all class template partial specializations in namespace std and
// records the verdict of constraintsAdmitNonProgramDefinedType, keyed by the
// spelling of the specialization, e.g. "hash<type-parameter-0-0 *>".
class Visitor : public RecursiveASTVisitor<Visitor> {
public:
  Visitor(Sema &S, llvm::StringMap<bool> &Results) : S(S), Results(Results) {}

  bool VisitClassTemplatePartialSpecializationDecl(
      ClassTemplatePartialSpecializationDecl *PSD) {
    if (!PSD->getDeclContext()->isStdNamespace())
      return true;
    std::string Label = PSD->getName().str();
    llvm::raw_string_ostream OS(Label);
    PrintingPolicy PP = S.getPrintingPolicy();
    PP.PrintAsCanonical = false;
    printTemplateArgumentList(OS, PSD->getTemplateArgs().asArray(), PP);
    Results[Label] = S.constraintsAdmitNonProgramDefinedType(PSD);
    return true;
  }

private:
  Sema &S;
  llvm::StringMap<bool> &Results;
};

class Consumer : public SemaConsumer {
public:
  Consumer(llvm::StringMap<bool> &Results) : Results(Results) {}
  void InitializeSema(Sema &S) override { SemaPtr = &S; }
  void HandleTranslationUnit(ASTContext &Ctx) override {
    Visitor V(*SemaPtr, Results);
    V.TraverseDecl(Ctx.getTranslationUnitDecl());
  }

private:
  Sema *SemaPtr = nullptr;
  llvm::StringMap<bool> &Results;
};

class Action : public ASTFrontendAction {
public:
  Action(llvm::StringMap<bool> &Results) : Results(Results) {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &,
                                                 StringRef) override {
    return std::make_unique<Consumer>(Results);
  }

private:
  llvm::StringMap<bool> &Results;
};

llvm::StringMap<bool> run(StringRef Code) {
  llvm::StringMap<bool> Results;
  auto A = std::make_unique<Action>(Results);
  EXPECT_TRUE(
      tooling::runToolOnCodeWithArgs(std::move(A), Code, {"-std=c++20"}));
  return Results;
}

constexpr StringRef Preamble = R"cpp(
namespace std {
  template <class T> struct hash;
  template <decltype(sizeof(int)) I, class T> struct tuple_element;
  template <class Derived, class Base>
  concept derived_from = __is_base_of(Base, Derived);
  template <class T> concept integral = __is_integral(T);
}
namespace prog {
  struct Base {};
  template <class...> struct Foo {};
  template <class> struct is_Foo { static constexpr bool value = false; };
  template <class... A> struct is_Foo<Foo<A...>> { static constexpr bool value = true; };
  template <class T> concept FooType = is_Foo<T>::value;
  template <class T> concept HasTraitsType = requires { typename T::traits_type; };
  template <class T> concept HasKeyCompare = requires { typename T::key_compare; };
  template <class T> concept HasArrayTag = requires { typename T::array_tag; };
}
)cpp";

// A fake <string> where std::string is a typedef, as in libstdc++.
constexpr StringRef StringTypedef = R"cpp(
namespace std {
  template <class C> struct char_traits {};
  template <class C, class Tr = char_traits<C>> struct basic_string {
    using traits_type = Tr;
  };
  typedef basic_string<char> string;
}
)cpp";

// A fake <string> where std::string is an alias declaration, as in libc++.
constexpr StringRef StringAlias = R"cpp(
namespace std {
  template <class C> struct char_traits {};
  template <class C, class Tr = char_traits<C>> struct basic_string {
    using traits_type = Tr;
  };
  using string = basic_string<char>;
}
)cpp";

constexpr StringRef Map = R"cpp(
namespace std {
  template <class T> struct less {};
  template <class K, class V, class Cmp = less<K>> struct map {
    using key_compare = Cmp;
  };
}
)cpp";

constexpr StringRef ArrayAndTuple = R"cpp(
namespace std {
  template <class T, decltype(sizeof(int)) N> struct array {
    using array_tag = void;
  };
  template <class... Ts> struct tuple {};
}
)cpp";

llvm::StringMap<bool> runParts(ArrayRef<StringRef> Parts) {
  std::string Code;
  for (StringRef P : Parts)
    Code += P;
  return run(Code);
}

std::string dump(const llvm::StringMap<bool> &R) {
  std::string S;
  for (const auto &E : R)
    S += (E.getKey() + " = " + (E.getValue() ? "true" : "false") + "\n").str();
  return S;
}

} // namespace

TEST(ConstraintAdmit, ProgramDefinedConcept) {
  // Constrained to prog::Foo<...>; no non-program-defined type satisfies it.
  auto R = runParts({Preamble, R"cpp(
    template <prog::FooType T> struct std::hash<T> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0>")) << dump(R);
  EXPECT_FALSE(R["hash<type-parameter-0-0>"]);
}

TEST(ConstraintAdmit, DerivedFromProgramDefinedBase) {
  auto R = runParts({Preamble, R"cpp(
    template <class T> requires std::derived_from<T, prog::Base>
    struct std::hash<T *> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0 *>")) << dump(R);
  EXPECT_FALSE(R["hash<type-parameter-0-0 *>"]);
}

TEST(ConstraintAdmit, StdIntegralAdmitsBuiltin) {
  auto R = runParts({Preamble, R"cpp(
    template <std::integral T> struct std::hash<T[2]> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0[2]>")) << dump(R);
  EXPECT_TRUE(R["hash<type-parameter-0-0[2]>"]);
}

TEST(ConstraintAdmit, UnconstrainedAdmitsEverything) {
  auto R = runParts({Preamble, R"cpp(
    template <class T> struct std::hash<T[4]> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0[4]>")) << dump(R);
  EXPECT_TRUE(R["hash<type-parameter-0-0[4]>"]);
}

TEST(ConstraintAdmit, StdStringTypedefIsProbed) {
  auto R = runParts({Preamble, StringTypedef, R"cpp(
    template <prog::HasTraitsType T> struct std::hash<T *> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0 *>")) << dump(R);
  EXPECT_TRUE(R["hash<type-parameter-0-0 *>"]);
}

TEST(ConstraintAdmit, StdStringAliasIsProbed) {
  auto R = runParts({Preamble, StringAlias, R"cpp(
    template <prog::HasTraitsType T> struct std::hash<T *> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0 *>")) << dump(R);
  EXPECT_TRUE(R["hash<type-parameter-0-0 *>"]);
}

TEST(ConstraintAdmit, StdStringAbsentIsNotProbed) {
  // The same constraint, but std::string is not declared in this TU.
  auto R = runParts({Preamble, R"cpp(
    template <prog::HasTraitsType T> struct std::hash<T *> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0 *>")) << dump(R);
  EXPECT_FALSE(R["hash<type-parameter-0-0 *>"]);
}

TEST(ConstraintAdmit, StdMapIsProbed) {
  auto R = runParts({Preamble, Map, R"cpp(
    template <prog::HasKeyCompare T> struct std::hash<T *> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0 *>")) << dump(R);
  EXPECT_TRUE(R["hash<type-parameter-0-0 *>"]);
}

TEST(ConstraintAdmit, StdArrayIsProbed) {
  // Only std::array has array_tag; std::tuple<int> does not satisfy it.
  auto R = runParts({Preamble, ArrayAndTuple, R"cpp(
    template <prog::HasArrayTag T> struct std::hash<T *> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0 *>")) << dump(R);
  EXPECT_TRUE(R["hash<type-parameter-0-0 *>"]);
}

TEST(ConstraintAdmit, StdArrayAbsentIsNotProbed) {
  auto R = runParts({Preamble, R"cpp(
    namespace std { template <class... Ts> struct tuple {}; }
    template <prog::HasArrayTag T> struct std::hash<T *> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0 *>")) << dump(R);
  EXPECT_FALSE(R["hash<type-parameter-0-0 *>"]);
}

TEST(ConstraintAdmit, NonTypeAndTypeParameter) {
  auto R = runParts({Preamble, R"cpp(
    template <decltype(sizeof(int)) I, prog::FooType T>
    struct std::tuple_element<I, T> {};
    template <decltype(sizeof(int)) I, class T> requires (I < 2)
    struct std::tuple_element<I, const T> {};
  )cpp"});
  ASSERT_EQ(R.size(), 2u) << dump(R);
  for (const auto &E : R) {
    bool ConstrainsI = StringRef(E.getKey()).contains("const");
    EXPECT_EQ(E.getValue(), ConstrainsI) << E.getKey().str();
  }
}

TEST(ConstraintAdmit, TwoTypeParameters) {
  auto R = runParts({Preamble, R"cpp(
    template <class...> struct P {};
    template <class T, class U> requires prog::FooType<U>
    struct std::hash<P<T, U> *> {};
    template <class T, class U> requires (!__is_same(T, U))
    struct std::hash<P<T, U> &> {};
  )cpp"});
  ASSERT_EQ(R.size(), 2u) << dump(R);
  for (const auto &E : R) {
    bool IsRef = StringRef(E.getKey()).ends_with("&>");
    EXPECT_EQ(E.getValue(), IsRef) << E.getKey().str();
  }
}

TEST(ConstraintAdmit, Packs) {
  auto R = runParts({Preamble, R"cpp(
    template <class...> struct P {};
    // Satisfied by an empty pack.
    template <class... Ts> requires (prog::FooType<Ts> && ...)
    struct std::hash<P<Ts...> *> {};
    // Needs at least one program-defined type.
    template <class... Ts> requires (prog::FooType<Ts> || ...)
    struct std::hash<P<Ts...> &> {};
  )cpp"});
  ASSERT_EQ(R.size(), 2u) << dump(R);
  for (const auto &E : R) {
    bool IsPtr = StringRef(E.getKey()).ends_with("*>");
    EXPECT_EQ(E.getValue(), IsPtr) << E.getKey().str();
  }
}

TEST(ConstraintAdmit, HardErrorDuringProbingIsSilent) {
  // Box<void> is ill-formed outside the immediate context. The error must not
  // be reported (run() expects no errors), and it must not stop the remaining
  // probes.
  auto R = runParts({Preamble, R"cpp(
    namespace prog { template <class T> struct Box { T Member; }; }
    template <class T> requires (sizeof(prog::Box<T>) > 0) && prog::FooType<T>
    struct std::hash<T *> {};
    template <class T> requires (sizeof(prog::Box<T>) > 0)
    struct std::hash<T &> {};
  )cpp"});
  ASSERT_EQ(R.size(), 2u) << dump(R);
  EXPECT_FALSE(R["hash<type-parameter-0-0 *>"]);
  EXPECT_TRUE(R["hash<type-parameter-0-0 &>"]);
}

TEST(ConstraintAdmit, StdByteIsProbed) {
  auto R = runParts({Preamble, R"cpp(
    namespace std { enum class byte : unsigned char {}; }
    template <class T> requires __is_enum(T) struct std::hash<T *> {};
  )cpp"});
  ASSERT_TRUE(R.count("hash<type-parameter-0-0 *>")) << dump(R);
  EXPECT_TRUE(R["hash<type-parameter-0-0 *>"]);
}
