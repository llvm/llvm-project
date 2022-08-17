//===-- TypeHierarchyTests.cpp  ---------------------------*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AST.h"
#include "Annotations.h"
#include "Matchers.h"
#include "ParsedAST.h"
#include "TestFS.h"
#include "TestTU.h"
#include "XRefs.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "llvm/Support/Path.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// GMock helpers for matching TypeHierarchyItem.
MATCHER_P(withName, N, "") { return arg.name == N; }
MATCHER_P(withKind, Kind, "") { return arg.kind == Kind; }
MATCHER_P(selectionRangeIs, R, "") { return arg.selectionRange == R; }
template <class... ParentMatchers>
::testing::Matcher<TypeHierarchyItem> parents(ParentMatchers... ParentsM) {
  return Field(&TypeHierarchyItem::parents,
               HasValue(UnorderedElementsAre(ParentsM...)));
}
template <class... ChildMatchers>
::testing::Matcher<TypeHierarchyItem> children(ChildMatchers... ChildrenM) {
  return Field(&TypeHierarchyItem::children,
               HasValue(UnorderedElementsAre(ChildrenM...)));
}
// Note: "not resolved" is different from "resolved but empty"!
MATCHER(parentsNotResolved, "") { return !arg.parents; }
MATCHER(childrenNotResolved, "") { return !arg.children; }
MATCHER_P(withResolveID, SID, "") { return arg.symbolID.str() == SID; }
MATCHER_P(withResolveParents, M, "") {
  return testing::ExplainMatchResult(M, arg.data.parents, result_listener);
}

TEST(FindRecordTypeAt, TypeOrVariable) {
  Annotations Source(R"cpp(
struct Ch^ild2 {
  int c;
};

using A^lias = Child2;

int main() {
  Ch^ild2 ch^ild2;
  ch^ild2.c = 1;
}
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  for (Position Pt : Source.points()) {
    auto Records = findRecordTypeAt(AST, Pt);
    ASSERT_THAT(Records, SizeIs(1));
    EXPECT_EQ(&findDecl(AST, "Child2"),
              static_cast<const NamedDecl *>(Records.front()));
  }
}

TEST(FindRecordTypeAt, Method) {
  Annotations Source(R"cpp(
struct Child2 {
  void met^hod ();
  void met^hod (int x);
};

int main() {
  Child2 child2;
  child2.met^hod(5);
}
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  for (Position Pt : Source.points()) {
    auto Records = findRecordTypeAt(AST, Pt);
    ASSERT_THAT(Records, SizeIs(1));
    EXPECT_EQ(&findDecl(AST, "Child2"),
              static_cast<const NamedDecl *>(Records.front()));
  }
}

TEST(FindRecordTypeAt, Field) {
  Annotations Source(R"cpp(
struct Child2 {
  int fi^eld;
};

int main() {
  Child2 child2;
  child2.fi^eld = 5;
}
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  for (Position Pt : Source.points()) {
    // A field does not unambiguously specify a record type
    // (possible associated reocrd types could be the field's type,
    // or the type of the record that the field is a member of).
    EXPECT_THAT(findRecordTypeAt(AST, Pt), SizeIs(0));
  }
}

TEST(TypeParents, SimpleInheritance) {
  Annotations Source(R"cpp(
struct Parent {
  int a;
};

struct Child1 : Parent {
  int b;
};

struct Child2 : Child1 {
  int c;
};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  const CXXRecordDecl *Parent =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Parent"));
  const CXXRecordDecl *Child1 =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Child1"));
  const CXXRecordDecl *Child2 =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Child2"));

  EXPECT_THAT(typeParents(Parent), ElementsAre());
  EXPECT_THAT(typeParents(Child1), ElementsAre(Parent));
  EXPECT_THAT(typeParents(Child2), ElementsAre(Child1));
}

TEST(TypeParents, MultipleInheritance) {
  Annotations Source(R"cpp(
struct Parent1 {
  int a;
};

struct Parent2 {
  int b;
};

struct Parent3 : Parent2 {
  int c;
};

struct Child : Parent1, Parent3 {
  int d;
};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  const CXXRecordDecl *Parent1 =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Parent1"));
  const CXXRecordDecl *Parent2 =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Parent2"));
  const CXXRecordDecl *Parent3 =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Parent3"));
  const CXXRecordDecl *Child = dyn_cast<CXXRecordDecl>(&findDecl(AST, "Child"));

  EXPECT_THAT(typeParents(Parent1), ElementsAre());
  EXPECT_THAT(typeParents(Parent2), ElementsAre());
  EXPECT_THAT(typeParents(Parent3), ElementsAre(Parent2));
  EXPECT_THAT(typeParents(Child), ElementsAre(Parent1, Parent3));
}

TEST(TypeParents, ClassTemplate) {
  Annotations Source(R"cpp(
struct Parent {};

template <typename T>
struct Child : Parent {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  const CXXRecordDecl *Parent =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Parent"));
  const CXXRecordDecl *Child =
      dyn_cast<ClassTemplateDecl>(&findDecl(AST, "Child"))->getTemplatedDecl();

  EXPECT_THAT(typeParents(Child), ElementsAre(Parent));
}

MATCHER_P(implicitSpecOf, ClassTemplate, "") {
  const ClassTemplateSpecializationDecl *CTS =
      dyn_cast<ClassTemplateSpecializationDecl>(arg);
  return CTS &&
         CTS->getSpecializedTemplate()->getTemplatedDecl() == ClassTemplate &&
         CTS->getSpecializationKind() == TSK_ImplicitInstantiation;
}

// This is similar to findDecl(AST, QName), but supports using
// a template-id as a query.
const NamedDecl &findDeclWithTemplateArgs(ParsedAST &AST,
                                          llvm::StringRef Query) {
  return findDecl(AST, [&Query](const NamedDecl &ND) {
    std::string QName;
    llvm::raw_string_ostream OS(QName);
    PrintingPolicy Policy(ND.getASTContext().getLangOpts());
    // Use getNameForDiagnostic() which includes the template
    // arguments in the printed name.
    ND.getNameForDiagnostic(OS, Policy, /*Qualified=*/true);
    OS.flush();
    return QName == Query;
  });
}

TEST(TypeParents, TemplateSpec1) {
  Annotations Source(R"cpp(
template <typename T>
struct Parent {};

template <>
struct Parent<int> {};

struct Child1 : Parent<float> {};

struct Child2 : Parent<int> {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  const CXXRecordDecl *Parent =
      dyn_cast<ClassTemplateDecl>(&findDecl(AST, "Parent"))->getTemplatedDecl();
  const CXXRecordDecl *ParentSpec =
      dyn_cast<CXXRecordDecl>(&findDeclWithTemplateArgs(AST, "Parent<int>"));
  const CXXRecordDecl *Child1 =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Child1"));
  const CXXRecordDecl *Child2 =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Child2"));

  EXPECT_THAT(typeParents(Child1), ElementsAre(implicitSpecOf(Parent)));
  EXPECT_THAT(typeParents(Child2), ElementsAre(ParentSpec));
}

TEST(TypeParents, TemplateSpec2) {
  Annotations Source(R"cpp(
struct Parent {};

template <typename T>
struct Child {};

template <>
struct Child<int> : Parent {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  const CXXRecordDecl *Parent =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Parent"));
  const CXXRecordDecl *Child =
      dyn_cast<ClassTemplateDecl>(&findDecl(AST, "Child"))->getTemplatedDecl();
  const CXXRecordDecl *ChildSpec =
      dyn_cast<CXXRecordDecl>(&findDeclWithTemplateArgs(AST, "Child<int>"));

  EXPECT_THAT(typeParents(Child), ElementsAre());
  EXPECT_THAT(typeParents(ChildSpec), ElementsAre(Parent));
}

TEST(TypeParents, DependentBase) {
  Annotations Source(R"cpp(
template <typename T>
struct Parent {};

template <typename T>
struct Child1 : Parent<T> {};

template <typename T>
struct Child2 : Parent<T>::Type {};

template <typename T>
struct Child3 : T {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  const CXXRecordDecl *Parent =
      dyn_cast<ClassTemplateDecl>(&findDecl(AST, "Parent"))->getTemplatedDecl();
  const CXXRecordDecl *Child1 =
      dyn_cast<ClassTemplateDecl>(&findDecl(AST, "Child1"))->getTemplatedDecl();
  const CXXRecordDecl *Child2 =
      dyn_cast<ClassTemplateDecl>(&findDecl(AST, "Child2"))->getTemplatedDecl();
  const CXXRecordDecl *Child3 =
      dyn_cast<ClassTemplateDecl>(&findDecl(AST, "Child3"))->getTemplatedDecl();

  // For "Parent<T>", use the primary template as a best-effort guess.
  EXPECT_THAT(typeParents(Child1), ElementsAre(Parent));
  // For "Parent<T>::Type", there is nothing we can do.
  EXPECT_THAT(typeParents(Child2), ElementsAre());
  // Likewise for "T".
  EXPECT_THAT(typeParents(Child3), ElementsAre());
}

TEST(TypeParents, IncompleteClass) {
  Annotations Source(R"cpp(
    class Incomplete;
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  const CXXRecordDecl *Incomplete =
      dyn_cast<CXXRecordDecl>(&findDecl(AST, "Incomplete"));
  EXPECT_THAT(typeParents(Incomplete), IsEmpty());
}

// Parts of getTypeHierarchy() are tested in more detail by the
// FindRecordTypeAt.* and TypeParents.* tests above. This test exercises the
// entire operation.
TEST(TypeHierarchy, Parents) {
  Annotations Source(R"cpp(
struct $Parent1Def[[Parent1]] {
  int a;
};

struct $Parent2Def[[Parent2]] {
  int b;
};

struct $Parent3Def[[Parent3]] : Parent2 {
  int c;
};

struct Ch^ild : Parent1, Parent3 {
  int d;
};

int main() {
  Ch^ild  ch^ild;

  ch^ild.a = 1;
}
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  for (Position Pt : Source.points()) {
    // Set ResolveLevels to 0 because it's only used for Children;
    // for Parents, getTypeHierarchy() always returns all levels.
    auto Result = getTypeHierarchy(AST, Pt, /*ResolveLevels=*/0,
                                   TypeHierarchyDirection::Parents);
    ASSERT_THAT(Result, SizeIs(1));
    EXPECT_THAT(
        Result.front(),
        AllOf(
            withName("Child"), withKind(SymbolKind::Struct),
            parents(AllOf(withName("Parent1"), withKind(SymbolKind::Struct),
                          selectionRangeIs(Source.range("Parent1Def")),
                          parents()),
                    AllOf(withName("Parent3"), withKind(SymbolKind::Struct),
                          selectionRangeIs(Source.range("Parent3Def")),
                          parents(AllOf(
                              withName("Parent2"), withKind(SymbolKind::Struct),
                              selectionRangeIs(Source.range("Parent2Def")),
                              parents()))))));
  }
}

TEST(TypeHierarchy, RecursiveHierarchyUnbounded) {
  Annotations Source(R"cpp(
  template <int N>
  struct $SDef[[S]] : S<N + 1> {};

  S^<0> s; // error-ok
  )cpp");

  TestTU TU = TestTU::withCode(Source.code());
  TU.ExtraArgs.push_back("-ftemplate-depth=10");
  auto AST = TU.build();

  // The compiler should produce a diagnostic for hitting the
  // template instantiation depth.
  ASSERT_TRUE(!AST.getDiagnostics()->empty());

  // Make sure getTypeHierarchy() doesn't get into an infinite recursion.
  // The parent is reported as "S" because "S<0>" is an invalid instantiation.
  // We then iterate once more and find "S" again before detecting the
  // recursion.
  auto Result = getTypeHierarchy(AST, Source.points()[0], 0,
                                 TypeHierarchyDirection::Parents);
  ASSERT_THAT(Result, SizeIs(1));
  EXPECT_THAT(
      Result.front(),
      AllOf(withName("S<0>"), withKind(SymbolKind::Struct),
            parents(
                AllOf(withName("S"), withKind(SymbolKind::Struct),
                      selectionRangeIs(Source.range("SDef")),
                      parents(AllOf(withName("S"), withKind(SymbolKind::Struct),
                                    selectionRangeIs(Source.range("SDef")),
                                    parents()))))));
}

TEST(TypeHierarchy, RecursiveHierarchyBounded) {
  Annotations Source(R"cpp(
  template <int N>
  struct $SDef[[S]] : S<N - 1> {};

  template <>
  struct S<0>{};

  S$SRefConcrete^<2> s;

  template <int N>
  struct Foo {
    S$SRefDependent^<N> s;
  };)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();

  // Make sure getTypeHierarchy() doesn't get into an infinite recursion
  // for either a concrete starting point or a dependent starting point.
  auto Result = getTypeHierarchy(AST, Source.point("SRefConcrete"), 0,
                                 TypeHierarchyDirection::Parents);
  ASSERT_THAT(Result, SizeIs(1));
  EXPECT_THAT(
      Result.front(),
      AllOf(withName("S<2>"), withKind(SymbolKind::Struct),
            parents(AllOf(
                withName("S<1>"), withKind(SymbolKind::Struct),
                selectionRangeIs(Source.range("SDef")),
                parents(AllOf(withName("S<0>"), withKind(SymbolKind::Struct),
                              parents()))))));
  Result = getTypeHierarchy(AST, Source.point("SRefDependent"), 0,
                            TypeHierarchyDirection::Parents);
  ASSERT_THAT(Result, SizeIs(1));
  EXPECT_THAT(
      Result.front(),
      AllOf(withName("S"), withKind(SymbolKind::Struct),
            parents(AllOf(withName("S"), withKind(SymbolKind::Struct),
                          selectionRangeIs(Source.range("SDef")), parents()))));
}

TEST(TypeHierarchy, DeriveFromImplicitSpec) {
  Annotations Source(R"cpp(
  template <typename T>
  struct Parent {};

  struct Child1 : Parent<int> {};

  struct Child2 : Parent<char> {};

  Parent<int> Fo^o;
  )cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  auto Result = getTypeHierarchy(AST, Source.points()[0], 2,
                                 TypeHierarchyDirection::Children, Index.get(),
                                 testPath(TU.Filename));
  ASSERT_THAT(Result, SizeIs(1));
  EXPECT_THAT(Result.front(),
              AllOf(withName("Parent"), withKind(SymbolKind::Struct),
                    children(AllOf(withName("Child1"),
                                   withKind(SymbolKind::Struct), children()),
                             AllOf(withName("Child2"),
                                   withKind(SymbolKind::Struct), children()))));
}

TEST(TypeHierarchy, DeriveFromPartialSpec) {
  Annotations Source(R"cpp(
  template <typename T> struct Parent {};
  template <typename T> struct Parent<T*> {};

  struct Child : Parent<int*> {};

  Parent<int> Fo^o;
  )cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  auto Result = getTypeHierarchy(AST, Source.points()[0], 2,
                                 TypeHierarchyDirection::Children, Index.get(),
                                 testPath(TU.Filename));
  ASSERT_THAT(Result, SizeIs(1));
  EXPECT_THAT(Result.front(), AllOf(withName("Parent"),
                                    withKind(SymbolKind::Struct), children()));
}

TEST(TypeHierarchy, DeriveFromTemplate) {
  Annotations Source(R"cpp(
  template <typename T>
  struct Parent {};

  template <typename T>
  struct Child : Parent<T> {};

  Parent<int> Fo^o;
  )cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  // FIXME: We'd like this to show the implicit specializations Parent<int>
  //        and Child<int>, but currently libIndex does not expose relationships
  //        between implicit specializations.
  auto Result = getTypeHierarchy(AST, Source.points()[0], 2,
                                 TypeHierarchyDirection::Children, Index.get(),
                                 testPath(TU.Filename));
  ASSERT_THAT(Result, SizeIs(1));
  EXPECT_THAT(Result.front(),
              AllOf(withName("Parent"), withKind(SymbolKind::Struct),
                    children(AllOf(withName("Child"),
                                   withKind(SymbolKind::Struct), children()))));
}

TEST(TypeHierarchy, Preamble) {
  Annotations SourceAnnotations(R"cpp(
struct Ch^ild : Parent {
  int b;
};)cpp");

  Annotations HeaderInPreambleAnnotations(R"cpp(
struct [[Parent]] {
  int a;
};)cpp");

  TestTU TU = TestTU::withCode(SourceAnnotations.code());
  TU.HeaderCode = HeaderInPreambleAnnotations.code().str();
  auto AST = TU.build();

  std::vector<TypeHierarchyItem> Result = getTypeHierarchy(
      AST, SourceAnnotations.point(), 1, TypeHierarchyDirection::Parents);

  ASSERT_THAT(Result, SizeIs(1));
  EXPECT_THAT(
      Result.front(),
      AllOf(withName("Child"),
            parents(AllOf(withName("Parent"),
                          selectionRangeIs(HeaderInPreambleAnnotations.range()),
                          parents()))));
}

SymbolID findSymbolIDByName(SymbolIndex *Index, llvm::StringRef Name,
                            llvm::StringRef TemplateArgs = "") {
  SymbolID Result;
  FuzzyFindRequest Request;
  Request.Query = std::string(Name);
  Request.AnyScope = true;
  bool GotResult = false;
  Index->fuzzyFind(Request, [&](const Symbol &S) {
    if (TemplateArgs == S.TemplateSpecializationArgs) {
      EXPECT_FALSE(GotResult);
      Result = S.ID;
      GotResult = true;
    }
  });
  EXPECT_TRUE(GotResult);
  return Result;
}

std::vector<SymbolID> collectSubtypes(SymbolID Subject, SymbolIndex *Index) {
  std::vector<SymbolID> Result;
  RelationsRequest Req;
  Req.Subjects.insert(Subject);
  Req.Predicate = RelationKind::BaseOf;
  Index->relations(Req,
                   [&Result](const SymbolID &Subject, const Symbol &Object) {
                     Result.push_back(Object.ID);
                   });
  return Result;
}

TEST(Subtypes, SimpleInheritance) {
  Annotations Source(R"cpp(
struct Parent {};
struct Child1a : Parent {};
struct Child1b : Parent {};
struct Child2 : Child1a {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto Index = TU.index();

  SymbolID Parent = findSymbolIDByName(Index.get(), "Parent");
  SymbolID Child1a = findSymbolIDByName(Index.get(), "Child1a");
  SymbolID Child1b = findSymbolIDByName(Index.get(), "Child1b");
  SymbolID Child2 = findSymbolIDByName(Index.get(), "Child2");

  EXPECT_THAT(collectSubtypes(Parent, Index.get()),
              UnorderedElementsAre(Child1a, Child1b));
  EXPECT_THAT(collectSubtypes(Child1a, Index.get()), ElementsAre(Child2));
}

TEST(Subtypes, MultipleInheritance) {
  Annotations Source(R"cpp(
struct Parent1 {};
struct Parent2 {};
struct Parent3 : Parent2 {};
struct Child : Parent1, Parent3 {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto Index = TU.index();

  SymbolID Parent1 = findSymbolIDByName(Index.get(), "Parent1");
  SymbolID Parent2 = findSymbolIDByName(Index.get(), "Parent2");
  SymbolID Parent3 = findSymbolIDByName(Index.get(), "Parent3");
  SymbolID Child = findSymbolIDByName(Index.get(), "Child");

  EXPECT_THAT(collectSubtypes(Parent1, Index.get()), ElementsAre(Child));
  EXPECT_THAT(collectSubtypes(Parent2, Index.get()), ElementsAre(Parent3));
  EXPECT_THAT(collectSubtypes(Parent3, Index.get()), ElementsAre(Child));
}

TEST(Subtypes, ClassTemplate) {
  Annotations Source(R"cpp(
struct Parent {};

template <typename T>
struct Child : Parent {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto Index = TU.index();

  SymbolID Parent = findSymbolIDByName(Index.get(), "Parent");
  SymbolID Child = findSymbolIDByName(Index.get(), "Child");

  EXPECT_THAT(collectSubtypes(Parent, Index.get()), ElementsAre(Child));
}

TEST(Subtypes, TemplateSpec1) {
  Annotations Source(R"cpp(
template <typename T>
struct Parent {};

template <>
struct Parent<int> {};

struct Child1 : Parent<float> {};

struct Child2 : Parent<int> {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto Index = TU.index();

  SymbolID Parent = findSymbolIDByName(Index.get(), "Parent");
  SymbolID ParentSpec = findSymbolIDByName(Index.get(), "Parent", "<int>");
  SymbolID Child1 = findSymbolIDByName(Index.get(), "Child1");
  SymbolID Child2 = findSymbolIDByName(Index.get(), "Child2");

  EXPECT_THAT(collectSubtypes(Parent, Index.get()), ElementsAre(Child1));
  EXPECT_THAT(collectSubtypes(ParentSpec, Index.get()), ElementsAre(Child2));
}

TEST(Subtypes, TemplateSpec2) {
  Annotations Source(R"cpp(
struct Parent {};

template <typename T>
struct Child {};

template <>
struct Child<int> : Parent {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto Index = TU.index();

  SymbolID Parent = findSymbolIDByName(Index.get(), "Parent");
  SymbolID ChildSpec = findSymbolIDByName(Index.get(), "Child", "<int>");

  EXPECT_THAT(collectSubtypes(Parent, Index.get()), ElementsAre(ChildSpec));
}

TEST(Subtypes, DependentBase) {
  Annotations Source(R"cpp(
template <typename T>
struct Parent {};

template <typename T>
struct Child : Parent<T> {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto Index = TU.index();

  SymbolID Parent = findSymbolIDByName(Index.get(), "Parent");
  SymbolID Child = findSymbolIDByName(Index.get(), "Child");

  EXPECT_THAT(collectSubtypes(Parent, Index.get()), ElementsAre(Child));
}

TEST(Subtypes, LazyResolution) {
  Annotations Source(R"cpp(
struct P^arent {};
struct Child1 : Parent {};
struct Child2a : Child1 {};
struct Child2b : Child1 {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  auto Result = getTypeHierarchy(AST, Source.point(), /*ResolveLevels=*/1,
                                 TypeHierarchyDirection::Children, Index.get(),
                                 testPath(TU.Filename));
  ASSERT_THAT(Result, SizeIs(1));
  EXPECT_THAT(
      Result.front(),
      AllOf(withName("Parent"), withKind(SymbolKind::Struct), parents(),
            children(AllOf(withName("Child1"), withKind(SymbolKind::Struct),
                           parentsNotResolved(), childrenNotResolved()))));

  resolveTypeHierarchy((*Result.front().children)[0], /*ResolveLevels=*/1,
                       TypeHierarchyDirection::Children, Index.get());

  EXPECT_THAT(
      (*Result.front().children)[0],
      AllOf(withName("Child1"), withKind(SymbolKind::Struct),
            parentsNotResolved(),
            children(AllOf(withName("Child2a"), withKind(SymbolKind::Struct),
                           parentsNotResolved(), childrenNotResolved()),
                     AllOf(withName("Child2b"), withKind(SymbolKind::Struct),
                           parentsNotResolved(), childrenNotResolved()))));
}

TEST(Standard, SubTypes) {
  Annotations Source(R"cpp(
struct Pare^nt1 {};
struct Parent2 {};
struct Child : Parent1, Parent2 {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  auto Result = getTypeHierarchy(AST, Source.point(), /*ResolveLevels=*/1,
                                 TypeHierarchyDirection::Children, Index.get(),
                                 testPath(TU.Filename));
  ASSERT_THAT(Result, SizeIs(1));
  auto Children = subTypes(Result.front(), Index.get());

  // Make sure parents are populated when getting children.
  // FIXME: This is partial.
  EXPECT_THAT(
      Children,
      UnorderedElementsAre(
          AllOf(withName("Child"),
                withResolveParents(HasValue(UnorderedElementsAre(withResolveID(
                    getSymbolID(&findDecl(AST, "Parent1")).str())))))));
}

TEST(Standard, SuperTypes) {
  Annotations Source(R"cpp(
struct Parent {};
struct Chil^d : Parent {};
)cpp");

  TestTU TU = TestTU::withCode(Source.code());
  auto AST = TU.build();
  auto Index = TU.index();

  auto Result = getTypeHierarchy(AST, Source.point(), /*ResolveLevels=*/1,
                                 TypeHierarchyDirection::Children, Index.get(),
                                 testPath(TU.Filename));
  ASSERT_THAT(Result, SizeIs(1));
  auto Parents = superTypes(Result.front(), Index.get());

  EXPECT_THAT(Parents, HasValue(UnorderedElementsAre(
                           AllOf(withName("Parent"),
                                 withResolveParents(HasValue(IsEmpty()))))));
}
} // namespace
} // namespace clangd
} // namespace clang
