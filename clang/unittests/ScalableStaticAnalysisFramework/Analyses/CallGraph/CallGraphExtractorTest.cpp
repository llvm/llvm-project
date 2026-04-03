//===- CallGraphExtractorTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFixture.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/CallGraph/CallGraphSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>

using namespace clang;
using namespace ssaf;

namespace {
AST_MATCHER(FunctionDecl, isPrimaryTemplate) {
  return Node.getDescribedFunctionTemplate() != nullptr;
}
} // namespace

static llvm::Expected<const NamedDecl *> findDecl(ASTContext &Ctx,
                                                  StringRef FnName) {
  using namespace ast_matchers;
  auto Matcher =
      functionDecl(hasName(FnName), unless(isPrimaryTemplate())).bind("decl");
  auto Matches = match(Matcher, Ctx);
  if (Matches.empty())
    return llvm::createStringError("No definition was found with name '" +
                                   FnName + "'");
  auto *ND = Matches[0].template getNodeAs<NamedDecl>("decl");
  assert(ND);
  return cast<NamedDecl>(ND->getCanonicalDecl());
}

// ============================================================================
// PrintTo overload for readable failure messages.
// Must live in the same namespace as Location (clang::ssaf) for ADL.
// ============================================================================

namespace clang::ssaf {
void PrintTo(const CallGraphSummary::Location &Loc, std::ostream *OS) {
  *OS << Loc.File << ":" << Loc.Line << ":" << Loc.Column;
}
void PrintTo(const CallGraphSummary &S, std::ostream *OS) {
  *OS << "CallGraphSummary { PrettyName: '" << S.PrettyName << "'"
      << ", Definition: ";
  PrintTo(S.Definition, OS);
  *OS << ", DirectCallees: " << S.DirectCallees.size()
      << ", VirtualCallees: " << S.VirtualCallees.size() << " }";
}
} // namespace clang::ssaf

namespace {

MATCHER_P3(DefinedAt, File, Line, Column,
           std::string(negation ? "is not" : "is") + " defined at " +
               std::string(File) + ":" + testing::PrintToString(Line) + ":" +
               testing::PrintToString(Column)) {
  const auto &D = arg.Definition;
  if (D.File != File || D.Line != Line || D.Column != Column) {
    *result_listener << "defined at " << D.File << ":" << D.Line << ":"
                     << D.Column;
    return false;
  }
  return true;
}

MATCHER_P(HasPrettyName, Name,
          std::string(negation ? "doesn't have" : "has") + " pretty name '" +
              testing::PrintToString(Name) + "'") {
  if (arg.PrettyName != std::string(Name)) {
    *result_listener << "has pretty name '" << arg.PrettyName << "'";
    return false;
  }
  return true;
}

MATCHER(HasNoDirectCallees,
        std::string(negation ? "has" : "has no") + " direct callees") {
  if (!arg.DirectCallees.empty()) {
    *result_listener << "has " << arg.DirectCallees.size()
                     << " direct callee(s)";
    return false;
  }
  return true;
}

MATCHER(HasNoVirtualCallees,
        std::string(negation ? "has" : "has no") + " virtual callees") {
  if (!arg.VirtualCallees.empty()) {
    *result_listener << "has " << arg.VirtualCallees.size()
                     << " virtual callee(s)";
    return false;
  }
  return true;
}

template <typename... Matchers> auto hasSummaryThat(const Matchers &...Ms) {
  using namespace testing;
  return llvm::HasValue(Pointee(AllOf(std::move(Ms)...)));
}

// ============================================================================
// Test fixture
// ============================================================================

struct CallGraphExtractorTest : ssaf::TestFixture {
  TUSummary Summary =
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp");
  TUSummaryBuilder Builder = TUSummaryBuilder(Summary);

  /// Creates the AST and extractor, then extracts the summaries from the AST.
  /// This will update the \c AST \c Builder and \c Summary data members.
  void runExtractor(StringRef Code, ArrayRef<std::string> Args = {}) {
    AST = tooling::buildASTFromCodeWithArgs(Code, Args);
    auto Consumer = makeTUSummaryExtractor("CallGraph", Builder);
    Consumer->HandleTranslationUnit(AST->getASTContext());
  }

  /// Tries to find the \c CallGraphSummary for the \p FnName function.
  llvm::Expected<const CallGraphSummary *>
  findSummary(llvm::StringRef FnName) const;

  /// Matcher factory: matches a summary whose direct callees are exactly the
  /// given set of function names (resolved to USRs via the entity table).
  /// Uses \c testing::ResultOf to transform the summary's EntityId set into
  /// USR strings before comparing with \c testing::ContainerEq.
  auto hasDirectCallees(llvm::ArrayRef<StringRef> Names)
      -> testing::Matcher<const CallGraphSummary &> {
    auto MaybeUSRs = asUSRs(Names);
    if (!MaybeUSRs) {
      ADD_FAILURE() << "Failed to resolve callee names to USRs: "
                    << llvm::toString(MaybeUSRs.takeError());
      return testing::An<const CallGraphSummary &>();
    }
    std::set<std::string> ExpectedUSRs = std::move(*MaybeUSRs);
    return testing::ResultOf(
        "direct callees",
        [this](const CallGraphSummary &S) {
          return getUSRsForCallees(S.DirectCallees);
        },
        testing::ContainerEq(ExpectedUSRs));
  }

  /// Matcher factory: same as \c hasDirectCallees but for virtual callees.
  auto hasVirtualCallees(llvm::ArrayRef<StringRef> Names)
      -> testing::Matcher<const CallGraphSummary &> {
    auto MaybeUSRs = asUSRs(Names);
    if (!MaybeUSRs) {
      ADD_FAILURE() << "Failed to resolve callee names to USRs: "
                    << llvm::toString(MaybeUSRs.takeError());
      return testing::A<const CallGraphSummary &>();
    }
    std::set<std::string> ExpectedUSRs = std::move(*MaybeUSRs);
    return testing::ResultOf(
        "virtual callees",
        [this](const CallGraphSummary &S) {
          return getUSRsForCallees(S.VirtualCallees);
        },
        testing::ContainerEq(ExpectedUSRs));
  }

private:
  std::unique_ptr<ASTUnit> AST;

  std::set<std::string>
  getUSRsForCallees(const std::set<EntityId> &Callees) const;

  /// Looks up the Decls for \p FnNames, and then transforms those into USRs.
  llvm::Expected<std::set<std::string>>
  asUSRs(llvm::ArrayRef<StringRef> FnNames);
};

llvm::Expected<const CallGraphSummary *>
CallGraphExtractorTest::findSummary(llvm::StringRef FnName) const {
  auto MaybeDecl = findDecl(AST->getASTContext(), FnName);
  if (!MaybeDecl)
    return MaybeDecl.takeError();

  std::optional<EntityName> EntName = getEntityName(*MaybeDecl);
  if (!EntName.has_value()) {
    return llvm::createStringError("Failed to create an entity name for '" +
                                   FnName + "'");
  }

  const auto &EntitiesTable = getEntities(getIdTable(Summary));
  auto It = EntitiesTable.find(EntName.value());
  if (It == EntitiesTable.end()) {
    return llvm::createStringError(
        "No entity ID was present in the entity table for '" + FnName + "'");
  }
  EntityId ID = It->second;
  auto &Data = getData(Summary);
  auto SummaryIt = Data.find(SummaryName("CallGraph"));
  if (SummaryIt == Data.end())
    return llvm::createStringError("There is no 'CallGraph' summary");
  auto EntityIt = SummaryIt->second.find(ID);
  if (EntityIt == SummaryIt->second.end()) {
    return llvm::createStringError(
        "There is no 'CallGraph' summary for entity ID " +
        std::to_string(getIndex(ID)) + " aka. '" + FnName + "'");
  }
  return static_cast<const CallGraphSummary *>(EntityIt->second.get());
}

std::set<std::string> CallGraphExtractorTest::getUSRsForCallees(
    const std::set<EntityId> &Callees) const {
  std::set<std::string> USRs;

  auto GatherCalleeUSRs = [&](const EntityName &Name, EntityId Id) {
    if (llvm::is_contained(Callees, Id))
      USRs.insert(TestFixture::getUSR(Name));
  };
  TestFixture::getIdTable(Summary).forEach(GatherCalleeUSRs);
  assert(Callees.size() == USRs.size());
  return USRs;
}

llvm::Expected<std::set<std::string>>
CallGraphExtractorTest::asUSRs(llvm::ArrayRef<StringRef> FnNames) {
  std::set<std::string> USRs;
  ASTContext &Ctx = AST->getASTContext();
  for (StringRef FnName : FnNames) {
    auto MaybeDecl = findDecl(Ctx, FnName);
    if (!MaybeDecl)
      return MaybeDecl.takeError();
    std::optional<EntityName> Name = getEntityName(MaybeDecl.get());
    if (!Name.has_value()) {
      return llvm::createStringError("Failed to get the USR of '" + FnName +
                                     "'");
    }
    USRs.insert(getUSR(Name.value()));
  }
  assert(USRs.size() == FnNames.size());
  return USRs;
}

// ============================================================================
// Tests
// ============================================================================

TEST_F(CallGraphExtractorTest, SimpleFunctionCalls) {
  runExtractor(R"cpp(
    void a();
    void b();
    void calls_a_and_b(bool coin) {
      if (coin)
        a();
      else
        b();
    }
  )cpp");

  ASSERT_THAT_EXPECTED(
      findSummary("calls_a_and_b"),
      hasSummaryThat(hasDirectCallees({"a", "b"}), HasNoVirtualCallees()));
}

TEST_F(CallGraphExtractorTest, NoCallees) {
  runExtractor(R"cpp(
    void leaf() {}
  )cpp");

  ASSERT_THAT_EXPECTED(
      findSummary("leaf"),
      hasSummaryThat(HasNoDirectCallees(), HasNoVirtualCallees()));
}

TEST_F(CallGraphExtractorTest, TransitiveCalls) {
  runExtractor(R"cpp(
    void c() { /*empty*/ }
    void b() { c(); }
    void a() { b(); }
  )cpp");

  // a calls b (not c — we only record direct callees).
  ASSERT_THAT_EXPECTED(findSummary("a"), hasSummaryThat(hasDirectCallees({"b"}),
                                                        HasNoVirtualCallees()));

  // b calls c.
  ASSERT_THAT_EXPECTED(findSummary("b"), hasSummaryThat(hasDirectCallees({"c"}),
                                                        HasNoVirtualCallees()));

  // c calls nothing.
  ASSERT_THAT_EXPECTED(findSummary("c"), hasSummaryThat(HasNoDirectCallees(),
                                                        HasNoVirtualCallees()));
}

TEST_F(CallGraphExtractorTest, VirtualCallsAreImprecise) {
  runExtractor(R"cpp(
    struct Base {
      virtual void virt();
    };
    struct Derived : Base {
      void virt() override;
    };
    void caller(Base &Obj) {
      Obj.virt();
    }
  )cpp");

  ASSERT_THAT_EXPECTED(
      findSummary("caller"),
      hasSummaryThat(HasNoDirectCallees(), hasVirtualCallees({"Base::virt"})));
}

TEST_F(CallGraphExtractorTest, MixedDirectAndVirtualCalls) {
  runExtractor(R"cpp(
    void direct_target();
    struct Base {
      virtual void virt();
    };
    void caller(Base &Obj) {
      direct_target();
      Obj.virt();
    }
  )cpp");

  ASSERT_THAT_EXPECTED(findSummary("caller"),
                       hasSummaryThat(hasDirectCallees({"direct_target"}),
                                      hasVirtualCallees({"Base::virt"})));
}

TEST_F(CallGraphExtractorTest, DeclarationsOnlyNoSummary) {
  runExtractor(R"cpp(
    void declared_only();
  )cpp");

  // No summary for functions without definitions.
  EXPECT_FALSE(llvm::is_contained(getData(Summary), SummaryName("CallGraph")));
}

TEST_F(CallGraphExtractorTest, DuplicateCallees) {
  runExtractor(R"cpp(
    void target();
    void caller() {
      target();
      target();
      target();
    }
  )cpp");

  // Despite three calls, there's only one unique callee.
  ASSERT_THAT_EXPECTED(
      findSummary("caller"),
      hasSummaryThat(hasDirectCallees({"target"}), HasNoVirtualCallees()));
}

TEST_F(CallGraphExtractorTest, NonVirtualMethodCalls) {
  runExtractor(R"cpp(
    struct S {
      void method();
    };
    void caller() {
      S s;
      s.method();
    }
  )cpp");

  ASSERT_THAT_EXPECTED(
      findSummary("caller"),
      hasSummaryThat(hasDirectCallees({"method"}), HasNoVirtualCallees()));
}

TEST_F(CallGraphExtractorTest, StaticMethodCalls) {
  runExtractor(R"cpp(
    struct S {
      static void staticMethod();
    };
    void caller() {
      S::staticMethod();
    }
  )cpp");

  ASSERT_THAT_EXPECTED(findSummary("caller"),
                       hasSummaryThat(hasDirectCallees({"staticMethod"}),
                                      HasNoVirtualCallees()));
}

TEST_F(CallGraphExtractorTest, FunctionPtrCall) {
  runExtractor(R"cpp(
    void caller(int (&fptr)()) {
      fptr();
    }
  )cpp");

  ASSERT_THAT_EXPECTED(
      findSummary("caller"),
      hasSummaryThat(HasNoDirectCallees(), HasNoVirtualCallees()));
}

TEST_F(CallGraphExtractorTest, ObjCMessageExprs) {
  runExtractor(R"cpp(
    @interface NSString
    - (id)stringByAppendingString:(id)str;
    @end

    void caller(void) {
        id msg = [@"Hello" stringByAppendingString:@", World!"];
    }
  )cpp",
               {"-x", "objective-c"});

  ASSERT_THAT_EXPECTED(
      findSummary("caller"),
      hasSummaryThat(HasNoDirectCallees(), HasNoVirtualCallees()));
}

TEST_F(CallGraphExtractorTest, DefinitionLocation) {
  runExtractor(R"cpp(
    void callee_with_def() {}
    void callee_without_def();
    void caller(int n) {
      if (n == 0) return;
      callee_with_def();
      callee_without_def();
      caller(n - 1);
    }
  )cpp");

  ASSERT_THAT_EXPECTED(
      findSummary("caller"),
      hasSummaryThat(
          hasDirectCallees({"caller", "callee_with_def", "callee_without_def"}),
          HasNoVirtualCallees(), DefinedAt("input.cc", 4U, 10U)));

  ASSERT_THAT_EXPECTED(findSummary("callee_with_def"),
                       hasSummaryThat(HasNoDirectCallees(),
                                      HasNoVirtualCallees(),
                                      DefinedAt("input.cc", 2U, 10U)));
}

TEST_F(CallGraphExtractorTest, PrettyName) {
  runExtractor(R"cpp(
    template <class T, int N>
    void templated_function(int *) {}
    void caller(int n) {
      templated_function<struct TypeTag, 404>(&n);
    }
  )cpp");

  ASSERT_THAT_EXPECTED(findSummary("caller"),
                       hasSummaryThat(hasDirectCallees({"templated_function"}),
                                      HasNoVirtualCallees(),
                                      HasPrettyName("caller(int)")));

  // FIXME: The template arguments are not spelled here.
  ASSERT_THAT_EXPECTED(
      findSummary("templated_function"),
      hasSummaryThat(HasNoDirectCallees(), HasNoVirtualCallees(),
                     HasPrettyName("templated_function(int *)")));
}

} // namespace
