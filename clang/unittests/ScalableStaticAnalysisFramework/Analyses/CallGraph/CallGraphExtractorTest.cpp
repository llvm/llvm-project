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
#include "gtest/gtest.h"
#include <cassert>

using namespace clang;
using namespace ssaf;

using llvm::Succeeded;

namespace {
AST_MATCHER(FunctionDecl, isPrimaryTemplate) {
  return Node.getDescribedFunctionTemplate() != nullptr;
}
} // namespace

static llvm::Expected<const FunctionDecl *> findFn(ASTContext &Ctx,
                                                   StringRef FnName) {
  using namespace ast_matchers;
  auto Matcher =
      functionDecl(hasName(FnName), unless(isPrimaryTemplate())).bind("decl");
  auto Matches = match(Matcher, Ctx);
  if (Matches.empty())
    return llvm::createStringError(
        "No FunctionDecl definition was found with name '" + FnName + "'");
  auto *FD = Matches[0].getNodeAs<FunctionDecl>("decl");
  assert(FD);
  return FD->getCanonicalDecl();
}

namespace {

struct CallGraphExtractorTest : ssaf::TestFixture {
  TUSummary Summary =
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp");
  TUSummaryBuilder Builder = TUSummaryBuilder(Summary);

  /// Creates the AST and extractor, then extracts the summaries from the AST.
  /// This will update the \c AST \c Builder and \c Summary data members.
  void runExtractor(StringRef Code) {
    AST = tooling::buildASTFromCode(Code);
    auto Consumer = makeTUSummaryExtractor("CallGraph", Builder);
    Consumer->HandleTranslationUnit(AST->getASTContext());
  }

  /// Tries to find the \c CallGraphSummary for the \p FnName function.
  llvm::Expected<const CallGraphSummary *>
  findSummary(llvm::StringRef FnName) const;

  /// Collects the USRs of all direct callees in CallGraphSummary \p S.
  std::set<std::string> getDirectCalleeUSRs(const CallGraphSummary *S) const;

  /// Looks up the Decls for \p FnNames, and then transforms those into USRs.
  llvm::Expected<std::set<std::string>>
  asUSRs(llvm::ArrayRef<StringRef> FnNames);

  /// Creates a GTest matcher selecting the direct callees of summary \p S.
  auto matchCalleeUSRs(const CallGraphSummary *S) const {
    return llvm::HasValue(testing::Eq(getDirectCalleeUSRs(S)));
  }

private:
  std::unique_ptr<ASTUnit> AST;
};

llvm::Expected<const CallGraphSummary *>
CallGraphExtractorTest::findSummary(llvm::StringRef FnName) const {
  auto MaybeFD = findFn(AST->getASTContext(), FnName);
  if (!MaybeFD)
    return MaybeFD.takeError();

  std::optional<EntityName> EntName = getEntityName(*MaybeFD);
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

std::set<std::string>
CallGraphExtractorTest::getDirectCalleeUSRs(const CallGraphSummary *S) const {
  const std::set<EntityId> &DirectCallees = S->DirectCallees;
  std::set<std::string> USRs;

  auto GatherCalleeUSRs = [&](const EntityName &Name, EntityId Id) {
    if (llvm::is_contained(DirectCallees, Id))
      USRs.insert(TestFixture::getUSR(Name));
  };
  TestFixture::getIdTable(Summary).forEach(GatherCalleeUSRs);
  assert(DirectCallees.size() == USRs.size());
  return USRs;
}

llvm::Expected<std::set<std::string>>
CallGraphExtractorTest::asUSRs(llvm::ArrayRef<StringRef> FnNames) {
  std::set<std::string> USRs;
  ASTContext &Ctx = AST->getASTContext();
  for (StringRef FnName : FnNames) {
    auto MaybeFD = findFn(Ctx, FnName);
    if (!MaybeFD)
      return MaybeFD.takeError();
    std::optional<EntityName> Name = getEntityName(MaybeFD.get());
    if (!Name.has_value()) {
      return llvm::createStringError("Failed to get the USR of '" + FnName +
                                     "'");
    }
    USRs.insert(getUSR(Name.value()));
  }
  assert(USRs.size() == FnNames.size());
  return USRs;
}

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

  const CallGraphSummary *S;
  ASSERT_THAT_ERROR(findSummary("calls_a_and_b").moveInto(S), Succeeded());
  EXPECT_FALSE(S->HasIndirectCalls);
  EXPECT_THAT_EXPECTED(asUSRs({"a", "b"}), matchCalleeUSRs(S));
}

TEST_F(CallGraphExtractorTest, NoCallees) {
  runExtractor(R"cpp(
    void leaf() {}
  )cpp");

  const CallGraphSummary *S;
  ASSERT_THAT_ERROR(findSummary("leaf").moveInto(S), Succeeded());
  EXPECT_FALSE(S->HasIndirectCalls);
  EXPECT_TRUE(S->DirectCallees.empty());
}

TEST_F(CallGraphExtractorTest, TransitiveCalls) {
  runExtractor(R"cpp(
    void c() { /*empty*/ }
    void b() { c(); }
    void a() { b(); }
  )cpp");

  { // a calls b (not c — we only record direct callees).
    const CallGraphSummary *SA;
    ASSERT_THAT_ERROR(findSummary("a").moveInto(SA), Succeeded());
    EXPECT_FALSE(SA->HasIndirectCalls);
    EXPECT_THAT_EXPECTED(asUSRs({"b"}), matchCalleeUSRs(SA));
  }

  { // b calls c.
    const CallGraphSummary *SB;
    ASSERT_THAT_ERROR(findSummary("b").moveInto(SB), Succeeded());
    EXPECT_FALSE(SB->HasIndirectCalls);
    EXPECT_THAT_EXPECTED(asUSRs({"c"}), matchCalleeUSRs(SB));
  }

  { // c calls nothing.
    const CallGraphSummary *SC;
    ASSERT_THAT_ERROR(findSummary("c").moveInto(SC), Succeeded());
    EXPECT_FALSE(SC->HasIndirectCalls);
    EXPECT_TRUE(SC->DirectCallees.empty());
  }
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
  const CallGraphSummary *S;
  ASSERT_THAT_ERROR(findSummary("caller").moveInto(S), Succeeded());

  // Virtual calls are treated as indirect calls.
  EXPECT_TRUE(S->HasIndirectCalls);

  // Virtual calls should not appear in DirectCallees.
  EXPECT_THAT_EXPECTED(asUSRs({}), matchCalleeUSRs(S));
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

  const CallGraphSummary *S;
  ASSERT_THAT_ERROR(findSummary("caller").moveInto(S), Succeeded());
  EXPECT_TRUE(S->HasIndirectCalls);
  EXPECT_THAT_EXPECTED(asUSRs({"direct_target"}), matchCalleeUSRs(S));
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

  const CallGraphSummary *S;
  ASSERT_THAT_ERROR(findSummary("caller").moveInto(S), Succeeded());
  EXPECT_FALSE(S->HasIndirectCalls);

  // Despite three calls, there's only one unique callee.
  EXPECT_THAT_EXPECTED(asUSRs({"target"}), matchCalleeUSRs(S));
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

  const CallGraphSummary *S;
  ASSERT_THAT_ERROR(findSummary("caller").moveInto(S), Succeeded());
  EXPECT_FALSE(S->HasIndirectCalls);
  EXPECT_THAT_EXPECTED(asUSRs({"method"}), matchCalleeUSRs(S));
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

  const CallGraphSummary *S;
  ASSERT_THAT_ERROR(findSummary("caller").moveInto(S), Succeeded());
  EXPECT_FALSE(S->HasIndirectCalls);
  EXPECT_THAT_EXPECTED(asUSRs({"staticMethod"}), matchCalleeUSRs(S));
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

  {
    const CallGraphSummary *S;
    ASSERT_THAT_ERROR(findSummary("caller").moveInto(S), Succeeded());
    EXPECT_FALSE(S->HasIndirectCalls);
    EXPECT_THAT_EXPECTED(
        asUSRs({"caller", "callee_with_def", "callee_without_def"}),
        matchCalleeUSRs(S));

    EXPECT_EQ(S->Definition.File, "input.cc");
    EXPECT_EQ(S->Definition.Line, 4U);
    EXPECT_EQ(S->Definition.Column, 10U);
  }

  {
    const CallGraphSummary *S;
    ASSERT_THAT_ERROR(findSummary("callee_with_def").moveInto(S), Succeeded());
    EXPECT_FALSE(S->HasIndirectCalls);
    EXPECT_TRUE(S->DirectCallees.empty());

    EXPECT_EQ(S->Definition.File, "input.cc");
    EXPECT_EQ(S->Definition.Line, 2U);
    EXPECT_EQ(S->Definition.Column, 10U);
  }
}

TEST_F(CallGraphExtractorTest, PrettyName) {
  runExtractor(R"cpp(
    template <class T, int N>
    void templated_function(int *) {}
    void caller(int n) {
      templated_function<struct TypeTag, 404>(&n);
    }
  )cpp");

  {
    const CallGraphSummary *S;
    ASSERT_THAT_ERROR(findSummary("caller").moveInto(S), Succeeded());
    EXPECT_FALSE(S->HasIndirectCalls);
    EXPECT_THAT_EXPECTED(asUSRs({"templated_function"}), matchCalleeUSRs(S));
    EXPECT_EQ(S->PrettyName, "caller(int)");
  }

  {
    const CallGraphSummary *S;
    ASSERT_THAT_ERROR(findSummary("templated_function").moveInto(S),
                      Succeeded());
    EXPECT_FALSE(S->HasIndirectCalls);
    EXPECT_TRUE(S->DirectCallees.empty());
    // FIXME: The template arguments are not spelled here.
    EXPECT_EQ(S->PrettyName, "templated_function(int *)");
  }
}

} // namespace
