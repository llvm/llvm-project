//=== unittests/Sema/CodeCompleteTest.cpp - Code Complete tests ==============//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace {

using namespace clang;
using namespace clang::tooling;
using ::testing::UnorderedElementsAre;

const char TestCCName[] = "test.cc";
using VisitedContextResults = std::vector<std::string>;

class VisitedContextFinder: public CodeCompleteConsumer {
public:
  VisitedContextFinder(VisitedContextResults &Results)
      : CodeCompleteConsumer(/*CodeCompleteOpts=*/{},
                             /*CodeCompleteConsumer*/ false),
        VCResults(Results),
        CCTUInfo(std::make_shared<GlobalCodeCompletionAllocator>()) {}

  void ProcessCodeCompleteResults(Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *Results,
                                  unsigned NumResults) override {
    VisitedContexts = Context.getVisitedContexts();
    VCResults = getVisitedNamespace();
  }

  CodeCompletionAllocator &getAllocator() override {
    return CCTUInfo.getAllocator();
  }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

  std::vector<std::string> getVisitedNamespace() const {
    std::vector<std::string> NSNames;
    for (const auto *Context : VisitedContexts)
      if (const auto *NS = llvm::dyn_cast<NamespaceDecl>(Context))
        NSNames.push_back(NS->getQualifiedNameAsString());
    return NSNames;
  }

private:
  VisitedContextResults& VCResults;
  CodeCompletionTUInfo CCTUInfo;
  CodeCompletionContext::VisitedContextSet VisitedContexts;
};

class CodeCompleteAction : public SyntaxOnlyAction {
public:
  CodeCompleteAction(ParsedSourceLocation P, VisitedContextResults &Results)
      : CompletePosition(std::move(P)), VCResults(Results) {}

  bool BeginInvocation(CompilerInstance &CI) override {
    CI.getFrontendOpts().CodeCompletionAt = CompletePosition;
    CI.setCodeCompletionConsumer(new VisitedContextFinder(VCResults));
    return true;
  }

private:
  // 1-based code complete position <Line, Col>;
  ParsedSourceLocation CompletePosition;
  VisitedContextResults& VCResults;
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

VisitedContextResults runCodeCompleteOnCode(StringRef Code) {
  VisitedContextResults Results;
  auto TokenOffset = Code.find('^');
  assert(TokenOffset != StringRef::npos &&
         "Completion token ^ wasn't found in Code.");
  std::string WithoutToken = Code.take_front(TokenOffset);
  WithoutToken += Code.drop_front(WithoutToken.size() + 1);
  assert(StringRef(WithoutToken).find('^') == StringRef::npos &&
         "expected exactly one completion token ^ inside the code");

  auto Action = llvm::make_unique<CodeCompleteAction>(
      offsetToPosition(WithoutToken, TokenOffset), Results);
  clang::tooling::runToolOnCodeWithArgs(Action.release(), Code, {"-std=c++11"},
                                        TestCCName);
  return Results;
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
  )cpp");
  EXPECT_THAT(VisitedNS, UnorderedElementsAre("foo", "ns1", "ns2", "ns3::nns3",
                                              "foo::(anonymous)"));
}

TEST(SemaCodeCompleteTest, VisitedNSForInvalideQualifiedId) {
  auto VisitedNS = runCodeCompleteOnCode(R"cpp(
     namespace ns { foo::^ }
  )cpp");
  EXPECT_TRUE(VisitedNS.empty());
}

} // namespace
