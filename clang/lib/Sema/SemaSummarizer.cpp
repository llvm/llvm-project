#include "clang/Sema/SemaSummarizer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Sema/SummaryConsumer.h"
#include "clang/Sema/SummaryAttribute.h"
#include <set>
#include <fstream>
#include <sstream>

namespace clang {
namespace {
class CallCollector : public ast_matchers::MatchFinder::MatchCallback {
  FunctionSummary *Summary;

  CallCollector(FunctionSummary &Summary) : Summary(&Summary) {}

  virtual void
  run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
    if (!Call)
      return;

    const auto *Callee = llvm::dyn_cast<FunctionDecl>(Call->getCalleeDecl());
    Summary->addCall(Callee);
  }

public:
  static void CollectCalledFunctions(const FunctionDecl *FD,
                                     FunctionSummary &Summary) {
    using namespace ast_matchers;
    MatchFinder Finder;
    CallCollector CC(Summary);

    Finder.addMatcher(functionDecl(forEachDescendant(callExpr().bind("call"))),
                      &CC);
    Finder.match(*FD, FD->getASTContext());
  }
};

class NoWriteGlobalDescription : public SummaryAttributeDescription {
  class Callback : public ast_matchers::MatchFinder::MatchCallback {
  public:
    bool WriteGlobal = false;

    void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
      const auto *Assignment =
          Result.Nodes.getNodeAs<BinaryOperator>("assignment");
      if (!Assignment)
        return;

      WriteGlobal = true;
    };
  };

public:
  NoWriteGlobalDescription()
      : SummaryAttributeDescription(NO_WRITE_GLOBAL, "no_write_global") {}

  bool predicate(const FunctionDecl *FD) override {
    using namespace ast_matchers;
    MatchFinder Finder;
    Callback CB;

    Finder.addMatcher(
        functionDecl(forEachDescendant(
            binaryOperator(isAssignmentOperator(),
                           hasLHS(declRefExpr(to(varDecl(hasGlobalStorage())))))
                .bind("assignment"))),
        &CB);
    Finder.match(*FD, FD->getASTContext());
    return !CB.WriteGlobal;
  };

  bool merge(FunctionSummary &Summary) override { return true; };
};
} // namespace

void FunctionSummary::addCall(const clang::FunctionDecl *FD) {
  SmallVector<char> Call;
  index::generateUSRForDecl(FD, Call);
  Calls.emplace(Call);
}

FunctionSummary::FunctionSummary(SmallVector<char> ID, std::set<SummaryAttribute> FunctionAttrs, std::set<SmallVector<char>> Calls) :
  ID(std::move(ID)), FunctionAttrs(std::move(FunctionAttrs)), Calls(std::move(Calls)) {}

FunctionSummary::FunctionSummary(const clang::FunctionDecl *FD) {
  index::generateUSRForDecl(FD, ID);
}

SummaryManager::SummaryManager() {
  AttributeDescriptions.emplace_back(std::make_unique<NoWriteGlobalDescription>());

  for(auto &&AttrDescr : AttributeDescriptions)
    AttrToDescription[AttrDescr->getAttribute()] = AttrDescr.get();
}

FunctionSummary SummaryManager::SummarizeFunctionBody(const FunctionDecl *FD) {
  auto Summary = std::make_unique<FunctionSummary>(FD);
  CallCollector::CollectCalledFunctions(FD, *Summary);

  for (auto &&AttrDesc : AttributeDescriptions) {
    if (const auto &Attr = AttrDesc->infer(FD))
      Summary->addAttribute(*Attr);
  }

  // FIXME: This is duplicated and hurts my eyes regardless
  std::string key(Summary->getID().begin(), Summary->getID().size());
  auto *SummaryPtr = FunctionSummaries.emplace_back(std::move(Summary)).get();
  IDToSummary[key] = SummaryPtr;
  return *SummaryPtr;
}

void SummaryManager::SerializeSummary(llvm::json::OStream &JOS, const FunctionSummary &Summary) const {
  JOS.object([&]{
    JOS.attribute("id", llvm::json::Value(Summary.getID()));
    JOS.attributeObject("attrs", [&]{
      JOS.attributeArray("function", [&]{
        for(auto &&Attr : Summary.getFunctionAttrs()) {
          JOS.value(llvm::json::Value(AttributeDescriptions[Attr]->serialize()));
        }
      });
    });
    JOS.attributeArray("calls", [&]{
      for(auto &&Call : Summary.getCalls()) {
        JOS.object([&]{
          JOS.attribute("id", llvm::json::Value(Call));
        });
      }
    });
  });
}

void SummaryManager::ParseSummaryFromJSON(StringRef path) {
  std::ifstream t(path.str());
  std::stringstream buffer;
  buffer << t.rdbuf();

  auto JSON = llvm::json::parse(buffer.str());
  if (!JSON)
    return;

  llvm::json::Array *Summaries = JSON->getAsArray();
  for(auto it = Summaries->begin(); it != Summaries->end(); ++it) {
    llvm::json::Object *Summary = it->getAsObject();

    SmallString<128> ID(*Summary->getString("id"));
    std::set<SummaryAttribute> FunctionAttrs;
    llvm::json::Array *FunctionAttributes = Summary->getObject("attrs")->getArray("function");
    for(auto attrIt = FunctionAttributes->begin(); attrIt != FunctionAttributes->end(); ++attrIt) {
      for(auto &&AttrDesc : AttributeDescriptions) {
        if(auto Attr = AttrDesc->parse(*attrIt->getAsString()))
          FunctionAttrs.emplace(*Attr);
      }
    }

    std::set<SmallVector<char>> Calls;
    llvm::json::Array *CallEntries = Summary->getArray("calls");
    for(auto callIt = CallEntries->begin(); callIt != CallEntries->end(); ++callIt) {
      auto *Obj = callIt->getAsObject();
      Calls.emplace(SmallString<128>(*Obj->getString("id")));
    }
    
    std::string key = ID.str().str();
    auto ParsedSummary = std::make_unique<FunctionSummary>(std::move(ID), std::move(FunctionAttrs), std::move(Calls));
    auto *ParsedSummaryPtr = FunctionSummaries.emplace_back(std::move(ParsedSummary)).get();
    IDToSummary[key] = ParsedSummaryPtr;
  }
}

void SummaryManager::ReduceSummaries() {
  // FIXME: implement
}

void SemaSummarizer::ActOnStartOfSourceFile() {
  if(TheSummaryConsumer)
    TheSummaryConsumer->ProcessStartOfSourceFile();
}

void SemaSummarizer::ActOnEndOfSourceFile() {
  if(TheSummaryConsumer)
    TheSummaryConsumer->ProcessEndOfSourceFile();
}

void SemaSummarizer::SummarizeFunctionBody(const FunctionDecl *FD) {
  FunctionSummary Summary = TheSummaryManager->SummarizeFunctionBody(FD);

  if(TheSummaryConsumer)
    TheSummaryConsumer->ProcessFunctionSummary(Summary);
}

} // namespace clang
