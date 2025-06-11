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

SmallVector<char> GetUSR(const FunctionDecl *FD) {
  SmallVector<char> USR;
  index::generateUSRForDecl(FD, USR);
  return USR;
}
} // namespace

void FunctionSummary::addCall(const clang::FunctionDecl *FD) {
  Calls.emplace(GetUSR(FD));
}

FunctionSummary::FunctionSummary(SmallVector<char> ID, std::set<const SummaryAttribute *> FunctionAttrs, std::set<SmallVector<char>> Calls) :
  ID(std::move(ID)), FunctionAttrs(std::move(FunctionAttrs)), Calls(std::move(Calls)) {}

FunctionSummary::FunctionSummary(const clang::FunctionDecl *FD) : ID(GetUSR(FD)) {}

SummaryManager::SummaryManager() {
  Attributes.emplace_back(std::make_unique<NoWriteGlobalDescription>());

  for(auto &&Attr : Attributes) {
    assert(KindToAttribute.count(Attr->getKind()) == 0 && "Attr already registered");
    KindToAttribute[Attr->getKind()] = Attr.get();
  }
}

void SummaryManager::SaveSummary(std::unique_ptr<FunctionSummary> Summary) {
  auto *SummaryPtr = FunctionSummaries.emplace_back(std::move(Summary)).get();
  IDToSummary[SummaryPtr->getID()] = SummaryPtr;
}

const FunctionSummary *SummaryManager::GetSummary(const FunctionDecl *FD) const { 
  auto USR = GetUSR(FD);
  if(!IDToSummary.count(USR))
    return nullptr;

  return IDToSummary.at(USR);
}

void SummaryManager::SummarizeFunctionBody(const FunctionDecl *FD) {
  auto Summary = std::make_unique<FunctionSummary>(FD);
  CallCollector::CollectCalledFunctions(FD, *Summary);

  for (auto &&Attr : Attributes) {
    if (Attr->infer(FD))
      Summary->addAttribute(Attr.get());
  }

  SaveSummary(std::move(Summary));
}

void SummaryManager::SerializeSummary(llvm::json::OStream &JOS, const FunctionSummary &Summary) const {
  JOS.object([&]{
    JOS.attribute("id", llvm::json::Value(Summary.getID()));
    JOS.attributeObject("attrs", [&]{
      JOS.attributeArray("function", [&]{
        for(auto &&Attr : Summary.getFunctionAttrs()) {
          JOS.value(llvm::json::Value(Attr->serialize()));
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
    std::set<const SummaryAttribute *> FunctionAttrs;
    llvm::json::Array *FunctionAttributes = Summary->getObject("attrs")->getArray("function");
    for(auto attrIt = FunctionAttributes->begin(); attrIt != FunctionAttributes->end(); ++attrIt) {
      for(auto &&Attr : Attributes) {
        if(Attr->parse(*attrIt->getAsString()))
          FunctionAttrs.emplace(Attr.get());
      }
    }

    std::set<SmallVector<char>> Calls;
    llvm::json::Array *CallEntries = Summary->getArray("calls");
    for(auto callIt = CallEntries->begin(); callIt != CallEntries->end(); ++callIt) {
      auto *Obj = callIt->getAsObject();
      Calls.emplace(SmallString<128>(*Obj->getString("id")));
    }
    
    SaveSummary(std::make_unique<FunctionSummary>(std::move(ID), std::move(FunctionAttrs), std::move(Calls)));
  }
}

bool SummaryManager::ReduceFunctionSummary(FunctionSummary &Function) {
  bool changed = false;

  for (auto &&call : Function.getCalls()) {
    std::set<const SummaryAttribute *> reducedAttrs;

    // If we don't have a summary about a called function, we forget
    // everything about the current one as well.
    if (!IDToSummary.count(call)) {
      Function.replaceAttributes(std::move(reducedAttrs));
      return true;
    }

    const FunctionSummary *callSummary = IDToSummary[call];

    for (auto &&Attr : Function.getFunctionAttrs()) {
      if (Attr->merge(*callSummary))
        reducedAttrs.emplace(Attr);
    }

    if (reducedAttrs != Function.getFunctionAttrs()) {
      Function.replaceAttributes(std::move(reducedAttrs));
      changed = true;
    }
  }

  return changed;
}

void SummaryManager::ReduceSummaries() {
  bool changed = true;
  while (changed) {
    changed = false;

    for (auto &&Function : FunctionSummaries)
      changed |= ReduceFunctionSummary(*Function);
  }
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
  TheSummaryManager->SummarizeFunctionBody(FD);

  if(TheSummaryConsumer)
    TheSummaryConsumer->ProcessFunctionSummary(*TheSummaryManager->GetSummary(FD));
}

} // namespace clang
