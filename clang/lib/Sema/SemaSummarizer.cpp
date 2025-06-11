#include "clang/Sema/SemaSummarizer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Sema/SummaryAttribute.h"
#include "clang/Sema/SummaryConsumer.h"
#include <set>

namespace clang {
namespace {
SmallVector<char> GetUSR(const FunctionDecl *FD) {
  SmallVector<char> USR;
  index::generateUSRForDecl(FD, USR);
  return USR;
}

class CallCollector : public ast_matchers::MatchFinder::MatchCallback {
  std::set<SmallVector<char>> Calls;

  virtual void
  run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
    if (!Call)
      return;

    const auto *Callee = llvm::dyn_cast<FunctionDecl>(Call->getCalleeDecl());
    Calls.emplace(GetUSR(Callee));
  }

public:
  std::set<SmallVector<char>> collect(const FunctionDecl *FD) {
    using namespace ast_matchers;
    MatchFinder Finder;

    Finder.addMatcher(functionDecl(forEachDescendant(callExpr().bind("call"))),
                      this);
    Finder.match(*FD, FD->getASTContext());

    return Calls;
  }
};
} // namespace

FunctionSummary::FunctionSummary(
    SmallVector<char> ID, std::set<const SummaryAttribute *> FunctionAttrs,
    std::set<SmallVector<char>> Calls)
    : ID(std::move(ID)), Attrs(std::move(FunctionAttrs)),
      Calls(std::move(Calls)) {}

SummaryManager::SummaryManager() {
  Attributes.emplace_back(std::make_unique<NoWriteGlobalDescription>());

  for (auto &&Attr : Attributes) {
    assert(KindToAttribute.count(Attr->getKind()) == 0 &&
           "Attr already registered");
    KindToAttribute[Attr->getKind()] = Attr.get();
  }
}

void SummaryManager::CreateSummary(SmallVector<char> ID,
                                   std::set<const SummaryAttribute *> Attrs,
                                   std::set<SmallVector<char>> Calls) {
  auto Summary = std::make_unique<FunctionSummary>(
      std::move(ID), std::move(Attrs), std::move(Calls));
  auto *SummaryPtr = FunctionSummaries.emplace_back(std::move(Summary)).get();
  IDToSummary[SummaryPtr->getID()] = SummaryPtr;
}

const FunctionSummary *
SummaryManager::GetSummary(const FunctionDecl *FD) const {
  auto USR = GetUSR(FD);
  return IDToSummary.count(USR) ? IDToSummary.at(USR) : nullptr;
}

void SummaryManager::SummarizeFunctionBody(const FunctionDecl *FD) {
  std::set<const SummaryAttribute *> Attrs;
  for (auto &&Attr : Attributes) {
    if (Attr->infer(FD))
      Attrs.emplace(Attr.get());
  }

  CreateSummary(GetUSR(FD), std::move(Attrs), CallCollector().collect(FD));
}

void SummaryManager::SerializeSummary(llvm::json::OStream &JOS, const FunctionSummary &Summary) const {
  JOS.object([&]{
    JOS.attribute("id", llvm::json::Value(Summary.getID()));
    JOS.attributeObject("attrs", [&] {
      JOS.attributeArray("function", [&] {
        for (auto &&Attr : Summary.getAttributes()) {
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

void SummaryManager::ParseSummaryFromJSON(const llvm::json::Array &Summary) {
  for (auto it = Summary.begin(); it != Summary.end(); ++it) {
    const llvm::json::Object *FunctionSummary = it->getAsObject();

    SmallString<128> ID(*FunctionSummary->getString("id"));
    std::set<const SummaryAttribute *> FunctionAttrs;
    const llvm::json::Array *FunctionAttributes =
        FunctionSummary->getObject("attrs")->getArray("function");
    for(auto attrIt = FunctionAttributes->begin(); attrIt != FunctionAttributes->end(); ++attrIt) {
      for (auto &&Attr : Attributes) {
        if (Attr->parse(*attrIt->getAsString()))
          FunctionAttrs.emplace(Attr.get());
      }
    }

    std::set<SmallVector<char>> Calls;
    const llvm::json::Array *CallEntries = FunctionSummary->getArray("calls");
    for(auto callIt = CallEntries->begin(); callIt != CallEntries->end(); ++callIt) {
      auto *Obj = callIt->getAsObject();
      Calls.emplace(SmallString<128>(*Obj->getString("id")));
    }

    CreateSummary(std::move(ID), std::move(FunctionAttrs), std::move(Calls));
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

    for (auto &&Attr : Function.getAttributes()) {
      if (Attr->merge(*callSummary))
        reducedAttrs.emplace(Attr);
    }

    if (reducedAttrs != Function.getAttributes()) {
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
    TheSummaryConsumer->ProcessFunctionSummary(
        *TheSummaryManager->GetSummary(FD));
}

} // namespace clang
