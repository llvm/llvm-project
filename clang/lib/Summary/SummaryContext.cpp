#include "clang/Summary/SummaryContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Summary/SummaryAttribute.h"
#include "clang/Summary/SummaryConsumer.h"
#include <set>

namespace clang {
namespace {
std::string GetUSR(const FunctionDecl *FD) {
  SmallString<32> USR;
  index::generateUSRForDecl(FD, USR);
  return USR.str().str();
}

class CallCollector : public ast_matchers::MatchFinder::MatchCallback {
  SummaryContext *Context;
  std::set<size_t> Calls;
  bool callsOpaqueSymbol = false;

  virtual void
  run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
    if (!Call)
      return;

    const auto *Callee =
        llvm::dyn_cast_or_null<FunctionDecl>(Call->getCalleeDecl());
    if (!Callee) {
      callsOpaqueSymbol = true;
      return;
    }

    if (Result.SourceManager->isInSystemHeader(Callee->getLocation()) ||
        Callee->getBuiltinID())
      return;

    if (const auto *MD = llvm::dyn_cast<CXXMethodDecl>(Callee);
        MD && MD->isVirtual()) {
      callsOpaqueSymbol = true;
      return;
    }

    Calls.emplace(Context->GetOrInsertStoredIdentifierIdx(GetUSR(Callee)));
  }

public:
  CallCollector(SummaryContext &Context) : Context(&Context) {}

  std::pair<std::set<size_t>, bool> collect(const FunctionDecl *FD) {
    using namespace ast_matchers;
    MatchFinder Finder;

    Finder.addMatcher(functionDecl(forEachDescendant(callExpr().bind("call"))),
                      this);
    Finder.match(*FD, FD->getASTContext());

    return {Calls, callsOpaqueSymbol};
  }
};
} // namespace

FunctionSummary::FunctionSummary(size_t ID,
                                 std::set<const SummaryAttr *> FunctionAttrs,
                                 std::set<size_t> Calls, bool CallsOpaque)
    : ID(ID), Attrs(std::move(FunctionAttrs)), Calls(std::move(Calls)),
      CallsOpaque(CallsOpaque) {}

template <typename T> void SummaryContext::RegisterAttr() {
  std::unique_ptr<T> attr(new T());
  SummaryAttrKind Kind = attr->getKind();

  if (KindToAttribute.count(Kind))
    return;

  if (!Attributes.empty())
    assert(Attributes.back()->getKind() == Kind - 1 &&
           "attributes are not stored continously");

  KindToAttribute[Kind] = Attributes.emplace_back(std::move(attr)).get();
}

SummaryContext::SummaryContext() { RegisterAttr<NoWriteGlobalAttr>(); }

size_t SummaryContext::GetOrInsertStoredIdentifierIdx(StringRef ID) {
  auto &&[Element, Inserted] =
      IdentifierToID.try_emplace(ID.str(), IdentifierToID.size());
  if (Inserted)
    Identifiers.emplace_back(Element->first);

  return Element->second;
}

std::optional<size_t>
SummaryContext::GetStoredIdentifierIdx(StringRef ID) const {
  if (IdentifierToID.count(ID.str()))
    return IdentifierToID.at(ID.str());

  return std::nullopt;
}

void SummaryContext::CreateSummary(size_t ID,
                                   std::set<const SummaryAttr *> Attrs,
                                   std::set<size_t> Calls, bool CallsOpaque) {
  if (IDToSummary.count(ID))
    return;

  auto Summary = std::make_unique<FunctionSummary>(
      ID, std::move(Attrs), std::move(Calls), CallsOpaque);
  auto *SummaryPtr = FunctionSummaries.emplace_back(std::move(Summary)).get();
  IDToSummary[SummaryPtr->getID()] = SummaryPtr;
}

const FunctionSummary *
SummaryContext::GetSummary(const FunctionDecl *FD) const {
  std::optional<size_t> ID = GetStoredIdentifierIdx(GetUSR(FD));
  return ID ? IDToSummary.at(*ID) : nullptr;
}

void SummaryContext::SummarizeFunctionBody(const FunctionDecl *FD) {
  std::set<const SummaryAttr *> Attrs;

  for (auto &&Attr : Attributes) {
    if (Attr->infer(FD))
      Attrs.emplace(Attr.get());
  }

  auto [CollectedCalls, Opaque] = CallCollector(*this).collect(FD);
  CreateSummary(GetOrInsertStoredIdentifierIdx(GetUSR(FD)), std::move(Attrs),
                std::move(CollectedCalls), Opaque);
}

bool SummaryContext::ReduceFunctionSummary(FunctionSummary &Function) {
  bool changed = false;

  for (auto &&call : Function.getCalls()) {
    std::set<const SummaryAttr *> reducedAttrs;

    const FunctionSummary *callSummary =
        IDToSummary.count(call) ? IDToSummary[call] : nullptr;
    for (auto &&Attr : Attributes) {
      if (Attr->merge(Function, callSummary))
        reducedAttrs.emplace(Attr.get());
    }

    if (reducedAttrs != Function.getAttributes()) {
      Function.replaceAttributes(std::move(reducedAttrs));
      changed = true;
    }
  }

  return changed;
}

void SummaryContext::ReduceSummaries() {
  bool changed = true;
  while (changed) {
    changed = false;

    for (auto &&Function : FunctionSummaries)
      changed |= ReduceFunctionSummary(*Function);
  }
}
} // namespace clang
