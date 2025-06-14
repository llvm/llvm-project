#ifndef LLVM_CLANG_SEMA_SEMASUMMARYCONTEXT_H
#define LLVM_CLANG_SEMA_SEMASUMMARYCONTEXT_H

#include "clang/Sema/SummaryAttribute.h"
#include "clang/Sema/SummaryConsumer.h"
#include <set>

namespace clang {
class FunctionSummary {
  SmallVector<char> ID;
  std::set<const SummaryAttribute *> Attrs;
  std::set<SmallVector<char>> Calls;

public:
  FunctionSummary(SmallVector<char> ID,
                  std::set<const SummaryAttribute *> Attrs,
                  std::set<SmallVector<char>> Calls);

  SmallVector<char> getID() const { return ID; }
  const std::set<const SummaryAttribute *> &getAttributes() const {
    return Attrs;
  }
  const std::set<SmallVector<char>> &getCalls() const { return Calls; }

  void replaceAttributes(std::set<const SummaryAttribute *> Attrs) {
    this->Attrs = std::move(Attrs);
  }
};

class SummaryContext {
  std::map<SmallVector<char>, const FunctionSummary *> IDToSummary;
  std::vector<std::unique_ptr<FunctionSummary>> FunctionSummaries;

  std::map<SummaryAttributeKind, const SummaryAttribute *> KindToAttribute;
  std::vector<std::unique_ptr<SummaryAttribute>> Attributes;

  void CreateSummary(SmallVector<char> ID,
                     std::set<const SummaryAttribute *> Attrs,
                     std::set<SmallVector<char>> Calls);
  bool ReduceFunctionSummary(FunctionSummary &FunctionSummary);

public:
  SummaryContext();

  const SummaryAttribute *GetAttribute(SummaryAttributeKind kind) const;
  const FunctionSummary *GetSummary(const FunctionDecl *FD) const;
  void SummarizeFunctionBody(const FunctionDecl *FD);

  void ParseSummaryFromJSON(const llvm::json::Array &Summary);
  void ReduceSummaries();
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASUMMARYCONTEXTH
