#ifndef LLVM_CLANG_SUMMARY_SUMMARYCONTEXT_H
#define LLVM_CLANG_SUMMARY_SUMMARYCONTEXT_H

#include "clang/Summary/SummaryAttribute.h"
#include "clang/Summary/SummaryConsumer.h"
#include <set>

namespace clang {
class FunctionSummary {
  SmallVector<char> ID;
  std::set<const SummaryAttr *> Attrs;
  std::set<SmallVector<char>> Calls;

public:
  FunctionSummary(SmallVector<char> ID, std::set<const SummaryAttr *> Attrs,
                  std::set<SmallVector<char>> Calls);

  SmallVector<char> getID() const { return ID; }
  const std::set<const SummaryAttr *> &getAttributes() const { return Attrs; }
  const std::set<SmallVector<char>> &getCalls() const { return Calls; }

  template <typename T> bool hasAttribute() const {
    for (auto &&attr : Attrs) {
      if (llvm::isa<T>(attr))
        return true;
    }

    return false;
  }

  void replaceAttributes(std::set<const SummaryAttr *> Attrs) {
    this->Attrs = std::move(Attrs);
  }
};

class SummaryContext {
  std::map<SmallVector<char>, const FunctionSummary *> IDToSummary;
  std::vector<std::unique_ptr<FunctionSummary>> FunctionSummaries;

  std::map<SummaryAttrKind, const SummaryAttr *> KindToAttribute;
  std::vector<std::unique_ptr<SummaryAttr>> Attributes;

  void CreateSummary(SmallVector<char> ID, std::set<const SummaryAttr *> Attrs,
                     std::set<SmallVector<char>> Calls);
  bool ReduceFunctionSummary(FunctionSummary &FunctionSummary);

  template <typename T> void registerAttr();

public:
  SummaryContext();

  const FunctionSummary *GetSummary(const FunctionDecl *FD) const;
  void SummarizeFunctionBody(const FunctionDecl *FD);

  void ParseSummaryFromJSON(const llvm::json::Array &Summary);
  void ReduceSummaries();
};
} // namespace clang

#endif // LLVM_CLANG_SUMMARY_SUMMARYCONTEXTH
