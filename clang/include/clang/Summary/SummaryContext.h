#ifndef LLVM_CLANG_SUMMARY_SUMMARYCONTEXT_H
#define LLVM_CLANG_SUMMARY_SUMMARYCONTEXT_H

#include "clang/Summary/SummaryAttribute.h"
#include "clang/Summary/SummaryConsumer.h"
#include "llvm/Support/YAMLTraits.h"
#include <set>

namespace clang {
class FunctionSummary {
  size_t ID;
  std::set<const SummaryAttr *> Attrs;
  std::set<size_t> Calls;
  bool CallsOpaque;

public:
  FunctionSummary(size_t ID, std::set<const SummaryAttr *> Attrs,
                  std::set<size_t> Calls, bool CallsOpaque);

  size_t getID() const { return ID; }
  const std::set<const SummaryAttr *> &getAttributes() const { return Attrs; }
  const std::set<size_t> &getCalls() const { return Calls; }
  bool callsOpaqueObject() const { return CallsOpaque; }

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
  std::map<std::string, size_t> IdentifierToID;
  std::vector<StringRef> Identifiers;

  std::map<size_t, const FunctionSummary *> IDToSummary;
  std::vector<std::unique_ptr<FunctionSummary>> FunctionSummaries;

  std::map<SummaryAttrKind, const SummaryAttr *> KindToAttribute;
  std::vector<std::unique_ptr<SummaryAttr>> Attributes;

  template <typename T> void RegisterAttr();

public:
  SummaryContext();

  size_t GetOrInsertStoredIdentifierIdx(StringRef ID);
  std::optional<size_t> GetStoredIdentifierIdx(StringRef ID) const;

  void CreateSummary(size_t ID, std::set<const SummaryAttr *> Attrs,
                     std::set<size_t> Calls, bool CallsOpaque);
  bool ReduceFunctionSummary(FunctionSummary &FunctionSummary);

  const std::vector<std::unique_ptr<FunctionSummary>> &GetSummaries() const {
    return FunctionSummaries;
  };
  const std::vector<std::unique_ptr<SummaryAttr>> &GetAttributes() const {
    return Attributes;
  };
  const std::vector<StringRef> &GetIdentifiers() const { return Identifiers; };

  const FunctionSummary *GetSummary(const FunctionDecl *FD) const;
  void SummarizeFunctionBody(const FunctionDecl *FD);
  void ReduceSummaries();
};
} // namespace clang

#endif // LLVM_CLANG_SUMMARY_SUMMARYCONTEXTH
