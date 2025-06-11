#ifndef LLVM_CLANG_SEMA_SEMASUMMARIZER_H
#define LLVM_CLANG_SEMA_SEMASUMMARIZER_H

#include "clang/Sema/SemaBase.h"
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

  friend class SummaryManager;
};

class SummaryManager {
  std::map<SmallVector<char>, const FunctionSummary *> IDToSummary;
  std::vector<std::unique_ptr<FunctionSummary>> FunctionSummaries;

  std::map<SummaryAttributeKind, const SummaryAttribute *> KindToAttribute;
  std::vector<std::unique_ptr<SummaryAttribute>> Attributes;

  void CreateSummary(SmallVector<char> ID,
                     std::set<const SummaryAttribute *> Attrs,
                     std::set<SmallVector<char>> Calls);
  bool ReduceFunctionSummary(FunctionSummary &FunctionSummary);

public:
  SummaryManager();

  const FunctionSummary *GetSummary(const FunctionDecl *FD) const;
  void SummarizeFunctionBody(const FunctionDecl *FD);

  void ParseSummaryFromJSON(const llvm::json::Array &Summary);
  void ReduceSummaries();
};

// FIXME: Is this class needed?
class SemaSummarizer : public SemaBase {
public:
  SummaryManager *TheSummaryManager;
  SummaryConsumer *TheSummaryConsumer;

  SemaSummarizer(Sema &S, SummaryManager &SummaryManager, SummaryConsumer *SummaryConsumer) 
    : SemaBase(S), TheSummaryManager(&SummaryManager), TheSummaryConsumer(SummaryConsumer) {};

  void ActOnStartOfSourceFile();
  void ActOnEndOfSourceFile();
  void SummarizeFunctionBody(const FunctionDecl *FD);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASUMMARIZE_H
