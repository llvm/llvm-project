#ifndef LLVM_CLANG_SEMA_SEMASUMMARIZER_H
#define LLVM_CLANG_SEMA_SEMASUMMARIZER_H

#include "clang/Sema/SemaBase.h"
#include "clang/Sema/SummaryAttribute.h"
#include "clang/Sema/SummaryConsumer.h"
#include <set>

namespace clang {
class FunctionSummary {
  SmallVector<char> ID;
  std::set<SummaryAttribute> FunctionAttrs;
  std::set<SmallVector<char>> Calls;

public:
  FunctionSummary(SmallVector<char> ID, std::set<SummaryAttribute> FunctionAttrs, std::set<SmallVector<char>> Calls);
  FunctionSummary(const clang::FunctionDecl *FD);

  SmallVector<char> getID() const { return ID; }
  const std::set<SummaryAttribute> &getFunctionAttrs() const { return FunctionAttrs; }
  const std::set<SmallVector<char>> &getCalls() const { return Calls; }

  void addAttribute(SummaryAttribute Attr) { FunctionAttrs.emplace(Attr); }
  bool hasAttribute(SummaryAttribute Attr) const { return FunctionAttrs.count(Attr); }

  void addCall(const clang::FunctionDecl *FD);
};

class SummaryManager {
  std::map<std::string, const FunctionSummary *> IDToSummary;
  std::vector<std::unique_ptr<FunctionSummary>> FunctionSummaries;
  
  std::map<SummaryAttribute, const SummaryAttributeDescription *> AttrToDescription;
  std::vector<std::unique_ptr<SummaryAttributeDescription>> AttributeDescriptions;

public:
  SummaryManager();

  FunctionSummary SummarizeFunctionBody(const FunctionDecl *FD);
  
  void SerializeSummary(llvm::json::OStream &, const FunctionSummary &) const;
  void ParseSummaryFromJSON(StringRef path);

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
