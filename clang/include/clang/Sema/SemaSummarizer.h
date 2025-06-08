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
  FunctionSummary(const clang::FunctionDecl *FD);

  void addAttribute(SummaryAttribute Attr) { FunctionAttrs.emplace(Attr); }
  bool hasAttribute(SummaryAttribute Attr) { return FunctionAttrs.count(Attr); }

  void addCall(const clang::FunctionDecl *FD);
};

class SemaSummarizer : public SemaBase {
public:
  SemaSummarizer(Sema &S, SummaryConsumer *SummaryConsumer);

  std::vector<std::unique_ptr<SummaryAttributeManager>> Attributes;
  SummaryConsumer *TheSummaryConsumer;

  void SummarizeFunctionBody(const FunctionDecl *FD);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASUMMARIZE_H
