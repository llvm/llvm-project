#ifndef LLVM_CLANG_SEMA_SEMASUMMARIZER_H
#define LLVM_CLANG_SEMA_SEMASUMMARIZER_H

#include "clang/Sema/SemaBase.h"
#include "clang/Sema/SummaryConsumer.h"

namespace clang {
class SemaSummarizer : public SemaBase {
public:
  SemaSummarizer(Sema &S, SummaryConsumer *SummaryConsumer)
      : SemaBase(S), SummaryConsumer(SummaryConsumer) {}

  SummaryConsumer *SummaryConsumer;

  void SummarizeFunctionBody(const FunctionDecl *FD);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMASUMMARIZE_H
