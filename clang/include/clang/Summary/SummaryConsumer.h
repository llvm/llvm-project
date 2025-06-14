#ifndef LLVM_CLANG_SUMMARY_SUMMARYCONSUMER_H
#define LLVM_CLANG_SUMMARY_SUMMARYCONSUMER_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/JSON.h"
namespace clang {
class FunctionSummary;
class SummaryContext;

class SummaryConsumer {
protected:
  const SummaryContext *SummaryCtx;

public:
  SummaryConsumer(const SummaryContext &SummaryCtx) : SummaryCtx(&SummaryCtx) {}
  virtual ~SummaryConsumer() = default;

  virtual void ProcessStartOfSourceFile(){};
  virtual void ProcessFunctionSummary(const FunctionSummary &){};
  virtual void ProcessEndOfSourceFile(){};
};

class PrintingSummaryConsumer : public SummaryConsumer {
public:
  PrintingSummaryConsumer(const SummaryContext &SummaryCtx, raw_ostream &OS)
      : SummaryConsumer(SummaryCtx) {}
};

class JSONPrintingSummaryConsumer : public PrintingSummaryConsumer {
  llvm::json::OStream JOS;

public:
  JSONPrintingSummaryConsumer(const SummaryContext &SummaryCtx, raw_ostream &OS)
      : PrintingSummaryConsumer(SummaryCtx, OS), JOS(OS, 2) {}

  void ProcessStartOfSourceFile() override { JOS.arrayBegin(); };
  void ProcessFunctionSummary(const FunctionSummary &) override;
  void ProcessEndOfSourceFile() override {
    JOS.arrayEnd();
    JOS.flush();
  };
};
} // namespace clang

#endif // LLVM_CLANG_SUMMARY_SUMMARYCONSUMER_H
