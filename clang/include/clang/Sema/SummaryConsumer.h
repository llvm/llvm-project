#ifndef LLVM_CLANG_SEMA_SUMMARYCONSUMER_H
#define LLVM_CLANG_SEMA_SUMMARYCONSUMER_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/JSON.h"
namespace clang {
class FunctionSummary;

class SummaryConsumer {
public:
    virtual ~SummaryConsumer() = default;

    virtual void ProcessStartOfSourceFile() {};
    virtual void ProcessFunctionSummary(const FunctionSummary&) {};
    virtual void ProcessEndOfSourceFile() {};
};

class PrintingSummaryConsumer : public SummaryConsumer {
public:
    PrintingSummaryConsumer(raw_ostream &OS)
      : SummaryConsumer() {}
};

class JSONPrintingSummaryConsumer : public PrintingSummaryConsumer {
    llvm::json::OStream JOS;

public:
    JSONPrintingSummaryConsumer(raw_ostream &OS) : PrintingSummaryConsumer(OS), JOS(OS, 2) {}

    void ProcessStartOfSourceFile() override { JOS.arrayBegin(); };
    void ProcessFunctionSummary(const FunctionSummary&) override;
    void ProcessEndOfSourceFile() override { JOS.arrayEnd(); };
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SUMMARYCONSUMER_H
