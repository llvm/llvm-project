#ifndef LLVM_CLANG_SEMA_SUMMARYCONSUMER_H
#define LLVM_CLANG_SEMA_SUMMARYCONSUMER_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/JSON.h"
namespace clang {
class FunctionSummary;
class SummaryManager;

class SummaryConsumer {
protected:
    const SummaryManager *TheSummaryManager;

public:
    SummaryConsumer(const SummaryManager &SummaryManager) : TheSummaryManager(&SummaryManager) {}
    virtual ~SummaryConsumer() = default;

    virtual void ProcessStartOfSourceFile() {};
    virtual void ProcessFunctionSummary(const FunctionSummary&) {};
    virtual void ProcessEndOfSourceFile() {};
};

class PrintingSummaryConsumer : public SummaryConsumer {
public:
    PrintingSummaryConsumer(const SummaryManager &SummaryManager, raw_ostream &OS)
      : SummaryConsumer(SummaryManager) {}
};

class JSONPrintingSummaryConsumer : public PrintingSummaryConsumer {
    llvm::json::OStream JOS;

public:
    JSONPrintingSummaryConsumer(const SummaryManager &SummaryManager, raw_ostream &OS) : PrintingSummaryConsumer(SummaryManager, OS), JOS(OS, 2) {}

    void ProcessStartOfSourceFile() override { JOS.arrayBegin(); };
    void ProcessFunctionSummary(const FunctionSummary&) override;
    void ProcessEndOfSourceFile() override { JOS.arrayEnd(); };
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SUMMARYCONSUMER_H
