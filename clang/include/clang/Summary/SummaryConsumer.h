#ifndef LLVM_CLANG_SUMMARY_SUMMARYCONSUMER_H
#define LLVM_CLANG_SUMMARY_SUMMARYCONSUMER_H

#include "llvm/Support/raw_ostream.h"
namespace clang {
class FunctionSummary;
class SummaryContext;
class SummarySerializer;

class SummaryConsumer {
protected:
  const SummaryContext *SummaryCtx;

public:
  SummaryConsumer(const SummaryContext &SummaryCtx) : SummaryCtx(&SummaryCtx) {}
  virtual ~SummaryConsumer() = default;

  virtual void ProcessStartOfSourceFile() {};
  virtual void ProcessFunctionSummary(const FunctionSummary &) {};
  virtual void ProcessEndOfSourceFile() {};
};

class SerializingSummaryConsumer : public SummaryConsumer {
  llvm::raw_ostream &OS;
  SummarySerializer *Serializer;

public:
  SerializingSummaryConsumer(SummarySerializer &Serializer,
                             llvm::raw_ostream &OS);

  void ProcessEndOfSourceFile() override;
};

} // namespace clang

#endif // LLVM_CLANG_SUMMARY_SUMMARYCONSUMER_H
