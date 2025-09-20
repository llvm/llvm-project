#include "clang/Summary/SummaryConsumer.h"
#include "clang/Summary/SummaryContext.h"
#include "clang/Summary/SummarySerialization.h"

namespace clang {
SerializingSummaryConsumer::SerializingSummaryConsumer(
    SummarySerializer &Serializer, llvm::raw_ostream &OS)
    : SummaryConsumer(*Serializer.getSummaryCtx()), OS(OS),
      Serializer(&Serializer) {}

void SerializingSummaryConsumer::ProcessEndOfSourceFile() {
  Serializer->serialize(OS);
}
} // namespace clang
