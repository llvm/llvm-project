#include "clang/Sema/SummaryConsumer.h"
#include "clang/Sema/SemaSummarizer.h"

namespace clang {
void JSONPrintingSummaryConsumer::ProcessFunctionSummary(const FunctionSummary &Summary) {
  TheSummaryManager->SerializeSummary(JOS, Summary);
}
} // namespace clang