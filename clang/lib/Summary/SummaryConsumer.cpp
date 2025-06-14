#include "clang/Summary/SummaryConsumer.h"
#include "clang/Summary/SummaryContext.h"

namespace clang {
void JSONPrintingSummaryConsumer::ProcessFunctionSummary(
    const FunctionSummary &Summary) {
  JOS.object([&] {
    JOS.attribute("id", llvm::json::Value(Summary.getID()));
    JOS.attributeObject("attrs", [&] {
      JOS.attributeArray("function", [&] {
        for (auto &&Attr : Summary.getAttributes()) {
          JOS.value(llvm::json::Value(Attr->serialize()));
        }
      });
    });
    JOS.attributeArray("calls", [&] {
      for (auto &&Call : Summary.getCalls()) {
        JOS.object([&] { JOS.attribute("id", llvm::json::Value(Call)); });
      }
    });
  });
}
} // namespace clang