#include "clang/Sema/SummaryConsumer.h"
#include "clang/Sema/SemaSummarizer.h"

namespace clang {
void JSONPrintingSummaryConsumer::ProcessFunctionSummary(const FunctionSummary &Summary) {
  JOS.object([&]{
    JOS.attribute("id", llvm::json::Value(Summary.getID()));
    JOS.attributeObject("attrs", [&]{
      JOS.attributeArray("function", [&]{
        for(auto &&Attr : Summary.getFunctionAttrs()) {
          JOS.value(llvm::json::Value(SummaryAttributeManager::serialize(Attr)));
        }
      });
    });
    JOS.attributeArray("calls", [&]{
      for(auto &&Call : Summary.getCalls()) {
        JOS.object([&]{
          JOS.attribute("id", llvm::json::Value(Call));
        });
      }
    });
  });
}
} // namespace clang