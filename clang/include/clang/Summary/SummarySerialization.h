#ifndef LLVM_CLANG_SUMMARY_SUMMARYSERIALIZATION_H
#define LLVM_CLANG_SUMMARY_SUMMARYSERIALIZATION_H

#include "clang/Summary/SummaryContext.h"

namespace clang {
class SummarySerializer {
protected:
  SummaryContext *SummaryCtx;

public:
  SummaryContext *getSummaryCtx() const { return SummaryCtx; }

  SummarySerializer(SummaryContext &SummaryCtx) : SummaryCtx(&SummaryCtx) {};
  virtual ~SummarySerializer() = default;

  virtual void serialize(const std::vector<std::unique_ptr<FunctionSummary>> &,
                         raw_ostream &OS) = 0;
  virtual void parse(StringRef) = 0;
};

class JSONSummarySerializer : public SummarySerializer {
public:
  JSONSummarySerializer(SummaryContext &SummaryCtx)
      : SummarySerializer(SummaryCtx) {};

  void serialize(const std::vector<std::unique_ptr<FunctionSummary>> &,
                 raw_ostream &OS) override;
  void parse(StringRef) override;
};

class YAMLSummarySerializer : public SummarySerializer {
public:
  YAMLSummarySerializer(SummaryContext &SummaryCtx)
      : SummarySerializer(SummaryCtx) {};

  void serialize(const std::vector<std::unique_ptr<FunctionSummary>> &,
                 raw_ostream &OS) override;
  void parse(StringRef) override;
};
} // namespace clang

#endif // LLVM_CLANG_SUMMARY_SUMMARYSERIALIZATION_H
