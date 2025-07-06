#ifndef LLVM_CLANG_SUMMARY_SUMMARYSERIALIZATION_H
#define LLVM_CLANG_SUMMARY_SUMMARYSERIALIZATION_H

#include "clang/Summary/SummaryContext.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Bitstream/BitstreamWriter.h"

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

class BinarySummarySerializer : public SummarySerializer {
  enum BlockIDs {
    ATTRIBUTE_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID,
    IDENTIFIER_BLOCK_ID,
    SUMMARY_BLOCK_ID
  };

  enum AttributeRecordTypes {
    ATTR = 1,
  };

  enum IdentifierRecordTypes {
    IDENTIFIER = 1,
  };

  enum SummaryRecordTypes { FUNCTION = 1 };

  // FIXME: get rid of this global state
  std::map<const SummaryAttr *, uint64_t> AttrIDs;
  std::map<std::string, uint64_t> FunctionIDs;

  std::vector<const SummaryAttr *> ParsedAttrIDs;
  std::vector<std::string> ParsedFunctionIDs;

  llvm::SmallVector<char, 32> Buffer;
  llvm::BitstreamWriter Stream;

  void PopulateBlockInfo();
  void EmitAttributeBlock();
  void EmitIdentifierBlock();
  void EmitSummaryBlock();

  void EmitBlock(unsigned ID, const char *Name);
  void EmitRecord(unsigned ID, const char *Name);

  llvm::Error handleBlockStartCommon(unsigned ID,
                                     llvm::BitstreamCursor &Stream);
  llvm::Error handleBlockRecordsCommon(
      llvm::BitstreamCursor &Stream,
      llvm::function_ref<void(const SmallVector<uint64_t, 64> &)>);

  llvm::Error parseAttributeBlock(llvm::BitstreamCursor &Stream);
  llvm::Error parseIdentifierBlock(llvm::BitstreamCursor &Stream);
  llvm::Error parseSummaryBlock(llvm::BitstreamCursor &Stream);
  llvm::Error parseBlock(unsigned ID, llvm::BitstreamCursor &Stream);
  llvm::Error parseImpl(StringRef Buffer);

public:
  BinarySummarySerializer(SummaryContext &SummaryCtx)
      : SummarySerializer(SummaryCtx), Stream(Buffer) {};

  void serialize(const std::vector<std::unique_ptr<FunctionSummary>> &,
                 raw_ostream &OS) override;
  void parse(StringRef) override;
};

} // namespace clang

#endif // LLVM_CLANG_SUMMARY_SUMMARYSERIALIZATION_H
