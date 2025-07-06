#include "clang/Summary/SummarySerialization.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Support/JSON.h"

namespace llvm {
namespace yaml {
template <> struct MappingTraits<clang::FunctionSummary> {
  static void mapping(IO &io, clang::FunctionSummary &FS) {
    io.mapRequired("id", FS.ID);

    std::vector<std::string> Attrs;
    for (auto &&Attr : FS.Attrs)
      Attrs.emplace_back(Attr->serialize());
    io.mapRequired("fn_attrs", Attrs);
    if (!io.outputting()) {
      std::set<const clang::SummaryAttr *> FunctionAttrs;
      for (auto parsedAttr : Attrs) {
        for (auto &&Attr :
             ((clang::SummaryContext *)io.getContext())->Attributes) {
          if (Attr->parse(parsedAttr))
            FunctionAttrs.emplace(Attr.get());
        }
      }

      FS.Attrs = std::move(FunctionAttrs);
    }

    io.mapRequired("opaque_calls", FS.CallsOpaque);

    std::vector<std::string> Calls(FS.Calls.begin(), FS.Calls.end());
    io.mapRequired("calls", Calls);
    if (!io.outputting())
      FS.Calls = std::set(Calls.begin(), Calls.end());
  }
};

template <>
struct SequenceTraits<std::vector<std::unique_ptr<clang::FunctionSummary>>> {
  static size_t
  size(IO &io, std::vector<std::unique_ptr<clang::FunctionSummary>> &seq) {
    return seq.size();
  }

  static clang::FunctionSummary &
  element(IO &io, std::vector<std::unique_ptr<clang::FunctionSummary>> &seq,
          size_t index) {
    if (index >= seq.size()) {
      seq.resize(index + 1);
      seq[index].reset(new clang::FunctionSummary("", {}, {}, false));
    }
    return *seq[index];
  }
};

} // namespace yaml
} // namespace llvm

namespace clang {
void JSONSummarySerializer::serialize(
    const std::vector<std::unique_ptr<FunctionSummary>> &Summaries,
    raw_ostream &OS) {
  llvm::json::OStream JOS(OS, 2);
  JOS.arrayBegin();

  for (auto &&Summary : Summaries) {
    JOS.object([&] {
      JOS.attribute("id", llvm::json::Value(Summary->getID()));
      JOS.attributeObject("attrs", [&] {
        JOS.attributeArray("function", [&] {
          for (auto &&Attr : Summary->getAttributes()) {
            JOS.value(llvm::json::Value(Attr->serialize()));
          }
        });
      });
      JOS.attributeObject("calls", [&] {
        JOS.attribute("opaque",
                      llvm::json::Value(Summary->callsOpaqueObject()));
        JOS.attributeArray("functions", [&] {
          for (auto &&Call : Summary->getCalls()) {
            JOS.object([&] { JOS.attribute("id", llvm::json::Value(Call)); });
          }
        });
      });
    });
  }

  JOS.arrayEnd();
  JOS.flush();
}

void JSONSummarySerializer::parse(StringRef Buffer) {
  auto JSON = llvm::json::parse(Buffer);
  if (!JSON) {
    llvm::handleAllErrors(JSON.takeError(), [](const llvm::ErrorInfoBase &EI) {
      std::ignore = EI.message();
    });
    return;
  }

  auto *JSONSummaries = JSON->getAsArray();
  if (!JSONSummaries)
    return;

  for (auto &&JSONSummary : *JSONSummaries) {
    const llvm::json::Object *JSONSummaryObject = JSONSummary.getAsObject();
    if (!JSONSummaryObject)
      continue;

    std::optional<StringRef> ID = JSONSummaryObject->getString("id");
    if (!ID)
      continue;

    const llvm::json::Object *JSONAttributes =
        JSONSummaryObject->getObject("attrs");
    if (!JSONAttributes)
      continue;

    const llvm::json::Array *JSONFunctionAttributes =
        JSONAttributes->getArray("function");
    if (!JSONFunctionAttributes)
      continue;

    std::set<const SummaryAttr *> FunctionAttrs;
    for (auto &&JSONAttr : *JSONFunctionAttributes)
      for (auto &&CtxAttr : SummaryCtx->Attributes)
        if (auto JSONAttrStr = JSONAttr.getAsString();
            JSONAttrStr && CtxAttr->parse(*JSONAttrStr))
          FunctionAttrs.emplace(CtxAttr.get());

    const llvm::json::Object *JSONCallsObject =
        JSONSummaryObject->getObject("calls");
    if (!JSONCallsObject)
      continue;

    std::optional<bool> CallsOpaue = *JSONCallsObject->getBoolean("opaque");
    if (!CallsOpaue)
      continue;

    std::set<std::string> Calls;
    const llvm::json::Array *JSONCallEntries =
        JSONCallsObject->getArray("functions");
    if (!JSONCallEntries)
      continue;

    for (auto &&JSONCall : *JSONCallEntries) {
      auto *JSONCallObj = JSONCall.getAsObject();
      if (!JSONCallObj)
        continue;

      std::optional<StringRef> CallID = JSONCallObj->getString("id");
      if (!CallID)
        continue;

      Calls.emplace(CallID->str());
    }

    SummaryCtx->CreateSummary(ID->str(), std::move(FunctionAttrs),
                              std::move(Calls), *CallsOpaue);
  }
}

void YAMLSummarySerializer::serialize(
    const std::vector<std::unique_ptr<FunctionSummary>> &Summaries,
    raw_ostream &OS) {
  llvm::yaml::Output YOUT(OS);
  YOUT << ((SummaryContext *)SummaryCtx)->FunctionSummaries;
  OS.flush();
}

void YAMLSummarySerializer::parse(StringRef Buffer) {
  std::vector<std::unique_ptr<clang::FunctionSummary>> summaries;

  llvm::yaml::Input YIN(Buffer, SummaryCtx);
  YIN >> summaries;

  for (auto &&summary : summaries)
    SummaryCtx->CreateSummary(summary->getID().str(), summary->getAttributes(),
                              summary->getCalls(),
                              summary->callsOpaqueObject());
}

void BinarySummarySerializer::PopulateBlockInfo() {
  Stream.EnterBlockInfoBlock();
  EmitBlock(ATTRIBUTE_BLOCK_ID, "ATTRIBUTES");
  EmitRecord(ATTR, "ATTR");
  EmitBlock(IDENTIFIER_BLOCK_ID, "IDENTIFIERS");
  EmitRecord(IDENTIFIER, "IDENTIFIER");
  EmitBlock(SUMMARY_BLOCK_ID, "SUMMARIES");
  EmitRecord(FUNCTION, "FUNCTION");
  Stream.ExitBlock();
}

void BinarySummarySerializer::EmitAttributeBlock() {
  Stream.EnterSubblock(ATTRIBUTE_BLOCK_ID, 3);

  auto Abv = std::make_shared<llvm::BitCodeAbbrev>();
  Abv->Add(llvm::BitCodeAbbrevOp(ATTR));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Array));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 8));
  unsigned Abbrev = Stream.EmitAbbrev(std::move(Abv));

  uint64_t ID = 0;
  uint64_t Record[] = {ATTR};

  for (auto &&Attr : SummaryCtx->Attributes) {
    AttrIDs[Attr.get()] = ID++;
    Stream.EmitRecordWithArray(Abbrev, Record, Attr->serialize());
  }

  Stream.ExitBlock();
}

void BinarySummarySerializer::EmitIdentifierBlock() {
  Stream.EnterSubblock(IDENTIFIER_BLOCK_ID, 3);

  auto Abv = std::make_shared<llvm::BitCodeAbbrev>();
  Abv->Add(llvm::BitCodeAbbrevOp(IDENTIFIER));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Array));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 8));
  unsigned Abbrev = Stream.EmitAbbrev(std::move(Abv));

  uint64_t ID = 0;
  uint64_t Record[] = {IDENTIFIER};

  for (auto &&Summary : SummaryCtx->FunctionSummaries) {
    FunctionIDs[Summary->getID().str()] = ID++;
    Stream.EmitRecordWithArray(Abbrev, Record, Summary->getID());

    for (auto &&Call : Summary->getCalls()) {
      if (FunctionIDs.count(Call))
        continue;

      FunctionIDs[Call] = ID++;
      Stream.EmitRecordWithArray(Abbrev, Record, Call);
    }
  }

  Stream.ExitBlock();
}

void BinarySummarySerializer::EmitSummaryBlock() {
  Stream.EnterSubblock(SUMMARY_BLOCK_ID, 3);

  auto Abv = std::make_shared<llvm::BitCodeAbbrev>();
  Abv->Add(llvm::BitCodeAbbrevOp(FUNCTION));
  // The number of attributes.
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 7));
  // The number of callees.
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 32));
  // Whether there are opaque callees or not.
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 1));
  // An array of the following form: [ID, Attr0...AttrN, Callee0...CalleeN]
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Array));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 32));
  unsigned Abbrev = Stream.EmitAbbrev(std::move(Abv));

  for (auto &&Summary : SummaryCtx->FunctionSummaries) {
    SmallVector<uint64_t, 64> Record;

    Record.push_back(Summary->getAttributes().size());
    Record.push_back(Summary->getCalls().size());
    Record.push_back(Summary->callsOpaqueObject());

    Record.push_back(1 + Summary->getAttributes().size() +
                     Summary->getCalls().size());
    Record.push_back(FunctionIDs[Summary->getID().str()]);
    for (auto &&Attr : Summary->getAttributes())
      Record.push_back(AttrIDs[Attr]);
    for (auto &&Call : Summary->getCalls())
      Record.push_back(FunctionIDs[Call]);

    Stream.EmitRecord(FUNCTION, Record, Abbrev);
  }

  Stream.ExitBlock();
}

void BinarySummarySerializer::EmitBlock(unsigned ID, const char *Name) {
  SmallVector<uint64_t, 64> Buffer;
  Buffer.push_back(ID);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETBID, Buffer);

  Buffer.clear();
  while (*Name)
    Buffer.push_back(*Name++);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_BLOCKNAME, Buffer);
}

void BinarySummarySerializer::EmitRecord(unsigned ID, const char *Name) {
  SmallVector<uint64_t, 64> Buffer;
  Buffer.push_back(ID);
  while (*Name)
    Buffer.push_back(*Name++);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETRECORDNAME, Buffer);
}

void BinarySummarySerializer::serialize(
    const std::vector<std::unique_ptr<FunctionSummary>> &, raw_ostream &OS) {
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'T', 8);
  Stream.Emit((unsigned)'U', 8);
  Stream.Emit((unsigned)'S', 8);

  PopulateBlockInfo();
  EmitAttributeBlock();
  EmitIdentifierBlock();
  EmitSummaryBlock();

  Stream.FlushToWord();
  OS << Buffer;
  OS.flush();
}

llvm::Error
BinarySummarySerializer::handleBlockStartCommon(unsigned ID,
                                                llvm::BitstreamCursor &Stream) {
  unsigned NumWords = 0;
  if (llvm::Error Err = Stream.EnterSubBlock(ID, &NumWords))
    return Err;

  llvm::BitstreamEntry Entry;
  if (llvm::Error E =
          Stream.advance(llvm::BitstreamCursor::AF_DontAutoprocessAbbrevs)
              .moveInto(Entry))
    return E;

  if (Entry.Kind != llvm::BitstreamEntry::Record &&
      Entry.ID != llvm::bitc::DEFINE_ABBREV)
    return llvm::createStringError("expected abbrev");

  if (llvm::Error Err = Stream.ReadAbbrevRecord())
    return Err;

  return llvm::Error::success();
}

llvm::Error BinarySummarySerializer::handleBlockRecordsCommon(
    llvm::BitstreamCursor &Stream,
    llvm::function_ref<void(const SmallVector<uint64_t, 64> &)> Callback) {
  while (true) {
    llvm::BitstreamEntry Entry;
    if (llvm::Error E =
            Stream.advance(llvm::BitstreamCursor::AF_DontAutoprocessAbbrevs)
                .moveInto(Entry))
      return E;

    if (Entry.Kind == llvm::BitstreamEntry::EndBlock)
      return llvm::Error::success();

    if (Entry.Kind != llvm::BitstreamEntry::Record)
      return llvm::createStringError("expected record");

    SmallVector<uint64_t, 64> Record;
    unsigned Code;
    if (llvm::Error E = Stream.readRecord(Entry.ID, Record).moveInto(Code))
      return E;

    Callback(Record);
  }
}

llvm::Error
BinarySummarySerializer::parseAttributeBlock(llvm::BitstreamCursor &Stream) {
  if (llvm::Error Err = handleBlockStartCommon(ATTRIBUTE_BLOCK_ID, Stream))
    return Err;

  ParsedAttrIDs.clear();
  if (llvm::Error Err = handleBlockRecordsCommon(Stream, [&](auto &&Record) {
        for (auto &&CtxAttr : SummaryCtx->Attributes) {
          llvm::SmallString<64> AttributeStr(Record.begin(), Record.end());

          if (CtxAttr->parse(AttributeStr.str())) {
            ParsedAttrIDs.push_back(CtxAttr.get());
            break;
          }
        }
      }))
    return Err;

  return llvm::Error::success();
}

llvm::Error
BinarySummarySerializer::parseIdentifierBlock(llvm::BitstreamCursor &Stream) {
  if (llvm::Error Err = handleBlockStartCommon(IDENTIFIER_BLOCK_ID, Stream))
    return Err;

  ParsedFunctionIDs.clear();
  if (llvm::Error Err = handleBlockRecordsCommon(Stream, [&](auto &&Record) {
        llvm::SmallString<64> IdentifierStr(Record.begin(), Record.end());
        ParsedFunctionIDs.emplace_back(IdentifierStr.str().str());
      }))
    return Err;

  return llvm::Error::success();
}

llvm::Error
BinarySummarySerializer::parseSummaryBlock(llvm::BitstreamCursor &Stream) {
  if (llvm::Error Err = handleBlockStartCommon(SUMMARY_BLOCK_ID, Stream))
    return Err;

  if (llvm::Error Err = handleBlockRecordsCommon(Stream, [&](auto &&Record) {
        int AttrCnt = Record[0];
        int CallCnt = Record[1];
        bool Opaque = Record[2];
        int ID = Record[4];
        int I = 0;

        std::set<const SummaryAttr *> Attrs;
        while (AttrCnt) {
          Attrs.emplace(ParsedAttrIDs[Record[5 + I]]);
          ++I;
          --AttrCnt;
        }

        std::set<std::string> Calls;
        while (CallCnt) {
          Calls.emplace(ParsedFunctionIDs[Record[5 + I]]);
          ++I;
          --CallCnt;
        }

        SummaryCtx->CreateSummary(ParsedFunctionIDs[ID], std::move(Attrs),
                                  std::move(Calls), Opaque);
      }))
    return Err;

  return llvm::Error::success();
}

llvm::Error BinarySummarySerializer::parseBlock(unsigned ID,
                                                llvm::BitstreamCursor &Stream) {
  if (ID == llvm::bitc::BLOCKINFO_BLOCK_ID) {
    std::optional<llvm::BitstreamBlockInfo> NewBlockInfo;
    if (llvm::Error Err = Stream.ReadBlockInfoBlock().moveInto(NewBlockInfo))
      return Err;
    if (!NewBlockInfo)
      return llvm::createStringError("expected block info");

    return llvm::Error::success();
  }

  if (ID == ATTRIBUTE_BLOCK_ID)
    return parseAttributeBlock(Stream);

  if (ID == IDENTIFIER_BLOCK_ID)
    return parseIdentifierBlock(Stream);

  if (ID == SUMMARY_BLOCK_ID)
    return parseSummaryBlock(Stream);

  return llvm::createStringError("unexpected block");
}

llvm::Error BinarySummarySerializer::parseImpl(StringRef Buffer) {
  llvm::BitstreamCursor Stream(Buffer);

  llvm::SimpleBitstreamCursor::word_t Magic[4] = {0};
  unsigned char ExpectedMagic[] = {'C', 'T', 'U', 'S'};
  for (int i = 0; i < 4; ++i) {
    if (llvm::Error Err = Stream.Read(8).moveInto(Magic[i]))
      return Err;

    if (Magic[i] != ExpectedMagic[i])
      return llvm::createStringError("invalid magic number");
  }

  while (!Stream.AtEndOfStream()) {
    Expected<unsigned> MaybeCode = Stream.ReadCode();
    if (!MaybeCode)
      return MaybeCode.takeError();
    if (MaybeCode.get() != llvm::bitc::ENTER_SUBBLOCK)
      return llvm::createStringError("expected record");

    Expected<unsigned> MaybeBlockID = Stream.ReadSubBlockID();
    if (!MaybeBlockID)
      return MaybeBlockID.takeError();

    if (llvm::Error Err = parseBlock(MaybeBlockID.get(), Stream))
      return Err;
  }

  return llvm::Error::success();
}

void BinarySummarySerializer::parse(StringRef Buffer) {
  if (llvm::Error Err = parseImpl(Buffer)) {
    handleAllErrors(std::move(Err), [&](const llvm::ErrorInfoBase &EI) {
      std::ignore = EI.message();
    });
  }
}
} // namespace clang
