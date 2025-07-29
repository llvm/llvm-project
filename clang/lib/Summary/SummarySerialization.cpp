#include "clang/Summary/SummarySerialization.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Support/JSON.h"

namespace llvm {
namespace yaml {
struct FunctionSummaryProxy {
  size_t ID;
  std::vector<size_t> Attrs;
  std::vector<size_t> Calls;
  bool CallsOpaque;

  FunctionSummaryProxy() = default;
  FunctionSummaryProxy(const clang::FunctionSummary &Summary)
      : ID(Summary.getID()), CallsOpaque(Summary.callsOpaqueObject()) {
    Attrs.reserve(Summary.getAttributes().size());
    for (auto &&Attr : Summary.getAttributes())
      Attrs.emplace_back(Attr->getKind());

    Calls.reserve(Summary.getCalls().size());
    for (auto &&Call : Summary.getCalls())
      Calls.emplace_back(Call);
  }
};

template <> struct MappingTraits<FunctionSummaryProxy> {
  static void mapping(IO &io, FunctionSummaryProxy &FS) {
    io.mapRequired("id", FS.ID);
    io.mapRequired("fn_attrs", FS.Attrs);
    io.mapRequired("opaque_calls", FS.CallsOpaque);
    io.mapRequired("calls", FS.Calls);
  }
};

template <> struct SequenceTraits<std::vector<FunctionSummaryProxy>> {
  static size_t size(IO &io, std::vector<FunctionSummaryProxy> &seq) {
    return seq.size();
  }

  static FunctionSummaryProxy &
  element(IO &io, std::vector<FunctionSummaryProxy> &seq, size_t index) {
    if (index >= seq.size())
      seq.emplace_back();

    return seq[index];
  }
};

template <> struct MappingTraits<clang::SummaryContext> {
  static void mapping(IO &io, clang::SummaryContext &Ctx) {
    if (io.outputting()) {
      std::vector<StringRef> Identifiers = Ctx.GetIdentifiers();
      io.mapRequired("identifiers", Identifiers);

      std::vector<std::string> Attributes;
      Attributes.reserve(Ctx.GetAttributes().size());
      for (auto &&Attr : Ctx.GetAttributes())
        Attributes.emplace_back(Attr->serialize());
      io.mapRequired("attributes", Attributes);

      std::vector<FunctionSummaryProxy> SummaryProxies;
      SummaryProxies.reserve(Ctx.GetSummaries().size());
      for (auto &&Summary : Ctx.GetSummaries())
        SummaryProxies.emplace_back(*Summary);
      io.mapRequired("summaries", SummaryProxies);

      return;
    }

    std::vector<StringRef> Identifiers;
    io.mapRequired("identifiers", Identifiers);
    std::map<size_t, size_t> LocalToContextID;
    for (auto &&ID : Identifiers) {
      LocalToContextID[LocalToContextID.size()] =
          Ctx.GetOrInsertStoredIdentifierIdx(ID);
    }

    std::vector<StringRef> Attributes;
    io.mapRequired("attributes", Attributes);
    std::map<size_t, const clang::SummaryAttr *> AttrIDToPtr;
    std::set<const clang::SummaryAttr *> Seen;
    for (auto &&Attribute : Attributes) {
      for (auto &&Attr : Ctx.GetAttributes())
        if (Attr->parse(Attribute)) {
          if (!Seen.emplace(Attr.get()).second)
            break;

          AttrIDToPtr[AttrIDToPtr.size()] = Attr.get();
          break;
        }
    }

    std::vector<FunctionSummaryProxy> SummaryProxies;
    io.mapRequired("summaries", SummaryProxies);
    for (auto &&Proxy : SummaryProxies) {
      if (Proxy.ID >= LocalToContextID.size())
        continue;

      std::set<const clang::SummaryAttr *> Attrs;
      for (auto &&Attr : Proxy.Attrs) {
        if (Attr >= AttrIDToPtr.size())
          continue;

        Attrs.emplace(AttrIDToPtr[Attr]);
      }

      std::set<size_t> Calls;
      for (auto &&Call : Proxy.Calls) {
        if (Call >= LocalToContextID.size())
          continue;

        Calls.emplace(LocalToContextID[Call]);
      }

      Ctx.CreateSummary(LocalToContextID[Proxy.ID], std::move(Attrs),
                        std::move(Calls), Proxy.CallsOpaque);
    }
  }
};
} // namespace yaml
} // namespace llvm

namespace clang {
void JSONSummarySerializer::serialize(raw_ostream &OS) {
  llvm::json::OStream JOS(OS, 2);
  JOS.objectBegin();

  JOS.attributeArray("identifiers", [&] {
    for (auto &&Identifier : SummaryCtx->GetIdentifiers())
      JOS.value(Identifier);
  });

  JOS.attributeArray("attributes", [&] {
    for (auto &&Attribute : SummaryCtx->GetAttributes())
      JOS.value(Attribute->serialize());
  });

  JOS.attributeArray("summaries", [&] {
    for (auto &&Summary : SummaryCtx->GetSummaries()) {
      JOS.object([&] {
        JOS.attribute("id", llvm::json::Value(Summary->getID()));
        JOS.attributeArray("fn_attrs", [&] {
          for (auto &&Attr : Summary->getAttributes()) {
            JOS.value(llvm::json::Value(static_cast<size_t>(Attr->getKind())));
          }
        });
        JOS.attribute("opaque_calls",
                      llvm::json::Value(Summary->callsOpaqueObject()));
        JOS.attributeArray("calls", [&] {
          for (auto &&Call : Summary->getCalls()) {
            JOS.value(llvm::json::Value(Call));
          }
        });
      });
    }
  });

  JOS.objectEnd();
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

  auto *JSONObject = JSON->getAsObject();
  if (!JSONObject)
    return;

  auto *JSONIdentifiers = JSONObject->getArray("identifiers");
  if (!JSONIdentifiers)
    return;

  std::map<size_t, size_t> LocalToContextID;
  for (auto &&Identifier : *JSONIdentifiers) {
    auto IdentifierStr = Identifier.getAsString();
    if (!IdentifierStr)
      return;

    LocalToContextID[LocalToContextID.size()] =
        SummaryCtx->GetOrInsertStoredIdentifierIdx(*IdentifierStr);
  }

  auto *JSONAttributes = JSONObject->getArray("attributes");
  if (!JSONAttributes)
    return;

  std::map<size_t, const SummaryAttr *> AttrIDToPtr;
  std::set<const SummaryAttr *> Seen;
  for (auto &&Attribute : *JSONAttributes) {
    auto AttributeStr = Attribute.getAsString();

    for (auto &&Attr : SummaryCtx->GetAttributes())
      if (Attr->parse(*AttributeStr)) {
        if (!Seen.emplace(Attr.get()).second)
          return;

        AttrIDToPtr[AttrIDToPtr.size()] = Attr.get();
        break;
      }
  }

  auto *JSONSummaries = JSONObject->getArray("summaries");
  if (!JSONSummaries)
    return;

  for (auto &&JSONSummary : *JSONSummaries) {
    auto *JSONSummaryObject = JSONSummary.getAsObject();
    if (!JSONSummaryObject)
      continue;

    std::optional<size_t> ID = JSONSummaryObject->getInteger("id");
    if (!ID || *ID >= LocalToContextID.size())
      continue;

    auto *JSONAttributes = JSONSummaryObject->getArray("fn_attrs");
    if (!JSONAttributes)
      continue;

    std::set<const SummaryAttr *> FunctionAttrs;
    for (auto &&JSONAttr : *JSONAttributes) {
      std::optional<size_t> AttrID = JSONAttr.getAsUINT64();
      if (!AttrID || *AttrID >= AttrIDToPtr.size())
        return;

      FunctionAttrs.emplace(AttrIDToPtr[*AttrID]);
    }

    std::optional<bool> CallsOpaue =
        *JSONSummaryObject->getBoolean("opaque_calls");
    if (!CallsOpaue)
      continue;

    std::set<size_t> Calls;
    auto *JSONCallEntries = JSONSummaryObject->getArray("calls");
    if (!JSONCallEntries)
      continue;

    for (auto &&JSONCall : *JSONCallEntries) {
      std::optional<size_t> CallID = JSONCall.getAsUINT64();
      if (!CallID || *CallID >= LocalToContextID.size())
        continue;

      Calls.emplace(LocalToContextID[*CallID]);
    }

    SummaryCtx->CreateSummary(LocalToContextID[*ID], std::move(FunctionAttrs),
                              std::move(Calls), *CallsOpaue);
  }
}

void YAMLSummarySerializer::serialize(raw_ostream &OS) {
  llvm::yaml::Output YOUT(OS);
  YOUT << *SummaryCtx;
  OS.flush();
}

void YAMLSummarySerializer::parse(StringRef Buffer) {
  llvm::yaml::Input YIN(Buffer);
  YIN >> *SummaryCtx;
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

void BinarySummarySerializer::EmitIdentifierBlock() {
  Stream.EnterSubblock(IDENTIFIER_BLOCK_ID, 3);

  auto Abv = std::make_shared<llvm::BitCodeAbbrev>();
  Abv->Add(llvm::BitCodeAbbrevOp(IDENTIFIER));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Array));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 8));
  unsigned Abbrev = Stream.EmitAbbrev(std::move(Abv));

  uint64_t Record[] = {IDENTIFIER};
  for (auto &&Identifier : SummaryCtx->GetIdentifiers())
    Stream.EmitRecordWithArray(Abbrev, Record, Identifier);

  Stream.ExitBlock();
}

void BinarySummarySerializer::EmitAttributeBlock() {
  Stream.EnterSubblock(ATTRIBUTE_BLOCK_ID, 3);

  auto Abv = std::make_shared<llvm::BitCodeAbbrev>();
  Abv->Add(llvm::BitCodeAbbrevOp(ATTR));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Array));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 8));
  unsigned Abbrev = Stream.EmitAbbrev(std::move(Abv));

  uint64_t Record[] = {ATTR};
  for (auto &&Attr : SummaryCtx->GetAttributes())
    Stream.EmitRecordWithArray(Abbrev, Record, Attr->serialize());

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

  for (auto &&Summary : SummaryCtx->GetSummaries()) {
    SmallVector<uint64_t, 64> Record;

    Record.push_back(Summary->getAttributes().size());
    Record.push_back(Summary->getCalls().size());
    Record.push_back(Summary->callsOpaqueObject());

    Record.push_back(1 + Summary->getAttributes().size() +
                     Summary->getCalls().size());
    Record.push_back(Summary->getID());
    for (auto &&Attr : Summary->getAttributes())
      Record.push_back(Attr->getKind());
    for (auto &&Call : Summary->getCalls())
      Record.push_back(Call);

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

void BinarySummarySerializer::serialize(raw_ostream &OS) {
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'T', 8);
  Stream.Emit((unsigned)'U', 8);
  Stream.Emit((unsigned)'S', 8);

  PopulateBlockInfo();
  EmitIdentifierBlock();
  EmitAttributeBlock();
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
BinarySummarySerializer::parseIdentifierBlock(llvm::BitstreamCursor &Stream) {
  if (llvm::Error Err = handleBlockStartCommon(IDENTIFIER_BLOCK_ID, Stream))
    return Err;

  if (llvm::Error Err = handleBlockRecordsCommon(Stream, [&](auto &&Record) {
        llvm::SmallString<64> IdentifierStr(Record.begin(), Record.end());
        LocalToContextID[LocalToContextID.size()] =
            SummaryCtx->GetOrInsertStoredIdentifierIdx(IdentifierStr);
      }))
    return Err;

  return llvm::Error::success();
}

llvm::Error
BinarySummarySerializer::parseAttributeBlock(llvm::BitstreamCursor &Stream) {
  if (llvm::Error Err = handleBlockStartCommon(ATTRIBUTE_BLOCK_ID, Stream))
    return Err;

  if (llvm::Error Err = handleBlockRecordsCommon(Stream, [&](auto &&Record) {
        std::set<const clang::SummaryAttr *> Seen;

        for (auto &&Attr : SummaryCtx->GetAttributes()) {
          llvm::SmallString<64> AttributeStr(Record.begin(), Record.end());

          if (Attr->parse(AttributeStr.str())) {
            if (!Seen.emplace(Attr.get()).second)
              break;

            AttrIDToPtr[AttrIDToPtr.size()] = Attr.get();
            break;
          }
        }
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
        size_t ID = Record[4];
        int I = 0;

        if (ID >= LocalToContextID.size())
          return;

        std::set<const SummaryAttr *> Attrs;
        while (AttrCnt) {
          size_t AttrID = Record[5 + I];
          if (AttrID >= AttrIDToPtr.size())
            return;

          Attrs.emplace(AttrIDToPtr[AttrID]);
          ++I;
          --AttrCnt;
        }

        std::set<size_t> Calls;
        while (CallCnt) {
          size_t CallID = Record[5 + I];
          if (CallID >= LocalToContextID.size())
            return;

          Calls.emplace(LocalToContextID[CallID]);
          ++I;
          --CallCnt;
        }

        SummaryCtx->CreateSummary(LocalToContextID[ID], std::move(Attrs),
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

  if (ID == IDENTIFIER_BLOCK_ID)
    return parseIdentifierBlock(Stream);

  if (ID == ATTRIBUTE_BLOCK_ID)
    return parseAttributeBlock(Stream);

  if (ID == SUMMARY_BLOCK_ID)
    return parseSummaryBlock(Stream);

  return llvm::createStringError("unexpected block");
}

llvm::Error BinarySummarySerializer::parseImpl(StringRef Buffer) {
  llvm::BitstreamCursor Stream(Buffer);
  LocalToContextID.clear();
  AttrIDToPtr.clear();

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
