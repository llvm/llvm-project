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

void BinarySummarySerializer::EmitString(StringRef Str,
                                         SmallVector<uint64_t, 64> &Buffer) {
  Buffer.push_back(Str.size());
  llvm::append_range(Buffer, Str);
}

// FIXME: clean this up
void BinarySummarySerializer::serialize(
    const std::vector<std::unique_ptr<FunctionSummary>> &, raw_ostream &OS) {
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'T', 8);
  Stream.Emit((unsigned)'U', 8);
  Stream.Emit((unsigned)'S', 8);

  Stream.EnterBlockInfoBlock();
  EmitBlock(ATTRIBUTE_BLOCK_ID, "ATTRIBUTES");
  EmitRecord(ATTR, "ATTR");
  EmitBlock(IDENTIFIER_BLOCK_ID, "IDENTIFIERS");
  EmitRecord(IDENTIFIER, "IDENTIFIER");
  EmitBlock(SUMMARY_BLOCK_ID, "SUMMARIES");
  EmitRecord(FUNCTION, "FUNCTION");
  Stream.ExitBlock();

  Stream.EnterSubblock(ATTRIBUTE_BLOCK_ID, 5);
  uint64_t ID = 0;
  // FIXME: Should we concatenate these for smaller size?
  for (auto &&Attr : SummaryCtx->Attributes) {
    AttrIDs[Attr.get()] = ID++;
    SmallVector<uint64_t, 64> Record;
    EmitString(Attr->serialize(), Record);
    Stream.EmitRecord(ATTR, Record);
  }
  Stream.ExitBlock();

  Stream.EnterSubblock(IDENTIFIER_BLOCK_ID, 5);
  ID = 0;
  for (auto &&Summary : SummaryCtx->FunctionSummaries) {
    FunctionIDs[Summary->getID()] = ID++;
    SmallVector<uint64_t, 64> Record;
    EmitString(Summary->getID(), Record);
    Stream.EmitRecord(IDENTIFIER, Record);

    for (auto &&Call : Summary->getCalls()) {
      if (FunctionIDs.count(Call))
        continue;

      FunctionIDs[Call] = ID++;
      SmallVector<uint64_t, 64> Record;
      EmitString(Call, Record);
      Stream.EmitRecord(IDENTIFIER, Record);
    }
  }
  Stream.ExitBlock();

  Stream.EnterSubblock(SUMMARY_BLOCK_ID, 5);
  for (auto &&Summary : SummaryCtx->FunctionSummaries) {
    SmallVector<uint64_t, 64> Record;

    Record.push_back(FUNCTION);
    Record.push_back(Summary->getAttributes().size());
    Record.push_back(Summary->getCalls().size());
    Record.push_back(Summary->callsOpaqueObject());

    Record.push_back(1 + Summary->getAttributes().size() +
                     Summary->getCalls().size());
    Record.push_back(FunctionIDs[Summary->getID()]);
    for (auto &&Attr : Summary->getAttributes())
      Record.push_back(AttrIDs[Attr]);
    for (auto &&Call : Summary->getCalls())
      Record.push_back(FunctionIDs[Call]);

    auto Abv = std::make_shared<llvm::BitCodeAbbrev>();
    Abv->Add(llvm::BitCodeAbbrevOp(FUNCTION));
    // The number of attributes.
    Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 32));
    // The number of callees.
    Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 32));
    // Whether there are opaque callees or not.
    Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 1));
    // An array of the following form: [ID, Attr0...AttrN, Callee0...CalleeN]
    Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Array));
    Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 32));
    unsigned Abbrev = Stream.EmitAbbrev(std::move(Abv));

    Stream.EmitRecord(FUNCTION, Record, Abbrev);
  }
  Stream.ExitBlock();

  Stream.FlushToWord();
  OS << Buffer;
  OS.flush();
}

void BinarySummarySerializer::parse(StringRef Buffer) {}
} // namespace clang
