#include "clang/Summary/SummarySerialization.h"
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
} // namespace clang
