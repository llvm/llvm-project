#include "clang/Summary/SummaryYamlMappings.h"

namespace llvm {
namespace yaml {
  void MappingTraits<clang::FunctionSummary>::mapping(IO &io, clang::FunctionSummary &FS) {
    io.mapRequired("id", FS.ID);

    std::vector<std::string> Attrs;
    for(auto &&Attr : FS.Attrs)
      Attrs.emplace_back(Attr->serialize());
    io.mapRequired("fn_attrs", Attrs);
    if(!io.outputting()) {
      std::set<const clang::SummaryAttr *> FunctionAttrs;
      for (auto parsedAttr : Attrs) {
        for (auto &&Attr : ((clang::SummaryContext*)io.getContext())->Attributes) {
          if (Attr->parse(parsedAttr))
            FunctionAttrs.emplace(Attr.get());
        }
      }

      FS.Attrs = std::move(FunctionAttrs);
    }

    io.mapRequired("opaque_calls", FS.CallsOpaque);

    std::vector<std::string> Calls(FS.Calls.begin(), FS.Calls.end());
    io.mapRequired("calls", Calls);
    if(!io.outputting())
      FS.Calls = std::set(Calls.begin(), Calls.end());
  }

  size_t
  SequenceTraits<std::vector<std::unique_ptr<clang::FunctionSummary>>>::size(IO &io, std::vector<std::unique_ptr<clang::FunctionSummary>> &seq) {
    return seq.size();
  }

  clang::FunctionSummary &
  SequenceTraits<std::vector<std::unique_ptr<clang::FunctionSummary>>>::element(IO &io, std::vector<std::unique_ptr<clang::FunctionSummary>> &seq,
          size_t index) {
    if (index >= seq.size()) {
      seq.resize(index + 1);
      seq[index].reset(new clang::FunctionSummary("", {}, {}, false));
    }
    return *seq[index];
  }
} // namespace yaml
} // namespace llvm