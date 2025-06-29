#ifndef LLVM_CLANG_SUMMARY_SUMMARYYAMLMAPPINGS_H
#define LLVM_CLANG_SUMMARY_SUMMARYYAMLMAPPINGS_H

#include "llvm/Support/YAMLTraits.h"
#include "clang/Summary/SummaryContext.h"

#include <memory>
#include <vector>

namespace llvm {
namespace yaml {
  template <> struct MappingTraits<clang::FunctionSummary> {
    static void mapping(IO &io, clang::FunctionSummary &FS);
  };

  template <>
  struct SequenceTraits<std::vector<std::unique_ptr<clang::FunctionSummary>>> {
    static size_t
    size(IO &io, std::vector<std::unique_ptr<clang::FunctionSummary>> &seq);
    static clang::FunctionSummary &
    element(IO &io, std::vector<std::unique_ptr<clang::FunctionSummary>> &seq,
            size_t index);
  };
} // namespace yaml
} // namespace llvm

#endif //LLVM_CLANG_SUMMARY_SUMMARYYAMLMAPPINGS_H
