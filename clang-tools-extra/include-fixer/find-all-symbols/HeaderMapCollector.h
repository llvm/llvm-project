//===-- HeaderMapCoolector.h - find all symbols------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_HEADER_MAP_COLLECTOR_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_HEADER_MAP_COLLECTOR_H

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Regex.h"
#include <string>
#include <vector>

namespace clang {
namespace find_all_symbols {

/// \brief HeaderMappCollector collects all remapping header files. This maps
/// complete header names or header name regex patterns to header names.
class HeaderMapCollector {
public:
  typedef llvm::StringMap<std::string> HeaderMap;
  typedef std::vector<std::pair<const char *, const char *>> RegexHeaderMap;

  HeaderMapCollector() : RegexHeaderMappingTable(nullptr) {}

  explicit HeaderMapCollector(const RegexHeaderMap *RegexHeaderMappingTable)
      : RegexHeaderMappingTable(RegexHeaderMappingTable) {}

  void addHeaderMapping(llvm::StringRef OrignalHeaderPath,
                        llvm::StringRef MappingHeaderPath) {
    HeaderMappingTable[OrignalHeaderPath] = MappingHeaderPath;
  };

  /// Check if there is a mapping from \p Header or a regex pattern that matches
  /// it to another header name.
  /// \param Header A header name.
  /// \return \p Header itself if there is no mapping for it; otherwise, return
  /// a mapped header name.
  llvm::StringRef getMappedHeader(llvm::StringRef Header) const;

private:
  /// A string-to-string map saving the mapping relationship.
  HeaderMap HeaderMappingTable;

  // A map from header patterns to header names.
  // This is a reference to a hard-coded map.
  const RegexHeaderMap *const RegexHeaderMappingTable;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_HEADER_MAP_COLLECTOR_H
