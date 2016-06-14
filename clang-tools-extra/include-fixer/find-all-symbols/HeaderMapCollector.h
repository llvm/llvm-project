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
#include <string>

namespace clang {
namespace find_all_symbols {

/// \brief HeaderMappCollector collects all remapping header files. This maps
/// complete header names or postfixes of header names to header names.
class HeaderMapCollector {
public:
  typedef llvm::StringMap<std::string> HeaderMap;

  HeaderMapCollector() : PostfixMappingTable(nullptr) {}

  explicit HeaderMapCollector(const HeaderMap *PostfixMap)
      : PostfixMappingTable(PostfixMap) {}

  void addHeaderMapping(llvm::StringRef OrignalHeaderPath,
                        llvm::StringRef MappingHeaderPath) {
    HeaderMappingTable[OrignalHeaderPath] = MappingHeaderPath;
  };

  /// Check if there is a mapping from \p Header or its postfix to another
  /// header name.
  /// \param Header A header name.
  /// \return \p Header itself if there is no mapping for it; otherwise, return
  /// a mapped header name.
  llvm::StringRef getMappedHeader(llvm::StringRef Header) const;

private:
  /// A string-to-string map saving the mapping relationship.
  HeaderMap HeaderMappingTable;

  // A postfix-to-header name map.
  // This is a reference to a hard-coded map.
  const HeaderMap *const PostfixMappingTable;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_HEADER_MAP_COLLECTOR_H
