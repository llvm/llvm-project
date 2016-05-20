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

/// \brief HeaderMappCollector collects all remapping header files.
class HeaderMapCollector {
public:
  typedef llvm::StringMap<std::string> HeaderMap;

  void addHeaderMapping(llvm::StringRef OrignalHeaderPath,
                        llvm::StringRef MappingHeaderPath) {
    HeaderMappingTable[OrignalHeaderPath] = MappingHeaderPath;
  };
  const HeaderMap &getHeaderMappingTable() { return HeaderMappingTable; };

private:
  /// A string-to-string map saving the mapping relationship.
  HeaderMap HeaderMappingTable;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_HEADER_MAP_COLLECTOR_H
