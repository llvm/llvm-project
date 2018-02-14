//===-- HeaderMapCoolector.h - find all symbols------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HeaderMapCollector.h"
#include "llvm/Support/Regex.h"

namespace clang {
namespace find_all_symbols {

HeaderMapCollector::HeaderMapCollector(
    const RegexHeaderMap *RegexHeaderMappingTable) {
  assert(RegexHeaderMappingTable);
  this->RegexHeaderMappingTable.reserve(RegexHeaderMappingTable->size());
  for (const auto &Entry : *RegexHeaderMappingTable) {
    this->RegexHeaderMappingTable.emplace_back(llvm::Regex(Entry.first),
                                               Entry.second);
  }
}

llvm::StringRef
HeaderMapCollector::getMappedHeader(llvm::StringRef Header) const {
  auto Iter = HeaderMappingTable.find(Header);
  if (Iter != HeaderMappingTable.end())
    return Iter->second;
  // If there is no complete header name mapping for this header, check the
  // regex header mapping.
  for (auto &Entry : RegexHeaderMappingTable) {
#ifndef NDEBUG
    std::string Dummy;
    assert(Entry.first.isValid(Dummy) && "Regex should never be invalid!");
#endif
    if (Entry.first.match(Header))
      return Entry.second;
  }
  return Header;
}

} // namespace find_all_symbols
} // namespace clang
