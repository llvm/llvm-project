//===-- InMemoryXrefsDB.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INMEMORYXREFSDB_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INMEMORYXREFSDB_H

#include "XrefsDB.h"
#include <map>
#include <string>
#include <vector>

namespace clang {
namespace include_fixer {

/// Xref database with fixed content.
class InMemoryXrefsDB : public XrefsDB {
public:
  InMemoryXrefsDB(std::map<std::string, std::vector<std::string>> LookupTable)
      : LookupTable(std::move(LookupTable)) {}

  std::vector<std::string> search(llvm::StringRef Identifier) override;

private:
  std::map<std::string, std::vector<std::string>> LookupTable;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INMEMORYXREFSDB_H
