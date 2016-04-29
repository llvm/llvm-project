//===-- YamlXrefsDB.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_YAMLXREFSDB_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_YAMLXREFSDB_H

#include "XrefsDB.h"
#include "find-all-symbols/SymbolInfo.h"
#include <map>
#include <vector>

namespace clang {
namespace include_fixer {

/// Yaml format database.
class YamlXrefsDB : public XrefsDB {
public:
  YamlXrefsDB(llvm::StringRef FilePath);

  std::vector<std::string> search(llvm::StringRef Identifier) override;

private:
  std::vector<clang::find_all_symbols::SymbolInfo> Symbols;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_YAMLXREFSDB_H
