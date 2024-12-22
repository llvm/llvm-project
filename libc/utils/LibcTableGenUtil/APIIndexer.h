//===-- A class to index libc API listed in tablegen files ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_LIBC_TABLE_GEN_UTILS_API_INDEXER_H
#define LLVM_LIBC_UTILS_LIBC_TABLE_GEN_UTILS_API_INDEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llvm_libc {

class APIIndexer {
private:
  std::optional<llvm::StringRef> StdHeader;

  // TableGen classes in spec.td.
  const llvm::Record *NamedTypeClass;
  const llvm::Record *PtrTypeClass;
  const llvm::Record *RestrictedPtrTypeClass;
  const llvm::Record *ConstTypeClass;
  const llvm::Record *StructClass;
  const llvm::Record *StandardSpecClass;
  const llvm::Record *PublicAPIClass;

  bool isaNamedType(const llvm::Record *Def);
  bool isaStructType(const llvm::Record *Def);
  bool isaPtrType(const llvm::Record *Def);
  bool isaConstType(const llvm::Record *Def);
  bool isaRestrictedPtrType(const llvm::Record *Def);
  bool isaStandardSpec(const llvm::Record *Def);
  bool isaPublicAPI(const llvm::Record *Def);

  void indexStandardSpecDef(const llvm::Record *StandardSpec);
  void indexPublicAPIDef(const llvm::Record *PublicAPI);
  void index(const llvm::RecordKeeper &Records);

public:
  using NameToRecordMapping =
      std::unordered_map<std::string, const llvm::Record *>;
  using NameSet = std::unordered_set<std::string>;

  // This indexes all headers, not just a specified one.
  explicit APIIndexer(const llvm::RecordKeeper &Records)
      : StdHeader(std::nullopt) {
    index(Records);
  }

  APIIndexer(llvm::StringRef Header, const llvm::RecordKeeper &Records)
      : StdHeader(Header) {
    index(Records);
  }

  // Mapping from names to records defining them.
  NameToRecordMapping MacroSpecMap;
  NameToRecordMapping TypeSpecMap;
  NameToRecordMapping EnumerationSpecMap;
  NameToRecordMapping FunctionSpecMap;
  NameToRecordMapping MacroDefsMap;
  NameToRecordMapping ObjectSpecMap;

  std::unordered_map<std::string, std::string> FunctionToHeaderMap;
  std::unordered_map<std::string, std::string> ObjectToHeaderMap;

  NameSet RequiredTypes;
  NameSet Structs;
  NameSet Enumerations;
  NameSet Functions;
  NameSet Objects;
  NameSet PublicHeaders;

  std::string getTypeAsString(const llvm::Record *TypeRecord);
};

} // namespace llvm_libc

#endif // LLVM_LIBC_UTILS_LIBC_TABLE_GEN_UTILS_API_INDEXER_H
