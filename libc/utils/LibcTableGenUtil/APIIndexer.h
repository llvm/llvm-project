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

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llvm_libc {

class APIIndexer {
private:
  llvm::Optional<llvm::StringRef> StdHeader;

  // TableGen classes in spec.td.
  llvm::Record *NamedTypeClass;
  llvm::Record *PtrTypeClass;
  llvm::Record *RestrictedPtrTypeClass;
  llvm::Record *ConstTypeClass;
  llvm::Record *StructClass;
  llvm::Record *StandardSpecClass;
  llvm::Record *PublicAPIClass;

  bool isaNamedType(llvm::Record *Def);
  bool isaStructType(llvm::Record *Def);
  bool isaPtrType(llvm::Record *Def);
  bool isaConstType(llvm::Record *Def);
  bool isaRestrictedPtrType(llvm::Record *Def);
  bool isaStandardSpec(llvm::Record *Def);
  bool isaPublicAPI(llvm::Record *Def);

  void indexStandardSpecDef(llvm::Record *StandardSpec);
  void indexPublicAPIDef(llvm::Record *PublicAPI);
  void index(llvm::RecordKeeper &Records);

public:
  using NameToRecordMapping = std::unordered_map<std::string, llvm::Record *>;
  using NameSet = std::unordered_set<std::string>;

  // This indexes all headers, not just a specified one.
  explicit APIIndexer(llvm::RecordKeeper &Records) : StdHeader(llvm::None) {
    index(Records);
  }

  APIIndexer(llvm::StringRef Header, llvm::RecordKeeper &Records)
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

  std::string getTypeAsString(llvm::Record *TypeRecord);
};

} // namespace llvm_libc

#endif // LLVM_LIBC_UTILS_LIBC_TABLE_GEN_UTILS_API_INDEXER_H
