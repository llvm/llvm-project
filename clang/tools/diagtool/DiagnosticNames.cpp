//===- DiagnosticNames.cpp - Defines a table of all builtin diagnostics ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DiagnosticNames.h"
#include "clang/Basic/AllDiagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringTable.h"
#include <array>
#include <cstddef>
#include <cstdint>

using namespace clang;
using namespace diagtool;

struct BuiltinDiagnosticNameStorage {
#define DIAG_NAME_INDEX(ENUM) char ENUM[sizeof(#ENUM)];
#include "clang/Basic/DiagnosticIndexName.inc"
#undef DIAG_NAME_INDEX
};

static_assert(sizeof(BuiltinDiagnosticNameStorage) <= uint64_t(1) << 24);

static constexpr BuiltinDiagnosticNameStorage BuiltinDiagnosticNames = {
#define DIAG_NAME_INDEX(ENUM) #ENUM,
#include "clang/Basic/DiagnosticIndexName.inc"
#undef DIAG_NAME_INDEX
};

#define DIAGNOSTIC_RECORD(ENUM)                                                \
  {diag::ENUM, uint16_t(offsetof(BuiltinDiagnosticNameStorage, ENUM)),         \
   uint8_t(offsetof(BuiltinDiagnosticNameStorage, ENUM) >> 16),                \
   STR_SIZE(#ENUM, uint8_t)}

static constexpr DiagnosticRecord BuiltinDiagnosticsByName[] = {
#define DIAG_NAME_INDEX(ENUM) DIAGNOSTIC_RECORD(ENUM),
#include "clang/Basic/DiagnosticIndexName.inc"
#undef DIAG_NAME_INDEX
};
static_assert(std::size(BuiltinDiagnosticsByName) < (1U << 16));
#undef DIAGNOSTIC_RECORD

llvm::ArrayRef<DiagnosticRecord> diagtool::getBuiltinDiagnosticsByName() {
  return llvm::ArrayRef(BuiltinDiagnosticsByName);
}

static constexpr auto BuiltinDiagnosticIndexByID = [] {
  std::array<uint16_t, diag::DIAG_UPPER_LIMIT> Result = {};
  uint16_t Index = 0;
#define DIAG_NAME_INDEX(ENUM) Result[diag::ENUM] = ++Index;
#include "clang/Basic/DiagnosticIndexName.inc"
#undef DIAG_NAME_INDEX
  return Result;
}();

llvm::StringRef DiagnosticRecord::getName() const {
  const char *Names = reinterpret_cast<const char *>(&BuiltinDiagnosticNames);
  uint32_t NameOffset =
      uint32_t(NameOffsetLow) | (uint32_t(NameOffsetHigh) << 16);
  return llvm::StringRef(Names + NameOffset, NameLen);
}

const DiagnosticRecord &diagtool::getDiagnosticForID(short DiagID) {
  assert(DiagID >= 0 &&
         static_cast<unsigned>(DiagID) < BuiltinDiagnosticIndexByID.size() &&
         BuiltinDiagnosticIndexByID[DiagID] != 0 && "diagnostic not found");
  return BuiltinDiagnosticsByName[BuiltinDiagnosticIndexByID[DiagID] - 1];
}

#define GET_DIAG_ARRAYS
#include "clang/Basic/DiagnosticGroups.inc"
#undef GET_DIAG_ARRAYS

// Second the table of options, sorted by name for fast binary lookup.
static const GroupRecord OptionTable[] = {
#define DIAG_ENTRY(GroupName, FlagNameOffset, Members, SubGroups, Docs)        \
  {FlagNameOffset, Members, SubGroups},
#include "clang/Basic/DiagnosticGroups.inc"
#undef DIAG_ENTRY
};

llvm::StringRef GroupRecord::getName() const {
  return DiagGroupNames[NameOffset];
}

GroupRecord::subgroup_iterator GroupRecord::subgroup_begin() const {
  return DiagSubGroups + SubGroups;
}

GroupRecord::subgroup_iterator GroupRecord::subgroup_end() const {
  return nullptr;
}

llvm::iterator_range<diagtool::GroupRecord::subgroup_iterator>
GroupRecord::subgroups() const {
  return llvm::make_range(subgroup_begin(), subgroup_end());
}

GroupRecord::diagnostics_iterator GroupRecord::diagnostics_begin() const {
  return DiagArrays + Members;
}

GroupRecord::diagnostics_iterator GroupRecord::diagnostics_end() const {
  return nullptr;
}

llvm::iterator_range<diagtool::GroupRecord::diagnostics_iterator>
GroupRecord::diagnostics() const {
  return llvm::make_range(diagnostics_begin(), diagnostics_end());
}

llvm::ArrayRef<GroupRecord> diagtool::getDiagnosticGroups() {
  return llvm::ArrayRef(OptionTable);
}
