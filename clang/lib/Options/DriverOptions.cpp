//===--- DriverOptions.cpp - Driver Options Table -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Options/Options.h"
#include "llvm/Option/OptTable.h"
#include <cassert>

using namespace clang::options;
using namespace llvm::opt;

#define OPTTABLE_STR_TABLE_CODE
#include "clang/Options/Options.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_VALUES_CODE
#include "clang/Options/Options.inc"
#undef OPTTABLE_VALUES_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "clang/Options/Options.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

#define OPTTABLE_PREFIXES_UNION_CODE
#include "clang/Options/Options.inc"
#undef OPTTABLE_PREFIXES_UNION_CODE

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "clang/Options/Options.inc"
#undef OPTION
};

namespace {

class DriverOptTable : public PrecomputedOptTable {
public:
  DriverOptTable()
      : PrecomputedOptTable(OptionStrTable, OptionPrefixesTable, InfoTable,
                            OptionPrefixesUnion) {}
};
} // anonymous namespace

const llvm::opt::OptTable &clang::getDriverOptTable() {
  static DriverOptTable Table;
  return Table;
}
