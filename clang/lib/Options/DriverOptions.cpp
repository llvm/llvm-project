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

#define OPTTABLE_VALUES_CODE
#include "clang/Options/Options.inc"

#define OPTTABLE_CODE
#include "clang/Options/Options.inc"

namespace {

class DriverOptTable : public OptTable {
public:
  DriverOptTable() : OptTable(OptionTables) {
    setValuesCodeFn(getOptionValuesCode);
  }
};
} // anonymous namespace

const llvm::opt::OptTable &clang::getDriverOptTable() {
  static const DriverOptTable Table;
  return Table;
}
