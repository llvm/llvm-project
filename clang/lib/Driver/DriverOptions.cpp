//===--- DriverOptions.cpp - Driver Options Table -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Options.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Option.h"

using namespace clang::driver;
using namespace clang::driver::options;

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)
#include "clang/Driver/Options.inc"
#undef OPTION
#undef PREFIX

static const OptTable::Info InfoTable[] = {
#define PREFIX(NAME, VALUE)
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, FLAGS, PARAM, \
               HELPTEXT, METAVAR)   \
  { PREFIX, NAME, HELPTEXT, METAVAR, OPT_##ID, Option::KIND##Class, PARAM, \
    FLAGS, OPT_##GROUP, OPT_##ALIAS },
#include "clang/Driver/Options.inc"
};

namespace {

class DriverOptTable : public OptTable {
public:
  DriverOptTable()
    : OptTable(InfoTable, sizeof(InfoTable) / sizeof(InfoTable[0])) {}
};

}

OptTable *clang::driver::createDriverOptTable() {
  return new DriverOptTable();
}
