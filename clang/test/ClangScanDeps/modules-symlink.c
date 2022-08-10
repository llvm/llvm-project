// RUN: rm -rf %t
// RUN: split-file %s %t
// Unsupported on AIX because we don't support the requisite "__clangast"
// section in XCOFF yet.
// UNSUPPORTED: system-windows, aix

//--- cdb_pch.json
[
  {
    "directory": "DIR",
    "command": "clang -x c-header DIR/pch.h -fmodules -gmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -o DIR/pch.h.gch",
    "file": "DIR/pch.h"
  }
]

//--- cdb_tu.json
[
  {
    "directory": "DIR",
    "command": "clang -c DIR/tu.c -fmodules -gmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -include DIR/pch.h -o DIR/tu.o",
    "file": "DIR/tu.c"
  }
]

//--- module.modulemap
module mod { header "symlink.h" }

//--- pch.h
#include "symlink.h"

//--- original.h
// Comment that will be stripped by the minimizer.
#define MACRO 1

//--- tu.c
#include "original.h"
static int foo = MACRO; // Macro usage that will trigger
                        // input file consistency checks.

// RUN: ln -s %t/original.h %t/symlink.h

// RUN: sed -e "s|DIR|%/t|g" %t/cdb_pch.json > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -module-files-dir %t/build > %t/result_pch.json
//
// RUN: %deps-to-rsp %t/result_pch.json --module-name=mod > %t/mod.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --tu-index=0 > %t/pch.rsp
//
// RUN: %clang @%t/mod.cc1.rsp
// RUN: %clang @%t/pch.rsp

// RUN: sed -e "s|DIR|%/t|g" %t/cdb_tu.json > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -module-files-dir %t/build > %t/result_tu.json
