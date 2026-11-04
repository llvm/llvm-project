// This test checks that implicit modules do not encounter a deadlock on a dependency cycle.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: not clang-scan-deps -mode preprocess -format experimental-full \
// RUN:   -compilation-database %t/cdb.json -j 2 -o %t/out 2> %t/err
// RUN: FileCheck %s --input-file %t/err
// CHECK-DAG: fatal error: cyclic dependency in module 'M': M -> N -> M
// CHECK-DAG: fatal error: cyclic dependency in module 'N': N -> M -> N

//--- cdb.json.in
[
  {
    "file": "DIR/tu1.c",
    "directory": "DIR",
    "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fimplicit-modules-lock-timeout=3 DIR/tu1.c -o DIR/tu1.o"
  },
  {
    "file": "DIR/tu2.c",
    "directory": "DIR",
    "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fimplicit-modules-lock-timeout=3 DIR/tu2.c -o DIR/tu2.o"
  }
]
//--- tu1.c
#include "m.h"
//--- tu2.c
#include "n.h"
//--- module.modulemap
module M { header "m.h" }
module N { header "n.h" }
//--- m.h
#pragma clang __debug sleep // Give enough time for tu2.c to start building N.
#include "n.h"
//--- n.h
#pragma clang __debug sleep // Give enough time for tu1.c to start building M.
#include "m.h"
