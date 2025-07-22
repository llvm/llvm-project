// Test correctly reporting what a module exports during dependency scanning.
// Module A depends on modules B, C and D, but only exports B and C.
// Module E depends on modules B, C and D, and exports all of them.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database \
// RUN:   %t/cdb.json -format experimental-full > %t/deps.db
// RUN: cat %t/deps.db | sed 's:\\\\\?:/:g' | FileCheck %s

//--- cdb.json.template
[
    {
      "directory": "DIR",
      "command": "clang -c DIR/test.c -I DIR/AH -I DIR/BH -I DIR/CH -I DIR/DH -I DIR/EH -fmodules -fmodules-cache-path=DIR/cache",
      "file": "DIR/test.c"
    },
]

//--- AH/A.h
#include "B.h"
#include "C.h"
#include "D.h"

int funcA();

//--- AH/module.modulemap
module A {
    header "A.h"

    export B
    export C
}

//--- BH/B.h
//--- BH/module.modulemap
module B {
    header "B.h"
}

//--- CH/C.h
//--- CH/module.modulemap
module C {
    header "C.h"
}

//--- DH/D.h
//--- DH/module.modulemap
module D {
    header "D.h"
}

//--- EH/E.h
#include "B.h"
#include "C.h"
#include "D.h"

//--- EH/module.modulemap
module E {
    header "E.h"
    export *
}

//--- test.c
#include "A.h"
#include "E.h"

int test1() {
  return funcA();
}

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_MOD_B:.*]]",
// CHECK-NEXT:               "module-name": "B",
// CHECK-NEXT:               "exported": "true"
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_MOD_C:.*]]",
// CHECK-NEXT:               "module-name": "C",
// CHECK-NEXT:               "exported": "true"
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_MOD_D:.*]]",
// CHECK-NEXT:               "module-name": "D"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "clang-modulemap-file":{{.*}},
// CHECK-NEXT:           "command-line": [
// CHECK:                 ],
// CHECK:                "name": "A"
// CHECK-NEXT:       }
// CHECK:            {
// CHECK:                 "name": "B"
// CHECK:            }
// CHECK:            {
// CHECK:                 "name": "C"
// CHECK:            }
// CHECK:            {
// CHECK:                 "name": "D"
// CHECK:            }
// CHECK:            {
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_MOD_B]]",
// CHECK-NEXT:               "module-name": "B",
// CHECK-NEXT:               "exported": "true"
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_MOD_C]]",
// CHECK-NEXT:               "module-name": "C",
// CHECK-NEXT:               "exported": "true"
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_MOD_D]]",
// CHECK-NEXT:               "module-name": "D",
// CHECK-NEXT:               "exported": "true"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "clang-modulemap-file":{{.*}},
// CHECK-NEXT:           "command-line": [
// CHECK:                 ],
// CHECK:                "name": "E"
// CHECK-NEXT:       }
// CHECK:   ]
// CHECK:   }



