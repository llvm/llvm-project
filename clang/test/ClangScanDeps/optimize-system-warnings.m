// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands.json.in > %t/build/compile-commands.json
// RUN: clang-scan-deps -compilation-database %t/build/compile-commands.json \
// RUN:   -j 1 -format experimental-full -optimize-args=system-warnings > %t/deps.db
// RUN: cat %t/deps.db | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file":
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-W
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "name": "A"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file":
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-W
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "name": "B"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK:        ]
// CHECK:      }

//--- build/compile-commands.json.in

[
{
  "directory": "DIR",
  "command": "clang -c DIR/A.m -isystem modules/A -I modules/B -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps",
  "file": "DIR/A.m"
},
{
  "directory": "DIR",
  "command": "clang -c DIR/B.m -isystem modules/A -I modules/B -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -Wmaybe-unused",
  "file": "DIR/B.m"
}
]

//--- modules/A/module.modulemap

module A {
  umbrella header "A.h"
}

//--- modules/A/A.h

typedef int A_t;

//--- modules/B/module.modulemap

module B [system] {
  umbrella header "B.h"
}

//--- modules/B/B.h

typedef int B_t;

//--- A.m

#include <A.h>
#include <B.h>

A_t a = 0;

//--- B.m

#include <A.h>
#include <B.h>

A_t b = 0;
