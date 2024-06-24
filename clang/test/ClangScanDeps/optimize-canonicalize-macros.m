// This test verifies that command lines with equivalent -D and -U arguments
// are canonicalized to the same module variant.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/compile-commands.json.in > %t/build/compile-commands.json
// RUN: clang-scan-deps -compilation-database %t/build/compile-commands.json \
// RUN:   -j 1 -format experimental-full -optimize-args=canonicalize-macros > %t/deps.db
// RUN: cat %t/deps.db | FileCheck %s -DPREFIX=%/t

// Verify that there are only two variants and that the expected merges have
// happened.

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file":
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "J=1"
// CHECK-NOT:          "J"
// CHECK-NOT:          "K"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "A"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file":
// CHECK-NEXT:       "command-line": [
// CHECK:              "Fඞ"
// CHECK:              0D9E
// CHECK:              "K"
// CHECK:              "K"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "A"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK:        ]
// CHECK:      }


//--- build/compile-commands.json.in

[
{
  "directory": "DIR",
  "command": "clang -c DIR/tu0.m -DJ=1 -UJ -DJ=2 -DI -DK(x)=x -I modules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps",
  "file": "DIR/tu0.m"
},
{
  "directory": "DIR",
  "command": "clang -c DIR/tu1.m -DK -DK(x)=x -DI -D \"J=2\" -I modules/A -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps",
  "file": "DIR/tu1.m"
},
{
  "directory": "DIR",
  "command": "clang -c DIR/tu2.m -I modules/A -DFඞ \"-DF\\\\u{0D9E}\" -DK -DK -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps",
  "file": "DIR/tu2.m"
}
]

//--- modules/A/module.modulemap

module A {
  umbrella header "A.h"
}

//--- modules/A/A.h

//--- tu0.m

#include <A.h>

//--- tu1.m

#include <A.h>

//--- tu2.m

#include <A.h>
