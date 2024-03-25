// This test checks that ordering of TUs in the input CDB is preserved in the full output.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "file": "DIR/tu.c",
    "command": "clang -fmodules -fmodules-cache-path=DIR/cache -c DIR/tu.c -DONE -o DIR/tu1.o"
  },
  {
    "directory": "DIR",
    "file": "DIR/tu.c",
    "command": "clang -fmodules -fmodules-cache-path=DIR/cache -c DIR/tu.c -DTWO -o DIR/tu2.o"
  }
]

//--- tu.c

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "clang-module-deps": [],
// CHECK-NEXT:           "command-line": [
// CHECK:                  "-D"
// CHECK-NEXT:             "ONE"
// CHECK:                ],
// CHECK-NEXT:           "executable": "clang",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          },
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "clang-module-deps": [],
// CHECK-NEXT:           "command-line": [
// CHECK:                  "-D"
// CHECK-NEXT:             "TWO"
// CHECK:                ],
// CHECK-NEXT:           "executable": "clang",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK:              }
// CHECK:            ]
// CHECK:          }
// CHECK:        ]
// CHECK:      }
