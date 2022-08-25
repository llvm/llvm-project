// This test checks that pruning of header search paths produces consistent dependency graphs.
//
// When pruning header search paths for a module, we can't remove any paths its dependencies use.
// Otherwise, we could get either of the following dependency graphs depending on the search path
// configuration of the particular TU that first discovered the module:
//   X:<hash1> -> Y:<hash2>
//   X:<hash1> -> Y:<hash3>
// We can't have the same version of module X depend on multiple different versions of Y based on
// the TU configuration.
//
// Keeping all header search paths (transitive) dependencies use will ensure we get consistent
// dependency graphs:
//   X:<hash1> -> Y:<hash2>
//   X:<hash4> -> Y:<hash3>

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- a/a.h
//--- b/b.h
//--- begin/begin.h
//--- end/end.h
//--- Y.h
#include "begin.h"
#if __has_include("a.h")
#include "a.h"
#endif
#include "end.h"

//--- X.h
#include "Y.h"

//--- module.modulemap
module Y { header "Y.h" }
module X { header "X.h" }

//--- test.c
#include "X.h"

//--- cdb_with_a.json.template
[{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang -c test.c -o DIR/test.o -fmodules -fimplicit-modules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -Ibegin -Ia -Ib -Iend"
}]

//--- cdb_without_a.json.template
[{
  "file": "DIR/test.c",
  "directory": "DIR",
  "command": "clang -c test.c -o DIR/test.o -fmodules -fimplicit-modules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -Ibegin     -Ib -Iend"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb_with_a.json.template    > %t/cdb_with_a.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb_without_a.json.template > %t/cdb_without_a.json

// RUN: clang-scan-deps -compilation-database %t/cdb_with_a.json    -format experimental-full -optimize-args >  %t/results.json
// RUN: clang-scan-deps -compilation-database %t/cdb_without_a.json -format experimental-full -optimize-args >> %t/results.json
// RUN: cat %t/results.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_Y_WITH_A:.*]]",
// CHECK-NEXT:           "module-name": "Y"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_X:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/./X.h",
// CHECK-NEXT:         "[[PREFIX]]/./module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "X"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_Y_WITH_A]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/./Y.h",
// CHECK-NEXT:         "[[PREFIX]]/./a/a.h",
// CHECK-NEXT:         "[[PREFIX]]/./begin/begin.h",
// CHECK-NEXT:         "[[PREFIX]]/./end/end.h",
// CHECK-NEXT:         "[[PREFIX]]/./module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "Y"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-context-hash": "{{.*}}",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_X]]",
// CHECK-NEXT:           "module-name": "X"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/test.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/test.c"
// CHECK-NEXT:     }

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_Y_WITHOUT_A:.*]]",
// CHECK-NEXT:           "module-name": "Y"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// Here is the actual check that this module X (which imports different version of Y)
// also has a different context hash from the first version of module X.
// CHECK-NOT:        "context-hash": "[[HASH_X]]",
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/./X.h",
// CHECK-NEXT:         "[[PREFIX]]/./module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "X"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_Y_WITHOUT_A]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/./Y.h",
// CHECK-NEXT:         "[[PREFIX]]/./begin/begin.h",
// CHECK-NEXT:         "[[PREFIX]]/./end/end.h",
// CHECK-NEXT:         "[[PREFIX]]/./module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "Y"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-context-hash": "{{.*}}",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "X"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/test.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/test.c"
// CHECK-NEXT:     }
