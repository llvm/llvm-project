// Check that the scanner can handle a response file input.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -format experimental-full -compilation-database %t/cdb.json > %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      "command-line": [
// CHECK:        "-fsyntax-only"
// CHECK:        "-x"
// CHECK-NEXT:   "c"
// CHECK:        "tu.c"
// CHECK:        "-I"
// CHECK-NEXT:   "include"
// CHECK:      ],
// CHECK:      "file-deps": [
// CHECK-NEXT:   "[[PREFIX]]/tu.c"
// CHECK-NEXT:   "[[PREFIX]]/include/header.h"
// CHECK-NEXT: ]

//--- cdb.json.template
[{
  "file": "DIR/t.c",
  "directory": "DIR",
  "command": "clang @DIR/args.txt"
}]

//--- args.txt
@args_nested.txt
-fsyntax-only tu.c

//--- args_nested.txt
-I include

//--- include/header.h

//--- tu.c
#include "header.h"
