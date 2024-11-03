// Check that we get the right file dependencies and not the declared paths from
// line directives.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -mode preprocess-dependency-directives -format experimental-full > %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      "file-deps": [
// CHECK-NEXT:   "[[PREFIX]]/tu.c"
// CHECK-NEXT:   "[[PREFIX]]/header.h"
// CHECK-NEXT: ]

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c"
}]

//--- other.h

//--- other.c

//--- header.h
#line 100 "other.h"

//--- tu.c
#include "header.h"
#line 100 "other.c"
