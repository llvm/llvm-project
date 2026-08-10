// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format make -j 1 > %t/result.txt
// RUN: FileCheck %s -input-file=%t/result.txt

// CHECK: {{/|\\}}dir1{{/|\\}}t1.c
// CHECK: {{/|\\}}dir1{{/|\\}}head.h
// CHECK: {{/|\\}}dir2{{/|\\}}t2.c
// CHECK: {{/|\\}}dir2{{/|\\}}head.h

//--- cdb.json.template
[
  {
    "directory": "DIR/dir1",
    "command": "clang -fsyntax-only t1.c",
    "file": "t1.c"
  },
  {
    "directory": "DIR/dir2",
    "command": "clang -fsyntax-only t2.c",
    "file": "t2.c"
  }
]

//--- dir1/t1.c
#include "head.h"

//--- dir1/head.h
#ifndef BBB
#define BBB
#endif

//--- dir2/t2.c
#include "head.h"

//--- dir2/head.h
