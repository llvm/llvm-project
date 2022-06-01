// Check dependency scanning when skipping an unmatched #if

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: not clang-scan-deps -compilation-database %t/cdb.json 2>&1 | FileCheck %s
// CHECK: header1.h:1:2: error: unterminated conditional directive

//--- cdb.json.template
[{
  "directory" : "DIR",
  "command" : "clang -target x86_64-apple-macosx10.7 -c DIR/test.cpp -o DIR/test.o",
  "file" : "DIR/test.o"
}]

//--- test.cpp
#include "header1.h"
#include "header2.h"

//--- header1.h
#if 0

//--- header2.h
#ifndef _HEADER2_H_
#define _HEADER2_H_
#endif
