// Test scanning deps works with _Pragma syntax when not inside a macro.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c",
  "file": "DIR/tu.c"
}]

//--- a.h
_Pragma("once")
#include "b.h"

//--- b.h
#include "a.h"

//--- tu.c
#include "a.h"
