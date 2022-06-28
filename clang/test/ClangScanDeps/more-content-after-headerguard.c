// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json | FileCheck %s

// CHECK: t.c
// CHECK: top.h
// CHECK: n1.h
// CHECK: n2.h
// CHECK: n3.h

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/t.c",
    "file": "DIR/t.c"
  }
]

//--- t.c

#include "top.h"
#define INCLUDE_N3
#include "top.h"

//--- top.h
#ifndef _TOP_H_
#define _TOP_H_

#include "n1.h"

#endif

// More stuff after following '#endif', should invalidate the macro guard optimization,
// and allow `top.h` to get re-included.
#include "n2.h"

//--- n1.h

//--- n2.h
#ifdef INCLUDE_N3
#include "n3.h"
#endif

//--- n3.h
