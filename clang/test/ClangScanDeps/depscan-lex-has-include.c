// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json | FileCheck %s
// CHECK: t.c
// CHECK: something.h

// RUN: sed -e "s|DIR|%/t|g" %t/cdb-error.json.template > %t/cdb-error.json
// RUN: not clang-scan-deps -compilation-database %t/cdb-error.json 2>&1 | FileCheck %s -check-prefix=ERROR
// ERROR: error: expected '>'
// ERROR: error: expected value in expression

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/t.c -I DIR",
    "file": "DIR/t.c"
  }
]

//--- cdb-error.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/error.c",
    "file": "DIR/error.c"
  }
]

//--- t.c

#define something

// Make sure the include is lexed as a literal, ignoring the macro.
#if __has_include(<something/something.h>)
#include <something/something.h>
#endif

//--- something/something.h

//--- error.c
#if __has_include(<something/something.h)
#define MAC
#endif
