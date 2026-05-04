// Test that -working-directory works even when it differs from the working
// directory of the filesystem.

// RUN: rm -rf %t
// RUN: mkdir -p %t/other
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   > %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      "file-deps": [
// CHECK-NEXT:   "[[PREFIX]]/cwd/t.c"
// CHECK-NEXT:   "[[PREFIX]]/cwd/relative/h1.h"
// CHECK-NEXT: ]
// CHECK-NEXT: "input-file": "[[PREFIX]]/cwd/t.c"

//--- cdb.json.template
[{
  "directory": "DIR/other",
  "command": "clang -c t.c -I relative -working-directory DIR/cwd",
  "file": "DIR/cwd/t.c"
}]

//--- cwd/relative/h1.h

//--- cwd/t.c
#include "h1.h"
