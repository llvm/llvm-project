// Test that a translation unit reports the directories it searches for headers.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -track-search-directories 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=ON

// ON:      "directory-deps": [
// ON-NEXT:   "[[PREFIX]]/angled",
// ON-NEXT:   "[[PREFIX]]/fw",
// ON-NEXT:   "[[PREFIX]]/missing",
// ON-NEXT:   "[[PREFIX]]/quoted"
// ON-NEXT: ]

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=OFF

// OFF-NOT: "directory-deps"

//--- fw/F.framework/Headers/F.h
//--- sysfw/SF.framework/Headers/SF.h
//--- quoted/Q.h
//--- angled/A.h
//--- sys/S.h
//--- notadir

//--- tu.c
#include "Q.h"
#include "A.h"
#include "S.h"
#include <F/F.h>
#include <SF/SF.h>

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -iquote DIR/quoted -I angled -isystem DIR/sys -F DIR/fw -iframework DIR/sysfw -I DIR/notadir -I DIR/missing -c DIR/tu.c -o DIR/tu.o"
}]

