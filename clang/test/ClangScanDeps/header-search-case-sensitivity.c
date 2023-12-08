// This test checks that reusing FileManager produces deterministic results on case-insensitive filesystems.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- dir1/arm/lower.h
//--- dir2/ARM/upper.h
//--- t1.c
#include "upper.h"
//--- t2.c
#include "arm/lower.h"

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/t1.c -I DIR/dir2/ARM -I DIR/dir1",
  "file": "DIR/t1.c"
},{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/t2.c -I DIR/dir2     -I DIR/dir1",
  "file": "DIR/t2.c"
}]

//--- cdb-rev.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/t2.c -I DIR/dir2     -I DIR/dir1",
  "file": "DIR/t2.c"
},{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/t1.c -I DIR/dir2/ARM -I DIR/dir1",
  "file": "DIR/t1.c"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template     > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb-rev.json.template > %t/cdb-rev.json

// RUN: clang-scan-deps -compilation-database=%t/cdb.json     -format make -j 1 | sed 's:\\\\\?:/:g' | FileCheck %s

// In the reversed case, Clang starts by scanning "t2.c". When looking up the "arm/lower.h" header,
// the string is appended to "DIR/dir2". That file ("DIR/dir2/arm/lower.h") doesn't exist, but when
// learning so, the FileManager stats and caches the parent directory ("DIR/dir2/arm"), using the
// UID as the key.
// When scanning "t1.c" later on, the "DIR/dir2/ARM" search directory is assigned the **same**
// directory entry (with lowercase "arm"), since they share the UID on case-insensitive filesystems.
// To preserve the correct case throughout the compiler for any file within that directory, it's
// important to use the spelling actually used, not just the cached one.
// RUN: clang-scan-deps -compilation-database=%t/cdb-rev.json -format make -j 1 | sed 's:\\\\\?:/:g' | FileCheck %s

// CHECK: ARM/upper.h
