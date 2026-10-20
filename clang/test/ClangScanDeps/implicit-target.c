// Check that we can detect an implicit target when clang is invoked as
// <triple->clang. Using an implicit triple requires that the target actually
// is available, too.
// REQUIRES: x86-registered-target

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

// Check that we can deduce this both when using a compilation database, and when using
// a literal command line.

// RUN: clang-scan-deps -format experimental-full -compilation-database %t/cdb.json | FileCheck %s

// RUN: clang-scan-deps -format experimental-full -- x86_64-w64-mingw32-clang %t/source.c -o %t/source.o | FileCheck %s

// CHECK: "-triple",
// CHECK-NEXT: "x86_64-w64-windows-gnu",


//--- cdb.json.in
[
  {
    "directory": "DIR"
    "command": "x86_64-w64-mingw32-clang -c DIR/source.c -o DIR/source.o"
    "file": "DIR/source.c"
  },
]

//--- source.c
void func(void) {}
