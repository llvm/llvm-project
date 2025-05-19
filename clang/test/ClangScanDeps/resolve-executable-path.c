// UNSUPPORTED: system-windows

// Check that we expand the executable name to an absolute path, when invoked
// with a plain executable name, which is implied to be found in PATH.
// REQUIRES: x86-registered-target

// RUN: rm -rf %t
// RUN: mkdir -p %t/bin
// RUN: ln -s %clang %t/bin/x86_64-w64-mingw32-clang
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

// Check that we can deduce this both when using a compilation database, and when using
// a literal command line.

// RUN: env "PATH=%t/bin:%PATH%" clang-scan-deps -format experimental-full -compilation-database %t/cdb.json | FileCheck %s -DBASE=%/t

// RUN: env "PATH=%t/bin:%PATH%" clang-scan-deps -format experimental-full -- x86_64-w64-mingw32-clang %t/source.c -o %t/source.o | FileCheck %s -DBASE=%/t

// CHECK: "executable": "[[BASE]]/bin/x86_64-w64-mingw32-clang"

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
