// When entry values are emitted, expect a subprogram for extern decls so that
// the dwarf generator can describe call site parameters at extern call sites.
//
// RUN: %clang -Xclang -femit-debug-entry-values -g -O2 -target x86_64-none-linux-gnu -S -emit-llvm %s -o - | FileCheck %s -check-prefix=ENTRY-VAL
// ENTRY-VAL: !DISubprogram(name: "fn1"

// Similarly, when the debugger tuning is gdb, expect a subprogram for extern
// decls so that the dwarf generator can describe information needed for tail
// call frame reconstrution.
//
// RUN: %clang -g -O2 -target x86_64-none-linux-gnu -ggdb -S -emit-llvm %s -o - | FileCheck %s -check-prefix=GDB
// GDB: !DISubprogram(name: "fn1"
//
// Do not emit a subprogram for extern decls when entry values are disabled and
// the tuning is not set to gdb.
//
// RUN: %clang -g -O2 -target x86_64-none-linux-gnu -gsce -S -emit-llvm %s -o - | FileCheck %s -check-prefix=SCE
// SCE-NOT: !DISubprogram(name: "fn1"

extern int fn1(int a, int b);

int fn2 () {
  int x = 4, y = 5;
  int res = fn1(x, y);

  return res;
}

