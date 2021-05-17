// When entry values are emitted, expect a subprogram for extern decls so that
// the dwarf generator can describe call site parameters at extern call sites.
//
// Initial implementation relied on the 'retainedTypes:' from the corresponding
// DICompileUnit, so we also ensure that we do not store the extern declaration
// subprogram into the 'retainedTypes:'.
//
// RUN: %clang -g -O2 -target x86_64-none-linux-gnu -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=DECLS-FOR-EXTERN

// Similarly, when the debugger tuning is gdb, expect a subprogram for extern
// decls so that the dwarf generator can describe information needed for tail
// call frame reconstrution.
//
// RUN: %clang -g -O2 -target x86_64-none-linux-gnu -ggdb -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=DECLS-FOR-EXTERN
//
// Do not emit a subprogram for extern decls when entry values are disabled and
// the tuning is not set to gdb.
//
// RUN: %clang -g -O2 -target x86_64-none-linux-gnu -gsce -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s -check-prefix=NO-DECLS-FOR-EXTERN

// DECLS-FOR-EXTERN-NOT: !DICompileUnit({{.*}}retainedTypes: !{{[0-9]+}}
// DECLS-FOR-EXTERN: !DISubprogram(name: "fn1"
// DECLS-FOR-EXTERN-NOT: !DISubprogram(name: "memcmp"
// DECLS-FOR-EXTERN-NOT: !DISubprogram(name: "__some_reserved_name"

// NO-DECLS-FOR-EXTERN-NOT: !DISubprogram(name: "fn1"
// NO-DECLS-FOR-EXTERN-NOT: !DISubprogram(name: "memcmp"
// NO-DECLS-FOR-EXTERN-NOT: !DISubprogram(name: "__some_reserved_name"

extern int fn1(int a, int b);
extern int memcmp(const void *s1, const void *s2, unsigned long n);
extern void __some_reserved_name(void);

int fn2 (int *src, int *dst) {
  int x = 4, y = 5;
  int res = fn1(x, y);
  int res2 = memcmp(dst, src, res);
  __some_reserved_name();
  return res + res2;
}

