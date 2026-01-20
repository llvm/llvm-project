// RUN: %clang -g -target bpf -S -emit-llvm %s -o - | FileCheck %s
//
// No debug info is produced for unreferenced functions.
// CHECK-NOT: !DISubprogram
void unref(void);
void unref2(typeof(unref));

// No debug info for unused extern variables as well.
// CHECK-NOT: !DiGlobalVariable
extern int unused;
extern int unused2[sizeof(unused)];
