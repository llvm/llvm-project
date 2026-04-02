// Check `a` and `d` have aliases while keeping their original symbols.
// GlobalOpt would usually replace these internal-linkage functions with
// their external-linkage aliases. That's currently prevented as a side effect
// of adding discardable functions to the compiler-used global.

// RUN: %clang -cc1 %s -emit-obj -O3 -debug-info-kind=limited -fdynamic-debugging -o - -triple x86_64-unknown-unknown | llvm-nm - | FileCheck %s
// CHECK: t a
// CHECK: T a.dyndbg.[[hash:[A-Z0-9]+]]
// CHECK: T b
// CHECK: U c
// CHECK: t d
// CHECK: T d.dyndbg.[[hash]]
// CHECK: B g

int g;
int c();

__attribute__((always_inline))
static inline int d() { return c(); }

__attribute__((always_inline))
static int a() { return g; }

int b() { return a() + d() ;}
