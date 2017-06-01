// Make sure -finline-functions family flags are behaving correctly.
//
// REQUIRES: x86-registered-target
//
// RUN: %clang_cc1 -triple i686-pc-win32 -emit-llvm %s -o - | FileCheck -check-prefix=NOINLINE %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fexperimental-new-pass-manager -emit-llvm %s -o - | FileCheck -check-prefix=NOINLINE %s
// RUN: %clang_cc1 -triple i686-pc-win32 -O3 -fno-inline-functions -emit-llvm %s -o - | FileCheck -check-prefix=NOINLINE %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fexperimental-new-pass-manager -O3 -fno-inline-functions -emit-llvm %s -o - | FileCheck -check-prefix=NOINLINE %s
// RUN: %clang_cc1 -triple i686-pc-win32 -O3 -finline-hint-functions -emit-llvm %s -o - | FileCheck -check-prefix=HINT %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fexperimental-new-pass-manager -O3 -finline-hint-functions -emit-llvm %s -o - | FileCheck -check-prefix=HINT %s
// RUN: %clang_cc1 -triple i686-pc-win32 -O3 -finline-functions -emit-llvm %s -o - | FileCheck -check-prefix=INLINE %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fexperimental-new-pass-manager -O3 -finline-functions -emit-llvm %s -o - | FileCheck -check-prefix=INLINE %s

inline int inline_hint(int a, int b) { return(a+b); }

int inline_no_hint(int a, int b) { return (a/b); }

inline __attribute__ ((__always_inline__)) int inline_always(int a, int b) { return(a*b); }

volatile int *pa = (int*) 0x1000;
void foo() {
// NOINLINE-LABEL: @foo
// HINT-LABEL: @foo
// INLINE-LABEL: @foo
// NOINLINE: call i32 @inline_hint
// HINT-NOT: call i32 @inline_hint
// INLINE-NOT: call i32 @inline_hint
    pa[0] = inline_hint(pa[1],pa[2]);
// NOINLINE-NOT: call i32 @inline_always
// HINT-NOT: call i32 @inline_always
// INLINE-NOT: call i32 @inline_always
    pa[3] = inline_always(pa[4],pa[5]);
// NOINLINE: call i32 @inline_no_hint
// HINT: call i32 @inline_no_hint
// INLINE-NOT: call i32 @inline_no_hint
    pa[6] = inline_no_hint(pa[7], pa[8]);
}
