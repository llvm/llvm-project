// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// XFAIL: mips

// CHECK-NOT: readonly
// CHECK-NOT: readnone

// The struct being passed byval means that we cannot mark the
// function readnone.  Readnone would allow stores to the arg to
// be deleted in the caller.  We also don't allow readonly since
// the callee might write to the byval parameter.  The inliner
// would have to assume the worse and introduce an explicit
// temporary when inlining such a function, which is costly for
// the common case in which the byval argument is not written.
struct S { int A[1000]; };
int __attribute__ ((const)) f(struct S x) { x.A[1] = 0; return x.A[0]; }
int g(struct S x) __attribute__ ((pure));
int h(struct S x) { return g(x); }
