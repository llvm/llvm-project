// An unimplemented ARM builtin must report NYI once and not crash. A vector
// result is the interesting case: it is stored through memory, so returning a
// null value here would reach createStore. Covers all four arms of the ARM
// dispatch in emitTargetArchBuiltinExpr.
//
// CIRGen stops after the first NYI, so only one such call fits per file.

// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -target-feature +neon -fclangir -emit-llvm %s -verify -o /dev/null
// RUN: %clang_cc1 -triple armebv7-unknown-linux-gnueabihf -target-feature +neon -fclangir -emit-llvm %s -verify -o /dev/null
// RUN: %clang_cc1 -triple thumbv7-unknown-linux-gnueabihf -target-feature +neon -fclangir -emit-llvm %s -verify -o /dev/null
// RUN: %clang_cc1 -triple thumbebv7-unknown-linux-gnueabihf -target-feature +neon -fclangir -emit-llvm %s -verify -o /dev/null

typedef __attribute__((neon_vector_type(4))) int int32x4_t;

// expected-error@+1 {{ClangIR code gen Not Yet Implemented: unimplemented ARM builtin call: __builtin_neon_vld1q_v}}
int32x4_t ld(const int *p) { int32x4_t r = __builtin_neon_vld1q_v(p, 34); return r; }
