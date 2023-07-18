// REQUIRES: arm-registered-target
// RUN: not %clang_cc1 -triple armv7 -target-feature +neon %s -S -o /dev/null 2>&1 | FileCheck %s

// rdar://13446483
typedef __attribute__((neon_vector_type(2))) long long int64x2_t;
typedef struct int64x2x4_t {
  int64x2_t val[4];
} int64x2x4_t;
int64x2x4_t t1(const long long a[]) {
  int64x2x4_t r;
  __asm__("vldm %[a], { %q[r0], %q[r1], %q[r2], %q[r3] }"
          : [r0] "=r"(r.val[0]), // expected-warning {{value size does not match register size specified by the constraint and modifier}}
            [r1] "=r"(r.val[1]), // expected-warning {{value size does not match register size specified by the constraint and modifier}}
            [r2] "=r"(r.val[2]), // expected-warning {{value size does not match register size specified by the constraint and modifier}}
            [r3] "=r"(r.val[3])  // expected-warning {{value size does not match register size specified by the constraint and modifier}}
          : [a] "r"(a));
  return r;
}
// We should see all four errors, rather than report a fatal error after the first.
// CHECK: error: non-trivial scalar-to-vector conversion, possible invalid constraint for vector type
// CHECK: error: non-trivial scalar-to-vector conversion, possible invalid constraint for vector type
// CHECK: error: non-trivial scalar-to-vector conversion, possible invalid constraint for vector type
// CHECK: error: non-trivial scalar-to-vector conversion, possible invalid constraint for vector type
