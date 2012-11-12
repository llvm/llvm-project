// RUN: %clang_cc1 -std=gnu89 -emit-llvm %s  -o /dev/null

/* This is apparently legal C.
 */
extern __inline__ void test() { }

void test() {
}
