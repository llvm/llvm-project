// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 %s -triple powerpc-pc-linux -emit-llvm -o - | grep {i32 32} | count 3
// XFAIL: *
//  Every printf has 'i32 0' for the GEP of the string; no point counting those.
typedef unsigned int Foo __attribute__((aligned(32)));
typedef union{Foo:0;}a;
typedef union{int x; Foo:0;}b;
extern int printf(const char*, ...);
main() {
  printf("%ld\n", sizeof(a));
  printf("%ld\n", __alignof__(a));
  printf("%ld\n", sizeof(b));
  printf("%ld\n", __alignof__(b));
}
