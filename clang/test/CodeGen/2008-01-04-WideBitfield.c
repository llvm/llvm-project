// RUN: %clang_cc1 -emit-llvm -o - %s
// PR1386
typedef unsigned long long uint64_t;
struct X {
  unsigned char pad : 4;
  uint64_t a : 64;
} __attribute__((packed)) x;

uint64_t f(void)
{
  return x.a;
}
