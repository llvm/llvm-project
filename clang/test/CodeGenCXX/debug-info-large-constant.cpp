// RUN: %clang_cc1 -g -triple=x86_64-apple-darwin %s -o /dev/null
// PR 8913

typedef __uint128_t word128;
static const word128 m126 = 0xffffffffffffffffULL;
word128 foo() {
  return m126;
}
