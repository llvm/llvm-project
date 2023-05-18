// RUN: %clang_cc1 -E %s | grep '1: aaab 2'
// RUN: %clang_cc1 -E %s | grep '2: 2 baaa'
// RUN: %clang_cc1 -E %s | grep '3: 2 xx'

#define a(n) aaa ## n
#define b 2
1: a(b b)   // aaab 2    2 gets expanded, not b.

#undef a
#undef b
#define a(n) n ## aaa
#define b 2
2: a(b b)   // 2 baaa    2 gets expanded, not b.

#define baaa xx
3: a(b b)   // 2 xx

