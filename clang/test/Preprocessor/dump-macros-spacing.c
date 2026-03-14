// RUN: %clang_cc1 -E -dD < %s | grep stdin | grep -v define
#define A A
/* 1
 * 2
 * 3
 * 4
 * 5
 * 6
 * 7
 * 8
 */
#define B B

