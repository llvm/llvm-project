// RUN: %clang_cc1 -E -x c -o - %s | grep '[.][*]'
// PR4395
#define X .*
X
