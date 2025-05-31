// RUN: %clang_cc1 -emit-llvm -triple hexagon-unknown-unknown %s -o /dev/null
// REQUIRES: hexagon-registered-target

// Testcase for bug 14744.  Empty file is sufficient, since the problem
// was a bad data layout string in the Hexagon target causing an ICE
// when compiling any Hexagon program.

int x;  // In C99, a translation unit needs to have at least one declaration.

