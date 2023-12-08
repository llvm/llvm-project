// RUN: %clang_cc1 -Wno-gnu-designator -verify %s
// expected-no-diagnostics
struct { int x, y, z[12]; } value = { x:17, .z [3 ... 5] = 7 };
