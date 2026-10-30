// RUN: %clang_cc1 -std=c11 -Wimplicit-float-conversion -fsyntax-only \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// Issue #93288 example
float f = 48 * (1.0 / 255.0);

// CHECK: {{.*}}:5:11: warning: implicit conversion loses floating-point precision: 'double' to 'float'
// CHECK: fix-it:"{{.*}}":{5:11-5:11}:"(float)("
// CHECK-NEXT: fix-it:"{{.*}}":{5:29-5:29}:")"
