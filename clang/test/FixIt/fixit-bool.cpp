// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -std=c++26 -Wint-in-bool-context %s 2>&1 | FileCheck %s

int x;
bool t1 = x << x;
// CHECK-LABEL: 4:13: warning: converting the result of '<<' to a boolean
// CHECK: fix-it:"{{.*}}":{4:11-4:11}:"("
// CHECK-NEXT: fix-it:"{{.*}}":{4:17-4:17}:") != 0"
