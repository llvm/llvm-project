// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple systemz--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} arch8
// CHECK-SAME: {{^}}, z10
// CHECK-SAME: {{^}}, arch9
// CHECK-SAME: {{^}}, z196
// CHECK-SAME: {{^}}, arch10
// CHECK-SAME: {{^}}, zEC12
// CHECK-SAME: {{^}}, arch11
// CHECK-SAME: {{^}}, z13
// CHECK-SAME: {{^}}, arch12
// CHECK-SAME: {{^}}, z14
// CHECK-SAME: {{^}}, arch13
// CHECK-SAME: {{^}}, z15
// CHECK-SAME: {{^}}, arch14
// CHECK-SAME: {{^}}, z16
// CHECK-SAME: {{^}}, arch15
// CHECK-SAME: {{$}}
