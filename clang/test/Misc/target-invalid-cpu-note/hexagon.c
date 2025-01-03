// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple hexagon--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} hexagonv5
// CHECK-SAME: {{^}}, hexagonv55
// CHECK-SAME: {{^}}, hexagonv60
// CHECK-SAME: {{^}}, hexagonv62
// CHECK-SAME: {{^}}, hexagonv65
// CHECK-SAME: {{^}}, hexagonv66
// CHECK-SAME: {{^}}, hexagonv67
// CHECK-SAME: {{^}}, hexagonv67t
// CHECK-SAME: {{^}}, hexagonv68
// CHECK-SAME: {{^}}, hexagonv69
// CHECK-SAME: {{^}}, hexagonv71
// CHECK-SAME: {{^}}, hexagonv71t
// CHECK-SAME: {{^}}, hexagonv73
// CHECK-SAME: {{^}}, hexagonv75
// CHECK-SAME: {{^}}, hexagonv79
// CHECK-SAME: {{$}}
