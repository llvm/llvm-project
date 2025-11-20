// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple r600--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} r600
// CHECK-SAME: {{^}}, rv630
// CHECK-SAME: {{^}}, rv635
// CHECK-SAME: {{^}}, r630
// CHECK-SAME: {{^}}, rs780
// CHECK-SAME: {{^}}, rs880
// CHECK-SAME: {{^}}, rv610
// CHECK-SAME: {{^}}, rv620
// CHECK-SAME: {{^}}, rv670
// CHECK-SAME: {{^}}, rv710
// CHECK-SAME: {{^}}, rv730
// CHECK-SAME: {{^}}, rv740
// CHECK-SAME: {{^}}, rv770
// CHECK-SAME: {{^}}, cedar
// CHECK-SAME: {{^}}, palm
// CHECK-SAME: {{^}}, cypress
// CHECK-SAME: {{^}}, hemlock
// CHECK-SAME: {{^}}, juniper
// CHECK-SAME: {{^}}, redwood
// CHECK-SAME: {{^}}, sumo
// CHECK-SAME: {{^}}, sumo2
// CHECK-SAME: {{^}}, barts
// CHECK-SAME: {{^}}, caicos
// CHECK-SAME: {{^}}, aruba
// CHECK-SAME: {{^}}, cayman
// CHECK-SAME: {{^}}, turks
// CHECK-SAME: {{$}}
