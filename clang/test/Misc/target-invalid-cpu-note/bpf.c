// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple bpf--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} generic
// CHECK-SAME: {{^}}, v1
// CHECK-SAME: {{^}}, v2
// CHECK-SAME: {{^}}, v3
// CHECK-SAME: {{^}}, v4
// CHECK-SAME: {{^}}, probe
// CHECK-SAME: {{$}}

