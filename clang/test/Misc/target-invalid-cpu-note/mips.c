// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple mips--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} mips1
// CHECK-SAME: {{^}}, mips2
// CHECK-SAME: {{^}}, mips3
// CHECK-SAME: {{^}}, mips4
// CHECK-SAME: {{^}}, mips5
// CHECK-SAME: {{^}}, mips32
// CHECK-SAME: {{^}}, mips32r2
// CHECK-SAME: {{^}}, mips32r3
// CHECK-SAME: {{^}}, mips32r5
// CHECK-SAME: {{^}}, mips32r6
// CHECK-SAME: {{^}}, mips64
// CHECK-SAME: {{^}}, mips64r2
// CHECK-SAME: {{^}}, mips64r3
// CHECK-SAME: {{^}}, mips64r5
// CHECK-SAME: {{^}}, mips64r6
// CHECK-SAME: {{^}}, octeon
// CHECK-SAME: {{^}}, octeon+
// CHECK-SAME: {{^}}, p5600
// CHECK-SAME: {{$}}
