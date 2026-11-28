// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple m68k--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} generic
// CHECK-SAME: {{^}}, M68000
// CHECK-SAME: {{^}}, M68010
// CHECK-SAME: {{^}}, M68020
// CHECK-SAME: {{^}}, M68030
// CHECK-SAME: {{^}}, M68040
// CHECK-SAME: {{^}}, M68060{{$}}
