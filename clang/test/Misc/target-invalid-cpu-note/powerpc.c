// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple powerpc--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} generic
// CHECK-SAME: {{^}}, 440
// CHECK-SAME: {{^}}, 440fp
// CHECK-SAME: {{^}}, ppc440
// CHECK-SAME: {{^}}, 450
// CHECK-SAME: {{^}}, 601
// CHECK-SAME: {{^}}, 602
// CHECK-SAME: {{^}}, 603
// CHECK-SAME: {{^}}, 603e
// CHECK-SAME: {{^}}, 603ev
// CHECK-SAME: {{^}}, 604
// CHECK-SAME: {{^}}, 604e
// CHECK-SAME: {{^}}, 620
// CHECK-SAME: {{^}}, 630
// CHECK-SAME: {{^}}, g3
// CHECK-SAME: {{^}}, 7400
// CHECK-SAME: {{^}}, g4
// CHECK-SAME: {{^}}, 7450
// CHECK-SAME: {{^}}, g4+
// CHECK-SAME: {{^}}, 750
// CHECK-SAME: {{^}}, 8548
// CHECK-SAME: {{^}}, ppc405
// CHECK-SAME: {{^}}, ppc464
// CHECK-SAME: {{^}}, ppc476
// CHECK-SAME: {{^}}, 970
// CHECK-SAME: {{^}}, ppc970
// CHECK-SAME: {{^}}, g5
// CHECK-SAME: {{^}}, a2
// CHECK-SAME: {{^}}, ppca2
// CHECK-SAME: {{^}}, ppc-cell-be
// CHECK-SAME: {{^}}, e500
// CHECK-SAME: {{^}}, e500mc
// CHECK-SAME: {{^}}, e5500
// CHECK-SAME: {{^}}, power3
// CHECK-SAME: {{^}}, pwr3
// CHECK-SAME: {{^}}, pwr4
// CHECK-SAME: {{^}}, power4
// CHECK-SAME: {{^}}, pwr5
// CHECK-SAME: {{^}}, power5
// CHECK-SAME: {{^}}, pwr5+
// CHECK-SAME: {{^}}, power5+
// CHECK-SAME: {{^}}, pwr5x
// CHECK-SAME: {{^}}, power5x
// CHECK-SAME: {{^}}, pwr6
// CHECK-SAME: {{^}}, power6
// CHECK-SAME: {{^}}, pwr6x
// CHECK-SAME: {{^}}, power6x
// CHECK-SAME: {{^}}, pwr7
// CHECK-SAME: {{^}}, power7
// CHECK-SAME: {{^}}, pwr8
// CHECK-SAME: {{^}}, power8
// CHECK-SAME: {{^}}, pwr9
// CHECK-SAME: {{^}}, power9
// CHECK-SAME: {{^}}, pwr10
// CHECK-SAME: {{^}}, power10
// CHECK-SAME: {{^}}, pwr11
// CHECK-SAME: {{^}}, power11
// CHECK-SAME: {{^}}, powerpc
// CHECK-SAME: {{^}}, ppc
// CHECK-SAME: {{^}}, ppc32
// CHECK-SAME: {{^}}, powerpc64
// CHECK-SAME: {{^}}, ppc64
// CHECK-SAME: {{^}}, powerpc64le
// CHECK-SAME: {{^}}, ppc64le
// CHECK-SAME: {{^}}, future
// CHECK-SAME: {{$}}
