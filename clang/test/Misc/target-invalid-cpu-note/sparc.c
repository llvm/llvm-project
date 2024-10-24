// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple sparc--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix SPARC
// SPARC: error: unknown target CPU 'not-a-cpu'
// SPARC-NEXT: note: valid target CPU values are:
// SPARC-SAME: {{^}} v8
// SPARC-SAME: {{^}}, supersparc
// SPARC-SAME: {{^}}, sparclite
// SPARC-SAME: {{^}}, f934
// SPARC-SAME: {{^}}, hypersparc
// SPARC-SAME: {{^}}, sparclite86x
// SPARC-SAME: {{^}}, sparclet
// SPARC-SAME: {{^}}, tsc701
// SPARC-SAME: {{^}}, v9
// SPARC-SAME: {{^}}, ultrasparc
// SPARC-SAME: {{^}}, ultrasparc3
// SPARC-SAME: {{^}}, niagara
// SPARC-SAME: {{^}}, niagara2
// SPARC-SAME: {{^}}, niagara3
// SPARC-SAME: {{^}}, niagara4
// SPARC-SAME: {{^}}, ma2100
// SPARC-SAME: {{^}}, ma2150
// SPARC-SAME: {{^}}, ma2155
// SPARC-SAME: {{^}}, ma2450
// SPARC-SAME: {{^}}, ma2455
// SPARC-SAME: {{^}}, ma2x5x
// SPARC-SAME: {{^}}, ma2080
// SPARC-SAME: {{^}}, ma2085
// SPARC-SAME: {{^}}, ma2480
// SPARC-SAME: {{^}}, ma2485
// SPARC-SAME: {{^}}, ma2x8x
// SPARC-SAME: {{^}}, leon2
// SPARC-SAME: {{^}}, at697e
// SPARC-SAME: {{^}}, at697f
// SPARC-SAME: {{^}}, leon3
// SPARC-SAME: {{^}}, ut699
// SPARC-SAME: {{^}}, gr712rc
// SPARC-SAME: {{^}}, leon4
// SPARC-SAME: {{^}}, gr740
// SPARC-SAME: {{$}}

// RUN: not %clang_cc1 -triple sparcv9--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix SPARCV9
// SPARCV9: error: unknown target CPU 'not-a-cpu'
// SPARCV9-NEXT: note: valid target CPU values are:
// SPARCV9-SAME: {{^}} v9
// SPARCV9-SAME: {{^}}, ultrasparc
// SPARCV9-SAME: {{^}}, ultrasparc3
// SPARCV9-SAME: {{^}}, niagara
// SPARCV9-SAME: {{^}}, niagara2
// SPARCV9-SAME: {{^}}, niagara3
// SPARCV9-SAME: {{^}}, niagara4
// SPARCV9-SAME: {{$}}
