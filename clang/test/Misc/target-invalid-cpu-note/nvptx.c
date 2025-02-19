// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple nvptx--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not={{[a-zA-Z0-9]}}
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} sm_20
// CHECK-SAME: {{^}}, sm_21
// CHECK-SAME: {{^}}, sm_30
// CHECK-SAME: {{^}}, sm_32
// CHECK-SAME: {{^}}, sm_35
// CHECK-SAME: {{^}}, sm_37
// CHECK-SAME: {{^}}, sm_50
// CHECK-SAME: {{^}}, sm_52
// CHECK-SAME: {{^}}, sm_53
// CHECK-SAME: {{^}}, sm_60
// CHECK-SAME: {{^}}, sm_61
// CHECK-SAME: {{^}}, sm_62
// CHECK-SAME: {{^}}, sm_70
// CHECK-SAME: {{^}}, sm_72
// CHECK-SAME: {{^}}, sm_75
// CHECK-SAME: {{^}}, sm_80
// CHECK-SAME: {{^}}, sm_86
// CHECK-SAME: {{^}}, sm_87
// CHECK-SAME: {{^}}, sm_89
// CHECK-SAME: {{^}}, sm_90
// CHECK-SAME: {{^}}, sm_90a
// CHECK-SAME: {{^}}, sm_100
// CHECK-SAME: {{^}}, sm_100a
// CHECK-SAME: {{^}}, gfx600
// CHECK-SAME: {{^}}, gfx601
// CHECK-SAME: {{^}}, gfx602
// CHECK-SAME: {{^}}, gfx700
// CHECK-SAME: {{^}}, gfx701
// CHECK-SAME: {{^}}, gfx702
// CHECK-SAME: {{^}}, gfx703
// CHECK-SAME: {{^}}, gfx704
// CHECK-SAME: {{^}}, gfx705
// CHECK-SAME: {{^}}, gfx801
// CHECK-SAME: {{^}}, gfx802
// CHECK-SAME: {{^}}, gfx803
// CHECK-SAME: {{^}}, gfx805
// CHECK-SAME: {{^}}, gfx810
// CHECK-SAME: {{^}}, gfx9-generic
// CHECK-SAME: {{^}}, gfx900
// CHECK-SAME: {{^}}, gfx902
// CHECK-SAME: {{^}}, gfx904
// CHECK-SAME: {{^}}, gfx906
// CHECK-SAME: {{^}}, gfx908
// CHECK-SAME: {{^}}, gfx909
// CHECK-SAME: {{^}}, gfx90a
// CHECK-SAME: {{^}}, gfx90c
// CHECK-SAME: {{^}}, gfx9-4-generic
// CHECK-SAME: {{^}}, gfx942
// CHECK-SAME: {{^}}, gfx950
// CHECK-SAME: {{^}}, gfx10-1-generic
// CHECK-SAME: {{^}}, gfx1010
// CHECK-SAME: {{^}}, gfx1011
// CHECK-SAME: {{^}}, gfx1012
// CHECK-SAME: {{^}}, gfx1013
// CHECK-SAME: {{^}}, gfx10-3-generic
// CHECK-SAME: {{^}}, gfx1030
// CHECK-SAME: {{^}}, gfx1031
// CHECK-SAME: {{^}}, gfx1032
// CHECK-SAME: {{^}}, gfx1033
// CHECK-SAME: {{^}}, gfx1034
// CHECK-SAME: {{^}}, gfx1035
// CHECK-SAME: {{^}}, gfx1036
// CHECK-SAME: {{^}}, gfx11-generic
// CHECK-SAME: {{^}}, gfx1100
// CHECK-SAME: {{^}}, gfx1101
// CHECK-SAME: {{^}}, gfx1102
// CHECK-SAME: {{^}}, gfx1103
// CHECK-SAME: {{^}}, gfx1150
// CHECK-SAME: {{^}}, gfx1151
// CHECK-SAME: {{^}}, gfx1152
// CHECK-SAME: {{^}}, gfx1153
// CHECK-SAME: {{^}}, gfx12-generic
// CHECK-SAME: {{^}}, gfx1200
// CHECK-SAME: {{^}}, gfx1201
// CHECK-SAME: {{^}}, amdgcnspirv
// CHECK-SAME: {{$}}
