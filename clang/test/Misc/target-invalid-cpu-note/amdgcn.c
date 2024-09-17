// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple amdgcn--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} gfx600
// CHECK-SAME: {{^}}, tahiti
// CHECK-SAME: {{^}}, gfx601
// CHECK-SAME: {{^}}, pitcairn
// CHECK-SAME: {{^}}, verde
// CHECK-SAME: {{^}}, gfx602
// CHECK-SAME: {{^}}, hainan
// CHECK-SAME: {{^}}, oland
// CHECK-SAME: {{^}}, gfx700
// CHECK-SAME: {{^}}, kaveri
// CHECK-SAME: {{^}}, gfx701
// CHECK-SAME: {{^}}, hawaii
// CHECK-SAME: {{^}}, gfx702
// CHECK-SAME: {{^}}, gfx703
// CHECK-SAME: {{^}}, kabini
// CHECK-SAME: {{^}}, mullins
// CHECK-SAME: {{^}}, gfx704
// CHECK-SAME: {{^}}, bonaire
// CHECK-SAME: {{^}}, gfx705
// CHECK-SAME: {{^}}, gfx801
// CHECK-SAME: {{^}}, carrizo
// CHECK-SAME: {{^}}, gfx802
// CHECK-SAME: {{^}}, iceland
// CHECK-SAME: {{^}}, tonga
// CHECK-SAME: {{^}}, gfx803
// CHECK-SAME: {{^}}, fiji
// CHECK-SAME: {{^}}, polaris10
// CHECK-SAME: {{^}}, polaris11
// CHECK-SAME: {{^}}, gfx805
// CHECK-SAME: {{^}}, tongapro
// CHECK-SAME: {{^}}, gfx810
// CHECK-SAME: {{^}}, stoney
// CHECK-SAME: {{^}}, gfx900
// CHECK-SAME: {{^}}, gfx902
// CHECK-SAME: {{^}}, gfx904
// CHECK-SAME: {{^}}, gfx906
// CHECK-SAME: {{^}}, gfx908
// CHECK-SAME: {{^}}, gfx909
// CHECK-SAME: {{^}}, gfx90a
// CHECK-SAME: {{^}}, gfx90c
// CHECK-SAME: {{^}}, gfx940
// CHECK-SAME: {{^}}, gfx941
// CHECK-SAME: {{^}}, gfx942
// CHECK-SAME: {{^}}, gfx1010
// CHECK-SAME: {{^}}, gfx1011
// CHECK-SAME: {{^}}, gfx1012
// CHECK-SAME: {{^}}, gfx1013
// CHECK-SAME: {{^}}, gfx1030
// CHECK-SAME: {{^}}, gfx1031
// CHECK-SAME: {{^}}, gfx1032
// CHECK-SAME: {{^}}, gfx1033
// CHECK-SAME: {{^}}, gfx1034
// CHECK-SAME: {{^}}, gfx1035
// CHECK-SAME: {{^}}, gfx1036
// CHECK-SAME: {{^}}, gfx1100
// CHECK-SAME: {{^}}, gfx1101
// CHECK-SAME: {{^}}, gfx1102
// CHECK-SAME: {{^}}, gfx1103
// CHECK-SAME: {{^}}, gfx1150
// CHECK-SAME: {{^}}, gfx1151
// CHECK-SAME: {{^}}, gfx1152
// CHECK-SAME: {{^}}, gfx1200
// CHECK-SAME: {{^}}, gfx1201
// CHECK-SAME: {{^}}, gfx9-generic
// CHECK-SAME: {{^}}, gfx10-1-generic
// CHECK-SAME: {{^}}, gfx10-3-generic
// CHECK-SAME: {{^}}, gfx11-generic
// CHECK-SAME: {{^}}, gfx12-generic
// CHECK-SAME: {{$}}
