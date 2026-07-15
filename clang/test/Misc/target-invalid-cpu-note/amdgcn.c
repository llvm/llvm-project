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
// CHECK-SAME: {{^}}, gfx942
// CHECK-SAME: {{^}}, gfx950
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
// CHECK-SAME: {{^}}, gfx1153
// CHECK-SAME: {{^}}, gfx1154
// CHECK-SAME: {{^}}, gfx1170
// CHECK-SAME: {{^}}, gfx1171
// CHECK-SAME: {{^}}, gfx1172
// CHECK-SAME: {{^}}, gfx1200
// CHECK-SAME: {{^}}, gfx1201
// CHECK-SAME: {{^}}, gfx1250
// CHECK-SAME: {{^}}, gfx1251
// CHECK-SAME: {{^}}, gfx1310
// CHECK-SAME: {{^}}, gfx9-generic
// CHECK-SAME: {{^}}, gfx9-4-generic
// CHECK-SAME: {{^}}, gfx10-1-generic
// CHECK-SAME: {{^}}, gfx10-3-generic
// CHECK-SAME: {{^}}, gfx11-generic
// CHECK-SAME: {{^}}, gfx11-7-generic
// CHECK-SAME: {{^}}, gfx12-generic
// CHECK-SAME: {{^}}, gfx12-5-generic
// CHECK-SAME: {{^}}, gfx13-generic
// CHECK-SAME: {{$}}

// When the triple carries a major-family subarch, only the GPUs in that family
// are valid (a CPU from another family is rejected).
// RUN: not %clang_cc1 -triple amdgpu9--- -target-cpu gfx1030 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=GFX9 %s
// GFX9: error: unknown target CPU 'gfx1030'
// GFX9-NEXT: note: valid target CPU values are:
// GFX9-SAME: {{^}} gfx900
// GFX9-SAME: {{^}}, gfx902
// GFX9-SAME: {{^}}, gfx904
// GFX9-SAME: {{^}}, gfx906
// GFX9-SAME: {{^}}, gfx909
// GFX9-SAME: {{^}}, gfx90c
// GFX9-SAME: {{^}}, gfx9-generic
// GFX9-SAME: {{$}}

// gfx908 and gfx90a are not part of the gfx9-generic family (they are their own
// major subarches), so they are rejected by the amdgpu9 triple above and only
// accepted by their own specific subarch triples.
// RUN: not %clang_cc1 -triple amdgpu9.08--- -target-cpu gfx900 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=GFX908 %s
// GFX908: error: unknown target CPU 'gfx900'
// GFX908-NEXT: note: valid target CPU values are:
// GFX908-SAME: {{^}} gfx908
// GFX908-SAME: {{$}}

// When the triple carries a specific subarch, only that GPU is valid, so even a
// CPU from the same major family but a different specific subarch is rejected.
// RUN: not %clang_cc1 -triple amdgpu9.0a--- -target-cpu gfx900 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=GFX90A %s
// GFX90A: error: unknown target CPU 'gfx900'
// GFX90A-NEXT: note: valid target CPU values are:
// GFX90A-SAME: {{^}} gfx90a
// GFX90A-SAME: {{$}}

// gfx810 is its own major subarch.
// RUN: not %clang_cc1 -triple amdgpu8.10--- -target-cpu gfx803 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=GFX810 %s
// GFX810: error: unknown target CPU 'gfx803'
// GFX810-NEXT: note: valid target CPU values are:
// GFX810-SAME: {{^}} gfx810
// GFX810-SAME: {{^}}, stoney
// GFX810-SAME: {{$}}

// amdgpu11.7 is a major-family subarch covering the gfx117x GPUs.
// RUN: not %clang_cc1 -triple amdgpu11.7--- -target-cpu gfx1100 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=GFX11_7 %s
// GFX11_7: error: unknown target CPU 'gfx1100'
// GFX11_7-NEXT: note: valid target CPU values are:
// GFX11_7-SAME: {{^}} gfx1170
// GFX11_7-SAME: {{^}}, gfx1171
// GFX11_7-SAME: {{^}}, gfx1172
// GFX11_7-SAME: {{$}}

// amdgpu13 is a major-family subarch covering the gfx131x GPUs.
// RUN: not %clang_cc1 -triple amdgpu13--- -target-cpu gfx1200 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=GFX13 %s
// GFX13: error: unknown target CPU 'gfx1200'
// GFX13-NEXT: note: valid target CPU values are:
// GFX13-SAME: {{^}} gfx1310
// GFX13-SAME: {{$}}
