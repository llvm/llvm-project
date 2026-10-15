// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// An AMDGPU name is an offload architecture but never an NVPTX GPU name.
// RUN: not %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu gfx906 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=AMD
// AMD: error: unknown target CPU 'gfx906'

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
// CHECK-SAME: {{^}}, sm_88
// CHECK-SAME: {{^}}, sm_89
// CHECK-SAME: {{^}}, sm_90
// CHECK-SAME: {{^}}, sm_90a
// CHECK-SAME: {{^}}, sm_100
// CHECK-SAME: {{^}}, sm_100a
// CHECK-SAME: {{^}}, sm_100f
// CHECK-SAME: {{^}}, sm_101
// CHECK-SAME: {{^}}, sm_101a
// CHECK-SAME: {{^}}, sm_101f
// CHECK-SAME: {{^}}, sm_103
// CHECK-SAME: {{^}}, sm_103a
// CHECK-SAME: {{^}}, sm_103f
// CHECK-SAME: {{^}}, sm_107
// CHECK-SAME: {{^}}, sm_107a
// CHECK-SAME: {{^}}, sm_107f
// CHECK-SAME: {{^}}, sm_110
// CHECK-SAME: {{^}}, sm_110a
// CHECK-SAME: {{^}}, sm_110f
// CHECK-SAME: {{^}}, sm_120
// CHECK-SAME: {{^}}, sm_120a
// CHECK-SAME: {{^}}, sm_120f
// CHECK-SAME: {{^}}, sm_121
// CHECK-SAME: {{^}}, sm_121a
// CHECK-SAME: {{^}}, sm_121f
// CHECK-SAME: {{$}}
