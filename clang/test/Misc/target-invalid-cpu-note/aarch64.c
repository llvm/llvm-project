// This test uses '<prefix>-SAME: {{^}}' to start matching immediately where the
// previous check finished matching (specifically, caret is not treated as
// matching a start of line when used like this in FileCheck).

// RUN: not %clang_cc1 -triple arm64--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple arm64--- -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: error: unknown target CPU 'not-a-cpu'
// CHECK-NEXT: note: valid target CPU values are:
// CHECK-SAME: {{^}} a64fx
// CHECK-SAME: {{^}}, ampere1
// CHECK-SAME: {{^}}, ampere1a
// CHECK-SAME: {{^}}, ampere1b
// CHECK-SAME: {{^}}, apple-a10
// CHECK-SAME: {{^}}, apple-a11
// CHECK-SAME: {{^}}, apple-a12
// CHECK-SAME: {{^}}, apple-a13
// CHECK-SAME: {{^}}, apple-a14
// CHECK-SAME: {{^}}, apple-a15
// CHECK-SAME: {{^}}, apple-a16
// CHECK-SAME: {{^}}, apple-a17
// CHECK-SAME: {{^}}, apple-a7
// CHECK-SAME: {{^}}, apple-a8
// CHECK-SAME: {{^}}, apple-a9
// CHECK-SAME: {{^}}, apple-m1
// CHECK-SAME: {{^}}, apple-m2
// CHECK-SAME: {{^}}, apple-m3
// CHECK-SAME: {{^}}, apple-m4
// CHECK-SAME: {{^}}, apple-s4
// CHECK-SAME: {{^}}, apple-s5
// CHECK-SAME: {{^}}, carmel
// CHECK-SAME: {{^}}, cobalt-100
// CHECK-SAME: {{^}}, cortex-a34
// CHECK-SAME: {{^}}, cortex-a35
// CHECK-SAME: {{^}}, cortex-a510
// CHECK-SAME: {{^}}, cortex-a520
// CHECK-SAME: {{^}}, cortex-a520ae
// CHECK-SAME: {{^}}, cortex-a53
// CHECK-SAME: {{^}}, cortex-a55
// CHECK-SAME: {{^}}, cortex-a57
// CHECK-SAME: {{^}}, cortex-a65
// CHECK-SAME: {{^}}, cortex-a65ae
// CHECK-SAME: {{^}}, cortex-a710
// CHECK-SAME: {{^}}, cortex-a715
// CHECK-SAME: {{^}}, cortex-a72
// CHECK-SAME: {{^}}, cortex-a720
// CHECK-SAME: {{^}}, cortex-a720ae
// CHECK-SAME: {{^}}, cortex-a725
// CHECK-SAME: {{^}}, cortex-a73
// CHECK-SAME: {{^}}, cortex-a75
// CHECK-SAME: {{^}}, cortex-a76
// CHECK-SAME: {{^}}, cortex-a76ae
// CHECK-SAME: {{^}}, cortex-a77
// CHECK-SAME: {{^}}, cortex-a78
// CHECK-SAME: {{^}}, cortex-a78ae
// CHECK-SAME: {{^}}, cortex-a78c
// CHECK-SAME: {{^}}, cortex-r82
// CHECK-SAME: {{^}}, cortex-r82ae
// CHECK-SAME: {{^}}, cortex-x1
// CHECK-SAME: {{^}}, cortex-x1c
// CHECK-SAME: {{^}}, cortex-x2
// CHECK-SAME: {{^}}, cortex-x3
// CHECK-SAME: {{^}}, cortex-x4
// CHECK-SAME: {{^}}, cortex-x925
// CHECK-SAME: {{^}}, cyclone
// CHECK-SAME: {{^}}, exynos-m3
// CHECK-SAME: {{^}}, exynos-m4
// CHECK-SAME: {{^}}, exynos-m5
// CHECK-SAME: {{^}}, falkor
// CHECK-SAME: {{^}}, fujitsu-monaka
// CHECK-SAME: {{^}}, generic
// CHECK-SAME: {{^}}, grace
// CHECK-SAME: {{^}}, kryo
// CHECK-SAME: {{^}}, neoverse-512tvb
// CHECK-SAME: {{^}}, neoverse-e1
// CHECK-SAME: {{^}}, neoverse-n1
// CHECK-SAME: {{^}}, neoverse-n2
// CHECK-SAME: {{^}}, neoverse-n3
// CHECK-SAME: {{^}}, neoverse-v1
// CHECK-SAME: {{^}}, neoverse-v2
// CHECK-SAME: {{^}}, neoverse-v3
// CHECK-SAME: {{^}}, neoverse-v3ae
// CHECK-SAME: {{^}}, oryon-1
// CHECK-SAME: {{^}}, saphira
// CHECK-SAME: {{^}}, thunderx
// CHECK-SAME: {{^}}, thunderx2t99
// CHECK-SAME: {{^}}, thunderx3t110
// CHECK-SAME: {{^}}, thunderxt81
// CHECK-SAME: {{^}}, thunderxt83
// CHECK-SAME: {{^}}, thunderxt88
// CHECK-SAME: {{^}}, tsv110
// CHECK-SAME: {{$}}

