// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only %s -verify
// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only %s 2>&1 | FileCheck %s

// expected-error@+2 {{resource ranges b[42;42] and b[42;42] overlap within space = 0 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("CBV(b42), CBV(b42)")]
void bad_root_signature_0() {}

// expected-error@+2 {{resource ranges t[0;0] and t[0;0] overlap within space = 3 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("SRV(t0, space = 3), SRV(t0, space = 3)")]
void bad_root_signature_1() {}

// expected-error@+2 {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("UAV(u0, visibility = SHADER_VISIBILITY_PIXEL), UAV(u0, visibility = SHADER_VISIBILITY_PIXEL)")]
void bad_root_signature_2() {}

// expected-error@+2 {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("UAV(u0, visibility = SHADER_VISIBILITY_ALL), UAV(u0, visibility = SHADER_VISIBILITY_PIXEL)")]
void bad_root_signature_3() {}

// expected-error@+2 {{resource ranges u[0;0] and u[0;0] overlap within space = 0 and visibility = Pixel}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("UAV(u0, visibility = SHADER_VISIBILITY_PIXEL), UAV(u0, visibility = SHADER_VISIBILITY_ALL)")]
void bad_root_signature_4() {}

// expected-error@+2 {{resource ranges b[0;0] and b[0;0] overlap within space = 0 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("RootConstants(num32BitConstants=4, b0), RootConstants(num32BitConstants=2, b0)")]
void bad_root_signature_5() {}

// expected-error@+2 {{resource ranges s[3;3] and s[3;3] overlap within space = 0 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("StaticSampler(s3), StaticSampler(s3)")]
void bad_root_signature_6() {}

// expected-error@+2 {{resource ranges t[2;5] and t[0;3] overlap within space = 0 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("DescriptorTable(SRV(t0, numDescriptors=4), SRV(t2, numDescriptors=4))")]
void bad_root_signature_7() {}

// expected-error@+2 {{resource ranges u[2;5] and u[0;unbounded) overlap within space = 0 and visibility = Hull}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility = SHADER_VISIBILITY_HULL), DescriptorTable(UAV(u2, numDescriptors=4))")]
void bad_root_signature_8() {}

// expected-error@+2 {{resource ranges b[2;2] and b[0;2] overlap within space = 0 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("RootConstants(num32BitConstants=4, b2), DescriptorTable(CBV(b0, numDescriptors=3))")]
void bad_root_signature_9() {}

// expected-error@+2 {{resource ranges s[17;17] and s[4;unbounded) overlap within space = 0 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("StaticSampler(s17), DescriptorTable(Sampler(s0, numDescriptors=3),Sampler(s4, numDescriptors=unbounded))")]
void bad_root_signature_10() {}

// expected-error@+2 {{resource ranges b[45;45] and b[4;unbounded) overlap within space = 0 and visibility = Geometry}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("DescriptorTable(CBV(b4, numDescriptors=unbounded)), CBV(b45, visibility = SHADER_VISIBILITY_GEOMETRY)")]
void bad_root_signature_11() {}

#define ReportFirstOverlap \
 "DescriptorTable( " \
 "  CBV(b4, numDescriptors = 4), " \
 "  CBV(b1, numDescriptors = 2), " \
 "  CBV(b0, numDescriptors = 8), " \
 ")"

// expected-error@+4 {{resource ranges b[4;7] and b[0;7] overlap within space = 0 and visibility = All}}
// expected-error@+3 {{resource ranges b[1;2] and b[0;7] overlap within space = 0 and visibility = All}}
// expected-note@+2 {{overlapping resource range here}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature(ReportFirstOverlap)]
void bad_root_signature_12() {}

// expected-error@+2 {{resource ranges s[2;2] and s[2;2] overlap within space = 0 and visibility = Vertex}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature("StaticSampler(s2, visibility=SHADER_VISIBILITY_ALL), DescriptorTable(Sampler(s2), visibility=SHADER_VISIBILITY_VERTEX)")]
void valid_root_signature_13() {}

#define DemoNoteSourceLocations \
 "DescriptorTable( " \
 "  CBV(b4, numDescriptors = 4), " \
 "  SRV(t22, numDescriptors = 1), " \
 "  UAV(u42, numDescriptors = 2), " \
 "  CBV(b9, numDescriptors = 8), " \
 "  SRV(t12, numDescriptors = 3), " \
 "  UAV(u3, numDescriptors = 16), " \
 "  SRV(t9, numDescriptors = 1), " \
 "  CBV(b1, numDescriptors = 2), " \
 "  SRV(t17, numDescriptors = 7), " \
 "  UAV(u0, numDescriptors = 3), " \
 ")"

// CHECK: [[@LINE-11]]:5: note: expanded from macro 'DemoNoteSourceLocations'
// CHECK-NEXT: [[@LINE-12]] | "  SRV(t22, numDescriptors = 1), "
// CHECK-NEXT:              |    ^
// CHECK: [[@LINE-7]]:5: note: expanded from macro 'DemoNoteSourceLocations'
// CHECK-NEXT: [[@LINE-8]]  | "  SRV(t17, numDescriptors = 7), " \
// CHECK-NEXT:              |    ^

// expected-error@+2 {{resource ranges t[22;22] and t[17;23] overlap within space = 0 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature(DemoNoteSourceLocations)]
void bad_root_signature_14() {}

#define DuplicatesRootSignature \
  "CBV(b0), CBV(b0), CBV(b0), DescriptorTable(CBV(b0, numDescriptors = 2))"

// expected-error@+6 {{resource ranges b[0;0] and b[0;0] overlap within space = 0 and visibility = All}}
// expected-note@+5 {{overlapping resource range here}}
// expected-error@+4 {{resource ranges b[0;0] and b[0;0] overlap within space = 0 and visibility = All}}
// expected-note@+3 {{overlapping resource range here}}
// expected-error@+2 {{resource ranges b[0;1] and b[0;0] overlap within space = 0 and visibility = All}}
// expected-note@+1 {{overlapping resource range here}}
[RootSignature(DuplicatesRootSignature)]
void valid_root_signature_15() {}

#define AppendingToUnbound \
  "DescriptorTable(CBV(b1, numDescriptors = unbounded), CBV(b0))"

// expected-error@+1 {{offset appends to unbounded descriptor range}}
[RootSignature(AppendingToUnbound)]
void append_to_unbound_signature() {}

#define DirectOffsetOverflow \
  "DescriptorTable(CBV(b0, offset = 4294967294 , numDescriptors = 6))"

// expected-error@+1 {{descriptor range offset overflows [4294967294, 4294967299]}}
[RootSignature(DirectOffsetOverflow)]
void direct_offset_overflow_signature() {}

#define AppendOffsetOverflow \
  "DescriptorTable(CBV(b0, offset = 4294967292), CBV(b1, numDescriptors = 7))"

// expected-error@+1 {{descriptor range offset overflows [4294967293, 4294967299]}}
[RootSignature(AppendOffsetOverflow)]
void append_offset_overflow_signature() {}

// expected-error@+1 {{descriptor range offset overflows [4294967292, 4294967296]}}
[RootSignature("DescriptorTable(CBV(b0, offset = 4294967292, numDescriptors = 5))")]
void offset_() {}
