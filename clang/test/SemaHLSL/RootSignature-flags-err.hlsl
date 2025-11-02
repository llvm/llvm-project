// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only \
// RUN:   -fdx-rootsignature-version=rootsig_1_0 %s -verify=v10
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only \
// RUN:   -fdx-rootsignature-version=rootsig_1_1 %s -verify=v11
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only \
// RUN:   -fdx-rootsignature-version=rootsig_1_2 %s -verify=v12
// Root Descriptor Flags:

// v10-error@+1 {{invalid flags for version 1.0}}
[RootSignature("CBV(b0, flags = DATA_STATIC)")]
void bad_root_descriptor_flags_0() {}

// v10-error@+1 {{invalid flags for version 1.0}}
[RootSignature("CBV(b0, flags = DATA_STATIC_WHILE_SET_AT_EXECUTE)")]
void bad_root_descriptor_flags_1() {}

// v10-error@+3 {{invalid flags for version 1.0}}
// v11-error@+2 {{invalid flags for version 1.1}}
// v12-error@+1 {{invalid flags for version 1.2}}
[RootSignature("CBV(b0, flags = DATA_STATIC | DATA_VOLATILE)")]
void bad_root_descriptor_flags_2() {}

// Descriptor Range Flags:

// v10-error@+1 {{invalid flags for version 1.0}}
[RootSignature("DescriptorTable(CBV(b0, flags = DATA_VOLATILE))")]
void bad_descriptor_range_flags_0() {}

// v10-error@+1 {{invalid flags for version 1.0}}
[RootSignature("DescriptorTable(CBV(b0, flags = DATA_STATIC))")]
void bad_descriptor_range_flags_1() {}

// v10-error@+1 {{invalid flags for version 1.0}}
[RootSignature("DescriptorTable(CBV(b0, flags = DATA_STATIC_WHILE_SET_AT_EXECUTE | DESCRIPTORS_VOLATILE))")]
void bad_descriptor_range_flags_2() {}

// v10-error@+1 {{invalid flags for version 1.0}}
[RootSignature("DescriptorTable(CBV(b0, flags = DESCRIPTORS_VOLATILE))")]
void bad_descriptor_range_flags_3() {}

// v10-error@+1 {{invalid flags for version 1.0}}
[RootSignature("DescriptorTable(CBV(b0, flags = DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS))")]
void bad_descriptor_range_flags_4() {}

// v10-error@+3 {{invalid flags for version 1.0}}
// v11-error@+2 {{invalid flags for version 1.1}}
// v12-error@+1 {{invalid flags for version 1.2}}
[RootSignature("DescriptorTable(CBV(b0, flags = DATA_STATIC | DATA_STATIC_WHILE_SET_AT_EXECUTE))")]
void bad_descriptor_range_flags_5() {}

// v10-error@+3 {{invalid flags for version 1.0}}
// v11-error@+2 {{invalid flags for version 1.1}}
// v12-error@+1 {{invalid flags for version 1.2}}
[RootSignature("DescriptorTable(CBV(b0, flags = DESCRIPTORS_VOLATILE | DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS))")]
void bad_descriptor_range_flags_6() {}

// v10-error@+3 {{invalid flags for version 1.0}}
// v11-error@+2 {{invalid flags for version 1.1}}
// v12-error@+1 {{invalid flags for version 1.2}}
[RootSignature("DescriptorTable(CBV(b0, flags = DESCRIPTORS_VOLATILE | DATA_STATIC))")]
void bad_descriptor_range_flags_7() {}
