// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -x hlsl -fsyntax-only -verify %s

// The entire uint32_t range is valid, including indices above INT32_MAX.
void largest_index(float V : USER4294967295) {}
void largest_index_with_zeroes(float V : USER000000000000000000004294967295) {}
void zero_index(float V : USER0000000000000000000000000000000000000000) {}

// Only trailing digits form the index.
void digits_in_name(float V : USER4294967296_NAME1) {}

void one_past_largest(float V : USER4294967296) {}
// expected-error@-1 {{semantic 'USER' index 4294967296 exceeds the maximum supported index 4294967295}}

void wraps_to_one(float V : USER4294967297) {}
// expected-error@-1 {{semantic 'USER' index 4294967297 exceeds the maximum supported index 4294967295}}

void overflow_with_zeroes(float V : USER0004294967296) {}
// expected-error@-1 {{semantic 'USER' index 0004294967296 exceeds the maximum supported index 4294967295}}

void largest_uint64(float V : USER18446744073709551615) {}
// expected-error@-1 {{semantic 'USER' index 18446744073709551615 exceeds the maximum supported index 4294967295}}

void larger_than_uint64(float V : USER18446744073709551616) {}
// expected-error@-1 {{semantic 'USER' index 18446744073709551616 exceeds the maximum supported index 4294967295}}

struct Input {
  float V : USER9999999999999999999999999999999999999999;
  // expected-error@-1 {{semantic 'USER' index 9999999999999999999999999999999999999999 exceeds the maximum supported index 4294967295}}
};

float return_semantic() : USER4294967296 { return 0; }
// expected-error@-1 {{semantic 'USER' index 4294967296 exceeds the maximum supported index 4294967295}}

#define OVERSIZED_SEMANTIC USER4294967296
void macro_semantic(float V : OVERSIZED_SEMANTIC) {}
// expected-error@-1 {{semantic 'USER' index 4294967296 exceeds the maximum supported index 4294967295}}

// Recover without losing the semantic or emitting a missing-semantic error.
[shader("compute")][numthreads(1, 1, 1)]
void compute_entry(int V : SV_GroupID4294967296) {}
// expected-error@-1 {{semantic 'SV_GroupID' index 4294967296 exceeds the maximum supported index 4294967295}}

[shader("pixel")]
float pixel_entry() : SV_Target4294967296 { return 0; }
// expected-error@-1 {{semantic 'SV_Target' index 4294967296 exceeds the maximum supported index 4294967295}}

// Parsing continues after an invalid annotation.
void valid_after_error(float V : USER1) {}
