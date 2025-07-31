// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -Wno-unused-value -verify=valid %s -o -
// RUN: %clang_cc1 -triple spirv32 -verify=invalid -Wno-unused-value %s -cl-std=CL3.0 -x cl -o -
// RUN: %clang_cc1 -triple spirv64 -verify=invalid -Wno-unused-value %s -cl-std=CL3.0 -x cl -o -

typedef float float2 __attribute__((ext_vector_type(2)));

// valid-no-diagnostics

void call(float2 X, float2 Y) {
  __builtin_spirv_reflect(X, Y);
  // invalid-error@-1 {{builtin requires spirv target}}
}

// no error
float valid_builtin(float2 X) { return __builtin_spirv_length(X); }
