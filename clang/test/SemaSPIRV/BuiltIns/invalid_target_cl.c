// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -Wno-unused-value -verify=invalid %s -o -
// RUN: %clang_cc1 -triple spirv32 -verify=valid -Wno-unused-value %s -cl-std=CL3.0 -x cl -o -
// RUN: %clang_cc1 -triple spirv64 -verify=valid -Wno-unused-value %s -cl-std=CL3.0 -x cl -o -

typedef float float2 __attribute__((ext_vector_type(2)));

// valid-no-diagnostics

void invalid_builtin_for_target(int* p) {
  __builtin_spirv_generic_cast_to_ptr_explicit(p, 7);
  // invalid-error@-1 {{builtin requires spirv32 or spirv64 target}}
}

// no error
float valid_builtin(float2 X) { return __builtin_spirv_length(X); }
