// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv64 -fsycl-is-device -verify %s -o -
// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv64 -verify %s -cl-std=CL3.0 -x cl -o -
// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv32 -verify %s -cl-std=CL3.0 -x cl -o -

typedef unsigned __attribute__((ext_vector_type(4))) int4;

void ballot(_Bool c) {
  int4 x;
  x = __builtin_spirv_subgroup_ballot(c);
  x = __builtin_spirv_subgroup_ballot(1);
  x = __builtin_spirv_subgroup_ballot(x); // expected-error{{parameter of incompatible type}}
  int y = __builtin_spirv_subgroup_ballot(c); // expected-error{{with an expression of incompatible type}}
}
