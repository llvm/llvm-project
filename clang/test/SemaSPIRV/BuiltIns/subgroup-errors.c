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

void shuffle() {
  int x = 0;
  long long l = 0;
  float f = 0;
  int [[clang::ext_vector_type(1)]] v;
  (void)__builtin_spirv_subgroup_shuffle(x, x);
  (void)__builtin_spirv_subgroup_shuffle(f, f);
  (void)__builtin_spirv_subgroup_shuffle(x, x, x); // expected-error{{too many arguments to function call, expected 2, have 3}}
  (void)__builtin_spirv_subgroup_shuffle(v, f); // expected-error{{1st argument must be a scalar type}}
  (void)__builtin_spirv_subgroup_shuffle(f, v); // expected-error{{to parameter of incompatible type}}
}
