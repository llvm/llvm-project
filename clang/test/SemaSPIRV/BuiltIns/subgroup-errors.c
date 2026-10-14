// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv64 -fsycl-is-device -x c++ -verify=cxx %s -o -
// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv64 -verify=cl %s -cl-std=CL3.0 -x cl -o -
// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv32 -verify=cl %s -cl-std=CL3.0 -x cl -o -

typedef unsigned __attribute__((ext_vector_type(4))) int4;

void ballot(bool c) {
  int4 x;
  x = __builtin_spirv_subgroup_ballot(c);
  x = __builtin_spirv_subgroup_ballot(1);
  x = __builtin_spirv_subgroup_ballot(x); // cxx-error{{cannot initialize a parameter of type 'bool' with an lvalue of type 'int4'}} cl-error{{parameter of incompatible type}}
  int y = __builtin_spirv_subgroup_ballot(c); // cxx-error{{cannot initialize a variable of type 'int' with an rvalue of type}} cl-error{{with an expression of incompatible type}}
}

void shuffle() {
  int x = 0;
  float f = 0;
  int [[clang::ext_vector_type(1)]] v;
  (void)__builtin_spirv_subgroup_shuffle(x, x);
  (void)__builtin_spirv_subgroup_shuffle(f, f);
  (void)__builtin_spirv_subgroup_shuffle(x, x, x); // cxx-error{{too many arguments to function call, expected 2, have 3}} cl-error{{too many arguments to function call, expected 2, have 3}}
  (void)__builtin_spirv_subgroup_shuffle(v, f); // cxx-error{{1st argument must be a scalar type}} cl-error{{1st argument must be a scalar type}}
  (void)__builtin_spirv_subgroup_shuffle(f, v); // cxx-error{{to parameter of incompatible type}} cl-error{{to parameter of incompatible type}}
}
