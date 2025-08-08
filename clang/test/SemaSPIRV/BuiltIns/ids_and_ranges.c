// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv64 -fsycl-is-device -verify %s -o -
// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv64 -verify %s -cl-std=CL3.0 -x cl -o -
// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv32 -verify %s -cl-std=CL3.0 -x cl -o -

void test_num_workgroups(int* p) {
  __builtin_spirv_num_workgroups(0);
  __builtin_spirv_num_workgroups(p); // expected-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_num_workgroups(0, 0); // expected-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_num_workgroups(); // expected-error{{too few arguments to function call, expected 1, have 0}}
}

void test_workgroup_size(int* p) {
  __builtin_spirv_workgroup_size(0);
  __builtin_spirv_workgroup_size(p); // expected-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_workgroup_size(0, 0); // expected-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_workgroup_size(); // expected-error{{too few arguments to function call, expected 1, have 0}}
}

void test_workgroup_id(int* p) {
  __builtin_spirv_workgroup_id(0);
  __builtin_spirv_workgroup_id(p); // expected-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_workgroup_id(0, 0); // expected-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_workgroup_id(); // expected-error{{too few arguments to function call, expected 1, have 0}}
}

void test_local_invocation_id(int* p) {
  __builtin_spirv_local_invocation_id(0);
  __builtin_spirv_local_invocation_id(p); // expected-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_local_invocation_id(0, 0); // expected-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_local_invocation_id(); // expected-error{{too few arguments to function call, expected 1, have 0}}
}

void test_global_invocation_id(int* p) {
  __builtin_spirv_global_invocation_id(0);
  __builtin_spirv_global_invocation_id(p); // expected-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_global_invocation_id(0, 0); // expected-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_global_invocation_id(); // expected-error{{too few arguments to function call, expected 1, have 0}}
}

void test_global_size(int* p) {
  __builtin_spirv_global_size(0);
  __builtin_spirv_global_size(p); // expected-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_global_size(0, 0); // expected-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_global_size(); // expected-error{{too few arguments to function call, expected 1, have 0}}
}

void test_global_offset(int* p) {
  __builtin_spirv_global_offset(0);
  __builtin_spirv_global_offset(p); // expected-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_global_offset(0, 0); // expected-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_global_offset(); // expected-error{{too few arguments to function call, expected 1, have 0}}
}

void test_subgroup_size() {
  __builtin_spirv_subgroup_size();
  __builtin_spirv_subgroup_size(0); // expected-error{{too many arguments to function call, expected 0, have 1}}
}

void test_subgroup_max_size() {
  __builtin_spirv_subgroup_max_size();
  __builtin_spirv_subgroup_max_size(0); // expected-error{{too many arguments to function call, expected 0, have 1}}
}

void test_num_subgroups() {
  __builtin_spirv_num_subgroups();
  __builtin_spirv_num_subgroups(0); // expected-error{{too many arguments to function call, expected 0, have 1}}
}

void test_subgroup_id() {
  __builtin_spirv_subgroup_id();
  __builtin_spirv_subgroup_id(0); // expected-error{{too many arguments to function call, expected 0, have 1}}
}

void test_subgroup_local_invocation_id() {
  __builtin_spirv_subgroup_local_invocation_id();
  __builtin_spirv_subgroup_local_invocation_id(0); // expected-error{{too many arguments to function call, expected 0, have 1}}
}
