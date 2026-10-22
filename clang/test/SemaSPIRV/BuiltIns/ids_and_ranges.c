// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv64 -fsycl-is-device -x c++ -verify=cxx %s -o -
// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv64 -verify=cl %s -cl-std=CL3.0 -x cl -o -
// RUN: %clang_cc1 -O1 -Wno-unused-value -triple spirv32 -verify=cl %s -cl-std=CL3.0 -x cl -o -

void test_num_workgroups(int* p) {
  __builtin_spirv_num_workgroups(0);
  __builtin_spirv_num_workgroups(p); // cxx-error{{cannot initialize a parameter of type 'int' with an lvalue of type 'int *'}} cl-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_num_workgroups(0, 0); // cxx-error{{too many arguments to function call, expected 1, have 2}} cl-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_num_workgroups(); // cxx-error{{too few arguments to function call, expected 1, have 0}} cl-error{{too few arguments to function call, expected 1, have 0}}
}

void test_workgroup_size(int* p) {
  __builtin_spirv_workgroup_size(0);
  __builtin_spirv_workgroup_size(p); // cxx-error{{cannot initialize a parameter of type 'int' with an lvalue of type 'int *'}} cl-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_workgroup_size(0, 0); // cxx-error{{too many arguments to function call, expected 1, have 2}} cl-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_workgroup_size(); // cxx-error{{too few arguments to function call, expected 1, have 0}} cl-error{{too few arguments to function call, expected 1, have 0}}
}

void test_workgroup_id(int* p) {
  __builtin_spirv_workgroup_id(0);
  __builtin_spirv_workgroup_id(p); // cxx-error{{cannot initialize a parameter of type 'int' with an lvalue of type 'int *'}} cl-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_workgroup_id(0, 0); // cxx-error{{too many arguments to function call, expected 1, have 2}} cl-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_workgroup_id(); // cxx-error{{too few arguments to function call, expected 1, have 0}} cl-error{{too few arguments to function call, expected 1, have 0}}
}

void test_local_invocation_id(int* p) {
  __builtin_spirv_local_invocation_id(0);
  __builtin_spirv_local_invocation_id(p); // cxx-error{{cannot initialize a parameter of type 'int' with an lvalue of type 'int *'}} cl-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_local_invocation_id(0, 0); // cxx-error{{too many arguments to function call, expected 1, have 2}} cl-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_local_invocation_id(); // cxx-error{{too few arguments to function call, expected 1, have 0}} cl-error{{too few arguments to function call, expected 1, have 0}}
}

void test_global_invocation_id(int* p) {
  __builtin_spirv_global_invocation_id(0);
  __builtin_spirv_global_invocation_id(p); // cxx-error{{cannot initialize a parameter of type 'int' with an lvalue of type 'int *'}} cl-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_global_invocation_id(0, 0); // cxx-error{{too many arguments to function call, expected 1, have 2}} cl-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_global_invocation_id(); // cxx-error{{too few arguments to function call, expected 1, have 0}} cl-error{{too few arguments to function call, expected 1, have 0}}
}

void test_global_size(int* p) {
  __builtin_spirv_global_size(0);
  __builtin_spirv_global_size(p); // cxx-error{{cannot initialize a parameter of type 'int' with an lvalue of type 'int *'}} cl-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_global_size(0, 0); // cxx-error{{too many arguments to function call, expected 1, have 2}} cl-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_global_size(); // cxx-error{{too few arguments to function call, expected 1, have 0}} cl-error{{too few arguments to function call, expected 1, have 0}}
}

void test_global_offset(int* p) {
  __builtin_spirv_global_offset(0);
  __builtin_spirv_global_offset(p); // cxx-error{{cannot initialize a parameter of type 'int' with an lvalue of type 'int *'}} cl-error{{incompatible pointer to integer conversion}}
  __builtin_spirv_global_offset(0, 0); // cxx-error{{too many arguments to function call, expected 1, have 2}} cl-error{{too many arguments to function call, expected 1, have 2}}
  __builtin_spirv_global_offset(); // cxx-error{{too few arguments to function call, expected 1, have 0}} cl-error{{too few arguments to function call, expected 1, have 0}}
}

void test_subgroup_size() {
  __builtin_spirv_subgroup_size();
  __builtin_spirv_subgroup_size(0); // cxx-error{{too many arguments to function call, expected 0, have 1}} cl-error{{too many arguments to function call, expected 0, have 1}}
}

void test_subgroup_max_size() {
  __builtin_spirv_subgroup_max_size();
  __builtin_spirv_subgroup_max_size(0); // cxx-error{{too many arguments to function call, expected 0, have 1}} cl-error{{too many arguments to function call, expected 0, have 1}}
}

void test_num_subgroups() {
  __builtin_spirv_num_subgroups();
  __builtin_spirv_num_subgroups(0); // cxx-error{{too many arguments to function call, expected 0, have 1}} cl-error{{too many arguments to function call, expected 0, have 1}}
}

void test_subgroup_id() {
  __builtin_spirv_subgroup_id();
  __builtin_spirv_subgroup_id(0); // cxx-error{{too many arguments to function call, expected 0, have 1}} cl-error{{too many arguments to function call, expected 0, have 1}}
}

void test_subgroup_local_invocation_id() {
  __builtin_spirv_subgroup_local_invocation_id();
  __builtin_spirv_subgroup_local_invocation_id(0); // cxx-error{{too many arguments to function call, expected 0, have 1}} cl-error{{too many arguments to function call, expected 0, have 1}}
}
