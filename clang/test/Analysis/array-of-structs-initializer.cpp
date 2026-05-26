// RUN: %clang_analyze_cc1 -xc -analyzer-checker=core,debug.ExprInspection -verify=expected,c %s
// RUN: %clang_analyze_cc1 -xc++ -DCPP -std=c++14 -analyzer-checker=core,debug.ExprInspection -verify=expected,cpp %s

void clang_analyzer_value(int);

struct CStruct {
  int a;
};

struct CStruct nonconst_c_struct_array[1] = {
  {11},
};

void use_nonconst_struct_array_c(void) {
  clang_analyzer_value(nonconst_c_struct_array->a); // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
}

const struct CStruct const_c_struct_array[1] = { {22} };

void use_const_struct_array_c(void) {
  clang_analyzer_value(const_c_struct_array->a); // expected-warning {{22}}
}

#ifdef CPP
struct CPPStruct {
  int a = 33;
};

CPPStruct nonconst_cpp_struct_array[1] = {};
const CPPStruct const_cpp_struct_array[1] = {};

struct CPPStructWithUserCtor {
  int a = 44;
  CPPStructWithUserCtor(): a(55) {}
};

CPPStructWithUserCtor nonconst_cpp_struct_wctor_array[1] = {};

void use_nonconst_struct_array_cpp(void) {
  clang_analyzer_value(nonconst_cpp_struct_array->a); // cpp-warning {{32s:{ [-2147483648, 2147483647] }}}
}

const CPPStructWithUserCtor const_cpp_struct_wctor_array[1] = {};
#endif

int main(int argc, char **argv) {
  // FIXME: In C++ mode, IsMainAnalysis is false because global constructors
  // may run before main(), so the initializer for non-const globals are not
  // considered. In C mode this correctly resolves to 11.
  clang_analyzer_value(nonconst_c_struct_array->a); // c-warning {{11}} cpp-warning {{32s:{ [-2147483648, 2147483647] }}}

#ifdef CPP
  // Default member initialization is resolved from the initializer.
  clang_analyzer_value(const_cpp_struct_array->a); // cpp-warning {{33}}

  // FIXME: In C++ mode, non-const globals are not trusted because global
  // constructors may run before main(). This should be 33.
  clang_analyzer_value(nonconst_cpp_struct_array->a); // cpp-warning {{32s:{ [-2147483648, 2147483647] }}}

  // FIXME: We do not model constructor calls in initializers. This should
  // be 55 (from the constructor's initializer list).
  clang_analyzer_value(const_cpp_struct_wctor_array->a); // cpp-warning {{32s:{ [-2147483648, 2147483647] }}}

  // FIXME: Non-const global in C++ mode, and also requires modeling
  // constructor calls. This should be 55.
  clang_analyzer_value(nonconst_cpp_struct_wctor_array->a); // cpp-warning {{32s:{ [-2147483648, 2147483647] }}}
#endif
}

struct Inner {
  int x;
  int y;
};

struct Outer {
  struct Inner arr[2];
  int z;
};

const struct Outer nested = {{{10, 20}, {30, 40}}, 50};

void test_nested_struct_array_field(void) {
  clang_analyzer_value(nested.arr[0].x); // expected-warning {{10}}
  clang_analyzer_value(nested.arr[0].y); // expected-warning {{20}}
  clang_analyzer_value(nested.arr[1].x); // expected-warning {{30}}
  clang_analyzer_value(nested.arr[1].y); // expected-warning {{40}}
  clang_analyzer_value(nested.z);        // expected-warning {{50}}
}

const struct CStruct matrix[2][2] = {{{1}, {2}}, {{3}, {4}}};

void test_2d_array_of_structs(void) {
  clang_analyzer_value(matrix[0][0].a); // expected-warning {{1}}
  clang_analyzer_value(matrix[0][1].a); // expected-warning {{2}}
  clang_analyzer_value(matrix[1][0].a); // expected-warning {{3}}
  clang_analyzer_value(matrix[1][1].a); // expected-warning {{4}}
}

const struct Inner partial_arr[3] = {{100, 200}};

void test_array_filler_zero_init(void) {
  clang_analyzer_value(partial_arr[0].x); // expected-warning {{100}}
  clang_analyzer_value(partial_arr[0].y); // expected-warning {{200}}
  clang_analyzer_value(partial_arr[1].x); // expected-warning {{0}}
  clang_analyzer_value(partial_arr[2].y); // expected-warning {{0}}
}

