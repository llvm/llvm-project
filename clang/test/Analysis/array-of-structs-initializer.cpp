// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -analyzer-checker=core,debug.ExprInspection -verify=expected,c -xc %s
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -analyzer-checker=core,debug.ExprInspection -verify=expected,cpp -xc++ -DCPP -std=c++14 %s

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

// Elided braces: see [dcl.init.aggr] p15.

const struct Outer nested_elided = {10, 20, 30, 40, 50};

void test_nested_elided_braces(void) {
  clang_analyzer_value(nested_elided.arr[0].x); // expected-warning {{10}}
  clang_analyzer_value(nested_elided.arr[0].y); // expected-warning {{20}}
  clang_analyzer_value(nested_elided.arr[1].x); // expected-warning {{30}}
  clang_analyzer_value(nested_elided.arr[1].y); // expected-warning {{40}}
  clang_analyzer_value(nested_elided.z);        // expected-warning {{50}}
}


const struct CStruct matrix[2][2] = {{{1}, {2}}, {{3}, {4}}};

void test_2d_array_of_structs(void) {
  clang_analyzer_value(matrix[0][0].a); // expected-warning {{1}}
  clang_analyzer_value(matrix[0][1].a); // expected-warning {{2}}
  clang_analyzer_value(matrix[1][0].a); // expected-warning {{3}}
  clang_analyzer_value(matrix[1][1].a); // expected-warning {{4}}
}

const struct CStruct matrix_elided[2][2] = {1, 2, 3, 4};

void test_matrix_elided_braces(void) {
  clang_analyzer_value(matrix_elided[0][0].a); // expected-warning {{1}}
  clang_analyzer_value(matrix_elided[0][1].a); // expected-warning {{2}}
  clang_analyzer_value(matrix_elided[1][0].a); // expected-warning {{3}}
  clang_analyzer_value(matrix_elided[1][1].a); // expected-warning {{4}}
}


const struct Inner partial_arr[3] = {{100, 200}};

void test_array_filler_zero_init(void) {
  clang_analyzer_value(partial_arr[0].x); // expected-warning {{100}}
  clang_analyzer_value(partial_arr[0].y); // expected-warning {{200}}
  clang_analyzer_value(partial_arr[1].x); // expected-warning {{0}}
  clang_analyzer_value(partial_arr[2].y); // expected-warning {{0}}
}

union IntOrChar {
  int i;
  char c;
};

const union IntOrChar u_int = {42};

void test_union_active_member(void) {
  clang_analyzer_value(u_int.i); // expected-warning {{42}}
}

void test_union_inactive_member(void) {
  clang_analyzer_value(u_int.c); // expected-warning {{8s:{ [-128, 127] }}}
}

const union IntOrChar u_char = {.c = 'x'};

void test_union_designated_active(void) {
  clang_analyzer_value(u_char.c); // expected-warning {{120}}
}

void test_union_designated_inactive(void) {
  clang_analyzer_value(u_char.i); // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
}

struct HasUnion {
  union IntOrChar u;
  int after;
};

const struct HasUnion hu = {{99}, 7};

void test_union_in_struct(void) {
  clang_analyzer_value(hu.u.i); // expected-warning {{99}}
  clang_analyzer_value(hu.u.c); // expected-warning {{8s:{ [-128, 127] }}}
  clang_analyzer_value(hu.after); // expected-warning {{7}}
}

// Empty initializer zero-initializes the first member.
const union IntOrChar u_empty = {};

void test_union_empty_init(void) {
  clang_analyzer_value(u_empty.i); // expected-warning {{0}}
  clang_analyzer_value(u_empty.c); // expected-warning {{8s:{ [-128, 127] }}}
}

const union IntOrChar u_arr[2] = {{10}, {.c = 'y'}};

void test_union_array(void) {
  clang_analyzer_value(u_arr[0].i); // expected-warning {{10}}
  clang_analyzer_value(u_arr[0].c); // expected-warning {{8s:{ [-128, 127] }}}
  clang_analyzer_value(u_arr[1].c); // expected-warning {{121}}
  clang_analyzer_value(u_arr[1].i); // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
}

struct Tagged {
  int tag;
  union {
    int ival;
    char cval;
  } payload;
};

const struct Tagged tagged_arr[2] = {
  {1, {.ival = 100}},
  {2, {.cval = 'Z'}},
};

void test_struct_union_array(void) {
  clang_analyzer_value(tagged_arr[0].tag);          // expected-warning {{1}}
  clang_analyzer_value(tagged_arr[0].payload.ival); // expected-warning {{100}}
  clang_analyzer_value(tagged_arr[0].payload.cval); // expected-warning {{8s:{ [-128, 127] }}}
  clang_analyzer_value(tagged_arr[1].tag);          // expected-warning {{2}}
  clang_analyzer_value(tagged_arr[1].payload.cval); // expected-warning {{90}}
  clang_analyzer_value(tagged_arr[1].payload.ival); // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
}
