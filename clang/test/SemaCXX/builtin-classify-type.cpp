// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s
// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s -fexperimental-new-constant-interpreter

// expected-no-diagnostics

enum gcc_type_class {
  no_type_class = -1,
  void_type_class, integer_type_class, char_type_class,
  enumeral_type_class, boolean_type_class,
  pointer_type_class, reference_type_class, offset_type_class,
  real_type_class, complex_type_class,
  function_type_class, method_type_class,
  record_type_class, union_type_class,
  array_type_class, string_type_class,
  lang_type_class, opaque_type_class,
  bitint_type_class, vector_type_class
};

class cl {
public:
    void bar() {}
    int baz;
};

int builtin_result;

void foo() {
  int i;
  char c;
  enum { red, green, blue} enum_obj;
  bool b;
  int *p;
  int &r = i;
  double d;
  extern void f();
  cl cl_obj;
  union { int a; float b; } u_obj;
  int arr[10];
  int (^block)();
  __attribute__((vector_size(16))) int vec;
  typedef __attribute__((ext_vector_type(4))) int evec_t;
  evec_t evec;
  _Atomic int atomic_i;
  _Atomic double atomic_d;
  _Complex int complex_i;
  _Complex double complex_d;
  _BitInt(32) bitint;

  int a1[__builtin_classify_type(f()) == void_type_class ? 1 : -1];
  int a2[__builtin_classify_type(i) == integer_type_class ? 1 : -1];
  int a3[__builtin_classify_type(c) == integer_type_class ? 1 : -1];
  int a4[__builtin_classify_type(enum_obj) == enumeral_type_class ? 1 : -1];
  int a5[__builtin_classify_type(b) == boolean_type_class ? 1 : -1];
  int a6[__builtin_classify_type(p) == pointer_type_class ? 1 : -1];
  int a7[__builtin_classify_type(r) == integer_type_class ? 1 : -1];
  int a8[__builtin_classify_type(&cl::baz) == offset_type_class ? 1 : -1];
  int a9[__builtin_classify_type(d) == real_type_class ? 1 : -1];
  int a10[__builtin_classify_type(f) == pointer_type_class ? 1 : -1];
  int a11[__builtin_classify_type(&cl::bar) == record_type_class ? 1 : -1];
  int a12[__builtin_classify_type(cl_obj) == record_type_class ? 1 : -1];
  int a13[__builtin_classify_type(u_obj) == union_type_class ? 1 : -1];
  int a14[__builtin_classify_type(arr) == pointer_type_class ? 1 : -1];
  int a15[__builtin_classify_type("abc") == pointer_type_class ? 1 : -1];
  int a16[__builtin_classify_type(block) == no_type_class ? 1 : -1];
  int a17[__builtin_classify_type(vec) == vector_type_class ? 1 : -1];
  int a18[__builtin_classify_type(evec) == vector_type_class ? 1 : -1];
  int a19[__builtin_classify_type(atomic_i) == integer_type_class ? 1 : -1];
  int a20[__builtin_classify_type(atomic_d) == real_type_class ? 1 : -1];
  int a21[__builtin_classify_type(complex_i) == complex_type_class ? 1 : -1];
  int a22[__builtin_classify_type(complex_d) == complex_type_class ? 1 : -1];
  int a23[__builtin_classify_type(bitint) == bitint_type_class ? 1 : -1];
}

