// RUN: %clang_cc1 -fsyntax-only -verify %s

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
  lang_type_class
};

void foo() {
  int i;
  char c;
  enum { red, green, blue } enum_obj;
  int *p;
  double d;
  _Complex double cc;
  extern void f();
  struct { int a; float b; } s_obj;
  union { int a; float b; } u_obj;
  int arr[10];

  int a1[__builtin_classify_type(f()) == void_type_class ? 1 : -1];
  int a2[__builtin_classify_type(i) == integer_type_class ? 1 : -1];
  int a3[__builtin_classify_type(c) == integer_type_class ? 1 : -1];
  int a4[__builtin_classify_type(enum_obj) == integer_type_class ? 1 : -1];
  int a5[__builtin_classify_type(p) == pointer_type_class ? 1 : -1];
  int a6[__builtin_classify_type(d) == real_type_class ? 1 : -1];
  int a7[__builtin_classify_type(cc) == complex_type_class ? 1 : -1];
  int a8[__builtin_classify_type(f) == pointer_type_class ? 1 : -1];
  int a0[__builtin_classify_type(s_obj) == record_type_class ? 1 : -1];
  int a10[__builtin_classify_type(u_obj) == union_type_class ? 1 : -1];
  int a11[__builtin_classify_type(arr) == pointer_type_class ? 1 : -1];
  int a12[__builtin_classify_type("abc") == pointer_type_class ? 1 : -1];
}

