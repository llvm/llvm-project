// RUN: %clang_cc1 -fsyntax-only -verify %s -fblocks
// RUN: %clang_cc1 -fsyntax-only -verify %s -fblocks -fexperimental-new-constant-interpreter

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

void foo(void) {
  int i;
  char c;
  enum { red, green, blue } enum_obj;
  int *p;
  double d;
  _Complex double cc;
  extern void f(void);
  struct { int a; float b; } s_obj;
  union { int a; float b; } u_obj;
  int arr[10];
  int (^block)(void);
  __attribute__((vector_size(16))) int vec;
  typedef __attribute__((ext_vector_type(4))) int evec_t;
  evec_t evec;
  typedef _BitInt(8) int8_t3 __attribute__((ext_vector_type(3)));
  int8_t3 t3;
  typedef _BitInt(16) int16_t3 __attribute__((ext_vector_type(4)));
  int16_t3 t4;
  typedef _BitInt(32) int32_t3 __attribute__((ext_vector_type(5)));
  int32_t3 t5;
  typedef _BitInt(64) int64_t3 __attribute__((ext_vector_type(6)));
  int64_t3 t6;
  typedef _BitInt(8) vint8_t3 __attribute__((vector_size(3)));
  vint8_t3 vt3;
  typedef _BitInt(16) vint16_t3 __attribute__((vector_size(4)));
  vint16_t3 vt4;
  typedef _BitInt(32) vint32_t3 __attribute__((vector_size(8)));
  vint32_t3 vt5;
  typedef _BitInt(64) vint64_t3 __attribute__((vector_size(16)));
  vint64_t3 vt6;
  _BitInt(16) bitint;

  _Atomic int atomic_i;
  _Atomic double atomic_d;
  _Complex int complex_i;
  _Complex double complex_d;

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
  int a13[__builtin_classify_type(block) == no_type_class ? 1 : -1];
  int a14[__builtin_classify_type(vec) == vector_type_class ? 1 : -1];
  int a15[__builtin_classify_type(evec) == vector_type_class ? 1 : -1];
  int a16[__builtin_classify_type(atomic_i) == integer_type_class ? 1 : -1];
  int a17[__builtin_classify_type(atomic_d) == real_type_class ? 1 : -1];
  int a18[__builtin_classify_type(complex_i) == complex_type_class ? 1 : -1];
  int a19[__builtin_classify_type(complex_d) == complex_type_class ? 1 : -1];
  int a20[__builtin_classify_type(bitint) == bitint_type_class ? 1 : -1];
}

extern int (^p)(void);
int n = __builtin_classify_type(p);
