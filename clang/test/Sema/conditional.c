// RUN: %clang_cc1 %s -fsyntax-only -fenable-matrix -verify

const char* test1 = 1 ? "i" : 1 == 1 ? "v" : "r";

void _efree(void *ptr);
void free(void *ptr);

int _php_stream_free1(void) {
  return (1 ? free(0) : _efree(0)); // expected-error {{returning 'void' from a function with incompatible result type 'int'}}
}

int _php_stream_free2(void) {
  return (1 ? _efree(0) : free(0));  // expected-error {{returning 'void' from a function with incompatible result type 'int'}}
}

void pr39809(void) {
  _Generic(0 ? (int const *)0 : (void *)0, int const *: (void)0);
  _Generic(0 ? (int const *)0 : (void *)1, void const *: (void)0);
  _Generic(0 ? (int volatile*)0 : (void const*)1, void volatile const*: (void)0);
  _Generic(0 ? (int volatile*)0 : (void const*)0, void volatile const*: (void)0);
}

// Ensure we compute the correct common type for extension types as well.
void GH69008(void) {
  typedef float mat4 __attribute((matrix_type(4, 4)));
  typedef float mat5 __attribute((matrix_type(5, 5)));

  mat4 transform;
  (void)(1 ? transform : transform); // ok

  mat5 other_transform;
  (void)(1 ? other_transform : transform); // expected-error {{incompatible operand types ('mat5' (aka 'float __attribute__((matrix_type(5, 5)))') and 'mat4' (aka 'float __attribute__((matrix_type(4, 4)))'))}}
}
