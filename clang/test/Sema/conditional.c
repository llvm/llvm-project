// RUN: %clang_cc1 %s -fsyntax-only -verify

const char* test1 = 1 ? "i" : 1 == 1 ? "v" : "r";

void _efree(void *ptr);
void free(void *ptr);

int _php_stream_free1() {
  return (1 ? free(0) : _efree(0)); // expected-error {{returning 'void' from a function with incompatible result type 'int'}}
}

int _php_stream_free2() {
  return (1 ? _efree(0) : free(0));  // expected-error {{returning 'void' from a function with incompatible result type 'int'}}
}

void pr39809() {
  _Generic(0 ? (int const *)0 : (void *)0, int const *: (void)0);
  _Generic(0 ? (int const *)0 : (void *)1, void const *: (void)0);
  _Generic(0 ? (int volatile*)0 : (void const*)1, void volatile const*: (void)0);
  _Generic(0 ? (int volatile*)0 : (void const*)0, void volatile const*: (void)0);
}
