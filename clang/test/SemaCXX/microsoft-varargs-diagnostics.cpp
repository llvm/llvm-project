// RUN: %clang_cc1 -triple thumbv7-windows -fms-compatibility -fsyntax-only %s -verify

extern "C" {
typedef char * va_list;
}

void test_no_arguments(int i, ...) {
  __va_start(); // expected-error{{too few arguments to function call, expected at least 3, have 0}}
}

void test_one_argument(int i, ...) {
  va_list ap;
  __va_start(&ap); // expected-error{{too few arguments to function call, expected at least 3, have 1}}
}

void test_two_arguments(int i, ...) {
  va_list ap;
  __va_start(&ap, &i); // expected-error{{too few arguments to function call, expected at least 3, have 2}}
}

void test_non_last_argument(int i, int j, ...) {
  va_list ap;
  __va_start(&ap, &i, 4);
  // expected-error@-1{{passing 'int *' to parameter of incompatible type 'const char *': type mismatch at 2nd parameter ('int *' vs 'const char *')}}
  // expected-error@-2{{passing 'int' to parameter of incompatible type 'unsigned int': type mismatch at 3rd parameter ('int' vs 'unsigned int')}}
}

void test_stack_allocated(int i, ...) {
  va_list ap;
  int j;
  __va_start(&ap, &j, 4);
  // expected-error@-1{{passing 'int *' to parameter of incompatible type 'const char *': type mismatch at 2nd parameter ('int *' vs 'const char *')}}
  // expected-error@-2{{passing 'int' to parameter of incompatible type 'unsigned int': type mismatch at 3rd parameter ('int' vs 'unsigned int')}}
}

void test_non_pointer_addressof(int i, ...) {
  va_list ap;
  __va_start(&ap, 1, 4);
  // expected-error@-1{{passing 'int' to parameter of incompatible type 'const char *': type mismatch at 2nd parameter ('int' vs 'const char *')}}
  // expected-error@-2{{passing 'int' to parameter of incompatible type 'unsigned int': type mismatch at 3rd parameter ('int' vs 'unsigned int')}}
}

