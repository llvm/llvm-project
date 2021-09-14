// RUN: %clang_cc1 -Wfortify-source -triple x86_64-apple-macosx10.14.0 %s -verify
// RUN: %clang_cc1 -Wfortify-source -xc++ -triple x86_64-apple-macosx10.14.0 %s -verify

typedef unsigned long size_t;

__attribute__((diagnose_as_builtin(__builtin_memcpy, 3, 1, 2))) int x; // expected-warning {{'diagnose_as_builtin' attribute only applies to functions}}

void *test_memcpy(const void *src, size_t c, void *dst) __attribute__((diagnose_as_builtin(__builtin_memcpy, 3, 1, 2))) {
  return __builtin_memcpy(dst, src, c);
}

void call_test_memcpy() {
  char bufferA[10];
  char bufferB[11];
  test_memcpy(bufferB, 10, bufferA);
  test_memcpy(bufferB, 11, bufferA); // expected-warning {{'memcpy' will always overflow; destination buffer has size 10, but size argument is 11}}
}

void failure_function_doesnt_exist() __attribute__((diagnose_as_builtin(__function_that_doesnt_exist))) {} // expected-error {{use of undeclared identifier '__function_that_doesnt_exist'}}

void not_a_builtin() {}

void failure_not_a_builtin() __attribute__((diagnose_as_builtin(not_a_builtin))) {} // expected-error {{'diagnose_as_builtin' attribute requires parameter 1 to be a builtin function}}

void failure_too_many_parameters(void *dst, const void *src, size_t count, size_t nothing) __attribute__((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3, 4))) {} // expected-error {{'diagnose_as_builtin' attribute references function '__builtin_memcpy', which takes exactly 3 arguments}}

void failure_too_few_parameters(void *dst, const void *src) __attribute__((diagnose_as_builtin(__builtin_memcpy, 1, 2))) {} // expected-error{{'diagnose_as_builtin' attribute references function '__builtin_memcpy', which takes exactly 3 arguments}}

void failure_not_an_integer(void *dst, const void *src, size_t count) __attribute__((diagnose_as_builtin(__builtin_memcpy, "abc", 2, 3))) {} // expected-error{{'diagnose_as_builtin' attribute requires parameter 2 to be an integer constant}}

void failure_not_a_builtin2() __attribute__((diagnose_as_builtin("abc"))) {} // expected-error{{'diagnose_as_builtin' attribute requires parameter 1 to be a builtin function}}

void failure_parameter_index_bounds(void *dst, const void *src) __attribute__((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3))) {} // expected-error{{'diagnose_as_builtin' attribute references parameter 3, but the function 'failure_parameter_index_bounds' has only 2 parameters}}

void failure_parameter_types(double dst, const void *src, size_t count) __attribute__((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3))) {} // expected-error{{'diagnose_as_builtin' attribute parameter types do not match: parameter 1 of function 'failure_parameter_types' has type 'double', but parameter 1 of function '__builtin_memcpy' has type 'void *'}}

void to_redeclare(void *dst, const void *src, size_t count);

void use_to_redeclare() {
  char src[10];
  char dst[9];
  // We shouldn't get an error yet.
  to_redeclare(dst, src, 10);
}

void to_redeclare(void *dst, const void *src, size_t count) __attribute((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3)));

void error_to_redeclare() {
  char src[10];
  char dst[9];
  // Now we get an error.
  to_redeclare(dst, src, 10); // expected-warning {{'memcpy' will always overflow; destination buffer has size 9, but size argument is 10}}
}

// Make sure this works even with extra qualifiers and the pass_object_size attribute.
void *memcpy2(void *const dst __attribute__((pass_object_size(0))), const void *src, size_t copy_amount) __attribute((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3)));

void call_memcpy2() {
  char buf1[10];
  char buf2[11];
  memcpy2(buf1, buf2, 11); // expected-warning {{'memcpy' will always overflow; destination buffer has size 10, but size argument is 11}}
}

// We require the types to be identical, modulo canonicalization and qualifiers.
// Maybe this could be relaxed if it proves too restrictive.
void failure_type(void *dest, char val, size_t n) __attribute__((diagnose_as_builtin(__builtin_memset, 1, 2, 3))) {} // expected-error {{'diagnose_as_builtin' attribute parameter types do not match: parameter 2 of function 'failure_type' has type 'char', but parameter 2 of function '__builtin_memset' has type 'int'}}

#ifdef __cplusplus
extern "C" {
#endif

extern int sscanf(const char *input, const char *format, ...);

#ifdef __cplusplus
}
#endif

// Variadics should work.
int mysscanf(const char *str, const char *format, ...) __attribute__((diagnose_as_builtin(sscanf, 1, 2)));

void warn_mysccanf() {
  char buf[10];
  mysscanf("", "%10s", buf); // expected-warning{{'sscanf' may overflow; destination buffer in argument 3 has size 10, but the corresponding specifier may require size 11}}
}

#ifdef __cplusplus

template <class T>
void some_templated_function(int x) {
  return;
}

void failure_template(int x) __attribute__((diagnose_as_builtin(some_templated_function, 1))) {} // expected-error{{'diagnose_as_builtin' attribute requires parameter 1 to be a builtin function}}

void failure_template_instantiation(int x) __attribute__((diagnose_as_builtin(some_templated_function<int>, 1))) {} // expected-error{{'diagnose_as_builtin' attribute requires parameter 1 to be a builtin function}}

void overloaded_function(unsigned);

void overloaded_function(int);

void failure_overloaded_function(int) __attribute__((diagnose_as_builtin(overloaded_function, 1))) {} // expected-error{{'diagnose_as_builtin' attribute requires parameter 1 to be a builtin function}}

struct S {
  __attribute__((diagnose_as_builtin(__builtin_memset))) void f(); // expected-error{{'diagnose_as_builtin' attribute cannot be applied to non-static member functions}}

  __attribute__((diagnose_as_builtin(__builtin_memcpy, 3, 1, 2))) static void *test_memcpy(const void *src, size_t c, void *dst) {
    return __builtin_memcpy(dst, src, c);
  }
};

void call_static_test_memcpy() {
  char bufferA[10];
  char bufferB[11];
  S::test_memcpy(bufferB, 10, bufferA);
  S::test_memcpy(bufferB, 11, bufferA); // expected-warning {{'memcpy' will always overflow; destination buffer has size 10, but size argument is 11}}
}

#endif
