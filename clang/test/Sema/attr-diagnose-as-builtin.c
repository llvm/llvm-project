// RUN: %clang_cc1 -Wfortify-source -triple x86_64-apple-macosx10.14.0 %s -verify
// RUN: %clang_cc1 -Wfortify-source -xc++ -triple x86_64-apple-macosx10.14.0 %s -verify

typedef unsigned long size_t;

__attribute__((diagnose_as_builtin(__builtin_memcpy, 3, 1, 2))) int x; // expected-warning {{'diagnose_as_builtin' attribute only applies to functions}}

void *test_memcpy(const void *src, size_t c, void *dst) __attribute__((diagnose_as_builtin(__builtin_memcpy, 3, 1, 2))) {
  return __builtin_memcpy(dst, src, c);
}

void call_test_memcpy(void) {
  char bufferA[10];
  char bufferB[11];
  test_memcpy(bufferB, 10, bufferA);
  test_memcpy(bufferB, 11, bufferA); // expected-warning {{'memcpy' will always overflow; destination buffer has size 10, but size argument is 11}}
}

void failure_function_doesnt_exist(void) __attribute__((diagnose_as_builtin(__function_that_doesnt_exist))) {} // expected-error {{use of undeclared identifier '__function_that_doesnt_exist'}}

void not_a_builtin(void) {}

void failure_not_a_builtin(void) __attribute__((diagnose_as_builtin(not_a_builtin))) {} // expected-error {{'diagnose_as_builtin' attribute requires parameter 1 to be a builtin function}}

void failure_too_many_parameters(void *dst, const void *src, size_t count, size_t nothing) __attribute__((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3, 4))) {} // expected-error {{'diagnose_as_builtin' attribute references function '__builtin_memcpy', which takes exactly 3 arguments}}

void failure_too_few_parameters(void *dst, const void *src) __attribute__((diagnose_as_builtin(__builtin_memcpy, 1, 2))) {} // expected-error{{'diagnose_as_builtin' attribute references function '__builtin_memcpy', which takes exactly 3 arguments}}

void failure_not_an_integer(void *dst, const void *src, size_t count) __attribute__((diagnose_as_builtin(__builtin_memcpy, "abc", 2, 3))) {} // expected-error{{'diagnose_as_builtin' attribute requires parameter 2 to be an integer constant}}

void failure_not_a_builtin2(void) __attribute__((diagnose_as_builtin("abc"))) {} // expected-error{{'diagnose_as_builtin' attribute requires parameter 1 to be a builtin function}}

void failure_parameter_index_bounds(void *dst, const void *src) __attribute__((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3))) {} // expected-error{{'diagnose_as_builtin' attribute references parameter 3, but the function 'failure_parameter_index_bounds' has only 2 parameters}}

void failure_parameter_types(double dst, const void *src, size_t count) __attribute__((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3))) {} // expected-error{{'diagnose_as_builtin' attribute parameter types do not match: parameter 1 of function 'failure_parameter_types' has type 'double', but parameter 1 of function '__builtin_memcpy' has type 'void *'}}

void to_redeclare(void *dst, const void *src, size_t count);

void use_to_redeclare(void) {
  char src[10];
  char dst[9];
  // We shouldn't get an error yet.
  to_redeclare(dst, src, 10);
}

void to_redeclare(void *dst, const void *src, size_t count) __attribute((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3)));

void error_to_redeclare(void) {
  char src[10];
  char dst[9];
  // Now we get an error.
  to_redeclare(dst, src, 10); // expected-warning {{'memcpy' will always overflow; destination buffer has size 9, but size argument is 10}}
}

// Make sure this works even with extra qualifiers and the pass_object_size attribute.
void *memcpy2(void *const dst __attribute__((pass_object_size(0))), const void *src, size_t copy_amount) __attribute((diagnose_as_builtin(__builtin_memcpy, 1, 2, 3)));

void call_memcpy2(void) {
  char buf1[10];
  char buf2[11];
  memcpy2(buf1, buf2, 11); // expected-warning {{'memcpy' will always overflow; destination buffer has size 10, but size argument is 11}}
}

// We require the types to be identical, modulo canonicalization and qualifiers.
// Maybe this could be relaxed if it proves too restrictive.
void failure_type(void *dest, char val, size_t n) __attribute__((diagnose_as_builtin(__builtin_memset, 1, 2, 3))) {} // expected-error {{'diagnose_as_builtin' attribute parameter types do not match: parameter 2 of function 'failure_type' has type 'char', but parameter 2 of function '__builtin_memset' has type 'int'}}

typedef struct _IO_FILE FILE;

#ifdef __cplusplus
extern "C" {
#endif

extern int sscanf(const char *input, const char *format, ...);
extern void *memccpy(void *dst, const void *src, int c, size_t n);
extern void *memset_explicit(void *dst, int c, size_t n);
extern void explicit_bzero(void *dst, size_t n);
extern size_t fread_unlocked(void *ptr, size_t size, size_t nmemb, FILE *stream);
extern size_t fwrite_unlocked(const void *ptr, size_t size, size_t nmemb, FILE *stream);
extern char *fgets_unlocked(char *s, int size, FILE *stream);
extern int getlogin_r(char *name, size_t namesize);
extern int ttyname_r(int fd, char *buf, size_t buflen);
extern int ptsname_r(int fd, char *buf, size_t buflen);
extern size_t confstr(int name, char *buf, size_t len);

#ifdef __cplusplus
}
#endif

// Variadics should work.
int mysscanf(const char *str, const char *format, ...) __attribute__((diagnose_as_builtin(sscanf, 1, 2)));

void warn_mysccanf(void) {
  char buf[10];
  mysscanf("", "%10s", buf); // expected-warning{{'sscanf' may overflow; destination buffer in argument 3 has size 10, but the corresponding specifier may require size 11}}
}

int my_vsprintf(const char *format, char *str, __builtin_va_list ap) __attribute__((diagnose_as_builtin(__builtin_vsprintf, 2, 1, 3)));

void warn_my_vsprintf(void) {
  char buf[6];
  __builtin_va_list list;
  my_vsprintf("hello", buf, list);
  my_vsprintf("hello!", buf, list); // expected-warning{{'vsprintf' will always overflow; destination buffer has size 6, but format string expands to at least 7}}
}

void *my_memccpy(int extra, void *dst, const void *src, int c, size_t n)
    __attribute__((diagnose_as_builtin(memccpy, 2, 3, 4, 5)));
void *my_memset_explicit(int extra, void *dst, int c, size_t n)
    __attribute__((diagnose_as_builtin(memset_explicit, 2, 3, 4)));
void my_explicit_bzero(int extra, void *dst, size_t n)
    __attribute__((diagnose_as_builtin(explicit_bzero, 2, 3)));

void warn_my_mem_zero_and_memccpy(void) {
  char src[20], dst[10];
  my_memccpy(0, dst, src, '\0', 10);
  my_memccpy(0, dst, src, '\0', 11); // expected-warning{{'memccpy' size argument is too large; destination buffer has size 10, but size argument is 11}}
  my_memset_explicit(0, dst, 0, 10);
  my_memset_explicit(0, dst, 0, 11); // expected-warning{{'memset_explicit' will always overflow; destination buffer has size 10, but size argument is 11}}
  my_explicit_bzero(0, dst, 10);
  my_explicit_bzero(0, dst, 11); // expected-warning{{'explicit_bzero' will always overflow; destination buffer has size 10, but size argument is 11}}
}

size_t my_fread_unlocked(int extra, void *ptr, size_t size, size_t nmemb, FILE *fp)
    __attribute__((diagnose_as_builtin(fread_unlocked, 2, 3, 4, 5)));
size_t my_fwrite_unlocked(int extra, const void *ptr, size_t size, size_t nmemb, FILE *fp)
    __attribute__((diagnose_as_builtin(fwrite_unlocked, 2, 3, 4, 5)));
char *my_fgets_unlocked(int extra, char *s, int size, FILE *fp)
    __attribute__((diagnose_as_builtin(fgets_unlocked, 2, 3, 4)));

void warn_my_stdio_unlocked(FILE *fp) {
  char buf[4];
  my_fread_unlocked(0, buf, 2, 2, fp);
  my_fread_unlocked(0, buf, 2, 3, fp); // expected-warning{{'fread_unlocked' will always overflow; destination buffer has size 4, but size argument is 6}}
  my_fwrite_unlocked(0, buf, 2, 2, fp);
  my_fwrite_unlocked(0, buf, 2, 3, fp); // expected-warning{{'fwrite_unlocked' will always read past the end of the source buffer; source buffer has size 4, but the size is 6}}
  my_fgets_unlocked(0, buf, 4, fp);
  my_fgets_unlocked(0, buf, 5, fp); // expected-warning{{'fgets_unlocked' size argument is too large; destination buffer has size 4, but size argument is 5}}
  my_fgets_unlocked(0, buf, -1, fp); // expected-warning{{'fgets_unlocked' size argument is negative}}
}

int my_getlogin_r(int extra, char *name, size_t namesize)
    __attribute__((diagnose_as_builtin(getlogin_r, 2, 3)));
int my_ttyname_r(int extra, int fd, char *buf, size_t buflen)
    __attribute__((diagnose_as_builtin(ttyname_r, 2, 3, 4)));
int my_ptsname_r(int extra, int fd, char *buf, size_t buflen)
    __attribute__((diagnose_as_builtin(ptsname_r, 2, 3, 4)));
size_t my_confstr(int extra, int name, char *buf, size_t len)
    __attribute__((diagnose_as_builtin(confstr, 2, 3, 4)));

void warn_my_posix_getters(int fd, int name) {
  char buf[10];
  my_getlogin_r(0, buf, 10);
  my_getlogin_r(0, buf, 11); // expected-warning{{'getlogin_r' size argument is too large; destination buffer has size 10, but size argument is 11}}
  my_ttyname_r(0, fd, buf, 10);
  my_ttyname_r(0, fd, buf, 11); // expected-warning{{'ttyname_r' size argument is too large; destination buffer has size 10, but size argument is 11}}
  my_ptsname_r(0, fd, buf, 10);
  my_ptsname_r(0, fd, buf, 11); // expected-warning{{'ptsname_r' size argument is too large; destination buffer has size 10, but size argument is 11}}
  my_confstr(0, name, buf, 10);
  my_confstr(0, name, buf, 11); // expected-warning{{'confstr' size argument is too large; destination buffer has size 10, but size argument is 11}}
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

void call_static_test_memcpy(void) {
  char bufferA[10];
  char bufferB[11];
  S::test_memcpy(bufferB, 10, bufferA);
  S::test_memcpy(bufferB, 11, bufferA); // expected-warning {{'memcpy' will always overflow; destination buffer has size 10, but size argument is 11}}
}

#endif
