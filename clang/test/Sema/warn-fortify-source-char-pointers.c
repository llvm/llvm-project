// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_BUFFER -DPOINTEE=int -verify=expected,c -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DWRONG_BUFFER -DPOINTEE=int -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DWRONG_BUFFER -DPOINTEE=void -verify=expected,c -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DWRONG_BUFFER -DPOINTEE=void -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DPOINTEE=int -verify=expected,c -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DPOINTEE=int -verify -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -DPOINTEE=void -verify=expected,c -Werror
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -DPOINTEE=void -verify -Werror

typedef __SIZE_TYPE__ size_t;

#ifdef WRONG_BUFFER
typedef const char *path_t;
typedef POINTEE *buffer_t;
#else
typedef const POINTEE *path_t;
typedef char *buffer_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif
long readlink(path_t, buffer_t, size_t);
long readlinkat(int, path_t, buffer_t, size_t);
#ifdef WRONG_BUFFER
char *getcwd(buffer_t, size_t);
// c-error@-1 {{incompatible redeclaration of library function 'getcwd'}}
// c-note@-2 {{'getcwd' is a builtin with type 'char *(char *, __size_t)'}}
#endif
long read(int, void *, size_t);
long write(int, const void *, size_t);
#ifdef __cplusplus
}
#endif

// Unrelated pointer types in either the path or buffer must not trigger
// fortify diagnostics, even when the count exceeds the buffer size.
void call_unrelated(void) {
  char buf[4];
  readlink((path_t)0, (buffer_t)buf, 8);
  readlinkat(0, (path_t)0, (buffer_t)buf, 8);
#ifdef WRONG_BUFFER
  getcwd((buffer_t)buf, 8);
#endif
}

// The void-pointer I/O functions must still check non-character buffers.
void call_io(void) {
  int buf[1];
  read(0, buf, 8); // expected-error {{'read' size argument is too large; destination buffer has size 4, but size argument is 8}}
  write(0, buf, 8); // expected-error {{'write' will always read past the end of the source buffer; source buffer has size 4, but the size is 8}}
}
