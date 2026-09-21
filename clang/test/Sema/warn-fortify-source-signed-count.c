// RUN: %clang_cc1 -triple i686-unknown-linux %s -verify=expected,signed32
// RUN: %clang_cc1 -triple i686-unknown-linux %s -fexperimental-new-constant-interpreter -verify=expected,signed32
// RUN: %clang_cc1 -triple i686-unknown-linux %s -DUNSIGNED_COUNT -verify=expected,unsigned32
// RUN: %clang_cc1 -triple i686-unknown-linux %s -DUNSIGNED_COUNT -fexperimental-new-constant-interpreter -verify=expected,unsigned32
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -verify=expected,signed64
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -fexperimental-new-constant-interpreter -verify=expected,signed64
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -DUNSIGNED_COUNT -verify=expected,unsigned64
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -DUNSIGNED_COUNT -fexperimental-new-constant-interpreter -verify=expected,unsigned64
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc %s -verify=expected,signed64
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc %s -fexperimental-new-constant-interpreter -verify=expected,signed64
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc %s -DUNSIGNED_COUNT -verify=expected,unsigned64
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc %s -DUNSIGNED_COUNT -fexperimental-new-constant-interpreter -verify=expected,unsigned64
// RUN: %clang_cc1 -triple i686-unknown-linux -x c++ %s -verify=expected,signed32
// RUN: %clang_cc1 -triple x86_64-unknown-linux -x c++ %s -verify=expected,signed64

// The count and return types of read/write need not match POSIX size_t and
// ssize_t. In particular, Windows read/write use unsigned int counts on LLP64.
#ifdef UNSIGNED_COUNT
typedef unsigned int count_t;
#else
typedef int count_t;
#endif
typedef __SIZE_TYPE__ size_t;
#ifdef __cplusplus
extern "C" {
#endif
int read(int, char *, count_t);
int write(int, const char *, count_t);
char *getcwd(char *, size_t);
#ifdef __cplusplus
}
#endif

void test_counts(count_t n) {
  char buf[4];
  read(0, buf, 4);
  write(0, buf, 4);
  read(0, buf, 8); // expected-warning {{'read' size argument is too large; destination buffer has size 4, but size argument is 8}}
  write(0, buf, 8); // expected-warning {{'write' will always read past the end of the source buffer; source buffer has size 4, but the size is 8}}
  read(0, buf, n);
  write(0, buf, n);
  read(0, buf, -1); // signed32-warning {{size argument is 4294967295}} signed64-warning {{size argument is 18446744073709551615}} unsigned32-warning {{size argument is 4294967295}} unsigned64-warning {{size argument is 4294967295}}
  write(0, buf, -1); // signed32-warning {{but the size is 4294967295}} signed64-warning {{but the size is 18446744073709551615}} unsigned32-warning {{but the size is 4294967295}} unsigned64-warning {{but the size is 4294967295}}
}

// Unlike read/write, getcwd has a complete builtin prototype using the
// target's size_t. Check that it is recognized on ILP32, LP64, and LLP64.
void test_getcwd(void) {
  char b[4];
  getcwd(b, sizeof(b));
  getcwd(b, 8); // expected-warning {{'getcwd' size argument is too large; destination buffer has size 4, but size argument is 8}}
}
