// RUN: %clang_cc1 -triple i686-unknown-linux %s -verify=expected,signed32
// RUN: %clang_cc1 -triple i686-unknown-linux %s -DUNSIGNED_COUNT -verify=expected,unsigned32
// RUN: %clang_cc1 -triple i686-unknown-linux %s -DUNSIGNED_COUNT -fexperimental-new-constant-interpreter -verify=expected,unsigned32
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -verify=expected,signed64
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -fexperimental-new-constant-interpreter -verify=expected,signed64
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -DUNSIGNED_COUNT -verify=expected,unsigned64
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc %s -DUNSIGNED_COUNT -verify=expected,unsigned64
// RUN: %clang_cc1 -triple x86_64-unknown-linux -x c++ %s -verify=expected,signed64
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -DWARNING_GROUP -Wno-stringop-overread -Werror=fortify-source -verify=fortify
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -DWARNING_GROUP -Wno-fortify-source -Wstringop-overread -verify=disabled
// disabled-no-diagnostics

// Cover signed and unsigned counts on ILP32 and LP64, Windows' unsigned int
// counts on LLP64, and both evaluators without repeating every combination.

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
int pwrite(int, const void *, size_t, long);
int pwrite64(int, const void *, size_t, long long);
char *getcwd(char *, size_t);
#ifdef __cplusplus
}
#endif

#ifndef WARNING_GROUP
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
#else
// These overread diagnostics belong to -Wfortify-source, independently of
// whether -Wstringop-overread is enabled. C++ calls are covered in the main test.
void test_write(size_t n) {
  char buf[4];
  write(0, buf, 4);
  pwrite(0, buf, 4, 0);
  pwrite64(0, buf, 4, 0);
  write(0, buf, n);
  pwrite(0, buf, n, 0);
  pwrite64(0, buf, n, 0);
  write(0, buf, 8); // fortify-error {{'write' will always read past the end of the source buffer; source buffer has size 4, but the size is 8}}
  pwrite(0, buf, 8, 0); // fortify-error {{'pwrite' will always read past the end of the source buffer; source buffer has size 4, but the size is 8}}
  pwrite64(0, buf, 8, 0); // fortify-error {{'pwrite64' will always read past the end of the source buffer; source buffer has size 4, but the size is 8}}
}
#endif
