// RUN: %clang_cc1 -triple i686-unknown-linux %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -verify

// Verify that the fortify dispatch is tolerant of libc typedef choices for
// ssize_t. glibc uses `long` while other libcs (musl, bionic) may use a
// type that matches Clang's signed counterpart of size_t. On ILP32 these
// differ canonically (`long` vs `int`) even though both are 32-bit signed
// integers; the gate must accept either as ssize_t.

typedef __SIZE_TYPE__ size_t;
typedef long ssize_t;

ssize_t read(int, void *, size_t);

void test_read(void) {
  char b[4];
  read(0, b, 8); // expected-warning {{'read' size argument is too large; destination buffer has size 4, but size argument is 8}}
}
