// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify -Werror

// A read() whose count parameter is a signed 'int' rather than POSIX size_t
// must not be treated as the libc read(): the size_t count slot is matched
// exactly, so this lookalike is not diagnosed.

typedef unsigned long size_t;
typedef long ssize_t;

// expected-no-diagnostics

ssize_t read(int fd, void *buf, int count);

void test_read_signed_count(void) {
  char b[4];
  read(0, b, 8);
}
