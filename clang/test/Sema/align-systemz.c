// RUN: %clang_cc1 -triple s390x-linux-gnu -fsyntax-only -verify %s
// expected-no-diagnostics

// SystemZ prefers to align all global variables to two bytes,
// but this should *not* be reflected in the ABI alignment as
// retrieved via __alignof__.

struct test {
  signed char a;
};

char c;
struct test s;

int chk1[__alignof__(c) == 1 ? 1 : -1];
int chk2[__alignof__(s) == 1 ? 1 : -1];

