// RUN: %clang_cl -Wpadded -Wno-msvc-not-found -fsyntax-only -- %s 2>&1 | FileCheck -check-prefix=WARN %s

struct Foo {
  int b : 1;
  char a;
};

int main () {Foo foo;}

// WARN: warning: padding struct 'Foo' with 31 bits to align 'a'
// WARN: warning: padding size of 'Foo' with 3 bytes to alignment boundary
