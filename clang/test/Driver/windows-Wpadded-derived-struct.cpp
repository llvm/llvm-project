// RUN: %clang_cl -Wpadded -Wno-msvc-not-found -fsyntax-only -- %s 2>&1 | FileCheck -check-prefix=WARN %s
// RUN: %clang -Wpadded -fsyntax-only -- %s 2>&1 | FileCheck -check-prefix=WARN %s

struct Base {
  int b;
};

struct Derived : public Base {
  char c;
};

int main() {Derived d;}
// WARN: warning: padding size of 'Derived' with 3 bytes to alignment boundary
