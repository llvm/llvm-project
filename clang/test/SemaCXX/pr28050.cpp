// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++11 %s -fsyntax-only
//
// expected-no-diagnostics

class A {
public:
  A(char *s) {}
  A(A &&) = delete;
};

int main() { A a("OK"); }
