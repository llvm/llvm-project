// RUN: %clang_cc1 -fsyntax-only -fdelayed-template-parsing -Wunused-private-field -Wused-but-marked-unused -Wno-uninitialized -verify -std=c++11 %s
// expected-no-diagnostics

class EverythingMayBeUsed {
  int x;
public:
  template <class T>
  void f() {
    x = 0;
  }
};
