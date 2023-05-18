// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -verify -fms-compatibility -Wno-microsoft -Wmicrosoft-template-shadow

template <typename T> // expected-note {{template parameter is declared here}}
struct Outmost {
  template <typename T> // expected-warning {{declaration of 'T' shadows template parameter}}
  struct Inner {
    void f() {
      T *var;
    }
  };
};
