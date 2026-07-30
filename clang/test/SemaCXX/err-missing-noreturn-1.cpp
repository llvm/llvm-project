// RUN: %clang_cc1 -fsyntax-only -verify %s -Wmissing-noreturn -Wreturn-type -Werror=return-type

struct rdar8875247 {
  ~rdar8875247 ();
};

int rdar8875247_test() {
  rdar8875247 f;
} // expected-error{{non-void function does not return a value}}
