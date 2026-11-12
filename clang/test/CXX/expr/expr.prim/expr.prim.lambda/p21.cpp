// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify
// expected-no-diagnostics

struct DirectInitOnly {
  explicit DirectInitOnly(DirectInitOnly&);
};

void direct_init_capture(DirectInitOnly &dio) {
  [dio] {}();
}
