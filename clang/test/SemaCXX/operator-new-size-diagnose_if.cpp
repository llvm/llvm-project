// RUN: %clang_cc1 %s -verify -fno-builtin -std=c++14

using size_t = decltype(sizeof(int));

#define _diagnose_if(...) __attribute__((diagnose_if(__VA_ARGS__)))

namespace operator_new {
struct T0 {
  int j = 0;
  static void *operator new(size_t i) _diagnose_if(i == sizeof(int), "yay", "warning"); // expected-note{{from 'diagnose_if'}}
};

struct T1 {
  int j = 0;
  static void *operator new[](size_t i) _diagnose_if(i == 8 * sizeof(int), "yay", "warning"); // expected-note 2{{from 'diagnose_if'}}
};

void run(int x) {
  new T0;       // expected-warning{{yay}}
  new T1[8];    // expected-warning{{yay}}
  new T1[4][2]; // expected-warning{{yay}}
  new T1[x];    // no warning.
}
} // namespace operator_new
