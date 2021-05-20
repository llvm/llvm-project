// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s

// p0388 conversions to unbounded array
// dcl.init.list/3

namespace One {
int ga[1];

auto &frob1() {
  int(&r1)[] = ga;
#if __cplusplus < 202002
  // expected-error@-2{{cannot bind to a value of unrelated type}}
#endif

  return r1;
}

auto &frob2(int (&arp)[1]) {
  int(&r2)[] = arp;
#if __cplusplus < 202002
  // expected-error@-2{{cannot bind to a value of unrelated type}}
#endif

  return r2;
}
} // namespace One

namespace Two {
int ga[1];

auto *frob1() {
  int(*r1)[] = &ga;
#if __cplusplus < 202002
  // expected-error@-2{{with an rvalue of type}}
#endif

  return r1;
}

auto *frob2(int (*arp)[1]) {
  int(*r2)[] = arp;
#if __cplusplus < 202002
  // expected-error@-2{{with an lvalue of type}}
#endif

  return r2;
}
} // namespace Two

namespace Four {
using Inc = int[2];
using Mat = Inc[1];
Mat *ga[2];

auto *frob1() {
  Inc(*const(*r1)[])[] = &ga;
#if __cplusplus < 202002
  // expected-error@-2{{with an rvalue of type}}
#else
  // missing a required 'const'
  Inc(*(*r2)[])[] = &ga; // expected-error{{cannot initialize}}
#endif

  return r1;
}

auto *frob2(Mat *(*arp)[1]) {
  Inc(*const(*r2)[])[] = arp;
#if __cplusplus < 202002
  // expected-error@-2{{with an lvalue of type}}
#else
  Inc(*(*r3)[])[] = arp; // expected-error{{cannot initialize}}
#endif

  return r2;
}

} // namespace Four
