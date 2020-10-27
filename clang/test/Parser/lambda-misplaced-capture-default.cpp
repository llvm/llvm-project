// RUN: %clang_cc1 -std=c++20 -Wno-unused-value -fsyntax-only -verify %s

namespace misplaced_capture_default {
void Test() {
  int i = 0;
  [&, i, &] {};   // expected-error {{expected variable name or 'this' in lambda capture list}}
  [&, i, = ] {};  // expected-error {{expected variable name or 'this' in lambda capture list}}
  [=, &i, &] {};  // expected-error {{expected variable name or 'this' in lambda capture list}}
  [=, &i, = ] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}

  [i, &] {};   // expected-error {{capture default must be first}}
  [i, = ] {};  // expected-error {{capture default must be first}}
  [i, = x] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
  [=, &i] {};  // ok
  [&, &i] {};  // expected-error {{'&' cannot precede a capture when the capture default is '&'}}
  [&x = i] {}; // ok
  [=, &x = i] {};  // ok
  [x = &i] {};     // ok
  [=, &x = &i] {}; // expected-error {{non-const lvalue reference to type 'int *' cannot bind to a temporary of type 'int *'}}
  [&, this] {}; // expected-error {{'this' cannot be captured in this context}}

  [i, &, x = 2] {}; // expected-error {{capture default must be first}}
  [i, =, x = 2] {}; // expected-error {{capture default must be first}}
}
} // namespace misplaced_capture_default

namespace misplaced_capture_default_pack {
template <typename... Args> void Test(Args... args) {
  [&, args...] {};         // ok
  [args..., &] {};         // expected-error {{capture default must be first}}
  [=, &args...] {};        // ok
  [&, ... xs = &args] {};  // ok
  [&, ... xs = &] {};      // expected-error {{expected expression}}
  [... xs = &] {};         // expected-error {{expected expression}}
  [... xs = &args, = ] {}; // expected-error {{capture default must be first}}
  [... xs = &args, &] {};  // expected-error {{capture default must be first}}
}
} // namespace misplaced_capture_default_pack
