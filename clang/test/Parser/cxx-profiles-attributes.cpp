// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=disabled %s

// Without -fexperimental-cxx-profiles, the attributes are ignored.

[[profiles::enforce(std::casting)]];
// disabled-warning@-1 {{attribute ignored}}

[[using profiles: enforce(std::casting)]];
// disabled-warning@-1 {{attribute ignored}}

[[profiles::enforce(acme::hardened(level: 3, mode: fast))]]; // #hardened
// expected-warning@-1 {{unknown profile 'acme::hardened'; it will not be enforced}}
// disabled-warning@-2 {{attribute ignored}}

[[profiles::enforce(acme::hardened(level: 4))]];
// expected-warning@-1 {{unknown profile 'acme::hardened'; it will not be enforced}}
// expected-error@-2 {{profile 'acme::hardened' is already enforced with designator 'acme::hardened(level: 3, mode: fast)'}}
// expected-note@#hardened {{profile 'acme::hardened' was enforced here}}
// disabled-warning@-4 {{attribute ignored}}

[[profiles::enforce(::std::casting)]];
// expected-error@-1 {{expected a profile name}}
// disabled-warning@-2 {{attribute ignored}}

[[profiles::enforce(std::)]];
// expected-error@-1 {{expected a profile name}}
// disabled-warning@-2 {{attribute ignored}}

[[profiles::enforce(std::casting, acme::other)]];
// expected-warning@-1 {{unknown profile 'acme::other'; it will not be enforced}}
// disabled-warning@-2 {{attribute ignored}}

void f(int *p) {
  [[profiles::suppress(std::casting)]] (void)p;
  // disabled-warning@-1 {{attribute ignored}}

  [[profiles::suppress(std::casting, rule: reinterpret_cast,
                       justification: "a reason")]] (void)p;
  // disabled-warning@-2 {{attribute ignored}}

  [[profiles::suppress(std::casting, rule)]] (void)p;
  // expected-error@-1 {{expected 'name: value' argument in}}
  // disabled-warning@-2 {{attribute ignored}}

  [[profiles::suppress(std::casting, rule: )]] (void)p;
  // expected-error@-1 {{invalid token in profile argument}}
  // disabled-warning@-2 {{attribute ignored}}

  [[profiles::suppress]] (void)p;
  // expected-error@-1 {{'profiles::suppress' attribute requires a profile name}}
  // disabled-warning@-2 {{attribute ignored}}
}
