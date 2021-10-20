// RUN: %clang_cc1 -std=c++2b -verify %s

namespace PR52206 {
constexpr auto f() {
  if consteval  { return 0;   }
  if !consteval { return 0.0; } // expected-error {{'auto' in return type deduced as 'double' here but deduced as 'int' in earlier return statement}}
}

constexpr auto g() {
  if !consteval { return 0;   }
  if consteval  { return 0.0; } // expected-error {{'auto' in return type deduced as 'double' here but deduced as 'int' in earlier return statement}}
}

constexpr auto h() {
  if consteval  { return 0; }
  if !consteval { return 0; } // okay
}

constexpr auto i() {
  if consteval {
    if consteval { // expected-warning {{consteval if is always true in an immediate context}}
	  return 1;
	}
	return 2;
  } else {
    return 1.0; // expected-error {{'auto' in return type deduced as 'double' here but deduced as 'int' in earlier return statement}}
  }
}

void test() {
  auto x1 = f();
  constexpr auto y1 = f();

  auto x2 = g();
  constexpr auto y2 = g();

  auto x3 = h();
  constexpr auto y3 = h();

  auto x4 = i();
  constexpr auto y4 = i();
}
} // namespace PR52206

consteval int *make() { return new int; }
auto f() {
  if constexpr (false) {
    if consteval {
      // Immediate function context, so call to `make()` is valid.
      // Discarded statement context, so `return 0;` is valid too.
      delete make();
      return 0;
    }
  }
  // FIXME: this error should not happen.
  return 0.0; // expected-error {{'auto' in return type deduced as 'double' here but deduced as 'int' in earlier return statement}}
}
