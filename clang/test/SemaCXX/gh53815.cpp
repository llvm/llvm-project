// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics

// Check that we don't crash due to forgetting to check for placeholders
// in the RHS of '.*'.

template <typename Fn>
static bool has_explicitly_named_overload() {
  return requires { Fn().*&Fn::operator(); };
}

int main() {
  has_explicitly_named_overload<decltype([](auto){})>();
}

template <typename Fn>
constexpr bool has_explicitly_named_overload_2() {
  return requires { Fn().*&Fn::operator(); };
}

static_assert(!has_explicitly_named_overload_2<decltype([](auto){})>());
