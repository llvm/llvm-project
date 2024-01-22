// RUN: %clang_cc1 -std=c++23 %s -verify

auto LL0 = [] {};
auto LL1 = []() {};
auto LL2 = []() mutable {};
auto LL3 = []() constexpr {};

auto L0 = [] constexpr {};
auto L1 = [] mutable {};
auto L2 = [] noexcept {};
auto L3 = [] constexpr mutable {};
auto L4 = [] mutable constexpr {};
auto L5 = [] constexpr mutable noexcept {};
auto L6 = [s = 1] mutable {};
auto L7 = [s = 1] constexpr mutable noexcept {};
auto L8 = [] -> bool { return true; };
auto L9 = []<typename T> { return true; };
auto L10 = []<typename T> noexcept { return true; };
auto L11 = []<typename T> -> bool { return true; };
auto L12 = [] consteval {};
auto L13 = []() requires true {}; // expected-error{{non-templated function cannot have a requires clause}}
auto L14 = []<auto> requires true() requires true {};
auto L15 = []<auto> requires true noexcept {};
auto L16 = [] [[maybe_unused]]{};

auto XL0 = [] mutable constexpr mutable {};    // expected-error{{cannot appear multiple times}}
auto XL1 = [] constexpr mutable constexpr {};  // expected-error{{cannot appear multiple times}}
auto XL2 = []) constexpr mutable constexpr {}; // expected-error{{expected body of lambda expression}}
auto XL3 = []( constexpr mutable constexpr {}; // expected-error{{invalid storage class specifier}} \
                                               // expected-error{{function parameter cannot be constexpr}} \
                                               // expected-error{{a type specifier is required}} \
                                               // expected-error{{expected ')'}} \
                                               // expected-note{{to match this '('}} \
                                               // expected-error{{expected body}} \
                                               // expected-warning{{duplicate 'constexpr'}}

// http://llvm.org/PR49736
auto XL4 = [] requires true {}; // expected-error{{expected body}}
auto XL5 = []<auto> requires true requires true {}; // expected-error{{expected body}}
auto XL6 = []<auto> requires true noexcept requires true {}; // expected-error{{expected body}}

auto XL7 = []() static static {}; // expected-error {{cannot appear multiple times}}
auto XL8 = []() static mutable {}; // expected-error {{cannot be both mutable and static}}
auto XL9 = []() static consteval {};
auto XL10 = []() static constexpr {};

auto XL11 = [] static {};
auto XL12 = []() static {};
auto XL13 = []() static extern {};  // expected-error {{expected body of lambda expression}}
auto XL14 = []() extern {};  // expected-error {{expected body of lambda expression}}


void static_captures() {
  int x;
  auto SC1 = [&]() static {}; // expected-error {{a static lambda cannot have any captures}}
  auto SC4 = [x]() static {}; // expected-error {{a static lambda cannot have any captures}}
  auto SC2 = [&x]() static {}; // expected-error {{a static lambda cannot have any captures}}
  auto SC3 = [y=x]() static {}; // expected-error {{a static lambda cannot have any captures}}
  auto SC5 = [&y = x]() static {}; // expected-error {{a static lambda cannot have any captures}}
  auto SC6 = [=]() static {}; // expected-error {{a static lambda cannot have any captures}}
  struct X {
    int z;
    void f() {
      [this]() static {}(); // expected-error {{a static lambda cannot have any captures}}
      [*this]() static {}(); // expected-error {{a static lambda cannot have any captures}}
    }
  };
}

constexpr auto static_capture_constexpr() {
  char n = 'n';
  return [n] static { return n; }(); // expected-error {{a static lambda cannot have any captures}}
}
static_assert(static_capture_constexpr()); // expected-error {{static assertion expression is not an integral constant expression}}

constexpr auto capture_constexpr() {
  char n = 'n';
  return [n] { return n; }();
}
static_assert(capture_constexpr());
