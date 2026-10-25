// RUN: %clang_cc1 -std=c++20 -verify %s -o -

struct empty {};
struct metre : empty { };
struct second : empty { };
template<auto, auto> struct divided_units : empty { };
template<auto> struct quantity { }; // #QUANT

void use() {
  quantity<divided_units<metre{}, second{}>{}> q{};
  quantity<metre{}> q2 = q;
  // expected-error@-1 {{no viable conversion from 'quantity<divided_units<metre{}, second{}>{}>' to 'quantity<metre{}>'}}
  // expected-note@#QUANT {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'quantity<divided_units<metre{}, second{}>{}>' to 'const quantity<metre{}> &' for 1st argument}}
  // expected-note@#QUANT {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'quantity<divided_units<metre{}, second{}>{}>' to 'quantity<metre{}> &&' for 1st argument}}
}


// An empty class has a single value, so the braces of its empty bases are not
// printed, however deep the hierarchy.
struct kilometre : metre { };
template<auto> struct quantity2 { }; // #QUANT2

void use_deep() {
  quantity2<kilometre{}> q{};
  quantity2<second{}> q2 = q;
  // expected-error@-1 {{no viable conversion from 'quantity2<kilometre{}>' to 'quantity2<second{}>'}}
  // expected-note@#QUANT2 2{{candidate constructor}}
}

// A class with data members keeps the braces that position each subobject.
struct tagged : empty { int v; };
template<auto> struct quantity3 { }; // #QUANT3

void use_non_empty() {
  quantity3<tagged{{}, 1}> q{};
  quantity3<tagged{{}, 2}> q2 = q;
  // expected-error@-1 {{{no viable conversion from 'quantity3<tagged{{}, 1}>' to 'quantity3<tagged{{}, 2}>'}}}
  // expected-note@#QUANT3 2{{candidate constructor}}
}
