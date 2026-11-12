// RUN: %clang_cc1 -std=c++20 -verify %s -o -

struct empty {};
struct metre : empty { };
struct second : empty { };
template<auto, auto> struct divided_units : empty { };
template<auto> struct quantity { }; // #QUANT

void use() {
  quantity<divided_units<metre{}, second{}>{}> q{};
  quantity<metre{}> q2 = q;
  // expected-error@-1 {{no viable conversion from 'quantity<divided_units<metre{}, second{}>{{}}>' to 'quantity<metre{{}}>'}}
  // expected-note@#QUANT {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'quantity<divided_units<metre{}, second{}>{}>' to 'const quantity<metre{{}}> &' for 1st argument}}
  // expected-note@#QUANT {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'quantity<divided_units<metre{}, second{}>{}>' to 'quantity<metre{{}}> &&' for 1st argument}}
}

