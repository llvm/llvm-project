// RUN: %clang_cc1 -std=c++11 -verify %s

// A function that is explicitly defaulted shall
// [...]
//   -- not have default arguments
struct DefArg {
  static DefArg &&make();
  DefArg(int n = 5) = default; // expected-error {{an explicitly-defaulted default constructor cannot have default arguments}}
  DefArg(const DefArg &DA = make()) = default; // expected-error {{an explicitly-defaulted default constructor cannot have default arguments}}
  DefArg(const DefArg &DA, int k = 3) = default; // expected-error {{an explicitly-defaulted copy constructor cannot have default arguments}}
  DefArg(DefArg &&DA = make()) = default; // expected-error {{an explicitly-defaulted default constructor cannot have default arguments}}
  DefArg(DefArg &&DA, int k = 3) = default; // expected-error {{an explicitly-defaulted move constructor cannot have default arguments}}
  DefArg &operator=(const DefArg&, int k = 4) = default; // expected-error {{parameter of overloaded 'operator=' cannot have a default argument}}
  DefArg &operator=(DefArg&&, int k = 4) = default; // expected-error {{parameter of overloaded 'operator=' cannot have a default argument}}
  ~DefArg(int k = 5) = default; // expected-error {{destructor cannot have any parameters}}
};
