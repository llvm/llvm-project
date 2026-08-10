// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// Test parsing + semantic analysis
template<typename ...Types> struct count_types {
  static const unsigned value = sizeof...(Types);
};

template<int ...Values> struct count_ints {
  static const unsigned value = sizeof...(Values);
};

// Test instantiation
int check_types[count_types<short, int, long>::value == 3? 1 : -1];
int check_ints[count_ints<1, 2, 3, 4, 5>::value == 5? 1 : -1];

// Test instantiation involving function parameter packs.
struct any {
  template<typename T> any(T);
};

template<typename ...Inits>
void init_me(Inits ...inits) {
  any array[sizeof...(inits)] = { inits... };
}

template void init_me<int, float, double*>(int, float, double*);

// Test parser and semantic recovery.
template<int Value> struct count_ints_2 {
  static const unsigned value = sizeof...(Value); // expected-error{{'Value' does not refer to the name of a parameter pack}}
};

template<typename ...Types> // expected-note{{parameter pack 'Types' declared here}}
struct count_types_2 {
  static const unsigned value = sizeof... Type; // expected-error{{missing parentheses around the size of parameter pack 'Type'}} \
  // expected-error{{'Type' does not refer to the name of a parameter pack; did you mean 'Types'?}}
};

