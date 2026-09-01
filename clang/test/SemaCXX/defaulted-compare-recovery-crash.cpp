// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace std {
  struct strong_ordering {
    int n;
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1}, strong_ordering::equal{0}, strong_ordering::greater{1};
  struct reverse_compare {
    constexpr explicit reverse_compare(std::strong_ordering o) : n(-o.n) {} // expected-error {{member initializer 'n' does not name a non-static data member or base class}}
  } // expected-error {{expected ';' after struct}}
  struct B { // expected-note 2{{definition of 'std::B' is not complete until the closing '}'}}
    friend reverse_compare operator<=>(const B&, const B&) = default; // expected-note {{while rewriting comparison as call to 'operator<=>' declared here}}
    static_assert(B{1, 2, 3, 4, 5} >= B{1, 2, 0, 40, 5}); // expected-error 2{{invalid use of incomplete type 'B'}} expected-error {{invalid operands to binary expression ('reverse_compare' and 'int')}}
  } // expected-error {{expected ';' after struct}}
}
