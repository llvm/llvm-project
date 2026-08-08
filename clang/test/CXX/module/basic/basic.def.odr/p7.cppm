// RUN: %clang_cc1 -std=c++20 -verify %s

// Global module fragment for named module 'a'.
module;

// Case 1: original crash – type parameter in GMF
template <class = int> class FooGMF;  // expected-note {{previous default template argument defined here}}
template <class = void> class FooGMF; // expected-error {{template parameter redefines default argument}}

// Case 2: same-TU redefinition without explicit template name
template<class T = int> class FooNonMod;   // expected-note {{previous default template argument defined here}}
template<class T = void> class FooNonMod;  // expected-error {{template parameter redefines default argument}}

// Case 3: non-type parameter in GMF
template <int N = 5> class NTGMF;          // expected-note {{previous default template argument defined here}}
template <int N = 7> class NTGMF;          // expected-error {{template parameter redefines default argument}}


// Case 4: legal vs illegal redefinitions across declaration/definition
template <class T = int> class Legal;
template <class T> class Legal { T value; };

template <class T = int> class Illegal;    // expected-note {{previous default template argument defined here}}
template <class T = void> class Illegal { T value; }; // expected-error {{template parameter redefines default argument}}

// Case 5: multiple redeclarations (type and non-type)
// When the 3rd decl fires, Clang notes every prior decl that had a default.
template <class = int> class Multi;        // expected-note {{previous default template argument defined here}}
template <class = int> class Multi;        // expected-error {{template parameter redefines default argument}} expected-note {{previous default template argument defined here}}
template <class = float> class Multi;      // expected-error {{template parameter redefines default argument}}

template <int N = 1> class MultiNT;        // expected-note {{previous default template argument defined here}}
template <int N = 1> class MultiNT;        // expected-error {{template parameter redefines default argument}} expected-note {{previous default template argument defined here}}
template <int N = 2> class MultiNT;        // expected-error {{template parameter redefines default argument}}

// Case 6: multiple parameters
// For multi-param templates, each parameter gets its own error+note.
// For a 3-decl chain, the 3rd decl notes all prior decls that had defaults (per param).
template <class T = int, class U = double> class Pair; // expected-note 2 {{previous default template argument defined here}}
template <class T = int, class U = double> class Pair; // expected-error 2 {{template parameter redefines default argument}} expected-note 2 {{previous default template argument defined here}}
template <class T = void, class U = float> class Pair; // expected-error 2 {{template parameter redefines default argument}}

template <int N = 5, char C = 'a'> class NTMulti;      // expected-note 2 {{previous default template argument defined here}}
template <int N = 5, char C = 'a'> class NTMulti;      // expected-error 2 {{template parameter redefines default argument}} expected-note 2 {{previous default template argument defined here}}
template <int N = 7, char C = 'b'> class NTMulti;      // expected-error 2 {{template parameter redefines default argument}}

// Case 7: template-template parameter defaults
// Same cascading note pattern for a 3-decl chain with 2 params.
template <class> class A;
template <class> class B;

template <template <class> class TT = A, class T = int> class TmplT; // expected-note 2 {{previous default template argument defined here}}
template <template <class> class TT = A, class T = int> class TmplT; // expected-error 2 {{template parameter redefines default argument}} expected-note 2 {{previous default template argument defined here}}
template <template <class> class TT = B, class T = void> class TmplT; // expected-error 2 {{template parameter redefines default argument}}

// Case 8: forward declaration pattern
template <class T = int> class Forward;   // expected-note {{previous default template argument defined here}}
template <class T> class Forward { T value; };
template <class T = void> class Forward;  // expected-error {{template parameter redefines default argument}}

export module a;

int main() {} // expected-warning {{'main' never has module linkage}}