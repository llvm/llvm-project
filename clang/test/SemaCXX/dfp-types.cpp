// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++2c -fexperimental-decimal-floating-point -fsyntax-only -verify %s

using D32 = float __attribute__((mode(SD)));
using D64 = float __attribute__((mode(DD)));
using D128 = float __attribute__((mode(TD)));

// Dependent type specifiers for the GNU mode attribute base type are ok, but
// must be of a valid type when instantiated.
template<typename T>
T __attribute((mode(SD))) dtamsd; // expected-error {{type of machine mode does not match type of base type}}
auto g1 = dtamsd<float>;
auto g2 = dtamsd<double>;
auto g3 = dtamsd<long double>;
auto g4 = dtamsd<int>; // expected-note {{in instantiation of variable template specialization 'dtamsd' requested here}}
