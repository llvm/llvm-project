// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s
// expected-no-diagnostics

enum X : short { A, B };
extern decltype(+A) x;
extern int x;

enum Y : long { C, D };
extern decltype(+C) y;
extern long y;

// An enum with a fixed underlying type has an integral promotion to that type,
// and to its promoted type.
enum B : bool { false_, true_ };
template<bool> struct T {};
T<false_> f;
T<true_> t;
// FIXME: DR1407 will make this ill-formed
T<+true_> q; // desired-error {{conversion from 'int' to 'bool'}}

enum B2 : bool {
  a = false,
  b = true,
  c = false_,
  d = true_,
  // FIXME: DR1407 will make this ill-formed
  e = +false_ // desired-error {{conversion from 'int' to 'bool'}}
};

namespace GH56560 {
enum GH56560_1 : bool;
bool GH56560_2(GH56560_1 a, GH56560_1 b) {
    return a == b;
}
} // namespace GH56560
