// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-dangling-field -Wno-dangling-gsl -verify=expected,perfunc %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety-dangling-field -Wno-dangling-gsl -verify=expected %s

#include "Inputs/lifetime-analysis.h"

std::string make();

// A default member initializer can bind a view member to a temporary that dies
// at the end of construction. This must be caught even when the constructor that
// applies the NSDMI is implicit, defaulted, or inheriting -- such synthesized
// bodies never reach the normal warning path.

struct ImplicitCtor {
  std::string_view v = make(); // expected-warning {{stack memory associated with temporary object escapes to the field 'v' which will dangle}}
                               // expected-note@-1 {{this field dangles}}
};
void use_implicit() { ImplicitCtor c; (void)c; } // perfunc-note {{in implicit default constructor for 'ImplicitCtor' first required here}}

struct DefaultedCtor {
  std::string_view v = make(); // expected-warning {{stack memory associated with temporary object escapes to the field 'v' which will dangle}}
                               // expected-note@-1 {{this field dangles}}
  DefaultedCtor() = default;
};
void use_defaulted() { DefaultedCtor c; (void)c; } // perfunc-note {{in defaulted default constructor for 'DefaultedCtor' first required here}}

struct UserCtor {
  std::string_view v = make(); // expected-warning {{stack memory associated with temporary object escapes to the field 'v' which will dangle}}
                               // expected-note@-1 {{this field dangles}}
  UserCtor() {}
};

struct Base {
  Base(int);
};
// An inheriting constructor applies the derived class's NSDMIs.
struct Inheriting : Base {
  using Base::Base;
  std::string_view v = make(); // expected-warning {{stack memory associated with temporary object escapes to the field 'v' which will dangle}}
                               // expected-note@-1 {{this field dangles}}
};
// No "first required here" note here: inheriting-constructor synthesis does not
// surface an instantiation context for the diagnostic.
void use_inheriting() { Inheriting i(0); (void)i; }

// A string literal has static storage, so binding a view to it does not dangle.
struct SafeLiteral {
  std::string_view v = "literal";
};
void use_safe() { SafeLiteral c; (void)c; }
