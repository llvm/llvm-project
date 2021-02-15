// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
// XFAIL: asserts

void clang_analyzer_eval(bool);

// TODO: The following test will work properly once reinterpret_cast on pointer-to-member is handled properly
namespace testReinterpretCasting {
struct Base {
  int field;
};

struct Derived : public Base {};

struct DoubleDerived : public Derived {};

struct Some {};

void f() {
  int DoubleDerived::*ddf = &Base::field;
  int Base::*bf = reinterpret_cast<int Base::*>(reinterpret_cast<int Derived::*>(reinterpret_cast<int Base::*>(ddf)));
  int Some::*sf = reinterpret_cast<int Some::*>(ddf);
  Base base;
  base.field = 13;
  clang_analyzer_eval(base.*bf == 13); // expected-warning{{TRUE}}
}
} // namespace testReinterpretCasting
