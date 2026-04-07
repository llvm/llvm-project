// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -Wlifetime-safety -Wno-dangling -verify %s

void throw_branches(bool cond, int *value) {
  (void)(cond ? throw 1 : value);
  (void)(cond ? throw 1 : throw 2);
}

void nested_throw_branches(bool cond, bool cond2, int *value) {
  (void)(cond ? (cond2 ? throw 1 : value) : throw 2);
  (void)(cond ? throw 1 : (cond2 ? value : throw 2));
}

int *f(int *p [[clang::lifetimebound]]);
[[noreturn]] int *noret_f(int *p [[clang::lifetimebound]]);


constexpr bool kTrue = true;
constexpr bool kFalse = false;

int *constexpr_dead_false(int *num) {
  int local = 0;
  return kTrue ? num : f(&local);
}

int *constexpr_dead_nested(int *num) {
  int local = 0;
  return kTrue ? (kTrue ? num : f(&local)) : num;
}

int *constexpr_live_false(int *num) {
  int local = 0;
  return kFalse ? num : f(&local); // expected-warning {{address of stack memory is returned later}} // expected-note {{returned here}}
}

int *constexpr_live_nested(int *num) {
  int local = 0;
  return kTrue ? (kFalse ? num : f(&local)) : num; } // expected-warning {{address of stack memory is returned later}} // expected-note {{returned here}}

int *noreturn_dead_false(bool cond, int *num) {
  int local = 0;
  return cond ? num : noret_f(&local);
}

int *noreturn_dead_nested(bool cond, bool cond2, int *num) {
  int local = 0;
  return cond ? (cond2 ? num : noret_f(&local)) : num;
}
