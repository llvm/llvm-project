// RUN: %clang_cc1 -fsyntax-only -verify=expected,c %s
// RUN: %clang_cc1 -x c++ -std=c++11 -fsyntax-only -verify=expected,cxx %s
// RUN: %clang_cc1 -x c++ -std=c++26 -fsyntax-only -verify=expected,cxx %s
// RUN: %clang_cc1 -x c++ -std=c++26 -fexperimental-new-constant-interpreter -fsyntax-only -verify=expected,cxx %s

#if !__has_builtin(__builtin_observable_checkpoint)
#error missing __builtin_observable_checkpoint
#endif

void test(void) {
  __builtin_observable_checkpoint();
  __builtin_observable_checkpoint(0); // expected-error {{too many arguments to function call, expected 0, have 1}}
  int x = __builtin_observable_checkpoint(); // c-error {{initializing 'int' with an expression of incompatible type 'void'}}
  // cxx-error@-1 {{cannot initialize a variable of type 'int' with an rvalue of type 'void'}}
}

#ifdef __cplusplus
static_assert(noexcept(__builtin_observable_checkpoint()), "must not throw");

constexpr int constant_evaluation(bool b) {
  return b ? (__builtin_observable_checkpoint(), 0) : 1;
}
static_assert(constant_evaluation(false) == 1, "unevaluated checkpoint is OK");
static_assert(constant_evaluation(true) == 0, ""); // expected-error {{static assertion expression is not an integral constant expression}}
// expected-note@-4 {{subexpression not valid in a constant expression}}
// expected-note@-2 {{in call to 'constant_evaluation(true)'}}
#endif
