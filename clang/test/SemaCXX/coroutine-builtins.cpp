// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -DERRORS %s

// Check that we don't crash when using __builtin_coro_* without the fcoroutine-ts or -std=c++20 option

#ifdef ERRORS
// expected-error@#A{{use of undeclared identifier '__builtin_coro_done'}}
// expected-error@#B{{use of undeclared identifier '__builtin_coro_id'}}
// expected-error@#C{{use of undeclared identifier '__builtin_coro_alloc'}}
#else
// expected-no-diagnostics
#endif

int main() {
  void *co_h;
  bool d = __builtin_coro_done(co_h); // #A
  __builtin_coro_id(32, 0, 0, 0);     // #B
  __builtin_coro_alloc();             // #C
}
