// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc18.0.0 -fcoroutines-ts -emit-llvm %s -o - -verify

void f(void) {
  __builtin_coro_alloc(); // expected-error {{this builtin expect that __builtin_coro_id}}
  __builtin_coro_begin(0); // expected-error {{this builtin expect that __builtin_coro_id}}
  __builtin_coro_free(0); // expected-error {{this builtin expect that __builtin_coro_id}}

  __builtin_coro_id(32, 0, 0, 0);
  __builtin_coro_id(32, 0, 0, 0); // expected-error {{only one __builtin_coro_id}}
}
