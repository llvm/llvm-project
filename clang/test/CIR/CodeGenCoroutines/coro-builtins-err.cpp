// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines -fclangir -emit-cir %s -o %t.cir -verify

void a(void) {
  __builtin_coro_alloc(); // expected-error {{this builtin expect that __builtin_coro_id has been used earlier in this function}}
  __builtin_coro_begin(0); // expected-error {{this builtin expect that __builtin_coro_id has been used earlier in this function}}
  __builtin_coro_free(0); // expected-error {{this builtin expect that __builtin_coro_id has been used earlier in this function}}

  __builtin_coro_id(32, 0, 0, 0);
  __builtin_coro_id(32, 0, 0, 0); // expected-error {{only one __builtin_coro_id}}
}
