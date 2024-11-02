// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fcuda-is-device -fsyntax-only -verify %s

void call_amdgpu_builtins() {
  __builtin_amdgcn_fence(); // expected-error {{too few arguments to function call, expected at least 2, have 0}}
  __builtin_amdgcn_atomic_inc32(); // expected-error {{too few arguments to function call, expected 4, have 0}}
  __builtin_amdgcn_atomic_inc32(0); // expected-error {{too few arguments to function call, expected 4, have 1}}
  __builtin_amdgcn_atomic_inc32(0, 0); // expected-error {{too few arguments to function call, expected 4, have 2}}
  __builtin_amdgcn_atomic_inc32(0, 0, 0); // expected-error {{too few arguments to function call, expected 4, have 3}}
  __builtin_amdgcn_atomic_inc64(); // expected-error {{too few arguments to function call, expected 4, have 0}}
  __builtin_amdgcn_atomic_dec32(); // expected-error {{too few arguments to function call, expected 4, have 0}}
  __builtin_amdgcn_atomic_dec64(); // expected-error {{too few arguments to function call, expected 4, have 0}}
  __builtin_amdgcn_div_scale(); // expected-error {{too few arguments to function call, expected 4, have 0}}
  __builtin_amdgcn_div_scale(0); // expected-error {{too few arguments to function call, expected 4, have 1}}
  __builtin_amdgcn_div_scale(0, 0); // expected-error {{too few arguments to function call, expected 4, have 2}}
  __builtin_amdgcn_div_scale(0, 0, 0); // expected-error {{too few arguments to function call, expected 4, have 3}}
  __builtin_amdgcn_div_scalef(); // expected-error {{too few arguments to function call, expected 4, have 0}}
  __builtin_amdgcn_ds_faddf(); // expected-error {{too few arguments to function call, expected 5, have 0}}
  __builtin_amdgcn_ds_fminf(); // expected-error {{too few arguments to function call, expected 5, have 0}}
  __builtin_amdgcn_ds_fmaxf(); // expected-error {{too few arguments to function call, expected 5, have 0}}
  __builtin_amdgcn_ds_append(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_amdgcn_ds_consume(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_amdgcn_is_shared(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_amdgcn_is_private(); // expected-error {{too few arguments to function call, expected 1, have 0}}
}

