// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -triple amdgcn-unknown-unknown -target-cpu gfx940 -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef unsigned int u32;

void test_global_load_lds_unsupported_size(global u32* src, local u32 *dst, u32 size) {
  __builtin_amdgcn_global_load_lds(src, dst, size, /*offset=*/0, /*aux=*/0); // expected-error{{expression is not an integer constant expression}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/5, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} expected-note {{size must be 1, 2, or 4}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/0, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} expected-note {{size must be 1, 2, or 4}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/3, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} expected-note {{size must be 1, 2, or 4}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/12, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} expected-note {{size must be 1, 2, or 4}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/16, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} expected-note {{size must be 1, 2, or 4}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/-1, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} expected-note {{size must be 1, 2, or 4}}
}
