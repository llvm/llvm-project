// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -triple amdgcn-unknown-unknown -target-cpu gfx942 -S -verify=gfx942,expected -o - %s
// RUN: %clang_cc1 -cl-std=CL2.0 -O0 -triple amdgcn-unknown-unknown -target-cpu gfx950 -S -verify=gfx950,expected -o - %s
// REQUIRES: amdgpu-registered-target

typedef unsigned int u32;

void test_global_load_lds_unsupported_size(global u32* src, local u32 *dst, u32 size, u32 offset, u32 aux) {
  __builtin_amdgcn_global_load_lds(src, dst, size, /*offset=*/0, /*aux=*/0); // expected-error{{argument to '__builtin_amdgcn_global_load_lds' must be a constant integer}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/4, offset, /*aux=*/0); // expected-error{{argument to '__builtin_amdgcn_global_load_lds' must be a constant integer}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/4, /*offset=*/0, aux); // expected-error{{argument to '__builtin_amdgcn_global_load_lds' must be a constant integer}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/5, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} gfx942-note {{size must be 1, 2, or 4}} gfx950-note {{size must be 1, 2, 4, 12 or 16}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/0, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} gfx942-note {{size must be 1, 2, or 4}} gfx950-note {{size must be 1, 2, 4, 12 or 16}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/3, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} gfx942-note {{size must be 1, 2, or 4}} gfx950-note {{size must be 1, 2, 4, 12 or 16}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/12, /*offset=*/0, /*aux=*/0); // gfx942-error{{invalid size value}} gfx942-note {{size must be 1, 2, or 4}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/16, /*offset=*/0, /*aux=*/0); // gfx942-error{{invalid size value}} gfx942-note {{size must be 1, 2, or 4}}
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/-1, /*offset=*/0, /*aux=*/0); // expected-error{{invalid size value}} gfx942-note {{size must be 1, 2, or 4}} gfx950-note {{size must be 1, 2, 4, 12 or 16}}
}

__attribute__((target("gfx950-insts")))
void test_global_load_lds_via_target_feature(global u32* src, local u32 *dst, u32 size, u32 offset, u32 aux) {
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/12, /*offset=*/0, /*aux=*/0);
  __builtin_amdgcn_global_load_lds(src, dst, /*size=*/16, /*offset=*/0, /*aux=*/0);
}
