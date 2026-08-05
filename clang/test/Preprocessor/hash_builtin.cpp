// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx906 -E %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -E %s -o - | FileCheck %s --check-prefix=SPIRV-AMDGCN

// CHECK: has_s_memtime_inst
// SPIRV-AMDGCN: has_s_memtime_inst
#if __has_builtin(__builtin_amdgcn_s_memtime)
  int has_s_memtime_inst;
#endif

// CHECK-NOT: has_gfx10_inst
// SPIRV-AMDGCN: has_gfx10_inst
#if __has_builtin(__builtin_amdgcn_mov_dpp8)
  int has_gfx10_inst;
#endif
