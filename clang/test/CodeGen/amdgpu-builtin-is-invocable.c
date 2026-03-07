// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=GFX1030
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=GFX900

void permlane16_path(void);
void permlane64_path(void);
void popcount_path(void);
void fallback_path(void);

// __builtin_amdgcn_permlane16 requires "gfx10-insts".
void test_is_invocable_permlane16(void) {
// GFX1030-LABEL: @test_is_invocable_permlane16
// GFX1030: call i1 @llvm.target.has.feature(metadata !"gfx10-insts")
//
// GFX900-LABEL: @test_is_invocable_permlane16
// GFX900: call i1 @llvm.target.has.feature(metadata !"gfx10-insts")
  if (__builtin_is_invocable(__builtin_amdgcn_permlane16))
    permlane16_path();
}

// __builtin_amdgcn_permlane64 requires "gfx11-insts".
void test_is_invocable_permlane64(void) {
// GFX1030-LABEL: @test_is_invocable_permlane64
// GFX1030: call i1 @llvm.target.has.feature(metadata !"gfx11-insts")
//
// GFX900-LABEL: @test_is_invocable_permlane64
// GFX900: call i1 @llvm.target.has.feature(metadata !"gfx11-insts")
  if (__builtin_is_invocable(__builtin_amdgcn_permlane64))
    permlane64_path();
}

// __builtin_popcount has no required features, fold to constant.
void test_is_invocable_popcount(void) {
// GFX1030-LABEL: @test_is_invocable_popcount
// GFX1030-NOT: call i1 @llvm.target.has.feature
// GFX1030: call void @popcount_path
//
// GFX900-LABEL: @test_is_invocable_popcount
// GFX900-NOT: call i1 @llvm.target.has.feature
// GFX900: call void @popcount_path
  if (__builtin_is_invocable(__builtin_popcount))
    popcount_path();
}

void test_dispatch(void) {
// GFX1030-LABEL: @test_dispatch
// GFX1030: call i1 @llvm.target.has.feature(metadata !"gfx10-insts")
//
// GFX900-LABEL: @test_dispatch
// GFX900: call i1 @llvm.target.has.feature(metadata !"gfx10-insts")
  if (__builtin_is_invocable(__builtin_amdgcn_permlane16))
    permlane16_path();
  else
    fallback_path();
}

// Calling a feature-gated builtin inside a guard must not produce an error.
void test_guarded_builtin_call(unsigned *out, unsigned a, unsigned b,
                               unsigned c, unsigned d) {
// GFX1030-LABEL: @test_guarded_builtin_call
// GFX1030: call i1 @llvm.target.has.feature(metadata !"gfx10-insts")
// GFX1030: call{{.*}} i32 @llvm.amdgcn.permlane16
//
// GFX900-LABEL: @test_guarded_builtin_call
// GFX900: call i1 @llvm.target.has.feature(metadata !"gfx10-insts")
// GFX900: call{{.*}} i32 @llvm.amdgcn.permlane16
  if (__builtin_is_invocable(__builtin_amdgcn_permlane16))
    *out = __builtin_amdgcn_permlane16(a, b, c, d, 0, 0);
}
