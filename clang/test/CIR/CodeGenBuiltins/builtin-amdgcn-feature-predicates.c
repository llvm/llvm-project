// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t-cir.ll
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG %s --input-file=%t.ll

void test_processor_is(void) {
  if (__builtin_amdgcn_processor_is("gfx900"))
    __builtin_trap();
}

// CIR-LABEL: cir.func {{.*}} @test_processor_is
// CIR: %[[ID:.*]] = cir.const #cir.int<4294967295> : !u32i
// CIR: %[[DEF:.*]] = cir.const #false
// CIR: %[[MD:.*]] = cir.metadata_as_value #cir.md_node<#cir.md_string<"is.gfx900">>
// CIR: cir.call_llvm_intrinsic "spv.named.boolean.spec.constant" %[[ID]], %[[DEF]], %[[MD]] : (!u32i, !cir.bool, !cir.metadata) -> !cir.bool

// LLVM-LABEL: define {{.*}} @test_processor_is
// LLVM: call{{.*}} i1 @llvm.spv.named.boolean.spec.constant(i32 -1, i1 false, metadata ![[IS_GFX900:[0-9]+]])

// OGCG-LABEL: define {{.*}} @test_processor_is
// OGCG: call{{.*}} i1 @llvm.spv.named.boolean.spec.constant(i32 -1, i1 false, metadata ![[IS_GFX900:[0-9]+]])

void test_is_invocable(void) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_permlanex16))
    __builtin_trap();
}

// CIR-LABEL: cir.func {{.*}} @test_is_invocable
// CIR: %[[MD:.*]] = cir.metadata_as_value #cir.md_node<#cir.md_string<"has.gfx10-insts">>
// CIR: cir.call_llvm_intrinsic "spv.named.boolean.spec.constant" %{{.*}}, %{{.*}}, %[[MD]] : (!u32i, !cir.bool, !cir.metadata) -> !cir.bool

// LLVM-LABEL: define {{.*}} @test_is_invocable
// LLVM: call{{.*}} i1 @llvm.spv.named.boolean.spec.constant(i32 -1, i1 false, metadata ![[HAS_GFX10:[0-9]+]])

// OGCG-LABEL: define {{.*}} @test_is_invocable
// OGCG: call{{.*}} i1 @llvm.spv.named.boolean.spec.constant(i32 -1, i1 false, metadata ![[HAS_GFX10:[0-9]+]])

// LLVM-DAG: ![[IS_GFX900]] = !{!"is.gfx900"}
// LLVM-DAG: ![[HAS_GFX10]] = !{!"has.gfx10-insts"}

// OGCG-DAG: ![[IS_GFX900]] = !{!"is.gfx900"}
// OGCG-DAG: ![[HAS_GFX10]] = !{!"has.gfx10-insts"}
