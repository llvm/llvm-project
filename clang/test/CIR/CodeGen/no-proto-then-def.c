// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -Wno-deprecated-non-prototype %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -Wno-deprecated-non-prototype %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -Wno-deprecated-non-prototype %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fclangir -emit-cir -Wno-deprecated-non-prototype %s -o %t-spirv.cir
// RUN: FileCheck --input-file=%t-spirv.cir %s -check-prefix=SPIRV

extern int default_proc();

int test_proc_ptr(int (*proc)()) {
  if (!proc)
    proc = default_proc;
  return 0;
}

int default_proc(int a) { return a; }

// Address of a no-proto decl is taken before the real definition is emitted; the
// final FuncOp must be bitcast back to the no-proto function pointer type for
// stores and other uses that were typed from the earlier declaration.
// CIR: cir.func{{.*}}@test_proc_ptr
// CIR:   cir.get_global @default_proc : !cir.ptr<!cir.func<(!s32i) -> !s32i>>
// CIR:   cir.cast bitcast %{{.*}} : !cir.ptr<!cir.func<(!s32i) -> !s32i>> -> !cir.ptr<!cir.func<(...) -> !s32i>>

// The retyped address keeps the program address space.
// SPIRV: cir.func{{.*}}@test_proc_ptr
// SPIRV:   cir.get_global @default_proc : !cir.ptr<!cir.func<(!s32i) -> !s32i>, target_address_space(4)>
// SPIRV:   cir.cast bitcast %{{.*}} : !cir.ptr<!cir.func<(!s32i) -> !s32i>, target_address_space(4)> -> !cir.ptr<!cir.func<(...) -> !s32i>, target_address_space(4)>

// LLVM: define{{.*}} @test_proc_ptr
// LLVM:   store ptr @default_proc, ptr
// LLVM:   define{{.*}} @default_proc
// LLVM:   ret i32
