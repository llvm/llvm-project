// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck --input-file=%t.pre.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// Accessing a data member through a member pointer yields an address in the
// address space of the object.

struct S {
  int a;
  int b;
};

[[clang::sycl_external]] int load(S &s, int S::*mp) { return s.*mp; }

// CIR: cir.func {{.*}} @_Z4loadR1SMS_i
// CIR:   %[[MEMBER:.*]] = cir.get_runtime_member %{{.*}}[%{{.*}} : !cir.data_member<!s32i in !rec_S>] : !cir.ptr<!rec_S, target_address_space(4)> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR:   cir.load {{.*}} %[[MEMBER]] : !cir.ptr<!s32i, target_address_space(4)>, !s32i

// LLVM: define spir_func noundef i32 @_Z4loadR1SMS_i(
// LLVM:   %[[MEMBER:.*]] = getelementptr {{.*}}i8, ptr addrspace(4) %{{.*}}, i64 %{{.*}}
// LLVM:   load i32, ptr addrspace(4) %[[MEMBER]], align 4
