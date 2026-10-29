// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=PRE --input-file=%t.pre.cir
// RUN: FileCheck %s --check-prefix=POST --input-file=%t.cir
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ll

// Language address spaces in record members are lowered.

struct S {
  __attribute__((opencl_global)) int *p;
  int n;
};

// PRE: !rec_S = !cir.struct<"S" {data !cir.ptr<!s32i, lang_address_space(offload_global)>, data !s32i}>
// POST: !rec_S = !cir.struct<"S" {data !cir.ptr<!s32i, target_address_space(1)>, data !s32i}>
// LLVM: %struct.S = type { ptr addrspace(1), i32 }

[[clang::sycl_external]] void store_member(__attribute__((opencl_global)) int *g) {
  S s;
  s.p = g;
}

// POST-LABEL: cir.func {{.*}}@_Z12store_memberPU3AS1i
// POST: cir.get_member %{{.*}}[0] {name = "p"} : !cir.ptr<!rec_S, target_address_space(4)> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>, target_address_space(4)>

// LLVM-LABEL: define {{.*}}@_Z12store_memberPU3AS1i
// LLVM: %[[P:.*]] = getelementptr inbounds nuw %struct.S, ptr {{.*}}, i32 0, i32 0
// LLVM: store ptr addrspace(1) %{{.*}}, ptr {{.*}}%[[P]], align 8
