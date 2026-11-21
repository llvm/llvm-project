// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

#define AS3 __attribute__((address_space(3)))

float &at(AS3 float *p, long i) {
  return ((float *)p)[i];
}

// CIR-LABEL: cir.func{{.*}} @_Z2atPU3AS3fl(
// CIR:   %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.ptr<!cir.float>>
// CIR:   %[[P:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!cir.float, target_address_space(3)>>, !cir.ptr<!cir.float, target_address_space(3)>
// CIR:   %[[G:.*]] = cir.cast address_space %[[P]] : !cir.ptr<!cir.float, target_address_space(3)> -> !cir.ptr<!cir.float>
// CIR:   %[[ELT:.*]] = cir.ptr_stride %[[G]], %{{.*}} : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CIR-NOT: cir.cast bitcast
// CIR:   cir.store %[[ELT]], %[[RETVAL]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>

// LLVM-LABEL: define {{.*}}@_Z2atPU3AS3fl(
// LLVM:   %[[G:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr
// LLVM:   getelementptr float, ptr %[[G]]

// OGCG-LABEL: define {{.*}}@_Z2atPU3AS3fl(
// OGCG:   %[[G:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr
// OGCG:   getelementptr inbounds float, ptr %[[G]]

int &at_int(AS3 float *p, long i) {
  return ((int *)p)[i];
}

// CIR-LABEL: cir.func{{.*}} @_Z6at_intPU3AS3fl(
// CIR:   %[[P:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!cir.float, target_address_space(3)>>, !cir.ptr<!cir.float, target_address_space(3)>
// CIR:   %[[BC:.*]] = cir.cast bitcast %[[P]] : !cir.ptr<!cir.float, target_address_space(3)> -> !cir.ptr<!s32i, target_address_space(3)>
// CIR:   %[[G:.*]] = cir.cast address_space %[[BC]] : !cir.ptr<!s32i, target_address_space(3)> -> !cir.ptr<!s32i>
// CIR:   cir.ptr_stride %[[G]], %{{.*}} : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>

// LLVM-LABEL: define {{.*}}@_Z6at_intPU3AS3fl(
// LLVM:   %[[G:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr
// LLVM:   getelementptr i32, ptr %[[G]]

// OGCG-LABEL: define {{.*}}@_Z6at_intPU3AS3fl(
// OGCG:   %[[G:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr
// OGCG:   getelementptr inbounds i32, ptr %[[G]]

float *g;
void store_addr(AS3 float *p, long i) {
  g = &((float *)p)[i];
}

// CIR-LABEL: cir.func{{.*}} @_Z10store_addrPU3AS3fl(
// CIR:   %[[P:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!cir.float, target_address_space(3)>>, !cir.ptr<!cir.float, target_address_space(3)>
// CIR:   %[[G:.*]] = cir.cast address_space %[[P]] : !cir.ptr<!cir.float, target_address_space(3)> -> !cir.ptr<!cir.float>
// CIR:   %[[ELT:.*]] = cir.ptr_stride %[[G]], %{{.*}} : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CIR:   %[[GADDR:.*]] = cir.get_global @g : !cir.ptr<!cir.ptr<!cir.float>>
// CIR-NOT: cir.cast bitcast
// CIR:   cir.store{{.*}} %[[ELT]], %[[GADDR]] : !cir.ptr<!cir.float>, !cir.ptr<!cir.ptr<!cir.float>>

// LLVM-LABEL: define {{.*}}@_Z10store_addrPU3AS3fl(
// LLVM:   %[[G:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr
// LLVM:   %[[ELT:.*]] = getelementptr float, ptr %[[G]]
// LLVM:   store ptr %[[ELT]], ptr @g

// OGCG-LABEL: define {{.*}}@_Z10store_addrPU3AS3fl(
// OGCG:   %[[G:.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr
// OGCG:   %[[ELT:.*]] = getelementptr inbounds float, ptr %[[G]]
// OGCG:   store ptr %[[ELT]], ptr @g
