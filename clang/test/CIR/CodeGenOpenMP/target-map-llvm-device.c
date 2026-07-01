// Two-step host-BC ->  device pipeline that mirrors the offloading driver.
//
// Step 1: Host compilation to bitcode (provides offload entry info to device pass).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -fclangir -emit-llvm-bc %s -o %t-cir-host.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -emit-llvm-bc %s -o %t-ogcg-host.bc
//
// Step 2: Device compilation using host BC.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-cir-host.bc \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-ogcg-host.bc \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OGCG

void use(int);

void target_map_to(int x) {
#pragma omp target map(to : x)
  {
    use(x);
  }
}

void target_map_from(int x) {
#pragma omp target map(from : x)
  {
    x = 42;
  }
}

void target_map_tofrom(int x) {
#pragma omp target map(tofrom : x)
  {
    x = x + 1;
  }
}

void target_map_multiple(int a, int b) {
#pragma omp target map(to : a) map(from : b)
  {
    b = a;
  }
}

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_map_to_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         %[[SLOT:.*]] = addrspacecast ptr addrspace(5) %{{.*}} to ptr
// LLVM:         store ptr %[[ARG]], ptr %[[SLOT]], align 8
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         %[[PTR:.*]] = load ptr, ptr %[[SLOT]], align 8
// LLVM:         %[[V:.*]] = load i32, ptr %[[PTR]], align 4
// LLVM:         call void @use(i32 {{.*}} %[[V]])
// LLVM:         call void @__kmpc_target_deinit()
// LLVM:         ret void

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_map_from_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         %[[SLOT:.*]] = addrspacecast ptr addrspace(5) %{{.*}} to ptr
// LLVM:         store ptr %[[ARG]], ptr %[[SLOT]], align 8
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         %[[PTR:.*]] = load ptr, ptr %[[SLOT]], align 8
// LLVM:         store i32 42, ptr %[[PTR]], align 4
// LLVM:         call void @__kmpc_target_deinit()
// LLVM:         ret void

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_map_tofrom_l
// LLVM-SAME:  (ptr %[[ARG:[^,]+]], ptr
// LLVM:         %[[SLOT:.*]] = addrspacecast ptr addrspace(5) %{{.*}} to ptr
// LLVM:         store ptr %[[ARG]], ptr %[[SLOT]], align 8
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         %[[PTR:.*]] = load ptr, ptr %[[SLOT]], align 8
// LLVM:         %[[LD:.*]] = load i32, ptr %[[PTR]], align 4
// LLVM:         %[[ADD:.*]] = add nsw i32 %[[LD]], 1
// LLVM:         store i32 %[[ADD]], ptr %[[PTR]], align 4
// LLVM:         call void @__kmpc_target_deinit()
// LLVM:         ret void

// LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_map_multiple_l
// LLVM-SAME:  (ptr %[[ARG_A:[^,]+]], ptr %[[ARG_B:[^,]+]], ptr
// LLVM:         %[[SLOT_A:.*]] = addrspacecast ptr addrspace(5) %{{.*}} to ptr
// LLVM:         store ptr %[[ARG_A]], ptr %[[SLOT_A]], align 8
// LLVM:         %[[SLOT_B:.*]] = addrspacecast ptr addrspace(5) %{{.*}} to ptr
// LLVM:         store ptr %[[ARG_B]], ptr %[[SLOT_B]], align 8
// LLVM:         call i32 @__kmpc_target_init(
// LLVM:       user_code.entry:
// LLVM:         %[[PTR_A:.*]] = load ptr, ptr %[[SLOT_A]], align 8
// LLVM:         %[[PTR_B:.*]] = load ptr, ptr %[[SLOT_B]], align 8
// LLVM:         %[[A:.*]] = load i32, ptr %[[PTR_A]], align 4
// LLVM:         store i32 %[[A]], ptr %[[PTR_B]], align 4
// LLVM:         call void @__kmpc_target_deinit()
// LLVM:         ret void

// OGCG-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_map_to_l
// OGCG:         call i32 @__kmpc_target_init(
// OGCG:       user_code.entry:
// OGCG:         %[[V:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         call void @use(i32 {{.*}} %[[V]])
// OGCG:         call void @__kmpc_target_deinit()
// OGCG:         ret void

// OGCG-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_map_from_l
// OGCG:         call i32 @__kmpc_target_init(
// OGCG:       user_code.entry:
// OGCG:         store i32 42, ptr %{{.*}}, align 4
// OGCG:         call void @__kmpc_target_deinit()
// OGCG:         ret void

// OGCG-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_map_tofrom_l
// OGCG:         call i32 @__kmpc_target_init(
// OGCG:       user_code.entry:
// OGCG:         %[[LD:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         %[[ADD:.*]] = add nsw i32 %[[LD]], 1
// OGCG:         store i32 %[[ADD]], ptr %{{.*}}, align 4
// OGCG:         call void @__kmpc_target_deinit()
// OGCG:         ret void

// OGCG-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_map_multiple_l
// OGCG:         call i32 @__kmpc_target_init(
// OGCG:       user_code.entry:
// OGCG:         %[[A:.*]] = load i32, ptr %{{.*}}, align 4
// OGCG:         store i32 %[[A]], ptr %{{.*}}, align 4
// OGCG:         call void @__kmpc_target_deinit()
// OGCG:         ret void
