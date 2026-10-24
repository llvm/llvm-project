// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

#pragma omp declare target

double param2d(double (*p)[5][10], int i, int j) { return (*p)[i][j]; }

// CIR-LABEL: cir.func {{.*}} @param2d(
// CIR:         cir.get_element %{{.*}}[%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>> -> !cir.ptr<!cir.array<!cir.double x 10>>
// CIR:         cir.get_element inbounds %{{.*}}[%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.double x 10>> -> !cir.ptr<!cir.double>

// LLVM-LABEL: define hidden double @param2d(
// LLVM:         %[[ROW:.*]] = getelementptr [5 x [10 x double]], ptr
// LLVM:         getelementptr inbounds [10 x double], ptr %[[ROW]],

double uparam2d(double (*p)[5][10], unsigned i, int j) { return (*p)[i][j]; }

// CIR-LABEL: cir.func {{.*}} @uparam2d(
// CIR:         cir.get_element nuw %{{.*}}[%{{.*}} : !u64i] : !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>> -> !cir.ptr<!cir.array<!cir.double x 10>>
// CIR:         cir.get_element inbounds %{{.*}}[%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.double x 10>> -> !cir.ptr<!cir.double>

// LLVM-LABEL: define hidden double @uparam2d(
// LLVM:         %[[ROW:.*]] = getelementptr nuw [5 x [10 x double]], ptr
// LLVM:         getelementptr inbounds [10 x double], ptr %[[ROW]],

double param3d(double (*p)[2][5][10], int i, int j, int k) {
  return (*p)[i][j][k];
}

// CIR-LABEL: cir.func {{.*}} @param3d(
// CIR:         %[[PLANE:.*]] = cir.get_element %{{.*}}[%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 10> x 5> x 2>> -> !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>>
// CIR:         %[[ROW:.*]] = cir.get_element %[[PLANE]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>> -> !cir.ptr<!cir.array<!cir.double x 10>>
// CIR:         cir.get_element inbounds %[[ROW]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.double x 10>> -> !cir.ptr<!cir.double>

// LLVM-LABEL: define hidden double @param3d(
// LLVM:         %[[PLANE:.*]] = getelementptr [2 x [5 x [10 x double]]], ptr
// LLVM:         %[[ROW:.*]] = getelementptr [5 x [10 x double]], ptr %[[PLANE]],
// LLVM:         getelementptr inbounds [10 x double], ptr %[[ROW]],

double local2d(int i, int j) {
  double a[5][10] = {{0}};
  return a[i][j];
}

// CIR-LABEL: cir.func {{.*}} @local2d(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>, target_address_space(5)>
// CIR:         %[[FLAT:.*]] = cir.cast address_space %[[A]] : !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>, target_address_space(5)> -> !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>>
// CIR:         cir.get_element inbounds %[[FLAT]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>> -> !cir.ptr<!cir.array<!cir.double x 10>>
// CIR:         cir.get_element inbounds %{{.*}}[%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.double x 10>> -> !cir.ptr<!cir.double>

// LLVM-LABEL: define hidden double @local2d(
// LLVMCIR:      alloca [5 x [10 x double]], align 8, addrspace(5)
// LLVMCIR:      %[[ROW:.*]] = getelementptr inbounds [5 x [10 x double]], ptr
// OGCG:         call align 8 ptr @__kmpc_alloc_shared(i64 400)
// OGCG:         %[[ROW:.*]] = getelementptr [5 x [10 x double]], ptr
// LLVM:         getelementptr inbounds [10 x double], ptr %[[ROW]],

double local3d(int i, int j, int k) {
  double a[2][5][10] = {{{0}}};
  return a[i][j][k];
}

// CIR-LABEL: cir.func {{.*}} @local3d(
// CIR:         %[[PLANE:.*]] = cir.get_element inbounds %{{.*}}[%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 10> x 5> x 2>> -> !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>>
// CIR:         %[[ROW:.*]] = cir.get_element inbounds %[[PLANE]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>> -> !cir.ptr<!cir.array<!cir.double x 10>>
// CIR:         cir.get_element inbounds %[[ROW]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.double x 10>> -> !cir.ptr<!cir.double>

// LLVM-LABEL: define hidden double @local3d(
// LLVMCIR:      alloca [2 x [5 x [10 x double]]], align 8, addrspace(5)
// LLVMCIR:      %[[PLANE:.*]] = getelementptr inbounds [2 x [5 x [10 x double]]], ptr
// LLVMCIR:      %[[ROW:.*]] = getelementptr inbounds [5 x [10 x double]], ptr %[[PLANE]],
// OGCG:         call align 8 ptr @__kmpc_alloc_shared(i64 800)
// OGCG:         %[[PLANE:.*]] = getelementptr [2 x [5 x [10 x double]]], ptr
// OGCG:         %[[ROW:.*]] = getelementptr [5 x [10 x double]], ptr %[[PLANE]],
// LLVM:         getelementptr inbounds [10 x double], ptr %[[ROW]],

double stride3d(int i, int j, int k) {
  double a[2][5][10] = {{{0}}};
  return (a + i)[0][j][k];
}

// CIR-LABEL: cir.func {{.*}} @stride3d(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 10> x 5> x 2>, target_address_space(5)>
// CIR:         %[[FLAT:.*]] = cir.cast address_space %[[A]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 10> x 5> x 2>, target_address_space(5)> -> !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 10> x 5> x 2>>
// CIR:         %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[FLAT]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!cir.double x 10> x 5> x 2>> -> !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>>
// CIR:         %[[ADD:.*]] = cir.ptr_stride %[[DECAY]], %{{.*}} : (!cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>>, !s32i) -> !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>>
// CIR:         %[[PLANE:.*]] = cir.ptr_stride %[[ADD]], %{{.*}} : (!cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>>, !s64i) -> !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>>
// CIR:         cir.get_element inbounds %[[PLANE]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!cir.array<!cir.double x 10> x 5>> -> !cir.ptr<!cir.array<!cir.double x 10>>

// LLVM-LABEL: define hidden double @stride3d(
// LLVMCIR:      alloca [2 x [5 x [10 x double]]], align 8, addrspace(5)
// LLVMCIR:      %[[ROW:.*]] = getelementptr inbounds [5 x [10 x double]], ptr
// OGCG:         call align 8 ptr @__kmpc_alloc_shared(i64 800)
// OGCG:         %[[ROW:.*]] = getelementptr [5 x [10 x double]], ptr %{{.*}}, i64 0, i64
// LLVM:         getelementptr inbounds [10 x double], ptr %[[ROW]],

#pragma omp end declare target
