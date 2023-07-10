// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix HOST --check-prefix CHECK
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix DEVICE --check-prefix CHECK
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --check-prefix DEVICE --check-prefix CHECK

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o -| FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -o - | FileCheck %s --check-prefix SIMD-ONLY

// expected-no-diagnostics

// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

#ifndef HEADER
#define HEADER

// HOST-DAG: @c = external global i32,
// HOST-DAG: @c_decl_tgt_ref_ptr = weak global ptr @c
// HOST-DAG: @[[D:.+]] = internal global i32 2
// HOST-DAG: @[[D_PTR:.+]] = weak global ptr @[[D]]
// DEVICE-NOT: @c =
// DEVICE: @c_decl_tgt_ref_ptr = weak global ptr null
// HOST: [[SIZES:@.+]] = private unnamed_addr constant [3 x i64] [i64 4, i64 4, i64 4]
// HOST: [[MAPTYPES:@.+]] = private unnamed_addr constant [3 x i64] [i64 35, i64 531, i64 531]
// HOST: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [{{[0-9]+}} x i8] c"c_decl_tgt_ref_ptr\00"
// HOST: @.omp_offloading.entry.c_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { ptr @c_decl_tgt_ref_ptr, ptr @.omp_offloading.entry_name, i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
// DEVICE-NOT: internal unnamed_addr constant [{{[0-9]+}} x i8] c"c_{{.*}}_decl_tgt_ref_ptr\00"
// HOST: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [{{[0-9]+}} x i8] c"_{{.*}}d_{{.*}}_decl_tgt_ref_ptr\00"
// HOST: @.omp_offloading.entry.[[D_PTR]] = weak constant %struct.__tgt_offload_entry { ptr @[[D_PTR]], ptr @.omp_offloading.entry_name{{.*}}

extern int c;
#pragma omp declare target link(c)

static int d = 2;
#pragma omp declare target link(d)

int maini1() {
  int a;
#pragma omp target map(tofrom : a)
  {
    a = c;
    d++;
  }
#pragma omp target
#pragma omp teams
  c = a;
  return 0;
}

// DEVICE: define weak_odr protected void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l42(ptr noundef nonnull align {{[0-9]+}} dereferenceable{{[^,]*}}
// DEVICE: [[C_REF:%.+]] = load ptr, ptr @c_decl_tgt_ref_ptr,
// DEVICE: [[C:%.+]] = load i32, ptr [[C_REF]],
// DEVICE: store i32 [[C]], ptr %

// HOST: define {{.*}}i32 @{{.*}}maini1{{.*}}()
// HOST: [[BASEPTRS:%.+]] = alloca [3 x ptr],
// HOST: [[PTRS:%.+]] = alloca [3 x ptr],
// HOST: getelementptr inbounds [3 x ptr], ptr [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: getelementptr inbounds [3 x ptr], ptr [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0

// HOST: [[BP1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// HOST: store ptr @c_decl_tgt_ref_ptr, ptr [[BP1]],
// HOST: [[P1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// HOST: store ptr @c, ptr [[P1]],

// HOST: [[BP2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// HOST: store ptr @[[D_PTR]], ptr [[BP2]],
// HOST: [[P2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// HOST: store ptr @[[D]], ptr [[P2]],

// HOST: [[BP0:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: [[P0:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr %{{.+}})
// HOST: call void @__omp_offloading_{{.*}}_{{.*}}_{{.*}}maini1{{.*}}_l42(ptr %{{[^,]+}})
// HOST: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 0, i32 0, ptr @.{{.+}}.region_id, ptr %{{.+}})

// HOST: define internal void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l42(ptr noundef nonnull align {{[0-9]+}} dereferenceable{{.*}})
// HOST: [[C:%.*]] = load i32, ptr @c,
// HOST: store i32 [[C]], ptr %

// CHECK: !{i32 1, !"c_decl_tgt_ref_ptr", i32 1, i32 {{[0-9]+}}}
#endif // HEADER
