// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix HOST --check-prefix CHECK
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix DEVICE --check-prefix CHECK
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --check-prefix DEVICE --check-prefix CHECK

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o -| FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -o - | FileCheck %s --check-prefix SIMD-ONLY

// expected-no-diagnostics

// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

#ifndef HEADER
#define HEADER

// HOST-DAG: @c = external global i32,
// HOST-DAG: @c_decl_tgt_ref_ptr = weak global i32* @c
// HOST-DAG: @[[D:.+]] = internal global i32 2
// HOST-DAG: @[[D_PTR:.+]] = weak global i32* @[[D]]
// DEVICE-NOT: @c =
// DEVICE: @c_decl_tgt_ref_ptr = weak global i32* null
// HOST: [[SIZES:@.+]] = private unnamed_addr constant [3 x i64] [i64 4, i64 4, i64 4]
// HOST: [[MAPTYPES:@.+]] = private unnamed_addr constant [3 x i64] [i64 35, i64 531, i64 531]
// HOST: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [{{[0-9]+}} x i8] c"c_decl_tgt_ref_ptr\00"
// HOST: @.omp_offloading.entry.c_decl_tgt_ref_ptr = weak constant %struct.__tgt_offload_entry { i8* bitcast (i32** @c_decl_tgt_ref_ptr to i8*), i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @.omp_offloading.entry_name, i32 0, i32 0), i64 8, i32 1, i32 0 }, section "omp_offloading_entries", align 1
// DEVICE-NOT: internal unnamed_addr constant [{{[0-9]+}} x i8] c"c_{{.*}}_decl_tgt_ref_ptr\00"
// HOST: @.omp_offloading.entry_name{{.*}} = internal unnamed_addr constant [{{[0-9]+}} x i8] c"_{{.*}}d_{{.*}}_decl_tgt_ref_ptr\00"
// HOST: @.omp_offloading.entry.[[D_PTR]] = weak constant %struct.__tgt_offload_entry { i8* bitcast (i32** @[[D_PTR]] to i8*), i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @.omp_offloading.entry_name{{.*}}, i32 0, i32 0

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

// DEVICE: define weak_odr void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l42(i32* noundef nonnull align {{[0-9]+}} dereferenceable{{[^,]*}}
// DEVICE: [[C_REF:%.+]] = load i32*, i32** @c_decl_tgt_ref_ptr,
// DEVICE: [[C:%.+]] = load i32, i32* [[C_REF]],
// DEVICE: store i32 [[C]], i32* %

// HOST: define {{.*}}i32 @{{.*}}maini1{{.*}}()
// HOST: [[BASEPTRS:%.+]] = alloca [3 x i8*],
// HOST: [[PTRS:%.+]] = alloca [3 x i8*],
// HOST: getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0

// HOST: [[BP1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// HOST: [[BP1_CAST:%.+]] = bitcast i8** [[BP1]] to i32***
// HOST: store i32** @c_decl_tgt_ref_ptr, i32*** [[BP1_CAST]],
// HOST: [[P1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// HOST: [[P1_CAST:%.+]] = bitcast i8** [[P1]] to i32**
// HOST: store i32* @c, i32** [[P1_CAST]],

// HOST: [[BP2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// HOST: [[BP2_CAST:%.+]] = bitcast i8** [[BP2]] to i32***
// HOST: store i32** @[[D_PTR]], i32*** [[BP2_CAST]],
// HOST: [[P2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// HOST: [[P2_CAST:%.+]] = bitcast i8** [[P2]] to i32**
// HOST: store i32* @[[D]], i32** [[P2_CAST]],

// HOST: [[BP0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: [[P0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 3, i8** [[BP0]], i8** [[P0]], i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[SIZES]], i{{[0-9]+}} 0, i{{[0-9]+}} 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[MAPTYPES]], i{{[0-9]+}} 0, i{{[0-9]+}} 0), i8** null, i8** null)
// HOST: call void @__omp_offloading_{{.*}}_{{.*}}_{{.*}}maini1{{.*}}_l42(i32* %{{[^,]+}})
// HOST: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @.__omp_offloading_{{.+}}_l47.region_id, i32 2, {{.+}})

// HOST: define internal void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l42(i32* noundef nonnull align {{[0-9]+}} dereferenceable{{.*}})
// HOST: [[C:%.*]] = load i32, i32* @c,
// HOST: store i32 [[C]], i32* %

// CHECK: !{i32 1, !"c_decl_tgt_ref_ptr", i32 1, i32 {{[0-9]+}}}
#endif // HEADER
