// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64  --check-prefix HCHECK
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32  --check-prefix HCHECK
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix HCHECK

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first. (no significant differences with host version of target region)
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: %ident_t = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: [[DEF_LOC_0:@.+]] = private unnamed_addr constant %ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// CHECK-DAG: [[DEF_LOC_DISTRIBUTE_0:@.+]] = private unnamed_addr constant %ident_t { i32 0, i32 2050, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

// CHECK-LABEL: define {{.*void}} @{{.*}}without_schedule_clause{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void without_schedule_clause(float *a, float *b, float *c, float *d) {
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute
  for (int i = 33; i < 32000000; i += 7) {
    a[i] = b[i] * c[i] * d[i];
  }
}

// CHECK: define {{.*}}void @{{.+}}(i32* noalias [[GBL_TIDP:%.+]], i32* noalias [[BND_TID:%.+]], float** dereferenceable({{[0-9]+}}) [[APTR:%.+]], float** dereferenceable({{[0-9]+}}) [[BPTR:%.+]], float** dereferenceable({{[0-9]+}}) [[CPTR:%.+]], float** dereferenceable({{[0-9]+}}) [[DPTR:%.+]])
// CHECK:  [[TID_ADDR:%.+]] = alloca i32*
// CHECK:  [[IV:%.+iv]] = alloca i32
// CHECK:  [[LB:%.+lb]] = alloca i32
// CHECK:  [[UB:%.+ub]] = alloca i32
// CHECK:  [[ST:%.+stride]] = alloca i32
// CHECK:  [[LAST:%.+last]] = alloca i32
// CHECK-DAG:  store i32* [[GBL_TIDP]], i32** [[TID_ADDR]]
// CHECK-DAG:  store i32 0, i32* [[LB]]
// CHECK-DAG:  store i32 4571423, i32* [[UB]]
// CHECK-DAG:  store i32 1, i32* [[ST]]
// CHECK-DAG:  store i32 0, i32* [[LAST]]
// CHECK-DAG:  [[GBL_TID:%.+]] = load i32*, i32** [[TID_ADDR]]
// CHECK-DAG:  [[GBL_TIDV:%.+]] = load i32, i32* [[GBL_TID]]
// CHECK:  call void @__kmpc_for_static_init_{{.+}}(%ident_t* [[DEF_LOC_DISTRIBUTE_0]], i32 [[GBL_TIDV]], i32 92, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1)
// CHECK-DAG:  [[UBV0:%.+]] = load i32, i32* [[UB]]
// CHECK-DAG:  [[USWITCH:%.+]] = icmp sgt i32 [[UBV0]], 4571423
// CHECK:  br i1 [[USWITCH]], label %[[BBCT:.+]], label %[[BBCF:.+]]
// CHECK-DAG:  [[BBCT]]:
// CHECK-DAG:  br label %[[BBCE:.+]]
// CHECK-DAG:  [[BBCF]]:
// CHECK-DAG:  [[UBV1:%.+]] = load i32, i32* [[UB]]
// CHECK-DAG:  br label %[[BBCE]]
// CHECK:  [[BBCE]]:
// CHECK:  [[SELUB:%.+]] = phi i32 [ 4571423, %[[BBCT]] ], [ [[UBV1]], %[[BBCF]] ]
// CHECK:  store i32 [[SELUB]], i32* [[UB]]
// CHECK:  [[LBV0:%.+]] = load i32, i32* [[LB]]
// CHECK:  store i32 [[LBV0]], i32* [[IV]]
// CHECK:  br label %[[BBINNFOR:.+]]
// CHECK:  [[BBINNFOR]]:
// CHECK:  [[IVVAL0:%.+]] = load i32, i32* [[IV]]
// CHECK:  [[UBV2:%.+]] = load i32, i32* [[UB]]
// CHECK:  [[IVLEUB:%.+]] = icmp sle i32 [[IVVAL0]], [[UBV2]]
// CHECK:  br i1 [[IVLEUB]], label %[[BBINNBODY:.+]], label %[[BBINNEND:.+]]
// CHECK:  [[BBINNBODY]]:
// CHECK:  {{.+}} = load i32, i32* [[IV]]
// ... loop body ...
// CHECK:  br label %[[BBBODYCONT:.+]]
// CHECK:  [[BBBODYCONT]]:
// CHECK:  br label %[[BBINNINC:.+]]
// CHECK:  [[BBINNINC]]:
// CHECK:  [[IVVAL1:%.+]] = load i32, i32* [[IV]]
// CHECK:  [[IVINC:%.+]] = add nsw i32 [[IVVAL1]], 1
// CHECK:  store i32 [[IVINC]], i32* [[IV]]
// CHECK:  br label %[[BBINNFOR]]
// CHECK:  [[BBINNEND]]:
// CHECK:  br label %[[LPEXIT:.+]]
// CHECK:  [[LPEXIT]]:
// CHECK:  call void @__kmpc_for_static_fini(%ident_t* [[DEF_LOC_DISTRIBUTE_0]], i32 [[GBL_TIDV]])
// CHECK:  ret void


// CHECK-LABEL: define {{.*void}} @{{.*}}static_not_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_not_chunked(float *a, float *b, float *c, float *d) {
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute dist_schedule(static)
  for (int i = 32000000; i > 33; i += -7) {
        a[i] = b[i] * c[i] * d[i];
  }
}

// CHECK: define {{.*}}void @.omp_outlined.{{.*}}(i32* noalias [[GBL_TIDP:%.+]], i32* noalias [[BND_TID:%.+]], float** dereferenceable({{[0-9]+}}) [[APTR:%.+]], float** dereferenceable({{[0-9]+}}) [[BPTR:%.+]], float** dereferenceable({{[0-9]+}}) [[CPTR:%.+]], float** dereferenceable({{[0-9]+}}) [[DPTR:%.+]])
// CHECK:  [[TID_ADDR:%.+]] = alloca i32*
// CHECK:  [[IV:%.+iv]] = alloca i32
// CHECK:  [[LB:%.+lb]] = alloca i32
// CHECK:  [[UB:%.+ub]] = alloca i32
// CHECK:  [[ST:%.+stride]] = alloca i32
// CHECK:  [[LAST:%.+last]] = alloca i32
// CHECK-DAG:  store i32* [[GBL_TIDP]], i32** [[TID_ADDR]]
// CHECK-DAG:  store i32 0, i32* [[LB]]
// CHECK-DAG:  store i32 4571423, i32* [[UB]]
// CHECK-DAG:  store i32 1, i32* [[ST]]
// CHECK-DAG:  store i32 0, i32* [[LAST]]
// CHECK-DAG:  [[GBL_TID:%.+]] = load i32*, i32** [[TID_ADDR]]
// CHECK-DAG:  [[GBL_TIDV:%.+]] = load i32, i32* [[GBL_TID]]
// CHECK:  call void @__kmpc_for_static_init_{{.+}}(%ident_t* [[DEF_LOC_DISTRIBUTE_0]], i32 [[GBL_TIDV]], i32 92, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1)
// CHECK-DAG:  [[UBV0:%.+]] = load i32, i32* [[UB]]
// CHECK-DAG:  [[USWITCH:%.+]] = icmp sgt i32 [[UBV0]], 4571423
// CHECK:  br i1 [[USWITCH]], label %[[BBCT:.+]], label %[[BBCF:.+]]
// CHECK-DAG:  [[BBCT]]:
// CHECK-DAG:  br label %[[BBCE:.+]]
// CHECK-DAG:  [[BBCF]]:
// CHECK-DAG:  [[UBV1:%.+]] = load i32, i32* [[UB]]
// CHECK-DAG:  br label %[[BBCE]]
// CHECK:  [[BBCE]]:
// CHECK:  [[SELUB:%.+]] = phi i32 [ 4571423, %[[BBCT]] ], [ [[UBV1]], %[[BBCF]] ]
// CHECK:  store i32 [[SELUB]], i32* [[UB]]
// CHECK:  [[LBV0:%.+]] = load i32, i32* [[LB]]
// CHECK:  store i32 [[LBV0]], i32* [[IV]]
// CHECK:  br label %[[BBINNFOR:.+]]
// CHECK:  [[BBINNFOR]]:
// CHECK:  [[IVVAL0:%.+]] = load i32, i32* [[IV]]
// CHECK:  [[UBV2:%.+]] = load i32, i32* [[UB]]
// CHECK:  [[IVLEUB:%.+]] = icmp sle i32 [[IVVAL0]], [[UBV2]]
// CHECK:  br i1 [[IVLEUB]], label %[[BBINNBODY:.+]], label %[[BBINNEND:.+]]
// CHECK:  [[BBINNBODY]]:
// CHECK:  {{.+}} = load i32, i32* [[IV]]
// ... loop body ...
// CHECK:  br label %[[BBBODYCONT:.+]]
// CHECK:  [[BBBODYCONT]]:
// CHECK:  br label %[[BBINNINC:.+]]
// CHECK:  [[BBINNINC]]:
// CHECK:  [[IVVAL1:%.+]] = load i32, i32* [[IV]]
// CHECK:  [[IVINC:%.+]] = add nsw i32 [[IVVAL1]], 1
// CHECK:  store i32 [[IVINC]], i32* [[IV]]
// CHECK:  br label %[[BBINNFOR]]
// CHECK:  [[BBINNEND]]:
// CHECK:  br label %[[LPEXIT:.+]]
// CHECK:  [[LPEXIT]]:
// CHECK:  call void @__kmpc_for_static_fini(%ident_t* [[DEF_LOC_DISTRIBUTE_0]], i32 [[GBL_TIDV]])
// CHECK:  ret void


// CHECK-LABEL: define {{.*void}} @{{.*}}static_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_chunked(float *a, float *b, float *c, float *d) {
  #pragma omp target
  #pragma omp teams
#pragma omp distribute dist_schedule(static, 5)
  for (unsigned i = 131071; i <= 2147483647; i += 127) {
    a[i] = b[i] * c[i] * d[i];
  }
}

// CHECK: define {{.*}}void @.omp_outlined.{{.*}}(i32* noalias [[GBL_TIDP:%.+]], i32* noalias [[BND_TID:%.+]], float** dereferenceable({{[0-9]+}}) [[APTR:%.+]], float** dereferenceable({{[0-9]+}}) [[BPTR:%.+]], float** dereferenceable({{[0-9]+}}) [[CPTR:%.+]], float** dereferenceable({{[0-9]+}}) [[DPTR:%.+]])
// CHECK:  [[TID_ADDR:%.+]] = alloca i32*
// CHECK:  [[IV:%.+iv]] = alloca i32
// CHECK:  [[LB:%.+lb]] = alloca i32
// CHECK:  [[UB:%.+ub]] = alloca i32
// CHECK:  [[ST:%.+stride]] = alloca i32
// CHECK:  [[LAST:%.+last]] = alloca i32
// CHECK-DAG:  store i32* [[GBL_TIDP]], i32** [[TID_ADDR]]
// CHECK-DAG:  store i32 0, i32* [[LB]]
// CHECK-DAG:  store i32 16908288, i32* [[UB]]
// CHECK-DAG:  store i32 1, i32* [[ST]]
// CHECK-DAG:  store i32 0, i32* [[LAST]]
// CHECK-DAG:  [[GBL_TID:%.+]] = load i32*, i32** [[TID_ADDR]]
// CHECK-DAG:  [[GBL_TIDV:%.+]] = load i32, i32* [[GBL_TID]]
// CHECK:  call void @__kmpc_for_static_init_{{.+}}(%ident_t* [[DEF_LOC_DISTRIBUTE_0]], i32 [[GBL_TIDV]], i32 91, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 5)
// CHECK-DAG:  [[UBV0:%.+]] = load i32, i32* [[UB]]
// CHECK-DAG:  [[USWITCH:%.+]] = icmp ugt i32 [[UBV0]], 16908288
// CHECK:  br i1 [[USWITCH]], label %[[BBCT:.+]], label %[[BBCF:.+]]
// CHECK-DAG:  [[BBCT]]:
// CHECK-DAG:  br label %[[BBCE:.+]]
// CHECK-DAG:  [[BBCF]]:
// CHECK-DAG:  [[UBV1:%.+]] = load i32, i32* [[UB]]
// CHECK-DAG:  br label %[[BBCE]]
// CHECK:  [[BBCE]]:
// CHECK:  [[SELUB:%.+]] = phi i32 [ 16908288, %[[BBCT]] ], [ [[UBV1]], %[[BBCF]] ]
// CHECK:  store i32 [[SELUB]], i32* [[UB]]
// CHECK:  [[LBV0:%.+]] = load i32, i32* [[LB]]
// CHECK:  store i32 [[LBV0]], i32* [[IV]]
// CHECK:  br label %[[BBINNFOR:.+]]
// CHECK:  [[BBINNFOR]]:
// CHECK:  [[IVVAL0:%.+]] = load i32, i32* [[IV]]
// CHECK:  [[UBV2:%.+]] = load i32, i32* [[UB]]
// CHECK:  [[IVLEUB:%.+]] = icmp ule i32 [[IVVAL0]], [[UBV2]]
// CHECK:  br i1 [[IVLEUB]], label %[[BBINNBODY:.+]], label %[[BBINNEND:.+]]
// CHECK:  [[BBINNBODY]]:
// CHECK:  {{.+}} = load i32, i32* [[IV]]
// ... loop body ...
// CHECK:  br label %[[BBBODYCONT:.+]]
// CHECK:  [[BBBODYCONT]]:
// CHECK:  br label %[[BBINNINC:.+]]
// CHECK:  [[BBINNINC]]:
// CHECK:  [[IVVAL1:%.+]] = load i32, i32* [[IV]]
// CHECK:  [[IVINC:%.+]] = add i32 [[IVVAL1]], 1
// CHECK:  store i32 [[IVINC]], i32* [[IV]]
// CHECK:  br label %[[BBINNFOR]]
// CHECK:  [[BBINNEND]]:
// CHECK:  br label %[[LPEXIT:.+]]
// CHECK:  [[LPEXIT]]:
// CHECK:  call void @__kmpc_for_static_fini(%ident_t* [[DEF_LOC_DISTRIBUTE_0]], i32 [[GBL_TIDV]])
// CHECK:  ret void

// CHECK-LABEL: test_precond
void test_precond() {
  char a = 0;
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute
  for(char i = a; i < 10; ++i);
}

// a is passed as a parameter to the outlined functions
// CHECK:  define {{.*}}void @.omp_outlined.{{.*}}(i32* noalias [[GBL_TIDP:%.+]], i32* noalias [[BND_TID:%.+]], i8* dereferenceable({{[0-9]+}}) [[APARM:%.+]])
// CHECK:  store i8* [[APARM]], i8** [[APTRADDR:%.+]]
// ..many loads of %0..
// CHECK:  [[A2:%.+]] = load i8*, i8** [[APTRADDR]]
// CHECK:  [[AVAL0:%.+]] = load i8, i8* [[A2]]
// CHECK:  store i8 [[AVAL0]], i8* [[CAP_EXPR:%.+]],
// CHECK:  [[AVAL1:%.+]] = load i8, i8* [[CAP_EXPR]]
// CHECK:  load i8, i8* [[CAP_EXPR]]
// CHECK:  [[AVAL2:%.+]] = load i8, i8* [[CAP_EXPR]]
// CHECK:  [[ACONV:%.+]] = sext i8 [[AVAL2]] to i32
// CHECK:  [[ACMP:%.+]] = icmp slt i32 [[ACONV]], 10
// CHECK:  br i1 [[ACMP]], label %[[PRECOND_THEN:.+]], label %[[PRECOND_END:.+]]
// CHECK:  [[PRECOND_THEN]]
// CHECK:  call void @__kmpc_for_static_init_4
// CHECK:  call void @__kmpc_for_static_fini
// CHECK:  [[PRECOND_END]]

// no templates for now, as these require special handling in target regions and/or declare target

// HCHECK-LABEL: fint
// HCHECK: call {{.*}}i32 {{.+}}ftemplate
// HCHECK: ret i32

// HCHECK: load i16, i16*
// HCHECK: store i16 %
// HCHECK: call i32 @__tgt_target_teams(
// HCHECK: call void @__kmpc_for_static_init_4(
template <typename T>
T ftemplate() {
  short aa = 0;

#pragma omp target
#pragma omp teams
#pragma omp distribute dist_schedule(static, aa)
  for (int i = 0; i < 100; i++) {
  }
  return T();
}

int fint(void) { return ftemplate<int>(); }

#endif
