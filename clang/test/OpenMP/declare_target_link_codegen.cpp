// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix HOST --check-prefix CHECK
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix DEVICE --check-prefix CHECK
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --check-prefix DEVICE --check-prefix CHECK

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o -| FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -o - | FileCheck %s --check-prefix SIMD-ONLY

// expected-no-diagnostics

// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

#ifndef HEADER
#define HEADER

// HOST: @c = external global i32,
// DEVICE-NOT: @c =
// CHECK: @c_decl_tgt_link_ptr = common global i32* null
// HOST: [[SIZES:@.+]] = private unnamed_addr constant [2 x i64] [i64 4, i64 4]
// HOST: [[MAPTYPES:@.+]] = private unnamed_addr constant [2 x i64] [i64 35, i64 531]
// CHECK: @llvm.used = appending global [1 x i8*] [i8* bitcast (i32** @c_decl_tgt_link_ptr to i8*)]

extern int c;
#pragma omp declare target link(c)

int maini1() {
  int a;
#pragma omp target map(tofrom : a)
  {
    a = c;
  }
  return 0;
}

// DEVICE: define void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l[[@LINE-7]](i32* dereferenceable{{[^,]*}}
// DEVICE: [[C_REF:%.+]] = load i32*, i32** @c_decl_tgt_link_ptr,
// DEVICE: [[C:%.+]] = load i32, i32* [[C_REF]],
// DEVICE: store i32 [[C]], i32* %

// HOST: define {{.*}}i32 @{{.*}}maini1{{.*}}()
// HOST: [[BASEPTRS:%.+]] = alloca [2 x i8*],
// HOST: [[PTRS:%.+]] = alloca [2 x i8*],
// HOST: getelementptr inbounds [2 x i8*], [2 x i8*]* [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: [[BP1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// HOST: [[BP1_CAST:%.+]] = bitcast i8** [[BP1]] to i32***
// HOST: store i32** @c_decl_tgt_link_ptr, i32*** [[BP1_CAST]],
// HOST: [[P1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// HOST: [[P1_CAST:%.+]] = bitcast i8** [[P1]] to i32**
// HOST: store i32* @c, i32** [[P1_CAST]],
// HOST: [[BP0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BASEPTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: [[P0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// HOST: call i32 @__tgt_target(i64 -1, i8* @{{[^,]+}}, i32 2, i8** [[BP0]], i8** [[P0]], i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[SIZES]], i{{[0-9]+}} 0, i{{[0-9]+}} 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAPTYPES]], i{{[0-9]+}} 0, i{{[0-9]+}} 0))
// HOST: call void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l[[@LINE-26]](i32* %{{[^,]+}})

// HOST: define internal void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l[[@LINE-28]](i32* dereferenceable{{.*}})
// HOST: [[C:%.*]] = load i32, i32* @c,
// HOST: store i32 [[C]], i32* %

#endif // HEADER
