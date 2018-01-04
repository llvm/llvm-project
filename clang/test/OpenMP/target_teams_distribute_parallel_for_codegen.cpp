// expected-no-diagnostics
#ifndef HEADER
#define HEADER
// Test host codegen.
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64 --check-prefix HCK1 --check-prefix HCK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64 --check-prefix HCK1 --check-prefix HCK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32 --check-prefix HCK1 --check-prefix HCK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32 --check-prefix HCK1 --check-prefix HCK1-64

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY1
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY1
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY1
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY1

// Test target codegen - host bc file has to be created first. (no significant differences with host version of target region)
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64 --check-prefix TCK1 --check-prefix TCK1-64 
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64 --check-prefix TCK1 --check-prefix TCK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc 
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32 --check-prefix TCK1 --check-prefix TCK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32 --check-prefix TCK1 --check-prefix TCK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix SIMD-ONLY1 
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY1
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix SIMD-ONLY1
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY1
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

#ifdef CK1

// HCK1: define{{.*}} i32 @{{.+}}target_teams_fun{{.*}}(
int target_teams_fun(int *g){
  int n = 1000;
  int a[1000];
  int te = n / 128;
  int th = 128;
  // discard n_addr
  // HCK1: alloca i32,
  // HCK1: [[TE:%.+]] alloca i32,
  // HCK1: [[TH:%.+]] = alloca i32,
  // discard capture expressions for te and th
  // HCK1: = alloca i32,
  // HCK1: = alloca i32,
  // HCK1: [[N_CAST:%.+]] = alloca i{{32|64}},
  // HCK1: [[TE_CAST:%.+]] = alloca i{{32|64}},
  // HCK1: [[TH_CAST:%.+]] = alloca i{{32|64}},
  // HCK1: [[N_PAR:%.+]] = load{{.+}}, {{.+}} [[N_CAST]],
  // HCK1: [[TE_PAR:%.+]] = load{{.+}}, {{.+}} [[TE_CAST]],
  // HCK1: [[TH_PAR:%.+]] = load{{.+}}, {{.+}} [[TH_CAST]],
  // HCK1: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 4, i8** %{{[^,]+}}, i8** %{{[^,]+}}, 

  // HCK1: call void @[[OFFL1:.+]](i{{32|64}} [[N_PAR]], {{.+}}, i{{32|64}} [[TE_PAR]], i{{32|64}} [[TH_PAR]])
  #pragma omp target teams distribute parallel for num_teams(te), thread_limit(th)
  for(int i = 0; i < n; i++) {
    a[i] = 0;
    #pragma omp cancel for
  }

  // HCK1: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0),
  // HCK1: call void @[[OFFL2:.+]](i{{64|32}} %{{.+}})
  {{{
  #pragma omp target teams distribute parallel for is_device_ptr(g)
  for(int i = 0; i < n; i++) {
    a[i] = g[0];
  }
  }}}

  // outlined target regions
  // HCK1: define internal void @[[OFFL1]](i{{32|64}} [[N_ARG:%.+]], i{{32|64}} [[TE_ARG:%.+]], i{{32|64}} [[TH_ARG:%.+]])
  // TCK1: define void @{{.+}}target_teams_fun{{.*}}(i{{32|64}} [[N_ARG:%.+]], {{.+}}, i{{32|64}} [[TE_ARG:%.+]], i{{32|64}} [[TH_ARG:%.+]])
  // CK1: [[N_ADDR:%.+]] = alloca i{{32|64}},
  // CK1: [[TE_ADDR:%.+]] = alloca i{{32|64}},
  // CK1: [[TH_ADDR:%.+]] = alloca i{{32|64}},
  // TCK1: store {{.+}} [[N_ARG]], {{.+}} [[N_ADDR]],
  // CK1: store{{.+}} [[TE_ARG]], {{.+}} [[TE_ADDR]],
  // CK1: store{{.+}} [[TH_ARG]], {{.+}} [[TH_ADDR]],
  // CK1-64: [[TE_CONV:%.+]] = bitcast{{.+}} [[TE_ADDR]] to
  // CK1-64: [[TH_CONV:%.+]] = bitcast{{.+}} [[TH_ADDR]] to
  // CK1-64: [[TE_VAL:%.+]] = load i32, i32* [[TE_CONV]],
  // CK1-64: [[TH_VAL:%.+]] = load i32, i32* [[TH_CONV]],
  // CK1-32: [[TE_VAL:%.+]] = load i32, i32* [[TE_ADDR]],
  // CK1-32: [[TH_VAL:%.+]] = load i32, i32* [[TH_ADDR]],
  // CK1: {{%.+}} = call i32 @__kmpc_push_num_teams({{.+}}, {{.+}}, i32 [[TE_VAL]], i32 [[TH_VAL]])
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 2, {{.+}} @[[OUTL1:.+]] to {{.+}}, {{.+}}, {{.+}})
  // CK1: ret void

  // CK1: define internal void @[[OUTL1]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.+}} @__kmpc_fork_call(
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // HCK1: define internal void @[[OFFL2]](
  // TCK1: define void @{{.+}}target_teams_fun{{.+}}(
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 3, {{.+}} @[[OUTL2:.+]] to {{.+}}, {{.+}}, {{.+}})
  // CK1: ret void

  // CK1: define internal void @[[OUTL2]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.+}} @__kmpc_fork_call(
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  return a[0];
}

#endif // CK1
#endif // HEADER 
