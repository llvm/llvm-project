// expected-no-diagnostics
#ifndef HEADER
#define HEADER
// Test host codegen.
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1

int Gbla;
long long Gblb;
int &Gblc = Gbla;

// CK1-LABEL: teams_argument_global_local
int teams_argument_global_local(int a){
  int comp = 1;

  int la = 23;
  float lc = 25.0;

  // CK1: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 1, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 0)
  // CK1: call void @{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target
  #pragma omp teams
  {
    ++comp;
  }

  // CK1: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 1, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 0)
  // CK1: call void @{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target
  {{{
    #pragma omp teams
    {
      ++comp;
    }
  }}}

  // CK1-DAG: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 2, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 0)
  // CK1-DAG: [[NT]] = load i32, i32* [[NTA:%[^,]+]],

  // CK1: call void @{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target
  #pragma omp teams num_teams(la)
  {
    ++comp;
  }

  // CK1-DAG: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 2, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 [[NT:%[^,]+]])
  // CK1-DAG: [[NT]] = load i32, i32* [[NTA:%[^,]+]],

  // CK1: call void @{{.+}}(i{{64|32}} %{{.+}})
  #pragma omp target
  #pragma omp teams thread_limit(la)
  {
    ++comp;
  }

  // CK1-DAG: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 5, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 [[TL:%[^,]+]])

  // CK1-DAG: [[NT]] = add nsw i32 [[NTA:%[^,]+]], [[NTB:%[^,]+]]
  // CK1-DAG: [[NTA]] = load i32, i32* @Gbla,
  // CK1-DAG: [[NTB]] = load i32, i32* %{{.+}},

  // CK1-DAG: [[TL]] = trunc i64 [[TLA:%[^,]+]] to i32
  // CK1-DAG: [[TLA]] = add nsw i64 [[TLB:%[^,]+]], [[TLC:%[^,]+]]
  // CK1-DAG: [[TLC]] = fptosi float [[TLD:%[^,]+]] to i64
  // CK1-DAG: [[TLD]] = load float, float* %{{.+}},
  // CK1-DAG: [[TLB]] = load i64, i64* @Gblb,

  // CK1: call void @{{.+}}(i{{.+}} {{.+}}, i{{.+}} {{.+}}, i{{.+}} {{.+}}, i{{.+}} {{.+}}, i{{.+}} {{.+}})
  #pragma omp target
  #pragma omp teams num_teams(Gbla+a) thread_limit(Gblb+(long long)lc)
  {
    ++comp;
  }

  // CK1-DAG: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 {{.+}}, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 [[TL:%[^,]+]])

  // CK1-DAG: [[NT]] = add nsw i32 [[NTA:%[^,]+]], 1
  // CK1-DAG: [[NTA]] = load i32, i32* @Gbla,

  // CK1-DAG: [[TL]] = add nsw i32 [[TLA:%[^,]+]], 2
  // CK1-DAG: [[TLA]] = load i32, i32* @Gbla,

  // CK1: call void @{{.+}}(i{{.+}} {{.+}}
  #pragma omp target
  #pragma omp teams num_teams(Gblc+1) thread_limit(Gblc+2)
  {
    comp += Gblc;
  }

  return comp;
}

#endif // CK1

// Test host codegen.
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2-DAG: [[SSI:%.+]] = type { i32, float }
// CK2-DAG: [[SSL:%.+]] = type { i64, float }
template <typename T>
struct SS{
  T a;
  float b;
};

SS<int> Gbla;
SS<long long> Gblb;

// CK2-LABEL: teams_template_arg
int teams_template_arg(void) {
  int comp = 1;

  SS<int> la;
  SS<long long> lb;

  // CK2-DAG: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 [[TL:%[^,]+]])

  // CK2-DAG: [[NT]] = load i32, i32* getelementptr inbounds ([[SSI]], [[SSI]]* @Gbla, i32 0, i32 0)

  // CK2-DAG: [[TL]] = trunc i64 [[TLA:%[^,]+]] to i32
  // CK2-DAG: [[TLA]] = fptosi float [[TLB:%[^,]+]] to i64
  // CK2-DAG: [[TLB]] = load float, float* [[TLC:%[^,]+]],
  // CK2-DAG: [[TLC]] = getelementptr inbounds [[SSI]], [[SSI]]* %{{.+}}, i32 0, i32 1

  // CK2: call void @{{.+}}({{.+}} {{.+}}, {{.+}} {{.+}}, {{.+}} {{.+}})
  #pragma omp target
  #pragma omp teams num_teams(Gbla.a) thread_limit((long long)la.b)
  {
    ++comp;
  }

  // CK2-DAG: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 [[TL:%[^,]+]])

  // CK2-DAG: [[TL]] = trunc i64 [[TLD:%[^,]+]] to i32
  // CK2-DAG: [[TLD]] = load i64, i64* getelementptr inbounds ([[SSL]], [[SSL]]* @Gblb, i32 0, i32 0),

  // CK2-DAG: [[NT]] = trunc i64 [[NTA:%[^,]+]] to i32
  // CK2-DAG: [[NTA]] = fptosi float [[NTB:%[^,]+]] to i64
  // CK2-DAG: [[NTB]] = load float, float* [[NTC:%[^,]+]],
  // CK2-DAG: [[NTC]] = getelementptr inbounds [[SSL]], [[SSL]]* %{{.+}}, i32 0, i32 1

  // CK2: call void @{{.+}}({{.+}} {{.+}}, {{.+}} {{.+}}, {{.+}} {{.+}})
  #pragma omp target
  #pragma omp teams num_teams((long long)lb.b) thread_limit(Gblb.a)
  {
    ++comp;
  }
  return comp;
}
#endif // CK2

// Test host codegen.
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3: [[SSI:%.+]] = type { i32, float }
// CK3-LABEL: teams_template_struct

template <typename T, int X, long long Y>
struct SS{
  T a;
  float b;

  int foo(void) {
    int comp = 1;

    // CK3-DAG: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* %{{[^,]+}}, i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 [[NT:%[^,]+]], i32 123)

    // CK3-DAG: [[NT]] = load i32, i32* [[NTA:%[^,]+]],
    // CK3-DAG: [[NTA]] = getelementptr inbounds [[SSI]], [[SSI]]* [[NTB:%[^,]+]], i32 0, i32 0
    // CK3-DAG: [[NTB]] = load [[SSI]]*, [[SSI]]** %{{.+}},

    // CK3: call void @{{.+}}({{.+}} {{.+}}, {{.+}} {{.+}})
    #pragma omp target
    #pragma omp teams num_teams(a) thread_limit(X)
    {
      ++comp;
    }

    // CK3-DAG: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* %{{[^,]+}}, i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 456, i32 [[TL:%[^,]+]])

    // CK3-DAG: [[TL]] = add nsw i32 [[TLA:%[^,]+]], 123
    // CK3-DAG: [[TLA]] = fptosi float [[TLB:%[^,]+]] to i32
    // CK3-DAG: [[TLB]] = load float, float* [[TLC:%[^,]+]],
    // CK3-DAG: [[TLC]] = getelementptr inbounds [[SSI]], [[SSI]]* [[THIS:%[^,]+]], i32 0, i32 1

    // CK3: call void @{{.+}}({{.+}} {{.+}}, {{.+}} {{.+}})
    #pragma omp target
    #pragma omp teams num_teams(Y) thread_limit((int)b+X)
    {
      ++comp;
    }
    return comp;
  }
};

int teams_template_struct(void) {
  SS<int, 123, 456> V;
  return V.foo();

}
#endif // CK3

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// SIMD-ONLY3-NOT: {{__kmpc|__tgt}}

#ifdef CK4

// CK4-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK4-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK4-DAG: [[DEF_LOC_0:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// CK4-DEBUG-DAG: [[LOC1:@.+]] = private unnamed_addr constant [{{.+}} x i8] c";{{.*}}teams_codegen.cpp;main;[[@LINE+14]];9;;\00"
// CK4-DEBUG-DAG: [[LOC2:@.+]] = private unnamed_addr constant [{{.+}} x i8] c";{{.*}}teams_codegen.cpp;tmain;[[@LINE+7]];9;;\00"

template <typename T>
int tmain(T argc) {
#pragma omp target
#pragma omp teams
  argc = 0;
  return 0;
}

int main (int argc, char **argv) {
#pragma omp target
#pragma omp teams
  argc = 0;
  return tmain(argv);
}

// CK4:  define {{.*}}void @{{[^,]+}}(i{{.+}} %[[ARGC:.+]])
// CK4:  [[ARGCADDR:%.+]] = alloca i{{.+}}
// CK4:  store i{{.+}} %[[ARGC]], i{{.+}}* [[ARGCADDR]]
// CK4-64:  [[CONV:%.+]] = bitcast i64* [[ARGCADDR]] to i32*
// CK4-64:  call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC_0]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*)* {{.+}} to void (i32*, i32*, ...)*), i32* [[CONV]])
// CK4-32:  call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC_0]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*)* {{.+}} to void (i32*, i32*, ...)*), i32* [[ARGCADDR]])
// CK4:  ret void
// CK4-NEXT: }

// CK4:  define {{.*}}void @{{[^,]+}}(i8** [[ARGC1:%.+]])
// CK4:  [[ARGCADDR1:%.+]] = alloca i8**
// CK4:  store i8** [[ARGC1]], i8*** [[ARGCADDR1]]
// CK4:  call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC_0]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i8***)* {{.+}} to void (i32*, i32*, ...)*), i8*** [[ARGCADDR1]])


#endif // CK4

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-32

// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// SIMD-ONLY4-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifdef CK5

// CK5-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK5-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK5-DAG: [[DEF_LOC_0:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// CK5-DEBUG-DAG: [[LOC1:@.+]] = private unnamed_addr constant [{{.+}} x i8] c";{{.*}}teams_codegen.cpp;main;[[@LINE+14]];9;;\00"
// CK5-DEBUG-DAG: [[LOC2:@.+]] = private unnamed_addr constant [{{.+}} x i8] c";{{.*}}teams_codegen.cpp;tmain;[[@LINE+7]];9;;\00"

template <typename T>
int tmain(T argc) {
  int a = 10;
  int b = 5;
#pragma omp target
#pragma omp teams num_teams(a) thread_limit(b)
  {
  argc = 0;
  }
  return 0;
}

int main (int argc, char **argv) {
  int a = 20;
  int b = 5;
#pragma omp target
#pragma omp teams num_teams(a) thread_limit(b)
  {
  argc = 0;
  }
  return tmain(argv);
}

// CK5:  define {{.*}}void @{{[^,]+}}(i{{.+}} [[AP:%.+]], i{{.+}} [[BP:%.+]], i{{.+}} [[ARGC:.+]])
// CK5:  [[AADDR:%.+]] = alloca i{{.+}}
// CK5:  [[BADDR:%.+]] = alloca i{{.+}}
// CK5:  [[ARGCADDR:%.+]] = alloca i{{.+}}
// CK5:  [[GBL_TH_NUM:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[DEF_LOC_0]])
// CK5:  store i{{.+}} [[AP]], i{{.+}}* [[AADDR]]
// CK5:  store i{{.+}} [[BP]], i{{.+}}* [[BADDR]]
// CK5:  store i{{.+}} [[ARGC]], i{{.+}}* [[ARGCADDR]]
// CK5-64:  [[ACONV:%.+]] = bitcast i64* [[AADDR]] to i32*
// CK5-64:  [[BCONV:%.+]] = bitcast i64* [[BADDR]] to i32*
// CK5-64:  [[CONV:%.+]] = bitcast i64* [[ARGCADDR]] to i32*
// CK5-64:  [[ACONVVAL:%.+]] = load i32, i32* [[ACONV]]
// CK5-64:  [[BCONVVAL:%.+]] = load i32, i32* [[BCONV]]
// CK5-32:  [[ACONVVAL:%.+]] = load i32, i32* [[AADDR]]
// CK5-32:  [[BCONVVAL:%.+]] = load i32, i32* [[BADDR]]
// CK5:  {{.+}} = call i32 @__kmpc_push_num_teams(%struct.ident_t* [[DEF_LOC_0]], i32 [[GBL_TH_NUM]], i32 [[ACONVVAL]], i32 [[BCONVVAL]])
// CK5-64:  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC_0]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*)* @.omp_outlined. to void (i32*, i32*, ...)*), i32* [[CONV]])
// CK5-32:  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC_0]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*)* @.omp_outlined. to void (i32*, i32*, ...)*), i32* [[ARGCADDR]])

// CK5:  define {{.*}}void @{{[^,]+}}(i{{.+}} [[AP:%.+]], i{{.+}} [[BP:%.+]], i{{.+}}** [[ARGC:%.+]])
// CK5:  [[AADDR:%.+]] = alloca i{{.+}}
// CK5:  [[BADDR:%.+]] = alloca i{{.+}}
// CK5:  [[ARGCADDR:%.+]] = alloca i{{.+}}**
// CK5:  [[GBL_TH_NUM:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[DEF_LOC_0]])
// CK5:  store i{{.+}} [[AP]], i{{.+}}* [[AADDR]]
// CK5:  store i{{.+}} [[BP]], i{{.+}}* [[BADDR]]
// CK5:  store i{{.+}}** [[ARGC]], i{{.+}}*** [[ARGCADDR]]
// CK5-64:  [[ACONV:%.+]] = bitcast i64* [[AADDR]] to i32*
// CK5-64:  [[BCONV:%.+]] = bitcast i64* [[BADDR]] to i32*
// CK5-64:  [[ACONVVAL:%.+]] = load i32, i32* [[ACONV]]
// CK5-64:  [[BCONVVAL:%.+]] = load i32, i32* [[BCONV]]
// CK5-64:  {{.+}} = call i32 @__kmpc_push_num_teams(%struct.ident_t* [[DEF_LOC_0]], i32 [[GBL_TH_NUM]], i32 [[ACONVVAL]], i32 [[BCONVVAL]])
// CK5-32:  [[A_VAL:%.+]] = load i32, i32* [[AADDR]]
// CK5-32:  [[B_VAL:%.+]] = load i32, i32* [[BADDR]]
// CK5-32:  {{.+}} = call i32 @__kmpc_push_num_teams(%struct.ident_t* [[DEF_LOC_0]], i32 [[GBL_TH_NUM]], i32 [[A_VAL]], i32 [[B_VAL]])
// CK5:  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC_0]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i{{.+}})* @.omp_outlined.{{.+}} to void (i32*, i32*, ...)*), i{{.+}}*** [[ARGCADDR]])
// CK5:  ret void
// CK5-NEXT: }

#endif // CK5

// Test host codegen.
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK6 -fopenmp-version=50 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-32

// RUN: %clang_cc1 -DCK6 -verify -fopenmp-version=50 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK6

// CK6-LABEL: foo
void foo() {
  // CK6: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* @0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* @{{.+}} to void (i32*, i32*, ...)*))
#pragma omp teams
  ;
}

#endif // CK6

#endif
