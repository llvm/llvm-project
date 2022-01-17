// Only test codegen on target side, as private clause does not require any action on the host side
// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

// TCHECK: [[TT:%.+]] = type { i64, i8 }
// TCHECK: [[S1:%.+]] = type { double }

int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;

  #pragma omp target private(a)
  {
  }

  // TCHECK:  define weak void @__omp_offloading_{{.+}}()
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK-NOT: store {{.+}}, {{.+}} [[A]],
  // TCHECK:  ret void

#pragma omp target private(a)
  {
    a = 1;
  }

  // TCHECK:  define weak void @__omp_offloading_{{.+}}()
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
  // TCHECK:  ret void

#pragma omp target private(a, aa)
  {
    a = 1;
    aa = 1;
  }

  // TCHECK:  define weak void @__omp_offloading_{{.+}}()
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
  // TCHECK:  ret void

  #pragma omp target private(a, b, bn, c, cn, d)
  {
    a = 1;
    b[2] = 1.0;
    bn[3] = 1.0;
    c[1][2] = 1.0;
    cn[1][3] = 1.0;
    d.X = 1;
    d.Y = 1;
  }
  // make sure that private variables are generated in all cases and that we use those instances for operations inside the
  // target region
  // TCHECK:  define weak void @__omp_offloading_{{.+}}(i{{[0-9]+}} noundef [[VLA:%.+]], i{{[0-9]+}} noundef [[VLA1:%.+]], i{{[0-9]+}} noundef [[VLA3:%.+]])
  // TCHECK:  [[VLA_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[VLA_ADDR2:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[VLA_ADDR4:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B:%.+]] = alloca [10 x float],
  // TCHECK:  [[SSTACK:%.+]] = alloca i8*,
  // TCHECK:  [[C:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[D:%.+]] = alloca [[TT]],
  // TCHECK:  store i{{[0-9]+}} [[VLA]], i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[VLA1]], i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK:  store i{{[0-9]+}} [[VLA3]], i{{[0-9]+}}* [[VLA_ADDR4]],
  // TCHECK:  [[VLA_ADDR_REF:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK:  [[VLA_ADDR_REF2:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK:  [[VLA_ADDR_REF4:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR4]],
  // TCHECK:  [[RET_STACK:%.+]] = call i8* @llvm.stacksave()
  // TCHECK:  store i8* [[RET_STACK]], i8** [[SSTACK]],
  // TCHECK:  [[VLA5:%.+]] = alloca float, i{{[0-9]+}} [[VLA_ADDR_REF]],
  // TCHECK:  [[VLA6_SIZE:%.+]] = mul{{.+}} i{{[0-9]+}} [[VLA_ADDR_REF2]], [[VLA_ADDR_REF4]]
  // TCHECK:  [[VLA6:%.+]] = alloca double, i{{[0-9]+}} [[VLA6_SIZE]],

  // a = 1
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],

  // b[2] = 1.0
  // TCHECK:  [[B_GEP:%.+]] = getelementptr inbounds [10 x float], [10 x float]* [[B]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // TCHECK:  store float 1.0{{.*}}, float* [[B_GEP]],

  // bn[3] = 1.0
  // TCHECK:  [[BN_GEP:%.+]] = getelementptr inbounds float, float* [[VLA5]], i{{[0-9]+}} 3
  // TCHECK:  store float 1.0{{.*}}, float* [[BN_GEP]],

  // c[1][2] = 1.0
  // TCHECK:  [[C_GEP1:%.+]] = getelementptr inbounds [5 x [10 x double]], [5 x [10 x double]]* [[C]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // TCHECK:  [[C_GEP2:%.+]] = getelementptr inbounds [10 x double], [10 x double]* [[C_GEP1]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // TCHECK:  store double 1.0{{.*}}, double* [[C_GEP2]],

  // cn[1][3] = 1.0
  // TCHECK:  [[CN_IND:%.+]] = mul{{.+}} i{{[0-9]+}} 1, [[VLA_ADDR_REF4]]
  // TCHECK:  [[CN_GEP_IND:%.+]] = getelementptr inbounds double, double* [[VLA6]], i{{[0-9]+}} [[CN_IND]]
  // TCHECK:  [[CN_GEP_3:%.+]] = getelementptr inbounds double, double* [[CN_GEP_IND]], i{{[0-9]+}} 3
  // TCHECK:  store double 1.0{{.*}}, double* [[CN_GEP_3]],

  // d.X = 1
  // [[X_FIELD:%.+]] = getelementptr inbounds [[TT]] [[TT]]* [[D]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // store i{{[0-9]+}} 1, i{{[0-9]+}}* [[X_FIELD]],

  // d.Y = 1
  // [[Y_FIELD:%.+]] = getelementptr inbounds [[TT]] [[TT]]* [[D]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // store i{{[0-9]+}} 1, i{{[0-9]+}}* [[Y_FIELD]],

  // finish
  // [[RELOAD_SSTACK:%.+]] = load i8*, i8** [[SSTACK]],
  // call ovid @llvm.stackrestore(i8* [[RELOAD_SSTACK]])
  // ret void

  return a;
}


template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

#pragma omp target private(a,aa,b)
  {
    a = 1;
    aa = 1;
    b[2] = 1;
  }

  return a;
}

static
int fstatic(int n) {
  int a = 0;
  short aa = 0;
  char aaa = 0;
  int b[10];

#pragma omp target private(a,aa,aaa,b)
  {
    a = 1;
    aa = 1;
    aaa = 1;
    b[2] = 1;
  }

  return a;
}

// TCHECK: define weak void @__omp_offloading_{{.+}}()
// TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A3]],
// TCHECK:  [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[B]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK:  ret void

struct S1 {
  double a;

  int r1(int n){
    int b = n+1;
    short int c[2][n];

#pragma omp target private(b,c)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

    return c[1][1] + (int)b;
  }

  // TCHECK: define weak void @__omp_offloading_{{.+}}([[S1]]* noundef [[TH:%.+]], i{{[0-9]+}} noundef [[VLA:%.+]], i{{[0-9]+}} noundef [[VLA1:%.+]])
  // TCHECK: [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK: [[VLA_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[VLA_ADDR2:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[B:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[SSTACK:%.+]] = alloca i8*,
  // TCHECK: store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK: store i{{[0-9]+}} [[VLA]], i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK: store i{{[0-9]+}} [[VLA1]], i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK: [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],
  // TCHECK: [[VLA_ADDR_REF:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK: [[VLA_ADDR_REF2:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK: [[RET_STACK:%.+]] = call i8* @llvm.stacksave()
  // TCHECK: store i8* [[RET_STACK:%.+]], i8** [[SSTACK]],

  // this->a = (double)b + 1.5;
  // TCHECK: [[VLA_IND:%.+]] = mul{{.+}} i{{[0-9]+}} [[VLA_ADDR_REF]], [[VLA_ADDR_REF2]]
  // TCHECK: [[VLA3:%.+]] = alloca i{{[0-9]+}}, i{{[0-9]+}} [[VLA_IND]],
  // TCHECK: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B]],
  // TCHECK: [[B_CONV:%.+]] = sitofp i{{[0-9]+}} [[B_VAL]] to double
  // TCHECK: [[NEW_A_VAL:%.+]] = fadd double [[B_CONV]], 1.5{{.+}}+00
  // TCHECK: [[A_FIELD:%.+]] = getelementptr inbounds [[S1]], [[S1]]* [[TH_ADDR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // TCHECK: store double [[NEW_A_VAL]], double* [[A_FIELD]],

  // c[1][1] = ++a;
  // TCHECK: [[A_FIELD4:%.+]] = getelementptr inbounds [[S1]], [[S1]]* [[TH_ADDR_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // TCHECK: [[A_FIELD4_VAL:%.+]] = load double, double* [[A_FIELD4]],
  // TCHECK: [[A_FIELD_INC:%.+]] = fadd double [[A_FIELD4_VAL]], 1.0{{.+}}+00
  // TCHECK: store double [[A_FIELD_INC]], double* [[A_FIELD4]],
  // TCHECK: [[A_FIELD_INC_CONV:%.+]] = fptosi double [[A_FIELD_INC]] to i{{[0-9]+}}
  // TCHECK: [[C_IND:%.+]] = mul{{.+}} i{{[0-9]+}} 1, [[VLA_ADDR_REF2]]
  // TCHECK: [[C_1_REF:%.+]] = getelementptr inbounds i{{[0-9]+}}, i{{[0-9]+}}* [[VLA3]], i{{[0-9]+}} [[C_IND]]
  // TCHECK: [[C_1_1_REF:%.+]] = getelementptr inbounds i{{[0-9]+}}, i{{[0-9]+}}* [[C_1_REF]], i{{[0-9]+}} 1
  // TCHECK: store i{{[0-9]+}} [[A_FIELD_INC_CONV]], i{{[0-9]+}}* [[C_1_1_REF]],

  // finish
  // TCHECK: [[RELOAD_SSTACK:%.+]] = load i8*, i8** [[SSTACK]],
  // TCHECK: call void @llvm.stackrestore(i8* [[RELOAD_SSTACK]])
  // TCHECK: ret void
};


int bar(int n){
  int a = 0;
  a += foo(n);
  S1 S;
  a += S.r1(n);
  a += fstatic(n);
  a += ftemplate<int>(n);

  return a;
}

// template
// TCHECK: define weak void @__omp_offloading_{{.+}}()
// TCHECK: [[A:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[A2:%.+]] = alloca i{{[0-9]+}},
// TCHECK: [[B:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A2]],
// TCHECK: [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[B]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK: ret void

#endif
