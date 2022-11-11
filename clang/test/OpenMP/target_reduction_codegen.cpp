// Only test codegen on target side, as private clause does not require any action on the host side
// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
  TT<tx, ty> operator*(const TT<tx, ty> &) { return *this; }
};

// TCHECK: [[S1:%.+]] = type { double }

int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;

  #pragma omp target reduction(*:a)
  {
  }

  // TCHECK: define weak_odr protected void @__omp_offloading_{{.+}}(i32*{{.+}} %{{.+}})
  // TCHECK: [[A:%.+]] = alloca i{{[0-9]+}}*,
  // TCHECK: store {{.+}}, {{.+}} [[A]],
  // TCHECK: load i32*, i32** [[A]],
  // TCHECK: ret void

#pragma omp target reduction(+:a)
  {
    a = 1;
  }

  // TCHECK:  define weak_odr protected void @__omp_offloading_{{.+}}(i32*{{.+}} %{{.+}})
  // TCHECK: [[A:%.+]] = alloca i{{[0-9]+}}*,
  // TCHECK: store {{.+}}, {{.+}} [[A]],
  // TCHECK: [[REF:%.+]] = load i32*, i32** [[A]],
  // TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[REF]],
  // TCHECK: ret void

  #pragma omp target reduction(-:a, aa)
  {
    a = 1;
    aa = 1;
  }

  // TCHECK:  define weak_odr protected void @__omp_offloading_{{.+}}(i32*{{.+}} [[A:%.+]], i16*{{.+}} [[AA:%.+]])
  // TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}}*,
  // TCHECK:  [[AA:%.+]] = alloca i{{[0-9]+}}*,
  // TCHECK: store {{.+}}, {{.+}} [[A]],
  // TCHECK: store {{.+}}, {{.+}} [[AA]],
  // TCHECK: [[A_REF:%.+]] = load i32*, i32** [[A]],
  // TCHECK: [[AA_REF:%.+]] = load i16*, i16** [[AA]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A_REF]],
  // TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[AA_REF]],
  // TCHECK:  ret void

  return a;
}


template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

#pragma omp target reduction(+:a,aa,b)
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

#pragma omp target reduction(-:a,aa,aaa,b)
  {
    a = 1;
    aa = 1;
    aaa = 1;
    b[2] = 1;
  }

  return a;
}

// TCHECK: define weak_odr protected void @__omp_offloading_{{.+}}(i32*{{.+}}, i16*{{.+}}, i8*{{.+}}, [10 x i32]*{{.+}})
// TCHECK:  [[A:%.+]] = alloca i{{[0-9]+}}*,
// TCHECK:  [[A2:%.+]] = alloca i{{[0-9]+}}*,
// TCHECK:  [[A3:%.+]] = alloca i{{[0-9]+}}*,
// TCHECK:  [[B:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK: store {{.+}}, {{.+}} [[A]],
// TCHECK: store {{.+}}, {{.+}} [[A2]],
// TCHECK: store {{.+}}, {{.+}} [[A3]],
// TCHECK: store {{.+}}, {{.+}} [[B]],
// TCHECK: [[A_REF:%.+]] = load i32*, i32** [[A]],
// TCHECK: [[AA_REF:%.+]] = load i16*, i16** [[AA]],
// TCHECK: [[A3_REF:%.+]] = load i8*, i8** [[A3]],
// TCHECK: [[B_REF:%.+]] = load {{.+}}*, {{.+}}** [[B]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A_REF]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[AA_REF]],
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A3_REF]],
// TCHECK:  [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[B_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK:  store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK:  ret void

struct S1 {
  double a;

  int r1(int n){
    int b = n+1;
    short int c[2][n];

#pragma omp target reduction(max:b,c)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

    return c[1][1] + (int)b;
  }

  // TCHECK: define weak_odr protected void @__omp_offloading_{{.+}}([[S1]]* noundef [[TH:%.+]], i32*{{.+}}, i{{[0-9]+}} noundef [[VLA:%.+]], i{{[0-9]+}} noundef [[VLA1:%.+]], i16*{{.+}})
  // TCHECK: [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK: [[B_ADDR:%.+]] = alloca i{{[0-9]+}}*,
  // TCHECK: [[VLA_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[VLA_ADDR2:%.+]] = alloca i{{[0-9]+}},
  // TCHECK: [[C_ADDR:%.+]] = alloca i{{[0-9]+}}*,
  // TCHECK: store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK: store i{{[0-9]+}}* {{.+}}, i{{[0-9]+}}** [[B_ADDR]],
  // TCHECK: store i{{[0-9]+}} [[VLA]], i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK: store i{{[0-9]+}} [[VLA1]], i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK: store i{{[0-9]+}}* {{.+}}, i{{[0-9]+}}** [[C_ADDR]],
  // TCHECK: [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],
  // TCHECK: [[B_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[B_ADDR]],
  // TCHECK: [[VLA_ADDR_REF:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK: [[VLA_ADDR_REF2:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK: [[C_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[C_ADDR]],

  // this->a = (double)b + 1.5;
  // TCHECK: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_REF]],
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
  // TCHECK: [[C_1_REF:%.+]] = getelementptr inbounds i{{[0-9]+}}, i{{[0-9]+}}* [[C_REF]], i{{[0-9]+}} [[C_IND]]
  // TCHECK: [[C_1_1_REF:%.+]] = getelementptr inbounds i{{[0-9]+}}, i{{[0-9]+}}* [[C_1_REF]], i{{[0-9]+}} 1
  // TCHECK: store i{{[0-9]+}} [[A_FIELD_INC_CONV]], i{{[0-9]+}}* [[C_1_1_REF]],

  // finish
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
// TCHECK: define weak_odr protected void @__omp_offloading_{{.+}}(i{{[0-9]+}}*{{.+}}, i{{[0-9]+}}*{{.+}}, [10 x i32]*{{.+}})
// TCHECK: [[A:%.+]] = alloca i{{[0-9]+}}*,
// TCHECK: [[A2:%.+]] = alloca i{{[0-9]+}}*,
// TCHECK: [[B:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK: store {{.+}}, {{.+}} [[A]],
// TCHECK: store {{.+}}, {{.+}} [[A2]],
// TCHECK: store {{.+}}, {{.+}} [[B]],
// TCHECK: [[A_REF:%.+]] = load i32*, i32** [[A]],
// TCHECK: [[AA_REF:%.+]] = load i16*, i16** [[AA]],
// TCHECK: [[B_REF:%.+]] = load {{.+}}*, {{.+}}** [[B]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[A_REF]],
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[AA_REF]],
// TCHECK: [[B_GEP:%.+]] = getelementptr inbounds [10 x i{{[0-9]+}}], [10 x i{{[0-9]+}}]* [[B_REF]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// TCHECK: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[B_GEP]],
// TCHECK: ret void

#endif
