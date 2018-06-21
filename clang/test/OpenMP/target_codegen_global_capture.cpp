// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER


// CHECK-DAG: [[GA:@.+]] = global double 1.000000e+00
// CHECK-DAG: [[GB:@.+]] = global double 2.000000e+00
// CHECK-DAG: [[GC:@.+]] = global double 3.000000e+00
// CHECK-DAG: [[GD:@.+]] = global double 4.000000e+00
// CHECK-DAG: [[FA:@.+]] = internal global float 5.000000e+00
// CHECK-DAG: [[FB:@.+]] = internal global float 6.000000e+00
// CHECK-DAG: [[FC:@.+]] = internal global float 7.000000e+00
// CHECK-DAG: [[FD:@.+]] = internal global float 8.000000e+00
// CHECK-DAG: [[BA:@.+]] = internal global float 9.000000e+00
// CHECK-DAG: [[BB:@.+]] = internal global float 1.000000e+01
// CHECK-DAG: [[BC:@.+]] = internal global float 1.100000e+01
// CHECK-DAG: [[BD:@.+]] = internal global float 1.200000e+01
// CHECK-DAG: [[TBA:@.+]] = {{.*}}global float 1.700000e+01
// CHECK-DAG: [[TBB:@.+]] = {{.*}}global float 1.800000e+01
// CHECK-DAG: [[TBC:@.+]] = {{.*}}global float 1.900000e+01
// CHECK-DAG: [[TBD:@.+]] = {{.*}}global float 2.000000e+01

double Ga = 1.0;
double Gb = 2.0;
double Gc = 3.0;
double Gd = 4.0;

// CHECK: define {{.*}} @{{.*}}foo{{.*}}(
// CHECK-SAME: i16 {{[^,]*}}[[A:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[B:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[C:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[D:%[^,]+]])
// CHECK: [[LA:%.+]] = alloca i16,
// CHECK: [[LB:%.+]] = alloca i16,
// CHECK: [[LC:%.+]] = alloca i16,
// CHECK: [[LD:%.+]] = alloca i16,
int foo(short a, short b, short c, short d){
  static float Sa = 5.0;
  static float Sb = 6.0;
  static float Sc = 7.0;
  static float Sd = 8.0;

  // CHECK-DAG:    [[VALLB:%.+]] = load i16, i16* [[LB]],
  // CHECK-64-DAG: [[VALGB:%.+]] = load double, double* [[GB]],
  // CHECK-DAG:    [[VALFB:%.+]] = load float, float* [[FB]],
  // CHECK-64-DAG: [[VALGC:%.+]] = load double, double* [[GC]],
  // CHECK-DAG:    [[VALLC:%.+]] = load i16, i16* [[LC]],
  // CHECK-DAG:    [[VALFC:%.+]] = load float, float* [[FC]],
  // CHECK-DAG:    [[VALLD:%.+]] = load i16, i16* [[LD]],
  // CHECK-64-DAG: [[VALGD:%.+]] = load double, double* [[GD]],
  // CHECK-DAG:    [[VALFD:%.+]] = load float, float* [[FD]],

  // 3 local vars being captured.

  // CHECK-DAG: store i16 [[VALLB]], i16* [[CONVLB:%.+]],
  // CHECK-DAG: [[CONVLB]] = bitcast i[[sz:64|32]]* [[CADDRLB:%.+]] to i16*
  // CHECK-DAG: [[CVALLB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLB]],
  // CHECK-DAG: store i[[sz]] [[CVALLB]], i[[sz]]* [[CBP:%.+]],
  // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLB:%.+]] to i[[sz]]*
  // CHECK-DAG: [[GEPLB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // CHECK-DAG: store i16 [[VALLC]], i16* [[CONVLC:%.+]],
  // CHECK-DAG: [[CONVLC]] = bitcast i[[sz]]* [[CADDRLC:%.+]] to i16*
  // CHECK-DAG: [[CVALLC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLC]],
  // CHECK-DAG: store i[[sz]] [[CVALLC]], i[[sz]]* [[CBP:%.+]],
  // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLC:%.+]] to i[[sz]]*
  // CHECK-DAG: [[GEPLC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // CHECK-DAG: store i16 [[VALLD]], i16* [[CONVLD:%.+]],
  // CHECK-DAG: [[CONVLD]] = bitcast i[[sz]]* [[CADDRLD:%.+]] to i16*
  // CHECK-DAG: [[CVALLD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLD]],
  // CHECK-DAG: store i[[sz]] [[CVALLD]], i[[sz]]* [[CBP:%.+]],
  // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLD:%.+]] to i[[sz]]*
  // CHECK-DAG: [[GEPLD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // 3 static vars being captured.

  // CHECK-DAG: store float [[VALFB]], float* [[CONVFB:%.+]],
  // CHECK-DAG: [[CONVFB]] = bitcast i[[sz]]* [[CADDRFB:%.+]] to float*
  // CHECK-DAG: [[CVALFB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFB]],
  // CHECK-DAG: store i[[sz]] [[CVALFB]], i[[sz]]* [[CBP:%.+]],
  // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFB:%.+]] to i[[sz]]*
  // CHECK-DAG: [[GEPFB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // CHECK-DAG: store float [[VALFC]], float* [[CONVFC:%.+]],
  // CHECK-DAG: [[CONVFC]] = bitcast i[[sz]]* [[CADDRFC:%.+]] to float*
  // CHECK-DAG: [[CVALFC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFC]],
  // CHECK-DAG: store i[[sz]] [[CVALFC]], i[[sz]]* [[CBP:%.+]],
  // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFC:%.+]] to i[[sz]]*
  // CHECK-DAG: [[GEPFC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // CHECK-DAG: store float [[VALFD]], float* [[CONVFD:%.+]],
  // CHECK-DAG: [[CONVFD]] = bitcast i[[sz]]* [[CADDRFD:%.+]] to float*
  // CHECK-DAG: [[CVALFD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFD]],
  // CHECK-DAG: store i[[sz]] [[CVALFD]], i[[sz]]* [[CBP:%.+]],
  // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFD:%.+]] to i[[sz]]*
  // CHECK-DAG: [[GEPFD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // 3 static global vars being captured.

  // CHECK-64-DAG: store double [[VALGB]], double* [[CONVGB:%.+]],
  // CHECK-64-DAG: [[CONVGB]] = bitcast i[[sz]]* [[CADDRGB:%.+]] to double*
  // CHECK-64-DAG: [[CVALGB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGB]],
  // CHECK-64-DAG: store i[[sz]] [[CVALGB]], i[[sz]]* [[CBP:%.+]],
  // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGB:%.+]] to i[[sz]]*
  // CHECK-32-DAG: store double* @Gb, double** [[CBP:%.+]],
  // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGB:%.+]] to double**
  // CHECK-DAG: [[GEPGB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // CHECK-64-DAG: store double [[VALGC]], double* [[CONVGC:%.+]],
  // CHECK-64-DAG: [[CONVGC]] = bitcast i[[sz]]* [[CADDRGC:%.+]] to double*
  // CHECK-64-DAG: [[CVALGC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGC]],
  // CHECK-64-DAG: store i[[sz]] [[CVALGC]], i[[sz]]* [[CBP:%.+]],
  // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGC:%.+]] to i[[sz]]*
  // CHECK-32-DAG: store double* @Gc, double** [[CBP:%.+]],
  // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGC:%.+]] to double**
  // CHECK-DAG: [[GEPGC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // CHECK-64-DAG: store double [[VALGD]], double* [[CONVGD:%.+]],
  // CHECK-64-DAG: [[CONVGD]] = bitcast i[[sz]]* [[CADDRGD:%.+]] to double*
  // CHECK-64-DAG: [[CVALGD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGD]],
  // CHECK-64-DAG: store i[[sz]] [[CVALGD]], i[[sz]]* [[CBP:%.+]],
  // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGD:%.+]] to i[[sz]]*
  // CHECK-32-DAG: store double* @Gd, double** [[CBP:%.+]],
  // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGD:%.+]] to double**
  // CHECK-DAG: [[GEPGD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

  // CHECK: call i32 @__tgt_target
  // CHECK: call void [[OFFLOADF:@.+]](
  // Capture b, Gb, Sb, Gc, c, Sc, d, Gd, Sd
  #pragma omp target if(Ga>0.0 && a>0 && Sa>0.0)
  {
    b += 1;
    Gb += 1.0;
    Sb += 1.0;

    // CHECK: define internal void [[OFFLOADF]]({{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}})
    // The parallel region only uses 3 captures.
    // CHECK:     call {{.*}}@__kmpc_fork_call(%ident_t* {{.+}}, i32 {{.+}}, void (i32*, i32*, ...)* bitcast ({{.*}}[[PARF:@.+]] to {{.*}}), {{.+}}* %{{.+}}, {{.+}}* %{{.+}}, {{.+}}* %{{.+}})
    // CHECK:     call void @.omp_outlined.(i32* %{{.+}}, i32* %{{.+}}, {{.+}}* %{{.+}}, {{.+}}* %{{.+}}, {{.+}}* %{{.+}})
    // Capture d, Gd, Sd,

    // CHECK: define internal void [[PARF]](i32* noalias %{{.*}}, i32* noalias %{{.*}},
    #pragma omp parallel if(Gc>0.0 && c>0 && Sc>0.0)
    {
      d += 1;
      Gd += 1.0;
      Sd += 1.0;
    }
  }
  return a + b + c + d + (int)Sa + (int)Sb + (int)Sc + (int)Sd;
}

// CHECK: define {{.*}} @{{.*}}bar{{.*}}(
// CHECK-SAME: i16 {{[^,]*}}[[A:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[B:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[C:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[D:%[^,]+]])
// CHECK: [[LA:%.+]] = alloca i16
// CHECK: [[LB:%.+]] = alloca i16
// CHECK: [[LC:%.+]] = alloca i16
// CHECK: [[LD:%.+]] = alloca i16
int bar(short a, short b, short c, short d){
  static float Sa = 9.0;
  static float Sb = 10.0;
  static float Sc = 11.0;
  static float Sd = 12.0;

  // CHECK: call void {{.*}}@__kmpc_fork_call(%ident_t* {{.+}}, i32 {{.+}}, void (i32*, i32*, ...)* bitcast ({{.*}}[[PARF:@.+]] to {{.*}}), i16* %{{.+}}, i16* %{{.+}}, i16* %{{.+}}, i16* %{{.+}})
  // CHECK: define internal void [[PARF]](i32* noalias %{{.*}}, i32* noalias %{{.*}}, i16* dereferenceable(2) [[A:%.+]], i16* dereferenceable(2) [[B:%.+]], i16* dereferenceable(2) [[C:%.+]], i16* dereferenceable(2) [[D:%.+]])
  // Capture a, b, c, d
  // CHECK: [[ALLOCLA:%.+]] = alloca i16
  // CHECK: [[ALLOCLB:%.+]] = alloca i16
  // CHECK: [[ALLOCLC:%.+]] = alloca i16
  // CHECK: [[ALLOCLD:%.+]] = alloca i16
  // CHECK: [[LLA:%.+]] = load i16*, i16** [[ALLOCLA]],
  // CHECK: [[LLB:%.+]] = load i16*, i16** [[ALLOCLB]],
  // CHECK: [[LLC:%.+]] = load i16*, i16** [[ALLOCLC]],
  // CHECK: [[LLD:%.+]] = load i16*, i16** [[ALLOCLD]],
  #pragma omp parallel
  {
    // CHECK-DAG:    [[VALLB:%.+]] = load i16, i16* [[LLB]],
    // CHECK-64-DAG: [[VALGB:%.+]] = load double, double* [[GB]],
    // CHECK-DAG:    [[VALFB:%.+]] = load float, float* [[BB]],
    // CHECK-64-DAG: [[VALGC:%.+]] = load double, double* [[GC]],
    // CHECK-DAG:    [[VALLC:%.+]] = load i16, i16* [[LLC]],
    // CHECK-DAG:    [[VALFC:%.+]] = load float, float* [[BC]],
    // CHECK-DAG:    [[VALLD:%.+]] = load i16, i16* [[LLD]],
    // CHECK-64-DAG: [[VALGD:%.+]] = load double, double* [[GD]],
    // CHECK-DAG:    [[VALFD:%.+]] = load float, float* [[BD]],

    // 3 local vars being captured.

    // CHECK-DAG: store i16 [[VALLB]], i16* [[CONVLB:%.+]],
    // CHECK-DAG: [[CONVLB]] = bitcast i[[sz:64|32]]* [[CADDRLB:%.+]] to i16*
    // CHECK-DAG: [[CVALLB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLB]],
    // CHECK-DAG: store i[[sz]] [[CVALLB]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLB:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPLB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-DAG: store i16 [[VALLC]], i16* [[CONVLC:%.+]],
    // CHECK-DAG: [[CONVLC]] = bitcast i[[sz]]* [[CADDRLC:%.+]] to i16*
    // CHECK-DAG: [[CVALLC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLC]],
    // CHECK-DAG: store i[[sz]] [[CVALLC]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLC:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPLC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-DAG: store i16 [[VALLD]], i16* [[CONVLD:%.+]],
    // CHECK-DAG: [[CONVLD]] = bitcast i[[sz]]* [[CADDRLD:%.+]] to i16*
    // CHECK-DAG: [[CVALLD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLD]],
    // CHECK-DAG: store i[[sz]] [[CVALLD]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLD:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPLD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // 3 static vars being captured.

    // CHECK-DAG: store float [[VALFB]], float* [[CONVFB:%.+]],
    // CHECK-DAG: [[CONVFB]] = bitcast i[[sz]]* [[CADDRFB:%.+]] to float*
    // CHECK-DAG: [[CVALFB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFB]],
    // CHECK-DAG: store i[[sz]] [[CVALFB]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFB:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPFB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-DAG: store float [[VALFC]], float* [[CONVFC:%.+]],
    // CHECK-DAG: [[CONVFC]] = bitcast i[[sz]]* [[CADDRFC:%.+]] to float*
    // CHECK-DAG: [[CVALFC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFC]],
    // CHECK-DAG: store i[[sz]] [[CVALFC]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFC:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPFC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-DAG: store float [[VALFD]], float* [[CONVFD:%.+]],
    // CHECK-DAG: [[CONVFD]] = bitcast i[[sz]]* [[CADDRFD:%.+]] to float*
    // CHECK-DAG: [[CVALFD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFD]],
    // CHECK-DAG: store i[[sz]] [[CVALFD]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFD:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPFD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // 3 static global vars being captured.

    // CHECK-64-DAG: store double [[VALGB]], double* [[CONVGB:%.+]],
    // CHECK-64-DAG: [[CONVGB]] = bitcast i[[sz]]* [[CADDRGB:%.+]] to double*
    // CHECK-64-DAG: [[CVALGB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGB]],
    // CHECK-64-DAG: store i[[sz]] [[CVALGB]], i[[sz]]* [[CBP:%.+]],
    // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGB:%.+]] to i[[sz]]*
    // CHECK-32-DAG: store double* @Gb, double** [[CBP:%.+]],
    // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGB:%.+]] to double**
    // CHECK-DAG: [[GEPGB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-64-DAG: store double [[VALGC]], double* [[CONVGC:%.+]],
    // CHECK-64-DAG: [[CONVGC]] = bitcast i[[sz]]* [[CADDRGC:%.+]] to double*
    // CHECK-64-DAG: [[CVALGC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGC]],
    // CHECK-64-DAG: store i[[sz]] [[CVALGC]], i[[sz]]* [[CBP:%.+]],
    // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGC:%.+]] to i[[sz]]*
    // CHECK-32-DAG: store double* @Gc, double** [[CBP:%.+]],
    // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGC:%.+]] to double**
    // CHECK-DAG: [[GEPGC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-64-DAG: store double [[VALGD]], double* [[CONVGD:%.+]],
    // CHECK-64-DAG: [[CONVGD]] = bitcast i[[sz]]* [[CADDRGD:%.+]] to double*
    // CHECK-64-DAG: [[CVALGD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGD]],
    // CHECK-64-DAG: store i[[sz]] [[CVALGD]], i[[sz]]* [[CBP:%.+]],
    // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGD:%.+]] to i[[sz]]*
    // CHECK-32-DAG: store double* @Gd, double** [[CBP:%.+]],
    // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGD:%.+]] to double**
    // CHECK-DAG: [[GEPGD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK: call i32 @__tgt_target
    // CHECK: call void [[OFFLOADF:@.+]](
    // Capture b, Gb, Sb, Gc, c, Sc, d, Gd, Sd
    #pragma omp target if(Ga>0.0 && a>0 && Sa>0.0)
    {
      b += 1;
      Gb += 1.0;
      Sb += 1.0;

      // CHECK: define internal void [[OFFLOADF]]({{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}})
      // CHECK: call void {{.*}}@__kmpc_fork_call(%ident_t* {{.+}}, i32 {{.+}}, void (i32*, i32*, ...)* bitcast ({{.*}}[[PARF:@.+]] to {{.*}})

      // CHECK: define internal void [[PARF]](i32* noalias %{{.*}}, i32* noalias %{{.*}}, {{.+}}* dereferenceable({{.+}}) %{{.+}}, {{.+}}* dereferenceable({{.+}}) %{{.+}}, {{.+}}* dereferenceable({{.+}}) %{{.+}})
      // Capture d, Gd, Sd
      #pragma omp parallel if(Gc>0.0 && c>0 && Sc>0.0)
      {
        d += 1;
        Gd += 1.0;
        Sd += 1.0;
      }
    }
  }
  return a + b + c + d + (int)Sa + (int)Sb + (int)Sc + (int)Sd;
}

///
/// Tests with template functions.
///

// CHECK: define {{.*}} @{{.*}}tbar2{{.*}}(

// CHECK: define {{.*}} @{{.*}}tbar{{.*}}(
// CHECK-SAME: i16 {{[^,]*}}[[A:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[B:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[C:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[D:%[^,]+]])
// CHECK: [[LA:%.+]] = alloca i16
// CHECK: [[LB:%.+]] = alloca i16
// CHECK: [[LC:%.+]] = alloca i16
// CHECK: [[LD:%.+]] = alloca i16
template<typename T>
int tbar(T a, T b, T c, T d){
  static float Sa = 17.0;
  static float Sb = 18.0;
  static float Sc = 19.0;
  static float Sd = 20.0;

  // CHECK: call void {{.*}}@__kmpc_fork_call(%ident_t* {{.+}}, i32 {{.+}}, void (i32*, i32*, ...)* bitcast ({{.*}}[[PARF:@.+]] to {{.*}}), i16* %{{.+}}, i16* %{{.+}}, i16* %{{.+}}, i16* %{{.+}})
  // CHECK: define internal void [[PARF]](i32* noalias %{{.*}}, i32* noalias %{{.*}}, i16* dereferenceable(2) [[A:%.+]], i16* dereferenceable(2) [[B:%.+]], i16* dereferenceable(2) [[C:%.+]], i16* dereferenceable(2) [[D:%.+]])
  // Capture a, b, c, d
  // CHECK: [[ALLOCLA:%.+]] = alloca i16
  // CHECK: [[ALLOCLB:%.+]] = alloca i16
  // CHECK: [[ALLOCLC:%.+]] = alloca i16
  // CHECK: [[ALLOCLD:%.+]] = alloca i16
  // CHECK: [[LLA:%.+]] = load i16*, i16** [[ALLOCLA]],
  // CHECK: [[LLB:%.+]] = load i16*, i16** [[ALLOCLB]],
  // CHECK: [[LLC:%.+]] = load i16*, i16** [[ALLOCLC]],
  // CHECK: [[LLD:%.+]] = load i16*, i16** [[ALLOCLD]],
  #pragma omp parallel
  {
    // CHECK-DAG:    [[VALLB:%.+]] = load i16, i16* [[LLB]],
    // CHECK-64-DAG: [[VALGB:%.+]] = load double, double* [[GB]],
    // CHECK-DAG:    [[VALFB:%.+]] = load float, float* [[TBB]],
    // CHECK-64-DAG: [[VALGC:%.+]] = load double, double* [[GC]],
    // CHECK-DAG:    [[VALLC:%.+]] = load i16, i16* [[LLC]],
    // CHECK-DAG:    [[VALFC:%.+]] = load float, float* [[TBC]],
    // CHECK-DAG:    [[VALLD:%.+]] = load i16, i16* [[LLD]],
    // CHECK-64-DAG: [[VALGD:%.+]] = load double, double* [[GD]],
    // CHECK-DAG:    [[VALFD:%.+]] = load float, float* [[TBD]],

    // 3 local vars being captured.

    // CHECK-DAG: store i16 [[VALLB]], i16* [[CONVLB:%.+]],
    // CHECK-DAG: [[CONVLB]] = bitcast i[[sz:64|32]]* [[CADDRLB:%.+]] to i16*
    // CHECK-DAG: [[CVALLB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLB]],
    // CHECK-DAG: store i[[sz]] [[CVALLB]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLB:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPLB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-DAG: store i16 [[VALLC]], i16* [[CONVLC:%.+]],
    // CHECK-DAG: [[CONVLC]] = bitcast i[[sz]]* [[CADDRLC:%.+]] to i16*
    // CHECK-DAG: [[CVALLC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLC]],
    // CHECK-DAG: store i[[sz]] [[CVALLC]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLC:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPLC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-DAG: store i16 [[VALLD]], i16* [[CONVLD:%.+]],
    // CHECK-DAG: [[CONVLD]] = bitcast i[[sz]]* [[CADDRLD:%.+]] to i16*
    // CHECK-DAG: [[CVALLD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRLD]],
    // CHECK-DAG: store i[[sz]] [[CVALLD]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPLD:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPLD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // 3 static vars being captured.

    // CHECK-DAG: store float [[VALFB]], float* [[CONVFB:%.+]],
    // CHECK-DAG: [[CONVFB]] = bitcast i[[sz]]* [[CADDRFB:%.+]] to float*
    // CHECK-DAG: [[CVALFB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFB]],
    // CHECK-DAG: store i[[sz]] [[CVALFB]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFB:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPFB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-DAG: store float [[VALFC]], float* [[CONVFC:%.+]],
    // CHECK-DAG: [[CONVFC]] = bitcast i[[sz]]* [[CADDRFC:%.+]] to float*
    // CHECK-DAG: [[CVALFC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFC]],
    // CHECK-DAG: store i[[sz]] [[CVALFC]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFC:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPFC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-DAG: store float [[VALFD]], float* [[CONVFD:%.+]],
    // CHECK-DAG: [[CONVFD]] = bitcast i[[sz]]* [[CADDRFD:%.+]] to float*
    // CHECK-DAG: [[CVALFD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRFD]],
    // CHECK-DAG: store i[[sz]] [[CVALFD]], i[[sz]]* [[CBP:%.+]],
    // CHECK-DAG: [[CBP]] = bitcast i8** [[GEPFD:%.+]] to i[[sz]]*
    // CHECK-DAG: [[GEPFD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // 3 static global vars being captured.

    // CHECK-64-DAG: store double [[VALGB]], double* [[CONVGB:%.+]],
    // CHECK-64-DAG: [[CONVGB]] = bitcast i[[sz]]* [[CADDRGB:%.+]] to double*
    // CHECK-64-DAG: [[CVALGB:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGB]],
    // CHECK-64-DAG: store i[[sz]] [[CVALGB]], i[[sz]]* [[CBP:%.+]],
    // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGB:%.+]] to i[[sz]]*
    // CHECK-32-DAG: store double* @Gb, double** [[CBP:%.+]],
    // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGB:%.+]] to double**
    // CHECK-DAG: [[GEPGB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-64-DAG: store double [[VALGC]], double* [[CONVGC:%.+]],
    // CHECK-64-DAG: [[CONVGC]] = bitcast i[[sz]]* [[CADDRGC:%.+]] to double*
    // CHECK-64-DAG: [[CVALGC:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGC]],
    // CHECK-64-DAG: store i[[sz]] [[CVALGC]], i[[sz]]* [[CBP:%.+]],
    // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGC:%.+]] to i[[sz]]*
    // CHECK-32-DAG: store double* @Gc, double** [[CBP:%.+]],
    // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGC:%.+]] to double**
    // CHECK-DAG: [[GEPGC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK-64-DAG: store double [[VALGD]], double* [[CONVGD:%.+]],
    // CHECK-64-DAG: [[CONVGD]] = bitcast i[[sz]]* [[CADDRGD:%.+]] to double*
    // CHECK-64-DAG: [[CVALGD:%.+]] = load i[[sz]], i[[sz]]* [[CADDRGD]],
    // CHECK-64-DAG: store i[[sz]] [[CVALGD]], i[[sz]]* [[CBP:%.+]],
    // CHECK-64-DAG: [[CBP]] = bitcast i8** [[GEPGD:%.+]] to i[[sz]]*
    // CHECK-32-DAG: store double* @Gd, double** [[CBP:%.+]],
    // CHECK-32-DAG: [[CBP]] = bitcast i8** [[GEPGD:%.+]] to double**
    // CHECK-DAG: [[GEPGD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{[0-8]}}

    // CHECK: call i32 @__tgt_target
    // CHECK: call void [[OFFLOADF:@.+]](
    // Capture b, Gb, Sb, Gc, c, Sc, d, Gd, Sd
    #pragma omp target if(Ga>0.0 && a>0 && Sa>0.0)
    {
      b += 1;
      Gb += 1.0;
      Sb += 1.0;

      // CHECK: define internal void [[OFFLOADF]]({{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}}, {{.+}} {{.*}}%{{.+}})
      // CHECK: call void {{.*}}@__kmpc_fork_call(%ident_t* {{.+}}, i32 {{.+}}, void (i32*, i32*, ...)* bitcast ({{.*}}[[PARF:@.+]] to {{.*}})

      // CHECK: define internal void [[PARF]](i32* noalias %{{.*}}, i32* noalias %{{.*}}, {{.+}}* dereferenceable({{.+}}) %{{.+}}, {{.+}}* dereferenceable({{.+}}) %{{.+}}, {{.+}}* dereferenceable({{.+}}) %{{.+}})
      // Capture d, Gd, Sd
      #pragma omp parallel if(Gc>0.0 && c>0 && Sc>0.0)
      {
        d += 1;
        Gd += 1.0;
        Sd += 1.0;
      }
    }
  }
  return a + b + c + d + (int)Sa + (int)Sb + (int)Sc + (int)Sd;
}

int tbar2(short a, short b, short c, short d){
  return tbar(a, b, c, d);
}

#endif
