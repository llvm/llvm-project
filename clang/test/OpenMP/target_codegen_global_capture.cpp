// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
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
double Ga = 1.0;
double Gb = 2.0;
double Gc = 3.0;
double Gd = 4.0;

// CHECK: define {{.*}} @{{.*}}foo{{.*}}(
// CHECK-SAME: i16 {{[^,]*}}[[A:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[B:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[C:%[^,]+]],
// CHECK-SAME: i16 {{[^,]*}}[[D:%[^,]+]])
// CHECK: [[LA:%.+]] = alloca i16
// CHECK: [[LB:%.+]] = alloca i16
// CHECK: [[LC:%.+]] = alloca i16
// CHECK: [[LD:%.+]] = alloca i16
int foo(short a, short b, short c, short d){
  static float Sa = 5.0;
  static float Sb = 6.0;
  static float Sc = 7.0;
  static float Sd = 8.0;

  // CHECK-DAG: [[REFB:%.+]] = bitcast i16* [[LB]] to i8*
  // CHECK-DAG: store i8* [[REFB]], i8** [[GEPB:%.+]], align
  // CHECK-DAG: [[REFC:%.+]] = bitcast i16* [[LC]] to i8*
  // CHECK-DAG: store i8* [[REFC]], i8** [[GEPC:%.+]], align
  // CHECK-DAG: [[REFD:%.+]] = bitcast i16* [[LD]] to i8*
  // CHECK-DAG: store i8* [[REFD]], i8** [[GEPD:%.+]], align
  // CHECK-DAG: store i8* bitcast (double* [[GB]] to i8*), i8** [[GEPGB:%.+]], align
  // CHECK-DAG: store i8* bitcast (double* [[GC]] to i8*), i8** [[GEPGC:%.+]], align
  // CHECK-DAG: store i8* bitcast (double* [[GD]] to i8*), i8** [[GEPGD:%.+]], align
  // CHECK-DAG: store i8* bitcast (float* [[FB]] to i8*), i8** [[GEPFB:%.+]], align
  // CHECK-DAG: store i8* bitcast (float* [[FC]] to i8*), i8** [[GEPFC:%.+]], align
  // CHECK-DAG: store i8* bitcast (float* [[FD]] to i8*), i8** [[GEPFD:%.+]], align
  // CHECK-DAG: [[GEPB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK-DAG: [[GEPC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK-DAG: [[GEPD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK-DAG: [[GEPGB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK-DAG: [[GEPGC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK-DAG: [[GEPGD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK-DAG: [[GEPFB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK-DAG: [[GEPFC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK-DAG: [[GEPFD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
  // CHECK: call i32 @__tgt_target
  // CHECK: call void [[OFFLOADF:@.+]](
  // Capture b, Gb, Sb, Gc, c, Sc, d, Gd, Sd
  #pragma omp target if(Ga>0.0 && a>0 && Sa>0.0)
  {
    b += 1;
    Gb += 1.0;
    Sb += 1.0;

    // CHECK: define internal void [[OFFLOADF]]({{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}})
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
  #pragma omp parallel
  {
    // CHECK: [[ADRA:%.+]] = alloca i16*, align
    // CHECK: [[ADRB:%.+]] = alloca i16*, align
    // CHECK: [[ADRC:%.+]] = alloca i16*, align
    // CHECK: [[ADRD:%.+]] = alloca i16*, align
    // CHECK: store i16* [[A]], i16** [[ADRA]], align
    // CHECK: store i16* [[B]], i16** [[ADRB]], align
    // CHECK: store i16* [[C]], i16** [[ADRC]], align
    // CHECK: store i16* [[D]], i16** [[ADRD]], align
    // CHECK: [[REFA:%.+]] = load i16*, i16** [[ADRA]],
    // CHECK: [[REFB:%.+]] = load i16*, i16** [[ADRB]],
    // CHECK: [[REFC:%.+]] = load i16*, i16** [[ADRC]],
    // CHECK: [[REFD:%.+]] = load i16*, i16** [[ADRD]],

    // CHECK: load float, float* [[BA]]

    // CHECK-DAG: [[CSTB:%.+]] = bitcast i16* [[REFB]] to i8*
    // CHECK-DAG: [[CSTC:%.+]] = bitcast i16* [[REFC]] to i8*
    // CHECK-DAG: [[CSTD:%.+]] = bitcast i16* [[REFD]] to i8*
    // CHECK-DAG: store i8* [[CSTB]], i8** [[GEPB:%.+]], align
    // CHECK-DAG: store i8* [[CSTC]], i8** [[GEPC:%.+]], align
    // CHECK-DAG: store i8* [[CSTD]], i8** [[GEPD:%.+]], align
    // CHECK-DAG: store i8* bitcast (double* [[GB]] to i8*), i8** [[GEPGB:%.+]], align
    // CHECK-DAG: store i8* bitcast (double* [[GC]] to i8*), i8** [[GEPGC:%.+]], align
    // CHECK-DAG: store i8* bitcast (double* [[GD]] to i8*), i8** [[GEPGD:%.+]], align
    // CHECK-DAG: store i8* bitcast (float* [[BB]] to i8*), i8** [[GEPBB:%.+]], align
    // CHECK-DAG: store i8* bitcast (float* [[BC]] to i8*), i8** [[GEPBC:%.+]], align
    // CHECK-DAG: store i8* bitcast (float* [[BD]] to i8*), i8** [[GEPBD:%.+]], align

    // CHECK-DAG: [[GEPB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK-DAG: [[GEPC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK-DAG: [[GEPD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK-DAG: [[GEPGB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK-DAG: [[GEPGC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK-DAG: [[GEPGD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK-DAG: [[GEPBB]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK-DAG: [[GEPBC]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK-DAG: [[GEPBD]] = getelementptr inbounds [9 x i8*], [9 x i8*]* %{{.+}}, i32 0, i32 {{.+}}
    // CHECK: call i32 @__tgt_target
    // CHECK: call void [[OFFLOADF:@.+]](
    // Capture b, Gb, Sb, Gc, c, Sc, d, Gd, Sd
    #pragma omp target if(Ga>0.0 && a>0 && Sa>0.0)
    {
      b += 1;
      Gb += 1.0;
      Sb += 1.0;

      // CHECK: define internal void [[OFFLOADF]]({{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}}, {{.+}}* {{.*}}%{{.+}})
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

#endif
