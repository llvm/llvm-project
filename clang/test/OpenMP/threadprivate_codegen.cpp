// RUN: %clang_cc1 -verify -fopenmp -fnoopenmp-use-tls -DBODY -triple x86_64-unknown-unknown -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=CHECK,OMP50
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fnoopenmp-use-tls -DBODY -triple x86_64-unknown-unknown -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=CHECK,OMP45

// RUN: %clang_cc1 -verify -fopenmp-simd -fnoopenmp-use-tls -DBODY -triple x86_64-unknown-unknown -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -DBODY -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -verify -fopenmp -DBODY -triple x86_64-unknown-unknown -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=CHECK-TLS,OMP50-TLS
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -DBODY -triple x86_64-unknown-unknown -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=CHECK-TLS,OMP45-TLS
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -DBODY -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefixes=CHECK-TLS,OMP50-TLS %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -DBODY -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefixes=CHECK-TLS,OMP45-TLS %s

// RUN: %clang_cc1 -verify -fopenmp-simd -DBODY -triple x86_64-unknown-unknown -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -DBODY -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s

// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -DBODY -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefixes=CHECK-DEBUG,OMP50-DEBUG %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fnoopenmp-use-tls -DBODY -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefixes=CHECK-DEBUG,OMP45-DEBUG %s

// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
//
#ifndef HEADER
#define HEADER
// CHECK-DAG: [[IDENT:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[S1:%.+]] = type { [[INT:i[0-9]+]] }
// CHECK-DAG: [[S2:%.+]] = type { [[INT]], double }
// CHECK-DAG: [[S3:%.+]] = type { [[INT]], float }
// CHECK-DAG: [[S4:%.+]] = type { [[INT]], [[INT]] }
// CHECK-DAG: [[S5:%.+]] = type { [[INT]], [[INT]], [[INT]] }
// CHECK-DAG: [[SMAIN:%.+]] = type { [[INT]], double, double }
// CHECK-DEBUG-DAG: [[IDENT:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DEBUG-DAG: [[S1:%.+]] = type { [[INT:i[0-9]+]] }
// CHECK-DEBUG-DAG: [[S2:%.+]] = type { [[INT]], double }
// CHECK-DEBUG-DAG: [[S3:%.+]] = type { [[INT]], float }
// CHECK-DEBUG-DAG: [[S4:%.+]] = type { [[INT]], [[INT]] }
// CHECK-DEBUG-DAG: [[S5:%.+]] = type { [[INT]], [[INT]], [[INT]] }
// CHECK-DEBUG-DAG: [[SMAIN:%.+]] = type { [[INT]], double, double }
// CHECK-TLS-DAG: [[S1:%.+]] = type { [[INT:i[0-9]+]] }
// CHECK-TLS-DAG: [[S2:%.+]] = type { [[INT]], double }
// CHECK-TLS-DAG: [[S3:%.+]] = type { [[INT]], float }
// CHECK-TLS-DAG: [[S4:%.+]] = type { [[INT]], [[INT]] }
// CHECK-TLS-DAG: [[S5:%.+]] = type { [[INT]], [[INT]], [[INT]] }
// CHECK-TLS-DAG: [[SMAIN:%.+]] = type { [[INT]], double, double }

struct S1 {
  int a;
  S1()
      : a(0) {
  }
  S1(int a)
      : a(a) {
  }
  S1(const S1 &s) {
    a = 12 + s.a;
  }
  ~S1() {
    a = 0;
  }
};

struct S2 {
  int a;
  double b;
  S2()
      : a(0) {
  }
  S2(int a)
      : a(a) {
  }
  S2(const S2 &s) {
    a = 12 + s.a;
  }
  ~S2() {
    a = 0;
  }
};

struct S3 {
  int a;
  float b;
  S3()
      : a(0) {
  }
  S3(int a)
      : a(a) {
  }
  S3(const S3 &s) {
    a = 12 + s.a;
  }
  ~S3() {
    a = 0;
  }
};

struct S4 {
  int a, b;
  S4()
      : a(0) {
  }
  S4(int a)
      : a(a) {
  }
  S4(const S4 &s) {
    a = 12 + s.a;
  }
  ~S4() {
    a = 0;
  }
};

struct S5 {
  int a, b, c;
  S5()
      : a(0) {
  }
  S5(int a)
      : a(a) {
  }
  S5(const S5 &s) {
    a = 12 + s.a;
  }
  ~S5() {
    a = 0;
  }
};

// CHECK-DAG:  [[GS1:@.+]] = internal global [[S1]] zeroinitializer
// CHECK-DAG:  [[GS1]].cache. = common global i8** null
// CHECK-DAG:  [[DEFAULT_LOC:@.+]] = private unnamed_addr constant [[IDENT]] { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* {{@.+}}, i32 0, i32 0) }
// CHECK-DAG:  [[GS2:@.+]] = internal global [[S2]] zeroinitializer
// CHECK-DAG:  [[ARR_X:@.+]] = global [2 x [3 x [[S1]]]] zeroinitializer
// CHECK-DAG:  [[ARR_X]].cache. = common global i8** null
// CHECK-DAG:  [[SM:@.+]] = internal global [[SMAIN]] zeroinitializer
// CHECK-DAG:  [[SM]].cache. = common global i8** null
// CHECK-DAG:  [[STATIC_S:@.+]] = external global [[S3]]
// CHECK-DAG:  [[STATIC_S]].cache. = common global i8** null
// CHECK-DAG:  [[GS3:@.+]] = external global [[S5]]
// CHECK-DAG:  [[GS3]].cache. = common global i8** null
// CHECK-DAG:  [[ST_INT_ST:@.+]] = linkonce_odr global i32 23
// CHECK-DAG:  [[ST_INT_ST]].cache. = common global i8** null
// CHECK-DAG:  [[ST_FLOAT_ST:@.+]] = linkonce_odr global float 2.300000e+01
// CHECK-DAG:  [[ST_FLOAT_ST]].cache. = common global i8** null
// CHECK-DAG:  [[ST_S4_ST:@.+]] = linkonce_odr global %struct.S4 zeroinitializer
// CHECK-DAG:  [[ST_S4_ST]].cache. = common global i8** null
// CHECK-NOT:  .cache. = common global i8** null
// There is no cache for gs2 - it is not threadprivate. Check that there is only
// 8 caches created (for Static::s, gs1, gs3, arr_x, main::sm, ST<int>::st,
// ST<float>::st, ST<S4>::st)
// CHECK-DEBUG-DAG: [[GS1:@.+]] = internal global [[S1]] zeroinitializer
// CHECK-DEBUG-DAG: [[GS2:@.+]] = internal global [[S2]] zeroinitializer
// CHECK-DEBUG-DAG: [[ARR_X:@.+]] = global [2 x [3 x [[S1]]]] zeroinitializer
// CHECK-DEBUG-DAG: [[SM:@.+]] = internal global [[SMAIN]] zeroinitializer
// CHECK-DEBUG-DAG: [[STATIC_S:@.+]] = external global [[S3]]
// CHECK-DEBUG-DAG: [[GS3:@.+]] = external global [[S5]]
// CHECK-DEBUG-DAG: [[ST_INT_ST:@.+]] = linkonce_odr global i32 23
// CHECK-DEBUG-DAG: [[ST_FLOAT_ST:@.+]] = linkonce_odr global float 2.300000e+01
// CHECK-DEBUG-DAG: [[ST_S4_ST:@.+]] = linkonce_odr global %struct.S4 zeroinitializer

// CHECK-DEBUG-DAG: [[LOC1:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;;249;1;;\00"
// CHECK-DEBUG-DAG: [[ID1:@.*]] = private unnamed_addr constant %struct.ident_t { {{.*}} [[LOC1]]
// CHECK-DEBUG-DAG: [[LOC2:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;;304;1;;\00"
// CHECK-DEBUG-DAG: [[ID2:@.*]] = private unnamed_addr constant %struct.ident_t { {{.*}} [[LOC2]]
// CHECK-DEBUG-DAG: [[LOC3:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;;422;19;;\00"
// CHECK-DEBUG-DAG: [[LOC4:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;459;1;;\00"
// CHECK-DEBUG-DAG: [[LOC5:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;476;9;;\00"
// CHECK-DEBUG-DAG: [[LOC6:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;498;10;;\00"
// CHECK-DEBUG-DAG: [[LOC7:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;521;10;;\00"
// CHECK-DEBUG-DAG: [[LOC8:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;557;10;;\00"
// CHECK-DEBUG-DAG: [[LOC9:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;586;10;;\00"
// CHECK-DEBUG-DAG: [[LOC10:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;606;10;;\00"
// CHECK-DEBUG-DAG: [[LOC11:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;629;27;;\00"
// CHECK-DEBUG-DAG: [[LOC12:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;main;652;10;;\00"
// CHECK-DEBUG-DAG: [[LOC13:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;foobar;774;9;;\00"
// CHECK-DEBUG-DAG: [[LOC14:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;foobar;797;10;;\00"
// CHECK-DEBUG-DAG: [[LOC15:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;foobar;833;10;;\00"
// CHECK-DEBUG-DAG: [[LOC16:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;foobar;862;10;;\00"
// CHECK-DEBUG-DAG: [[LOC17:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;foobar;882;10;;\00"
// CHECK-DEBUG-DAG: [[LOC18:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;foobar;905;27;;\00"
// CHECK-DEBUG-DAG: [[LOC19:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;foobar;928;10;;\00"
// CHECK-DEBUG-DAG: [[LOC20:@.*]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";{{.*}}threadprivate_codegen.cpp;;363;1;;\00"

// CHECK-TLS-DAG:  [[GS1:@.+]] = internal thread_local global [[S1]] zeroinitializer
// CHECK-TLS-DAG:  [[GS2:@.+]] = internal global [[S2]] zeroinitializer
// CHECK-TLS-DAG:  [[ARR_X:@.+]] = thread_local global [2 x [3 x [[S1]]]] zeroinitializer
// CHECK-TLS-DAG:  [[SM:@.+]] = internal thread_local global [[SMAIN]] zeroinitializer
// CHECK-TLS-DAG:  [[SM_GUARD:@_ZGVZ4mainE2sm]] = internal thread_local global i8 0
// CHECK-TLS-DAG:  [[STATIC_S:@.+]] = external thread_local global [[S3]]
// CHECK-TLS-DAG:  [[GS3:@.+]] = external thread_local global [[S5]]
// CHECK-TLS-DAG:  [[ST_INT_ST:@.+]] = linkonce_odr thread_local global i32 23
// CHECK-TLS-DAG:  [[ST_FLOAT_ST:@.+]] = linkonce_odr thread_local global float 2.300000e+01
// CHECK-TLS-DAG:  [[ST_S4_ST:@.+]] = linkonce_odr thread_local global %struct.S4 zeroinitializer
// CHECK-TLS-DAG:  [[ST_S4_ST_GUARD:@_ZGVN2STI2S4E2stE]] = linkonce_odr thread_local global i64 0
// CHECK-TLS-DAG:  @__tls_guard = internal thread_local global i8 0
// CHECK-TLS-DAG:  @__dso_handle = external hidden global i8
// CHECK-TLS-DAG:  [[GS1_TLS_INIT:@_ZTHL3gs1]] = internal alias void (), void ()* @__tls_init
// CHECK-TLS-DAG:  [[ARR_X_TLS_INIT:@_ZTH5arr_x]] = alias void (), void ()* @__tls_init
// CHECK-TLS-DAG:  [[ST_S4_ST_TLS_INIT:@_ZTHN2STI2S4E2stE]] = linkonce_odr alias void (), void ()* [[ST_S4_ST_CXX_INIT:@[^, ]*]]

// OMP50-TLS: define internal void [[GS1_CXX_INIT:@.*]]()
// OMP50-TLS: call void [[GS1_CTOR1:@.*]]([[S1]]* [[GS1]], i32 5)
// OMP50-TLS: call i32 @__cxa_thread_atexit(void (i8*)* bitcast (void ([[S1]]*)* [[GS1_DTOR1:.*]] to void (i8*)*), i8* bitcast ([[S1]]* [[GS1]] to i8*)
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[GS1_CTOR1]]([[S1]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: call void [[GS1_CTOR2:@.*]]([[S1]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[GS1_DTOR1]]([[S1]]* {{.*}})
// OMP50-TLS: call void [[GS1_DTOR2:@.*]]([[S1]]* {{.*}})
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[GS1_CTOR2]]([[S1]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: define {{.*}}void [[GS1_DTOR2]]([[S1]]* {{.*}})

// OMP50-TLS: define internal void [[GS2_CXX_INIT:@.*]]()
// OMP50-TLS: call void [[GS2_CTOR1:@.*]]([[S2]]* [[GS2]], i32 27)
// OMP50-TLS: call i32 @__cxa_atexit(void (i8*)* bitcast (void ([[S2]]*)* [[GS2_DTOR1:.*]] to void (i8*)*), i8* bitcast ([[S2]]* [[GS2]] to i8*)
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[GS2_CTOR1]]([[S2]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: call void [[GS2_CTOR2:@.*]]([[S2]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[GS2_DTOR1]]([[S2]]* {{.*}})
// OMP50-TLS: call void [[GS2_DTOR2:@.*]]([[S2]]* {{.*}})
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[GS2_CTOR2]]([[S2]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: define {{.*}}void [[GS2_DTOR2]]([[S2]]* {{.*}})

// OMP50-TLS: define internal void [[ARR_X_CXX_INIT:@.*]]()
// OMP50-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 0, i{{.*}} 0), i{{.*}} 1)
// OMP50-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 0, i{{.*}} 1), i{{.*}} 2)
// OMP50-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 0, i{{.*}} 2), i{{.*}} 3)
// OMP50-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 1, i{{.*}} 0), i{{.*}} 4)
// OMP50-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 1, i{{.*}} 1), i{{.*}} 5)
// OMP50-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 1, i{{.*}} 2), i{{.*}} 6)
// OMP50-TLS: call i32 @__cxa_thread_atexit(void (i8*)* [[ARR_X_CXX_DTOR:@[^,]+]]
// OMP50-TLS: define internal void [[ARR_X_CXX_DTOR]](i8* %0)
// OMP50-TLS: void [[GS1_DTOR1]]([[S1]]* {{.*}})

struct Static {
  static S3 s;
#pragma omp threadprivate(s)
};

static S1 gs1(5);
#pragma omp threadprivate(gs1)
#pragma omp threadprivate(gs1)
// CHECK:      define {{.*}} [[S1_CTOR:@.*]]([[S1]]* {{.*}},
// CHECK:      define {{.*}} [[S1_DTOR:@.*]]([[S1]]* {{.*}})
// CHECK:      define internal {{.*}}i8* [[GS1_CTOR:@\.__kmpc_global_ctor_\..*]](i8* %0)
// CHECK:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[S1]]*
// CHECK-NEXT: call {{.*}} [[S1_CTOR]]([[S1]]* [[RES]], {{.*}} 5)
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK:      ret i8* [[ARG]]
// CHECK-NEXT: }
// CHECK:      define internal {{.*}}void [[GS1_DTOR:@\.__kmpc_global_dtor_\..*]](i8* %0)
// CHECK:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[S1]]*
// CHECK-NEXT: call {{.*}} [[S1_DTOR]]([[S1]]* [[RES]])
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK:      define internal {{.*}}void [[GS1_INIT:@\.__omp_threadprivate_init_\..*]]()
// CHECK:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* [[DEFAULT_LOC]], i8* bitcast ([[S1]]* [[GS1]] to i8*), i8* (i8*)* [[GS1_CTOR]], i8* (i8*, i8*)* null, void (i8*)* [[GS1_DTOR]])
// CHECK-NEXT: ret void
// CHECK-NEXT: }



// CHECK-DEBUG:      @__kmpc_global_thread_num
// CHECK-DEBUG:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* [[ID1]], i8* bitcast ([[S1]]* [[GS1]] to i8*), i8* (i8*)* [[GS1_CTOR:@\.__kmpc_global_ctor_\..*]], i8* (i8*, i8*)* null, void (i8*)* [[GS1_DTOR:@\.__kmpc_global_dtor_\..*]])
// CHECK-DEBUG:      define internal {{.*}}i8* [[GS1_CTOR]](i8* %0)
// CHECK-DEBUG:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK-DEBUG:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK-DEBUG:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[S1]]*
// CHECK-DEBUG-NEXT: call {{.*}} [[S1_CTOR:@.+]]([[S1]]* [[RES]], {{.*}} 5){{.*}}, !dbg
// CHECK-DEBUG:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK-DEBUG:      ret i8* [[ARG]]
// CHECK-DEBUG-NEXT: }
// CHECK-DEBUG:      define {{.*}} [[S1_CTOR]]([[S1]]* {{.*}},
// CHECK-DEBUG:      define internal {{.*}}void [[GS1_DTOR]](i8* %0)
// CHECK-DEBUG:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK-DEBUG:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK-DEBUG:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[S1]]*
// CHECK-DEBUG-NEXT: call {{.*}} [[S1_DTOR:@.+]]([[S1]]* [[RES]]){{.*}}, !dbg
// CHECK-DEBUG-NEXT: ret void
// CHECK-DEBUG-NEXT: }
// CHECK-DEBUG:      define {{.*}} [[S1_DTOR]]([[S1]]* {{.*}})
static S2 gs2(27);
// CHECK:      define {{.*}} [[S2_CTOR:@.*]]([[S2]]* {{.*}},
// CHECK:      define {{.*}} [[S2_DTOR:@.*]]([[S2]]* {{.*}})
// No another call for S2 constructor because it is not threadprivate
// CHECK-NOT:  call {{.*}} [[S2_CTOR]]([[S2]]*
// CHECK-DEBUG:      define {{.*}} [[S2_CTOR:@.*]]([[S2]]* {{.*}},
// CHECK-DEBUG:      define {{.*}} [[S2_DTOR:@.*]]([[S2]]* {{.*}})
// No another call for S2 constructor because it is not threadprivate
// CHECK-DEBUG-NOT:  call {{.*}} [[S2_CTOR]]([[S2]]*
S1 arr_x[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
#pragma omp threadprivate(arr_x)
// CHECK:      define internal {{.*}}i8* [[ARR_X_CTOR:@\.__kmpc_global_ctor_\..*]](i8* %0)
// CHECK:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK:      [[RES:%.*]] = bitcast i8* [[ARG]] to [2 x [3 x [[S1]]]]*
// CHECK:      [[ARR1:%.*]] = getelementptr inbounds [2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[RES]], i{{.*}} 0, i{{.*}} 0
// CHECK:      [[ARR:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR1]], i{{.*}} 0, i{{.*}} 0
// CHECK:      invoke {{.*}} [[S1_CTOR]]([[S1]]* [[ARR]], [[INT]] {{.*}}1)
// CHECK:      [[ARR_ELEMENT:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR]], i{{.*}} 1
// CHECK:      invoke {{.*}} [[S1_CTOR]]([[S1]]* [[ARR_ELEMENT]], [[INT]] {{.*}}2)
// CHECK:      [[ARR_ELEMENT2:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_ELEMENT]], i{{.*}} 1
// CHECK:      invoke {{.*}} [[S1_CTOR]]([[S1]]* [[ARR_ELEMENT2]], [[INT]] {{.*}}3)
// CHECK:      [[ARR_ELEMENT3:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR1]], i{{.*}} 1
// CHECK:      [[ARR_:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR_ELEMENT3]], i{{.*}} 0, i{{.*}} 0
// CHECK:      invoke {{.*}} [[S1_CTOR]]([[S1]]* [[ARR_]], [[INT]] {{.*}}4)
// CHECK:      [[ARR_ELEMENT:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_]], i{{.*}} 1
// CHECK:      invoke {{.*}} [[S1_CTOR]]([[S1]]* [[ARR_ELEMENT]], [[INT]] {{.*}}5)
// CHECK:      [[ARR_ELEMENT2:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_ELEMENT]], i{{.*}} 1
// CHECK:      invoke {{.*}} [[S1_CTOR]]([[S1]]* [[ARR_ELEMENT2]], [[INT]] {{.*}}6)
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK:      ret i8* [[ARG]]
// CHECK:      }
// CHECK:      define internal {{.*}}void [[ARR_X_DTOR:@\.__kmpc_global_dtor_\..*]](i8* %0)
// CHECK:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK:      [[ARR_BEGIN:%.*]] = bitcast i8* [[ARG]] to [[S1]]*
// CHECK-NEXT: [[ARR_CUR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_BEGIN]], i{{.*}} 6
// CHECK-NEXT: br label %[[ARR_LOOP:.*]]
// CHECK:      {{.*}}[[ARR_LOOP]]{{.*}}
// CHECK-NEXT: [[ARR_ELEMENTPAST:%.*]] = phi [[S1]]* [ [[ARR_CUR]], {{.*}} ], [ [[ARR_ELEMENT:%.*]], {{.*}} ]
// CHECK-NEXT: [[ARR_ELEMENT:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_ELEMENTPAST]], i{{.*}} -1
// CHECK-NEXT: {{call|invoke}} {{.*}} [[S1_DTOR]]([[S1]]* [[ARR_ELEMENT]])
// CHECK:      [[ARR_DONE:%.*]] = icmp eq [[S1]]* [[ARR_ELEMENT]], [[ARR_BEGIN]]
// CHECK-NEXT: br i1 [[ARR_DONE]], label %[[ARR_EXIT:.*]], label %[[ARR_LOOP]]
// CHECK:      {{.*}}[[ARR_EXIT]]{{.*}}
// CHECK-NEXT: ret void
// CHECK:      }
// CHECK:      define internal {{.*}}void [[ARR_X_INIT:@\.__omp_threadprivate_init_\..*]]()
// CHECK:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* [[DEFAULT_LOC]], i8* bitcast ([2 x [3 x [[S1]]]]* [[ARR_X]] to i8*), i8* (i8*)* [[ARR_X_CTOR]], i8* (i8*, i8*)* null, void (i8*)* [[ARR_X_DTOR]])
// CHECK-NEXT: ret void
// CHECK-NEXT: }



// CHECK-DEBUG:      @__kmpc_global_thread_num
// CHECK-DEBUG:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* [[ID2]], i8* bitcast ([2 x [3 x [[S1]]]]* [[ARR_X]] to i8*), i8* (i8*)* [[ARR_X_CTOR:@\.__kmpc_global_ctor_\..*]], i8* (i8*, i8*)* null, void (i8*)* [[ARR_X_DTOR:@\.__kmpc_global_dtor_\..*]])
// CHECK-DEBUG:      define internal {{.*}}i8* [[ARR_X_CTOR]](i8* %0)
// CHECK-DEBUG:      }
// CHECK-DEBUG:      define internal {{.*}}void [[ARR_X_DTOR]](i8* %0)
// CHECK-DEBUG:      }
extern S5 gs3;
#pragma omp threadprivate(gs3)
// No call for S5 constructor because gs3 has just declaration, not a definition.
// CHECK-NOT:  call {{.*}}([[S5]]*
// CHECK-DEBUG-NOT:  call {{.*}}([[S5]]*

template <class T>
struct ST {
  static T st;
#pragma omp threadprivate(st)
};





// OMP50-DEBUG:      @__kmpc_global_thread_num
// OMP50-DEBUG:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* {{.*}}, i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*), i8* (i8*)* [[ST_S4_ST_CTOR:@\.__kmpc_global_ctor_\..+]], i8* (i8*, i8*)* null, void (i8*)* [[ST_S4_ST_DTOR:@\.__kmpc_global_dtor_\..+]])
// OMP50-DEBUG:      define internal {{.*}}i8* [[ST_S4_ST_CTOR]](i8* %0)
// OMP50-DEBUG:      }
// OMP50-DEBUG:      define {{.*}} [[S4_CTOR:@.*]]([[S4]]* {{.*}},
// OMP50-DEBUG:      define internal {{.*}}void [[ST_S4_ST_DTOR]](i8* %0)
// OMP50-DEBUG:      }
// OMP50-DEBUG:      define {{.*}} [[S4_DTOR:@.*]]([[S4]]* {{.*}})

// OMP50:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* [[DEFAULT_LOC]], i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*), i8* (i8*)* [[ST_S4_ST_CTOR:@\.__kmpc_global_ctor_\..+]], i8* (i8*, i8*)* null, void (i8*)* [[ST_S4_ST_DTOR:@\.__kmpc_global_dtor_\..+]])
// OMP50:      define internal {{.*}}i8* [[ST_S4_ST_CTOR]](i8* %0)
// OMP50:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// OMP50:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// OMP50:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[S4]]*
// OMP50-NEXT: call {{.*}} [[S4_CTOR:@.+]]([[S4]]* [[RES]], {{.*}} 23)
// OMP50:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// OMP50-NEXT: ret i8* [[ARG]]
// OMP50-NEXT: }
// OMP50:      define {{.*}} [[S4_CTOR]]([[S4]]* {{.*}},
// OMP50:      define internal {{.*}}void [[ST_S4_ST_DTOR]](i8* %0)
// OMP50:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// OMP50:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// OMP50:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[S4]]*
// OMP50-NEXT: call {{.*}} [[S4_DTOR:@.+]]([[S4]]* [[RES]])
// OMP50-NEXT: ret void
// OMP50-NEXT: }
// OMP50:      define {{.*}} [[S4_DTOR]]([[S4]]* {{.*}})
template <class T>
T ST<T>::st(23);

// CHECK-LABEL:  @main()
// CHECK-DEBUG-LABEL: @main()
int main() {

  int Res;
  struct Smain {
    int a;
    double b, c;
    Smain()
        : a(0) {
    }
    Smain(int a)
        : a(a) {
    }
    Smain(const Smain &s) {
      a = 12 + s.a;
    }
    ~Smain() {
      a = 0;
    }
  };

  static Smain sm(gs1.a);
// CHECK:      [[THREAD_NUM:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT]]* [[DEFAULT_LOC]])
// CHECK:      call {{.*}}i{{.*}} @__cxa_guard_acquire
// CHECK:      call {{.*}}i32 @__kmpc_global_thread_num([[IDENT]]* [[DEFAULT_LOC]])
// CHECK:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* [[DEFAULT_LOC]], i8* bitcast ([[SMAIN]]* [[SM]] to i8*), i8* (i8*)* [[SM_CTOR:@\.__kmpc_global_ctor_\..+]], i8* (i8*, i8*)* null, void (i8*)* [[SM_DTOR:@\.__kmpc_global_dtor_\..+]])
// CHECK:      [[GS1_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S1]]* [[GS1]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[GS1]].cache.)
// CHECK-NEXT: [[GS1_ADDR:%.*]] = bitcast i8* [[GS1_TEMP_ADDR]] to [[S1]]*
// CHECK-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
// CHECK-NEXT: [[GS1_A:%.*]] = load [[INT]], [[INT]]* [[GS1_A_ADDR]]
// CHECK-NEXT: invoke {{.*}} [[SMAIN_CTOR:.*]]([[SMAIN]]* [[SM]], [[INT]] {{.*}}[[GS1_A]])
// CHECK:      call {{.*}}void @__cxa_guard_release



// CHECK-DEBUG:      call {{.*}}i{{.*}} @__cxa_guard_acquire
// CHECK-DEBUG:      call {{.*}}i32 @__kmpc_global_thread_num([[IDENT]]* [[KMPC_LOC:@.+]])
// CHECK-DEBUG:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* [[KMPC_LOC]], i8* bitcast ([[SMAIN]]* [[SM]] to i8*), i8* (i8*)* [[SM_CTOR:@\.__kmpc_global_ctor_\..+]], i8* (i8*, i8*)* null, void (i8*)* [[SM_DTOR:@\.__kmpc_global_dtor_\..+]])
// CHECK-DEBUG:      [[GS1_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S1]]* [[GS1]] to i8*), i{{.*}} {{[0-9]+}}, i8***


// CHECK-DEBUG-NEXT: [[GS1_ADDR:%.*]] = bitcast i8* [[GS1_TEMP_ADDR]] to [[S1]]*
// CHECK-DEBUG-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
// CHECK-DEBUG-NEXT: [[GS1_A:%.*]] = load [[INT]], [[INT]]* [[GS1_A_ADDR]]
// CHECK-DEBUG-NEXT: invoke {{.*}} [[SMAIN_CTOR:.*]]([[SMAIN]]* [[SM]], [[INT]] {{.*}}[[GS1_A]])
// CHECK-DEBUG:      call {{.*}}void @__cxa_guard_release
// CHECK-TLS:      [[IS_INIT_INT:%.*]] = load i8, i8* [[SM_GUARD]]
// CHECK-TLS-NEXT: [[IS_INIT_BOOL:%.*]] = icmp eq i8 [[IS_INIT_INT]], 0
// CHECK-TLS-NEXT: br i1 [[IS_INIT_BOOL]], label %[[INIT_LABEL:.*]], label %[[INIT_DONE:[^,]+]]{{.*}}
// CHECK-TLS:      [[INIT_LABEL]]
// CHECK-TLS-NEXT: [[GS1_ADDR:%.*]] = call [[S1]]* [[GS1_TLS_INITD:@[^,]+]]
// CHECK-TLS-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i32 0, i32 0
// CHECK-TLS-NEXT: [[GS1_A_VAL:%.*]] = load i32, i32* [[GS1_A_ADDR]]
// CHECK-TLS-NEXT: call void [[SM_CTOR1:@.*]]([[SMAIN]]* [[SM]], i32 [[GS1_A_VAL]])
// CHECK-TLS-NEXT: call i32 @__cxa_thread_atexit(void (i8*)* bitcast (void ([[SMAIN]]*)* [[SM_DTOR1:@.*]] to void (i8*)*), i8* bitcast ([[SMAIN]]* [[SM]] to i8*), i8* @__dso_handle)
// CHECK-TLS-NEXT: store i8 1, i8* [[SM_GUARD]]
// CHECK-TLS-NEXT: br label %[[INIT_DONE]]
// CHECK-TLS:      [[INIT_DONE]]
#pragma omp threadprivate(sm)
  // CHECK:      [[STATIC_S_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S3]]* [[STATIC_S]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[STATIC_S]].cache.)
  // CHECK-NEXT: [[STATIC_S_ADDR:%.*]] = bitcast i8* [[STATIC_S_TEMP_ADDR]] to [[S3]]*
  // CHECK-NEXT: [[STATIC_S_A_ADDR:%.*]] = getelementptr inbounds [[S3]], [[S3]]* [[STATIC_S_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[STATIC_S_A:%.*]] = load [[INT]], [[INT]]* [[STATIC_S_A_ADDR]]
  // CHECK-NEXT: store [[INT]] [[STATIC_S_A]], [[INT]]* [[RES_ADDR:[^,]+]]
  // CHECK-DEBUG:[[STATIC_S_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S3]]* [[STATIC_S]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[STATIC_S_ADDR:%.*]] = bitcast i8* [[STATIC_S_TEMP_ADDR]] to [[S3]]*
  // CHECK-DEBUG-NEXT: [[STATIC_S_A_ADDR:%.*]] = getelementptr inbounds [[S3]], [[S3]]* [[STATIC_S_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[STATIC_S_A:%.*]] = load [[INT]], [[INT]]* [[STATIC_S_A_ADDR]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[STATIC_S_A]], [[INT]]* [[RES_ADDR:[^,]+]]
  // CHECK-TLS:      [[STATIC_S_ADDR:%.*]] = call [[S3]]* [[STATIC_S_TLS_INITD:@[^,]+]]
  // CHECK-TLS-NEXT: [[STATIC_S_A_ADDR:%.*]] = getelementptr inbounds [[S3]], [[S3]]* [[STATIC_S_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT: [[STATIC_S_A:%.*]] = load i32, i32* [[STATIC_S_A_ADDR]]
  // CHECK-TLS-NEXT: store i32 [[STATIC_S_A]], i32* [[RES_ADDR:[^,]+]]
  Res = Static::s.a;
  // CHECK:      [[SM_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[SMAIN]]* [[SM]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[SM]].cache.)
  // CHECK-NEXT: [[SM_ADDR:%.*]] = bitcast i8* [[SM_TEMP_ADDR]] to [[SMAIN]]*
  // CHECK-NEXT: [[SM_A_ADDR:%.*]] = getelementptr inbounds [[SMAIN]], [[SMAIN]]* [[SM_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[SM_A:%.*]] = load [[INT]], [[INT]]* [[SM_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[SM_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[SM_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[SMAIN]]* [[SM]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[SM_ADDR:%.*]] = bitcast i8* [[SM_TEMP_ADDR]] to [[SMAIN]]*
  // CHECK-DEBUG-NEXT: [[SM_A_ADDR:%.*]] = getelementptr inbounds [[SMAIN]], [[SMAIN]]* [[SM_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[SM_A:%.*]] = load [[INT]], [[INT]]* [[SM_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[SM_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // [[SM]] was initialized already, so it can be used directly
  // CHECK-TLS:      [[SM_A:%.*]] = load i32, i32* getelementptr inbounds ([[SMAIN]], [[SMAIN]]* [[SM]], i{{.*}} 0, i{{.*}} 0)
  // CHECK-TLS-NEXT: [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add {{.*}} i32 [[RES]], [[SM_A]]
  // CHECK-TLS-NEXT: store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += sm.a;
  // CHECK:      [[GS1_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S1]]* [[GS1]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[GS1]].cache.)
  // CHECK-NEXT: [[GS1_ADDR:%.*]] = bitcast i8* [[GS1_TEMP_ADDR]] to [[S1]]*
  // CHECK-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[GS1_A:%.*]] = load [[INT]], [[INT]]* [[GS1_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS1_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[GS1_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S1]]* [[GS1]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[GS1_ADDR:%.*]] = bitcast i8* [[GS1_TEMP_ADDR]] to [[S1]]*
  // CHECK-DEBUG-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[GS1_A:%.*]] = load [[INT]], [[INT]]* [[GS1_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS1_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:      [[GS1_ADDR:%.*]] = call [[S1]]* [[GS1_TLS_INITD]]
  // CHECK-TLS-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT: [[GS1_A:%.*]] = load i32, i32* [[GS1_A_ADDR]]
  // CHECK-TLS-NEXT: [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add {{.*}} i32 [[RES]], [[GS1_A]]
  // CHECK-TLS-NEXT: store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += gs1.a;
  // CHECK:      [[GS2_A:%.*]] = load [[INT]], [[INT]]* getelementptr inbounds ([[S2]], [[S2]]* [[GS2]], i{{.*}} 0, i{{.*}} 0)
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS2_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG:      [[GS2_A:%.*]] = load [[INT]], [[INT]]* getelementptr inbounds ([[S2]], [[S2]]* [[GS2]], i{{.*}} 0, i{{.*}} 0)
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS2_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:      [[GS2_A:%.*]] = load [[INT]], [[INT]]* getelementptr inbounds ([[S2]], [[S2]]* [[GS2]], i{{.*}} 0, i{{.*}} 0)
  // CHECK-TLS-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS2_A]]
  // CHECK-TLS-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  Res += gs2.a;
  // CHECK:      [[GS3_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S5]]* [[GS3]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[GS3]].cache.)
  // CHECK-NEXT: [[GS3_ADDR:%.*]] = bitcast i8* [[GS3_TEMP_ADDR]] to [[S5]]*
  // CHECK-NEXT: [[GS3_A_ADDR:%.*]] = getelementptr inbounds [[S5]], [[S5]]* [[GS3_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[GS3_A:%.*]] = load [[INT]], [[INT]]* [[GS3_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS3_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[GS3_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S5]]* [[GS3]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[GS3_ADDR:%.*]] = bitcast i8* [[GS3_TEMP_ADDR]] to [[S5]]*
  // CHECK-DEBUG-NEXT: [[GS3_A_ADDR:%.*]] = getelementptr inbounds [[S5]], [[S5]]* [[GS3_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[GS3_A:%.*]] = load [[INT]], [[INT]]* [[GS3_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS3_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:      [[GS3_ADDR:%.*]] = call [[S5]]* [[GS3_TLS_INITD:[^,]+]]
  // CHECK-TLS-NEXT: [[GS3_A_ADDR:%.*]] = getelementptr inbounds [[S5]], [[S5]]* [[GS3_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT: [[GS3_A:%.*]] = load i32, i32* [[GS3_A_ADDR]]
  // CHECK-TLS-NEXT: [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add nsw i32 [[RES]], [[GS3_A]]
  // CHECK-TLS-NEXT: store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += gs3.a;
  // CHECK:      [[ARR_X_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([2 x [3 x [[S1]]]]* [[ARR_X]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[ARR_X]].cache.)
  // CHECK-NEXT: [[ARR_X_ADDR:%.*]] = bitcast i8* [[ARR_X_TEMP_ADDR]] to [2 x [3 x [[S1]]]]*
  // CHECK-NEXT: [[ARR_X_1_ADDR:%.*]] = getelementptr inbounds [2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-NEXT: [[ARR_X_1_1_ADDR:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR_X_1_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-NEXT: [[ARR_X_1_1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_X_1_1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[ARR_X_1_1_A:%.*]] = load [[INT]], [[INT]]* [[ARR_X_1_1_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ARR_X_1_1_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT:      [[ARR_X_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([2 x [3 x [[S1]]]]* [[ARR_X]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[ARR_X_ADDR:%.*]] = bitcast i8* [[ARR_X_TEMP_ADDR]] to [2 x [3 x [[S1]]]]*
  // CHECK-DEBUG-NEXT: [[ARR_X_1_ADDR:%.*]] = getelementptr inbounds [2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-DEBUG-NEXT: [[ARR_X_1_1_ADDR:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR_X_1_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-DEBUG-NEXT: [[ARR_X_1_1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_X_1_1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[ARR_X_1_1_A:%.*]] = load [[INT]], [[INT]]* [[ARR_X_1_1_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ARR_X_1_1_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:       [[ARR_X_ADDR:%.*]] = call [2 x [3 x [[S1]]]]* [[ARR_X_TLS_INITD:[^,]+]]
  // CHECK-TLS-NEXT:  [[ARR_X_1_ADDR:%.*]] = getelementptr inbounds [2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-TLS-NEXT:  [[ARR_X_1_1_ADDR:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR_X_1_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-TLS-NEXT:  [[ARR_X_1_1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_X_1_1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT:  [[ARR_X_1_1_A:%.*]] = load i32, i32* [[ARR_X_1_1_A_ADDR]]
  // CHECK-TLS-NEXT:  [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT:  [[ADD:%.*]] = add {{.*}} i32 [[RES]], [[ARR_X_1_1_A]]
  // CHECK-TLS-NEXT:  store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += arr_x[1][1].a;
  // CHECK:      [[ST_INT_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[INT]]* [[ST_INT_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[ST_INT_ST]].cache.)
  // CHECK-NEXT: [[ST_INT_ST_ADDR:%.*]] = bitcast i8* [[ST_INT_ST_TEMP_ADDR]] to [[INT]]*
  // CHECK-NEXT: [[ST_INT_ST_VAL:%.*]] = load [[INT]], [[INT]]* [[ST_INT_ST_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_INT_ST_VAL]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[ST_INT_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[INT]]* [[ST_INT_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[ST_INT_ST_ADDR:%.*]] = bitcast i8* [[ST_INT_ST_TEMP_ADDR]] to [[INT]]*
  // CHECK-DEBUG-NEXT: [[ST_INT_ST_VAL:%.*]] = load [[INT]], [[INT]]* [[ST_INT_ST_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_INT_ST_VAL]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:       [[ST_INT_ST_ADDR:%.*]] = call i32* [[ST_INT_ST_TLS_INITD:[^,]+]]
  // CHECK-TLS-NEXT:  [[ST_INT_ST_VAL:%.*]] = load i32, i32* [[ST_INT_ST_ADDR]]
  // CHECK-TLS-NEXT:  [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT:  [[ADD:%.*]] = add {{.*}} i32 [[RES]], [[ST_INT_ST_VAL]]
  // CHECK-TLS-NEXT:  store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += ST<int>::st;
  // CHECK:      [[ST_FLOAT_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast (float* [[ST_FLOAT_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[ST_FLOAT_ST]].cache.)
  // CHECK-NEXT: [[ST_FLOAT_ST_ADDR:%.*]] = bitcast i8* [[ST_FLOAT_ST_TEMP_ADDR]] to float*
  // CHECK-NEXT: [[ST_FLOAT_ST_VAL:%.*]] = load float, float* [[ST_FLOAT_ST_ADDR]]
  // CHECK-NEXT: [[FLOAT_TO_INT_CONV:%.*]] = fptosi float [[ST_FLOAT_ST_VAL]] to [[INT]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[FLOAT_TO_INT_CONV]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[ST_FLOAT_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast (float* [[ST_FLOAT_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[ST_FLOAT_ST_ADDR:%.*]] = bitcast i8* [[ST_FLOAT_ST_TEMP_ADDR]] to float*
  // CHECK-DEBUG-NEXT: [[ST_FLOAT_ST_VAL:%.*]] = load float, float* [[ST_FLOAT_ST_ADDR]]
  // CHECK-DEBUG-NEXT: [[FLOAT_TO_INT_CONV:%.*]] = fptosi float [[ST_FLOAT_ST_VAL]] to [[INT]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[FLOAT_TO_INT_CONV]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS: [[ST_FLOAT_ST_ADDR:%.*]]  = call float* [[ST_FLOAT_ST_TLS_INITD:[^,]+]]
  // CHECK-TLS-NEXT: [[ST_FLOAT_ST_VAL:%.*]]  = load float, float* [[ST_FLOAT_ST_ADDR]]
  // CHECK-TLS-NEXT: [[FLOAT_TO_INT_CONV:%.*]] = fptosi float [[ST_FLOAT_ST_VAL]]  to i32
  // CHECK-TLS-NEXT: [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add {{.*}} i32 [[RES]], [[FLOAT_TO_INT_CONV]]
  // CHECK-TLS-NEXT: store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += static_cast<int>(ST<float>::st);
  // CHECK:      [[ST_S4_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[ST_S4_ST]].cache.)
  // CHECK-NEXT: [[ST_S4_ST_ADDR:%.*]] = bitcast i8* [[ST_S4_ST_TEMP_ADDR]] to [[S4]]*
  // CHECK-NEXT: [[ST_S4_ST_A_ADDR:%.*]] = getelementptr inbounds [[S4]], [[S4]]* [[ST_S4_ST_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[ST_S4_ST_A:%.*]] = load [[INT]], [[INT]]* [[ST_S4_ST_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_S4_ST_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[ST_S4_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[ST_S4_ST_ADDR:%.*]] = bitcast i8* [[ST_S4_ST_TEMP_ADDR]] to [[S4]]*
  // CHECK-DEBUG-NEXT: [[ST_S4_ST_A_ADDR:%.*]] = getelementptr inbounds [[S4]], [[S4]]* [[ST_S4_ST_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[ST_S4_ST_A:%.*]] = load [[INT]], [[INT]]* [[ST_S4_ST_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_S4_ST_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:       [[ST_S4_ST_ADDR:%.*]] = call [[S4]]* [[ST_S4_ST_TLS_INITD:[^,]+]]
  // CHECK-TLS-NEXT:  [[ST_S4_ST_A_ADDR:%.*]] = getelementptr inbounds [[S4]], [[S4]]* [[ST_S4_ST_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT:  [[ST_S4_ST_A:%.*]] = load i32, i32* [[ST_S4_ST_A_ADDR]]
  // CHECK-TLS-NEXT:  [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT:  [[ADD:%.*]] = add {{.*}} i32 [[RES]], [[ST_S4_ST_A]]
  // CHECK-TLS-NEXT:  store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += ST<S4>::st.a;
  // CHECK:      [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: ret [[INT]] [[RES]]
  // CHECK-DEBUG:      [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: ret [[INT]] [[RES]]
  // CHECK-TLS:      [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT: ret i32 [[RES]]
  return Res;
}
// CHECK: }

// CHECK:      define internal {{.*}}i8* [[SM_CTOR]](i8* %0)
// CHECK:      [[THREAD_NUM:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT]]* [[DEFAULT_LOC]])
// CHECK:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[SMAIN]]*
// CHECK:      [[GS1_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S1]]* [[GS1]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[GS1]].cache.)
// CHECK-NEXT: [[GS1_ADDR:%.*]] = bitcast i8* [[GS1_TEMP_ADDR]] to [[S1]]*
// CHECK-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
// CHECK-NEXT: [[GS1_A:%.*]] = load [[INT]], [[INT]]* [[GS1_A_ADDR]]
// CHECK-NEXT: call {{.*}} [[SMAIN_CTOR:@.+]]([[SMAIN]]* [[RES]], [[INT]] {{.*}}[[GS1_A]])
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK-NEXT: ret i8* [[ARG]]
// CHECK-NEXT: }
// CHECK:      define {{.*}} [[SMAIN_CTOR]]([[SMAIN]]* {{.*}},
// CHECK:      define internal {{.*}}void [[SM_DTOR]](i8* %0)
// CHECK:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[SMAIN]]*
// CHECK-NEXT: call {{.*}} [[SMAIN_DTOR:@.+]]([[SMAIN]]* [[RES]])
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK:      define {{.*}} [[SMAIN_DTOR]]([[SMAIN]]* {{.*}})
// CHECK-DEBUG:      define internal {{.*}}i8* [[SM_CTOR]](i8* %0)
// CHECK-DEBUG:      [[THREAD_NUM:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT]]* {{.*}})



// CHECK-DEBUG:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// CHECK-DEBUG:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK-DEBUG:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[SMAIN]]*
// CHECK-DEBUG:      [[GS1_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S1]]* [[GS1]] to i8*), i{{.*}} {{[0-9]+}}, i8***


// CHECK-DEBUG-NEXT: [[GS1_ADDR:%.*]] = bitcast i8* [[GS1_TEMP_ADDR]] to [[S1]]*
// CHECK-DEBUG-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
// CHECK-DEBUG-NEXT: [[GS1_A:%.*]] = load [[INT]], [[INT]]* [[GS1_A_ADDR]]
// CHECK-DEBUG-NEXT: call {{.*}} [[SMAIN_CTOR:@.+]]([[SMAIN]]* [[RES]], [[INT]] {{.*}}[[GS1_A]])
// CHECK-DEBUG:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// CHECK-DEBUG-NEXT: ret i8* [[ARG]]
// CHECK-DEBUG-NEXT: }
// CHECK-DEBUG:      define {{.*}} [[SMAIN_CTOR]]([[SMAIN]]* {{.*}},
// CHECK-DEBUG:      define internal {{.*}} [[SM_DTOR:@.+]](i8* %0)
// CHECK-DEBUG:      call {{.*}} [[SMAIN_DTOR:@.+]]([[SMAIN]]*
// CHECK-DEBUG:      }
// CHECK-DEBUG:      define {{.*}} [[SMAIN_DTOR]]([[SMAIN]]* {{.*}})
// CHECK-TLS:      define internal [[S1]]* [[GS1_TLS_INITD]] {{#[0-9]+}} {
// CHECK-TLS-NEXT: call void [[GS1_TLS_INIT]]
// CHECK-TLS-NEXT: ret [[S1]]* [[GS1]]
// CHECK-TLS-NEXT: }
// CHECK-TLS: define internal void [[SM_CTOR1]]([[SMAIN]]* %this, i32 {{.*}}) {{.*}} {
// CHECK-TLS: void [[SM_CTOR2:@.*]]([[SMAIN]]* {{.*}}, i32 {{.*}})
// CHECK-TLS: }
// CHECK-TLS: define internal void [[SM_DTOR1]]([[SMAIN]]* %this) {{.*}} {
// CHECK-TLS: void [[SM_DTOR2:@.*]]([[SMAIN]]* {{.*}})
// CHECK-TLS: }
// CHECK-TLS: define {{.*}} [[S3]]* [[STATIC_S_TLS_INITD]]
// CHECK-TLS: call void [[STATIC_S_TLS_INIT:[^,]+]]
// CHECK-TLS: ret [[S3]]* [[STATIC_S]]
// CHECK-TLS: }
// CHECK-TLS: define {{.*}} [[S5]]* [[GS3_TLS_INITD]]
// CHECK-TLS:   call void [[GS3_TLS_INIT:@[^,]+]]
// CHECK-TLS:   ret [[S5]]* [[GS3]]
// CHECK-TLS: }
// CHECK-TLS: define {{.*}} [2 x [3 x [[S1]]]]* [[ARR_X_TLS_INITD]]
// CHECK-TLS:   call void [[ARR_X_TLS_INIT]]
// CHECK-TLS:   ret [2 x [3 x [[S1]]]]* [[ARR_X]]
// CHECK-TLS: }
// CHECK-TLS: define {{.*}} i32* [[ST_INT_ST_TLS_INITD]] {{#[0-9]+}} comdat {
// CHECK-TLS-NOT:   call
// CHECK-TLS:   ret i32* [[ST_INT_ST]]
// CHECK-TLS: }
// CHECK-TLS: define {{.*}} float* [[ST_FLOAT_ST_TLS_INITD]] {{#[0-9]+}} comdat {
// CHECK-TLS-NOT:   call
// CHECK-TLS:   ret float* [[ST_FLOAT_ST]]
// CHECK-TLS: }
// CHECK-TLS: define {{.*}} [[S4]]* [[ST_S4_ST_TLS_INITD]] {{#[0-9]+}} comdat {
// CHECK-TLS:   call void [[ST_S4_ST_TLS_INIT]]
// CHECK-TLS:   ret [[S4]]* [[ST_S4_ST]]
// CHECK-TLS: }

#endif
// OMP50-TLS: define {{.*}}void [[SM_CTOR2]]([[SMAIN]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: define {{.*}}void [[SM_DTOR2]]([[SMAIN]]* {{.*}})

#ifdef BODY
// CHECK-LABEL:  @{{.*}}foobar{{.*}}()
// CHECK-DEBUG-LABEL: @{{.*}}foobar{{.*}}()
// CHECK-TLS-LABEL: @{{.*}}foobar{{.*}}()
int foobar() {

  int Res;
  // CHECK:      [[THREAD_NUM:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT]]* [[DEFAULT_LOC]])
  // CHECK:      [[STATIC_S_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S3]]* [[STATIC_S]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[STATIC_S]].cache.)
  // CHECK-NEXT: [[STATIC_S_ADDR:%.*]] = bitcast i8* [[STATIC_S_TEMP_ADDR]] to [[S3]]*
  // CHECK-NEXT: [[STATIC_S_A_ADDR:%.*]] = getelementptr inbounds [[S3]], [[S3]]* [[STATIC_S_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[STATIC_S_A:%.*]] = load [[INT]], [[INT]]* [[STATIC_S_A_ADDR]]
  // CHECK-NEXT: store [[INT]] [[STATIC_S_A]], [[INT]]* [[RES_ADDR:[^,]+]]
  // CHECK-DEBUG:      [[THREAD_NUM:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT]]* {{.*}})
  // CHECK-DEBUG:      [[STATIC_S_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S3]]* [[STATIC_S]] to i8*), i{{.*}} {{[0-9]+}}, i8***




  // CHECK-DEBUG-NEXT: [[STATIC_S_ADDR:%.*]] = bitcast i8* [[STATIC_S_TEMP_ADDR]] to [[S3]]*
  // CHECK-DEBUG-NEXT: [[STATIC_S_A_ADDR:%.*]] = getelementptr inbounds [[S3]], [[S3]]* [[STATIC_S_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[STATIC_S_A:%.*]] = load [[INT]], [[INT]]* [[STATIC_S_A_ADDR]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[STATIC_S_A]], [[INT]]* [[RES_ADDR:[^,]+]]
  // CHECK-TLS:      [[STATIC_S_ADDR:%.*]]  = call [[S3]]* [[STATIC_S_TLS_INITD]]
  // CHECK-TLS-NEXT: [[STATIC_S_A_ADDR:%.*]] = getelementptr inbounds [[S3]], [[S3]]* [[STATIC_S_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT: [[STATIC_S_A:%.*]] = load i32, i32* [[STATIC_S_A_ADDR]]
  // CHECK-TLS-NEXT: store i32 [[STATIC_S_A]], i32* [[RES_ADDR:[^,]+]]
  Res = Static::s.a;
  // CHECK:      [[GS1_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S1]]* [[GS1]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[GS1]].cache.)
  // CHECK-NEXT: [[GS1_ADDR:%.*]] = bitcast i8* [[GS1_TEMP_ADDR]] to [[S1]]*
  // CHECK-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[GS1_A:%.*]] = load [[INT]], [[INT]]* [[GS1_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS1_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[GS1_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S1]]* [[GS1]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[GS1_ADDR:%.*]] = bitcast i8* [[GS1_TEMP_ADDR]] to [[S1]]*
  // CHECK-DEBUG-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[GS1_A:%.*]] = load [[INT]], [[INT]]* [[GS1_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS1_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:      [[GS1_ADDR:%.*]] = call [[S1]]* [[GS1_TLS_INITD]]
  // CHECK-TLS-NEXT: [[GS1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[GS1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT: [[GS1_A:%.*]] = load i32, i32* [[GS1_A_ADDR]]
  // CHECK-TLS-NEXT: [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add {{.*}} i32 [[RES]], [[GS1_A]]
  // CHECK-TLS-NEXT: store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += gs1.a;
  // CHECK:      [[GS2_A:%.*]] = load [[INT]], [[INT]]* getelementptr inbounds ([[S2]], [[S2]]* [[GS2]], i{{.*}} 0, i{{.*}} 0)
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS2_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG:      [[GS2_A:%.*]] = load [[INT]], [[INT]]* getelementptr inbounds ([[S2]], [[S2]]* [[GS2]], i{{.*}} 0, i{{.*}} 0)
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS2_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:      [[GS2_A:%.*]] = load i32, i32* getelementptr inbounds ([[S2]], [[S2]]* [[GS2]], i{{.*}} 0, i{{.*}} 0)
  // CHECK-TLS-NEXT: [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add {{.*}} i32 [[RES]], [[GS2_A]]
  // CHECK-TLS-NEXT: store i32 [[ADD]], i32* [[RES:.+]]
  Res += gs2.a;
  // CHECK:      [[GS3_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S5]]* [[GS3]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[GS3]].cache.)
  // CHECK-NEXT: [[GS3_ADDR:%.*]] = bitcast i8* [[GS3_TEMP_ADDR]] to [[S5]]*
  // CHECK-NEXT: [[GS3_A_ADDR:%.*]] = getelementptr inbounds [[S5]], [[S5]]* [[GS3_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[GS3_A:%.*]] = load [[INT]], [[INT]]* [[GS3_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS3_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[GS3_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S5]]* [[GS3]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[GS3_ADDR:%.*]] = bitcast i8* [[GS3_TEMP_ADDR]] to [[S5]]*
  // CHECK-DEBUG-NEXT: [[GS3_A_ADDR:%.*]] = getelementptr inbounds [[S5]], [[S5]]* [[GS3_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[GS3_A:%.*]] = load [[INT]], [[INT]]* [[GS3_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[GS3_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:       [[GS3_ADDR:%.*]] = call [[S5]]* [[GS3_TLS_INITD]]
  // CHECK-TLS-DEBUG: [[GS3_A_ADDR:%.*]] = getelementptr inbounds [[S5]], [[S5]]* [[GS3_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-DEBUG: [[GS3_A:%.*]] = load i32, i32* [[GS3_A_ADDR]]
  // CHECK-TLS-DEBUG: [[RES:%.*]] = load i32, i32* [[RES_ADDR]]
  // CHECK-TLS-DEBUG: [[ADD:%.*]]= add nsw i32 [[RES]], [[GS3_A]]
  // CHECK-TLS-DEBUG: store i32 [[ADD]], i32* [[RES_ADDR]]
  Res += gs3.a;
  // CHECK:      [[ARR_X_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([2 x [3 x [[S1]]]]* [[ARR_X]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[ARR_X]].cache.)
  // CHECK-NEXT: [[ARR_X_ADDR:%.*]] = bitcast i8* [[ARR_X_TEMP_ADDR]] to [2 x [3 x [[S1]]]]*
  // CHECK-NEXT: [[ARR_X_1_ADDR:%.*]] = getelementptr inbounds [2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-NEXT: [[ARR_X_1_1_ADDR:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR_X_1_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-NEXT: [[ARR_X_1_1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_X_1_1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[ARR_X_1_1_A:%.*]] = load [[INT]], [[INT]]* [[ARR_X_1_1_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ARR_X_1_1_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT:      [[ARR_X_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([2 x [3 x [[S1]]]]* [[ARR_X]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[ARR_X_ADDR:%.*]] = bitcast i8* [[ARR_X_TEMP_ADDR]] to [2 x [3 x [[S1]]]]*
  // CHECK-DEBUG-NEXT: [[ARR_X_1_ADDR:%.*]] = getelementptr inbounds [2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-DEBUG-NEXT: [[ARR_X_1_1_ADDR:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR_X_1_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-DEBUG-NEXT: [[ARR_X_1_1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_X_1_1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[ARR_X_1_1_A:%.*]] = load [[INT]], [[INT]]* [[ARR_X_1_1_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ARR_X_1_1_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:      [[ARR_X_ADDR:%.*]] = call [2 x [3 x [[S1]]]]* [[ARR_X_TLS_INITD]]
  // CHECK-TLS-NEXT: [[ARR_X_1_ADDR:%.*]] = getelementptr inbounds [2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-TLS-NEXT: [[ARR_X_1_1_ADDR:%.*]] = getelementptr inbounds [3 x [[S1]]], [3 x [[S1]]]* [[ARR_X_1_ADDR]], i{{.*}} 0, i{{.*}} 1
  // CHECK-TLS-NEXT: [[ARR_X_1_1_A_ADDR:%.*]] = getelementptr inbounds [[S1]], [[S1]]* [[ARR_X_1_1_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT: [[ARR_X_1_1_A:%.*]] = load [[INT]], [[INT]]* [[ARR_X_1_1_A_ADDR]]
  // CHECK-TLS-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ARR_X_1_1_A]]
  // CHECK-TLS-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  Res += arr_x[1][1].a;
  // CHECK:      [[ST_INT_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[INT]]* [[ST_INT_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[ST_INT_ST]].cache.)
  // CHECK-NEXT: [[ST_INT_ST_ADDR:%.*]] = bitcast i8* [[ST_INT_ST_TEMP_ADDR]] to [[INT]]*
  // CHECK-NEXT: [[ST_INT_ST_VAL:%.*]] = load [[INT]], [[INT]]* [[ST_INT_ST_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_INT_ST_VAL]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[ST_INT_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[INT]]* [[ST_INT_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[ST_INT_ST_ADDR:%.*]] = bitcast i8* [[ST_INT_ST_TEMP_ADDR]] to [[INT]]*
  // CHECK-DEBUG-NEXT: [[ST_INT_ST_VAL:%.*]] = load [[INT]], [[INT]]* [[ST_INT_ST_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_INT_ST_VAL]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // OMP45-TLS:      [[ST_INT_ST_ADDR:%.*]] = call i32* [[ST_INT_ST_TLS_INITD]]
  // OMP45-TLS-NEXT: [[ST_INT_ST_VAL:%.*]] = load [[INT]], [[INT]]* [[ST_INT_ST_ADDR]]
  // OMP45-TLS-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // OMP45-TLS-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_INT_ST_VAL]]
  // OMP45-TLS-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  Res += ST<int>::st;
  // CHECK:      [[ST_FLOAT_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast (float* [[ST_FLOAT_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[ST_FLOAT_ST]].cache.)
  // CHECK-NEXT: [[ST_FLOAT_ST_ADDR:%.*]] = bitcast i8* [[ST_FLOAT_ST_TEMP_ADDR]] to float*
  // CHECK-NEXT: [[ST_FLOAT_ST_VAL:%.*]] = load float, float* [[ST_FLOAT_ST_ADDR]]
  // CHECK-NEXT: [[FLOAT_TO_INT_CONV:%.*]] = fptosi float [[ST_FLOAT_ST_VAL]] to [[INT]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[FLOAT_TO_INT_CONV]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[ST_FLOAT_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast (float* [[ST_FLOAT_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[ST_FLOAT_ST_ADDR:%.*]] = bitcast i8* [[ST_FLOAT_ST_TEMP_ADDR]] to float*
  // CHECK-DEBUG-NEXT: [[ST_FLOAT_ST_VAL:%.*]] = load float, float* [[ST_FLOAT_ST_ADDR]]
  // CHECK-DEBUG-NEXT: [[FLOAT_TO_INT_CONV:%.*]] = fptosi float [[ST_FLOAT_ST_VAL]] to [[INT]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[FLOAT_TO_INT_CONV]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // OMP45-TLS:      [[ST_FLOAT_ST_ADDR:%.*]] = call float* [[ST_FLOAT_ST_TLS_INITD]]
  // OMP45-TLS-NEXT: [[ST_FLOAT_ST_VAL:%.*]] = load float, float* [[ST_FLOAT_ST_ADDR]]
  // OMP45-TLS-NEXT: [[FLOAT_TO_INT_CONV:%.*]] = fptosi float [[ST_FLOAT_ST_VAL]] to [[INT]]
  // OMP45-TLS-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // OMP45-TLS-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[FLOAT_TO_INT_CONV]]
  // OMP45-TLS-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  Res += static_cast<int>(ST<float>::st);
  // CHECK:      [[ST_S4_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* [[DEFAULT_LOC]], i32 {{.*}}, i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8*** [[ST_S4_ST]].cache.)
  // CHECK-NEXT: [[ST_S4_ST_ADDR:%.*]] = bitcast i8* [[ST_S4_ST_TEMP_ADDR]] to [[S4]]*
  // CHECK-NEXT: [[ST_S4_ST_A_ADDR:%.*]] = getelementptr inbounds [[S4]], [[S4]]* [[ST_S4_ST_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-NEXT: [[ST_S4_ST_A:%.*]] = load [[INT]], [[INT]]* [[ST_S4_ST_A_ADDR]]
  // CHECK-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_S4_ST_A]]
  // CHECK-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-DEBUG-NEXT: [[ST_S4_ST_TEMP_ADDR:%.*]] = call {{.*}}i8* @__kmpc_threadprivate_cached([[IDENT]]* {{.*}}, i32 {{.*}}, i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*), i{{.*}} {{[0-9]+}}, i8***


  // CHECK-DEBUG-NEXT: [[ST_S4_ST_ADDR:%.*]] = bitcast i8* [[ST_S4_ST_TEMP_ADDR]] to [[S4]]*
  // CHECK-DEBUG-NEXT: [[ST_S4_ST_A_ADDR:%.*]] = getelementptr inbounds [[S4]], [[S4]]* [[ST_S4_ST_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-DEBUG-NEXT: [[ST_S4_ST_A:%.*]] = load [[INT]], [[INT]]* [[ST_S4_ST_A_ADDR]]
  // CHECK-DEBUG-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_S4_ST_A]]
  // CHECK-DEBUG-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  // CHECK-TLS:      [[ST_S4_ST_ADDR:%.*]] = call [[S4]]* [[ST_S4_ST_TLS_INITD]]
  // CHECK-TLS-NEXT: [[ST_S4_ST_A_ADDR:%.*]] = getelementptr inbounds [[S4]], [[S4]]* [[ST_S4_ST_ADDR]], i{{.*}} 0, i{{.*}} 0
  // CHECK-TLS-NEXT: [[ST_S4_ST_A:%.*]] = load [[INT]], [[INT]]* [[ST_S4_ST_A_ADDR]]
  // CHECK-TLS-NEXT: [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-TLS-NEXT: [[ADD:%.*]] = add {{.*}} [[INT]] [[RES]], [[ST_S4_ST_A]]
  // CHECK-TLS-NEXT: store [[INT]] [[ADD]], [[INT]]* [[RES:.+]]
  Res += ST<S4>::st.a;
  // CHECK:      [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-NEXT: ret [[INT]] [[RES]]
  // CHECK-DEBUG:      [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-DEBUG-NEXT: ret [[INT]] [[RES]]
  // CHECK-TLS:      [[RES:%.*]] = load [[INT]], [[INT]]* [[RES_ADDR]]
  // CHECK-TLS-NEXT: ret [[INT]] [[RES]]
  return Res;
}
#endif

// OMP45:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* [[DEFAULT_LOC]], i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*), i8* (i8*)* [[ST_S4_ST_CTOR:@\.__kmpc_global_ctor_\..+]], i8* (i8*, i8*)* null, void (i8*)* [[ST_S4_ST_DTOR:@\.__kmpc_global_dtor_\..+]])
// OMP45:      define internal {{.*}}i8* [[ST_S4_ST_CTOR]](i8* %0)
// OMP45:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// OMP45:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// OMP45:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[S4]]*
// OMP45-NEXT: call {{.*}} [[S4_CTOR:@.+]]([[S4]]* [[RES]], {{.*}} 23)
// OMP45:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// OMP45-NEXT: ret i8* [[ARG]]
// OMP45-NEXT: }
// OMP45:      define {{.*}} [[S4_CTOR]]([[S4]]* {{.*}},
// OMP45:      define internal {{.*}}void [[ST_S4_ST_DTOR]](i8* %0)
// OMP45:      store i8* %0, i8** [[ARG_ADDR:%.*]],
// OMP45:      [[ARG:%.+]] = load i8*, i8** [[ARG_ADDR]]
// OMP45:      [[RES:%.*]] = bitcast i8* [[ARG]] to [[S4]]*
// OMP45-NEXT: call {{.*}} [[S4_DTOR:@.+]]([[S4]]* [[RES]])
// OMP45-NEXT: ret void
// OMP45-NEXT: }
// OMP45:      define {{.*}} [[S4_DTOR]]([[S4]]* {{.*}})




// OMP45-DEBUG:      @__kmpc_global_thread_num
// OMP45-DEBUG:      call {{.*}}void @__kmpc_threadprivate_register([[IDENT]]* {{.*}}, i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*), i8* (i8*)* [[ST_S4_ST_CTOR:@\.__kmpc_global_ctor_\..+]], i8* (i8*, i8*)* null, void (i8*)* [[ST_S4_ST_DTOR:@\.__kmpc_global_dtor_\..+]])
// OMP45-DEBUG:      define internal {{.*}}i8* [[ST_S4_ST_CTOR]](i8* %0)
// OMP45-DEBUG:      }
// OMP45-DEBUG:      define {{.*}} [[S4_CTOR:@.*]]([[S4]]* {{.*}},
// OMP45-DEBUG:      define internal {{.*}}void [[ST_S4_ST_DTOR]](i8* %0)
// OMP45-DEBUG:      }
// OMP45-DEBUG:      define {{.*}} [[S4_DTOR:@.*]]([[S4]]* {{.*}})

// CHECK:      define internal {{.*}}void {{@.*}}()
// CHECK-DAG:  call {{.*}}void [[GS1_INIT]]()
// CHECK-DAG:  call {{.*}}void [[ARR_X_INIT]]()
// CHECK:      ret void
// CHECK-DEBUG:      define internal {{.*}}void {{@.*}}()
// CHECK-DEBUG:      ret void

// OMP45-TLS: define internal void [[GS1_CXX_INIT:@.*]]()
// OMP45-TLS: call void [[GS1_CTOR1:@.*]]([[S1]]* [[GS1]], i32 5)
// OMP45-TLS: call i32 @__cxa_thread_atexit(void (i8*)* bitcast (void ([[S1]]*)* [[GS1_DTOR1:.*]] to void (i8*)*), i8* bitcast ([[S1]]* [[GS1]] to i8*)
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[GS1_CTOR1]]([[S1]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: call void [[GS1_CTOR2:@.*]]([[S1]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[GS1_DTOR1]]([[S1]]* {{.*}})
// OMP45-TLS: call void [[GS1_DTOR2:@.*]]([[S1]]* {{.*}})
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[GS1_CTOR2]]([[S1]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: define {{.*}}void [[GS1_DTOR2]]([[S1]]* {{.*}})

// OMP45-TLS: define internal void [[GS2_CXX_INIT:@.*]]()
// OMP45-TLS: call void [[GS2_CTOR1:@.*]]([[S2]]* [[GS2]], i32 27)
// OMP45-TLS: call i32 @__cxa_atexit(void (i8*)* bitcast (void ([[S2]]*)* [[GS2_DTOR1:.*]] to void (i8*)*), i8* bitcast ([[S2]]* [[GS2]] to i8*)
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[GS2_CTOR1]]([[S2]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: call void [[GS2_CTOR2:@.*]]([[S2]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[GS2_DTOR1]]([[S2]]* {{.*}})
// OMP45-TLS: call void [[GS2_DTOR2:@.*]]([[S2]]* {{.*}})
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[GS2_CTOR2]]([[S2]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: define {{.*}}void [[GS2_DTOR2]]([[S2]]* {{.*}})

// OMP45-TLS: define internal void [[ARR_X_CXX_INIT:@.*]]()
// OMP45-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 0, i{{.*}} 0), i{{.*}} 1)
// OMP45-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 0, i{{.*}} 1), i{{.*}} 2)
// OMP45-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 0, i{{.*}} 2), i{{.*}} 3)
// OMP45-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 1, i{{.*}} 0), i{{.*}} 4)
// OMP45-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 1, i{{.*}} 1), i{{.*}} 5)
// OMP45-TLS: invoke void [[GS1_CTOR1]]([[S1]]* getelementptr inbounds ([2 x [3 x [[S1]]]], [2 x [3 x [[S1]]]]* [[ARR_X]], i{{.*}} 0, i{{.*}} 1, i{{.*}} 2), i{{.*}} 6)
// OMP45-TLS: call i32 @__cxa_thread_atexit(void (i8*)* [[ARR_X_CXX_DTOR:@[^,]+]]
// OMP45-TLS: define internal void [[ARR_X_CXX_DTOR]](i8* %0)
// OMP45-TLS: void [[GS1_DTOR1]]([[S1]]* {{.*}})

// OMP45-TLS: define {{.*}}void [[SM_CTOR2]]([[SMAIN]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: define {{.*}}void [[SM_DTOR2]]([[SMAIN]]* {{.*}})

// OMP45-TLS: define internal void [[ST_S4_ST_CXX_INIT]]()
// OMP45-TLS: call void [[ST_S4_ST_CTOR1:@.*]]([[S4]]* [[ST_S4_ST]], i32 23)
// OMP45-TLS: call i32 @__cxa_thread_atexit(void (i8*)* bitcast (void ([[S4]]*)* [[ST_S4_ST_DTOR1:.*]] to void (i8*)*), i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*)
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[ST_S4_ST_CTOR1]]([[S4]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: call void [[ST_S4_ST_CTOR2:@.*]]([[S4]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[ST_S4_ST_DTOR1]]([[S4]]* {{.*}})
// OMP45-TLS: call void [[ST_S4_ST_DTOR2:@.*]]([[S4]]* {{.*}})
// OMP45-TLS: }
// OMP45-TLS: define {{.*}}void [[ST_S4_ST_CTOR2]]([[S4]]* {{.*}}, i32 {{.*}})
// OMP45-TLS: define {{.*}}void [[ST_S4_ST_DTOR2]]([[S4]]* {{.*}})

// OMP50-TLS: define internal void [[ST_S4_ST_CXX_INIT]]()
// OMP50-TLS: call void [[ST_S4_ST_CTOR1:@.*]]([[S4]]* [[ST_S4_ST]], i32 23)

// OMP50-TLS: call i32 @__cxa_thread_atexit(void (i8*)* bitcast (void ([[S4]]*)* [[ST_S4_ST_DTOR1:.*]] to void (i8*)*), i8* bitcast ([[S4]]* [[ST_S4_ST]] to i8*)
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[ST_S4_ST_CTOR1]]([[S4]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: call void [[ST_S4_ST_CTOR2:@.*]]([[S4]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[ST_S4_ST_DTOR1]]([[S4]]* {{.*}})
// OMP50-TLS: call void [[ST_S4_ST_DTOR2:@.*]]([[S4]]* {{.*}})
// OMP50-TLS: }
// OMP50-TLS: define {{.*}}void [[ST_S4_ST_CTOR2]]([[S4]]* {{.*}}, i32 {{.*}})
// OMP50-TLS: define {{.*}}void [[ST_S4_ST_DTOR2]]([[S4]]* {{.*}})

// CHECK-TLS:      define internal void @__tls_init()
// CHECK-TLS:      [[GRD:%.*]] = load i8, i8* @__tls_guard
// CHECK-TLS-NEXT: [[IS_INIT:%.*]] = icmp eq i8 [[GRD]], 0
// CHECK-TLS-NEXT: br i1 [[IS_INIT]], label %[[INIT_LABEL:[^,]+]], label %[[DONE_LABEL:[^,]+]]{{.*}}
// CHECK-TLS:      [[INIT_LABEL]]
// CHECK-TLS-NEXT: store i8 1, i8* @__tls_guard
// CHECK-TLS:      call void [[GS1_CXX_INIT]]
// CHECK-TLS-NOT:  call void [[GS2_CXX_INIT]]
// CHECK-TLS:      call void [[ARR_X_CXX_INIT]]
// CHECK-TLS-NOT:  call void [[ST_S4_ST_CXX_INIT]]
// CHECK-TLS:      [[DONE_LABEL]]

// CHECK-TLS-DAG:      declare {{.*}} void [[GS3_TLS_INIT]]
// CHECK-TLS-DAG:      declare {{.*}} void [[STATIC_S_TLS_INIT]]
