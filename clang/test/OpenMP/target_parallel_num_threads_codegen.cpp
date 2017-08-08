// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: %ident_t = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: [[DEF_LOC:@.+]] = private unnamed_addr constant %ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

// CHECK-DAG: [[S1:%.+]] = type { double }
// CHECK-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i[[SZ:32|64]], i32, i32 }
// CHECK-DAG: [[DEVTY:%.+]] = type { i8*, i8*, [[ENTTY]]*, [[ENTTY]]* }
// CHECK-DAG: [[DSCTY:%.+]] = type { i32, [[DEVTY]]*, [[ENTTY]]*, [[ENTTY]]* }

// TCHECK: [[ENTTY:%.+]] = type { i8*, i8*, i{{32|64}}, i32, i32 }

// CHECK-DAG: $[[REGFN:\.omp_offloading\..+]] = comdat

// We have 6 target regions

// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0
// CHECK-DAG: @{{.*}} = private constant i8 0

// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]
// TCHECK: @{{.+}} = constant [[ENTTY]]

// Check if offloading descriptor is created.
// CHECK: [[ENTBEGIN:@.+]] = external constant [[ENTTY]]
// CHECK: [[ENTEND:@.+]] = external constant [[ENTTY]]
// CHECK: [[DEVBEGIN:@.+]] = external constant i8
// CHECK: [[DEVEND:@.+]] = external constant i8
// CHECK: [[IMAGES:@.+]] = internal unnamed_addr constant [1 x [[DEVTY]]] [{{.+}} { i8* [[DEVBEGIN]], i8* [[DEVEND]], [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }], comdat($[[REGFN]])
// CHECK: [[DESC:@.+]] = internal constant [[DSCTY]] { i32 1, [[DEVTY]]* getelementptr inbounds ([1 x [[DEVTY]]], [1 x [[DEVTY]]]* [[IMAGES]], i32 0, i32 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }, comdat($[[REGFN]])

// Check target registration is registered as a Ctor.
// CHECK: appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* bitcast (void (i8*)* @[[REGFN]] to void ()*), i8* bitcast (void (i8*)* @[[REGFN]] to i8*) }]


template<typename tx>
tx ftemplate(int n) {
  tx a = 0;

  #pragma omp target parallel num_threads(tx(20))
  {
  }

  short b = 1;
  #pragma omp target parallel num_threads(b)
  {
    a += b;
  }

  return a;
}

static
int fstatic(int n) {

  #pragma omp target parallel num_threads(n)
  {
  }

  #pragma omp target parallel num_threads(32+n)
  {
  }

  return n+1;
}

struct S1 {
  double a;

  int r1(int n){
    int b = 1;

    #pragma omp target parallel num_threads(n-b)
    {
      this->a = (double)b + 1.5;
    }

    #pragma omp target parallel num_threads(1024)
    {
      this->a = 2.5;
    }

    return (int)a;
  }
};

// CHECK: define {{.*}}@{{.*}}bar{{.*}}
int bar(int n){
  int a = 0;

  S1 S;
  // CHECK: call {{.*}}i32 [[FS1:@.+]]([[S1]]* {{.*}}, i32 {{.*}})
  a += S.r1(n);

  // CHECK: call {{.*}}i32 [[FSTATIC:@.+]](i32 {{.*}})
  a += fstatic(n);

  // CHECK: call {{.*}}i32 [[FTEMPLATE:@.+]](i32 {{.*}})
  a += ftemplate<int>(n);

  return a;
}



//
// CHECK: define {{.*}}[[FS1]]([[S1]]* {{%.+}}, i32 {{[^%]*}}[[PARM:%.+]])
//
// CHECK-DAG:   store i32 [[PARM]], i32* [[N_ADDR:%.+]], align
// CHECK:       store i32 1, i32* [[B:%.+]], align
// CHECK:       [[NV:%.+]] = load i32, i32* [[N_ADDR]], align
// CHECK:       [[BV:%.+]] = load i32, i32* [[B]], align
// CHECK:       [[SUB:%.+]] = sub nsw i32 [[NV]], [[BV]]
// CHECK:       store i32 [[SUB]], i32* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CEV:%.+]] = load i32, i32* [[CAPE_ADDR]], align
// CHECK-64:    [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPEC_ADDR:%.+]] to i32*
// CHECK-64:    store i32 [[CEV]], i32* [[CONV]], align
// CHECK-32:    store i32 [[CEV]], i32* [[CAPEC_ADDR:%.+]], align
// CHECK:       [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[CAPEC_ADDR]], align
// CHECK:       [[THREADS:%.+]] = load i32, i32* [[CAPE_ADDR]], align
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 3, {{.*}}, i32 1, i32 [[THREADS]])
// CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align
// CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK:       br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT1:@.+]]([[S1]]* {{%.+}}, i[[SZ]] {{%.+}}, i[[SZ]] [[ARG]])
// CHECK:       br label {{%?}}[[END]]
// CHECK:       [[END]]
//
//
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 1, {{.+}}, i32 1, i32 1024)
// CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align
// CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK:       br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT2:@.+]]([[S1]]* {{[^,]+}})
// CHECK:       br label {{%?}}[[END]]
// CHECK:       [[END]]
//






//
// CHECK: define {{.*}}[[FSTATIC]](i32 {{[^%]*}}[[PARM:%.+]])
//
// CHECK-DAG:   store i32 [[PARM]], i32* [[N_ADDR:%.+]], align
// CHECK:       [[NV:%.+]] = load i32, i32* [[N_ADDR]], align
// CHECK:       store i32 [[NV]], i32* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CEV:%.+]] = load i32, i32* [[CAPE_ADDR]], align
// CHECK-64:    [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPEC_ADDR:%.+]] to i32*
// CHECK-64:    store i32 [[CEV]], i32* [[CONV]], align
// CHECK-32:    store i32 [[CEV]], i32* [[CAPEC_ADDR:%.+]], align
// CHECK:       [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[CAPEC_ADDR]], align
// CHECK:       [[THREADS:%.+]] = load i32, i32* [[CAPE_ADDR]], align
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 1, {{.*}}, i32 1, i32 [[THREADS]])
// CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align
// CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK:       br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT3:@.+]](i[[SZ]] [[ARG]])
// CHECK:       br label {{%?}}[[END]]
// CHECK:       [[END]]
//
//
//
// CHECK:       [[NV:%.+]] = load i32, i32* [[N_ADDR]], align
// CHECK:       [[ADD:%.+]] = add nsw i32 32, [[NV]]
// CHECK:       store i32 [[ADD]], i32* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CEV:%.+]] = load i32, i32* [[CAPE_ADDR]], align
// CHECK-64:    [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPEC_ADDR:%.+]] to i32*
// CHECK-64:    store i32 [[CEV]], i32* [[CONV]], align
// CHECK-32:    store i32 [[CEV]], i32* [[CAPEC_ADDR:%.+]], align
// CHECK:       [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[CAPEC_ADDR]], align
// CHECK:       [[THREADS:%.+]] = load i32, i32* [[CAPE_ADDR]], align
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 1, {{.*}}, i32 1, i32 [[THREADS]])
// CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align
// CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK:       br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT4:@.+]](i[[SZ]] [[ARG]])
// CHECK:       br label {{%?}}[[END]]
// CHECK:       [[END]]
//






//
// CHECK: define {{.*}}[[FTEMPLATE]]
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 0, {{.*}}, i32 1, i32 20)
// CHECK-NEXT:  store i32 [[RET]], i32* [[RHV:%.+]], align
// CHECK-NEXT:  [[RET2:%.+]] = load i32, i32* [[RHV]], align
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT5:@.+]]()
// CHECK:       br label {{%?}}[[END]]
//
// CHECK:       [[END]]
//
//
//
// CHECK:       store i16 1, i16* [[B:%.+]], align
// CHECK:       [[BV:%.+]] = load i16, i16* [[B]], align
// CHECK:       store i16 [[BV]], i16* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CEV:%.+]] = load i16, i16* [[CAPE_ADDR]], align
// CHECK:       [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPEC_ADDR:%.+]] to i16*
// CHECK:       store i16 [[CEV]], i16* [[CONV]], align
// CHECK:       [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[CAPEC_ADDR]], align
// CHECK:       [[T:%.+]] = load i16, i16* [[CAPE_ADDR]], align
// CHECK:       [[THREADS:%.+]] = sext i16 [[T]] to i32
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i32 -1, i8* @{{[^,]+}}, i32 3, {{.*}}, i32 1, i32 [[THREADS]])
// CHECK:       store i32 [[RET]], i32* [[RHV:%.+]], align
// CHECK:       [[RET2:%.+]] = load i32, i32* [[RHV]], align
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET2]], 0
// CHECK:       br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT6:@.+]](i[[SZ]] {{%.+}}, i[[SZ]] {{%.+}}, i[[SZ]] [[ARG]])
// CHECK:       br label {{%?}}[[END]]
// CHECK:       [[END]]
//






// Check that the offloading functions are emitted and that the parallel function
// is appropriately guarded.

// CHECK:       define internal void [[HVT1]]([[S1]]* {{%.+}}, i[[SZ]] [[PARM1:%.+]], i[[SZ]] [[PARM2:%.+]])
// CHECK-DAG:   store i[[SZ]] [[PARM2]], i[[SZ]]* [[CAPE_ADDR:%.+]], align
// CHECK-64:    [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPE_ADDR]] to i32*
// CHECK-64:    [[NT:%.+]] = load i32, i32* [[CONV]], align
// CHECK-32:    [[NT:%.+]] = load i32, i32* [[CAPE_ADDR]], align
// CHECK:       call void @__kmpc_push_num_threads(%ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 [[NT]])
// CHECK:       call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%ident_t* [[DEF_LOC]], i32 2,
//
//


// CHECK:       define internal void [[HVT2]]([[S1]]* {{%.+}})
// CHECK:       call void @__kmpc_push_num_threads(%ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 1024)
// CHECK:       call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%ident_t* [[DEF_LOC]], i32 1,
//
//








// CHECK:       define internal void [[HVT3]](i[[SZ]] [[PARM:%.+]])
// CHECK-DAG:   store i[[SZ]] [[PARM]], i[[SZ]]* [[CAPE_ADDR:%.+]], align
// CHECK-64:    [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPE_ADDR]] to i32*
// CHECK-64:    [[NT:%.+]] = load i32, i32* [[CONV]], align
// CHECK-32:    [[NT:%.+]] = load i32, i32* [[CAPE_ADDR]], align
// CHECK:       call void @__kmpc_push_num_threads(%ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 [[NT]])
// CHECK:       call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%ident_t* [[DEF_LOC]], i32 0,
//
//
// CHECK:       define internal void [[HVT4]](i[[SZ]] [[PARM:%.+]])
// CHECK-DAG:   store i[[SZ]] [[PARM]], i[[SZ]]* [[CAPE_ADDR:%.+]], align
// CHECK-64:    [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPE_ADDR]] to i32*
// CHECK-64:    [[NT:%.+]] = load i32, i32* [[CONV]], align
// CHECK-32:    [[NT:%.+]] = load i32, i32* [[CAPE_ADDR]], align
// CHECK:       call void @__kmpc_push_num_threads(%ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 [[NT]])
// CHECK:       call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%ident_t* [[DEF_LOC]], i32 0,
//
//





// CHECK:       define internal void [[HVT5]](
// CHECK:       call void @__kmpc_push_num_threads(%ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 20)
// CHECK:       call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%ident_t* [[DEF_LOC]], i32 0,
//
//


// CHECK:       define internal void [[HVT6]](i[[SZ]] [[PARM1:%.+]], i[[SZ]] [[PARM2:%.+]], i[[SZ]] [[PARM3:%.+]])
// CHECK-DAG:   store i[[SZ]] [[PARM3]], i[[SZ]]* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPE_ADDR]] to i16*
// CHECK:       [[T:%.+]] = load i16, i16* [[CONV]], align
// CHECK:       [[NT:%.+]] = sext i16 [[T]] to i32
// CHECK:       call void @__kmpc_push_num_threads(%ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 [[NT]])
// CHECK:       call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%ident_t* [[DEF_LOC]], i32 2,
//
//



#endif
