// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: [[DEF_LOC:@.+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

// CHECK-DAG: [[S1:%.+]] = type { double }
// CHECK-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i[[SZ:32|64]], i32, i32 }

// TCHECK: [[ENTTY:%.+]] = type { i8*, i8*, i{{32|64}}, i32, i32 }

// We have 6 target regions

// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0

// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]

// Check target registration is registered as a Ctor.
// CHECK: appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* @.omp_offloading.requires_reg, i8* null }]


template<typename tx>
tx ftemplate(int n) {
  tx a = 0;

  #pragma omp target parallel if(parallel: 0)
  {
    a += 1;
  }

  short b = 1;
  #pragma omp target parallel if(parallel: 1)
  {
    a += b;
  }

  return a;
}

static
int fstatic(int n) {

  #pragma omp target parallel if(n>1)
  {
  }

  #pragma omp target parallel if(target: n-2>2)
  {
  }

  return n+1;
}

struct S1 {
  double a;

  int r1(int n){
    int b = 1;

    #pragma omp target parallel if(parallel: n>3)
    {
      this->a = (double)b + 1.5;
    }

    #pragma omp target parallel if(target: n>4) if(parallel: n>5)
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
// CHECK:       [[NV:%.+]] = load i32, i32* [[N_ADDR]], align
// CHECK:       [[CMP:%.+]] = icmp sgt i32 [[NV]], 3
// CHECK:       [[FB:%.+]] = zext i1 [[CMP]] to i8
// CHECK:       store i8 [[FB]], i8* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CAPE:%.+]] = load i8, i8* [[CAPE_ADDR]], align
// CHECK:       [[TB:%.+]] = trunc i8 [[CAPE]] to i1
// CHECK:       [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPEC_ADDR:%.+]] to i8*
// CHECK:       [[FB:%.+]] = zext i1 [[TB]] to i8
// CHECK:       store i8 [[FB]], i8* [[CONV]], align
// CHECK:       [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[CAPEC_ADDR]], align
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 4, {{.*}}, i32 1, i32 [[NT:%.+]])
// CHECK-DAG:   [[NT]] = select i1 %{{.+}}, i32 0, i32 1
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK:       br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT1:@.+]]([[S1]]* {{%.+}}, i[[SZ]] {{%.+}}, i[[SZ]] [[ARG]])
// CHECK:       br label {{%?}}[[END]]
// CHECK:       [[END]]
//
//
//
// CHECK:       [[NV:%.+]] = load i32, i32* [[N_ADDR]], align
// CHECK:       [[CMP:%.+]] = icmp sgt i32 [[NV]], 5
// CHECK:       [[FB:%.+]] = zext i1 [[CMP]] to i8
// CHECK:       store i8 [[FB]], i8* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CAPE:%.+]] = load i8, i8* [[CAPE_ADDR]], align
// CHECK:       [[TB:%.+]] = trunc i8 [[CAPE]] to i1
// CHECK:       [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPEC_ADDR:%.+]] to i8*
// CHECK:       [[FB:%.+]] = zext i1 [[TB]] to i8
// CHECK:       store i8 [[FB]], i8* [[CONV]], align
// CHECK:       [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[CAPEC_ADDR]], align
// CHECK:       [[NV:%.+]] = load i32, i32* [[N_ADDR]], align
// CHECK:       [[CMP:%.+]] = icmp sgt i32 [[NV]], 4
// CHECK:       br i1 [[CMP]], label {{%?}}[[IF_THEN:.+]], label {{%?}}[[IF_ELSE:.+]]
//
// CHECK:       [[IF_THEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 3, {{.*}}, i32 1, i32 [[NT:%.+]])
// CHECK-DAG:   [[NT]] = select i1 %{{.+}}, i32 0, i32 1
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT2:@.+]]([[S1]]* {{%.+}}, i[[SZ]] [[ARG]])
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IF_ELSE]]
// CHECK:       call void [[HVT2]]([[S1]]* {{%.+}}, i[[SZ]] [[ARG]])
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]

//
// CHECK: define {{.*}}[[FSTATIC]](i32 {{[^%]*}}[[PARM:%.+]])
//
// CHECK-DAG:   store i32 [[PARM]], i32* [[N_ADDR:%.+]], align
// CHECK:       [[NV:%.+]] = load i32, i32* [[N_ADDR]], align
// CHECK:       [[CMP:%.+]] = icmp sgt i32 [[NV]], 1
// CHECK:       [[FB:%.+]] = zext i1 [[CMP]] to i8
// CHECK:       store i8 [[FB]], i8* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CAPE:%.+]] = load i8, i8* [[CAPE_ADDR]], align
// CHECK:       [[TB:%.+]] = trunc i8 [[CAPE]] to i1
// CHECK:       [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPEC_ADDR:%.+]] to i8*
// CHECK:       [[FB:%.+]] = zext i1 [[TB]] to i8
// CHECK:       store i8 [[FB]], i8* [[CONV]], align
// CHECK:       [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[CAPEC_ADDR]], align
// CHECK:       [[CAPE2:%.+]] = load i8, i8* [[CAPE_ADDR]], align
// CHECK:       [[TB:%.+]] = trunc i8 [[CAPE2]] to i1
// CHECK:       br i1 [[TB]], label {{%?}}[[IF_THEN:.+]], label {{%?}}[[IF_ELSE:.+]]
//
// CHECK:       [[IF_THEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 1, {{.*}}, i32 1, i32 [[NT:%.+]])
// CHECK-DAG:   [[NT]] = select i1 %{{.+}}, i32 0, i32 1
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT3:@.+]](i[[SZ]] [[ARG]])
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IF_ELSE]]
// CHECK:       call void [[HVT3]](i[[SZ]] [[ARG]])
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]
//
//
//
// CHECK-DAG:   [[NV:%.+]] = load i32, i32* [[N_ADDR]], align
// CHECK:       [[SUB:%.+]] = sub nsw i32 [[NV]], 2
// CHECK:       [[CMP:%.+]] = icmp sgt i32 [[SUB]], 2
// CHECK:       br i1 [[CMP]], label {{%?}}[[IF_THEN:.+]], label {{%?}}[[IF_ELSE:.+]]
//
// CHECK:       [[IF_THEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 0, {{.*}}, i32 1, i32 0)
// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT4:@.+]]()
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IF_ELSE]]
// CHECK:       call void [[HVT4]]()
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]






//
// CHECK: define {{.*}}[[FTEMPLATE]]
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 1, {{.*}}, i32 1, i32 1)
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT5:@.+]]({{[^,]+}})
// CHECK:       br label {{%?}}[[END]]
//
// CHECK:       [[END]]
//
//
//
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 2, {{.*}}, i32 1, i32 0)
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
//
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT6:@.+]]({{[^,]+}}, {{[^,]+}})
// CHECK:       br label {{%?}}[[END]]
// CHECK:       [[END]]






// Check that the offloading functions are emitted and that the parallel function
// is appropriately guarded.

// CHECK:       define internal void [[HVT1]]([[S1]]* {{%.+}}, i[[SZ]] [[PARM1:%.+]], i[[SZ]] [[PARM2:%.+]])
// CHECK-DAG:   store i[[SZ]] [[PARM1]], i[[SZ]]* [[B_ADDR:%.+]], align
// CHECK-DAG:   store i[[SZ]] [[PARM2]], i[[SZ]]* [[CAPE_ADDR:%.+]], align
// CHECK-64:    [[CONVB:%.+]] = bitcast i[[SZ]]* [[B_ADDR]] to i32*
// CHECK:       [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPE_ADDR]] to i8*
// CHECK-64:    [[BV:%.+]] = load i32, i32* [[CONVB]], align
// CHECK-32:    [[BV:%.+]] = load i32, i32* [[B_ADDR]], align
// CHECK-64:    [[BC:%.+]] = bitcast i64* [[ARGA:%.+]] to i32*
// CHECK-64:    store i32 [[BV]], i32* [[BC]], align
// CHECK-64:    [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[ARGA]], align
// CHECK-32:    store i32 [[BV]], i32* [[ARGA:%.+]], align
// CHECK-32:    [[ARG:%.+]] = load i[[SZ]], i[[SZ]]* [[ARGA]], align
// CHECK:       [[IFC:%.+]] = load i8, i8* [[CONV]], align
// CHECK:       [[TB:%.+]] = trunc i8 [[IFC]] to i1
// CHECK:       br i1 [[TB]], label {{%?}}[[IF_THEN:.+]], label {{%?}}[[IF_ELSE:.+]]
//
// CHECK:       [[IF_THEN]]
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [[S1]]*, i[[SZ]])* [[OMP_OUTLINED3:@.+]] to void (i32*, i32*, ...)*), [[S1]]* {{.+}}, i[[SZ]] [[ARG]])
// CHECK:       br label {{%?}}[[END:.+]]
//
// CHECK:       [[IF_ELSE]]
// CHECK:       call void @__kmpc_serialized_parallel(
// CHECK:       call void [[OMP_OUTLINED3]](i32* {{%.+}}, i32* {{%.+}}, [[S1]]* {{.+}}, i[[SZ]] [[ARG]])
// CHECK:       call void @__kmpc_end_serialized_parallel(
// CHECK:       br label {{%?}}[[END]]
//
// CHECK:       [[END]]
//
//


// CHECK:       define internal void [[HVT2]]([[S1]]* {{%.+}}, i[[SZ]] [[PARM:%.+]])
// CHECK-DAG:   store i[[SZ]] [[PARM]], i[[SZ]]* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPE_ADDR]] to i8*
// CHECK:       [[IFC:%.+]] = load i8, i8* [[CONV]], align
// CHECK:       [[TB:%.+]] = trunc i8 [[IFC]] to i1
// CHECK:       br i1 [[TB]], label {{%?}}[[IF_THEN:.+]], label {{%?}}[[IF_ELSE:.+]]
//
// CHECK:       [[IF_THEN]]
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [[S1]]*)* [[OMP_OUTLINED4:@.+]] to void (i32*, i32*, ...)*), [[S1]]* {{.+}})
// CHECK:       br label {{%?}}[[END:.+]]
//
// CHECK:       [[IF_ELSE]]
// CHECK:       call void @__kmpc_serialized_parallel(
// CHECK:       call void [[OMP_OUTLINED4]](i32* {{%.+}}, i32* {{%.+}}, [[S1]]* {{.+}})
// CHECK:       call void @__kmpc_end_serialized_parallel(
// CHECK:       br label {{%?}}[[END]]
//
// CHECK:       [[END]]
//
//








// CHECK:       define internal void [[HVT3]](i[[SZ]] [[PARM:%.+]])
// CHECK-DAG:   store i[[SZ]] [[PARM]], i[[SZ]]* [[CAPE_ADDR:%.+]], align
// CHECK:       [[CONV:%.+]] = bitcast i[[SZ]]* [[CAPE_ADDR]] to i8*
// CHECK:       [[IFC:%.+]] = load i8, i8* [[CONV]], align
// CHECK:       [[TB:%.+]] = trunc i8 [[IFC]] to i1
// CHECK:       br i1 [[TB]], label {{%?}}[[IF_THEN:.+]], label {{%?}}[[IF_ELSE:.+]]
//
// CHECK:       [[IF_THEN]]
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* [[OMP_OUTLINED1:@.+]] to void (i32*, i32*, ...)*))
// CHECK:       br label {{%?}}[[END:.+]]
//
// CHECK:       [[IF_ELSE]]
// CHECK:       call void @__kmpc_serialized_parallel(
// CHECK:       call void [[OMP_OUTLINED1]](i32* {{%.+}}, i32* {{%.+}})
// CHECK:       call void @__kmpc_end_serialized_parallel(
// CHECK:       br label {{%?}}[[END]]
//
// CHECK:       [[END]]
//
//
// CHECK:       define internal void [[HVT4]]()
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* [[OMP_OUTLINED2:@.+]] to void (i32*, i32*, ...)*))
// CHECK-NEXT:  ret
//
//





// CHECK:       define internal void [[HVT5]](
// CHECK-NOT:   @__kmpc_fork_call
// CHECK:       call void @__kmpc_serialized_parallel(
// CHECK:       call void [[OMP_OUTLINED5:@.+]](i32* {{%.+}}, i32* {{%.+}}, i[[SZ]] {{.+}})
// CHECK:       call void @__kmpc_end_serialized_parallel(
// CHECK:       ret
//
//


// CHECK:       define internal void [[HVT6]](
// CHECK-NOT:   call void @__kmpc_serialized_parallel(
// CHECK-NOT:   call void [[OMP_OUTLINED5:@.+]](i32* {{%.+}}, i32* {{%.+}}, i[[SZ]] {{.+}})
// CHECK-NOT:   call void @__kmpc_end_serialized_parallel(
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i[[SZ]], i[[SZ]])* [[OMP_OUTLINED5:@.+]] to void (i32*, i32*, ...)*),
// CHECK:       ret
//
//



#endif
