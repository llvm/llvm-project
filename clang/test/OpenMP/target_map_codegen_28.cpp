// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32

// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32

// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32

// RUN: %clang_cc1 -DCK29 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY28 %s
// RUN: %clang_cc1 -DCK29 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY28 %s
// RUN: %clang_cc1 -DCK29 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY28 %s
// RUN: %clang_cc1 -DCK29 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY28 %s
// SIMD-ONLY28-NOT: {{__kmpc|__tgt}}
#ifdef CK29

// CK29: [[SSA:%.+]] = type { ptr, ptr }
// CK29: [[SSB:%.+]]  = type { ptr, ptr }

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[SIZE00:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|16}}, i64 80, i64 {{4|8}}]
// CK29: [[MTYPE00:@.+]] = private {{.*}}constant [3 x i64] [i64 [[#0x223]], i64 3, i64 [[#0x8000]]]

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[SIZE01:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|16}}, i64 80, i64 {{4|8}}]
// CK29: [[MTYPE01:@.+]] = private {{.*}}constant [3 x i64] [i64 [[#0x223]], i64 3, i64 [[#0x8000]]]

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[SIZE02:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|16}}, i64 80, i64 {{4|8}}]
// CK29: [[MTYPE02:@.+]] = private {{.*}}constant [3 x i64] [i64 [[#0x223]], i64 3, i64 [[#0x8000]]]

struct SSA{
  double *p;
  double *&pr;
  SSA(double *&pr) : pr(pr) {}
};

struct SSB{
  SSA *p;
  SSA *&pr;
  SSB(SSA *&pr) : pr(pr) {}

  // CK29-LABEL: define {{.+}}foo
  void foo() {

// Region 00

// &this[0],  &this[0],  sizeof(this[0]),       TO | FROM | IMPLICIT
// &p->pr[0], &p->pr[0], 10 * sizeof(p->pr[0]), TO | FROM
// &p->pr,    &p->pr[0], sizeof(void*),         ATTACH

// CK29-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK29-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK29-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK29-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK29-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]

// CK29-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK29-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: store ptr [[THIS:%.+]], ptr [[BP0]]
// CK29-DAG: store ptr [[THIS]], ptr [[P0]]

// CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: store ptr [[PR_DEREF_LOAD:%.+]], ptr [[BP1]]
// CK29-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]

// CK29-DAG: [[PR_DEREF_LOAD]] = load ptr, ptr [[PR_DEREF:%[^,]+]]
// CK29-DAG: [[PR_DEREF]] = load ptr, ptr [[PR:%[^,]+]]
// CK29-DAG: [[PR]] = getelementptr inbounds nuw %struct.SSA, ptr [[P_LOAD:%.+]], i32 0, i32 1
// CK29-DAG: [[P_LOAD]] = load ptr, ptr [[THIS_P:%[^,]+]]
// CK29-DAG: [[THIS_P]] = getelementptr inbounds nuw %struct.SSB, ptr [[THIS]], i32 0, i32 0

// CK29-DAG: [[SEC1]] = getelementptr inbounds nuw double, ptr [[PR_DEREF_LOAD1:%.+]], i{{.*}} 0
// CK29-DAG: [[PR_DEREF_LOAD1]] = load ptr, ptr [[PR_DEREF1:%[^,]+]]
// CK29-DAG: [[PR_DEREF]] = load ptr, ptr [[PR1:%[^,]+]]
// CK29-DAG: [[PR1]] = getelementptr inbounds nuw %struct.SSA, ptr [[P_LOAD1:%.+]], i32 0, i32 1
// CK29-DAG: [[P_LOAD1]] = load ptr, ptr [[THIS_P1:%[^,]+]]
// CK29-DAG: [[THIS_P1]] = getelementptr inbounds nuw %struct.SSB, ptr [[THIS]], i32 0, i32 0

// CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK29-DAG: store ptr [[PR_DEREF:%.+]], ptr [[BP2]]
// CK29-DAG: store ptr [[SEC1:%.+]], ptr [[P2]]

// CK29: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->pr[:10])
    {
      p->pr++;
    }

// Region 01

// &this[0],  &this[0],  sizeof(this[0]),       TO | FROM | IMPLICIT
// &pr->p[0], &pr->p[0], 10 * sizeof(pr->p[0]), TO | FROM
// &pr->p,    &pr->p[0], sizeof(void*),         ATTACH

// CK29-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK29-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK29-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK29-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK29-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]

// CK29-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK29-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: store ptr [[THIS:%.+]], ptr [[BP0]]
// CK29-DAG: store ptr [[THIS]], ptr [[P0]]

// CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: store ptr [[PR_P_LOAD:%.+]], ptr [[BP1]]
// CK29-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]

// CK29-DAG: [[PR_P_LOAD]] = load ptr, ptr [[PR_P:%[^,]+]]
// CK29-DAG: [[PR_P]] = getelementptr inbounds nuw %struct.SSA, ptr [[PR_DEREF_LOAD:%.+]], i32 0, i32 0
// CK29-DAG: [[PR_DEREF_LOAD]] = load ptr, ptr [[PR_DEREF:%[^,]+]]
// CK29-DAG: [[PR_DEREF]] = load ptr, ptr [[PR:%[^,]+]]
// CK29-DAG: [[PR]] = getelementptr inbounds nuw %struct.SSB, ptr [[THIS]], i32 0, i32 1

// CK29-DAG: [[SEC1]] = getelementptr inbounds nuw double, ptr [[PR_P_LOAD1:%.+]], i{{.*}} 0
// CK29-DAG: [[PR_P_LOAD1]] = load ptr, ptr [[PR_P1:%[^,]+]]
// CK29-DAG: [[PR_P1]] = getelementptr inbounds nuw %struct.SSA, ptr [[PR_DEREF_LOAD1:%.+]], i32 0, i32 0
// CK29-DAG: [[PR_DEREF_LOAD1]] = load ptr, ptr [[PR_DEREF1:%[^,]+]]
// CK29-DAG: [[PR_DEREF1]] = load ptr, ptr [[PR1:%[^,]+]]
// CK29-DAG: [[PR1]] = getelementptr inbounds nuw %struct.SSB, ptr [[THIS]], i32 0, i32 1

// CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK29-DAG: store ptr [[PR_P]], ptr [[BP2]]
// CK29-DAG: store ptr [[SEC1:%.+]], ptr [[P2]]

// CK29: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(pr->p[:10])
    {
      pr->p++;
    }

// Region 02

// &this[0],   &this[0],   sizeof(this[0]),        TO | FROM | IMPLICIT
// &pr->pr[0], &pr->pr[0], 10 * sizeof(pr->pr[0]), TO | FROM
// &pr->pr,    &pr->pr[0], sizeof(void*),          ATTACH

// CK29-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK29-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK29-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK29-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK29-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]

// CK29-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK29-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: store ptr [[THIS:%.+]], ptr [[BP0]]
// CK29-DAG: store ptr [[THIS]], ptr [[P0]]

// CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: store ptr [[PR_PR_DEREF_LOAD:%.+]], ptr [[BP1]]
// CK29-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]

// CK29-DAG: [[PR_PR_DEREF_LOAD]] = load ptr, ptr [[PR_PR_DEREF:%[^,]+]]
// CK29-DAG: [[PR_PR_DEREF]] = load ptr, ptr [[PR_PR:%[^,]+]]
// CK29-DAG: [[PR_PR]] = getelementptr inbounds nuw %struct.SSA, ptr [[PR_DEREF_LOAD:%.+]], i32 0, i32 1
// CK29-DAG: [[PR_DEREF_LOAD]] = load ptr, ptr [[PR_DEREF:%[^,]+]]
// CK29-DAG: [[PR_DEREF]] = load ptr, ptr [[PR:%[^,]+]]
// CK29-DAG: [[PR]] = getelementptr inbounds nuw %struct.SSB, ptr [[THIS]], i32 0, i32 1

// CK29-DAG: [[SEC1]] = getelementptr inbounds nuw double, ptr [[PR_PR_DEREF_LOAD1:%.+]], i{{.*}} 0
// CK29-DAG: [[PR_PR_DEREF_LOAD1]] = load ptr, ptr [[PR_PR_DEREF1:%[^,]+]]
// CK29-DAG: [[PR_PR_DEREF1]] = load ptr, ptr [[PR_PR1:%[^,]+]]
// CK29-DAG: [[PR_PR1]] = getelementptr inbounds nuw %struct.SSA, ptr [[PR_DEREF_LOAD1:%.+]], i32 0, i32 1
// CK29-DAG: [[PR_DEREF_LOAD1]] = load ptr, ptr [[PR_DEREF1:%[^,]+]]
// CK29-DAG: [[PR_DEREF1]] = load ptr, ptr [[PR1:%[^,]+]]
// CK29-DAG: [[PR1]] = getelementptr inbounds nuw %struct.SSB, ptr [[THIS]], i32 0, i32 1

// CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK29-DAG: store ptr [[PR_PR_DEREF:%.+]], ptr [[BP2]]
// CK29-DAG: store ptr [[SEC1:%.+]], ptr [[P2]]

// CK29: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(pr->pr[:10])
    {
      pr->pr++;
    }
  }
};

void explicit_maps_member_pointer_references(SSA *sap) {
  double *d;
  SSA sa(d);
  SSB sb(sap);
  sb.foo();
}
#endif // CK29
#endif
