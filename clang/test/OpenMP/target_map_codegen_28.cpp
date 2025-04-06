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
// CK29: [[SIZE00:@.+]] = private {{.*}}constant [2 x i64] [i64 0, i64 80]
// CK29: [[MTYPE00:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710675]

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[SIZE01:@.+]] = private {{.*}}constant [3 x i64] [i64 0, i64 {{8|4}}, i64 80]
// CK29: [[MTYPE01:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710672, i64 19]

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[SIZE02:@.+]] = private {{.*}}constant [2 x i64] [i64 0, i64 80]
// CK29: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710675]

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
// CK29-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK29-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK29-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK29-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK29-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK29-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK29-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]

// CK29-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK29-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK29-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK29-DAG: store ptr [[VAR00:%.+]], ptr [[P0]]
// CK29-DAG: store i64 %{{.+}}, ptr [[S0]]
// CK29-DAG: [[VAR0]] = load ptr, ptr %
// CK29-DAG: [[VAR00]] = getelementptr inbounds nuw [[SSB]], ptr [[VAR0]], i32 0, i32 0

// CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: store ptr [[VAR1:%.+]], ptr [[BP2]]
// CK29-DAG: store ptr [[VAR2:%.+]], ptr [[P2]]
// CK29-DAG: [[VAR1]] = getelementptr inbounds nuw [[SSA]], ptr %{{.+}}, i32 0, i32 1
// CK29-DAG: [[VAR2]] = getelementptr inbounds nuw double, ptr [[VAR22:%.+]], i{{.+}} 0
// CK29-DAG: [[VAR22]] = load ptr, ptr %{{.+}},

// CK29: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(p->pr[:10])
    {
      p->pr++;
    }

// Region 01
// CK29-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK29-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK29-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK29-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK29-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK29-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK29-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]

// CK29-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK29-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK29-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: store ptr [[VAR0]], ptr [[BP0]]
// CK29-DAG: store ptr [[VAR000:%.+]], ptr [[P0]]
// CK29-DAG: store i64 %{{.+}}, ptr [[S0]]
// CK29-DAG: [[VAR000]] = getelementptr inbounds nuw [[SSB]], ptr [[VAR0]], i32 0, i32 1

// CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: store ptr [[VAR000]], ptr [[BP1]]
// CK29-DAG: store ptr [[VAR1:%.+]], ptr [[P1]]
// CK29-DAG: [[VAR1]] = getelementptr inbounds nuw [[SSA]], ptr %{{.+}}, i32 0, i32 0

// CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK29-DAG: store ptr [[VAR1]], ptr [[BP2]]
// CK29-DAG: store ptr [[VAR2:%.+]], ptr [[P2]]
// CK29-DAG: [[VAR2]] = getelementptr inbounds nuw double, ptr [[VAR22:%.+]], i{{.+}} 0
// CK29-DAG: [[VAR22]] = load ptr, ptr %{{.+}},

// CK29: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(pr->p[:10])
    {
      pr->p++;
    }

// Region 02
// CK29-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK29-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK29-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK29-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK29-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK29-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK29-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]

// CK29-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK29-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK29-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK29-DAG: store ptr [[VAR0]], ptr [[BP0]]
// CK29-DAG: store ptr [[VAR000:%.+]], ptr [[P0]]
// CK29-DAG: store i64 %{{.+}}, ptr [[S0]]
// CK29-DAG: [[VAR000]] = getelementptr inbounds nuw [[SSB]], ptr [[VAR0]], i32 0, i32 1

// CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK29-DAG: store ptr [[VAR1:%.+]], ptr [[BP2]]
// CK29-DAG: store ptr [[VAR2:%.+]], ptr [[P2]]
// CK29-DAG: [[VAR1]] = getelementptr inbounds nuw [[SSA]], ptr %{{.+}}, i32 0, i32 1
// CK29-DAG: [[VAR2]] = getelementptr inbounds nuw double, ptr [[VAR22:%.+]], i{{.+}} 0
// CK29-DAG: [[VAR22]] = load ptr, ptr %{{.+}},

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
