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

// CK29: [[SSA:%.+]] = type { double*, double** }
// CK29: [[SSB:%.+]]  = type { [[SSA]]*, [[SSA]]** }

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[MTYPE00:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710675]

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[MTYPE01:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710672, i64 19]

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
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
    // CK29-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z:64|32]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null, i8** null)

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK29-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SSB]]**
    // CK29-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SSA]]***
    // CK29-DAG: store [[SSB]]* [[VAR0:%.+]], [[SSB]]** [[CBP0]]
    // CK29-DAG: store [[SSA]]** [[VAR00:%.+]], [[SSA]]*** [[CP0]]
    // CK29-DAG: store i64 %{{.+}}, i64* [[S0]]
    // CK29-DAG: [[VAR0]] = load [[SSB]]*, [[SSB]]** %
    // CK29-DAG: [[VAR00]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 0

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double****
    // CK29-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
    // CK29-DAG: store double*** [[VAR1:%.+]], double**** [[CBP2]]
    // CK29-DAG: store double* [[VAR2:%.+]], double** [[CP2]]
    // CK29-DAG: store i64 80, i64* [[S2]]
    // CK29-DAG: [[VAR1]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 1
    // CK29-DAG: [[VAR2]] = getelementptr inbounds double, double* [[VAR22:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR22]] = load double*, double** %{{.+}},

    // CK29: call void [[CALL00:@.+]]([[SSB]]* {{[^,]+}})
    #pragma omp target map(p->pr[:10])
    {
      p->pr++;
    }

    // Region 01
    // CK29-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null, i8** null)

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK29-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SSB]]**
    // CK29-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SSA]]****
    // CK29-DAG: store [[SSB]]* [[VAR0]], [[SSB]]** [[CBP0]]
    // CK29-DAG: store [[SSA]]*** [[VAR000:%.+]], [[SSA]]**** [[CP0]]
    // CK29-DAG: store i64 %{{.+}}, i64* [[S0]]
    // CK29-DAG: [[VAR000]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 1

    // CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SSA]]****
    // CK29-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double***
    // CK29-DAG: store [[SSA]]*** [[VAR000]], [[SSA]]**** [[CBP1]]
    // CK29-DAG: store double** [[VAR1:%.+]], double*** [[CP1]]
    // CK29-DAG: store i64 {{8|4}}, i64* [[S1]]
    // CK29-DAG: [[VAR1]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 0

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double***
    // CK29-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
    // CK29-DAG: store double** [[VAR1]], double*** [[CBP2]]
    // CK29-DAG: store double* [[VAR2:%.+]], double** [[CP2]]
    // CK29-DAG: store i64 80, i64* [[S2]]
    // CK29-DAG: [[VAR2]] = getelementptr inbounds double, double* [[VAR22:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR22]] = load double*, double** %{{.+}},

    // CK29: call void [[CALL00:@.+]]([[SSB]]* {{[^,]+}})
    #pragma omp target map(pr->p[:10])
    {
      pr->p++;
    }

    // Region 02
    // CK29-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null, i8** null)

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK29-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SSB]]**
    // CK29-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SSA]]****
    // CK29-DAG: store [[SSB]]* [[VAR0]], [[SSB]]** [[CBP0]]
    // CK29-DAG: store [[SSA]]*** [[VAR000:%.+]], [[SSA]]**** [[CP0]]
    // CK29-DAG: store i64 %{{.+}}, i64* [[S0]]
    // CK29-DAG: [[VAR000]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 1

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double****
    // CK29-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
    // CK29-DAG: store double*** [[VAR1:%.+]], double**** [[CBP2]]
    // CK29-DAG: store double* [[VAR2:%.+]], double** [[CP2]]
    // CK29-DAG: store i64 80, i64* [[S2]]
    // CK29-DAG: [[VAR1]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 1
    // CK29-DAG: [[VAR2]] = getelementptr inbounds double, double* [[VAR22:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR22]] = load double*, double** %{{.+}},

    // CK29: call void [[CALL00:@.+]]([[SSB]]* {{[^,]+}})
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
