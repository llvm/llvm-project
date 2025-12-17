// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1

double *g;

// CK1: @g ={{.*}} global ptr
// CK1: [[MTYPE00:@.+]] = {{.*}}constant [2 x i64] [i64 67, i64 32768]
// CK1: [[MTYPE01:@.+]] = {{.*}}constant [2 x i64] [i64 67, i64 32768]
// CK1: [[MTYPE03:@.+]] = {{.*}}constant [2 x i64] [i64 67, i64 32768]
// CK1: [[MTYPE04:@.+]] = {{.*}}constant [2 x i64] [i64 67, i64 32768]
// CK1: [[MTYPE05:@.+]] = {{.*}}constant [2 x i64] [i64 67, i64 32768]
// CK1: [[MTYPE06:@.+]] = {{.*}}constant [2 x i64] [i64 67, i64 32768]
// CK1: [[MTYPE07:@.+]] = {{.*}}constant [2 x i64] [i64 67, i64 32768]
// CK1: [[MTYPE08:@.+]] = {{.*}}constant [4 x i64] [i64 67, i64 32768, i64 3, i64 32768]
// CK1: [[MTYPE09:@.+]] = {{.*}}constant [4 x i64] [i64 67, i64 32768, i64 67, i64 32768]
// CK1: [[MTYPE10:@.+]] = {{.*}}constant [4 x i64] [i64 67, i64 32768, i64 67, i64 32768]
// CK1: [[MTYPE11:@.+]] = {{.*}}constant [3 x i64] [i64 3, i64 32768, i64 64]
// CK1: [[MTYPE12:@.+]] = {{.*}}constant [3 x i64] [i64 3, i64 32768, i64 64]

// CK1-LABEL: @_Z3foo
template<typename T>
void foo(float *&lr, T *&tr) {
  float *l;
  T *t;

  // &g[0], &g[0], 10 * sizeof(g[0]), TO | FROM | RETURN_PARAM
  // &g,    &g[0], sizeof(void*),     ATTACH
  //
  // CK1:     [[T:%.+]] = load ptr, ptr [[DECL:@g]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
  // CK1:     store ptr [[T]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE00]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1-NOT: store ptr [[VAL]], ptr [[DECL]],
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[TT:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw double, ptr [[TT]], i32 1
  #pragma omp target data map(g[0:10]) use_device_ptr(g)
  {
    ++g;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE00]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw double, ptr [[TTT]], i32 1
  ++g;

  // &l[0], &l[0], 10 * sizeof(l[0]), TO | FROM | RETURN_PARAM
  // &l,    &l[0], sizeof(void*),     ATTACH
  //
  // CK1:     [[T1:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE01]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1-NOT: store ptr [[VAL]], ptr [[DECL]],
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TT1]], i32 1
  #pragma omp target data map(l[0:10]) use_device_ptr(l)
  {
    ++l;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE01]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TTT]], i32 1
  ++l;

  // CK1-NOT: call void @__tgt_target
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TTT]], i32 1
  #pragma omp target data map(l[:10]) use_device_ptr(l) if(0)
  {
    ++l;
  }
  // CK1-NOT: call void @__tgt_target
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TTT]], i32 1
  ++l;

  // CK1:     [[T1:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE03]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1-NOT: store ptr [[VAL]], ptr [[DECL]],
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TT1]], i32 1
  #pragma omp target data map(l[:10]) use_device_ptr(l) if(1)
  {
    ++l;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE03]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TTT]], i32 1
  ++l;

  // CK1:     [[CMP:%.+]] = icmp ne ptr %{{.+}}, null
  // CK1:     br i1 [[CMP]], label %[[BTHEN:.+]], label %[[BELSE:.+]]

  // CK1:     [[BTHEN]]:
  // CK1:     [[T1:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE04]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1-NOT: store ptr [[VAL]], ptr [[DECL]],
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TT1]], i32 1
  // CK1:     br label %[[BEND:.+]]

  // CK1:     [[BELSE]]:
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TTT]], i32 1
  // CK1:     br label %[[BEND]]
  #pragma omp target data map(l[:10]) use_device_ptr(l) if(lr != 0)
  {
    ++l;
  }
  // CK1:     [[BEND]]:
  // CK1:     br i1 [[CMP]], label %[[BTHEN:.+]], label %[[BELSE:.+]]

  // CK1:     [[BTHEN]]:
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE04]]
  // CK1:     br label %[[BEND:.+]]

  // CK1:     [[BELSE]]:
  // CK1:     br label %[[BEND]]

  // CK1:     [[BEND]]:
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TTT]], i32 1
  ++l;

  // &(ptee(lr)[0]), &(ptee(lr)[0]), 10 * sizeof(lr[0]), TO | FROM | RETURN_PARAM
  // &(ptee(lr)),    &(ptee(lr)[0]), sizeof(void*),      ATTACH
  //
  // CK1:     [[T2:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[T1:%.+]] = load ptr, ptr [[T2]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE05]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1:     store ptr [[VAL]], ptr [[PVTV:%.+]],
  // CK1-NOT: store ptr [[PVTV]], ptr [[DECL]],
  // CK1:     store ptr [[PVTV]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     [[TT2:%.+]] = load ptr, ptr [[TT1]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TT2]], i32 1
  #pragma omp target data map(lr[:10]) use_device_ptr(lr)
  {
    ++lr;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE05]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     [[TTTT:%.+]] = load ptr, ptr [[TTT]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TTTT]], i32 1
  ++lr;

  // &t[0], &t[0], 10 * sizeof(t[0]), TO | FROM | RETURN_PARAM
  // &t,    &t[1], sizeof(void*),     ATTACH
  //
  // CK1:     [[T1:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE06]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1-NOT: store ptr [[VAL]], ptr [[DECL]],
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TT1]], i32 1
  #pragma omp target data map(t[:10]) use_device_ptr(t)
  {
    ++t;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE06]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TTT]], i32 1
  ++t;

  // &(ptee(tr)[0]), &(ptee(tr)[0]), 10 * sizeof(tr[0]), TO | FROM | RETURN_PARAM
  // &(ptee(tr)),    &(ptee(tr)[0]), sizeof(void*),      ATTACH
  //
  // CK1:     [[T2:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[T1:%.+]] = load ptr, ptr [[T2]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE07]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1:     store ptr [[VAL]], ptr [[PVTV:%.+]],
  // CK1-NOT: store ptr [[PVTV]], ptr [[DECL]],
  // CK1:     store ptr [[PVTV]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     [[TT2:%.+]] = load ptr, ptr [[TT1]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TT2]], i32 1
  #pragma omp target data map(tr[:10]) use_device_ptr(tr)
  {
    ++tr;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE07]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     [[TTTT:%.+]] = load ptr, ptr [[TTT]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TTTT]], i32 1
  ++tr;

  // &l[0], &l[0], 10 * sizeof(l[0]), TO | FROM | RETURN_PARAM
  // &l,    &l[0], sizeof(void*),     ATTACH
  // &t[0], &t[0], 10 * sizeof(t[0]), TO | FROM
  // &t,    &t[0], sizeof(void*),     ATTACH
  //
  // CK1:     [[T1:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [4 x ptr], ptr %{{.+}}, i32 0, i32 0
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE08]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1-NOT: store ptr [[VAL]], ptr [[DECL]],
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TT1]], i32 1
  #pragma omp target data map(l[:10], t[:10]) use_device_ptr(l)
  {
    ++l; ++t;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE08]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw float, ptr [[TTT]], i32 1
  ++l; ++t;


  // &l[0], &l[0], 10 * sizeof(l[0]), TO | FROM | RETURN_PARAM
  // &l,    &l[0], sizeof(void*),     ATTACH
  // &t[0], &t[0], 10 * sizeof(t[0]), TO | FROM | RETURN_PARAM
  // &t,    &t[0], sizeof(void*),     ATTACH
  //
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE09]]
  // CK1:     [[_VAL:%.+]] = load ptr, ptr {{%.+}},
  // CK1:     store ptr [[_VAL]], ptr [[_PVT:%.+]],
  // CK1:     [[VAL:%.+]] = load ptr, ptr {{%.+}},
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[_TT1:%.+]] = load ptr, ptr [[_PVT]],
  // CK1:     getelementptr inbounds nuw float, ptr [[_TT1]], i32 1
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TT1]], i32 1
  #pragma omp target data map(l[:10], t[:10]) use_device_ptr(l) use_device_ptr(t)
  {
    ++l; ++t;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE09]]
  // CK1:     [[_TTT:%.+]] = load ptr, ptr {{%.+}},
  // CK1:     getelementptr inbounds nuw float, ptr [[_TTT]], i32 1
  // CK1:     [[TTT:%.+]] = load ptr, ptr {{%.+}},
  // CK1:     getelementptr inbounds nuw i32, ptr [[TTT]], i32 1
  ++l; ++t;

  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE10]]
  // CK1:     [[_VAL:%.+]] = load ptr, ptr {{%.+}},
  // CK1:     store ptr [[_VAL]], ptr [[_PVT:%.+]],
  // CK1:     [[VAL:%.+]] = load ptr, ptr {{%.+}},
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[_TT1:%.+]] = load ptr, ptr [[_PVT]],
  // CK1:     getelementptr inbounds nuw float, ptr [[_TT1]], i32 1
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TT1]], i32 1
  #pragma omp target data map(l[:10], t[:10]) use_device_ptr(l,t)
  {
    ++l; ++t;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE10]]
  // CK1:     [[_TTT:%.+]] = load ptr, ptr {{%.+}},
  // CK1:     getelementptr inbounds nuw float, ptr [[_TTT]], i32 1
  // CK1:     [[TTT:%.+]] = load ptr, ptr {{%.+}},
  // CK1:     getelementptr inbounds nuw i32, ptr [[TTT]], i32 1
  ++l; ++t;

  // &l[0], &l[0], 10 * sizeof(l[0]), TO | FROM
  // &l,    &l[0], sizeof(void*),     ATTACH
  // &t[0], &t[0], 0,                 RETURN_PARAM
  //
  // CK1:     [[T1:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [3 x ptr], ptr %{{.+}}, i32 0, i32 2
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE11]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1-NOT: store ptr [[VAL]], ptr [[DECL]],
  // CK1:     store ptr [[VAL]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TT1]], i32 1
  #pragma omp target data map(l[:10]) use_device_ptr(t)
  {
    ++l; ++t;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE11]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TTT]], i32 1
  ++l; ++t;

  // &l[0],            &l[0],            10 * sizeof(l[0]), TO | FROM
  // &l,               &l[0],            sizeof(void*),     ATTACH
  // &ref_ptee(tr)[0], &ref_ptee(tr)[0], 0,                 RETURN_PARAM
  //
  // CK1:     [[T2:%.+]] = load ptr, ptr [[DECL:%.+]],
  // CK1:     [[T1:%.+]] = load ptr, ptr [[T2]],
  // CK1:     [[BP:%.+]] = getelementptr inbounds [3 x ptr], ptr %{{.+}}, i32 0, i32 2
  // CK1:     store ptr [[T1]], ptr [[BP]],
  // CK1:     call void @__tgt_target_data_begin{{.+}}[[MTYPE12]]
  // CK1:     [[VAL:%.+]] = load ptr, ptr [[BP]],
  // CK1:     store ptr [[VAL]], ptr [[PVTV:%.+]],
  // CK1-NOT: store ptr [[PVTV]], ptr [[DECL]],
  // CK1:     store ptr [[PVTV]], ptr [[PVT:%.+]],
  // CK1:     [[TT1:%.+]] = load ptr, ptr [[PVT]],
  // CK1:     [[TT2:%.+]] = load ptr, ptr [[TT1]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TT2]], i32 1
  #pragma omp target data map(l[:10]) use_device_ptr(tr)
  {
    ++l; ++tr;
  }
  // CK1:     call void @__tgt_target_data_end{{.+}}[[MTYPE12]]
  // CK1:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
  // CK1:     [[TTTT:%.+]] = load ptr, ptr [[TTT]],
  // CK1:     getelementptr inbounds nuw i32, ptr [[TTTT]], i32 1
  ++l; ++tr;

}

void bar(float *&a, int *&b) {
  foo<int>(a,b);
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2: [[ST:%.+]] = type { ptr, ptr }
// CK2: [[MTYPE00:@.+]] = {{.*}}constant [2 x i64] [i64 [[#0x43]], i64 [[#0x8000]]]
// CK2: [[MTYPE01:@.+]] = {{.*}}constant [2 x i64] [i64 [[#0x43]], i64 [[#0x8000]]]
// CK2: [[MTYPE02:@.+]] = {{.*}}constant [3 x i64] [i64 3, i64 [[#0x8000]], i64 [[#0x40]]]
// CK2: [[MTYPE03:@.+]] = {{.*}}constant [3 x i64] [i64 [[#0x43]], i64 [[#0x8000]], i64 [[#0x40]]]

template <typename T>
struct ST {
  T *a;
  double *&b;
  ST(double *&b) : a(0), b(b) {}

  // CK2-LABEL: @{{.*}}foo{{.*}}
  void foo(double *&arg) {
    int *la = 0;

    // &a[0], &a[0], 10 * sizeof(a[0]), TO | FROM | RETURN_PARAM
    // &a,    &a[0], sizeof(void*),     ATTACH
    //
    // CK2:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
    // CK2:     store ptr [[RVAL:%.+]], ptr [[BP]],
    // CK2:     call void @__tgt_target_data_begin{{.+}}[[MTYPE00]]
    // CK2:     [[VAL:%.+]] = load ptr, ptr [[BP]],
    // CK2:     store ptr [[VAL]], ptr [[PVT:%.+]],
    // CK2:     store ptr [[PVT]], ptr [[PVT2:%.+]],
    // CK2:     [[TT1:%.+]] = load ptr, ptr [[PVT2]],
    // CK2:     [[TT2:%.+]] = load ptr, ptr [[TT1]],
    // CK2:     getelementptr inbounds nuw double, ptr [[TT2]], i32 1
    #pragma omp target data map(a[:10]) use_device_ptr(a)
    {
      a++;
    }
    // CK2:     call void @__tgt_target_data_end{{.+}}[[MTYPE00]]
    // CK2:     [[DECL:%.+]] = getelementptr inbounds nuw [[ST]], ptr %this1, i32 0, i32 0
    // CK2:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
    // CK2:     getelementptr inbounds nuw double, ptr [[TTT]], i32 1
    a++;

    // &ptee(b)[0], &ptee(b)[0], 10 * sizeof(ptee(b)[0]), TO | FROM | RETURN_PARAM
    // &ptee(b),    &ptee(b)[0], sizeof(void*),           ATTACH
    //
    // CK2:     [[BP:%.+]] = getelementptr inbounds [2 x ptr], ptr %{{.+}}, i32 0, i32 0
    // CK2:     store ptr [[RVAL:%.+]], ptr [[BP]],
    // CK2:     call void @__tgt_target_data_begin{{.+}}[[MTYPE01]]
    // CK2:     [[VAL:%.+]] = load ptr, ptr [[BP]],
    // CK2:     store ptr [[VAL]], ptr [[PVT:%.+]],
    // CK2:     store ptr [[PVT]], ptr [[PVT2:%.+]],
    // CK2:     [[TT1:%.+]] = load ptr, ptr [[PVT2]],
    // CK2:     [[TT2:%.+]] = load ptr, ptr [[TT1]],
    // CK2:     getelementptr inbounds nuw double, ptr [[TT2]], i32 1
    #pragma omp target data map(b[:10]) use_device_ptr(b)
    {
      b++;
    }
    // CK2:     call void @__tgt_target_data_end{{.+}}[[MTYPE01]]
    // CK2:     [[DECL:%.+]] = getelementptr inbounds nuw [[ST]], ptr %{{.+}}, i32 0, i32 1
    // CK2:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
    // CK2:     [[TTTT:%.+]] = load ptr, ptr [[TTT]],
    // CK2:     getelementptr inbounds nuw double, ptr [[TTTT]], i32 1
    b++;

    // &la[0], &la[0], 10 * sizeof(la[0]), TO | FROM
    // &la,    &la[0], sizeof(void*),      ATTACH
    // &a[0],  &a[0],  sizeof(this[0].a),  RETURN_PARAM
    //
    // CK2:     [[BP:%.+]] = getelementptr inbounds [3 x ptr], ptr %{{.+}}, i32 0, i32 2
    // CK2:     store ptr [[RVAL:%.+]], ptr [[BP]],
    // CK2:     call void @__tgt_target_data_begin{{.+}}[[MTYPE02]]
    // CK2:     [[VAL:%.+]] = load ptr, ptr [[BP]],
    // CK2:     store ptr [[VAL]], ptr [[PVT:%.+]],
    // CK2:     store ptr [[PVT]], ptr [[PVT2:%.+]],
    // CK2:     [[TT1:%.+]] = load ptr, ptr [[PVT2]],
    // CK2:     [[TT2:%.+]] = load ptr, ptr [[TT1]],
    // CK2:     getelementptr inbounds nuw double, ptr [[TT2]], i32 1
    #pragma omp target data map(la[:10]) use_device_ptr(a)
    {
      a++;
      la++;
    }
    // CK2:     call void @__tgt_target_data_end{{.+}}[[MTYPE02]]
    // CK2:     [[DECL:%.+]] = getelementptr inbounds nuw [[ST]], ptr %this1, i32 0, i32 0
    // CK2:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
    // CK2:     getelementptr inbounds nuw double, ptr [[TTT]], i32 1
    a++;
    la++;

    // &ptee(b)[0], &ptee(b)[0], 10 * sizeof(ptee(b)[0]), TO | FROM | RETURN_PARAM
    // &ptee(b),    &ptee(b)[0], sizeof(void*),           ATTACH
    // &a[0],       &a[0],       0,                       RETURN_PARAM
    //
    // CK2:     [[BP1:%.+]] = getelementptr inbounds [3 x ptr], ptr %{{.+}}, i32 0, i32 0
    // CK2:     store ptr [[RVAL1:%.+]], ptr [[BP1]],
    // CK2:     [[BP2:%.+]] = getelementptr inbounds [3 x ptr], ptr %{{.+}}, i32 0, i32 2
    // CK2:     store ptr [[RVAL2:%.+]], ptr [[BP2]],
    // CK2:     call void @__tgt_target_data_begin{{.+}}[[MTYPE03]]
    // CK2:     [[VAL1:%.+]] = load ptr, ptr [[BP1]],
    // CK2:     store ptr [[VAL1]], ptr [[PVT1:%.+]],
    // CK2:     [[VAL2:%.+]] = load ptr, ptr [[BP2]],
    // CK2:     store ptr [[VAL2]], ptr [[PVT2:%.+]],
    // CK2:     store ptr [[PVT2]], ptr [[_PVT2:%.+]],
    // CK2:     store ptr [[PVT1]], ptr [[_PVT1:%.+]],
    // CK2:     [[TT2:%.+]] = load ptr, ptr [[_PVT2]],
    // CK2:     [[_TT2:%.+]] = load ptr, ptr [[TT2]],
    // CK2:     getelementptr inbounds nuw double, ptr [[_TT2]], i32 1
    // CK2:     [[TT1:%.+]] = load ptr, ptr [[_PVT1]],
    // CK2:     [[_TT1:%.+]] = load ptr, ptr [[TT1]],
    // CK2:     getelementptr inbounds nuw double, ptr [[_TT1]], i32 1
    #pragma omp target data map(b[:10]) use_device_ptr(a, b)
    {
      a++;
      b++;
    }
    // CK2:     call void @__tgt_target_data_end{{.+}}[[MTYPE03]]
    // CK2:     [[DECL:%.+]] = getelementptr inbounds nuw [[ST]], ptr %this1, i32 0, i32 0
    // CK2:     [[TTT:%.+]] = load ptr, ptr [[DECL]],
    // CK2:     getelementptr inbounds nuw double, ptr [[TTT]], i32 1
    // CK2:     [[_DECL:%.+]] = getelementptr inbounds nuw [[ST]], ptr %this1, i32 0, i32 1
    // CK2:     [[_TTT:%.+]] = load ptr, ptr [[_DECL]],
    // CK2:     [[_TTTT:%.+]] = load ptr, ptr [[_TTT]],
    // CK2:     getelementptr inbounds nuw double, ptr [[_TTTT]], i32 1
    a++;
    b++;
  }
};

void bar(double *arg){
  ST<double> A(arg);
  A.foo(arg);
  ++arg;
}
#endif
#endif
