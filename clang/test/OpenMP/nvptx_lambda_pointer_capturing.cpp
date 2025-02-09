// Test host codegen only.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[ANON_T:%.+]] = type { ptr, ptr }
// CHECK-DAG: [[SIZES_TEMPLATE:@.+]] = private {{.+}} constant [5 x i[[PTRSZ:32|64]]] [i{{32|64}} 4, i{{32|64}} 4, i{{32|64}} {{8|16}}, i{{32|64}} 0, i{{32|64}} 0]
// CHECK-DAG: [[TYPES_TEMPLATE:@.+]] = private {{.+}} constant [5 x i64] [i64 800, i64 800, i64 673, i64 844424930132752, i64 844424930132752]
// CHECK-DAG: [[SIZES:@.+]] = private {{.+}} constant [3 x i[[PTRSZ:32|64]]] [i{{32|64}} {{8|16}}, i{{32|64}} 0, i{{32|64}} 0]
// CHECK-DAG: [[TYPES:@.+]] = private {{.+}} constant [3 x i64] [i64 673, i64 281474976711440, i64 281474976711440]
// CHECK-DAG: [[TYPES3:@.+]] = private {{.+}} constant [3 x i64] [i64 545, i64 281474976711440, i64 800]
// CHECK-DAG: [[TYPES11:@.+]] = private {{.+}} constant [5 x i64] [i64 800, i64 800, i64 549, i64 844424930132752, i64 844424930132752]
// CHECK-DAG: [[TYPES13:@.+]] = private {{.+}} constant [2 x i64] [i64 545, i64 281474976711440]
// CHECK-DAG: [[TYPES15:@.+]] = private {{.+}} constant [2 x i64] [i64 673, i64 281474976711440]

template <typename F>
void omp_loop(int start, int end, F body) {
#pragma omp target teams distribute parallel for
  for (int i = start; i < end; ++i) {
    body(i);
  }
}

template <typename F>
void omp_loop_ref(int start, int end, F& body) {
#pragma omp target teams distribute parallel for map(always, to: body)
  for (int i = start; i < end; ++i) {
    body(i);
  }
  int *p;
  const auto &body_ref = [=](int i) {p[i]=0;};
  #pragma omp target map(to: body_ref)
  body_ref(10);
  #pragma omp target
  body_ref(10);
}

template <class FTy>
struct C {
  static void xoo(const FTy& f) {
    int x = 10;
    #pragma omp target map(to:f)
      f(x);
  }
};

template <class FTy>
void zoo(const FTy &functor) {
  C<FTy>::xoo(functor);
}

// CHECK: define {{.*}}[[MAIN:@.+]](
int main()
{
  int* p = new int[100];
  int* q = new int[100];
  auto body = [=](int i){
    p[i] = q[i];
  };
  zoo([=](int i){p[i] = 0;});

#pragma omp target teams distribute parallel for
  for (int i = 0; i < 100; ++i) {
    body(i);
  }

// CHECK: [[BASE_PTRS:%.+]] = alloca [3 x ptr]{{.+}}
// CHECK: [[PTRS:%.+]] = alloca [3 x ptr]{{.+}}

// First gep of pointers inside lambdas to store the values across function call need to be ignored
// CHECK: {{%.+}} = getelementptr inbounds nuw [[ANON_T]], ptr %{{.+}}, i{{.+}} 0, i{{.+}} 0
// CHECK: {{%.+}} = getelementptr inbounds nuw [[ANON_T]], ptr %{{.+}}, i{{.+}} 0, i{{.+}} 1

// access of pointers inside lambdas
// CHECK: [[BASE_PTR1:%.+]] = getelementptr inbounds nuw [[ANON_T]], ptr %{{.+}}, i{{.+}} 0, i{{.+}} 0
// CHECK: [[PTR1:%.+]] = load ptr, ptr [[BASE_PTR1]]
// CHECK: [[BASE_PTR2:%.+]] = getelementptr inbounds nuw [[ANON_T]], ptr %{{.+}}, i{{.+}} 0, i{{.+}} 1
// CHECK: [[PTR2:%.+]] = load ptr, ptr [[BASE_PTR2]]

// storage of pointers in baseptrs and ptrs arrays
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 0
// CHECK: store ptr %{{.+}}, ptr [[LOC_LAMBDA]]{{.+}}
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PTRS]], i{{.+}} 0, i{{.+}} 0
// CHECK: store ptr %{{.+}}, ptr [[LOC_LAMBDA]]{{.+}}

// CHECK: [[LOC_PTR1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 1
// CHECK: store ptr [[BASE_PTR1]], ptr [[LOC_PTR1]]{{.+}}
// CHECK: [[LOC_PTR1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PTRS]], i{{.+}} 0, i{{.+}} 1
// CHECK: store ptr [[PTR1]], ptr [[LOC_PTR1]]{{.+}}


// CHECK: [[LOC_PTR2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 2
// CHECK: store ptr [[BASE_PTR2]], ptr [[LOC_PTR2]]{{.+}}
// CHECK: [[LOC_PTR2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PTRS]], i{{.+}} 0, i{{.+}} 2
// CHECK: store ptr [[PTR2]], ptr [[LOC_PTR2]]{{.+}}

  // actual target invocation
  // CHECK: [[BASES_GEP:%.+]] = getelementptr {{.+}} [3 x ptr], ptr [[BASE_PTRS]], {{.+}} 0, {{.+}} 0
  // CHECK: [[PTRS_GEP:%.+]] = getelementptr {{.+}} [3 x ptr], ptr [[PTRS]], {{.+}} 0, {{.+}} 0
  // CHECK: {{%.+}} = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 0, i32 0, ptr @.{{.+}}.region_id, ptr %{{.+}})

  omp_loop(0,100,body);
  omp_loop_ref(0,100,body);
}

// CHECK: [[BASE_PTRS:%.+]] = alloca [5 x ptr]{{.+}}
// CHECK: [[PTRS:%.+]] = alloca [5 x ptr]{{.+}}

// access of pointers inside lambdas
// CHECK: [[BASE_PTR1:%.+]] = getelementptr inbounds nuw [[ANON_T]], ptr %{{.+}}, i{{.+}} 0, i{{.+}} 0
// CHECK: [[PTR1:%.+]] = load ptr, ptr [[BASE_PTR1]]
// CHECK: [[BASE_PTR2:%.+]] = getelementptr inbounds nuw [[ANON_T]], ptr %{{.+}}, i{{.+}} 0, i{{.+}} 1
// CHECK: [[PTR2:%.+]] = load ptr, ptr [[BASE_PTR2]]

// storage of pointers in baseptrs and ptrs arrays
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 2
// CHECK: store ptr %{{.+}}, ptr [[LOC_LAMBDA]]{{.+}}
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i{{.+}} 0, i{{.+}} 2
// CHECK: store ptr %{{.+}}, ptr [[LOC_LAMBDA]]{{.+}}

// CHECK: [[LOC_PTR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 3
// CHECK: store ptr [[BASE_PTR1]], ptr [[LOC_PTR1]]{{.+}}
// CHECK: [[LOC_PTR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i{{.+}} 0, i{{.+}} 3
// CHECK: store ptr [[PTR1]], ptr [[LOC_PTR1]]{{.+}}


// CHECK: [[LOC_PTR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 4
// CHECK: store ptr [[BASE_PTR2]], ptr [[LOC_PTR2]]{{.+}}
// CHECK: [[LOC_PTR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[PTRS]], i{{.+}} 0, i{{.+}} 4
// CHECK: store ptr [[PTR2]], ptr [[LOC_PTR2]]{{.+}}

// actual target invocation
// CHECK: [[BASES_GEP:%.+]] = getelementptr {{.+}} [5 x ptr], ptr [[BASE_PTRS]], {{.+}} 0, {{.+}} 0
// CHECK: [[PTRS_GEP:%.+]] = getelementptr {{.+}} [5 x ptr], ptr [[PTRS]], {{.+}} 0, {{.+}} 0

// CHECK: define internal void @{{.+}}omp_loop_ref{{.+}}(
// CHECK: [[BODY:%body.addr]] = alloca ptr
// CHECK: [[TMP:%tmp]] = alloca ptr
// CHECK: [[BODY_REF:%body_ref]] = alloca ptr
// CHECK: [[REF_TMP:%ref.tmp]] = alloca %class.anon.1
// CHECK: [[TMP8:%tmp.+]] = alloca ptr
// CHECK: [[L0:%.+]] = load ptr, ptr [[BODY]]
// CHECK: store ptr [[L0]], ptr [[TMP]]
// CHECK: [[L5:%.+]] = load ptr, ptr [[TMP]]
// CHECK-NOT [[L6:%.+]] = load ptr, ptr [[TMP]]
// CHECK-NOT [[L7:%.+]] = load ptr, ptr [[TMP]]
// CHECK: store ptr [[REF_TMP]], ptr [[BODY_REF]]
// CHECK:[[L47:%.+]] =  load ptr, ptr [[BODY_REF]]
// CHECK: store ptr [[L47]], ptr [[TMP8]]
// CHECK: [[L48:%.+]] = load ptr, ptr [[TMP8]]
// CHECK-NOT: [[L49:%.+]] = load ptr, ptr [[TMP8]]
// CHECK-NOT: [[L50:%.+]] = load ptr, ptr [[TMP8]]
// CHECK: ret void

// CHECK: define internal void @{{.+}}xoo{{.+}}(
// CHECK: [[FADDR:%f.addr]] = alloca ptr
// CHECK: [[L0:%.+]] = load ptr, ptr [[FADDR]]
// CHECK: store ptr [[L0]], ptr [[TMP:%tmp]]
// CHECK: [[L1:%.+]] = load ptr, ptr [[TMP]]
// CHECK-NOT: %4 = load ptr, ptr [[TMP]]
// CHECK-NOT: %5 = load ptr, ptr [[TMP]]
// CHECK: [[L4:%.+]] = getelementptr inbounds nuw %class.anon.0, ptr [[L1]], i32 0, i32 0
// CHECK: [[L5:%.+]] = load ptr, ptr [[L4]]
// CHECK: ret void

#endif
