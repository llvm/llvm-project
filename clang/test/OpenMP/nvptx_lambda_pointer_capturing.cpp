// Test host codegen only.
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[ANON_T:%.+]] = type { i32*, i32* }
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

// CHECK: [[BASE_PTRS:%.+]] = alloca [3 x i8*]{{.+}}
// CHECK: [[PTRS:%.+]] = alloca [3 x i8*]{{.+}}

// First gep of pointers inside lambdas to store the values across function call need to be ignored
// CHECK: {{%.+}} = getelementptr inbounds [[ANON_T]], [[ANON_T]]* %{{.+}}, i{{.+}} 0, i{{.+}} 0
// CHECK: {{%.+}} = getelementptr inbounds [[ANON_T]], [[ANON_T]]* %{{.+}}, i{{.+}} 0, i{{.+}} 1

// access of pointers inside lambdas
// CHECK: [[BASE_PTR1:%.+]] = getelementptr inbounds [[ANON_T]], [[ANON_T]]* %{{.+}}, i{{.+}} 0, i{{.+}} 0
// CHECK: [[PTR1:%.+]] = load i32*, i32** [[BASE_PTR1]]
// CHECK: [[BASE_PTR2:%.+]] = getelementptr inbounds [[ANON_T]], [[ANON_T]]* %{{.+}}, i{{.+}} 0, i{{.+}} 1
// CHECK: [[PTR2:%.+]] = load i32*, i32** [[BASE_PTR2]]

// storage of pointers in baseptrs and ptrs arrays
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[CAST_LAMBDA:%.+]] = bitcast i8** [[LOC_LAMBDA]] to [[ANON_T]]**
// CHECK: store [[ANON_T]]* %{{.+}}, [[ANON_T]]** [[CAST_LAMBDA]]{{.+}}
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[CAST_LAMBDA:%.+]] = bitcast i8** [[LOC_LAMBDA]] to [[ANON_T]]**
// CHECK: store [[ANON_T]]* %{{.+}}, [[ANON_T]]** [[CAST_LAMBDA]]{{.+}}

// CHECK: [[LOC_PTR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[CAST_PTR1:%.+]] = bitcast i8** [[LOC_PTR1]] to i32***
// CHECK: store i32** [[BASE_PTR1]], i32*** [[CAST_PTR1]]{{.+}}
// CHECK: [[LOC_PTR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[CAST_PTR1:%.+]] = bitcast i8** [[LOC_PTR1]] to i32**
// CHECK: store i32* [[PTR1]], i32** [[CAST_PTR1]]{{.+}}


// CHECK: [[LOC_PTR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[CAST_PTR2:%.+]] = bitcast i8** [[LOC_PTR2]] to i32***
// CHECK: store i32** [[BASE_PTR2]], i32*** [[CAST_PTR2]]{{.+}}
// CHECK: [[LOC_PTR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[CAST_PTR2:%.+]] = bitcast i8** [[LOC_PTR2]] to i32**
// CHECK: store i32* [[PTR2]], i32** [[CAST_PTR2]]{{.+}}

  // actual target invocation
  // CHECK: [[BASES_GEP:%.+]] = getelementptr {{.+}} [3 x {{.+}}*], [3 x {{.+}}*]* [[BASE_PTRS]], {{.+}} 0, {{.+}} 0
  // CHECK: [[PTRS_GEP:%.+]] = getelementptr {{.+}} [3 x {{.+}}*], [3 x {{.+}}*]* [[PTRS]], {{.+}} 0, {{.+}} 0
  // CHECK: {{%.+}} = call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 0, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* %{{.+}})

  omp_loop(0,100,body);
  omp_loop_ref(0,100,body);
}

// CHECK: [[BASE_PTRS:%.+]] = alloca [5 x i8*]{{.+}}
// CHECK: [[PTRS:%.+]] = alloca [5 x i8*]{{.+}}

// access of pointers inside lambdas
// CHECK: [[BASE_PTR1:%.+]] = getelementptr inbounds [[ANON_T]], [[ANON_T]]* %{{.+}}, i{{.+}} 0, i{{.+}} 0
// CHECK: [[PTR1:%.+]] = load i32*, i32** [[BASE_PTR1]]
// CHECK: [[BASE_PTR2:%.+]] = getelementptr inbounds [[ANON_T]], [[ANON_T]]* %{{.+}}, i{{.+}} 0, i{{.+}} 1
// CHECK: [[PTR2:%.+]] = load i32*, i32** [[BASE_PTR2]]

// storage of pointers in baseptrs and ptrs arrays
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[CAST_LAMBDA:%.+]] = bitcast i8** [[LOC_LAMBDA]] to [[ANON_T]]**
// CHECK: store [[ANON_T]]* %{{.+}}, [[ANON_T]]** [[CAST_LAMBDA]]{{.+}}
// CHECK: [[LOC_LAMBDA:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[CAST_LAMBDA:%.+]] = bitcast i8** [[LOC_LAMBDA]] to [[ANON_T]]**
// CHECK: store [[ANON_T]]* %{{.+}}, [[ANON_T]]** [[CAST_LAMBDA]]{{.+}}

// CHECK: [[LOC_PTR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 3
// CHECK: [[CAST_PTR1:%.+]] = bitcast i8** [[LOC_PTR1]] to i32***
// CHECK: store i32** [[BASE_PTR1]], i32*** [[CAST_PTR1]]{{.+}}
// CHECK: [[LOC_PTR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i{{.+}} 0, i{{.+}} 3
// CHECK: [[CAST_PTR1:%.+]] = bitcast i8** [[LOC_PTR1]] to i32**
// CHECK: store i32* [[PTR1]], i32** [[CAST_PTR1]]{{.+}}


// CHECK: [[LOC_PTR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BASE_PTRS]], i{{.+}} 0, i{{.+}} 4
// CHECK: [[CAST_PTR2:%.+]] = bitcast i8** [[LOC_PTR2]] to i32***
// CHECK: store i32** [[BASE_PTR2]], i32*** [[CAST_PTR2]]{{.+}}
// CHECK: [[LOC_PTR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS]], i{{.+}} 0, i{{.+}} 4
// CHECK: [[CAST_PTR2:%.+]] = bitcast i8** [[LOC_PTR2]] to i32**
// CHECK: store i32* [[PTR2]], i32** [[CAST_PTR2]]{{.+}}

// actual target invocation
// CHECK: [[BASES_GEP:%.+]] = getelementptr {{.+}} [5 x {{.+}}*], [5 x {{.+}}*]* [[BASE_PTRS]], {{.+}} 0, {{.+}} 0
// CHECK: [[PTRS_GEP:%.+]] = getelementptr {{.+}} [5 x {{.+}}*], [5 x {{.+}}*]* [[PTRS]], {{.+}} 0, {{.+}} 0

// CHECK: define internal void @{{.+}}omp_loop_ref{{.+}}(
// CHECK: [[BODY:%body.addr]] = alloca %class.anon*
// CHECK: [[TMP:%tmp]] = alloca %class.anon*
// CHECK: [[BODY_REF:%body_ref]] = alloca %class.anon.1*
// CHECK: [[REF_TMP:%ref.tmp]] = alloca %class.anon.1
// CHECK: [[TMP8:%tmp.+]] = alloca %class.anon.1*
// CHECK: [[L0:%.+]] = load %class.anon*, %class.anon** [[BODY]]
// CHECK: store %class.anon* [[L0]], %class.anon** [[TMP]]
// CHECK: [[L5:%.+]] = load %class.anon*, %class.anon** [[TMP]]
// CHECK-NOT [[L6:%.+]] = load %class.anon*, %class.anon** [[TMP]]
// CHECK-NOT [[L7:%.+]] = load %class.anon*, %class.anon** [[TMP]]
// CHECK: store %class.anon.1* [[REF_TMP]], %class.anon.1** [[BODY_REF]]
// CHECK:[[L47:%.+]] =  load %class.anon.1*, %class.anon.1** [[BODY_REF]]
// CHECK: store %class.anon.1* [[L47]], %class.anon.1** [[TMP8]]
// CHECK: [[L48:%.+]] = load %class.anon.1*, %class.anon.1** [[TMP8]]
// CHECK-NOT: [[L49:%.+]] = load %class.anon.1*, %class.anon.1** [[TMP8]]
// CHECK-NOT: [[L50:%.+]] = load %class.anon.1*, %class.anon.1** [[TMP8]]
// CHECK: ret void

// CHECK: define internal void @{{.+}}xoo{{.+}}(
// CHECK: [[FADDR:%f.addr]] = alloca %class.anon.0*
// CHECK: [[L0:%.+]] = load %class.anon.0*, %class.anon.0** [[FADDR]]
// CHECK: store %class.anon.0* [[L0]], %class.anon.0** [[TMP:%tmp]]
// CHECK: [[L1:%.+]] = load %class.anon.0*, %class.anon.0** [[TMP]]
// CHECK-NOT: %4 = load %class.anon.0*, %class.anon.0** [[TMP]]
// CHECK-NOT: %5 = load %class.anon.0*, %class.anon.0** [[TMP]]
// CHECK: [[L4:%.+]] = getelementptr inbounds %class.anon.0, %class.anon.0* [[L1]], i32 0, i32 0
// CHECK: [[L5:%.+]] = load i{{.*}}*, i{{.*}}** [[L4]]
// CHECK: ret void

#endif
