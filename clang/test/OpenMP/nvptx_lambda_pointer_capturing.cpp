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

// CHECK: [[ANON_T:%.+]] = type { i32*, i32* }
// CHECK-DAG: [[SIZES_TEMPLATE:@.+]] = private {{.+}} constant [5 x i[[PTRSZ:32|64]]] [i{{32|64}} 4, i{{32|64}} 4, i{{32|64}} {{8|16}}, i{{32|64}} 0, i{{32|64}} 0]
// CHECK-DAG: [[TYPES_TEMPLATE:@.+]] = private {{.+}} constant [5 x i64] [i64 800, i64 800, i64 673, i64 844424930132752, i64 844424930132752]
// CHECK-DAG: [[SIZES:@.+]] = private {{.+}} constant [3 x i[[PTRSZ:32|64]]] [i{{32|64}} {{8|16}}, i{{32|64}} 0, i{{32|64}} 0]
// CHECK-DAG: [[TYPES:@.+]] = private {{.+}} constant [3 x i64] [i64 673, i64 281474976711440, i64 281474976711440]

template <typename F>
void omp_loop(int start, int end, F body) {
#pragma omp target teams distribute parallel for
  for (int i = start; i < end; ++i) {
    body(i);
  }
}

// CHECK: define {{.*}}[[MAIN:@.+]](
int main()
{
  int* p = new int[100];
  int* q = new int[100];
  auto body = [=](int i){
    p[i] = q[i];
  };

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
// CHECK: {{%.+}} = call{{.+}} @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, {{.+}}, {{.+}}, {{.+}}, i8** [[BASES_GEP]], i8** [[PTRS_GEP]], i[[PTRSZ]]* getelementptr inbounds ([3 x i{{.+}}], [3 x i{{.+}}]* [[SIZES]], i{{.+}} 0, i{{.+}} 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[TYPES]], i{{.+}} 0, i{{.+}} 0), i8** null, i8** null, {{.+}}, {{.+}})


  omp_loop(0,100,body);
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
// CHECK: {{%.+}} = call{{.+}} @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, {{.+}}, {{.+}}, {{.+}}, i8** [[BASES_GEP]], i8** [[PTRS_GEP]], i[[PTRSZ]]* getelementptr inbounds ([5 x i{{.+}}], [5 x i{{.+}}]* [[SIZES_TEMPLATE]], i{{.+}} 0, i{{.+}} 0), i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[TYPES_TEMPLATE]], i{{.+}} 0, i{{.+}} 0), i8** null, i8** null, {{.+}}, {{.+}})

#endif
