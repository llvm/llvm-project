// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ \
// RUN:  -triple powerpc64le-unknown-unknown -DCUDA \
// RUN:  -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o \
// RUN:  %t-ppc-host.bc

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ \
// RUN:  -triple nvptx64-unknown-unknown -DCUA \
// RUN:  -fopenmp-targets=nvptx64-nvidia-cuda -DCUDA -emit-llvm %s \
// RUN:  -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc \
// RUN:  -o - | FileCheck %s --check-prefix CHECK

// RUN: %clang_cc1 -verify -fopenmp -x c++ \
// RUN:   -triple powerpc64le-unknown-unknown -DDIAG\
// RUN:   -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm \
// RUN:   %s -o - | FileCheck  %s \
// RUN:   --check-prefix=CHECK1

// RUN: %clang_cc1 -verify -fopenmp -x c++ \
// RUN:   -triple i386-unknown-unknown \
// RUN:   -fopenmp-targets=i386-pc-linux-gnu -emit-llvm \
// RUN:   %s -o - | FileCheck  %s \
// RUN:   --check-prefix=CHECK2


#if defined(CUDA)
// expected-no-diagnostics

int foo(int n) {
  double *e;
  //no error and no implicit map generated for e[:1]
  #pragma omp target parallel reduction(+: e[:1])
    *e=10;
  ;
  return 0;
}
// CHECK-NOT @.offload_maptypes
// CHECK: call void @__kmpc_nvptx_end_reduce_nowait(
#elif defined(DIAG)
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  S2(S2 &s2):a(s2.a) { }
  S2 &operator +(S2 &s);
};
int bar() {
 S2 o[5];
  //warnig "copyable and not guaranteed to be mapped correctly" and
  //implicit map generated.
#pragma omp target parallel reduction(+:o[0]) //expected-warning {{Type 'S2' is not trivially copyable and not guaranteed to be mapped correctly}}
  for (int i = 0; i < 10; i++);
  double b[10][10][10];
  //no error no implicit map generated, the map for b is generated but not
  //for b[0:2][2:4][1].
#pragma omp target parallel for reduction(task, +: b[0:2][2:4][1])
  for (long long i = 0; i < 10; ++i);
  return 0;
}
// map for variable o
// CHECK1: offload_sizes = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK1: offload_maptypes = private unnamed_addr constant [1 x i64] [i64 547]
// map for b:
// CHECK1: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 8000]
// CHECK1: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 547]
#else
// expected-no-diagnostics

// generate implicit map for array elements or array sections in reduction
// clause. In following case: the implicit map is generate for output[0]
// with map size 4 and output[:3] with map size 12.
void sum(int* input, int size, int* output)
{
#pragma omp target teams distribute parallel for reduction(+: output[0]) \
                                                 map(to: input [0:size])
  for (int i = 0; i < size; i++)
    output[0] += input[i];
#pragma omp target teams distribute parallel for reduction(+: output[:3])  \
                                                 map(to: input [0:size])
  for (int i = 0; i < size; i++)
    output[0] += input[i];
  int a[10];
#pragma omp target parallel reduction(+: a[:2])
  for (int i = 0; i < size; i++)
    ;
#pragma omp target parallel reduction(+: a[3])
  for (int i = 0; i < size; i++)
    ;
}
//CHECK2: @.offload_sizes = private unnamed_addr constant [2 x i64] [i64 4, i64 8]
//CHECK2: @.offload_maptypes.10 = private unnamed_addr constant [2 x i64] [i64 800, i64 547]
//CHECK2: @.offload_sizes.13 = private unnamed_addr constant [2 x i64] [i64 4, i64 4]
//CHECK2: @.offload_maptypes.14 = private unnamed_addr constant [2 x i64] [i64 800, i64 547]
//CHECK2: define dso_local void @_Z3sumPiiS_
//CHECK2-NEXT: entry
//CHECK2-NEXT: [[INP:%.*]] = alloca i32*
//CHECK2-NEXT: [[SIZE:%.*]] = alloca i32
//CHECK2-NEXT: [[OUTP:%.*]] = alloca i32*
//CHECK2:      [[OFFSIZE:%.*]] = alloca [3 x i64]
//CHECK2:      [[OFFSIZE10:%.*]] = alloca [3 x i64]
//CHECK2:      [[T15:%.*]] = getelementptr inbounds [3 x i64], [3 x i64]* [[OFFSIZE]], i32 0, i32 0
//CHECK2-NEXT: store i64 4, i64* [[T15]]
//CHECK2:      [[T21:%.*]] = getelementptr inbounds [3 x i64], [3 x i64]* [[OFFSIZE]], i32 0, i32 1
//CHECK2-NEXT: store i64 4, i64* [[T21]]
//CHECK2:     [[T53:%.*]] = getelementptr inbounds [3 x i64], [3 x i64]* [[OFFSIZE10]], i32 0, i32 0
//CHECK2-NEXT: store i64 4, i64* [[T53]]
//CHECK2:     [[T59:%.*]] = getelementptr inbounds [3 x i64], [3 x i64]* [[OFFSIZE10]], i32 0, i32 1
//CHECK2-NEXT: store i64 12, i64* [[T59]]
#endif
int main()
{
#if defined(CUDA)
  int a = foo(10);
#elif defined(DIAG)
  int a = bar();
#else
  const int size = 100;
  int *array = new int[size];
  int result = 0;
  sum(array, size, &result);
#endif
  return 0;
}
