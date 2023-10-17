// RUN: %clang_cc1 -verify -fopenmp -x c++ -emit-llvm %s -triple x86_64-linux -fexceptions -fcxx-exceptions -o - -femit-all-decls -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -femit-all-decls -disable-llvm-passes
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls -disable-llvm-passes | FileCheck --check-prefixes=CHECK-LOAD,OMP50-LOAD %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -emit-llvm %s -triple x86_64-linux -fexceptions -fcxx-exceptions -o - -femit-all-decls -disable-llvm-passes | FileCheck %s --check-prefixes=CHECK,OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple x86_64-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -femit-all-decls -disable-llvm-passes
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls -disable-llvm-passes | FileCheck --check-prefixes=CHECK-LOAD,OMP45-LOAD %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -emit-llvm %s -triple x86_64-linux -fexceptions -fcxx-exceptions -o - -femit-all-decls -disable-llvm-passes | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s -femit-all-decls -disable-llvm-passes
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls -disable-llvm-passes | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK: [[SSS_INT:.+]] = type { i32 }
// CHECK-LOAD: [[SSS_INT:.+]] = type { i32 }

// OMP45: add
void add(short &out, short &in) {}

#pragma omp declare reduction(my_add : short : add(omp_out, omp_in))

// OMP45: define internal void @.
// OMP45: call void @{{.+}}add{{.+}}(
// OMP45: ret void

// OMP45: foo_reduction_array
void foo_reduction_array() {
  short y[1];
  // OMP45: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(
#pragma omp parallel for reduction(my_add : y)
  for (int i = 0; i < 1; i++) {
  }
}

// OMP45: define internal void @

#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)
// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: [[MUL:%.+]] = mul nsw i32 %{{.+}}, %{{.+}}
// CHECK-NEXT: store i32 [[MUL]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK-LOAD: [[MUL:%.+]] = mul nsw i32 %{{.+}}, %{{.+}}
// CHECK-LOAD-NEXT: store i32 [[MUL]], ptr
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: sext i8
// CHECK: sext i8
// CHECK: [[MUL:%.+]] = mul nsw i32 %{{.+}}, %{{.+}}
// CHECK-NEXT: [[TRUNC:%.+]] = trunc i32 [[MUL]] to i8
// CHECK-NEXT: store i8 [[TRUNC]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK-LOAD: sext i8
// CHECK-LOAD: sext i8
// CHECK-LOAD: [[MUL:%.+]] = mul nsw i32 %{{.+}}, %{{.+}}
// CHECK-LOAD-NEXT: [[TRUNC:%.+]] = trunc i32 [[MUL]] to i8
// CHECK-LOAD-NEXT: store i8 [[TRUNC]], ptr
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

template <class T>
struct SSS {
  T a;
  SSS() : a() {}
#pragma omp declare reduction(fun : T : omp_out ^= omp_in) initializer(omp_priv = 24 + omp_orig)
#pragma omp declare reduction(sssss : T : ssssss(omp_in)) initializer(omp_priv = 18 + omp_orig)
  static void ssssss(T &x);
};

SSS<int> d;

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: [[XOR:%.+]] = xor i32 %{{.+}}, %{{.+}}
// CHECK-NEXT: store i32 [[XOR]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: [[ADD:%.+]] = add nsw i32 24, %{{.+}}
// CHECK-NEXT: store i32 [[ADD]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: call void @_ZN3SSSIiE6ssssssERi(ptr noundef nonnull align {{[0-9]+}} dereferenceable{{.*}})
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: [[ADD:%.+]] = add nsw i32 18, %{{.+}}
// CHECK-NEXT: store i32 [[ADD]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }

template <typename T>
void init(T &lhs, T &rhs) {}

#pragma omp declare reduction(fun : SSS < int > : omp_out = omp_in) initializer(init(omp_priv, omp_orig))
// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: call void @llvm.memcpy
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: call {{.*}}void @_Z4initI3SSSIiEEvRT_S3_(
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK-LOAD: call void @llvm.memcpy
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK-LOAD: call {{.*}}void @_Z4initI3SSSIiEEvRT_S3_(
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define {{.*}}void @_Z4initI3SSSIiEEvRT_S3_(ptr {{.+}}, ptr {{.+}})
// CHECK-LOAD: define {{.*}}void @_Z4initI3SSSIiEEvRT_S3_(ptr {{.+}}, ptr {{.+}})

template <typename T>
T foo(T a) {
#pragma omp declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = 15 * omp_orig)
  {
#pragma omp declare reduction(fun : T : omp_out /= omp_in) initializer(omp_priv = 11 - omp_orig)
  }
  return a;
}

struct Summary {
  void merge(const Summary& other) {}
};

template <typename K>
void work() {
  Summary global_summary;
#pragma omp declare reduction(+ : Summary : omp_out.merge(omp_in))
#pragma omp parallel for reduction(+ : global_summary)
  for (int k = 1; k <= 100; ++k) {
  }
}

struct A {};


// CHECK-LABEL: @main
int main() {
  int i = 0;
  SSS<int> sss;
#pragma omp parallel reduction(SSS < int > ::fun : i)
  {
    i += 1;
  }
#pragma omp parallel reduction(::fun : sss)
  {
  }
#pragma omp declare reduction(fun : SSS < int > : init(omp_out, omp_in))
#pragma omp parallel reduction(fun : sss)
  {
  }
  // CHECK: call {{.*}}void (ptr, i32, ptr, ...) @__kmpc_fork_call(
  // CHECK: call {{.*}}void (ptr, i32, ptr, ...) @__kmpc_fork_call(
  // CHECK: call {{.*}}void (ptr, i32, ptr, ...) @__kmpc_fork_call({{[^@]*}} @{{[^@]*}}[[REGION:@[^,]+]]
  // CHECK-LABEL: work
  work<A>();
  // CHECK-LABEL: foo
  return foo(15);
}

// CHECK: define internal {{.*}}void [[REGION]](
// CHECK: [[SSS_PRIV:%.+]] = alloca %struct.SSS,
// CHECK: invoke {{.*}} @_ZN3SSSIiEC1Ev(ptr {{[^,]*}} [[SSS_PRIV]])
// CHECK-NOT: {{call |invoke }}
// CHECK: call {{.*}}i32 @__kmpc_reduce_nowait(

// CHECK-LABEL: i32 @{{.+}}foo{{[^(].+}}(i32
// CHECK-LOAD-LABEL: i32 @{{.+}}foo{{[^(].+}}(i32

// OMP45-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// OMP45-LOAD: [[XOR:%.+]] = xor i32 %{{.+}}, %{{.+}}
// OMP45-LOAD-NEXT: store i32 [[XOR]], ptr
// OMP45-LOAD-NEXT: ret void
// OMP45-LOAD-NEXT: }

// OMP45-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// OMP45-LOAD: [[ADD:%.+]] = add nsw i32 24, %{{.+}}
// OMP45-LOAD-NEXT: store i32 [[ADD]], ptr
// OMP45-LOAD-NEXT: ret void
// OMP45-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: [[ADD:%.+]] = add nsw i32 %{{.+}}, %{{.+}}
// CHECK-NEXT: store i32 [[ADD]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK-LOAD: [[ADD:%.+]] = add nsw i32 %{{.+}}, %{{.+}}
// CHECK-LOAD-NEXT: store i32 [[ADD]], ptr
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: [[MUL:%.+]] = mul nsw i32 15, %{{.+}}
// CHECK-NEXT: store i32 [[MUL]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK-LOAD: [[MUL:%.+]] = mul nsw i32 15, %{{.+}}
// CHECK-LOAD-NEXT: store i32 [[MUL]], ptr
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: [[DIV:%.+]] = sdiv i32 %{{.+}}, %{{.+}}
// CHECK-NEXT: store i32 [[DIV]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK-LOAD: [[DIV:%.+]] = sdiv i32 %{{.+}}, %{{.+}}
// CHECK-LOAD-NEXT: store i32 [[DIV]], ptr
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: [[SUB:%.+]] = sub nsw i32 11, %{{.+}}
// CHECK-NEXT: store i32 [[SUB]], ptr
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK-LOAD: [[SUB:%.+]] = sub nsw i32 11, %{{.+}}
// CHECK-LOAD-NEXT: store i32 [[SUB]], ptr
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

#endif
