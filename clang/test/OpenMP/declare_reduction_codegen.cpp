// RUN: %clang_cc1 -verify -fopenmp -x c++ -emit-llvm %s -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -o - -femit-all-decls -disable-llvm-optzns | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -emit-pch -o %t %s -femit-all-decls -disable-llvm-optzns
// RUN: %clang_cc1 -fopenmp -x c++ -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls -disable-llvm-optzns | FileCheck --check-prefix=CHECK-LOAD %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK: [[SSS_INT:.+]] = type { i32 }
// CHECK-LOAD: [[SSS_INT:.+]] = type { i32 }

#pragma omp declare reduction(+ : int, char : omp_out *= omp_in)
// CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK: [[MUL:%.+]] = mul nsw i32
// CHECK-NEXT: store i32 [[MUL]], i32*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK-LOAD: [[MUL:%.+]] = mul nsw i32
// CHECK-LOAD-NEXT: store i32 [[MUL]], i32*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(i8* noalias, i8* noalias)
// CHECK: sext i8
// CHECK: sext i8
// CHECK: [[MUL:%.+]] = mul nsw i32
// CHECK-NEXT: [[TRUNC:%.+]] = trunc i32 [[MUL]] to i8
// CHECK-NEXT: store i8 [[TRUNC]], i8*
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i8* noalias, i8* noalias)
// CHECK-LOAD: sext i8
// CHECK-LOAD: sext i8
// CHECK-LOAD: [[MUL:%.+]] = mul nsw i32
// CHECK-LOAD-NEXT: [[TRUNC:%.+]] = trunc i32 [[MUL]] to i8
// CHECK-LOAD-NEXT: store i8 [[TRUNC]], i8*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

template <class T>
struct SSS {
  T a;
  SSS() : a() {}
#pragma omp declare reduction(fun : T : omp_out ^= omp_in) initializer(omp_priv = 24 + omp_orig)
};

SSS<int> d;

// CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK: [[XOR:%.+]] = xor i32
// CHECK-NEXT: store i32 [[XOR]], i32*
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK: [[ADD:%.+]] = add nsw i32 24,
// CHECK-NEXT: store i32 [[ADD]], i32*
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK: define {{.*}}void [[INIT:@[^(]+]]([[SSS_INT]]*
// CHECK-LOAD: define {{.*}}void [[INIT:@[^(]+]]([[SSS_INT]]*
void init(SSS<int> &lhs, SSS<int> &rhs) {}

#pragma omp declare reduction(fun : SSS < int > : omp_out = omp_in) initializer(init(omp_priv, omp_orig))
// CHECK: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias, [[SSS_INT]]* noalias)
// CHECK: call void @llvm.memcpy
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias, [[SSS_INT]]* noalias)
// CHECK: call {{.*}}void [[INIT]](
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias, [[SSS_INT]]* noalias)
// CHECK-LOAD: call void @llvm.memcpy
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}([[SSS_INT]]* noalias, [[SSS_INT]]* noalias)
// CHECK-LOAD: call {{.*}}void [[INIT]](
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

template <typename T>
T foo(T a) {
#pragma omp declare reduction(fun : T : omp_out += omp_in) initializer(omp_priv = 15 * omp_orig)
  {
#pragma omp declare reduction(fun : T : omp_out /= omp_in) initializer(omp_priv = 11 - omp_orig)
  }
  return a;
}

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
  // CHECK: call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
  // CHECK: call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
  // CHECK: call {{.*}}void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call({{[^@]*}} @{{[^@]*}}[[REGION:@[^ ]+]]
  // CHECK-LABEL: foo
  return foo(15);
}

// CHECK: define internal {{.*}}void [[REGION]](
// CHECK: [[SSS_PRIV:%.+]] = alloca %struct.SSS,
// CHECK: invoke {{.*}} @_ZN3SSSIiEC1Ev(%struct.SSS* [[SSS_PRIV]])
// CHECK-NOT: {{call |invoke }}
// CHECK: call {{.*}}i32 @__kmpc_reduce_nowait(

// CHECK-LABEL: i32 @{{.+}}foo{{[^(].+}}(i32
// CHECK-LOAD-LABEL: i32 @{{.+}}foo{{[^(].+}}(i32

// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK-LOAD: [[XOR:%.+]] = xor i32
// CHECK-LOAD-NEXT: store i32 [[XOR]], i32*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK-LOAD: [[ADD:%.+]] = add nsw i32 24,
// CHECK-LOAD-NEXT: store i32 [[ADD]], i32*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK: [[ADD:%.+]] = add nsw i32
// CHECK-NEXT: store i32 [[ADD]], i32*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK-LOAD: [[ADD:%.+]] = add nsw i32
// CHECK-LOAD-NEXT: store i32 [[ADD]], i32*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK: [[MUL:%.+]] = mul nsw i32 15,
// CHECK-NEXT: store i32 [[MUL]], i32*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK-LOAD: [[MUL:%.+]] = mul nsw i32 15,
// CHECK-LOAD-NEXT: store i32 [[MUL]], i32*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK: [[DIV:%.+]] = sdiv i32
// CHECK-NEXT: store i32 [[DIV]], i32*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK-LOAD: [[DIV:%.+]] = sdiv i32
// CHECK-LOAD-NEXT: store i32 [[DIV]], i32*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

// CHECK: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK: [[SUB:%.+]] = sub nsw i32 11,
// CHECK-NEXT: store i32 [[SUB]], i32*
// CHECK-NEXT: ret void
// CHECK-NEXT: }
// CHECK-LOAD: define internal {{.*}}void @{{[^(]+}}(i32* noalias, i32* noalias)
// CHECK-LOAD: [[SUB:%.+]] = sub nsw i32 11,
// CHECK-LOAD-NEXT: store i32 [[SUB]], i32*
// CHECK-LOAD-NEXT: ret void
// CHECK-LOAD-NEXT: }

#endif
