// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s --implicit-check-not=@_Z5alignm --implicit-check-not=@_Z4flagb
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -o - %s -DLITERAL | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -ast-dump %s | FileCheck %s --check-prefix=AST

// A constexpr call as a _Constant argument is emitted like the literal.

#ifdef LITERAL
#define C(x) (x)
#define B(x) (x)
#define CE(x) (x)
#else
constexpr unsigned long align(unsigned long a) { return a; }
constexpr bool flag(bool b) { return b; }
consteval unsigned long align_ce(unsigned long a) { return a; }
#define C(x) align(x)
#define B(x) flag(x)
#define CE(x) align_ce(x)
#endif

// CHECK-LABEL: @_Z4callm(
// CHECK: alloca i8, i64 %{{.*}}, align 8
// AST-LABEL: FunctionDecl {{.*}} call 'void *(unsigned long)'
// AST: ConstantExpr {{.*}} 'unsigned long'
// AST-NEXT: value: Int 64
// AST-NEXT: CallExpr {{.*}} 'unsigned long'
void *call(unsigned long n) { return __builtin_alloca_with_align(n, C(64)); }

template <unsigned long A> void *dependent(unsigned long n) {
  return __builtin_alloca_with_align(n, C(A));
}

void *instantiate(unsigned long n) { return dependent<128>(n); }

// The template pattern stays unwrapped; the instantiation stores the value.
// AST-LABEL: FunctionTemplateDecl {{.*}} dependent
// AST: FunctionDecl {{.*}} dependent 'void *(unsigned long)'{{$}}
// AST-NOT: ConstantExpr
// AST: implicit_instantiation
// AST: ConstantExpr {{.*}} 'unsigned long'
// AST-NEXT: value: Int 128
// CHECK-LABEL: @_Z9dependentILm128EEPvm(
// CHECK: alloca i8, i64 %{{.*}}, align 16

// An immediate invocation is not wrapped again.
// CHECK-LABEL: @_Z9immediatem(
// CHECK: alloca i8, i64 %{{.*}}, align 32
void *immediate(unsigned long n) {
  return __builtin_alloca_with_align(n, CE(256));
}

// A _Constant bool is emitted as an i1 constant.
// CHECK-LABEL: @_Z7suspendv(
// CHECK: call i8 @llvm.coro.suspend(token none, i1 true)
// AST-LABEL: FunctionDecl {{.*}} suspend 'char ()'
// AST: ConstantExpr {{.*}} 'bool'
// AST-NEXT: value: Int 1
// AST-NEXT: CallExpr {{.*}} 'bool'
char suspend() { return __builtin_coro_suspend(B(true)); }
