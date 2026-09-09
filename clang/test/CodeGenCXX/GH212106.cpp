// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -ffp-exception-behavior=strict -emit-llvm -o - %s | FileCheck %s

// A file-scope compound literal is a constant-initialized global even when its
// address is taken from a non-constant context.

struct RR { int r; };
struct Z { int x; const RR* y; int z; };
inline int f() { return 0; }
Z z2 = { 10, (const RR[1]){__builtin_constant_p(z2.x)}, z2.y->r+f() };

// CHECK-DAG: @z2 = {{.*}}global %struct.Z zeroinitializer
// CHECK-DAG: [[Z2CL:@.compoundliteral(\.[0-9]+)?]] = internal constant [1 x %struct.RR] zeroinitializer

namespace reduced {
struct Z { const int* y; int z; };
int f();
Z z2 = { (int[1]){__builtin_constant_p(z2.z)}, f() };
}

// CHECK-DAG: @_ZN7reduced2z2E = {{.*}}global %"struct.reduced::Z" zeroinitializer
// CHECK-DAG: @.compoundliteral{{(\.[0-9]+)?}} = internal global [1 x i32] zeroinitializer

struct F { int a; const float *fp; };
int g();
F fl = { g(), (float[1]){0.1} };

// CHECK-DAG: @fl = {{.*}}global %struct.F zeroinitializer
// CHECK-DAG: @.compoundliteral{{(\.[0-9]+)?}} = internal global [1 x float] [float 1.000000e-01]

const RR *p = (const RR[1]){__builtin_constant_p(1)};

// CHECK-DAG: @.compoundliteral{{(\.[0-9]+)?}} = internal constant [1 x %struct.RR] [%struct.RR { i32 1 }]
// CHECK-DAG: @p = {{.*}}global ptr @.compoundliteral{{(\.[0-9]+)?}}

// A default member initializer at namespace scope is also a file-scope compound
// literal, reached here from a constructor emitted for a local variable.
extern int n;
struct Q { const int *m = (const int[1]){__builtin_constant_p(n)}; };
void h() { Q q; }

// CHECK-DAG: [[QCL:@.compoundliteral(\.[0-9]+)?]] = internal constant [1 x i32] zeroinitializer

struct SP { int a; const char *const *s; };
SP sp = { g(), (const char *const[1]){__builtin_constant_p(n) ? "a" : "b"} };

// CHECK-DAG: @.compoundliteral{{(\.[0-9]+)?}} = internal constant [1 x ptr] [ptr @.str{{(\.[0-9]+)?}}]

// A reference member binds its element as an lvalue.
int gv;
struct RS { const int &r; int v; };
RS rs = (RS){gv, __builtin_constant_p(n)};

// CHECK-DAG: @rs = {{.*}}global { ptr, i32 } { ptr @gv, i32 0 }

// An immediate invocation is already a ConstantExpr.
consteval int cf() { return 3; }
const int *cp = (const int[1]){cf()};

// CHECK-DAG: @.compoundliteral{{(\.[0-9]+)?}} = internal constant [1 x i32] [i32 3]

// A pointer cast to an integer is stored as an lvalue.
struct LI { int a; const long *l; };
LI li = { g(), (const long[1]){(long)"x"} };

// CHECK-DAG: @.compoundliteral{{(\.[0-9]+)?}} = internal constant [1 x i64] [i64 ptrtoint (ptr @.str{{(\.[0-9]+)?}} to i64)]

namespace std { class type_info; }
const std::type_info *const *tp = (const std::type_info *const[1]){&typeid(int)};

// CHECK-DAG: @.compoundliteral{{(\.[0-9]+)?}} = internal constant [1 x ptr] [ptr @_ZTIi]

// CHECK-LABEL: define internal void @__cxx_global_var_init()
// CHECK: store ptr [[Z2CL]], ptr getelementptr inbounds{{.*}}(i8, ptr @z2, i64 8)

// CHECK-LABEL: define {{.*}}void @_ZN1QC2Ev(
// CHECK: store ptr [[QCL]], ptr
