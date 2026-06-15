// REQUIRES: amdgpu-registered-target

// Regression test for issue #199563: -ast-print on
// __builtin_amdgcn_is_invocable / __builtin_amdgcn_processor_is used to
// crash because the expansion produced an IntegerLiteral typed _Bool/bool.

// C (C11 and C23): IntegerLiteral 'int' implicit-cast to _Bool.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -std=c11 \
// RUN:   -ast-print %s | FileCheck --check-prefix=INT-TRUE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -std=c11 \
// RUN:   -ast-print %s | FileCheck --check-prefix=INT-FALSE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -std=c11 \
// RUN:   -ast-dump %s | FileCheck --check-prefix=INT-DUMP %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -std=c23 \
// RUN:   -ast-print %s | FileCheck --check-prefix=INT-TRUE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -std=c23 \
// RUN:   -ast-print %s | FileCheck --check-prefix=INT-FALSE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -std=c23 \
// RUN:   -ast-dump %s | FileCheck --check-prefix=INT-DUMP %s

// C++ / HIP device: CXXBoolLiteralExpr 'bool'.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x c++ \
// RUN:   -std=c++17 -ast-print %s | FileCheck --check-prefix=BOOL-TRUE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -x c++ \
// RUN:   -std=c++17 -ast-print %s | FileCheck --check-prefix=BOOL-FALSE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x c++ \
// RUN:   -std=c++17 -ast-dump %s | FileCheck --check-prefix=BOOL-DUMP %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -ast-print %s \
// RUN:   | FileCheck --check-prefix=BOOL-TRUE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -x hip \
// RUN:   -fcuda-is-device -ast-print %s \
// RUN:   | FileCheck --check-prefix=BOOL-FALSE %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -ast-dump %s \
// RUN:   | FileCheck --check-prefix=BOOL-DUMP %s

#if defined(__HIP__) || defined(__CUDA__)
#define __device__ __attribute__((device))
#else
#define __device__
#endif

__device__ void use_is_invocable(void) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fadd_f32))
    (void)0;
}

__device__ void use_processor_is(void) {
  if (__builtin_amdgcn_processor_is("gfx942"))
    (void)0;
}

// INT-TRUE-LABEL: use_is_invocable
// INT-TRUE:       if (1)
// INT-TRUE-LABEL: use_processor_is
// INT-TRUE:       if (1)

// INT-FALSE-LABEL: use_is_invocable
// INT-FALSE:       if (0)
// INT-FALSE-LABEL: use_processor_is
// INT-FALSE:       if (0)

// BOOL-TRUE-LABEL: use_is_invocable
// BOOL-TRUE:       if (true)
// BOOL-TRUE-LABEL: use_processor_is
// BOOL-TRUE:       if (true)

// BOOL-FALSE-LABEL: use_is_invocable
// BOOL-FALSE:       if (false)
// BOOL-FALSE-LABEL: use_processor_is
// BOOL-FALSE:       if (false)

// INT-DUMP-LABEL: FunctionDecl {{.*}} use_is_invocable
// INT-DUMP:       IfStmt
// INT-DUMP-NEXT:  ImplicitCastExpr {{.*}} {{'_Bool'|'bool'}} <IntegralToBoolean>
// INT-DUMP-NEXT:  IntegerLiteral {{.*}} 'int' 1
// INT-DUMP-LABEL: FunctionDecl {{.*}} use_processor_is
// INT-DUMP:       IfStmt
// INT-DUMP-NEXT:  ImplicitCastExpr {{.*}} {{'_Bool'|'bool'}} <IntegralToBoolean>
// INT-DUMP-NEXT:  IntegerLiteral {{.*}} 'int' 1

// BOOL-DUMP-LABEL: FunctionDecl {{.*}} use_is_invocable
// BOOL-DUMP:       IfStmt
// BOOL-DUMP-NEXT:  CXXBoolLiteralExpr {{.*}} 'bool' true
// BOOL-DUMP-LABEL: FunctionDecl {{.*}} use_processor_is
// BOOL-DUMP:       IfStmt
// BOOL-DUMP-NEXT:  CXXBoolLiteralExpr {{.*}} 'bool' true
