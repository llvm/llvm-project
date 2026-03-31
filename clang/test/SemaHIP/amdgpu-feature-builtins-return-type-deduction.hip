// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx900 -ast-dump -ast-dump-decl-types %s | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1201 -ast-dump -ast-dump-decl-types %s | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -ast-dump -ast-dump-decl-types %s | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -triple x86_64 -aux-triple amdgcn -ast-dump -ast-dump-decl-types %s | FileCheck %s --strict-whitespace
// RUN: %clang_cc1 -triple x86_64 -aux-triple spirv64-amd-amdhsa -ast-dump -ast-dump-decl-types %s | FileCheck %s --strict-whitespace

__attribute__((device)) auto foo() {
  return __builtin_amdgcn_processor_is("gfx900");
}

__attribute__((device)) decltype(auto) bar() {
  return __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep);
}

// CHECK: |-TypedefDecl {{.*}} implicit __amdgpu_feature_predicate_t '__amdgpu_feature_predicate_t'
// CHECK-NEXT: | `-BuiltinType {{.*}} '__amdgpu_feature_predicate_t'
// CHECK-DAG: |-FunctionDecl {{.*}} foo '__amdgpu_feature_predicate_t ()'
// CHECK-NEXT: |-CompoundStmt {{.*}}
// CHECK-NEXT: | `-ReturnStmt {{.*}}
// CHECK-NEXT: |   `-CallExpr {{.*}} '__amdgpu_feature_predicate_t'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.*}} '__amdgpu_feature_predicate_t (*)(const char *) noexcept'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} Function {{.*}} '__builtin_amdgcn_processor_is' '__amdgpu_feature_predicate_t (const char *) noexcept'
// CHECK-NEXT: |     `-StringLiteral {{.*}} "gfx900"
// CHECK-DAG: |-FunctionDecl {{.*}} bar '__amdgpu_feature_predicate_t ()'
// CHECK-NEXT: |-CompoundStmt {{.*}}
// CHECK-NEXT: | `-ReturnStmt {{.*}}
// CHECK-NEXT: |   `-CallExpr {{.*}} '__amdgpu_feature_predicate_t'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.*}} '__amdgpu_feature_predicate_t (*)() noexcept' <BuiltinFnToFnPtr>
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} Function {{.*}} '__builtin_amdgcn_is_invocable' '__amdgpu_feature_predicate_t () noexcept'
