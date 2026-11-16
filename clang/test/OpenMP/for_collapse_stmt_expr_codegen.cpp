// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fexperimental-new-constant-interpreter -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -verify -fopenmp-simd -fexperimental-new-constant-interpreter -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// expected-no-diagnostics

// CHECK-LABEL: define {{.*}}void @_Z23collapse_stmt_expr_initv(
// CHECK:         %a = alloca float,
// CHECK-NOT:     %a{{[0-9]+}} = alloca
// CHECK:         store float 0.000000e+00, ptr %a,
// CHECK-NOT:     store float 0.000000e+00, ptr %a,
// CHECK:         ret void
void collapse_stmt_expr_init() {
#pragma omp for collapse(2)
  for (int i = ({float a = 0;a; }); i < 10; i++)
    for (int j = i; j < 10 + i; j++)
    ;
}

// CHECK-LABEL: define {{.*}}void @_Z30collapse_stmt_expr_init_unusedv(
// CHECK:         %a = alloca i32,
// CHECK-NOT:     %a{{[0-9]+}} = alloca
// CHECK:         store i32 0, ptr %a,
// CHECK-NOT:     store i32 0, ptr %a,
// CHECK:         ret void
void collapse_stmt_expr_init_unused() {
#pragma omp for collapse(2)
  for (int i = ({int a = 0; 0; }); i < 10; i++)
    for (int j = i; j < 10 + i; j++)
    ;
}

// CHECK-LABEL: define {{.*}}void @_Z23collapse_stmt_expr_condv(
// CHECK:         %b = alloca i32,
// CHECK-NOT:     %b{{[0-9]+}} = alloca
// CHECK:         store i32 10, ptr %b,
// CHECK-NOT:     store i32 10, ptr %b,
// CHECK:         ret void
void collapse_stmt_expr_cond() {
#pragma omp for collapse(2)
  for (int i = 0; i < ({int b = 10; 10; }); i++)
    for (int j = i; j < 10 + i; j++)
    ;
}

// CHECK-LABEL: define {{.*}}void @_Z23collapse_stmt_expr_stepv(
// CHECK:         %c = alloca i32,
// CHECK-NOT:     %c{{[0-9]+}} = alloca
// CHECK:         store i32 1, ptr %c,
// CHECK-NOT:     store i32 1, ptr %c,
// CHECK:         ret void
void collapse_stmt_expr_step() {
#pragma omp for collapse(2)
  for (int i = 0; i < 10; i += ({int c = 1; 1; }))
    for (int j = i; j < 10 + i; j++)
    ;
}

// CHECK-LABEL: define {{.*}}void @_Z33collapse_stmt_expr_lb_non_rect_ubv(
// CHECK:         %c = alloca i32,
// CHECK-NOT:     %c{{[0-9]+}} = alloca
// CHECK:         store i32 0, ptr %c,
// CHECK-NOT:     store i32 0, ptr %c,
// CHECK:         ret void
void collapse_stmt_expr_lb_non_rect_ub() {
#pragma omp for collapse(2)
  for (int i = 0; i < 10; i++)
    for (int j = ({int c = 0; 0; }); j < 10 + i; j++)
    ;
}

// CHECK-LABEL: define {{.*}}void @_Z14stmt_expr_initv(
// CHECK:         %a = alloca i32,
// CHECK-NOT:     %a{{[0-9]+}} = alloca
// CHECK:         store i32 0, ptr %a,
// CHECK-NOT:     store i32 0, ptr %a,
// CHECK:         ret void
void stmt_expr_init() {
#pragma omp for
  for (int i = ({int a = 0; 0; }); i < 10; i++)
    ;
}

// CHECK-LABEL: define {{.*}}void @_Z19simd_stmt_expr_initv(
// CHECK:         %a = alloca i32,
// CHECK-NOT:     %a{{[0-9]+}} = alloca
// CHECK:         store i32 0, ptr %a,
// CHECK-NOT:     store i32 0, ptr %a,
// CHECK:         ret void
void simd_stmt_expr_init() {
#pragma omp simd collapse(2)
  for (int i = ({int a = 0; 0; }); i < 10; i++)
    for (int j = i; j < 10 + i; j++)
    ;
}

// CHECK-LABEL: define {{.*}}void @_Z17collapse_baselinev(
// CHECK:         ret void
void collapse_baseline() {
#pragma omp for collapse(2)
  for (int i = 0; i < 10; i++)
    for (int j = i; j < 10 + i; j++)
    ;
}
