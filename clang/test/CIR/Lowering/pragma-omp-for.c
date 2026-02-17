// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --input-file %t-cir.ll

void before(int);
void during(int);
void after(int);

// Test simple for loop with constant bounds: for (int i = 0; i < 10; i++)
void emit_simple_for() {
  int j = 5;
  before(j);
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
      during(j);
    }
  }
  after(j);
}

// CHECK-LABEL: define dso_local void @emit_simple_for()
// CHECK: call void @before(i32 %{{.*}})
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 1, ptr @emit_simple_for..omp_par, ptr %{{.*}})
// CHECK: call void @after(i32 %{{.*}})

// CHECK-LABEL: define internal void @emit_simple_for..omp_par(
// CHECK: store i32 0, ptr %p.lowerbound
// CHECK: store i32 9, ptr %p.upperbound
// CHECK: store i32 1, ptr %p.stride
// CHECK: call void @__kmpc_for_static_init_4u(
// CHECK: omp_loop.body:
// CHECK: omp.loop_nest.region:
// CHECK: store i32 %{{.*}}, ptr %{{.*}}, align 4
// CHECK: call void @during(i32 %{{.*}})
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_barrier(

// Test for loop with variable bounds and type conversions
void emit_for_with_vars() {
  int j = 5;
  before(j);
#pragma omp parallel
  {
    int lb = 1;
    long ub = 10;
    short step = 1;
#pragma omp for
    for (int i = 0; i < ub; i = i + step) {
      during(j);
    }
  }
  after(j);
}

// CHECK-LABEL: define dso_local void @emit_for_with_vars()
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 1, ptr @emit_for_with_vars..omp_par, ptr %{{.*}})

// CHECK-LABEL: define internal void @emit_for_with_vars..omp_par(
// variable upper bound: loaded and truncated from i64 to i32
// CHECK: %{{.*}} = trunc i64 %{{.*}} to i32
// variable step: loaded and sign-extended from i16 to i32
// CHECK: %{{.*}} = sext i16 %{{.*}} to i32
// CHECK: call void @__kmpc_for_static_init_4u(
// CHECK: omp.loop_nest.region:
// CHECK: call void @during(i32 %{{.*}})
// CHECK: call void @__kmpc_for_static_fini(

// Test induction variable is accessible in the loop body: during(i)
void emit_for_with_induction_var() {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
      during(i);
    }
  }
}

// CHECK-LABEL: define internal void @emit_for_with_induction_var..omp_par(
// CHECK: store i32 0, ptr %p.lowerbound
// CHECK: store i32 9, ptr %p.upperbound
// CHECK: omp.loop_nest.region:
// IV is stored to the alloca and then loaded for during(i)
// CHECK: store i32 %{{.*}}, ptr %[[IV_PTR:.*]], align 4
// CHECK: %[[IV_LOAD:.*]] = load i32, ptr %[[IV_PTR]], align 4
// CHECK: call void @during(i32 %[[IV_LOAD]])

// Test inclusive upper bound: for (int i = 0; i <= 9; i++)
void emit_for_inclusive_bound() {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i <= 9; i++) {
      during(i);
    }
  }
}

// CHECK-LABEL: define internal void @emit_for_inclusive_bound..omp_par(
// inclusive i <= 9 has same trip count as i < 10
// CHECK: store i32 0, ptr %p.lowerbound
// CHECK: store i32 9, ptr %p.upperbound
// CHECK: call void @__kmpc_for_static_init_4u(
// CHECK: omp.loop_nest.region:
// CHECK: call void @during(i32 %{{.*}})

// Test reversed comparison: for (int i = 0; 10 > i; i++)
void emit_for_reversed_cmp() {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; 10 > i; i++) {
      during(i);
    }
  }
}

// CHECK-LABEL: define internal void @emit_for_reversed_cmp..omp_par(
// reversed cmp (10 > i) produces same bounds as (i < 10)
// CHECK: store i32 0, ptr %p.lowerbound
// CHECK: store i32 9, ptr %p.upperbound
// CHECK: call void @__kmpc_for_static_init_4u(

// Test reversed inclusive comparison: for (int i = 0; 9 >= i; i++)
void emit_for_reversed_inclusive_cmp() {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; 9 >= i; i++) {
      during(i);
    }
  }
}

// CHECK-LABEL: define internal void @emit_for_reversed_inclusive_cmp..omp_par(
// reversed inclusive cmp (9 >= i) produces same bounds as (i <= 9)
// CHECK: store i32 0, ptr %p.lowerbound
// CHECK: store i32 9, ptr %p.upperbound
// CHECK: call void @__kmpc_for_static_init_4u(

// Test compound assignment step: for (int i = 0; i < 20; i += 2)
void emit_for_compound_step() {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 20; i += 2) {
      during(i);
    }
  }
}

// CHECK-LABEL: define internal void @emit_for_compound_step..omp_par(
// step = 2 visible in the loop body IV computation
// CHECK: call void @__kmpc_for_static_init_4u(
// CHECK: omp_loop.body:
// CHECK: %{{.*}} = mul i32 %{{.*}}, 2
// CHECK: omp.loop_nest.region:
// CHECK: call void @during(i32 %{{.*}})

// Test commuted step expression: for (int i = 0; i < 30; i = step + i)
void emit_for_commuted_step() {
  short step = 3;
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 30; i = step + i) {
      during(i);
    }
  }
}

// CHECK-LABEL: define internal void @emit_for_commuted_step..omp_par(
// variable step loaded and sign-extended from i16
// CHECK: %{{.*}} = sext i16 %{{.*}} to i32
// CHECK: call void @__kmpc_for_static_init_4u(
// CHECK: omp_loop.body:
// step is variable, multiplied into IV
// CHECK: %{{.*}} = mul i32 %{{.*}}, %{{.*}}
// CHECK: omp.loop_nest.region:
// CHECK: call void @during(i32 %{{.*}})

// Verify OpenMP runtime declarations
// CHECK: declare i32 @__kmpc_global_thread_num(ptr)
// CHECK: declare void @__kmpc_for_static_init_4u(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)
// CHECK: declare void @__kmpc_for_static_fini(ptr, i32)
// CHECK: declare void @__kmpc_barrier(ptr, i32)
// CHECK: declare {{.*}}void @__kmpc_fork_call(ptr, i32, ptr, ...)
