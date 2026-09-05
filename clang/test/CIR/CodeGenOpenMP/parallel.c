// RUN: not %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void before(int);
void during(int);
void after(int);

void emit_simple_parallel() {
  // CHECK: cir.func{{.*}}@emit_simple_parallel
  int i = 5;
  before(i);
  // CHECK: %[[I_LOAD:.*]] = cir.load{{.*}}
  // CHECK-NEXT: cir.call @before(%[[I_LOAD]])

#pragma omp parallel
  {}
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel
  {
    during(i);
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: {{.*}} = cir.load align(4) %{{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: cir.call @during(%{{.*}}) : (!s32i {{.*}}) -> ()
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

  after(i);
  // CHECK: %[[I_LOAD:.*]] = cir.load{{.*}}
  // CHECK-NEXT: cir.call @after(%[[I_LOAD]])
}

void parallel_with_operations() {
  // CHECK: cir.func{{.*}}@parallel_with_operations
  int a, b;
  // CHECK-NEXT: cir.alloca "a"
  // CHECK-NEXT: cir.alloca "b"
  // TODO(OMP): At the moment this results in 3 NYI diagnostics, 1 each for the
  // clauses + 1 for the CapturedStmt. When those are implemented, the check
  // lines will need updating.
#pragma omp parallel shared(a) firstprivate(b)
  {
   a = a + 1;
   b = b + 1;
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: cir.load align(4) %{{.*}}
  // CHECK-NEXT: cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: cir.add nsw %{{.*}}, %{{.*}} : !s32i
  // CHECK-NEXT: cir.store align(4) %{{.*}}, %{{.*}} : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.load align(4) %{{.*}}
  // CHECK-NEXT: cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: cir.add nsw %{{.*}}, %{{.*}} : !s32i
  // CHECK-NEXT: cir.store align(4) %{{.*}}, %{{.*}} : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}
void proc_bind_parallel() {
  // CHECK: cir.func{{.*}}@proc_bind_parallel
#pragma omp parallel proc_bind(master)
  {}
  // CHECK-NEXT: omp.parallel proc_bind(master) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel proc_bind(close)
  {}
  // CHECK-NEXT: omp.parallel proc_bind(close) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel proc_bind(spread)
  {}
  // CHECK-NEXT: omp.parallel proc_bind(spread) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel proc_bind(primary)
  {}
  // CHECK-NEXT: omp.parallel proc_bind(primary) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
#pragma omp parallel proc_bind(default)
  {}
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}

void if_parallel() {
  // CHECK: cir.func{{.*}}@if_parallel

  int validCondition = 10;
  int invalidCondition = 0;
  void *nullPtr = ((void *)0);

  // CHECK-NEXT: %[[VALID_CONDITION_ADDR:.*]] = cir.alloca "validCondition"
  // CHECK-NEXT: %[[INVALID_CONDITION_ADDR:.*]] = cir.alloca "invalidCondition"
  // CHECK-NEXT: %[[NULL_ADDR:.*]] = cir.alloca "nullPtr"

  #pragma omp parallel if (1)
  {}
  // CHECK: %[[ONE_CONST:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_BOOL:.*]] = cir.cast int_to_bool %[[ONE_CONST]]
  // CHECK-NEXT: %[[ONE_U1:.*]] = cir.cast bool_to_int %[[ONE_BOOL]]
  // CHECK-NEXT: %[[ONE_I1:.*]] = cir.builtin_int_cast %[[ONE_U1]]
  // CHECK-NEXT: omp.parallel if(%[[ONE_I1]]) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel if (validCondition)
  {}
  // CHECK-NEXT: %[[VALID_CONDITION_PTR:.*]] = cir.load align(4) %[[VALID_CONDITION_ADDR]]
  // CHECK-NEXT: %[[VALID_CONDITION_BOOL:.*]] = cir.cast int_to_bool %[[VALID_CONDITION_PTR]]
  // CHECK-NEXT: %[[VALID_CONDITION_U1:.*]] = cir.cast bool_to_int %[[VALID_CONDITION_BOOL]]
  // CHECK-NEXT: %[[VALID_CONDITION_I1:.*]] = cir.builtin_int_cast %[[VALID_CONDITION_U1]]
  // CHECK-NEXT: omp.parallel if(%[[VALID_CONDITION_I1]]) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel if (nullPtr)
  {}
  // CHECK-NEXT: %[[NULL_PTR:.*]] = cir.load align(8) %[[NULL_ADDR]]
  // CHECK-NEXT: %[[NULL_BOOL:.*]] = cir.cast ptr_to_bool %[[NULL_PTR]]
  // CHECK-NEXT: %[[NULL_U1:.*]] = cir.cast bool_to_int %[[NULL_BOOL]]
  // CHECK-NEXT: %[[NULL_I1:.*]] = cir.builtin_int_cast %[[NULL_U1]]
  // CHECK-NEXT: omp.parallel if(%[[NULL_I1]]) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel if (invalidCondition) 
  {}
  // CHECK-NEXT: %[[INVALID_CONDITION_PTR:.*]] = cir.load align(4) %[[INVALID_CONDITION_ADDR]]
  // CHECK-NEXT: %[[INVALID_CONDITION_BOOL:.*]] = cir.cast int_to_bool %[[INVALID_CONDITION_PTR]]
  // CHECK-NEXT: %[[INVALID_CONDITION_U1:.*]] = cir.cast bool_to_int %[[INVALID_CONDITION_BOOL]]
  // CHECK-NEXT: %[[INVALID_CONDITION_I1:.*]] = cir.builtin_int_cast %[[INVALID_CONDITION_U1]]
  // CHECK-NEXT: omp.parallel if(%[[INVALID_CONDITION_I1]]) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel if (parallel: validCondition)
  {}
  // CHECK-NEXT: %[[VALID_CONDITION_DIRECTIVE_PTR:.*]] = cir.load align(4) %[[VALID_CONDITION_ADDR]]
  // CHECK-NEXT: %[[VALID_CONDITION_DIRECTIVE_BOOL:.*]] = cir.cast int_to_bool %[[VALID_CONDITION_DIRECTIVE_PTR]]
  // CHECK-NEXT: %[[VALID_CONDITION_DIRECTIVE_U1:.*]] = cir.cast bool_to_int %[[VALID_CONDITION_DIRECTIVE_BOOL]]
  // CHECK-NEXT: %[[VALID_CONDITION_DIRECTIVE_I1:.*]] = cir.builtin_int_cast %[[VALID_CONDITION_DIRECTIVE_U1]]
  // CHECK-NEXT: omp.parallel if(%[[VALID_CONDITION_DIRECTIVE_I1]]) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel if (parallel: invalidCondition)
  {}
  // CHECK-NEXT: %[[INVALID_CONDITION_DIRECTIVE_PTR:.*]] = cir.load align(4) %[[INVALID_CONDITION_ADDR]]
  // CHECK-NEXT: %[[INVALID_CONDITION_DIRECTIVE_BOOL:.*]] = cir.cast int_to_bool %[[INVALID_CONDITION_DIRECTIVE_PTR]]
  // CHECK-NEXT: %[[INVALID_CONDITION_DIRECTIVE_U1:.*]] = cir.cast bool_to_int %[[INVALID_CONDITION_DIRECTIVE_BOOL]]
  // CHECK-NEXT: %[[INVALID_CONDITION_DIRECTIVE_I1:.*]] = cir.builtin_int_cast %[[INVALID_CONDITION_DIRECTIVE_U1]]
  // CHECK-NEXT: omp.parallel if(%[[INVALID_CONDITION_DIRECTIVE_I1]]) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}
