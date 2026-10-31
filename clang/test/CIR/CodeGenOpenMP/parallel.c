// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -emit-cir -fclangir %s -o - | FileCheck %s

// CHECK: cir.global "private" constant cir_private @__const.default_parallel.arr {{.*}} !cir.array<!s32i x 1> align(4)

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
#pragma omp parallel
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

void default_parallel() {
  // CHECK: cir.func {{.*}} @default_parallel

  int x = 0;
  int y = 0;
  int *p = &y;
  int arr[1] = {0}; // Checked at the beginning of the file.

  // CHECK-NEXT: %[[X_ADDR:.*]] = cir.alloca "x"
  // CHECK-NEXT: %[[Y_ADDR:.*]] = cir.alloca "y"
  // CHECK-NEXT: %[[P_ADDR:.*]] = cir.alloca "p"
  // CHECK-NEXT: %[[ARR_ADDR:.*]] = cir.alloca "arr"

#pragma omp parallel default(none)
  {}
  // CHECK: omp.parallel {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel default(none)
  {
    int inside = 10;
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: %[[INSIDE_ADDR:.*]] = cir.alloca "inside"
  // CHECK-NEXT: %[[TEN_ADDR:.*]] = cir.const #cir.int<10>
  // CHECK-NEXT: cir.store align(4) %[[TEN_ADDR]], %[[INSIDE_ADDR]]
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel default(shared)
  {}
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel default(shared)
  {
    x += 5;
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: %[[FIVE_ADDR:.*]] = cir.const #cir.int<5>
  // CHECK-NEXT: %[[PTR_X_ADDR:.*]] = cir.load align(4) %[[X_ADDR]]
  // CHECK-NEXT: %[[INCREMENT_X_ADDR:.*]] = cir.add nsw %[[PTR_X_ADDR]], %[[FIVE_ADDR]]
  // CHECK-NEXT: cir.store align(4) %[[INCREMENT_X_ADDR]], %[[X_ADDR]]
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel default(shared)
  {
    arr[0] += x;
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: %[[PTR_X_ADDR:.*]] = cir.load align(4) %[[X_ADDR]]
  // CHECK-NEXT: %[[ZERO_IDX_ADDR:.*]] = cir.const #cir.int<0>
  // CHECK-NEXT: %[[ARR_ELEMENT_ADDR:.*]] = cir.get_element %[[ARR_ADDR]][%[[ZERO_IDX_ADDR]] : !s64i]
  // CHECK-NEXT: %[[PTR_ARR_ELEMENT_ADDR:.*]] = cir.load align(4) %[[ARR_ELEMENT_ADDR]]
  // CHECK-NEXT: %[[INCREMENT_ELEMENT_ADDR:.*]] = cir.add nsw %[[PTR_ARR_ELEMENT_ADDR]], %[[PTR_X_ADDR]]
  // CHECK-NEXT: cir.store align(4) %[[INCREMENT_ELEMENT_ADDR]], %[[ARR_ELEMENT_ADDR]]
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

#pragma omp parallel default(shared)
  {
    *p += x;
  }
  // CHECK-NEXT: omp.parallel {
  // CHECK-NEXT: %[[PTR_X_ADDR:.*]] = cir.load align(4) %[[X_ADDR]]
  // CHECK-NEXT: %[[DEREF_P_ADDR:.*]] = cir.load deref align(8) %[[P_ADDR]]
  // CHECK-NEXT: %[[LOAD_DEREF_P_ADDR:.*]] = cir.load align(4) %[[DEREF_P_ADDR]]
  // CHECK-NEXT: %[[INCREMENT_P_ADDR:.*]] = cir.add nsw %[[LOAD_DEREF_P_ADDR]], %[[PTR_X_ADDR]]
  // CHECK-NEXT: cir.store align(4) %[[INCREMENT_P_ADDR]], %[[DEREF_P_ADDR]]
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
}

void num_threads_parallel() {
  // CHECK: cir.func{{.*}}@num_threads_parallel

  int numThreads = 4;

  // CHECK-NEXT: %[[NUM_THREADS_ADDR:.*]] = cir.alloca "numThreads"
  // CHECK-NEXT: %[[CONST_4:.*]] = cir.const #cir.int<4>
  // CHECK-NEXT: cir.store align(4) %[[CONST_4]], %[[NUM_THREADS_ADDR]]
  #pragma omp parallel num_threads(16)
  {}
  // CHECK-NEXT: %[[CONST_16:.*]] = cir.const #cir.int<16>
  // CHECK-NEXT: %[[CONST_16_I32:.*]] = cir.builtin_int_cast %[[CONST_16]]
  // CHECK-NEXT: omp.parallel num_threads(%[[CONST_16_I32]] : i32) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }

  #pragma omp parallel num_threads(numThreads) 
  {}
  // CHECK-NEXT: %[[NUM_THREADS_PTR:.*]] = cir.load align(4) %[[NUM_THREADS_ADDR]]
  // CHECK-NEXT: %[[NUM_THREADS_I32:.*]] = cir.builtin_int_cast %[[NUM_THREADS_PTR]]
  // CHECK-NEXT: omp.parallel num_threads(%[[NUM_THREADS_I32]] : i32) {
  // CHECK-NEXT: omp.terminator
  // CHECK-NEXT: }
}
