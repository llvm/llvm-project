// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 \
// RUN:   -fopenmp-metadirective-runtime-selection -x c++ \
// RUN:   -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

void bar();

// The condition selects the 'when' variant; otherwise the 'otherwise' variant
// runs.
// CHECK-LABEL: define {{.*}}void @_Z14when_otherwisei(
// CHECK:       [[FLAG:%.*]] = load i32, ptr %flag.addr
// CHECK-NEXT:  [[COND:%.*]] = icmp ne i32 [[FLAG]], 0
// CHECK-NEXT:  br i1 [[COND]], label %[[WHEN:omp.metadirective.when[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK:       call {{.*}}@__kmpc_fork_call(
// CHECK:       br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK:       call i32 @__kmpc_single(
// CHECK:       br label %[[END]]
// CHECK:       [[END]]:
// CHECK-NEXT:  ret void
void when_otherwise(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
  bar();
}

// 'when' clauses are tried in order; the first true condition wins.
// CHECK-LABEL: define {{.*}}void @_Z9two_whensii(
// CHECK:       [[A:%.*]] = load i32, ptr %a.addr
// CHECK-NEXT:  [[CONDA:%.*]] = icmp ne i32 [[A]], 0
// CHECK-NEXT:  br i1 [[CONDA]], label %[[WHEN0:omp.metadirective.when[0-9]*]], label %[[ELSE0:omp.metadirective.else[0-9]*]]
// CHECK:       [[WHEN0]]:
// CHECK:       call {{.*}}@__kmpc_fork_call(
// CHECK:       br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[ELSE0]]:
// CHECK:       [[B:%.*]] = load i32, ptr %b.addr
// CHECK-NEXT:  [[CONDB:%.*]] = icmp ne i32 [[B]], 0
// CHECK-NEXT:  br i1 [[CONDB]], label %[[WHEN1:omp.metadirective.when[0-9]*]], label %[[ELSE1:omp.metadirective.else[0-9]*]]
// CHECK:       [[WHEN1]]:
// CHECK:       call i32 @__kmpc_single(
// CHECK:       br label %[[END]]
// CHECK:       [[ELSE1]]:
// CHECK-NEXT:  call void @_Z3barv()
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]:
// CHECK-NEXT:  ret void
void two_whens(int a, int b) {
#pragma omp metadirective when(user={condition(a)}: parallel) \
                          when(user={condition(b)}: single) otherwise()
  bar();
}

// Without 'otherwise', the associated statement runs when no condition holds.
// CHECK-LABEL: define {{.*}}void @_Z12no_otherwisei(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.when[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK:       call {{.*}}@__kmpc_fork_call(
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  call void @_Z3barv()
// CHECK-NEXT:  br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[END]]:
// CHECK-NEXT:  ret void
void no_otherwise(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel)
  bar();
}

// vendor(llvm) always matches, so 'parallel' is always selected.
// CHECK-LABEL: define {{.*}}void @_Z18static_match_firsti(
// CHECK-NOT:   icmp
// CHECK:       call {{.*}}@__kmpc_fork_call(
// CHECK-NOT:   __kmpc_single
// CHECK:       ret void
void static_match_first(int flag) {
#pragma omp metadirective when(implementation={vendor(llvm)}: parallel) \
                          when(user={condition(flag)}: single) otherwise()
  bar();
}

// vendor(amd) never matches, so its variant is not emitted.
// CHECK-LABEL: define {{.*}}void @_Z14static_nomatchi(
// CHECK-NOT:   __kmpc_fork_call
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.when[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK:       call i32 @__kmpc_single(
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  call void @_Z3barv()
// CHECK-NOT:   __kmpc_fork_call
// CHECK:       ret void
void static_nomatch(int flag) {
#pragma omp metadirective when(implementation={vendor(amd)}: parallel) \
                          when(user={condition(flag)}: single) otherwise()
  bar();
}

// vendor(llvm) runs when the condition fails; plain bar() is never called.
// CHECK-LABEL: define {{.*}}void @_Z17static_match_lasti(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.when[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK:       call i32 @__kmpc_single(
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK-NEXT:  br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[END]]:
// CHECK-NEXT:  ret void
void static_match_last(int flag) {
#pragma omp metadirective when(user={condition(flag)}: single) \
                          when(implementation={vendor(llvm)}: parallel)
  bar();
}

// kind(host) scores higher than the condition, so 'parallel' always wins.
// CHECK-LABEL: define {{.*}}void @_Z18device_kind_secondi(
// CHECK-NOT:   icmp
// CHECK:       call {{.*}}@__kmpc_fork_call(
// CHECK-NOT:   __kmpc_single
// CHECK:       ret void
void device_kind_second(int flag) {
#pragma omp metadirective when(user={condition(flag)}: single) \
                          when(device={kind(host)}: parallel)
  bar();
}

// score(5) makes the second variant the first one tried.
// CHECK-LABEL: define {{.*}}void @_Z14score_reordersi(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.when[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK:       call i32 @__kmpc_single(
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK-NEXT:  br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[END]]:
// CHECK-NEXT:  ret void
void score_reorders(int flag) {
#pragma omp metadirective when(implementation={vendor(llvm)}: parallel) \
    when(user={condition(flag)}, implementation={vendor(score(5): llvm)}: single)
  bar();
}
