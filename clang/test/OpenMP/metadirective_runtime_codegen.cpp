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
// CHECK-NEXT:  br i1 [[COND]], label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK-NEXT:  br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[OTHER]]:
// CHECK-NEXT:  call i32 @__kmpc_single(
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
// CHECK-NEXT:  br i1 [[CONDA]], label %[[V0:omp.metadirective.variant[0-9]*]], label %[[ELSE0:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE0]]:
// CHECK-NEXT:  [[B:%.*]] = load i32, ptr %b.addr
// CHECK-NEXT:  [[CONDB:%.*]] = icmp ne i32 [[B]], 0
// CHECK-NEXT:  br i1 [[CONDB]], label %[[V1:omp.metadirective.variant[0-9]*]], label %[[ELSE1:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE1]]:
// CHECK-NEXT:  br label %[[V2:omp.metadirective.variant[0-9]*]]
// CHECK:       [[V0]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK:       [[V1]]:
// CHECK-NEXT:  call i32 @__kmpc_single(
// CHECK:       [[V2]]:
// CHECK-NEXT:  call void @_Z3barv()
// CHECK-NEXT:  br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[END]]:
// CHECK-NEXT:  ret void
void two_whens(int a, int b) {
#pragma omp metadirective when(user={condition(a)}: parallel) \
                          when(user={condition(b)}: single) otherwise()
  bar();
}

// Without 'otherwise', the associated statement runs when no condition holds.
// CHECK-LABEL: define {{.*}}void @_Z12no_otherwisei(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK:       [[OTHER]]:
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
// CHECK:       br label %[[SEL:omp.metadirective.variant[0-9]*]]
// CHECK:       [[SEL]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK:       omp.metadirective.variant{{[0-9]+}}:{{.*}}No predecessors!
// CHECK-NEXT:  call i32 @__kmpc_single(
// CHECK:       omp.metadirective.variant{{[0-9]+}}:{{.*}}No predecessors!
// CHECK-NEXT:  call void @_Z3barv()
void static_match_first(int flag) {
#pragma omp metadirective when(implementation={vendor(llvm)}: parallel) \
                          when(user={condition(flag)}: single) otherwise()
  bar();
}

// vendor(amd) never matches, so its variant is never selected.
// CHECK-LABEL: define {{.*}}void @_Z14static_nomatchi(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       omp.metadirective.variant{{[0-9]*}}:{{.*}}No predecessors!
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK:       [[WHEN]]:
// CHECK-NEXT:  call i32 @__kmpc_single(
// CHECK:       [[OTHER]]:
// CHECK-NEXT:  call void @_Z3barv()
void static_nomatch(int flag) {
#pragma omp metadirective when(implementation={vendor(amd)}: parallel) \
                          when(user={condition(flag)}: single) otherwise()
  bar();
}

// vendor(llvm) runs when the condition fails; plain bar() is never called.
// CHECK-LABEL: define {{.*}}void @_Z17static_match_lasti(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK-NEXT:  call i32 @__kmpc_single(
// CHECK:       [[OTHER]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK:       omp.metadirective.variant{{[0-9]+}}:{{.*}}No predecessors!
// CHECK-NEXT:  call void @_Z3barv()
void static_match_last(int flag) {
#pragma omp metadirective when(user={condition(flag)}: single) \
                          when(implementation={vendor(llvm)}: parallel)
  bar();
}

// kind(host) scores higher than the condition, so 'parallel' always wins.
// CHECK-LABEL: define {{.*}}void @_Z18device_kind_secondi(
// CHECK-NOT:   icmp
// CHECK:       br label %[[SEL:omp.metadirective.variant[0-9]*]]
// CHECK:       omp.metadirective.variant{{[0-9]*}}:{{.*}}No predecessors!
// CHECK-NEXT:  call i32 @__kmpc_single(
// CHECK:       [[SEL]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
void device_kind_second(int flag) {
#pragma omp metadirective when(user={condition(flag)}: single) \
                          when(device={kind(host)}: parallel)
  bar();
}

// score(5) makes the second variant the first one tried.
// CHECK-LABEL: define {{.*}}void @_Z14score_reordersi(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[OTHER]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call(
// CHECK:       [[WHEN]]:
// CHECK-NEXT:  call i32 @__kmpc_single(
void score_reorders(int flag) {
#pragma omp metadirective when(implementation={vendor(llvm)}: parallel) \
    when(user={condition(flag)}, implementation={vendor(score(5): llvm)}: single)
  bar();
}

void baz(int);

// Loop variants: each one runs its own loop.
// CHECK-LABEL: define {{.*}}void @_Z13loop_variantsii(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call({{.*}}@_Z13loop_variantsii.omp_outlined
// CHECK:       [[OTHER]]:
// CHECK:       call void @__kmpc_for_static_init_4(
// CHECK:       call void @_Z3bazi(
// CHECK:       call void @__kmpc_for_static_fini(
// CHECK:       call void @__kmpc_barrier(
// CHECK:       define internal void @_Z13loop_variantsii.omp_outlined(
// CHECK:       call void @__kmpc_for_static_init_4(
// CHECK:       call void @_Z3bazi(
// CHECK:       call void @__kmpc_for_static_fini(
void loop_variants(int flag, int n) {
#pragma omp metadirective when(user={condition(flag)}: parallel for) otherwise(for)
  for (int i = 0; i < n; ++i)
    baz(i);
}

// A metadirective inside a variant of another one. The empty 'otherwise()'
// runs the block, so the inner metadirective runs on both paths.
// CHECK-LABEL: define {{.*}}void @_Z6nestedii(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call({{.*}}@_Z6nestedii.omp_outlined, ptr %b.addr)
// CHECK:       [[OTHER]]:
// CHECK-NEXT:  {{%.*}} = load i32, ptr %b.addr
// CHECK-NEXT:  {{%.*}} = icmp ne i32
// CHECK-NEXT:  br i1 {{.*}}, label %[[IWHEN:omp.metadirective.variant[0-9]*]], label %[[IELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[IELSE]]:
// CHECK-NEXT:  br label %[[IOTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[IWHEN]]:
// CHECK-NEXT:  call i32 @__kmpc_single(
// CHECK:       [[IOTHER]]:
// CHECK-NEXT:  call i32 @__kmpc_masked(
// CHECK:       define internal void @_Z6nestedii.omp_outlined(
// CHECK:       br i1 {{.*}}, label %[[OWHEN:omp.metadirective.variant[0-9]*]], label %[[OELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[OELSE]]:
// CHECK-NEXT:  br label %[[OOTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[OWHEN]]:
// CHECK:       call i32 @__kmpc_single(
// CHECK:       [[OOTHER]]:
// CHECK:       call i32 @__kmpc_masked(
void nested(int a, int b) {
#pragma omp metadirective when(user={condition(a)}: parallel) otherwise()
  {
#pragma omp metadirective when(user={condition(b)}: single) otherwise(masked)
    bar();
  }
}

// construct={parallel} matches inside the parallel region and scores higher,
// so 'single' is always selected.
// CHECK-LABEL: define internal void @_Z12runtime_pathi.omp_outlined(
// CHECK-NOT:   icmp
// CHECK:       br label %[[SEL:omp.metadirective.variant[0-9]*]]
// CHECK:       [[SEL]]:
// CHECK:       call i32 @__kmpc_single(
// CHECK:       omp.metadirective.variant{{[0-9]+}}:{{.*}}No predecessors!
// CHECK:       call i32 @__kmpc_masked(
void runtime_path(int flag) {
#pragma omp parallel
  {
#pragma omp metadirective when(construct={parallel}: single) \
                          when(user={condition(flag)}: masked)
    bar();
  }
}

// A standalone variant runs its directive and then the statement.
// CHECK-LABEL: define internal void @_Z5mixedi.omp_outlined(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK:       call void @__kmpc_barrier(
// CHECK-NEXT:  call void @_Z3barv()
// CHECK-NEXT:  br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[OTHER]]:
// CHECK:       call i32 @__kmpc_single(
// CHECK:       [[END]]:
// CHECK-NEXT:  ret void
void mixed(int flag) {
#pragma omp parallel
  {
#pragma omp metadirective when(user={condition(flag)}: barrier) otherwise(single)
    bar();
  }
}

// All variants are standalone: the statement runs once, after the selection.
// CHECK-LABEL: define internal void @_Z14all_standalonei.omp_outlined(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK:       call void @__kmpc_barrier(
// CHECK-NEXT:  br label %[[END:omp.metadirective.end[0-9]*]]
// CHECK:       [[OTHER]]:
// CHECK:       call i32 @__kmpc_omp_taskyield(
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]:
// CHECK-NEXT:  call void @_Z3barv()
// CHECK-NEXT:  ret void
void all_standalone(int flag) {
#pragma omp parallel
  {
#pragma omp metadirective when(user={condition(flag)}: barrier) otherwise(taskyield)
    bar();
  }
}

// Instantiated template. Kept last: the instantiation may be emitted at the
// end of the file.
// CHECK-LABEL: define {{.*}}void @_Z4tmplIiEvT_(
// CHECK:       br i1 {{.*}}, label %[[WHEN:omp.metadirective.variant[0-9]*]], label %[[ELSE:omp.metadirective.else[0-9]*]]
// CHECK:       [[ELSE]]:
// CHECK-NEXT:  br label %[[OTHER:omp.metadirective.variant[0-9]*]]
// CHECK:       [[WHEN]]:
// CHECK-NEXT:  call {{.*}}@__kmpc_fork_call({{.*}}@_Z4tmplIiEvT_.omp_outlined
// CHECK:       [[OTHER]]:
// CHECK-NEXT:  call i32 @__kmpc_single(
template <typename T> void tmpl(T flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
  bar();
}
void inst() { tmpl(1); }
