// REQUIRES: amdgpu-registered-target

// On AMDGPU, whose lanes run in lockstep, an omp_set_lock ... omp_unset_lock
// sequence with straight-line control flow between the calls is run by the
// lanes of a wavefront one at a time, so that lanes contending for the lock do
// not deadlock. Anything that could leave a lock held from one lane's turn to
// the next, or leave the sequence other than at its end, is left alone.

// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// expected-no-diagnostics

typedef struct omp_lock_t {
  void *_lk;
} omp_lock_t;
typedef struct omp_nest_lock_t {
  void *_lk;
} omp_nest_lock_t;
extern void omp_set_lock(omp_lock_t *);
extern void omp_unset_lock(omp_lock_t *);
extern int omp_test_lock(omp_lock_t *);
extern void omp_set_nest_lock(omp_nest_lock_t *);
extern void omp_unset_nest_lock(omp_nest_lock_t *);

// HOST-NOT: __kmpc_warp_active_thread_mask

#pragma omp declare target
omp_lock_t L, M, Ls[4];
omp_nest_lock_t N;
int X;
void use(int);
void with_lock_routine(void (*)(omp_lock_t *), omp_lock_t *);

// CHECK-LABEL: define {{.*}}void @straight(
// CHECK:       %omp.lanes.mask = call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       omp.lanes.header:
// CHECK:       omp.lanes.body:
// CHECK-NEXT:    call void @omp_set_lock(ptr noundef {{.*}}@L{{.*}})
// CHECK:         call void @omp_unset_lock(ptr noundef {{.*}}@L{{.*}})
// CHECK:       omp.lanes.latch:
// CHECK-NEXT:    call void @__kmpc_syncwarp(i64 %omp.lanes.mask)
// CHECK:       omp.lanes.exit:
// CHECK:         call void @use(
void straight(void) {
  omp_set_lock(&L);
  X++;
  omp_unset_lock(&L);
  use(X);
}

// Control flow that stays inside the region, and a lock chosen per thread.
// CHECK-LABEL: define {{.*}}void @nested_flow(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       omp.lanes.body:
// CHECK:         call void @omp_set_lock(
// CHECK:         call void @use(
// CHECK:         call void @omp_unset_lock(
void nested_flow(int N) {
  omp_set_lock(&Ls[N % 4]);
  for (int I = 0; I < N; ++I) {
    if (I == 3)
      break;
    use(I);
  }
  omp_unset_lock(&Ls[N % 4]);
}

// The lanes are serialized from each omp_set_lock to its own omp_unset_lock.
// CHECK-LABEL: define {{.*}}void @two_regions(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:         call void @omp_set_lock(ptr noundef {{.*}}@L{{.*}})
// CHECK:         call void @omp_unset_lock(ptr noundef {{.*}}@L{{.*}})
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:         call void @omp_set_lock(ptr noundef {{.*}}@M{{.*}})
// CHECK:         call void @omp_unset_lock(ptr noundef {{.*}}@M{{.*}})
// CHECK:       ret void
void two_regions(void) {
  omp_set_lock(&L);
  X++;
  omp_unset_lock(&L);
  omp_set_lock(&M);
  X++;
  omp_unset_lock(&M);
}

// A pair nested properly inside another is part of the outer sequence.
// CHECK-LABEL: define {{.*}}void @nested_pairs(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:         call void @omp_set_lock(ptr noundef {{.*}}@L{{.*}})
// CHECK:         call void @omp_set_lock(ptr noundef {{.*}}@M{{.*}})
// CHECK:         call void @omp_unset_lock(ptr noundef {{.*}}@M{{.*}})
// CHECK:         call void @omp_unset_lock(ptr noundef {{.*}}@L{{.*}})
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void nested_pairs(void) {
  omp_set_lock(&L);
  omp_set_lock(&M);
  X++;
  omp_unset_lock(&M);
  omp_unset_lock(&L);
}

// A sequence inside a nested block, and a switch inside the sequence.
// CHECK-LABEL: define {{.*}}void @in_block(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       ret void
void in_block(int C) {
  if (C) {
    omp_set_lock(&L);
    switch (C) {
    case 1:
      X++;
      break;
    default:
      break;
    }
    omp_unset_lock(&L);
  }
}

// The following are left alone.

// CHECK-LABEL: define {{.*}}void @mismatched(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void mismatched(void) {
  omp_set_lock(&L);
  X++;
  omp_unset_lock(&M);
}

// CHECK-LABEL: define {{.*}}void @early_return(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void early_return(int C) {
  omp_set_lock(&L);
  if (C)
    return;
  omp_unset_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @escaping_break(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void escaping_break(int C) {
  for (;;) {
    omp_set_lock(&L);
    if (C)
      break;
    omp_unset_lock(&L);
  }
}

// CHECK-LABEL: define {{.*}}void @declaration(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void declaration(void) {
  omp_set_lock(&L);
  int Y = X;
  omp_unset_lock(&L);
  use(Y);
}

// CHECK-LABEL: define {{.*}}void @directive(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void directive(void) {
  omp_set_lock(&L);
#pragma omp barrier
  omp_unset_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @side_effect(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void side_effect(int N) {
  omp_set_lock(&Ls[N++]);
  omp_unset_lock(&Ls[N++]);
}

// The unset does not follow the set on every path.

// CHECK-LABEL: define {{.*}}void @set_only(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void set_only(void) {
  omp_set_lock(&L);
  X++;
}

// CHECK-LABEL: define {{.*}}void @conditional_unset(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void conditional_unset(int C) {
  omp_set_lock(&L);
  X++;
  if (C)
    omp_unset_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @unset_in_both_branches(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void unset_in_both_branches(int C) {
  omp_set_lock(&L);
  if (C) {
    X++;
    omp_unset_lock(&L);
  } else {
    omp_unset_lock(&L);
  }
}

// CHECK-LABEL: define {{.*}}void @unset_before_set(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void unset_before_set(void) {
  omp_unset_lock(&L);
  X++;
  omp_set_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @escaping_continue(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void escaping_continue(int C) {
  for (int I = 0; I < C; ++I) {
    omp_set_lock(&L);
    if (I == 3)
      continue;
    omp_unset_lock(&L);
  }
}

// CHECK-LABEL: define {{.*}}void @goto_out(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void goto_out(int C) {
  omp_set_lock(&L);
  if (C)
    goto out;
  omp_unset_lock(&L);
out:
  X++;
}

// CHECK-LABEL: define {{.*}}void @label_inside(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void label_inside(int C) {
  if (C)
    goto in;
  omp_set_lock(&L);
in:
  X++;
  omp_unset_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @case_label_inside(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void case_label_inside(int C) {
  switch (C) {
  case 0:
    omp_set_lock(&L);
  case 1:
    X++;
    omp_unset_lock(&L);
  }
}

// A lock could be held from one lane's turn to the next.

// CHECK-LABEL: define {{.*}}void @overlapping_pairs(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void overlapping_pairs(void) {
  omp_set_lock(&L);
  omp_set_lock(&M);
  X++;
  omp_unset_lock(&L);
  omp_unset_lock(&M);
}

// The first omp_set_lock stays outside; the pair that follows it is serialized.
// CHECK-LABEL: define {{.*}}void @relocked(
// CHECK:         call void @omp_set_lock(ptr noundef {{.*}}@L{{.*}})
// CHECK-NEXT:    %omp.lanes.mask = call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       ret void
void relocked(void) {
  omp_set_lock(&L);
  omp_set_lock(&L);
  omp_unset_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @unset_and_reset_conditionally(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void unset_and_reset_conditionally(int C) {
  omp_set_lock(&L);
  if (C) {
    omp_unset_lock(&L);
    omp_set_lock(&L);
  }
  omp_unset_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @test_lock_inside(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void test_lock_inside(void) {
  omp_set_lock(&L);
  if (omp_test_lock(&M))
    X++;
  omp_unset_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @nest_lock_inside(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void nest_lock_inside(void) {
  omp_set_lock(&L);
  omp_set_nest_lock(&N);
  omp_unset_nest_lock(&N);
  omp_unset_lock(&L);
}

// CHECK-LABEL: define {{.*}}void @lock_routine_as_value(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void
void lock_routine_as_value(void) {
  omp_set_lock(&L);
  with_lock_routine(omp_set_lock, &M);
  omp_unset_lock(&L);
}
#pragma omp end declare target
