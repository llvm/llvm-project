// REQUIRES: amdgpu-registered-target

// At -O1 and above, locals get lifetime markers whose end markers are emitted
// from a cir.cleanup.scope. These tests check that such cleanup scopes stay
// inside the OpenMP region they belong to, and that FlattenCFG can flatten a
// cleanup scope or loop containing an OpenMP construct.

// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -O2 -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -O2 -emit-llvm -fclangir %s -o - | FileCheck %s --check-prefix=LLVM

void use(int);

// A local declared in the parallel region is ended inside the region, before
// the region's terminator.
void local_in_parallel(int n) {
#pragma omp parallel
  {
    int x = n;
    use(x);
  }
}

// CIR-LABEL: cir.func{{.*}}@local_in_parallel
// CIR:         omp.parallel {
// CIR:           %[[X:.*]] = cir.alloca "x"
// CIR:           cir.lifetime.start %[[X]]
// CIR:           cir.cleanup.scope {
// CIR:             cir.call @use(
// CIR:             cir.yield
// CIR:           } cleanup normal {
// CIR:             cir.lifetime.end %[[X]]
// CIR:             cir.yield
// CIR:           }
// CIR-NEXT:      omp.terminator
// CIR-NEXT:    }

// Same for a target region.
void local_in_target(int n) {
#pragma omp target map(tofrom : n)
  {
    int x = n;
    n = x + 1;
  }
}

// CIR-LABEL: cir.func{{.*}}@local_in_target
// CIR:         omp.target
// CIR:           %[[X:.*]] = cir.alloca "x"
// CIR:           cir.lifetime.start %[[X]]
// CIR:           cir.cleanup.scope {
// CIR:           } cleanup normal {
// CIR:             cir.lifetime.end %[[X]]
// CIR:           }
// CIR-NEXT:      omp.terminator
// CIR-NEXT:    }

// A local declared before the construct puts the omp.parallel inside the
// local's cleanup scope, which must still be flattened.
int local_before_parallel(void) {
  int s = 0;
#pragma omp parallel
  use(s);
  return s;
}

// CIR-LABEL: cir.func{{.*}}@local_before_parallel
// CIR:         cir.cleanup.scope {
// CIR:           omp.parallel {
// CIR:         } cleanup normal {
// CIR:           cir.lifetime.end

// LLVM-LABEL: define{{.*}} i32 @local_before_parallel(
// LLVM:         call void @llvm.lifetime.start.p0(ptr {{.*}}%[[S:[0-9]+]])
// LLVM:         call void {{.*}}@__kmpc_fork_call(
// LLVM:         call void @llvm.lifetime.end.p0(ptr {{.*}}%[[S]])

// A loop containing an OpenMP construct must also be flattened.
void parallel_in_loop(int n) {
  for (int i = 0; i < n; i = i + 1) {
#pragma omp parallel
    use(i);
  }
}

// CIR-LABEL: cir.func{{.*}}@parallel_in_loop
// CIR:         cir.for : cond {
// CIR:         } body {
// CIR:           omp.parallel {

// LLVM-LABEL: define{{.*}} void @parallel_in_loop(
// LLVM:         br i1 {{.*}}, label %[[BODY:.*]], label
// LLVM:       [[BODY]]:
// LLVM:         call void {{.*}}@__kmpc_fork_call(
// LLVM:         br i1 {{.*}}, label %[[BODY]], label

// A local in the loop body is ended inside the omp.loop_nest, before its
// yield.
void local_in_for(int *a, int n) {
#pragma omp for
  for (int i = 0; i < n; i = i + 1) {
    int t = i * 2;
    a[i] = t;
  }
}

// CIR-LABEL: cir.func{{.*}}@local_in_for
// CIR:         omp.wsloop
// CIR:           omp.loop_nest
// CIR:             %[[T:.*]] = cir.alloca "t"
// CIR:             cir.lifetime.start %[[T]]
// CIR:             cir.cleanup.scope {
// CIR:             } cleanup normal {
// CIR:               cir.lifetime.end %[[T]]
// CIR:             }
// CIR-NEXT:        omp.yield

// Sema's loop helper variables (pre-init captures, the iteration variable and
// the private counter) are emitted without lifetime markers, so nothing comes
// between the combined omp.parallel and its omp.wsloop.
void local_in_parallel_for(int *a, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i = i + 1) {
    int t = i * 2;
    a[i] = t;
  }
}

// CIR-LABEL: cir.func{{.*}}@local_in_parallel_for
// CIR:         omp.parallel {
// CIR-NOT:       cir.lifetime.start
// CIR-NOT:       cir.cleanup.scope
// CIR:           omp.wsloop
// CIR:             omp.loop_nest
// CIR:               cir.lifetime.start
// CIR:               cir.cleanup.scope {

// LLVM-LABEL: define{{.*}} void @local_in_parallel_for(
// LLVM:         call void {{.*}}@__kmpc_fork_call(

// A local in the teams region is ended inside the region, before its
// terminator.
void local_in_teams(void) {
#pragma omp target teams
  {
    int t = 2;
    use(t);
  }
}

// CIR-LABEL: cir.func{{.*}}@local_in_teams
// CIR:         omp.target
// CIR:           omp.teams {
// CIR:             %[[T:.*]] = cir.alloca "t"
// CIR:             cir.lifetime.start %[[T]]
// CIR:             cir.cleanup.scope {
// CIR:             } cleanup normal {
// CIR:               cir.lifetime.end %[[T]]
// CIR:             }
// CIR-NEXT:        omp.terminator

// LLVM-LABEL: define{{.*}} void @local_in_teams(
// LLVM:         call void {{.*}}@__kmpc_fork_teams(

// The composite loop's iteration variable is emitted without lifetime markers,
// so the composite omp.parallel, omp.distribute and omp.wsloop stay directly
// nested and the kernel is still SPMD.
void local_in_teams_distribute_parallel_for(void) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 10; i++) {
    int t = i * 2;
    use(t);
  }
}

// CIR-LABEL: cir.func{{.*}}@local_in_teams_distribute_parallel_for
// CIR:         omp.target kernel_type(spmd)
// CIR:           omp.teams {
// CIR:             omp.parallel
// CIR-NOT:           cir.lifetime.start
// CIR-NOT:           cir.cleanup.scope
// CIR:               omp.distribute {
// CIR-NEXT:            omp.wsloop {
// CIR-NEXT:              omp.loop_nest
// CIR:                     cir.lifetime.start
// CIR:                     cir.cleanup.scope {
// CIR:                     omp.yield

// LLVM-LABEL: define{{.*}} void @local_in_teams_distribute_parallel_for(
// LLVM:         call i32 @__tgt_target_kernel(
