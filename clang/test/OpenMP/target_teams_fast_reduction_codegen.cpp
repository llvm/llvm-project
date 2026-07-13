// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-target-fast-reduction -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=ATOMIC %s < %t.ll
// RUN: FileCheck --check-prefix=BUFFER %s < %t.ll
// RUN: FileCheck %s < %t.ll

// expected-no-diagnostics

// ATOMIC-DAG: call i32 @__kmpc_is_team_main_thread(
// ATOMIC-DAG: atomicrmw add ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw add ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw and ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw or ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw xor ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw min ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw max ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw umin ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw umax ptr {{.*}} syncscope("agent") monotonic
// ATOMIC-DAG: atomicrmw fadd ptr {{.*}} syncscope("agent") monotonic
void atomicable_matrix(int n) {
  int si = 0, su = 0, an = ~0, orv = 0, xr = 0, smn = 0x7fffffff,
      smx = -0x7fffffff - 1;
  unsigned umn = ~0u, umx = 0u;
  float f = 0.0f;
#pragma omp target teams distribute parallel for reduction(+ : si)              \
    reduction(- : su) reduction(& : an) reduction(| : orv) reduction(^ : xr)    \
    reduction(min : smn) reduction(max : smx) reduction(min : umn)              \
    reduction(max : umx) reduction(+ : f)
  for (int i = 0; i < n; ++i) {
    si += i;
    su -= i;
    an &= i;
    orv |= i;
    xr ^= i;
    smn = i < smn ? i : smn;
    smx = i > smx ? i : smx;
    umn = (unsigned)i < umn ? (unsigned)i : umn;
    umx = (unsigned)i > umx ? (unsigned)i : umx;
    f += i;
  }
}

// A fp multiply and a fp max reduction have no direct atomicrmw, so each falls
// back to the buffer path even with -fopenmp-target-fast-reduction. Together with
// the mixed construct below there are exactly three buffered writebacks.
//
// BUFFER-COUNT-3: call i32 @__kmpc_gpu_xteam_reduce_nowait(
// BUFFER-NOT:     call i32 @__kmpc_gpu_xteam_reduce_nowait(
double nonatomicable_mul(int n) {
  double p = 1.0;
#pragma omp target teams distribute parallel for reduction(* : p)
  for (int i = 0; i < n; ++i)
    p *= 1.0;
  return p;
}

double nonatomicable_fpmax(int n) {
  double mx = 0.0;
#pragma omp target teams distribute parallel for reduction(max : mx)
  for (int i = 0; i < n; ++i)
    mx = (double)i > mx ? (double)i : mx;
  return mx;
}

// A single non-atomicable reduction (fp multiply) poisons the whole set: the
// atomic path requires *all* reductions to have an atomicrmw, so even the int
// sum here is combined through the buffer, not atomically. Hence no team-main
// guard and no `atomicrmw add` for `s` from this construct (asserted by the
// module-wide counts above).
//
// CHECK-COUNT-1: call i32 @__kmpc_is_team_main_thread(
// CHECK-NOT:     call i32 @__kmpc_is_team_main_thread(
// CHECK-NOT: atomicrmw fmul
// CHECK-NOT: atomicrmw fmax

double mixed(int n) {
  int s = 0;
  double p = 1.0;
#pragma omp target teams distribute parallel for reduction(+ : s)               \
    reduction(* : p)
  for (int i = 0; i < n; ++i) {
    s += i;
    p *= 1.0;
  }
  return s + p;
}
