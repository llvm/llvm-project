// REQUIRES: amdgpu-registered-target

// omp_set_lock ... omp_unset_lock sequences in C++ on AMDGPU: in templates,
// where the decision is made for each instantiation, and alongside lambdas,
// classes and exceptions.

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++17 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++17 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --implicit-check-not=invoke --implicit-check-not=landingpad
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++17 -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host-eh.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++17 -fexceptions -fcxx-exceptions -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-eh.bc -o - | FileCheck %s --implicit-check-not=invoke --implicit-check-not=landingpad
// expected-no-diagnostics

struct omp_lock_t {
  void *_lk;
};
extern "C" void omp_set_lock(omp_lock_t *);
extern "C" void omp_unset_lock(omp_lock_t *);

// Not the OpenMP routines.
namespace mine {
void omp_set_lock(omp_lock_t *);
void omp_unset_lock(omp_lock_t *);
} // namespace mine

void may_throw();

// Exceptions are not supported in device code, so with or without
// -fexceptions no call in a sequence can leave it by unwinding: there is no
// invoke or landingpad anywhere in the output.

#pragma omp declare target
omp_lock_t L, Ls[4];
int X;

template <typename T> void bump(T &V) {
  omp_set_lock(&L);
  V += 1;
  may_throw();
  omp_unset_lock(&L);
}

template <int I> void bump_indexed() {
  omp_set_lock(&Ls[I]);
  X++;
  omp_unset_lock(&Ls[I]);
}

template <int I, int J> void bump_pair() {
  omp_set_lock(&Ls[I]);
  X++;
  omp_unset_lock(&Ls[J]);
}

template <bool B> void unset_if() {
  omp_set_lock(&L);
  X++;
  if constexpr (B)
    omp_unset_lock(&L);
}

template <typename T> void return_if(T C) {
  omp_set_lock(&L);
  if (C)
    return;
  omp_unset_lock(&L);
}

template <typename T> struct Guarded {
  omp_lock_t Lock;
  T Val;
  void bump() {
    omp_set_lock(&Lock);
    ++Val;
    omp_unset_lock(&Lock);
  }
};

void with_lambda(int C) {
  omp_set_lock(&L);
  [&] {
    if (C)
      return;
    X++;
  }();
  omp_unset_lock(&L);
}

void not_openmp() {
  mine::omp_set_lock(&L);
  X++;
  mine::omp_unset_lock(&L);
}

void instantiate(int &I, double &D, Guarded<int> &G) {
  bump(I);
  bump(D);
  bump_indexed<2>();
  bump_pair<1, 1>();
  bump_pair<1, 2>();
  unset_if<true>();
  unset_if<false>();
  return_if(I);
  G.bump();
}
#pragma omp end declare target

// A lambda's return leaves only the lambda.
// CHECK-LABEL: define {{.*}}void @_Z11with_lambdai(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       ret void

// CHECK-LABEL: define {{.*}}void @_Z10not_openmpv(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void

// Each instantiation is considered on its own.
// CHECK-LABEL: define {{.*}}void @_Z4bumpIiEvRT_(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       omp.lanes.body:
// CHECK-NEXT:    call void @omp_set_lock(
// CHECK:         call void @_Z9may_throwv()
// CHECK:         call void @omp_unset_lock(
// CHECK:       ret void

// CHECK-LABEL: define {{.*}}void @_Z4bumpIdEvRT_(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       ret void

// CHECK-LABEL: define {{.*}}void @_Z12bump_indexedILi2EEvv(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       ret void

// The two lock expressions are the same once I and J are substituted...
// CHECK-LABEL: define {{.*}}void @_Z9bump_pairILi1ELi1EEvv(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       ret void

// ...but not here.
// CHECK-LABEL: define {{.*}}void @_Z9bump_pairILi1ELi2EEvv(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void

// The unset is conditional in the template, whichever branch survives.
// CHECK-LABEL: define {{.*}}void @_Z8unset_ifILb1EEvv(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void

// CHECK-LABEL: define {{.*}}void @_Z8unset_ifILb0EEvv(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void

// CHECK-LABEL: define {{.*}}void @_Z9return_ifIiEvT_(
// CHECK-NOT:   __kmpc_warp_active_thread_mask
// CHECK:       ret void

// CHECK-LABEL: define {{.*}}void @_ZN7GuardedIiE4bumpEv(
// CHECK:       call i64 @__kmpc_warp_active_thread_mask()
// CHECK:       ret void
