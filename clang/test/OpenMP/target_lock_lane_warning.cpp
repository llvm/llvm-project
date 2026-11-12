// REQUIRES: amdgpu-registered-target

// On a GPU, whose threads of a warp may run in lockstep, warn about an
// omp_set_lock that several threads of a warp may reach together but that
// cannot be serialized across them, and say why.

// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc -verify=host
// RUN: %clang_cc1 -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o /dev/null -verify
// RUN: %clang_cc1 -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -fsyntax-only %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -verify
// RUN: %clang_cc1 -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o /dev/null -Wno-openmp-lockstep-lock -verify=quiet
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host-nv.bc
// RUN: %clang_cc1 -fopenmp -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host-nv.bc -o /dev/null -verify

// host-no-diagnostics
// quiet-no-diagnostics

struct omp_lock_t {
  void *_lk;
};
extern "C" void omp_set_lock(omp_lock_t *);
extern "C" void omp_unset_lock(omp_lock_t *);

#pragma omp declare target
omp_lock_t L, M, Ls[4];
int X;
void use(int);
// The callers are not known, so there is nothing to say.
void acquire() { omp_set_lock(&L); }
#pragma omp end declare target

void reasons(int C, int N) {
  // expected-note@+1 7 {{threads of a warp may execute this '#pragma omp target parallel' region together; use 'single', 'masked' or 'critical' to run code on one thread at a time}}
#pragma omp target parallel
  {
    // Each case is in a block of its own, since the search for the matching
    // omp_unset_lock stops at the end of the block.
    {
      omp_set_lock(&L); // expected-warning {{because no matching 'omp_unset_lock' follows it in the same block; threads of a warp that contend for the lock will trap or hang}}
      X++;
    }
    {
      omp_set_lock(&M); // expected-warning {{because no matching 'omp_unset_lock' follows it in the same block}}
      X++;
    }
    for (int I = 0; I < N; ++I) {
      omp_set_lock(&L); // expected-warning {{because control can enter or leave the locked code other than at its ends}}
      if (I == 3)
        continue;
      omp_unset_lock(&L);
    }
    {
      omp_set_lock(&L); // expected-warning {{because the locked code uses a lock routine other than in properly nested set/unset pairs}}
      X++;
      if (C)
        omp_unset_lock(&L);
    }
    {
      omp_set_lock(&L); // expected-warning {{because the locked code contains an OpenMP directive}}
#pragma omp barrier
      omp_unset_lock(&L);
    }
    {
      omp_set_lock(&L); // expected-warning {{because the locked code declares a variable}}
      int Y = X;
      omp_unset_lock(&L);
      use(Y);
    }
    {
      omp_set_lock(&Ls[N++]); // expected-warning {{because the lock expression has side effects}}
      omp_unset_lock(&Ls[N]);
    }
  }
}

void contexts(int C) {
  // A worksharing loop inside a combined construct.
  // expected-note@+1 {{this '#pragma omp target teams distribute parallel for' region}}
#pragma omp target teams distribute parallel for
  for (int I = 0; I < 100; ++I) {
    omp_set_lock(&L); // expected-warning {{because the locked code uses a lock routine}}
    if (C)
      omp_unset_lock(&L);
  }

  // Nothing to say where one thread at a time runs the code, or where it is
  // serialized.
#pragma omp target parallel
  {
    omp_set_lock(&L);
    X++;
    omp_unset_lock(&L);

#pragma omp single
    {
      omp_set_lock(&L);
      if (C)
        omp_unset_lock(&L);
    }
#pragma omp masked
    omp_set_lock(&M);
#pragma omp critical
    {
      omp_set_lock(&M);
    }
  }

  // The sequential part of a target region, and one thread per team.
#pragma omp target
  {
    omp_set_lock(&L);
    if (C)
      omp_unset_lock(&L);
  }
#pragma omp target teams distribute
  for (int I = 0; I < 100; ++I) {
    omp_set_lock(&L);
    if (C)
      omp_unset_lock(&L);
  }
}

void one_thread(int C) {
  // The masked part of a combined construct runs the code on one thread.
#pragma omp target
#pragma omp parallel masked
  {
    omp_set_lock(&L);
    if (C)
      omp_unset_lock(&L);
  }

  // So does each section, explicit or implicit, of a sections region.
#pragma omp target
#pragma omp parallel sections
  {
    {
      omp_set_lock(&L);
      if (C)
        omp_unset_lock(&L);
    }
#pragma omp section
    {
      omp_set_lock(&L);
      if (C)
        omp_unset_lock(&L);
    }
  }
}

void statement_forms(int C) {
  // expected-note@+1 2 {{this '#pragma omp target parallel' region}}
#pragma omp target parallel
  {
    switch (C) {
    case 1:
      omp_set_lock(&L); // expected-warning {{because the call is labelled; put the locked code in a block of its own}}
      X++;
      omp_unset_lock(&L);
      break;
    case 2: {
      omp_set_lock(&L);
      X++;
      omp_unset_lock(&L);
      break;
    }
    }
    if (C) {
      omp_set_lock(&M);
      omp_unset_lock(&M);
    }
    if (C)
      omp_set_lock(&M); // expected-warning {{because the call is not a statement of its own in a block}}
  }

  // A construct whose body is not a block.
  // expected-note@+1 {{this '#pragma omp target parallel' region}}
#pragma omp target parallel
  omp_set_lock(&L); // expected-warning {{because the call is not a statement of its own in a block}}

  // The callers of a lambda are not known.
#pragma omp target parallel
  {
    auto F = [&] {
      omp_set_lock(&L);
      if (C)
        omp_unset_lock(&L);
    };
    F();
  }
}

// A function reached from a target region is emitted for the device; one only
// called on the host is not.
void device_helper(int C) {
  // expected-note@+1 {{this '#pragma omp parallel' region}}
#pragma omp parallel
  {
    omp_set_lock(&L); // expected-warning {{because the locked code uses a lock routine}}
    if (C)
      omp_unset_lock(&L);
  }
}

void host_helper(int C) {
#pragma omp parallel
  {
    omp_set_lock(&L);
    if (C)
      omp_unset_lock(&L);
  }
}

void callers(int C) {
#pragma omp target
  device_helper(C); // expected-note {{called by 'callers'}}
  host_helper(C);
}

// One warning however many instantiations there are.
template <int N> void instantiated(int C) {
  // expected-note@+1 {{this '#pragma omp target parallel' region}}
#pragma omp target parallel
  {
    omp_set_lock(&Ls[N]); // expected-warning {{because the locked code uses a lock routine}}
    if (C)
      omp_unset_lock(&Ls[N]);
  }
}

void instantiate(int C) {
  instantiated<1>(C); // expected-note {{in instantiation of function template specialization 'instantiated<1>' requested here}}
  instantiated<2>(C);
}
