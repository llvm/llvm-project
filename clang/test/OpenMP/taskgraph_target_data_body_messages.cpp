// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -fopenmp-targets=amdgcn-amd-amdhsa -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -verify=quiet -fopenmp -fopenmp-version=60 -fopenmp-targets=amdgcn-amd-amdhsa -Wno-openmp-taskgraph -ferror-limit 100 -o - %s

// quiet-no-diagnostics

// The structured block of a 'target data' recorded by a taskgraph is a task
// of its own, whose region is not part of the taskgraph region.  Device
// constructs written in it are not recorded, and dependences on storage from
// outside it order nothing outside it; both are diagnosed, lexically.

typedef void *omp_depend_t;

int Global;

void device_constructs(int *A, int N) {
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : A[0 : N]) // expected-note 6 {{body of this 'target data' construct is a task of its own}}
    {
#pragma omp target // expected-warning {{'#pragma omp target' in the body of a replayed 'target data' construct is not part of the taskgraph}}
      A[0] = 1;
#pragma omp target enter data map(to : A[0 : N]) // expected-warning {{'#pragma omp target_enter_data' in the body}}
#pragma omp target exit data map(release : A[0 : N]) // expected-warning {{'#pragma omp target_exit_data' in the body}}
#pragma omp target update to(A[0 : N]) // expected-warning {{'#pragma omp target_update' in the body}}
#pragma omp target data map(to : A[0 : N]) // expected-warning {{'#pragma omp target_data' in the body}}
      {
        A[1] = 2;
      }
#pragma omp task
      {
#pragma omp target teams // expected-warning {{'#pragma omp target teams' in the body}}
        A[2] = 3;
      }
    }
  }
}

// A 'replayable' clause makes the body a task outside a taskgraph too.
void replayable_clause(int *A, int N) {
#pragma omp target data map(tofrom : A[0 : N]) replayable // expected-note {{body of this 'target data' construct is a task of its own}}
  {
#pragma omp target // expected-warning {{'#pragma omp target' in the body}}
    A[0] = 1;
  }
}

// A target region inside the body starts a region of its own: what is nested
// in it is not the body's business.
void stops_at_target(int *A, int N) {
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : A[0 : N]) // expected-note {{body of this 'target data' construct is a task of its own}}
    {
#pragma omp target map(tofrom : A[0 : N]) // expected-warning {{'#pragma omp target' in the body}}
      {
#pragma omp parallel for
        for (int I = 0; I < N; ++I)
          A[I] = I;
      }
    }
  }
}

void depends(int *A, int N, omp_depend_t Obj) {
  int Outer = 0;
  int Arr[4];
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : A[0 : N]) // expected-note 5 {{body of this 'target data' construct is a task of its own}}
    {
      int Inner = 0;
      int InnerArr[4];
#pragma omp task depend(out : Outer) // expected-warning {{dependence on 'Outer' in the body of a replayed 'target data' construct does not order the construct with tasks outside it}}
      Outer = 1;
#pragma omp task depend(in : Global) // expected-warning {{dependence on 'Global'}}
      (void)Global;
#pragma omp task depend(inout : A[0 : N]) // expected-warning {{dependence on 'A'}}
      A[0] = 1;
#pragma omp task depend(in : Arr[1]) // expected-warning {{dependence on 'Arr'}}
      (void)Arr[1];
#pragma omp taskwait depend(in : Outer) // expected-warning {{dependence on 'Outer'}}
      // Storage declared in the body is the body's own.
#pragma omp task depend(out : Inner)
      Inner = 1;
#pragma omp task depend(inout : InnerArr[0 : 2])
      InnerArr[0] = 1;
#pragma omp task depend(in : Inner)
      {
        int Nested = 0;
#pragma omp task depend(out : Nested)
        Nested = Inner;
      }
      // Dependence objects and iterators are left alone.
#pragma omp task depend(depobj : Obj)
      A[1] = 1;
#pragma omp task depend(iterator(I = 0 : 4), in : Arr[I])
      A[2] = 1;
    }
  }
}

// Only a body the taskgraph records is a task: outside a taskgraph, and
// without 'replayable', the block is merged into the encountering task.
void not_recorded(int *A, int N) {
  int Outer = 0;
#pragma omp target data map(tofrom : A[0 : N])
  {
#pragma omp target
    A[0] = 1;
#pragma omp task depend(out : Outer)
    Outer = 1;
  }
#pragma omp taskgraph
  {
#pragma omp task
    {
#pragma omp target data map(tofrom : A[0 : N])
      {
#pragma omp target
        A[0] = 1;
      }
    }
  }
}

// A taskgraph nested in the body starts a region of its own.
void stops_at_taskgraph(int *A, int N) {
  int Outer = 0;
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : A[0 : N])
    {
#pragma omp taskgraph
      {
#pragma omp task depend(out : Outer)
        Outer = 1;
      }
    }
  }
}

template <typename T> void in_template(T *A, int N) {
  int Outer = 0;
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : A[0 : N]) // expected-note 2 {{body of this 'target data' construct is a task of its own}}
    {
#pragma omp target // expected-warning {{'#pragma omp target' in the body}}
      A[0] = 1;
#pragma omp task depend(out : Outer) // expected-warning {{dependence on 'Outer'}}
      Outer = 1;
    }
  }
}

template void in_template<int>(int *, int);
template void in_template<float>(float *, int);
