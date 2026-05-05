// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -verify -fsyntax-only %s

// OpenMP 6.0 [14.3, taskgraph Construct, Restrictions]: task-generating
// constructs are the only constructs that may be encountered as part of the
// taskgraph region.  A construct that is not task-generating records no node,
// so it would run on the recording execution and then be missing from every
// replay.  The 6.1 draft keeps the restriction and spells it with the sharper
// explicit-task-generating property.  The same cases are covered for flang in
// flang/test/Semantics/OpenMP/taskgraph.f90.

void not_task_generating(int n) {
#pragma omp taskgraph
  {
    // A taskgraph is itself not task-generating, so it cannot be nested.
#pragma omp taskgraph // expected-error {{'taskgraph' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
#pragma omp parallel // expected-error {{'parallel' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
#pragma omp taskgroup // expected-error {{'taskgroup' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
#pragma omp critical // expected-error {{'critical' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
#pragma omp barrier // expected-error {{'barrier' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
  }

#pragma omp taskgraph
  {
#pragma omp taskyield // expected-error {{'taskyield' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
  }

#pragma omp taskgraph
  {
#pragma omp flush // expected-error {{'flush' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
  }

#pragma omp taskgraph
  {
#pragma omp single // expected-error {{'single' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
    int x = 0;
#pragma omp atomic // expected-error {{'atomic' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    x++;
  }

#pragma omp taskgraph
  {
#pragma omp unroll partial(2) // expected-error {{'unroll' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    for (int i = 0; i < n; ++i)
      ;
  }
}

// Only the outermost leaf of a compound construct decides.  It is the leaf that
// generates the explicit task; the remaining leaves are encountered inside that
// task's region, which [2, region] excludes from the taskgraph region.
void compound(int n) {
#pragma omp taskgraph
  {
#pragma omp masked taskloop // expected-error {{'masked taskloop' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    for (int i = 0; i < n; ++i)
      ;
  }

#pragma omp taskgraph
  {
#pragma omp parallel masked taskloop // expected-error {{'parallel masked taskloop' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    for (int i = 0; i < n; ++i)
      ;
  }

  // Ok: the target leaf generates the target task.
#pragma omp taskgraph
  {
#pragma omp target teams distribute parallel for
    for (int i = 0; i < n; ++i)
      ;
#pragma omp target parallel
    {}
#pragma omp taskloop simd
    for (int i = 0; i < n; ++i)
      ;
  }
}

// The task-generating constructs, which are also the ones the replayable clause
// may appear on, plus target data: a composite of target_enter_data, a task and
// target_exit_data, so task-generating as well.
void task_generating(int n) {
  int x = 0;

#pragma omp taskgraph
  {
#pragma omp task
    {}
#pragma omp taskloop
    for (int i = 0; i < n; ++i)
      ;
#pragma omp target
    {}
#pragma omp target enter data map(to : x)
#pragma omp target update to(x)
#pragma omp target exit data map(from : x)
#pragma omp target data map(tofrom : x)
    {}
#pragma omp taskwait depend(in : x)
  }
}

// A taskwait is task-generating only with a depend clause, which has its own
// diagnostic rather than this one.
void taskwait_without_depend() {
#pragma omp taskgraph
  {
#pragma omp taskwait // expected-error {{directive '#pragma omp taskwait' within '#pragma omp taskgraph' must use 'depend' clause to be task-generating}}
  }
}

// Only constructs are restricted, so the informational and utility directives
// are left alone.
void not_a_construct() {
#pragma omp taskgraph
  {
#pragma omp nothing
#pragma omp assume no_openmp_routines
    {}
  }
}

// [2, region] puts neither the body of a generated task nor a target region in
// the region of the encountering thread, so a construct below one of those is
// not encountered in the taskgraph region.
void below_a_generated_task(int n) {
#pragma omp taskgraph
  {
#pragma omp task
    {
#pragma omp parallel
      {}
#pragma omp taskgraph
      {}
    }

#pragma omp taskloop
    for (int i = 0; i < n; ++i) {
#pragma omp critical
      {}
    }

#pragma omp target
    {
#pragma omp parallel
      {}
    }

    // The body of a target data region is that of its constituent task.
#pragma omp target data map(tofrom : n)
    {
#pragma omp taskgroup
      {}
    }
  }
}

// Outside a taskgraph region none of this applies.
void outside() {
#pragma omp parallel
  {
#pragma omp taskgroup
    {
#pragma omp taskgraph
      {
#pragma omp task
        {}
      }
    }
  }
}

template <typename T> void templated(T n) {
#pragma omp taskgraph
  {
#pragma omp parallel // expected-error {{'parallel' directive is not task-generating and cannot be used within '#pragma omp taskgraph'}}
    {}
#pragma omp task
    {}
  }
}

void instantiate() { templated<int>(4); }
