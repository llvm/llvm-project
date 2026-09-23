// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -verify -fsyntax-only %s

// OpenMP 6.0 [14.3]: a replayable construct in a taskgraph region must not
// generate a detachable, transparent or undeferred task.  Each restriction is
// on the replayable construct, so replayable(false) is the way to write one of
// these inside a taskgraph region.  The same cases are covered for flang in
// flang/test/Semantics/OpenMP/taskgraph.f90.

typedef unsigned long omp_event_handle_t;

// The impex constants are not required to be constant expressions in C/C++,
// so a 'transparent' argument the compiler cannot read is left alone.  Spell
// them as an enum here to exercise the cases where it can.
enum omp_impex_t {
  omp_not_impex = 0,
  omp_import = 1,
  omp_export = 2,
  omp_impex = 3
};

void detachable() {
  omp_event_handle_t ev;

#pragma omp taskgraph
  {
#pragma omp task detach(ev) // expected-error {{detachable replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }

  // Ok: detachable, but not replayable.
#pragma omp taskgraph
  {
#pragma omp task detach(ev) replayable(0)
    {}
  }

  // Ok: outside any taskgraph region.
#pragma omp task detach(ev)
  {}
}

void transparent(enum omp_impex_t which) {
#pragma omp taskgraph
  {
#pragma omp task transparent // expected-error {{transparent replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
#pragma omp task transparent(omp_impex) // expected-error {{transparent replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }

  // Ok: omp_not_impex leaves the task non-transparent.
#pragma omp taskgraph
  {
#pragma omp task transparent(omp_not_impex)
    {}
  }

  // Ok: transparent, but not replayable.
#pragma omp taskgraph
  {
#pragma omp task transparent replayable(0)
    {}
  }

  // Ok: the impex value is not known here, so it is not diagnosed.
#pragma omp taskgraph
  {
#pragma omp task transparent(which)
    {}
  }
}

void undeferred(int cond) {
#pragma omp taskgraph
  {
#pragma omp task if (0) // expected-error {{undeferred replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
#pragma omp task if (task : 0) // expected-error {{undeferred replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
#pragma omp taskloop if (0) // expected-error {{undeferred replayable task is not allowed within '#pragma omp taskgraph'}}
    for (int i = 0; i < 4; ++i)
      ;
  }

  // Ok: the condition is not known to be false, so no undeferred task is
  // necessarily generated.  A runtime that records one anyway is on its own.
#pragma omp taskgraph
  {
#pragma omp task if (cond)
    {}
  }

  // Ok: true is a deferred task.
#pragma omp taskgraph
  {
#pragma omp task if (1)
    {}
  }

  // Ok: undeferred, but not replayable.
#pragma omp taskgraph
  {
#pragma omp task if (0) replayable(0)
    {}
  }
}

// A bare replayable clause, and one whose argument is not a constant, both
// leave the construct replayable.
void replayable_forms(int cond) {
  omp_event_handle_t ev;

#pragma omp taskgraph
  {
#pragma omp task detach(ev) replayable // expected-error {{detachable replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }

#pragma omp taskgraph
  {
#pragma omp task detach(ev) replayable(1) // expected-error {{detachable replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }

  // Not a constant, so it may reach a recording: diagnosed, as in flang.
#pragma omp taskgraph
  {
#pragma omp task detach(ev) replayable(cond) // expected-error {{detachable replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }
}

// The body of a generated task is not part of the taskgraph region, so the
// restrictions do not reach into it.
void nested() {
  omp_event_handle_t ev;

#pragma omp taskgraph
  {
#pragma omp task
    {
#pragma omp task detach(ev)
      {}
    }
  }
}

// A target data region's structured block, on the other hand, is executed by
// the encountering task, so its constructs are in the taskgraph region.
void within_a_target_data(int n) {
  omp_event_handle_t ev;

#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : n)
    {
#pragma omp task detach(ev) // expected-error {{detachable replayable task is not allowed within '#pragma omp taskgraph'}}
      {}
    }
  }
}

template <int N> void templated() {
  omp_event_handle_t ev;

#pragma omp taskgraph
  {
#pragma omp task detach(ev) replayable(N) // expected-error {{detachable replayable task is not allowed within '#pragma omp taskgraph'}}
    {}
  }
}

void instantiate() {
  templated<1>(); // expected-note {{in instantiation of function template specialization 'templated<1>' requested here}}
  templated<0>();
}
