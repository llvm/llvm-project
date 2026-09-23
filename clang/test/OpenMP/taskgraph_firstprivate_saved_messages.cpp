// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s

// Tests the OpenMP 6.0 'saved' modifier on the 'firstprivate' clause.  The
// modifier is meaningful only on constructs that create tasks or taskloops
// (the units of work that can participate in taskgraph replay).  Every other
// directive that admits a 'firstprivate' clause must reject it.

void unknown_modifier() {
  int a = 0;
  // The diagnostic comes from the generic "expected <list> in OpenMP clause"
  // path and enumerates the legal modifier names ('saved' in OpenMP 6.0).
  #pragma omp task firstprivate(bogus: a) // expected-error {{expected 'saved' in OpenMP clause 'firstprivate'}}
  { (void)a; }
}

void rejected_on_non_tasking_constructs() {
  int a = 0;
  int b[8];

  // parallel
  #pragma omp parallel firstprivate(saved: a) // expected-error {{'saved' modifier on 'firstprivate' clause is only allowed on a 'task', 'taskloop', or 'target' construct}}
  { (void)a; }

  // for (worksharing)
  #pragma omp parallel
  {
    #pragma omp for firstprivate(saved: a) // expected-error {{'saved' modifier on 'firstprivate' clause is only allowed on a 'task', 'taskloop', or 'target' construct}}
    for (int i = 0; i < 4; ++i) (void)a;
  }

  // sections
  #pragma omp parallel
  {
    #pragma omp sections firstprivate(saved: a) // expected-error {{'saved' modifier on 'firstprivate' clause is only allowed on a 'task', 'taskloop', or 'target' construct}}
    {
      (void)a;
    }
  }

  // single
  #pragma omp parallel
  {
    #pragma omp single firstprivate(saved: a) // expected-error {{'saved' modifier on 'firstprivate' clause is only allowed on a 'task', 'taskloop', or 'target' construct}}
    { (void)a; }
  }

  // teams (inside target -- the inner directive is a standalone 'teams',
  // not a 'target teams' combined directive, so it is not a target
  // execution directive in its own right).
  #pragma omp target
  #pragma omp teams firstprivate(saved: a) // expected-error {{'saved' modifier on 'firstprivate' clause is only allowed on a 'task', 'taskloop', or 'target' construct}}
  { (void)a; }

  // distribute (inside teams) -- standalone 'distribute' is not a target
  // execution directive either.
  #pragma omp target teams
  #pragma omp distribute firstprivate(saved: a) // expected-error {{'saved' modifier on 'firstprivate' clause is only allowed on a 'task', 'taskloop', or 'target' construct}}
  for (int i = 0; i < 4; ++i) (void)a;
}

void accepted_on_task_taskloop_and_target() {
  int a = 0;

  // Bare task (no enclosing taskgraph, no replayable clause): accepted.
  // Per OpenMP 6.0 [7.2] the 'saved' modifier silently has no effect on a
  // non-replayable construct; we only enforce the directive-kind check
  // statically.  In this implementation a bare task / taskloop is never
  // recorded into a taskgraph -- recording requires either lexical
  // nesting inside a '#pragma omp taskgraph' or an explicit 'replayable'
  // clause -- so the modifier is a well-defined no-op here.
  #pragma omp task firstprivate(saved: a)
  { (void)a; }

  // Bare taskloop: accepted, same rationale.
  #pragma omp taskloop firstprivate(saved: a)
  for (int i = 0; i < 4; ++i) (void)a;

  // Replayable task: explicitly opted-in for replay.
  #pragma omp task replayable firstprivate(saved: a)
  { (void)a; }

  // Replayable taskloop.
  #pragma omp taskloop replayable firstprivate(saved: a)
  for (int i = 0; i < 4; ++i) (void)a;

  // Task lexically nested inside a taskgraph.
  #pragma omp taskgraph
  {
    #pragma omp task firstprivate(saved: a)
    { (void)a; }
  }

  // Bare target: accepted on the same well-formed-but-no-effect grounds
  // as a bare task.  Per OpenMP 6.0 [14.6] the 'target' construct admits
  // both 'firstprivate' and 'replayable', so 'saved' is meaningful as
  // soon as a 'replayable' clause is added or the construct is nested
  // inside a 'taskgraph' region.
  #pragma omp target firstprivate(saved: a)
  { (void)a; }

  // Replayable target: explicitly opted-in for replay.
  #pragma omp target replayable firstprivate(saved: a)
  { (void)a; }

  // Combined target construct (target + parallel): accepted because the
  // composite directive is a target execution directive in its own
  // right, so the captured snapshot at the target boundary belongs to a
  // construct that may participate in taskgraph replay.
  #pragma omp target parallel firstprivate(saved: a)
  { (void)a; }

  // Combined target teams distribute parallel for: same rationale.
  #pragma omp target teams distribute parallel for firstprivate(saved: a)
  for (int i = 0; i < 4; ++i) (void)a;
}
