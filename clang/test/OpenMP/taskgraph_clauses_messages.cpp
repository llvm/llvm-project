// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -verify=expected,omp51 -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -verify=expected,omp60 -fsyntax-only %s

// Tests that the 'graph_id' and 'graph_reset' clauses are accepted in OpenMP 6.0
// and rejected in prior versions on the 'taskgraph' directive. Also tests
// duplicate-clause and invalid-condition diagnostics.

void foo() {}

void taskgraph_clauses_messages() {
  int A = 1;

  // Basic version error tests.
  #pragma omp taskgraph graph_id(0) // omp51-error {{unexpected OpenMP clause 'graph_id' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}}
  {}

  #pragma omp taskgraph graph_reset(true) // omp51-error {{unexpected OpenMP clause 'graph_reset' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}}
  {}

  // Same, without argument.
  #pragma omp taskgraph graph_reset // omp51-error {{unexpected OpenMP clause 'graph_reset' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}}
  {}

  // Duplicate clause tests (OMP 6.0 only; in OMP 5.1 both are unexpected).
  #pragma omp taskgraph graph_id(0) graph_id(1) // omp51-error {{unexpected OpenMP clause 'graph_id' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP clause 'graph_id' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}} expected-error {{directive '#pragma omp taskgraph' cannot contain more than one 'graph_id' clause}}
  {}

  #pragma omp taskgraph graph_reset(true) graph_reset(false) // omp51-error {{unexpected OpenMP clause 'graph_reset' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP clause 'graph_reset' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}} expected-error {{directive '#pragma omp taskgraph' cannot contain more than one 'graph_reset' clause}}
  {}

  #pragma omp taskgraph graph_id(foo()) // omp51-error {{unexpected OpenMP clause 'graph_id' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}} omp60-error {{expression must have integral or unscoped enumeration type, not 'void'}}
  {}

  #pragma omp taskgraph graph_reset(foo()) // omp51-error {{unexpected OpenMP clause 'graph_reset' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}} omp60-error {{value of type 'void' is not contextually convertible to 'bool'}}
  {}
}
