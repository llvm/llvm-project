// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -emit-pch -o %t %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -ast-dump %s | FileCheck %s --check-prefix=DUMP

#ifndef HEADER
#define HEADER

typedef void *omp_interop_t;

void brace_specs() {
  omp_interop_t obj;

  // Single brace entry with fr() only.
  // PRINT: #pragma omp interop init(prefer_type({fr("sycl")}), targetsync : obj)
  #pragma omp interop init(prefer_type({fr("sycl")}), targetsync: obj)

  // Multiple brace entries, each with fr() only.
  // PRINT: #pragma omp interop init(prefer_type({fr("sycl")}, {fr("level_zero")}, {fr("opencl")}), targetsync : obj)
  #pragma omp interop init(prefer_type({fr("sycl")}, {fr("level_zero")}, \
                                       {fr("opencl")}), targetsync: obj)

  // fr() + attr() on one entry.
  // PRINT: #pragma omp interop init(prefer_type({fr("sycl"), attr("ompx_propX")}), targetsync : obj)
  #pragma omp interop init(prefer_type({fr("sycl"), attr("ompx_propX")}), \
                           targetsync: obj)

  // attr() only (no fr()) -- any runtime.
  // PRINT: #pragma omp interop init(prefer_type({attr("ompx_propX", "ompx_propY")}), targetsync : obj)
  #pragma omp interop init(prefer_type({attr("ompx_propX", "ompx_propY")}), \
                           targetsync: obj)

  // Integer expression inside fr() on brace-grouped entry.
  // PRINT: #pragma omp interop init(prefer_type({fr(4)}, {fr(6)}), targetsync : obj)
  #pragma omp interop init(prefer_type({fr(4)}, {fr(6)}), targetsync: obj)

  // Mixed flat + brace entries
  // PRINT: #pragma omp interop init(prefer_type({fr("sycl")}, {fr("opencl")}), targetsync : obj)
  #pragma omp interop init(prefer_type("sycl", {fr("opencl")}), \
                           targetsync: obj)

  // Multiple pref-specs mixing fr-only, fr+attr, and attr-only.
  // PRINT: #pragma omp interop init(prefer_type({fr("sycl"), attr("ompx_propX")}, {fr("level_zero")}, {attr("ompx_propY")}), targetsync : obj)
  #pragma omp interop init(prefer_type({fr("sycl"), attr("ompx_propX")}, {fr("level_zero")}, {attr("ompx_propY")}), targetsync: obj)
}

template <int N>
void tmpl(omp_interop_t obj) {
  // PRINT: #pragma omp interop init(prefer_type({fr(N), attr("ompx_propX")}), targetsync : obj)
  #pragma omp interop init(prefer_type({fr(N), attr("ompx_propX")}), targetsync: obj)
}

// DUMP: FunctionDecl {{.*}} tmpl 'void (omp_interop_t)' explicit_instantiation_definition
// DUMP: TemplateArgument integral '7'
// DUMP: OMPInitClause
// DUMP: IntegerLiteral {{.*}} 'int' 7
// DUMP: StringLiteral {{.*}} "ompx_propX"
template void tmpl<7>(omp_interop_t);
#endif // HEADER
