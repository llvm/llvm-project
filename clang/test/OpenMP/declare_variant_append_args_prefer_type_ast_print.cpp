// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -fsyntax-only -verify %s

// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -emit-pch -o %t %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -include-pch %t -ast-print %s \
// RUN:   | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -ast-dump %s \
// RUN:   | FileCheck %s --check-prefix=DUMP

#ifndef HEADER
#define HEADER

typedef void *omp_interop_t;

// Basic append_args with prefer_type (non-template).
void foo_v1(float *A, omp_interop_t IOp);

// PRINT: #pragma omp declare variant(foo_v1) match(construct={dispatch}) append_args(interop(prefer_type({fr("cuda")}),target))
#pragma omp declare variant(foo_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr("cuda")}), target))
void foo(float *A) {}

// append_args with prefer_type containing fr() + attr().
void bar_v1(float *A, omp_interop_t IOp);

// PRINT: #pragma omp declare variant(bar_v1) match(construct={dispatch}) append_args(interop(prefer_type({fr("sycl"),attr("ompx_gpu")}),targetsync))
#pragma omp declare variant(bar_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr("sycl"), attr("ompx_gpu")}), targetsync))
void bar(float *A) {}

// Template: prefer_type with integer expression in fr().
template <typename T>
void tmpl_v1(T *A, omp_interop_t IOp);

template <typename T>
void tmpl_bar(T *A);

// PRINT: #pragma omp declare variant(tmpl_v1<int>) match(construct={dispatch}) append_args(interop(prefer_type({fr(1)}),target))
#pragma omp declare variant(tmpl_v1<int>) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr(1)}), target))
void tmpl_bar(int *A) {}

// Template with dependent expression in fr().
template <int N>
void dep_v1(float *A, omp_interop_t IOp);

template <int N>
void dep_bar(float *A);

// PRINT: #pragma omp declare variant(dep_v1<N>) match(construct={dispatch}) append_args(interop(prefer_type({fr(N)}),target))
#pragma omp declare variant(dep_v1<N>) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr(N)}), target))
template <int N>
void dep_bar(float *A) {}

// DUMP: FunctionDecl{{.*}}dep_bar 'void (float *)' explicit_instantiation_definition
// DUMP: OMPDeclareVariantAttr
// DUMP: IntegerLiteral{{.*}}'int' 4
template void dep_bar<4>(float *);

// Multiple prefer_type entries with attr() only.
void multi_v1(float *A, omp_interop_t IOp);

// PRINT: #pragma omp declare variant(multi_v1) match(construct={dispatch}) append_args(interop(prefer_type({attr("ompx_propA")},{fr(2),attr("ompx_propB")}),target))
#pragma omp declare variant(multi_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr("ompx_propA")}, {fr(2), attr("ompx_propB")}), target))
void multi(float *A) {}

#endif // HEADER
