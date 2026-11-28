// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -fopenmp -disable-llvm-passes %s -o /dev/null -verify
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -emit-pch -o %t %s
// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -fopenmp -disable-llvm-passes -include-pch %t %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PCH --implicit-check-not=error:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -fsyntax-only %s -verify=syntax

// The parts of the dispatch construct that are not implemented yet are
// diagnosed when generating code; the construct is otherwise well-formed.

// syntax-no-diagnostics

// What Sema records about a variant's 'adjust_args' and 'append_args' clauses
// is preserved in a PCH.
// PCH: dispatch_unsupported.c:{{[0-9]+}}:1: error: cannot compile this OpenMP dispatch directive with a 'device' clause yet
// PCH: dispatch_unsupported.c:{{[0-9]+}}:1: error: cannot compile this OpenMP dispatch directive calling a function variant with an 'adjust_args' clause yet
// PCH: dispatch_unsupported.c:{{[0-9]+}}:1: error: cannot compile this OpenMP dispatch directive calling a function variant with an 'adjust_args' clause yet
// PCH: dispatch_unsupported.c:{{[0-9]+}}:1: error: cannot compile this OpenMP dispatch directive calling a function variant with an 'append_args' clause yet

#ifndef HEADER
#define HEADER

typedef void *omp_interop_t;

void f(void);

void adj_v(int *);
#pragma omp declare variant(adj_v) match(construct = {dispatch})              \
    adjust_args(need_device_ptr : p)
void adj(int *p);

// 'adjust_args(nothing: ...)' does not adjust anything.
void nothing_v(int *);
#pragma omp declare variant(nothing_v) match(construct = {dispatch})          \
    adjust_args(nothing : p)
void nothing(int *p);

void app_v(int, omp_interop_t);
#pragma omp declare variant(app_v) match(construct = {dispatch})              \
    append_args(interop(targetsync))
void app(int);

// A variant that cannot be selected on this target.
void other_v(int, omp_interop_t);
#pragma omp declare variant(other_v)                                          \
    match(construct = {dispatch}, device = {arch(nvptx64)})                   \
    append_args(interop(targetsync))
void other(int);

void test(int d, int *p, int x) {
  // expected-error@+1 {{cannot compile this OpenMP dispatch directive with a 'device' clause yet}}
#pragma omp dispatch device(d)
  f();

  // expected-error@+1 {{cannot compile this OpenMP dispatch directive calling a function variant with an 'adjust_args' clause yet}}
#pragma omp dispatch
  adj(p);

  // 'novariants' does not avoid the error: its condition is only known at run
  // time.
  // expected-error@+1 {{cannot compile this OpenMP dispatch directive calling a function variant with an 'adjust_args' clause yet}}
#pragma omp dispatch novariants(x)
  adj(p);

#pragma omp dispatch
  nothing(p);

  // expected-error@+1 {{cannot compile this OpenMP dispatch directive calling a function variant with an 'append_args' clause yet}}
#pragma omp dispatch
  app(x);

#pragma omp dispatch
  other(x);
}

#endif
