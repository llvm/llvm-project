// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify %s

namespace {
// TODO this must be fixed. This warning shouldn't be generated.
// expected-warning@+1{{function '(anonymous namespace)::bar' has internal linkage but is not defined}}
void bar();
} // namespace

#pragma omp begin declare variant match(user = {condition(1)})
void bar() {
}
#pragma omp end declare variant

// expected-warning@+1{{function 'baz' has internal linkage but is not defined}}
static void baz();
#pragma omp begin declare variant match(device = {kind(nohost)})
static void baz() {}
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(host)})
static void foo() {}
#pragma omp end declare variant

int main() {
  foo();
  // expected-note@+1{{used here}}
  baz();
  // expected-note@+1{{used here}}
  bar();

  return 0;
}
