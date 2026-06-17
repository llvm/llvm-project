// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -std=c++20 \
// RUN: -triple x86_64-unknown-unknown -emit-llvm %s -o -

struct Point { int x, y; };
struct ArrayStruct { int arr[2]; };

#pragma omp declare reduction(mysum: int: omp_out += omp_in) initializer(omp_pr\
iv = 0)

void test_array_reduction() {
  ArrayStruct s{{1, 2}};
  auto [arr] = s;

#pragma omp parallel for reduction(+:arr) // expected-error {{array-type reduct\
ions on structured bindings are not yet supported}}                             
  for (int i = 0; i < 10; ++i) {
    arr[0] += i;
  }
}

void test_udr_reduction() {
  Point p{0, 0};
  auto [a, b] = p;

#pragma omp parallel for reduction(mysum:a) // expected-error {{user-defined re\
ductions on structured bindings are not yet supported}}                         
  for (int i = 0; i < 10; ++i) {
    a += i;
  }
}

void test_simple_scalar_reduction() {
  Point p{0, 0};
  auto [a, b] = p;

#pragma omp parallel for reduction(+:a)
  for (int i = 0; i < 10; ++i) {
    a += i;
  }
}
