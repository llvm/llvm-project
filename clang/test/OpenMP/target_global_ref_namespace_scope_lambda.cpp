// RUN: %clang_cc1 -verify -fopenmp -fblocks -fsyntax-only %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fblocks -fsyntax-only %s

// A reference without local storage used in a target region inside a lambda or
// block at namespace scope used to assert in SemaOpenMP::isOpenMPCapturedDecl.

int x;
int &ref = x;

auto lambda = []() {
#pragma omp target
  ref = 42;
};

auto nested_lambda = []() {
  return []() {
#pragma omp target
    ref = 42;
  };
};

auto combined_directive = []() {
#pragma omp target teams
  ref = 42;
};

auto static_local = []() {
  static int &local_ref = x;
#pragma omp target
  local_ref = 42;
};

void (^block)() = ^{
#pragma omp target
  ref = 42;
};

void default_argument(int = []() {
#pragma omp target
  ref = 42;
  return 0;
}());

template <int N> int variable_template = []() {
#pragma omp target
  ref = N;
  return 0;
}();
int instantiation = variable_template<1>;

// Reproducer from GH223397.
int &foo = []() { // expected-error {{non-const lvalue reference to type 'int' cannot bind to a temporary of type '(lambda at}}
#pragma omp target
  foo(42); // expected-error {{called object type 'int' is not a function or function pointer}}
};
