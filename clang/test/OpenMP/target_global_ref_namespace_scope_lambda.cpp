// RUN: %clang_cc1 -verify -fopenmp -fblocks -fsyntax-only %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fblocks -fsyntax-only %s
// RUN: %clang_cc1 -fopenmp -fblocks -DCODEGEN -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

// A reference without local storage used in a target region inside a lambda or
// block at namespace scope used to assert in SemaOpenMP::isOpenMPCapturedDecl,
// and such target regions have no parent function declaration in CodeGen.

int x;
int &ref = x;

auto lambda = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = 42;
};

auto nested_lambda = []() {
  return []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}_l[[#@LINE+1]]{{"?}}(
#pragma omp target
    ref = 42;
  };
};

auto combined_directive = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}_l[[#@LINE+1]]{{"?}}(
#pragma omp target teams
  ref = 42;
};

auto static_local = []() {
  static int &local_ref = x;
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  local_ref = 42;
};

void (^block)() = ^{
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = 42;
};

void default_argument(int = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = 42;
  return 0;
}());

template <int N> int variable_template = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = N;
  return 0;
}();
int instantiation = variable_template<1>;

void use() {
  lambda();
  nested_lambda()();
  combined_directive();
  static_local();
  default_argument();
}

#ifndef CODEGEN
// Reproducer from GH223397.
int &foo = []() { // expected-error {{non-const lvalue reference to type 'int' cannot bind to a temporary of type '(lambda at}}
#pragma omp target
  foo(42); // expected-error {{called object type 'int' is not a function or function pointer}}
};
#endif
