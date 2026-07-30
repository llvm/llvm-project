// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -fsyntax-only

thread_local const int prohobit_ns_scope = 0;
thread_local int prohobit_ns_scope2 = 0;
thread_local const int allow_ns_scope = 0;

struct S {
  static const thread_local int prohibit_static_member;
  static thread_local int prohibit_static_member2;
};

struct T {
  static const thread_local int allow_static_member;
};

void foo() {
  // expected-error@+1{{thread-local storage is not supported for the current target}}
  thread_local const int prohibit_local = 0;
  // expected-error@+1{{thread-local storage is not supported for the current target}}
  thread_local int prohibit_local2;
}

void bar() { thread_local int allow_local; }

void usage() {
  // expected-note@+1 {{called by}}
  foo();
  // expected-error@+1 {{thread-local storage is not supported for the current target}}
  (void)prohobit_ns_scope;
  // expected-error@+1 {{thread-local storage is not supported for the current target}}
  (void)prohobit_ns_scope2;
  // expected-error@+1 {{thread-local storage is not supported for the current target}}
  (void)S::prohibit_static_member;
  // expected-error@+1 {{thread-local storage is not supported for the current target}}
  (void)S::prohibit_static_member2;
}

int main() {
  // expected-note@+2 2{{called by}}
#pragma omp target
  usage();
  return 0;
}
