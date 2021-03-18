// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -verify -fsyntax-only %s

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

template <typename name, typename Func>
__attribute__((sycl_kernel))
// expected-note@+2 2{{called by}}
void
kernel_single_task(Func kernelFunc) { kernelFunc(); }

int main() {
  // expected-note@+1 2{{called by}}
  kernel_single_task<class fake_kernel>([]() { usage(); });
  return 0;
}
