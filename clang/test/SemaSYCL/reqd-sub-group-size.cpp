// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -DTRIGGER_ERROR %s
// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s

[[cl::intel_reqd_sub_group_size(4)]] void foo() {} // expected-note {{conflicting attribute is here}}
// expected-note@-1 {{conflicting attribute is here}}
[[cl::intel_reqd_sub_group_size(32)]] void baz() {} // expected-note {{conflicting attribute is here}}

class Functor16 {
public:
  [[cl::intel_reqd_sub_group_size(16)]] void operator()() {}
};

class Functor8 { // expected-error {{conflicting attributes applied to a SYCL kernel}}
public:
  [[cl::intel_reqd_sub_group_size(8)]] void operator()() { // expected-note {{conflicting attribute is here}}
    foo();
  }
};

class Functor {
public:
  void operator()() {
    foo();
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor16 f16;
  kernel<class kernel_name1>(f16);

  Functor f;
  kernel<class kernel_name2>(f);

#ifdef TRIGGER_ERROR
  Functor8 f8;
  kernel<class kernel_name3>(f8);

  kernel<class kernel_name4>([]() { // expected-error {{conflicting attributes applied to a SYCL kernel}}
    foo();
    baz();
  });
#endif
}

// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name1
// CHECK: IntelReqdSubGroupSizeAttr {{.*}} 16
// CHECK: FunctionDecl {{.*}} {{.*}}kernel_name2
// CHECK: IntelReqdSubGroupSizeAttr {{.*}} 4
