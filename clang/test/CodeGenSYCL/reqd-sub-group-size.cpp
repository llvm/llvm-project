// RUN: %clang_cc1 -std=c++11 -disable-llvm-passes -fsycl-is-device -emit-llvm -o - %s | FileCheck %s

class Functor16 {
public:
  [[cl::intel_reqd_sub_group_size(16)]] void operator()() {}
};

[[cl::intel_reqd_sub_group_size(8)]] void foo() {}

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
}

// CHECK: define spir_kernel void @{{.*}}kernel_name1() {{.*}} !intel_reqd_sub_group_size ![[SGSIZE16:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name2() {{.*}} !intel_reqd_sub_group_size ![[SGSIZE8:[0-9]+]]
// CHECK: ![[SGSIZE16]] = !{i32 16}
// CHECK: ![[SGSIZE8]] = !{i32 8}

