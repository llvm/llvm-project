// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -x c++ -emit-llvm-only %s
// expected-no-diagnostics
// Should never fail, since the type is never used in kernel code.

struct Base {
  virtual void f(){}
};

struct Inherit : Base {
  virtual void f() override {}
};

void always_uses() {
  Inherit u;
}

void usage() {
}


template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}
int main() {
  always_uses();
  kernel_single_task<class fake_kernel>([]() { usage(); });
  return 0;
}

