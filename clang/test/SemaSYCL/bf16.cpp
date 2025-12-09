// RUN: %clang_cc1 -triple spir64 -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -verify -fsyntax-only %s
// expected-no-diagnostics

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

void host_ok(void) {
  __bf16 A;
}

int main()
{  host_ok();
  __bf16 var;
  kernel<class variables>([=]() {
    (void)var;
    int B = sizeof(__bf16);
  });

  return 0;
}

