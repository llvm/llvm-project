// RUN: %clang_cc1 -triple spir64 -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -verify -fsyntax-only %s

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc(); // expected-note {{called by 'kernel}}
}

void host_ok(void) {
  __bf16 A;
}

int main()
{  host_ok();
  __bf16 var; // expected-note {{'var' defined here}}
  kernel<class variables>([=]() {
    (void)var; // expected-error {{'var' requires 16 bit size '__bf16' type support, but target 'spir64' does not support it}}
    int B = sizeof(__bf16);
  });

  return 0;
}

