// RUN: %clang_cc1 -triple spir64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:    -fsycl-is-device -verify -fsyntax-only %s

// Verify that __int256 is rejected in SYCL device code on targets
// that don't support it, mirroring the __int128 restriction test.

typedef __uint256_t BIGTY;

template <class T>
class Z {
public:
  // expected-note@+1 {{'field' defined here}}
  T field;
  // expected-note@+1 2{{'field1' defined here}}
  __int256 field1;
};

void host_ok(void) {
  __int256 A;
  int B = sizeof(__int256);
  Z<__int256> C;
  C.field1 = A;
}

void usage() {
  // expected-note@+1 {{'A' defined here}}
  __int256 A;
  Z<__int256> C;
  // expected-error@+3 2{{expression requires 256 bit size '__int256' type support, but target 'spir64' does not support it}}
  // expected-error@+2 {{'A' requires 256 bit size '__int256' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'field1' requires 256 bit size '__int256' type support, but target 'spir64' does not support it}}
  C.field1 = A;
}

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  // expected-note@+1 2{{called by 'kernel}}
  kernelFunc();
}

int main() {
  // expected-note@+1 {{'CapturedToDevice' defined here}}
  __int256 CapturedToDevice = 1;
  host_ok();
  kernel<class variables>([=]() {
    // expected-error@+1 {{'CapturedToDevice' requires 256 bit size '__int256' type support, but target 'spir64' does not support it}}
    auto C = CapturedToDevice;
    Z<__int256> S;
    // expected-error@+2 {{expression requires 256 bit size '__int256' type support, but target 'spir64' does not support it}}
    // expected-error@+1 {{'field1' requires 256 bit size '__int256' type support, but target 'spir64' does not support it}}
    S.field1 += 1;
    // expected-error@+2 {{expression requires 256 bit size '__int256' type support, but target 'spir64' does not support it}}
    // expected-error@+1 {{'field' requires 256 bit size '__int256' type support, but target 'spir64' does not support it}}
    S.field = 1;
  });

  kernel<class functions>([=]() {
    // expected-note@+1 {{called by 'operator()'}}
    usage();
  });

  kernel<class ok>([=]() {
    Z<__int256> S;
    auto A = sizeof(CapturedToDevice);
  });

  return 0;
}

// no error expected for host-side functions
BIGTY zoo(BIGTY h) {
  h = 1;
  return h;
}
