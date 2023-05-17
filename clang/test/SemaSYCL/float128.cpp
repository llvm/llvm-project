// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device -fsyntax-only %s

typedef __float128 BIGTY;

template <class T>
class Z {
public:
  // expected-note@+1 {{'field' defined here}}
  T field;
  // expected-note@+1 2{{'field1' defined here}}
  __float128 field1;
  using BIGTYPE = __float128;
  // expected-note@+1 {{'bigfield' defined here}}
  BIGTYPE bigfield;
};

void host_ok(void) {
  __float128 A;
  int B = sizeof(__float128);
  Z<__float128> C;
  C.field1 = A;
}

long double ld_func(long double arg);

void usage() {
  // expected-note@+1 3{{'A' defined here}}
  __float128 A;
  Z<__float128> C;
  // expected-error@+3 2{{expression requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
  // expected-error@+2 {{'A' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'field1' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
  C.field1 = A;
  // expected-error@+2 {{expression requires 128 bit size 'BIGTYPE' (aka '__float128') type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'bigfield' requires 128 bit size 'BIGTYPE' (aka '__float128') type support, but target 'spir64' does not support it}}
  C.bigfield += 1.0;

  // expected-error@+1 {{'A' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
  auto foo1 = [=]() {
    __float128 AA;
    // expected-note@+2 {{'BB' defined here}}
    // expected-error@+1 {{'A' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    auto BB = A;
    // expected-error@+2 {{expression requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    // expected-error@+1 {{'BB' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    BB += 1;

    float F1 = 0.1f;
    float F2 = 0.1f;
    // expected-error@+1 3{{expression requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    float F3 = ((__float128)F1 * (__float128)F2) / 2.0f;

    // assume that long double is supported
    float F4 = ld_func(F3);
  };

  // expected-note@+1 {{called by 'usage'}}
  foo1();
}

template <typename t>
void foo2(){};

// expected-note@+3 {{'P' defined here}}
// expected-error@+2 {{'P' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
// expected-note@+1 2{{'foo' defined here}}
__float128 foo(__float128 P) { return P; }

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  // expected-note@+1 5{{called by 'kernel}}
  kernelFunc();
}

int main() {
  // expected-note@+1 {{'CapturedToDevice' defined here}}
  __float128 CapturedToDevice = 1;
  host_ok();
  kernel<class variables>([=]() {
    decltype(CapturedToDevice) D;
    // expected-error@+1 {{'CapturedToDevice' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    auto C = CapturedToDevice;
    Z<__float128> S;
    // expected-error@+2 {{expression requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    // expected-error@+1 {{'field1' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    S.field1 += 1;
    // expected-error@+2 {{expression requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    // expected-error@+1 {{'field' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    S.field = 1;
  });

  kernel<class functions>([=]() {
    // expected-note@+1 2{{called by 'operator()'}}
    usage();
    // expected-note@+1 {{'BBBB' defined here}}
    BIGTY BBBB;
    // expected-note@+3 {{called by 'operator()'}}
    // expected-error@+2 {{'BBBB' requires 128 bit size 'BIGTY' (aka '__float128') type support, but target 'spir64' does not support it}}
    // expected-error@+1 2{{'foo' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
    auto A = foo(BBBB);
  });

  kernel<class ok>([=]() {
    Z<__float128> S;
    foo2<__float128>();
    auto A = sizeof(CapturedToDevice);
  });

  return 0;
}
