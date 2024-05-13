// RUN: %clang_cc1 -triple spir64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:    -fsycl-is-device -verify -fsyntax-only %s

typedef __uint128_t BIGTY;

template <class T>
class Z {
public:
  // expected-note@+1 {{'field' defined here}}
  T field;
  // expected-note@+1 2{{'field1' defined here}}
  __int128 field1;
  using BIGTYPE = __int128;
  // expected-note@+1 {{'bigfield' defined here}}
  BIGTYPE bigfield;
};

void host_ok(void) {
  __int128 A;
  int B = sizeof(__int128);
  Z<__int128> C;
  C.field1 = A;
}

void usage() {
  // expected-note@+1 3{{'A' defined here}}
  __int128 A;
  Z<__int128> C;
  // expected-error@+3 2{{expression requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  // expected-error@+2 {{'A' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'field1' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  C.field1 = A;
  // expected-error@+2 {{expression requires 128 bit size 'BIGTYPE' (aka '__int128') type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'bigfield' requires 128 bit size 'BIGTYPE' (aka '__int128') type support, but target 'spir64' does not support it}}
  C.bigfield += 1.0;

  // expected-error@+1 {{'A' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  auto foo1 = [=]() {
    __int128 AA;
    // expected-note@+2 {{'BB' defined here}}
    // expected-error@+1 {{'A' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    auto BB = A;
    // expected-error@+2 {{expression requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    // expected-error@+1 {{'BB' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    BB += 1;
  };

  // expected-note@+1 {{called by 'usage'}}
  foo1();
}

template <typename t>
void foo2(){};

// expected-note@+3 {{'P' defined here}}
// expected-error@+2 {{'P' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
// expected-note@+1 2{{'foo' defined here}}
__int128 foo(__int128 P) { return P; }

void foobar() {
  // expected-note@+1 {{'operator __int128' defined here}}
  struct X { operator  __int128() const; } x;
  bool a = false;
  // expected-error@+2 2{{expression requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'operator __int128' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  a = x == __int128(0);
}

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  // expected-note@+1 6{{called by 'kernel}}
  kernelFunc();
}

int main() {
  // expected-note@+1 {{'CapturedToDevice' defined here}}
  __int128 CapturedToDevice = 1;
  host_ok();
  kernel<class variables>([=]() {
    decltype(CapturedToDevice) D;
    // expected-error@+1 {{'CapturedToDevice' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    auto C = CapturedToDevice;
    Z<__int128> S;
    // expected-error@+2 {{expression requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    // expected-error@+1 {{'field1' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    S.field1 += 1;
    // expected-error@+2 {{expression requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    // expected-error@+1 {{'field' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    S.field = 1;
  });

  kernel<class functions>([=]() {
    // expected-note@+1 2{{called by 'operator()'}}
    usage();
    // expected-note@+1 {{'BBBB' defined here}}
    BIGTY BBBB;
    // expected-error@+3 {{'BBBB' requires 128 bit size 'BIGTY' (aka 'unsigned __int128') type support, but target 'spir64' does not support it}}
    // expected-error@+2 2{{'foo' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
    // expected-note@+1 1{{called by 'operator()'}}
    auto A = foo(BBBB);
    // expected-note@+1 {{called by 'operator()'}}
    foobar();
  });

  kernel<class ok>([=]() {
    Z<__int128> S;
    foo2<__int128>();
    auto A = sizeof(CapturedToDevice);
  });

  return 0;
}

// no error expected
BIGTY zoo(BIGTY h) {
  h = 1;
  return h;
}

namespace PR12964 {
  struct X { operator  __int128() const; } x;
  bool a = x == __int128(0);
}

