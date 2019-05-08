// RUN: %clang_cc1 -x c++ -fsycl-is-device -std=c++11 -fsyntax-only -verify -pedantic %s

// Test for Intel FPGA loop attributes applied not to a loop
void foo() {
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while or do statements}}
  [[intelfpga::ivdep]] int a[10];
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while or do statements}}
  [[intelfpga::ivdep(2)]] int b[10];
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while or do statements}}
  [[intelfpga::ii(2)]] int c[10];
  // expected-error@+1 {{intelfpga loop attributes must be applied to for, while or do statements}}
  [[intelfpga::max_concurrency(2)]] int d[10];
}

// Test for incorrect number of arguments for Intel FPGA loop attributes
void boo() {
  int a[10];
  // expected-error@+1 {{'ivdep' attribute takes no more than 1 argument}}
  [[intelfpga::ivdep(2,2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'ii' attribute takes at least 1 argument}}
  [[intelfpga::ii]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'ii' attribute takes no more than 1 argument}}
  [[intelfpga::ii(2,2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute takes at least 1 argument}}
  [[intelfpga::max_concurrency]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute takes no more than 1 argument}}
  [[intelfpga::max_concurrency(2,2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

// Test for incorrect argument value for Intel FPGA loop attributes
void goo() {
  int a[10];
  // expected-error@+1 {{'ivdep' attribute requires a positive integral compile time constant expression}}
  [[intelfpga::ivdep(0)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'ii' attribute requires a positive integral compile time constant expression}}
  [[intelfpga::ii(0)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute requires a positive integral compile time constant expression}}
  [[intelfpga::max_concurrency(0)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'ivdep' attribute requires an integer constant}}
  [[intelfpga::ivdep("test123")]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'ii' attribute requires an integer constant}}
  [[intelfpga::ii("test123")]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // expected-error@+1 {{'max_concurrency' attribute requires an integer constant}}
  [[intelfpga::max_concurrency("test123")]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

// Test for Intel FPGA loop attributes dublication
void zoo() {
  int a[10];
  // no diagnostics are expected
  [[intelfpga::ivdep]]
  [[intelfpga::max_concurrency(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'ivdep'}}
  [[intelfpga::ivdep]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'ivdep'}}
  [[intelfpga::ivdep(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ivdep(2)]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'ivdep'}}
  [[intelfpga::ivdep(4)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::max_concurrency(2)]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'max_concurrency'}}
  [[intelfpga::max_concurrency(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ii(2)]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'ii'}}
  [[intelfpga::ii(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  [[intelfpga::ii(2)]]
  // expected-error@-1 {{duplicate Intel FPGA loop attribute 'ii'}}
  [[intelfpga::max_concurrency(2)]]
  [[intelfpga::ii(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
    boo();
    goo();
    zoo();
  });
  return 0;
}
