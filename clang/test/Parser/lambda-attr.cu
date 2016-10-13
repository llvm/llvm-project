// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcuda-is-device -verify %s

__attribute__((device)) void device_fn() {}
__attribute__((host, device)) void hd_fn() {}

__attribute__((device)) void device_attr() {
  ([]() __attribute__((device)) { device_fn(); })();
  // expected-warning@-1 {{nvcc does not allow '__device__' to appear after '()' in lambdas}}
  ([] __attribute__((device)) () { device_fn(); })();
  ([] __attribute__((device)) { device_fn(); })();

  ([&]() __attribute__((device)){ device_fn(); })();
  // expected-warning@-1 {{nvcc does not allow '__device__' to appear after '()' in lambdas}}
  ([&] __attribute__((device)) () { device_fn(); })();
  ([&] __attribute__((device)) { device_fn(); })();

  ([&](int) __attribute__((device)){ device_fn(); })(0);
  // expected-warning@-1 {{nvcc does not allow '__device__' to appear after '()' in lambdas}}
  ([&] __attribute__((device)) (int) { device_fn(); })(0);
}

__attribute__((host)) __attribute__((device)) void host_device_attrs() {
  ([]() __attribute__((host)) __attribute__((device)){ hd_fn(); })();
  // expected-warning@-1 {{nvcc does not allow '__host__' to appear after '()' in lambdas}}
  // expected-warning@-2 {{nvcc does not allow '__device__' to appear after '()' in lambdas}}
  ([] __attribute__((host)) __attribute__((device)) () { hd_fn(); })();
  ([] __attribute__((host)) __attribute__((device)) { hd_fn(); })();

  ([&]() __attribute__((host)) __attribute__((device)){ hd_fn(); })();
  // expected-warning@-1 {{nvcc does not allow '__host__' to appear after '()' in lambdas}}
  // expected-warning@-2 {{nvcc does not allow '__device__' to appear after '()' in lambdas}}
  ([&] __attribute__((host)) __attribute__((device)) () { hd_fn(); })();
  ([&] __attribute__((host)) __attribute__((device)) { hd_fn(); })();

  ([&](int) __attribute__((host)) __attribute__((device)){ hd_fn(); })(0);
  // expected-warning@-1 {{nvcc does not allow '__host__' to appear after '()' in lambdas}}
  // expected-warning@-2 {{nvcc does not allow '__device__' to appear after '()' in lambdas}}
  ([&] __attribute__((host)) __attribute__((device)) (int) { hd_fn(); })(0);
}

// TODO: Add tests for __attribute__((global)) once we support global lambdas.
