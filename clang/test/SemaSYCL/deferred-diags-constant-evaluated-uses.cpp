// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu -std=c++20 -fsycl-is-device -fsyntax-only -verify %s

// Functions used only in constant evaluation are not diagnosed.

int vfn(int, ...);

constexpr int cf(bool b) {
  if (b)
    return vfn(1, 2);
  return 1;
}
consteval int ce(bool b) {
  if (b)
    return vfn(1, 2);
  return 2;
}
template <int N> struct S { static constexpr int value = N; };

[[clang::sycl_external]] int constexpr_var_init() {
  constexpr int v = cf(false);
  return v;
}
[[clang::sycl_external]] int const_var_init() {
  const int v = cf(false);
  return v;
}
[[clang::sycl_external]] int immediate_invocation() { return ce(false); }
[[clang::sycl_external]] int case_label(int x) {
  switch (x) {
  case cf(false):
    return 1;
  }
  return 0;
}
[[clang::sycl_external]] int template_arg() { return S<cf(false)>::value; }

constexpr int cr(bool b) {
  if (b)
    return vfn(1, 2); // expected-error {{SYCL device code does not support variadic functions}}
  return 3;
}
[[clang::sycl_external]] int runtime_call(bool b) {
  return cr(b); // expected-note {{called by 'runtime_call'}}
}
