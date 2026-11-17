// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -target-feature +sse2 \
// RUN:   -fsyntax-only -verify %s

// Check that functions requiring host target features or a host CPU are
// diagnosed only when called from code that is reachable from device code.

__attribute__((target("sse2"))) int sse2_add(int x) { return x; }
__attribute__((target("arch=haswell"))) int haswell_add(int x) { return x; }
__attribute__((target("sse2"))) constexpr int sse2_const(int x) { return x; }

[[clang::sycl_external]] int direct(int x) {
  // expected-error@+2 {{SYCL device code cannot call 'sse2_add', which requires the 'sse2' host feature}}
  // expected-error@+1 {{SYCL device code cannot call 'haswell_add', which requires a 'haswell' host}}
  return sse2_add(x) + haswell_add(x);
}

inline int helper(int x) {
  // expected-error@+1 {{SYCL device code cannot call 'sse2_add', which requires the 'sse2' host feature}}
  return sse2_add(x);
}

// expected-note@+1 {{called by 'indirect'}}
[[clang::sycl_external]] int indirect(int x) { return helper(x); }

// Not reachable from device code.
int host_only(int x) { return sse2_add(x) + haswell_add(x); }

// Not evaluated.
[[clang::sycl_external]] int unevaluated(int x) { return sizeof(sse2_add(x)); }

// Constant evaluated.
[[clang::sycl_external]] void constant_evaluated() {
  static_assert(sse2_const(1) == 1);
}
