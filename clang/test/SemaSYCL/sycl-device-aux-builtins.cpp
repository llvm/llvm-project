// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

// Check that host target builtins are diagnosed only in code that is
// reachable from device code.

typedef short v8hi __attribute__((__vector_size__(16)));
typedef int v4si __attribute__((__vector_size__(16)));

[[clang::sycl_external]] void direct() {
  // expected-error@+1 {{builtin is not supported on this target}}
  __builtin_ia32_pause();
}

inline v4si madd(v8hi a, v8hi b) {
  // expected-error@+1 {{builtin is not supported on this target}}
  return __builtin_ia32_pmaddwd128(a, b);
}

// expected-note@+1 {{called by 'indirect'}}
[[clang::sycl_external]] v4si indirect(v8hi a, v8hi b) { return madd(a, b); }

// Not reachable from device code.
inline void unused() { __builtin_ia32_pause(); }
void host_only() { __builtin_ia32_pause(); }

// Not evaluated.
[[clang::sycl_external]] int unevaluated() {
  return sizeof(__builtin_ia32_pmaddwd128(v8hi{}, v8hi{}));
}
