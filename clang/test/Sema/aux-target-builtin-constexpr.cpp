// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device \
// RUN:   -triple aarch64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -aux-target-feature +avx512f -std=c++17 -fsyntax-only -verify %s

// expected-no-diagnostics

// When compiling for a device target (here aarch64) with an auxiliary x86
// target, x86 builtins are registered as auxiliary builtins whose IDs are
// shifted past the primary target's builtins. The constant evaluator must
// translate such aux IDs back to their canonical X86::BI* values before
// dispatching the target-specific constexpr switches. Without that
// normalization the shifted ID misses its intended case and the call fails to
// fold ("not an integral constant expression").

typedef short __v8hi __attribute__((__vector_size__(16)));

constexpr __v8hi V = {0, 10, 20, 30, 40, 50, 60, 70};

static_assert(__builtin_ia32_vec_ext_v8hi(V, 3) == 30);
static_assert(__builtin_ia32_vec_ext_v8hi(V, 7) == 70);
