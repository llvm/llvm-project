// RUN: rm -rf %t
// RUN: %clang -fsyntax-only -fmodules -fmodules-cache-path=%t -D__need_wint_t %s -I%S/Inputs/System/usr/include -Xclang -verify
// RUN: %clang -fsyntax-only -std=c99 -fmodules -fmodules-cache-path=%t -D__need_wint_t %s -I%S/Inputs/System/usr/include -Xclang -verify
// expected-no-diagnostics

#ifdef __SSE__
@import _Builtin_intrinsics.intel.sse;
#endif

#ifdef __AVX2__
@import _Builtin_intrinsics.intel.avx2;
#endif
