// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -I%S/Inputs/System/usr/include -verify
// RUN: %clang_cc1 -fsyntax-only -std=c99 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -I%S/Inputs/System/usr/include -verify
// RUN: %clang_cc1 -fsyntax-only -fmodules -fmodule-map-file=%resource_dir/module.modulemap -fmodules-cache-path=%t %s -I%S/Inputs/System/usr/include -verify
// expected-no-diagnostics

#ifdef __SSE__
@import _Builtin_intrinsics.intel.sse;
#endif

#ifdef __AVX2__
@import _Builtin_intrinsics.intel.avx2;
#endif
