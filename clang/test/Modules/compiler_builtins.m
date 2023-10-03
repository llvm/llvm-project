// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -internal-isystem %S/Inputs/System/usr/include -verify
// RUN: %clang_cc1 -fsyntax-only -std=c99 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -internal-isystem %S/Inputs/System/usr/include -verify
// RUN: %clang_cc1 -fsyntax-only -fmodules -fmodule-map-file=%resource_dir/module.modulemap -fmodules-cache-path=%t %s -I%S/Inputs/System/usr/include -DNO_SYSTEM_MODULES -verify
// expected-no-diagnostics

#ifdef __SSE__
@import _Builtin_intrinsics.intel.sse;
#endif

#ifdef __AVX2__
@import _Builtin_intrinsics.intel.avx2;
#endif

#ifndef NO_SYSTEM_MODULES
@import _Builtin_float;
@import _Builtin_inttypes;
@import _Builtin_iso646;
@import _Builtin_limits;
@import _Builtin_stdalign;
@import _Builtin_stdarg;
@import _Builtin_stdatomic;
@import _Builtin_stdbool;
@import _Builtin_stddef;
@import _Builtin_stdint;
@import _Builtin_stdnoreturn;
@import _Builtin_tgmath;
@import _Builtin_unwind;
#endif
