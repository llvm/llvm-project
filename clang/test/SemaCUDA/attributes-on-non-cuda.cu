// Tests that CUDA attributes are warnings when compiling C files, but not when
// compiling CUDA files.
//
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fcuda-is-device -verify %s
// Now pretend that we're compiling a C file. There should be warnings.
// RUN: %clang_cc1 -DEXPECT_WARNINGS -fsyntax-only -verify -x c %s

#if defined(EXPECT_WARNINGS)
// expected-warning@+15 {{'device' attribute ignored}}
// expected-warning@+15 {{'global' attribute ignored}}
// expected-warning@+15 {{'constant' attribute ignored}}
// expected-warning@+15 {{'shared' attribute ignored}}
// expected-warning@+15 {{'host' attribute ignored}}
// expected-warning@+21 {{'device_builtin_surface_type' attribute ignored}}
// expected-warning@+21 {{'device_builtin_texture_type' attribute ignored}}
//
// NOTE: IgnoredAttr in clang which is used for the rest of
// attributes ignores LangOpts, so there are no warnings.
#else
// expected-warning@+15 {{'device_builtin_surface_type' attribute only applies to classes}}
// expected-warning@+15 {{'device_builtin_texture_type' attribute only applies to classes}}
#endif

__attribute__((device)) void f_device();
__attribute__((global)) void f_global();
__attribute__((constant)) int* g_constant;
__attribute__((shared)) float *g_shared;
__attribute__((host)) void f_host();
__attribute__((device_builtin)) void f_device_builtin();
typedef __attribute__((device_builtin)) const void *t_device_builtin;
enum __attribute__((device_builtin)) e_device_builtin {E};
__attribute__((device_builtin)) int v_device_builtin;
__attribute__((cudart_builtin)) void f_cudart_builtin();
__attribute__((nv_weak)) void f_nv_weak();
__attribute__((device_builtin_surface_type)) unsigned long long surface_var;
__attribute__((device_builtin_texture_type)) unsigned long long texture_var;
