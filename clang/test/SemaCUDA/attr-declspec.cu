// Test the __declspec spellings of CUDA attributes.
//
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify %s
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -fcuda-is-device -verify %s
// Now pretend that we're compiling a C file. There should be warnings.
// RUN: %clang_cc1 -DEXPECT_WARNINGS -fms-extensions -fsyntax-only -verify -x c %s

#if defined(EXPECT_WARNINGS)
// expected-warning@+12 {{'__device__' attribute ignored}}
// expected-warning@+12 {{'__global__' attribute ignored}}
// expected-warning@+12 {{'__constant__' attribute ignored}}
// expected-warning@+12 {{'__shared__' attribute ignored}}
// expected-warning@+12 {{'__host__' attribute ignored}}
//
// (Currently we don't for the other attributes. They are implemented with
// IgnoredAttr, which is ignored irrespective of any LangOpts.)
#else
// expected-no-diagnostics
#endif

__declspec(__device__) void f_device();
__declspec(__global__) void f_global();
__declspec(__constant__) int* g_constant;
__declspec(__shared__) float *g_shared;
__declspec(__host__) void f_host();
__declspec(__device_builtin__) void f_device_builtin();
typedef __declspec(__device_builtin__) const void *t_device_builtin;
enum __declspec(__device_builtin__) e_device_builtin {E};
__declspec(__device_builtin__) int v_device_builtin;
__declspec(__cudart_builtin__) void f_cudart_builtin();
__declspec(__device_builtin_surface_type__) unsigned long long surface_var;
__declspec(__device_builtin_texture_type__) unsigned long long texture_var;

// Note that there's no __declspec spelling of nv_weak.
