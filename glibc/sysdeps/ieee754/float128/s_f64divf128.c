#define f32xdivf64x __hide_f32xdivf64x
#define f32xdivf128 __hide_f32xdivf128
#define f64divf64x __hide_f64divf64x
#define f64divf128 __hide_f64divf128
#include <float128_private.h>
#undef f32xdivf64x
#undef f32xdivf128
#undef f64divf64x
#undef f64divf128
#include "../ldbl-128/s_ddivl.c"
