#define f32xsubf64x __hide_f32xsubf64x
#define f32xsubf128 __hide_f32xsubf128
#define f64subf64x __hide_f64subf64x
#define f64subf128 __hide_f64subf128
#include <float128_private.h>
#undef f32xsubf64x
#undef f32xsubf128
#undef f64subf64x
#undef f64subf128
#include "../ldbl-128/s_dsubl.c"
