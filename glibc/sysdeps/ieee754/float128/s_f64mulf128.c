#define f32xmulf64x __hide_f32xmulf64x
#define f32xmulf128 __hide_f32xmulf128
#define f64mulf64x __hide_f64mulf64x
#define f64mulf128 __hide_f64mulf128
#include <float128_private.h>
#undef f32xmulf64x
#undef f32xmulf128
#undef f64mulf64x
#undef f64mulf128
#include "../ldbl-128/s_dmull.c"
