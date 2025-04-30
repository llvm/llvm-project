#define f32mulf64x __hide_f32mulf64x
#define f32mulf128 __hide_f32mulf128
#include <float128_private.h>
#undef f32mulf64x
#undef f32mulf128
#include "../ldbl-128/s_fmull.c"
