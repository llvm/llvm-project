#define f32divf64x __hide_f32divf64x
#define f32divf128 __hide_f32divf128
#include <float128_private.h>
#undef f32divf64x
#undef f32divf128
#include "../ldbl-128/s_fdivl.c"
