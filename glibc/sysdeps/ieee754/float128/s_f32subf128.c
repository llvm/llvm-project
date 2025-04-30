#define f32subf64x __hide_f32subf64x
#define f32subf128 __hide_f32subf128
#include <float128_private.h>
#undef f32subf64x
#undef f32subf128
#include "../ldbl-128/s_fsubl.c"
