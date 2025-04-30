#define f32addf64x __hide_f32addf64x
#define f32addf128 __hide_f32addf128
#include <float128_private.h>
#undef f32addf64x
#undef f32addf128
#include "../ldbl-128/s_faddl.c"
