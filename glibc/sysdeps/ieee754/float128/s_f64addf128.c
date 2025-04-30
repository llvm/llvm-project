#define f32xaddf64x __hide_f32xaddf64x
#define f32xaddf128 __hide_f32xaddf128
#define f64addf64x __hide_f64addf64x
#define f64addf128 __hide_f64addf128
#include <float128_private.h>
#undef f32xaddf64x
#undef f32xaddf128
#undef f64addf64x
#undef f64addf128
#include "../ldbl-128/s_daddl.c"
