#include <clc/clc.h>

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define __CLC_BODY <prefetch.inc>
#include <clc/async/gentype.inc>
#undef __CLC_BODY
