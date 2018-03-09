#include <clc/clc.h>

#include "math.h"

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define __CLC_BODY <frexp.inc>
#include <clc/math/gentype.inc>
