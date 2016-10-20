#include <clc/clc.h>

#include "../clcmacro.h"
#include "math.h"

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define __CLC_BODY <lgamma_r.inc>
#include <clc/math/gentype.inc>
