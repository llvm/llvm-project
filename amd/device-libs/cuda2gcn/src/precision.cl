/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"

#define ATTR

#define FUNC1F(root) \
  ATTR float __nv_fast_##root##f(float x) { return __ocml_##root##_f32(x); }
#define FUNC1(root) FUNC1F(root)

#define FUNC2F(root) \
  ATTR float __nv_fast_##root##f(float x, float y) { return __ocml_##root##_f32(x, y); }
#define FUNC2(root) FUNC2F(root)

#define FUNC3F(root) \
  ATTR float __nv_fast_##root##f(float x, float y, float z) { return __ocml_##root##_f32(x, y, z); }
#define FUNC3(root) FUNC3F(root)

//-------- T __nv_fast_cosf
FUNC1(cos)

//-------- T __nv_fast_exp10f
FUNC1(exp10)

//-------- T __nv_fast_expf
FUNC1(exp)

//-------- T __nv_fast_log10f
FUNC1(log10)

//-------- T __nv_fast_log2f
FUNC1(log2)

//-------- T __nv_fast_logf
FUNC1(log)

//-------- T __nv_fast_powf
FUNC2(pow)

//-------- T __nv_fast_sinf
FUNC1(sin)

//-------- T __nv_fast_tanf
FUNC1(tan)

//-------- T __nv_fast_fdividef
ATTR float __nv_fast_fdividef(float x, float y) { return native_divide(x, y); }

//-------- T __nv_fast_sincosf
ATTR void __nv_fast_sincosf(float x, __private float * sptr, __private float *cptr) { (*sptr)=__ocml_sincos_f32(x, cptr); }

