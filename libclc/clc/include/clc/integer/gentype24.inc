#include <clc/clcfunc.h>
#include <clc/clctypes.h>

#define __CLC_GENSIZE 32
#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE int

#define __CLC_GENTYPE int
#define __CLC_U_GENTYPE uint
#define __CLC_S_GENTYPE int
#define __CLC_SCALAR 1
#define __CLC_VECSIZE
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int2
#define __CLC_U_GENTYPE uint2
#define __CLC_S_GENTYPE int2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int3
#define __CLC_U_GENTYPE uint3
#define __CLC_S_GENTYPE int3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int4
#define __CLC_U_GENTYPE uint4
#define __CLC_S_GENTYPE int4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int8
#define __CLC_U_GENTYPE uint8
#define __CLC_S_GENTYPE int8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE int16
#define __CLC_U_GENTYPE uint16
#define __CLC_S_GENTYPE int16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#undef __CLC_SCALAR_GENTYPE
#define __CLC_SCALAR_GENTYPE uint

#define __CLC_GENTYPE uint
#define __CLC_U_GENTYPE uint
#define __CLC_S_GENTYPE int
#define __CLC_SCALAR 1
#define __CLC_VECSIZE
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_SCALAR
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint2
#define __CLC_U_GENTYPE uint2
#define __CLC_S_GENTYPE int2
#define __CLC_VECSIZE 2
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint3
#define __CLC_U_GENTYPE uint3
#define __CLC_S_GENTYPE int3
#define __CLC_VECSIZE 3
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint4
#define __CLC_U_GENTYPE uint4
#define __CLC_S_GENTYPE int4
#define __CLC_VECSIZE 4
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint8
#define __CLC_U_GENTYPE uint8
#define __CLC_S_GENTYPE int8
#define __CLC_VECSIZE 8
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#define __CLC_GENTYPE uint16
#define __CLC_U_GENTYPE uint16
#define __CLC_S_GENTYPE int16
#define __CLC_VECSIZE 16
#include __CLC_BODY
#undef __CLC_VECSIZE
#undef __CLC_GENTYPE
#undef __CLC_U_GENTYPE
#undef __CLC_S_GENTYPE

#undef __CLC_GENSIZE
#undef __CLC_SCALAR_GENTYPE
#undef __CLC_BODY
