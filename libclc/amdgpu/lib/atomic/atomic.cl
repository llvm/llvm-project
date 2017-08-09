#include <clc/clc.h>

#define ATOMIC_FUNC_DEFINE(RET_SIGN, ARG_SIGN, TYPE, CL_FUNCTION, CLC_FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
_CLC_OVERLOAD _CLC_DEF RET_SIGN TYPE CL_FUNCTION (volatile CL_ADDRSPACE RET_SIGN TYPE *p, RET_SIGN TYPE val) { \
	return (RET_SIGN TYPE)__clc_##CLC_FUNCTION##_addr##LLVM_ADDRSPACE((volatile CL_ADDRSPACE ARG_SIGN TYPE*)p, (ARG_SIGN TYPE)val); \
}

/* For atomic functions that don't need different bitcode dependending on argument signedness */
#define ATOMIC_FUNC_SIGN(TYPE, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
	_CLC_DECL signed TYPE __clc_##FUNCTION##_addr##LLVM_ADDRSPACE(volatile CL_ADDRSPACE signed TYPE*, signed TYPE); \
	ATOMIC_FUNC_DEFINE(signed, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
	ATOMIC_FUNC_DEFINE(unsigned, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE)

#define ATOMIC_FUNC_ADDRSPACE(TYPE, FUNCTION) \
	ATOMIC_FUNC_SIGN(TYPE, FUNCTION, global, 1) \
	ATOMIC_FUNC_SIGN(TYPE, FUNCTION, local, 3)

#define ATOMIC_FUNC(FUNCTION) \
	ATOMIC_FUNC_ADDRSPACE(int, FUNCTION)

#define ATOMIC_FUNC_DEFINE_3_ARG(RET_SIGN, ARG_SIGN, TYPE, CL_FUNCTION, CLC_FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
_CLC_OVERLOAD _CLC_DEF RET_SIGN TYPE CL_FUNCTION (volatile CL_ADDRSPACE RET_SIGN TYPE *p, RET_SIGN TYPE cmp, RET_SIGN TYPE val) { \
	return (RET_SIGN TYPE)__clc_##CLC_FUNCTION##_addr##LLVM_ADDRSPACE((volatile CL_ADDRSPACE ARG_SIGN TYPE*)p, (ARG_SIGN TYPE)cmp, (ARG_SIGN TYPE)val); \
}

/* For atomic functions that don't need different bitcode dependending on argument signedness */
#define ATOMIC_FUNC_SIGN_3_ARG(TYPE, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
	_CLC_DECL signed TYPE __clc_##FUNCTION##_addr##LLVM_ADDRSPACE(volatile CL_ADDRSPACE signed TYPE*, signed TYPE, signed TYPE); \
	ATOMIC_FUNC_DEFINE_3_ARG(signed, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE) \
	ATOMIC_FUNC_DEFINE_3_ARG(unsigned, signed, TYPE, FUNCTION, FUNCTION, CL_ADDRSPACE, LLVM_ADDRSPACE)

#define ATOMIC_FUNC_ADDRSPACE_3_ARG(TYPE, FUNCTION) \
	ATOMIC_FUNC_SIGN_3_ARG(TYPE, FUNCTION, global, 1) \
	ATOMIC_FUNC_SIGN_3_ARG(TYPE, FUNCTION, local, 3)

#define ATOMIC_FUNC_3_ARG(FUNCTION) \
	ATOMIC_FUNC_ADDRSPACE_3_ARG(int, FUNCTION)

ATOMIC_FUNC(atomic_add)
ATOMIC_FUNC(atomic_and)
ATOMIC_FUNC(atomic_or)
ATOMIC_FUNC(atomic_sub)
ATOMIC_FUNC(atomic_xchg)
ATOMIC_FUNC(atomic_xor)
ATOMIC_FUNC_3_ARG(atomic_cmpxchg)

_CLC_DECL signed int __clc_atomic_max_addr1(volatile global signed int*, signed int);
_CLC_DECL signed int __clc_atomic_max_addr3(volatile local signed int*, signed int);
_CLC_DECL uint __clc_atomic_umax_addr1(volatile global uint*, uint);
_CLC_DECL uint __clc_atomic_umax_addr3(volatile local uint*, uint);

ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_max, atomic_max, global, 1)
ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_max, atomic_max, local, 3)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_max, atomic_umax, global, 1)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_max, atomic_umax, local, 3)

_CLC_DECL signed int __clc_atomic_min_addr1(volatile global signed int*, signed int);
_CLC_DECL signed int __clc_atomic_min_addr3(volatile local signed int*, signed int);
_CLC_DECL uint __clc_atomic_umin_addr1(volatile global uint*, uint);
_CLC_DECL uint __clc_atomic_umin_addr3(volatile local uint*, uint);

ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_min, atomic_min, global, 1)
ATOMIC_FUNC_DEFINE(signed, signed, int, atomic_min, atomic_min, local, 3)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_min, atomic_umin, global, 1)
ATOMIC_FUNC_DEFINE(unsigned, unsigned, int, atomic_min, atomic_umin, local, 3)
