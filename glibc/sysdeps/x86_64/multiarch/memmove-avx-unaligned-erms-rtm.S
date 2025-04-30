#if IS_IN (libc)
# define VEC_SIZE	32
# define VEC(i)		ymm##i
# define VMOVNT		vmovntdq
# define VMOVU		vmovdqu
# define VMOVA		vmovdqa

# define ZERO_UPPER_VEC_REGISTERS_RETURN \
  ZERO_UPPER_VEC_REGISTERS_RETURN_XTEST

# define VZEROUPPER_RETURN jmp	 L(return)

# define SECTION(p)		p##.avx.rtm
# define MEMMOVE_SYMBOL(p,s)	p##_avx_##s##_rtm

# include "memmove-vec-unaligned-erms.S"
#endif
