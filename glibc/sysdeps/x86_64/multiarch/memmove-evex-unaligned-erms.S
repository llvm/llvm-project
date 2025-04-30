#if IS_IN (libc)
# define VEC_SIZE	32
# define XMM0		xmm16
# define XMM1		xmm17
# define YMM0		ymm16
# define YMM1		ymm17
# define VEC0		ymm16
# define VEC1		ymm17
# define VEC2		ymm18
# define VEC3		ymm19
# define VEC4		ymm20
# define VEC5		ymm21
# define VEC6		ymm22
# define VEC7		ymm23
# define VEC8		ymm24
# define VEC9		ymm25
# define VEC10		ymm26
# define VEC11		ymm27
# define VEC12		ymm28
# define VEC13		ymm29
# define VEC14		ymm30
# define VEC15		ymm31
# define VEC(i)		VEC##i
# define VMOVNT		vmovntdq
# define VMOVU		vmovdqu64
# define VMOVA		vmovdqa64
# define VZEROUPPER

# define SECTION(p)		p##.evex
# define MEMMOVE_SYMBOL(p,s)	p##_evex_##s

# include "memmove-vec-unaligned-erms.S"
#endif
