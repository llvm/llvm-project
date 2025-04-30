#if IS_IN (libc)
# define VEC_SIZE	64
# define XMM0		xmm16
# define YMM0		ymm16
# define VEC0		zmm16
# define VEC(i)		VEC##i
# define VMOVU		vmovdqu64
# define VMOVA		vmovdqa64
# define VZEROUPPER

# define MEMSET_VDUP_TO_VEC0_AND_SET_RETURN(d, r) \
  movq r, %rax; \
  vpbroadcastb d, %VEC0

# define WMEMSET_VDUP_TO_VEC0_AND_SET_RETURN(d, r) \
  movq r, %rax; \
  vpbroadcastd d, %VEC0

# define SECTION(p)		p##.evex512
# define MEMSET_SYMBOL(p,s)	p##_avx512_##s
# define WMEMSET_SYMBOL(p,s)	p##_avx512_##s
# define USE_LESS_VEC_MASK_STORE	1
# include "memset-vec-unaligned-erms.S"
#endif
