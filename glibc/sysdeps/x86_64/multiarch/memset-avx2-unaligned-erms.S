#if IS_IN (libc)
# define VEC_SIZE	32
# define VEC(i)		ymm##i
# define VMOVU		vmovdqu
# define VMOVA		vmovdqa

# define MEMSET_VDUP_TO_VEC0_AND_SET_RETURN(d, r) \
  vmovd d, %xmm0; \
  movq r, %rax; \
  vpbroadcastb %xmm0, %ymm0

# define WMEMSET_VDUP_TO_VEC0_AND_SET_RETURN(d, r) \
  vmovd d, %xmm0; \
  movq r, %rax; \
  vpbroadcastd %xmm0, %ymm0

# ifndef SECTION
#  define SECTION(p)		p##.avx
# endif
# ifndef MEMSET_SYMBOL
#  define MEMSET_SYMBOL(p,s)	p##_avx2_##s
# endif
# ifndef WMEMSET_SYMBOL
#  define WMEMSET_SYMBOL(p,s)	p##_avx2_##s
# endif

# include "memset-vec-unaligned-erms.S"
#endif
