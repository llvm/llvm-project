/* Simple transformations functions - s390 version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#if defined HAVE_S390_VX_ASM_SUPPORT
# include <ifunc-resolve.h>

# if defined HAVE_S390_VX_GCC_SUPPORT
#  define ASM_CLOBBER_VR(NR) , NR
# else
#  define ASM_CLOBBER_VR(NR)
# endif

# define ICONV_C_NAME(NAME) __##NAME##_c
# define ICONV_VX_NAME(NAME) __##NAME##_vx
# ifdef HAVE_S390_MIN_Z13_ZARCH_ASM_SUPPORT
/* We support z13 instructions by default -> Just use the vector variant.  */
#  define ICONV_VX_IFUNC(FUNC) strong_alias (ICONV_VX_NAME (FUNC), FUNC)
# else
/* We have to use ifunc to determine if z13 instructions are supported.  */
#  define ICONV_VX_IFUNC(FUNC)						\
  s390_libc_ifunc_expr (ICONV_C_NAME (FUNC), FUNC,			\
			(hwcap & HWCAP_S390_VX)				\
			? ICONV_VX_NAME (FUNC)				\
			: ICONV_C_NAME (FUNC)				\
			)
# endif
# define ICONV_VX_SINGLE(NAME)						\
  static __typeof (NAME##_single) __##NAME##_vx_single __attribute__((alias(#NAME "_single")));

/* Generate the transformations which are used, if the target machine does not
   support vector instructions.  */
# define __gconv_transform_ascii_internal		\
  ICONV_C_NAME (__gconv_transform_ascii_internal)
# define __gconv_transform_internal_ascii		\
  ICONV_C_NAME (__gconv_transform_internal_ascii)
# define __gconv_transform_internal_ucs4le		\
  ICONV_C_NAME (__gconv_transform_internal_ucs4le)
# define __gconv_transform_ucs4_internal		\
  ICONV_C_NAME (__gconv_transform_ucs4_internal)
# define __gconv_transform_ucs4le_internal		\
  ICONV_C_NAME (__gconv_transform_ucs4le_internal)
# define __gconv_transform_ucs2_internal		\
  ICONV_C_NAME (__gconv_transform_ucs2_internal)
# define __gconv_transform_ucs2reverse_internal		\
  ICONV_C_NAME (__gconv_transform_ucs2reverse_internal)
# define __gconv_transform_internal_ucs2		\
  ICONV_C_NAME (__gconv_transform_internal_ucs2)
# define __gconv_transform_internal_ucs2reverse		\
  ICONV_C_NAME (__gconv_transform_internal_ucs2reverse)


# include <iconv/gconv_simple.c>

# undef __gconv_transform_ascii_internal
# undef __gconv_transform_internal_ascii
# undef __gconv_transform_internal_ucs4le
# undef __gconv_transform_ucs4_internal
# undef __gconv_transform_ucs4le_internal
# undef __gconv_transform_ucs2_internal
# undef __gconv_transform_ucs2reverse_internal
# undef __gconv_transform_internal_ucs2
# undef __gconv_transform_internal_ucs2reverse

/* Now define the functions with vector support.  */
# if defined __s390x__
#  define CONVERT_32BIT_SIZE_T(REG)
# else
#  define CONVERT_32BIT_SIZE_T(REG) "llgfr %" #REG ",%" #REG "\n\t"
# endif

/* Convert from ISO 646-IRV to the internal (UCS4-like) format.  */
# define DEFINE_INIT		0
# define DEFINE_FINI		0
# define MIN_NEEDED_FROM	1
# define MIN_NEEDED_TO		4
# define FROM_DIRECTION		1
# define FROM_LOOP		ICONV_VX_NAME (ascii_internal_loop)
# define TO_LOOP		ICONV_VX_NAME (ascii_internal_loop) /* This is not used.  */
# define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_ascii_internal)
# define ONE_DIRECTION		1

# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define LOOPFCT		FROM_LOOP
# define BODY_ORIG_ERROR						\
    /* The value is too large.  We don't try transliteration here since \
       this is not an error because of the lack of possibilities to	\
       represent the result.  This is a genuine bug in the input since	\
       ASCII does not allow such values.  */				\
    STANDARD_FROM_LOOP_ERR_HANDLER (1);

# define BODY_ORIG							\
  {									\
    if (__glibc_unlikely (*inptr > '\x7f'))				\
      {									\
	BODY_ORIG_ERROR							\
      }									\
    else								\
      {									\
	/* It's an one byte sequence.  */				\
	*((uint32_t *) outptr) = *inptr++;				\
	outptr += sizeof (uint32_t);					\
      }									\
  }
# define BODY								\
  {									\
    size_t len = inend - inptr;						\
    if (len > (outend - outptr) / 4)					\
      len = (outend - outptr) / 4;					\
    size_t loop_count, tmp;						\
    __asm__ volatile (".machine push\n\t"				\
		      ".machine \"z13\"\n\t"				\
		      ".machinemode \"zarch_nohighgprs\"\n\t"		\
		      CONVERT_32BIT_SIZE_T ([R_LEN])			\
		      "    vrepib %%v30,0x7f\n\t" /* For compare > 0x7f.  */ \
		      "    srlg %[R_LI],%[R_LEN],4\n\t"			\
		      "    vrepib %%v31,0x20\n\t"			\
		      "    clgije %[R_LI],0,1f\n\t"			\
		      "0:  \n\t" /* Handle 16-byte blocks.  */		\
		      "    vl %%v16,0(%[R_IN])\n\t"			\
		      /* Checking for values > 0x7f.  */		\
		      "    vstrcbs %%v17,%%v16,%%v30,%%v31\n\t"		\
		      "    jno 10f\n\t"					\
		      /* Enlarge to UCS4.  */				\
		      "    vuplhb %%v17,%%v16\n\t"			\
		      "    vupllb %%v18,%%v16\n\t"			\
		      "    vuplhh %%v19,%%v17\n\t"			\
		      "    vupllh %%v20,%%v17\n\t"			\
		      "    vuplhh %%v21,%%v18\n\t"			\
		      "    vupllh %%v22,%%v18\n\t"			\
		      /* Store 64bytes to buf_out.  */			\
		      "    vstm %%v19,%%v22,0(%[R_OUT])\n\t"		\
		      "    la %[R_IN],16(%[R_IN])\n\t"			\
		      "    la %[R_OUT],64(%[R_OUT])\n\t"		\
		      "    brctg %[R_LI],0b\n\t"			\
		      "    lghi %[R_LI],15\n\t"				\
		      "    ngr %[R_LEN],%[R_LI]\n\t"			\
		      "    je 20f\n\t" /* Jump away if no remaining bytes.  */ \
		      /* Handle remaining bytes.  */			\
		      "1: aghik %[R_LI],%[R_LEN],-1\n\t"		\
		      "    jl 20f\n\t" /* Jump away if no remaining bytes.  */ \
		      "    vll %%v16,%[R_LI],0(%[R_IN])\n\t"		\
		      /* Checking for values > 0x7f.  */		\
		      "    vstrcbs %%v17,%%v16,%%v30,%%v31\n\t"		\
		      "    vlgvb %[R_TMP],%%v17,7\n\t"			\
		      "    clr %[R_TMP],%[R_LI]\n\t"			\
		      "    locrh %[R_TMP],%[R_LEN]\n\t"			\
		      "    locghih %[R_LEN],0\n\t"			\
		      "    j 12f\n\t"					\
		      "10:\n\t"						\
		      /* Found a value > 0x7f.				\
			 Store the preceding chars.  */			\
		      "    vlgvb %[R_TMP],%%v17,7\n\t"			\
		      "12: la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		      "    sllk %[R_TMP],%[R_TMP],2\n\t"		\
		      "    ahi %[R_TMP],-1\n\t"				\
		      "    jl 20f\n\t"					\
		      "    lgr %[R_LI],%[R_TMP]\n\t"			\
		      "    vuplhb %%v17,%%v16\n\t"			\
		      "    vuplhh %%v19,%%v17\n\t"			\
		      "    vstl %%v19,%[R_LI],0(%[R_OUT])\n\t"		\
		      "    ahi %[R_LI],-16\n\t"				\
		      "    jl 11f\n\t"					\
		      "    vupllh %%v20,%%v17\n\t"			\
		      "    vstl %%v20,%[R_LI],16(%[R_OUT])\n\t"		\
		      "    ahi %[R_LI],-16\n\t"				\
		      "    jl 11f\n\t"					\
		      "    vupllb %%v18,%%v16\n\t"			\
		      "    vuplhh %%v21,%%v18\n\t"			\
		      "    vstl %%v21,%[R_LI],32(%[R_OUT])\n\t"		\
		      "    ahi %[R_LI],-16\n\t"				\
		      "    jl 11f\n\t"					\
		      "    vupllh %%v22,%%v18\n\t"			\
		      "    vstl %%v22,%[R_LI],48(%[R_OUT])\n\t"		\
		      "11:\n\t"						\
		      "    la %[R_OUT],1(%[R_TMP],%[R_OUT])\n\t"	\
		      "20:\n\t"						\
		      ".machine pop"					\
		      : /* outputs */ [R_OUT] "+a" (outptr)		\
			, [R_IN] "+a" (inptr)				\
			, [R_LEN] "+d" (len)				\
			, [R_LI] "=d" (loop_count)			\
			, [R_TMP] "=a" (tmp)				\
		      : /* inputs */					\
		      : /* clobber list*/ "memory", "cc"		\
			ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")	\
			ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")	\
			ASM_CLOBBER_VR ("v20") ASM_CLOBBER_VR ("v21")	\
			ASM_CLOBBER_VR ("v22") ASM_CLOBBER_VR ("v30")	\
			ASM_CLOBBER_VR ("v31")				\
		      );						\
    if (len > 0)							\
      {									\
	/* Found an invalid character at the next input byte.  */	\
	BODY_ORIG_ERROR							\
      }									\
  }

# define LOOP_NEED_FLAGS
# include <iconv/loop.c>
# include <iconv/skeleton.c>
# undef BODY_ORIG
# undef BODY_ORIG_ERROR
ICONV_VX_IFUNC (__gconv_transform_ascii_internal)

/* Convert from the internal (UCS4-like) format to ISO 646-IRV.  */
# define DEFINE_INIT		0
# define DEFINE_FINI		0
# define MIN_NEEDED_FROM	4
# define MIN_NEEDED_TO		1
# define FROM_DIRECTION		1
# define FROM_LOOP		ICONV_VX_NAME (internal_ascii_loop)
# define TO_LOOP		ICONV_VX_NAME (internal_ascii_loop) /* This is not used.  */
# define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_internal_ascii)
# define ONE_DIRECTION		1

# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define LOOPFCT		FROM_LOOP
# define BODY_ORIG_ERROR						\
  UNICODE_TAG_HANDLER (*((const uint32_t *) inptr), 4);			\
  STANDARD_TO_LOOP_ERR_HANDLER (4);

# define BODY_ORIG							\
  {									\
    if (__glibc_unlikely (*((const uint32_t *) inptr) > 0x7f))		\
      {									\
	BODY_ORIG_ERROR							\
      }									\
    else								\
      {									\
	/* It's an one byte sequence.  */				\
	*outptr++ = *((const uint32_t *) inptr);			\
	inptr += sizeof (uint32_t);					\
      }									\
  }
# define BODY								\
  {									\
    size_t len = (inend - inptr) / 4;					\
    if (len > outend - outptr)						\
      len = outend - outptr;						\
    size_t loop_count, tmp, tmp2;					\
    __asm__ volatile (".machine push\n\t"				\
		      ".machine \"z13\"\n\t"				\
		      ".machinemode \"zarch_nohighgprs\"\n\t"		\
		      CONVERT_32BIT_SIZE_T ([R_LEN])			\
		      /* Setup to check for ch > 0x7f.  */		\
		      "    vzero %%v21\n\t"				\
		      "    srlg %[R_LI],%[R_LEN],4\n\t"			\
		      "    vleih %%v21,8192,0\n\t"  /* element 0:   >  */ \
		      "    vleih %%v21,-8192,2\n\t" /* element 1: =<>  */ \
		      "    vleif %%v20,127,0\n\t"   /* element 0: 127  */ \
		      "    lghi %[R_TMP],0\n\t"				\
		      "    clgije %[R_LI],0,1f\n\t"			\
		      "0:\n\t"						\
		      "    vlm %%v16,%%v19,0(%[R_IN])\n\t"		\
		      /* Shorten to byte values.  */			\
		      "    vpkf %%v23,%%v16,%%v17\n\t"			\
		      "    vpkf %%v24,%%v18,%%v19\n\t"			\
		      "    vpkh %%v23,%%v23,%%v24\n\t"			\
		      /* Checking for values > 0x7f.  */		\
		      "    vstrcfs %%v22,%%v16,%%v20,%%v21\n\t"		\
		      "    jno 10f\n\t"					\
		      "    vstrcfs %%v22,%%v17,%%v20,%%v21\n\t"		\
		      "    jno 11f\n\t"					\
		      "    vstrcfs %%v22,%%v18,%%v20,%%v21\n\t"		\
		      "    jno 12f\n\t"					\
		      "    vstrcfs %%v22,%%v19,%%v20,%%v21\n\t"		\
		      "    jno 13f\n\t"					\
		      /* Store 16bytes to outptr.  */			\
		      "    vst %%v23,0(%[R_OUT])\n\t"			\
		      "    la %[R_IN],64(%[R_IN])\n\t"			\
		      "    la %[R_OUT],16(%[R_OUT])\n\t"		\
		      "    brctg %[R_LI],0b\n\t"			\
		      "    lghi %[R_LI],15\n\t"				\
		      "    ngr %[R_LEN],%[R_LI]\n\t"			\
		      "    je 20f\n\t" /* Jump away if no remaining bytes.  */ \
		      /* Handle remaining bytes.  */			\
		      "1: sllg %[R_LI],%[R_LEN],2\n\t"			\
		      "    aghi %[R_LI],-1\n\t"				\
		      "    jl 20f\n\t" /* Jump away if no remaining bytes.  */ \
		      /* Load remaining 1...63 bytes.  */		\
		      "    vll %%v16,%[R_LI],0(%[R_IN])\n\t"		\
		      "    ahi %[R_LI],-16\n\t"				\
		      "    jl 2f\n\t"					\
		      "    vll %%v17,%[R_LI],16(%[R_IN])\n\t"		\
		      "    ahi %[R_LI],-16\n\t"				\
		      "    jl 2f\n\t"					\
		      "    vll %%v18,%[R_LI],32(%[R_IN])\n\t"		\
		      "    ahi %[R_LI],-16\n\t"				\
		      "    jl 2f\n\t"					\
		      "    vll %%v19,%[R_LI],48(%[R_IN])\n\t"		\
		      "2:\n\t"						\
		      /* Shorten to byte values.  */			\
		      "    vpkf %%v23,%%v16,%%v17\n\t"			\
		      "    vpkf %%v24,%%v18,%%v19\n\t"			\
		      "    vpkh %%v23,%%v23,%%v24\n\t"			\
		      "    sllg %[R_LI],%[R_LEN],2\n\t"			\
		      "    aghi %[R_LI],-16\n\t"			\
		      "    jl 3f\n\t" /* v16 is not fully loaded.  */	\
		      "    vstrcfs %%v22,%%v16,%%v20,%%v21\n\t"		\
		      "    jno 10f\n\t"					\
		      "    aghi %[R_LI],-16\n\t"			\
		      "    jl 4f\n\t" /* v17 is not fully loaded.  */	\
		      "    vstrcfs %%v22,%%v17,%%v20,%%v21\n\t"		\
		      "    jno 11f\n\t"					\
		      "    aghi %[R_LI],-16\n\t"			\
		      "    jl 5f\n\t" /* v18 is not fully loaded.  */	\
		      "    vstrcfs %%v22,%%v18,%%v20,%%v21\n\t"		\
		      "    jno 12f\n\t"					\
		      "    aghi %[R_LI],-16\n\t"			\
		      /* v19 is not fully loaded. */			\
		      "    lghi %[R_TMP],12\n\t"			\
		      "    vstrcfs %%v22,%%v19,%%v20,%%v21\n\t"		\
		      "6: vlgvb %[R_I],%%v22,7\n\t"			\
		      "    aghi %[R_LI],16\n\t"				\
		      "    clrjl %[R_I],%[R_LI],14f\n\t"		\
		      "    lgr %[R_I],%[R_LEN]\n\t"			\
		      "    lghi %[R_LEN],0\n\t"				\
		      "    j 15f\n\t"					\
		      "3: vstrcfs %%v22,%%v16,%%v20,%%v21\n\t"		\
		      "    j 6b\n\t"					\
		      "4: vstrcfs %%v22,%%v17,%%v20,%%v21\n\t"		\
		      "    lghi %[R_TMP],4\n\t"				\
		      "    j 6b\n\t"					\
		      "5: vstrcfs %%v22,%%v17,%%v20,%%v21\n\t"		\
		      "    lghi %[R_TMP],8\n\t"				\
		      "    j 6b\n\t"					\
		      /* Found a value > 0x7f.  */			\
		      "13: ahi %[R_TMP],4\n\t"				\
		      "12: ahi %[R_TMP],4\n\t"				\
		      "11: ahi %[R_TMP],4\n\t"				\
		      "10: vlgvb %[R_I],%%v22,7\n\t"			\
		      "14: srlg %[R_I],%[R_I],2\n\t"			\
		      "    agr %[R_I],%[R_TMP]\n\t"			\
		      "    je 20f\n\t"					\
		      /* Store characters before invalid one...  */	\
		      "15: aghi %[R_I],-1\n\t"				\
		      "    vstl %%v23,%[R_I],0(%[R_OUT])\n\t"		\
		      /* ... and update pointers.  */			\
		      "    la %[R_OUT],1(%[R_I],%[R_OUT])\n\t"		\
		      "    sllg %[R_I],%[R_I],2\n\t"			\
		      "    la %[R_IN],4(%[R_I],%[R_IN])\n\t"		\
		      "20:\n\t"						\
		      ".machine pop"					\
		      : /* outputs */ [R_OUT] "+a" (outptr)		\
			, [R_IN] "+a" (inptr)				\
			, [R_LEN] "+d" (len)				\
			, [R_LI] "=d" (loop_count)			\
			, [R_I] "=a" (tmp2)				\
			, [R_TMP] "=d" (tmp)				\
		      : /* inputs */					\
		      : /* clobber list*/ "memory", "cc"		\
			ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")	\
			ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")	\
			ASM_CLOBBER_VR ("v20") ASM_CLOBBER_VR ("v21")	\
			ASM_CLOBBER_VR ("v22") ASM_CLOBBER_VR ("v23")	\
			ASM_CLOBBER_VR ("v24")				\
		      );						\
    if (len > 0)							\
      {									\
	/* Found an invalid character > 0x7f at next character.  */	\
	BODY_ORIG_ERROR							\
      }									\
  }
# define LOOP_NEED_FLAGS
# include <iconv/loop.c>
# include <iconv/skeleton.c>
# undef BODY_ORIG
# undef BODY_ORIG_ERROR
ICONV_VX_IFUNC (__gconv_transform_internal_ascii)


/* Convert from internal UCS4 to UCS4 little endian form.  */
# define DEFINE_INIT		0
# define DEFINE_FINI		0
# define MIN_NEEDED_FROM	4
# define MIN_NEEDED_TO		4
# define FROM_DIRECTION		1
# define FROM_LOOP		ICONV_VX_NAME (internal_ucs4le_loop)
# define TO_LOOP		ICONV_VX_NAME (internal_ucs4le_loop) /* This is not used.  */
# define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_internal_ucs4le)
# define ONE_DIRECTION		0

static inline int
__attribute ((always_inline))
ICONV_VX_NAME (internal_ucs4le_loop) (struct __gconv_step *step,
				      struct __gconv_step_data *step_data,
				      const unsigned char **inptrp,
				      const unsigned char *inend,
				      unsigned char **outptrp,
				      const unsigned char *outend,
				      size_t *irreversible)
{
  const unsigned char *inptr = *inptrp;
  unsigned char *outptr = *outptrp;
  int result;
  size_t len = MIN (inend - inptr, outend - outptr) / 4;
  size_t loop_count;
  __asm__ volatile (".machine push\n\t"
		    ".machine \"z13\"\n\t"
		    ".machinemode \"zarch_nohighgprs\"\n\t"
		    CONVERT_32BIT_SIZE_T ([R_LEN])
		    "    bras %[R_LI],1f\n\t"
		    /* Vector permute mask:  */
		    "    .long 0x03020100,0x7060504,0x0B0A0908,0x0F0E0D0C\n\t"
		    "1:  vl %%v20,0(%[R_LI])\n\t"
		    /* Process 64byte (16char) blocks.  */
		    "    srlg %[R_LI],%[R_LEN],4\n\t"
		    "    clgije %[R_LI],0,10f\n\t"
		    "0:  vlm %%v16,%%v19,0(%[R_IN])\n\t"
		    "    vperm %%v16,%%v16,%%v16,%%v20\n\t"
		    "    vperm %%v17,%%v17,%%v17,%%v20\n\t"
		    "    vperm %%v18,%%v18,%%v18,%%v20\n\t"
		    "    vperm %%v19,%%v19,%%v19,%%v20\n\t"
		    "    vstm %%v16,%%v19,0(%[R_OUT])\n\t"
		    "    la %[R_IN],64(%[R_IN])\n\t"
		    "    la %[R_OUT],64(%[R_OUT])\n\t"
		    "    brctg %[R_LI],0b\n\t"
		    "    llgfr %[R_LEN],%[R_LEN]\n\t"
		    "    nilf %[R_LEN],15\n\t"
		    /* Process 16byte (4char) blocks.  */
		    "10: srlg %[R_LI],%[R_LEN],2\n\t"
		    "    clgije %[R_LI],0,20f\n\t"
		    "11: vl %%v16,0(%[R_IN])\n\t"
		    "    vperm %%v16,%%v16,%%v16,%%v20\n\t"
		    "    vst %%v16,0(%[R_OUT])\n\t"
		    "    la %[R_IN],16(%[R_IN])\n\t"
		    "    la %[R_OUT],16(%[R_OUT])\n\t"
		    "    brctg %[R_LI],11b\n\t"
		    "    nill %[R_LEN],3\n\t"
		    /* Process <16bytes.  */
		    "20: sll %[R_LEN],2\n\t"
		    "    ahi %[R_LEN],-1\n\t"
		    "    jl 30f\n\t"
		    "    vll %%v16,%[R_LEN],0(%[R_IN])\n\t"
		    "    vperm %%v16,%%v16,%%v16,%%v20\n\t"
		    "    vstl %%v16,%[R_LEN],0(%[R_OUT])\n\t"
		    "    la %[R_IN],1(%[R_LEN],%[R_IN])\n\t"
		    "    la %[R_OUT],1(%[R_LEN],%[R_OUT])\n\t"
		    "30: \n\t"
		    ".machine pop"
		    : /* outputs */ [R_OUT] "+a" (outptr)
		      , [R_IN] "+a" (inptr)
		      , [R_LI] "=a" (loop_count)
		      , [R_LEN] "+a" (len)
		    : /* inputs */
		    : /* clobber list*/ "memory", "cc"
		      ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")
		      ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")
		      ASM_CLOBBER_VR ("v20")
		    );
  *inptrp = inptr;
  *outptrp = outptr;

  /* Determine the status.  */
  if (*inptrp == inend)
    result = __GCONV_EMPTY_INPUT;
  else if (*outptrp + 4 > outend)
    result = __GCONV_FULL_OUTPUT;
  else
    result = __GCONV_INCOMPLETE_INPUT;

  return result;
}

ICONV_VX_SINGLE (internal_ucs4le_loop)
# include <iconv/skeleton.c>
ICONV_VX_IFUNC (__gconv_transform_internal_ucs4le)


/* Transform from UCS4 to the internal, UCS4-like format.  Unlike
   for the other direction we have to check for correct values here.  */
# define DEFINE_INIT		0
# define DEFINE_FINI		0
# define MIN_NEEDED_FROM	4
# define MIN_NEEDED_TO		4
# define FROM_DIRECTION		1
# define FROM_LOOP		ICONV_VX_NAME (ucs4_internal_loop)
# define TO_LOOP		ICONV_VX_NAME (ucs4_internal_loop) /* This is not used.  */
# define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_ucs4_internal)
# define ONE_DIRECTION		0


static inline int
__attribute ((always_inline))
ICONV_VX_NAME (ucs4_internal_loop) (struct __gconv_step *step,
				    struct __gconv_step_data *step_data,
				    const unsigned char **inptrp,
				    const unsigned char *inend,
				    unsigned char **outptrp,
				    const unsigned char *outend,
				    size_t *irreversible)
{
  int flags = step_data->__flags;
  const unsigned char *inptr = *inptrp;
  unsigned char *outptr = *outptrp;
  int result;
  size_t len, loop_count;
  do
    {
      len = MIN (inend - inptr, outend - outptr) / 4;
      __asm__ volatile (".machine push\n\t"
			".machine \"z13\"\n\t"
			".machinemode \"zarch_nohighgprs\"\n\t"
			CONVERT_32BIT_SIZE_T ([R_LEN])
			/* Setup to check for ch > 0x7fffffff.  */
			"    larl %[R_LI],9f\n\t"
			"    vlm %%v20,%%v21,0(%[R_LI])\n\t"
			"    srlg %[R_LI],%[R_LEN],2\n\t"
			"    clgije %[R_LI],0,1f\n\t"
			/* Process 16byte (4char) blocks.  */
			"0:  vl %%v16,0(%[R_IN])\n\t"
			"    vstrcfs %%v22,%%v16,%%v20,%%v21\n\t"
			"    jno 10f\n\t"
			"    vst %%v16,0(%[R_OUT])\n\t"
			"    la %[R_IN],16(%[R_IN])\n\t"
			"    la %[R_OUT],16(%[R_OUT])\n\t"
			"    brctg %[R_LI],0b\n\t"
			"    llgfr %[R_LEN],%[R_LEN]\n\t"
			"    nilf %[R_LEN],3\n\t"
			/* Process <16bytes.  */
			"1:  sll %[R_LEN],2\n\t"
			"    ahik %[R_LI],%[R_LEN],-1\n\t"
			"    jl 20f\n\t" /* No further bytes available.  */
			"    vll %%v16,%[R_LI],0(%[R_IN])\n\t"
			"    vstrcfs %%v22,%%v16,%%v20,%%v21\n\t"
			"    vlgvb %[R_LI],%%v22,7\n\t"
			"    clr %[R_LI],%[R_LEN]\n\t"
			"    locgrhe %[R_LI],%[R_LEN]\n\t"
			"    locghihe %[R_LEN],0\n\t"
			"    j 11f\n\t"
			/* v20: Vector string range compare values.  */
			"9:  .long 0x7fffffff,0x0,0x0,0x0\n\t"
			/* v21: Vector string range compare control-bits.
			   element 0: >; element 1: =<> (always true)  */
			"    .long 0x20000000,0xE0000000,0x0,0x0\n\t"
			/* Found a value > 0x7fffffff.  */
			"10: vlgvb %[R_LI],%%v22,7\n\t"
			/* Store characters before invalid one.  */
			"11: aghi %[R_LI],-1\n\t"
			"    jl 20f\n\t"
			"    vstl %%v16,%[R_LI],0(%[R_OUT])\n\t"
			"    la %[R_IN],1(%[R_LI],%[R_IN])\n\t"
			"    la %[R_OUT],1(%[R_LI],%[R_OUT])\n\t"
			"20:\n\t"
			".machine pop"
			: /* outputs */ [R_OUT] "+a" (outptr)
			  , [R_IN] "+a" (inptr)
			  , [R_LI] "=a" (loop_count)
			  , [R_LEN] "+d" (len)
			: /* inputs */
			: /* clobber list*/ "memory", "cc"
			  ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v20")
			  ASM_CLOBBER_VR ("v21") ASM_CLOBBER_VR ("v22")
			);
      if (len > 0)
	{
	  /* The value is too large.  We don't try transliteration here since
	     this is not an error because of the lack of possibilities to
	     represent the result.  This is a genuine bug in the input since
	     UCS4 does not allow such values.  */
	  if (irreversible == NULL)
	    /* We are transliterating, don't try to correct anything.  */
	    return __GCONV_ILLEGAL_INPUT;

	  if (flags & __GCONV_IGNORE_ERRORS)
	    {
	      /* Just ignore this character.  */
	      ++*irreversible;
	      inptr += 4;
	      continue;
	    }

	  *inptrp = inptr;
	  *outptrp = outptr;
	  return __GCONV_ILLEGAL_INPUT;
	}
    }
  while (len > 0);

  *inptrp = inptr;
  *outptrp = outptr;

  /* Determine the status.  */
  if (*inptrp == inend)
    result = __GCONV_EMPTY_INPUT;
  else if (*outptrp + 4 > outend)
    result = __GCONV_FULL_OUTPUT;
  else
    result = __GCONV_INCOMPLETE_INPUT;

  return result;
}

ICONV_VX_SINGLE (ucs4_internal_loop)
# include <iconv/skeleton.c>
ICONV_VX_IFUNC (__gconv_transform_ucs4_internal)


/* Transform from UCS4-LE to the internal encoding.  */
# define DEFINE_INIT		0
# define DEFINE_FINI		0
# define MIN_NEEDED_FROM	4
# define MIN_NEEDED_TO		4
# define FROM_DIRECTION		1
# define FROM_LOOP		ICONV_VX_NAME (ucs4le_internal_loop)
# define TO_LOOP		ICONV_VX_NAME (ucs4le_internal_loop) /* This is not used.  */
# define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_ucs4le_internal)
# define ONE_DIRECTION		0

static inline int
__attribute ((always_inline))
ICONV_VX_NAME (ucs4le_internal_loop) (struct __gconv_step *step,
				      struct __gconv_step_data *step_data,
				      const unsigned char **inptrp,
				      const unsigned char *inend,
				      unsigned char **outptrp,
				      const unsigned char *outend,
				      size_t *irreversible)
{
  int flags = step_data->__flags;
  const unsigned char *inptr = *inptrp;
  unsigned char *outptr = *outptrp;
  int result;
  size_t len, loop_count;
  do
    {
      len = MIN (inend - inptr, outend - outptr) / 4;
      __asm__ volatile (".machine push\n\t"
			".machine \"z13\"\n\t"
			".machinemode \"zarch_nohighgprs\"\n\t"
			CONVERT_32BIT_SIZE_T ([R_LEN])
			/* Setup to check for ch > 0x7fffffff.  */
			"    larl %[R_LI],9f\n\t"
			"    vlm %%v20,%%v22,0(%[R_LI])\n\t"
			"    srlg %[R_LI],%[R_LEN],2\n\t"
			"    clgije %[R_LI],0,1f\n\t"
			/* Process 16byte (4char) blocks.  */
			"0:  vl %%v16,0(%[R_IN])\n\t"
			"    vperm %%v16,%%v16,%%v16,%%v22\n\t"
			"    vstrcfs %%v23,%%v16,%%v20,%%v21\n\t"
			"    jno 10f\n\t"
			"    vst %%v16,0(%[R_OUT])\n\t"
			"    la %[R_IN],16(%[R_IN])\n\t"
			"    la %[R_OUT],16(%[R_OUT])\n\t"
			"    brctg %[R_LI],0b\n\t"
			"    llgfr %[R_LEN],%[R_LEN]\n\t"
			"    nilf %[R_LEN],3\n\t"
			/* Process <16bytes.  */
			"1:  sll %[R_LEN],2\n\t"
			"    ahik %[R_LI],%[R_LEN],-1\n\t"
			"    jl 20f\n\t" /* No further bytes available.  */
			"    vll %%v16,%[R_LI],0(%[R_IN])\n\t"
			"    vperm %%v16,%%v16,%%v16,%%v22\n\t"
			"    vstrcfs %%v23,%%v16,%%v20,%%v21\n\t"
			"    vlgvb %[R_LI],%%v23,7\n\t"
			"    clr %[R_LI],%[R_LEN]\n\t"
			"    locgrhe %[R_LI],%[R_LEN]\n\t"
			"    locghihe %[R_LEN],0\n\t"
			"    j 11f\n\t"
			/* v20: Vector string range compare values.  */
			"9: .long 0x7fffffff,0x0,0x0,0x0\n\t"
			/* v21: Vector string range compare control-bits.
			   element 0: >; element 1: =<> (always true)  */
			"    .long 0x20000000,0xE0000000,0x0,0x0\n\t"
			/* v22: Vector permute mask.  */
			"    .long 0x03020100,0x7060504,0x0B0A0908,0x0F0E0D0C\n\t"
			/* Found a value > 0x7fffffff.  */
			"10: vlgvb %[R_LI],%%v23,7\n\t"
			/* Store characters before invalid one.  */
			"11: aghi %[R_LI],-1\n\t"
			"    jl 20f\n\t"
			"    vstl %%v16,%[R_LI],0(%[R_OUT])\n\t"
			"    la %[R_IN],1(%[R_LI],%[R_IN])\n\t"
			"    la %[R_OUT],1(%[R_LI],%[R_OUT])\n\t"
			"20:\n\t"
			".machine pop"
			: /* outputs */ [R_OUT] "+a" (outptr)
			  , [R_IN] "+a" (inptr)
			  , [R_LI] "=a" (loop_count)
			  , [R_LEN] "+d" (len)
			: /* inputs */
			: /* clobber list*/ "memory", "cc"
			  ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v20")
			  ASM_CLOBBER_VR ("v21") ASM_CLOBBER_VR ("v22")
			  ASM_CLOBBER_VR ("v23")
			);
      if (len > 0)
	{
	  /* The value is too large.  We don't try transliteration here since
	     this is not an error because of the lack of possibilities to
	     represent the result.  This is a genuine bug in the input since
	     UCS4 does not allow such values.  */
	  if (irreversible == NULL)
	    /* We are transliterating, don't try to correct anything.  */
	    return __GCONV_ILLEGAL_INPUT;

	  if (flags & __GCONV_IGNORE_ERRORS)
	    {
	      /* Just ignore this character.  */
	      ++*irreversible;
	      inptr += 4;
	      continue;
	    }

	  *inptrp = inptr;
	  *outptrp = outptr;
	  return __GCONV_ILLEGAL_INPUT;
	}
    }
  while (len > 0);

  *inptrp = inptr;
  *outptrp = outptr;

  /* Determine the status.  */
  if (*inptrp == inend)
    result = __GCONV_EMPTY_INPUT;
  else if (*inptrp + 4 > inend)
    result = __GCONV_INCOMPLETE_INPUT;
  else
    {
      assert (*outptrp + 4 > outend);
      result = __GCONV_FULL_OUTPUT;
    }

  return result;
}
ICONV_VX_SINGLE (ucs4le_internal_loop)
# include <iconv/skeleton.c>
ICONV_VX_IFUNC (__gconv_transform_ucs4le_internal)

/* Convert from UCS2 to the internal (UCS4-like) format.  */
# define DEFINE_INIT		0
# define DEFINE_FINI		0
# define MIN_NEEDED_FROM	2
# define MIN_NEEDED_TO		4
# define FROM_DIRECTION		1
# define FROM_LOOP		ICONV_VX_NAME (ucs2_internal_loop)
# define TO_LOOP		ICONV_VX_NAME (ucs2_internal_loop) /* This is not used.  */
# define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_ucs2_internal)
# define ONE_DIRECTION		1

# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define LOOPFCT		FROM_LOOP
# define BODY_ORIG_ERROR						\
  /* Surrogate characters in UCS-2 input are not valid.  Reject		\
     them.  (Catching this here is not security relevant.)  */		\
  STANDARD_FROM_LOOP_ERR_HANDLER (2);
# define BODY_ORIG							\
  {									\
    uint16_t u1 = get16 (inptr);					\
									\
    if (__glibc_unlikely (u1 >= 0xd800 && u1 < 0xe000))			\
      {									\
	BODY_ORIG_ERROR							\
      }									\
									\
    *((uint32_t *) outptr) = u1;					\
    outptr += sizeof (uint32_t);					\
    inptr += 2;								\
  }
# define BODY								\
  {									\
    size_t len, tmp, tmp2;						\
    len = MIN ((inend - inptr) / 2, (outend - outptr) / 4);		\
    __asm__ volatile (".machine push\n\t"				\
		      ".machine \"z13\"\n\t"				\
		      ".machinemode \"zarch_nohighgprs\"\n\t"		\
		      CONVERT_32BIT_SIZE_T ([R_LEN])			\
		      /* Setup to check for ch >= 0xd800 && ch < 0xe000.  */ \
		      "    larl %[R_TMP],9f\n\t"			\
		      "    vlm %%v20,%%v21,0(%[R_TMP])\n\t"		\
		      "    srlg %[R_TMP],%[R_LEN],3\n\t"		\
		      "    clgije %[R_TMP],0,1f\n\t"			\
		      /* Process 16byte (8char) blocks.  */		\
		      "0:  vl %%v16,0(%[R_IN])\n\t"			\
		      "    vstrchs %%v19,%%v16,%%v20,%%v21\n\t"		\
		      /* Enlarge UCS2 to UCS4.  */			\
		      "    vuplhh %%v17,%%v16\n\t"			\
		      "    vupllh %%v18,%%v16\n\t"			\
		      "    jno 10f\n\t"					\
		      /* Store 32bytes to buf_out.  */			\
		      "    vstm %%v17,%%v18,0(%[R_OUT])\n\t"		\
		      "    la %[R_IN],16(%[R_IN])\n\t"			\
		      "    la %[R_OUT],32(%[R_OUT])\n\t"		\
		      "    brctg %[R_TMP],0b\n\t"			\
		      "    llgfr %[R_LEN],%[R_LEN]\n\t"			\
		      "    nilf %[R_LEN],7\n\t"				\
		      /* Process <16bytes.  */				\
		      "1:  sll %[R_LEN],1\n\t"				\
		      "    ahik %[R_TMP],%[R_LEN],-1\n\t"		\
		      "    jl 20f\n\t" /* No further bytes available.  */ \
		      "    vll %%v16,%[R_TMP],0(%[R_IN])\n\t"		\
		      "    vstrchs %%v19,%%v16,%%v20,%%v21\n\t"		\
		      /* Enlarge UCS2 to UCS4.  */			\
		      "    vuplhh %%v17,%%v16\n\t"			\
		      "    vupllh %%v18,%%v16\n\t"			\
		      "    vlgvb %[R_TMP],%%v19,7\n\t"			\
		      "    clr %[R_TMP],%[R_LEN]\n\t"			\
		      "    locgrhe %[R_TMP],%[R_LEN]\n\t"		\
		      "    locghihe %[R_LEN],0\n\t"			\
		      "    j 11f\n\t"					\
		      /* v20: Vector string range compare values.  */	\
		      "9:  .short 0xd800,0xe000,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		      /* v21: Vector string range compare control-bits.	\
			 element 0: =>; element 1: <  */		\
		      "    .short 0xa000,0x4000,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		      /* Found an element: ch >= 0xd800 && ch < 0xe000  */ \
		      "10: vlgvb %[R_TMP],%%v19,7\n\t"			\
		      "11: la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		      "    sll %[R_TMP],1\n\t"				\
		      "    lgr %[R_TMP2],%[R_TMP]\n\t"			\
		      "    ahi %[R_TMP],-1\n\t"				\
		      "    jl 20f\n\t"					\
		      "    vstl %%v17,%[R_TMP],0(%[R_OUT])\n\t"		\
		      "    ahi %[R_TMP],-16\n\t"			\
		      "    jl 19f\n\t"					\
		      "    vstl %%v18,%[R_TMP],16(%[R_OUT])\n\t"	\
		      "19: la %[R_OUT],0(%[R_TMP2],%[R_OUT])\n\t"	\
		      "20: \n\t"					\
		      ".machine pop"					\
		      : /* outputs */ [R_OUT] "+a" (outptr)		\
			, [R_IN] "+a" (inptr)				\
			, [R_TMP] "=a" (tmp)				\
			, [R_TMP2] "=a" (tmp2)				\
			, [R_LEN] "+d" (len)				\
		      : /* inputs */					\
		      : /* clobber list*/ "memory", "cc"		\
			ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")	\
			ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")	\
			ASM_CLOBBER_VR ("v20") ASM_CLOBBER_VR ("v21")	\
		      );						\
    if (len > 0)							\
      {									\
	/* Found an invalid character at next input-char.  */		\
	BODY_ORIG_ERROR							\
      }									\
  }

# define LOOP_NEED_FLAGS
# include <iconv/loop.c>
# include <iconv/skeleton.c>
# undef BODY_ORIG
# undef BODY_ORIG_ERROR
ICONV_VX_IFUNC (__gconv_transform_ucs2_internal)

/* Convert from UCS2 in other endianness to the internal (UCS4-like) format. */
# define DEFINE_INIT		0
# define DEFINE_FINI		0
# define MIN_NEEDED_FROM	2
# define MIN_NEEDED_TO		4
# define FROM_DIRECTION		1
# define FROM_LOOP		ICONV_VX_NAME (ucs2reverse_internal_loop)
# define TO_LOOP		ICONV_VX_NAME (ucs2reverse_internal_loop) /* This is not used.*/
# define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_ucs2reverse_internal)
# define ONE_DIRECTION		1

# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define LOOPFCT		FROM_LOOP
# define BODY_ORIG_ERROR						\
  /* Surrogate characters in UCS-2 input are not valid.  Reject		\
     them.  (Catching this here is not security relevant.)  */		\
  if (! ignore_errors_p ())						\
    {									\
      result = __GCONV_ILLEGAL_INPUT;					\
      break;								\
    }									\
  inptr += 2;								\
  ++*irreversible;							\
  continue;

# define BODY_ORIG \
  {									\
    uint16_t u1 = bswap_16 (get16 (inptr));				\
									\
    if (__glibc_unlikely (u1 >= 0xd800 && u1 < 0xe000))			\
      {									\
	BODY_ORIG_ERROR							\
      }									\
									\
    *((uint32_t *) outptr) = u1;					\
    outptr += sizeof (uint32_t);					\
    inptr += 2;								\
  }
# define BODY								\
  {									\
    size_t len, tmp, tmp2;						\
    len = MIN ((inend - inptr) / 2, (outend - outptr) / 4);		\
    __asm__ volatile (".machine push\n\t"				\
		      ".machine \"z13\"\n\t"				\
		      ".machinemode \"zarch_nohighgprs\"\n\t"		\
		      CONVERT_32BIT_SIZE_T ([R_LEN])			\
		      /* Setup to check for ch >= 0xd800 && ch < 0xe000.  */ \
		      "    larl %[R_TMP],9f\n\t"			\
		      "    vlm %%v20,%%v22,0(%[R_TMP])\n\t"		\
		      "    srlg %[R_TMP],%[R_LEN],3\n\t"		\
		      "    clgije %[R_TMP],0,1f\n\t"			\
		      /* Process 16byte (8char) blocks.  */		\
		      "0:  vl %%v16,0(%[R_IN])\n\t"			\
		      "    vperm %%v16,%%v16,%%v16,%%v22\n\t"		\
		      "    vstrchs %%v19,%%v16,%%v20,%%v21\n\t"		\
		      /* Enlarge UCS2 to UCS4.  */			\
		      "    vuplhh %%v17,%%v16\n\t"			\
		      "    vupllh %%v18,%%v16\n\t"			\
		      "    jno 10f\n\t"					\
		      /* Store 32bytes to buf_out.  */			\
		      "    vstm %%v17,%%v18,0(%[R_OUT])\n\t"		\
		      "    la %[R_IN],16(%[R_IN])\n\t"			\
		      "    la %[R_OUT],32(%[R_OUT])\n\t"		\
		      "    brctg %[R_TMP],0b\n\t"			\
		      "    llgfr %[R_LEN],%[R_LEN]\n\t"			\
		      "    nilf %[R_LEN],7\n\t"				\
		      /* Process <16bytes.  */				\
		      "1:  sll %[R_LEN],1\n\t"				\
		      "    ahik %[R_TMP],%[R_LEN],-1\n\t"		\
		      "    jl 20f\n\t" /* No further bytes available.  */ \
		      "    vll %%v16,%[R_TMP],0(%[R_IN])\n\t"		\
		      "    vperm %%v16,%%v16,%%v16,%%v22\n\t"		\
		      "    vstrchs %%v19,%%v16,%%v20,%%v21\n\t"		\
		      /* Enlarge UCS2 to UCS4.  */			\
		      "    vuplhh %%v17,%%v16\n\t"			\
		      "    vupllh %%v18,%%v16\n\t"			\
		      "    vlgvb %[R_TMP],%%v19,7\n\t"			\
		      "    clr %[R_TMP],%[R_LEN]\n\t"			\
		      "    locgrhe %[R_TMP],%[R_LEN]\n\t"		\
		      "    locghihe %[R_LEN],0\n\t"			\
		      "    j 11f\n\t"					\
		      /* v20: Vector string range compare values.  */	\
		      "9:  .short 0xd800,0xe000,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		      /* v21: Vector string range compare control-bits.	\
			 element 0: =>; element 1: <  */		\
		      "    .short 0xa000,0x4000,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		      /* v22: Vector permute mask.  */			\
		      "    .short 0x0100,0x0302,0x0504,0x0706\n\t"	\
		      "    .short 0x0908,0x0b0a,0x0d0c,0x0f0e\n\t"	\
		      /* Found an element: ch >= 0xd800 && ch < 0xe000  */ \
		      "10: vlgvb %[R_TMP],%%v19,7\n\t"			\
		      "11: la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		      "    sll %[R_TMP],1\n\t"				\
		      "    lgr %[R_TMP2],%[R_TMP]\n\t"			\
		      "    ahi %[R_TMP],-1\n\t"				\
		      "    jl 20f\n\t"					\
		      "    vstl %%v17,%[R_TMP],0(%[R_OUT])\n\t"		\
		      "    ahi %[R_TMP],-16\n\t"			\
		      "    jl 19f\n\t"					\
		      "    vstl %%v18,%[R_TMP],16(%[R_OUT])\n\t"	\
		      "19: la %[R_OUT],0(%[R_TMP2],%[R_OUT])\n\t"	\
		      "20: \n\t"					\
		      ".machine pop"					\
		      : /* outputs */ [R_OUT] "+a" (outptr)		\
			, [R_IN] "+a" (inptr)				\
			, [R_TMP] "=a" (tmp)				\
			, [R_TMP2] "=a" (tmp2)				\
			, [R_LEN] "+d" (len)				\
		      : /* inputs */					\
		      : /* clobber list*/ "memory", "cc"		\
			ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")	\
			ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")	\
			ASM_CLOBBER_VR ("v20") ASM_CLOBBER_VR ("v21")	\
			ASM_CLOBBER_VR ("v22")				\
		      );						\
    if (len > 0)							\
      {									\
	/* Found an invalid character at next input-char.  */		\
	BODY_ORIG_ERROR							\
      }									\
  }
# define LOOP_NEED_FLAGS
# include <iconv/loop.c>
# include <iconv/skeleton.c>
# undef BODY_ORIG
# undef BODY_ORIG_ERROR
ICONV_VX_IFUNC (__gconv_transform_ucs2reverse_internal)

/* Convert from the internal (UCS4-like) format to UCS2.  */
#define DEFINE_INIT		0
#define DEFINE_FINI		0
#define MIN_NEEDED_FROM		4
#define MIN_NEEDED_TO		2
#define FROM_DIRECTION		1
#define FROM_LOOP		ICONV_VX_NAME (internal_ucs2_loop)
#define TO_LOOP			ICONV_VX_NAME (internal_ucs2_loop) /* This is not used.  */
#define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_internal_ucs2)
#define ONE_DIRECTION		1

#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY_ORIG							\
  {									\
    uint32_t val = *((const uint32_t *) inptr);				\
									\
    if (__glibc_unlikely (val >= 0x10000))				\
      {									\
	UNICODE_TAG_HANDLER (val, 4);					\
	STANDARD_TO_LOOP_ERR_HANDLER (4);				\
      }									\
    else if (__glibc_unlikely (val >= 0xd800 && val < 0xe000))		\
      {									\
	/* Surrogate characters in UCS-4 input are not valid.		\
	   We must catch this, because the UCS-2 output might be	\
	   interpreted as UTF-16 by other programs.  If we let		\
	   surrogates pass through, attackers could make a security	\
	   hole exploit by synthesizing any desired plane 1-16		\
	   character.  */						\
	result = __GCONV_ILLEGAL_INPUT;					\
	if (! ignore_errors_p ())					\
	  break;							\
	inptr += 4;							\
	++*irreversible;						\
	continue;							\
      }									\
    else								\
      {									\
	put16 (outptr, val);						\
	outptr += sizeof (uint16_t);					\
	inptr += 4;							\
      }									\
  }
# define BODY								\
  {									\
    if (__builtin_expect (inend - inptr < 32, 1)			\
	|| outend - outptr < 16)					\
      /* Convert remaining bytes with c code.  */			\
      BODY_ORIG								\
    else								\
      {									\
	/* Convert in 32 byte blocks.  */				\
	size_t loop_count = (inend - inptr) / 32;			\
	size_t tmp, tmp2;						\
	if (loop_count > (outend - outptr) / 16)			\
	  loop_count = (outend - outptr) / 16;				\
	__asm__ volatile (".machine push\n\t"				\
			  ".machine \"z13\"\n\t"			\
			  ".machinemode \"zarch_nohighgprs\"\n\t"	\
			  CONVERT_32BIT_SIZE_T ([R_LI])			\
			  "    larl %[R_I],3f\n\t"			\
			  "    vlm %%v20,%%v23,0(%[R_I])\n\t"		\
			  "0:  \n\t"					\
			  "    vlm %%v16,%%v17,0(%[R_IN])\n\t"		\
			  /* Shorten UCS4 to UCS2.  */			\
			  "    vpkf %%v18,%%v16,%%v17\n\t"		\
			  "    vstrcfs %%v19,%%v16,%%v20,%%v21\n\t"	\
			  "    jno 11f\n\t"				\
			  "1:  vstrcfs %%v19,%%v17,%%v20,%%v21\n\t"	\
			  "    jno 10f\n\t"				\
			  /* Store 16bytes to buf_out.  */		\
			  "2:  vst %%v18,0(%[R_OUT])\n\t"		\
			  "    la %[R_IN],32(%[R_IN])\n\t"		\
			  "    la %[R_OUT],16(%[R_OUT])\n\t"		\
			  "    brctg %[R_LI],0b\n\t"			\
			  "    j 20f\n\t"				\
			  /* Setup to check for ch >= 0xd800. (v20, v21)  */ \
			  "3:  .long 0xd800,0xd800,0x0,0x0\n\t"		\
			  "    .long 0xa0000000,0xa0000000,0x0,0x0\n\t"	\
			  /* Setup to check for ch >= 0xe000		\
			     && ch < 0x10000. (v22,v23)  */		\
			  "    .long 0xe000,0x10000,0x0,0x0\n\t"	\
			  "    .long 0xa0000000,0x40000000,0x0,0x0\n\t"	\
			  /* v16 contains only valid chars. Check in v17: \
			     ch >= 0xe000 && ch <= 0xffff.  */		\
			  "10: vstrcfs %%v19,%%v17,%%v22,%%v23,8\n\t"	\
			  "    jo 2b\n\t" /* All ch's in this range, proceed.   */ \
			  "    lghi %[R_TMP],16\n\t"			\
			  "    j 12f\n\t"				\
			  /* Maybe v16 contains invalid chars.		\
			     Check ch >= 0xe000 && ch <= 0xffff.  */	\
			  "11: vstrcfs %%v19,%%v16,%%v22,%%v23,8\n\t"	\
			  "    jo 1b\n\t" /* All ch's in this range, proceed.   */ \
			  "    lghi %[R_TMP],0\n\t"			\
			  "12: vlgvb %[R_I],%%v19,7\n\t"		\
			  "    agr %[R_I],%[R_TMP]\n\t"			\
			  "    la %[R_IN],0(%[R_I],%[R_IN])\n\t"	\
			  "    srl %[R_I],1\n\t"			\
			  "    ahi %[R_I],-1\n\t"			\
			  "    jl 20f\n\t"				\
			  "    vstl %%v18,%[R_I],0(%[R_OUT])\n\t"	\
			  "    la %[R_OUT],1(%[R_I],%[R_OUT])\n\t"	\
			  "20:\n\t"					\
			  ".machine pop"				\
			  : /* outputs */ [R_OUT] "+a" (outptr)		\
			    , [R_IN] "+a" (inptr)			\
			    , [R_LI] "+d" (loop_count)			\
			    , [R_I] "=a" (tmp2)				\
			    , [R_TMP] "=d" (tmp)			\
			  : /* inputs */				\
			  : /* clobber list*/ "memory", "cc"		\
			    ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17") \
			    ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19") \
			    ASM_CLOBBER_VR ("v20") ASM_CLOBBER_VR ("v21") \
			    ASM_CLOBBER_VR ("v22") ASM_CLOBBER_VR ("v23") \
			  );						\
	if (loop_count > 0)						\
	  {								\
	    /* Found an invalid character at next character.  */	\
	    BODY_ORIG							\
	  }								\
      }									\
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>
#include <iconv/skeleton.c>
# undef BODY_ORIG
ICONV_VX_IFUNC (__gconv_transform_internal_ucs2)

/* Convert from the internal (UCS4-like) format to UCS2 in other endianness. */
#define DEFINE_INIT		0
#define DEFINE_FINI		0
#define MIN_NEEDED_FROM		4
#define MIN_NEEDED_TO		2
#define FROM_DIRECTION		1
#define FROM_LOOP		ICONV_VX_NAME (internal_ucs2reverse_loop)
#define TO_LOOP			ICONV_VX_NAME (internal_ucs2reverse_loop)/* This is not used.*/
#define FUNCTION_NAME		ICONV_VX_NAME (__gconv_transform_internal_ucs2reverse)
#define ONE_DIRECTION		1

#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY_ORIG							\
  {									\
    uint32_t val = *((const uint32_t *) inptr);				\
    if (__glibc_unlikely (val >= 0x10000))				\
      {									\
	UNICODE_TAG_HANDLER (val, 4);					\
	STANDARD_TO_LOOP_ERR_HANDLER (4);				\
      }									\
    else if (__glibc_unlikely (val >= 0xd800 && val < 0xe000))		\
      {									\
	/* Surrogate characters in UCS-4 input are not valid.		\
	   We must catch this, because the UCS-2 output might be	\
	   interpreted as UTF-16 by other programs.  If we let		\
	   surrogates pass through, attackers could make a security	\
	   hole exploit by synthesizing any desired plane 1-16		\
	   character.  */						\
	if (! ignore_errors_p ())					\
	  {								\
	    result = __GCONV_ILLEGAL_INPUT;				\
	    break;							\
	  }								\
	inptr += 4;							\
	++*irreversible;						\
	continue;							\
      }									\
    else								\
      {									\
	put16 (outptr, bswap_16 (val));					\
	outptr += sizeof (uint16_t);					\
	inptr += 4;							\
      }									\
  }
# define BODY								\
  {									\
    if (__builtin_expect (inend - inptr < 32, 1)			\
	|| outend - outptr < 16)					\
      /* Convert remaining bytes with c code.  */			\
      BODY_ORIG								\
    else								\
      {									\
	/* Convert in 32 byte blocks.  */				\
	size_t loop_count = (inend - inptr) / 32;			\
	size_t tmp, tmp2;						\
	if (loop_count > (outend - outptr) / 16)			\
	  loop_count = (outend - outptr) / 16;				\
	__asm__ volatile (".machine push\n\t"				\
			  ".machine \"z13\"\n\t"			\
			  ".machinemode \"zarch_nohighgprs\"\n\t"	\
			  CONVERT_32BIT_SIZE_T ([R_LI])			\
			  "    larl %[R_I],3f\n\t"			\
			  "    vlm %%v20,%%v24,0(%[R_I])\n\t"		\
			  "0:  \n\t"					\
			  "    vlm %%v16,%%v17,0(%[R_IN])\n\t"		\
			  /* Shorten UCS4 to UCS2 and byteswap.  */	\
			  "    vpkf %%v18,%%v16,%%v17\n\t"		\
			  "    vperm %%v18,%%v18,%%v18,%%v24\n\t"	\
			  "    vstrcfs %%v19,%%v16,%%v20,%%v21\n\t"	\
			  "    jno 11f\n\t"				\
			  "1:  vstrcfs %%v19,%%v17,%%v20,%%v21\n\t"	\
			  "    jno 10f\n\t"				\
			  /* Store 16bytes to buf_out.  */		\
			  "2: vst %%v18,0(%[R_OUT])\n\t"		\
			  "    la %[R_IN],32(%[R_IN])\n\t"		\
			  "    la %[R_OUT],16(%[R_OUT])\n\t"		\
			  "    brctg %[R_LI],0b\n\t"			\
			  "    j 20f\n\t"				\
			  /* Setup to check for ch >= 0xd800. (v20, v21)  */ \
			  "3: .long 0xd800,0xd800,0x0,0x0\n\t"		\
			  "    .long 0xa0000000,0xa0000000,0x0,0x0\n\t"	\
			  /* Setup to check for ch >= 0xe000		\
			     && ch < 0x10000. (v22,v23)  */		\
			  "    .long 0xe000,0x10000,0x0,0x0\n\t"	\
			  "    .long 0xa0000000,0x40000000,0x0,0x0\n\t"	\
			  /* Vector permute mask (v24)  */		\
			  "    .short 0x0100,0x0302,0x0504,0x0706\n\t"	\
			  "    .short 0x0908,0x0b0a,0x0d0c,0x0f0e\n\t"	\
			  /* v16 contains only valid chars. Check in v17: \
			     ch >= 0xe000 && ch <= 0xffff.  */		\
			  "10: vstrcfs %%v19,%%v17,%%v22,%%v23,8\n\t"	\
			  "    jo 2b\n\t" /* All ch's in this range, proceed.  */ \
			  "    lghi %[R_TMP],16\n\t"			\
			  "    j 12f\n\t"				\
			  /* Maybe v16 contains invalid chars.		\
			     Check ch >= 0xe000 && ch <= 0xffff.  */	\
			  "11: vstrcfs %%v19,%%v16,%%v22,%%v23,8\n\t"	\
			  "    jo 1b\n\t" /* All ch's in this range, proceed.  */ \
			  "    lghi %[R_TMP],0\n\t"			\
			  "12: vlgvb %[R_I],%%v19,7\n\t"		\
			  "    agr %[R_I],%[R_TMP]\n\t"			\
			  "    la %[R_IN],0(%[R_I],%[R_IN])\n\t"	\
			  "    srl %[R_I],1\n\t"			\
			  "    ahi %[R_I],-1\n\t"			\
			  "    jl 20f\n\t"				\
			  "    vstl %%v18,%[R_I],0(%[R_OUT])\n\t"	\
			  "    la %[R_OUT],1(%[R_I],%[R_OUT])\n\t"	\
			  "20:\n\t"					\
			  ".machine pop"				\
			  : /* outputs */ [R_OUT] "+a" (outptr)		\
			    , [R_IN] "+a" (inptr)			\
			    , [R_LI] "+d" (loop_count)			\
			    , [R_I] "=a" (tmp2)				\
			    , [R_TMP] "=d" (tmp)			\
			  : /* inputs */				\
			  : /* clobber list*/ "memory", "cc"		\
			    ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17") \
			    ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19") \
			    ASM_CLOBBER_VR ("v20") ASM_CLOBBER_VR ("v21") \
			    ASM_CLOBBER_VR ("v22") ASM_CLOBBER_VR ("v23") \
			    ASM_CLOBBER_VR ("v24")			\
			  );						\
	if (loop_count > 0)						\
	  {								\
	    /* Found an invalid character at next character.  */	\
	    BODY_ORIG							\
	  }								\
      }									\
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>
#include <iconv/skeleton.c>
# undef BODY_ORIG
ICONV_VX_IFUNC (__gconv_transform_internal_ucs2reverse)


#else
/* Generate the internal transformations without ifunc if build environment
   lacks vector support. Instead simply include the common version.  */
# include <iconv/gconv_simple.c>
#endif /* !defined HAVE_S390_VX_ASM_SUPPORT */
