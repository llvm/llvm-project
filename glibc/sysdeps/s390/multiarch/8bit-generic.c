/* Generic conversion to and from 8bit charsets - S390 version.
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

# if defined HAVE_S390_VX_GCC_SUPPORT
#  define ASM_CLOBBER_VR(NR) , NR
# else
#  define ASM_CLOBBER_VR(NR)
# endif

/* Generate the conversion loop routines without vector instructions as
   fallback, if vector instructions aren't available at runtime.  */
# define IGNORE_ICONV_SKELETON
# define from_generic __from_generic_c
# define to_generic __to_generic_c
# include "iconvdata/8bit-generic.c"
# undef IGNORE_ICONV_SKELETON
# undef from_generic
# undef to_generic

/* Generate the converion routines with vector instructions. The vector
   routines can only be used with charsets where the maximum UCS4 value
   fits in 1 byte size. Then the hardware translate-instruction is used
   to translate between multiple generic characters and "1 byte UCS4"
   characters at once. The vector instructions are used to convert between
   the "1 byte UCS4" and UCS4.  */
# include <ifunc-resolve.h>

# undef FROM_LOOP
# undef TO_LOOP
# define FROM_LOOP		__from_generic_vx
# define TO_LOOP		__to_generic_vx

# define MIN_NEEDED_FROM	1
# define MIN_NEEDED_TO		4
# define ONE_DIRECTION		0

/* First define the conversion function from the 8bit charset to UCS4.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define LOOPFCT		FROM_LOOP
# define BODY_FROM_ORIG \
  {									      \
    uint32_t ch = to_ucs4[*inptr];					      \
									      \
    if (HAS_HOLES && __builtin_expect (ch == L'\0', 0) && *inptr != '\0')     \
      {									      \
	/* This is an illegal character.  */				      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
    ++inptr;								      \
  }

# define BODY								\
  {									\
    if (__builtin_expect (inend - inptr < 16, 1)			\
	|| outend - outptr < 64)					\
      /* Convert remaining bytes with c code.  */			\
      BODY_FROM_ORIG							\
    else								\
       {								\
	 /* Convert 16 ... 256 bytes at once with tr-instruction.  */	\
	 size_t index;							\
	 char buf[256];							\
	 size_t loop_count = (inend - inptr) / 16;			\
	 if (loop_count > (outend - outptr) / 64)			\
	   loop_count = (outend - outptr) / 64;				\
	 if (loop_count > 16)						\
	   loop_count = 16;						\
	 __asm__ volatile (".machine push\n\t"				\
			   ".machine \"z13\"\n\t"			\
			   ".machinemode \"zarch_nohighgprs\"\n\t"	\
			   "    sllk %[R_I],%[R_LI],4\n\t"		\
			   "    ahi %[R_I],-1\n\t"			\
			   /* Execute mvc and tr with correct len.  */	\
			   "    exrl %[R_I],21f\n\t"			\
			   "    exrl %[R_I],22f\n\t"			\
			   /* Post-processing.  */			\
			   "    lghi %[R_I],0\n\t"			\
			   "    vzero %%v0\n\t"				\
			   "0:  \n\t"					\
			   /* Find invalid character - value is zero.  */ \
			   "    vl %%v16,0(%[R_I],%[R_BUF])\n\t"	\
			   "    vceqbs %%v23,%%v0,%%v16\n\t"		\
			   "    jle 10f\n\t"				\
			   "1:  \n\t"					\
			   /* Enlarge to UCS4.  */			\
			   "    vuplhb %%v17,%%v16\n\t"			\
			   "    vupllb %%v18,%%v16\n\t"			\
			   "    vuplhh %%v19,%%v17\n\t"			\
			   "    vupllh %%v20,%%v17\n\t"			\
			   "    vuplhh %%v21,%%v18\n\t"			\
			   "    vupllh %%v22,%%v18\n\t"			\
			   /* Store 64bytes to buf_out.  */		\
			   "    vstm %%v19,%%v22,0(%[R_OUT])\n\t"	\
			   "    aghi %[R_I],16\n\t"			\
			   "    la %[R_OUT],64(%[R_OUT])\n\t"		\
			   "    brct %[R_LI],0b\n\t"			\
			   "    la %[R_IN],0(%[R_I],%[R_IN])\n\t"	\
			   "    j 20f\n\t"				\
			   "21: mvc 0(1,%[R_BUF]),0(%[R_IN])\n\t"	\
			   "22: tr 0(1,%[R_BUF]),0(%[R_TBL])\n\t"	\
			   /* Possibly invalid character found.  */	\
			   "10: \n\t"					\
			   /* Test if input was zero, too.  */		\
			   "    vl %%v24,0(%[R_I],%[R_IN])\n\t"		\
			   "    vceqb %%v24,%%v0,%%v24\n\t"		\
			   /* Zeros in buf (v23) and inptr (v24) are marked \
			      with one bits. After xor, invalid characters \
			      are marked as one bits. Proceed, if no	\
			      invalid characters are found.  */		\
			   "    vx %%v24,%%v23,%%v24\n\t"		\
			   "    vfenebs %%v24,%%v24,%%v0\n\t"		\
			   "    jo 1b\n\t"				\
			   /* Found an invalid translation.		\
			      Store the preceding chars.  */		\
			   "    la %[R_IN],0(%[R_I],%[R_IN])\n\t"	\
			   "    vlgvb %[R_I],%%v24,7\n\t"		\
			   "    la %[R_IN],0(%[R_I],%[R_IN])\n\t"	\
			   "    sll %[R_I],2\n\t"			\
			   "    ahi %[R_I],-1\n\t"			\
			   "    jl 20f\n\t"				\
			   "    lgr %[R_LI],%[R_I]\n\t"			\
			   "    vuplhb %%v17,%%v16\n\t"			\
			   "    vuplhh %%v19,%%v17\n\t"			\
			   "    vstl %%v19,%[R_I],0(%[R_OUT])\n\t"	\
			   "    ahi %[R_I],-16\n\t"			\
			   "    jl 11f\n\t"				\
			   "    vupllh %%v20,%%v17\n\t"			\
			   "    vstl %%v20,%[R_I],16(%[R_OUT])\n\t"	\
			   "    ahi %[R_I],-16\n\t"			\
			   "    jl 11f\n\t"				\
			   "    vupllb %%v18,%%v16\n\t"			\
			   "    vuplhh %%v21,%%v18\n\t"			\
			   "    vstl %%v21,%[R_I],32(%[R_OUT])\n\t"	\
			   "    ahi %[R_I],-16\n\t"			\
			   "    jl 11f\n\t"				\
			   "    vupllh %%v22,%%v18\n\t"			\
			   "    vstl %%v22,%[R_I],48(%[R_OUT])\n\t"	\
			   "11: \n\t"					\
			   "    la %[R_OUT],1(%[R_LI],%[R_OUT])\n\t"	\
			   "20: \n\t"					\
			   ".machine pop"				\
			   : /* outputs */ [R_IN] "+a" (inptr)		\
			     , [R_OUT] "+a" (outptr), [R_I] "=&a" (index) \
			     , [R_LI] "+a" (loop_count)			\
			   : /* inputs */ [R_BUF] "a" (buf)		\
			     , [R_TBL] "a" (to_ucs1)			\
			   : /* clobber list*/ "memory", "cc"		\
			     ASM_CLOBBER_VR ("v0")  ASM_CLOBBER_VR ("v16") \
			     ASM_CLOBBER_VR ("v17") ASM_CLOBBER_VR ("v18") \
			     ASM_CLOBBER_VR ("v19") ASM_CLOBBER_VR ("v20") \
			     ASM_CLOBBER_VR ("v21") ASM_CLOBBER_VR ("v22") \
			     ASM_CLOBBER_VR ("v23") ASM_CLOBBER_VR ("v24") \
			   );						\
	 /* Error occured?  */						\
	 if (loop_count != 0)						\
	   {								\
	     /* Found an invalid character!  */				\
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				\
	  }								\
      }									\
    }

# define LOOP_NEED_FLAGS
# include <iconv/loop.c>

/* Next, define the other direction - from UCS4 to 8bit charset.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_TO
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
# define LOOPFCT		TO_LOOP
# define BODY_TO_ORIG \
  {									      \
    uint32_t ch = get32 (inptr);					      \
									      \
    if (__builtin_expect (ch >= sizeof (from_ucs4) / sizeof (from_ucs4[0]), 0)\
	|| (__builtin_expect (from_ucs4[ch], '\1') == '\0' && ch != 0))	      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
									      \
	/* This is an illegal character.  */				      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
									      \
    *outptr++ = from_ucs4[ch];						      \
    inptr += 4;								      \
  }
# define BODY								\
  {									\
    if (__builtin_expect (inend - inptr < 64, 1)			\
	|| outend - outptr < 16)					\
      /* Convert remaining bytes with c code.  */			\
      BODY_TO_ORIG							\
    else								\
      {									\
	/* Convert 64 ... 1024 bytes at once with tr-instruction.  */	\
	size_t index, tmp;						\
	char buf[256];							\
	size_t loop_count = (inend - inptr) / 64;			\
	uint32_t max = sizeof (from_ucs4) / sizeof (from_ucs4[0]);	\
	if (loop_count > (outend - outptr) / 16)			\
	  loop_count = (outend - outptr) / 16;				\
	if (loop_count > 16)						\
	  loop_count = 16;						\
	size_t remaining_loop_count = loop_count;			\
	/* Step 1: Check for ch>=max, ch == 0 and shorten to bytes.	\
	   (ch == 0 is no error, but is handled differently)  */	\
	__asm__ volatile (".machine push\n\t"				\
			  ".machine \"z13\"\n\t"			\
			  ".machinemode \"zarch_nohighgprs\"\n\t"	\
			  /* Setup to check for ch >= max.  */		\
			  "    vzero %%v21\n\t"				\
			  "    vleih %%v21,-24576,0\n\t" /* element 0:   >  */ \
			  "    vleih %%v21,-8192,2\n\t"  /* element 1: =<>  */ \
			  "    vlvgf %%v20,%[R_MAX],0\n\t" /* element 0: val  */ \
			  /* Process in 64byte - 16 characters blocks.  */ \
			  "    lghi %[R_I],0\n\t"			\
			  "    lghi %[R_TMP],0\n\t"			\
			  "0:  \n\t"					\
			  "    vlm %%v16,%%v19,0(%[R_IN])\n\t"		\
			  /* Test for ch >= max and ch == 0.  */	\
			  "    vstrczfs %%v22,%%v16,%%v20,%%v21\n\t"	\
			  "    jno 10f\n\t"				\
			  "    vstrczfs %%v22,%%v17,%%v20,%%v21\n\t"	\
			  "    jno 11f\n\t"				\
			  "    vstrczfs %%v22,%%v18,%%v20,%%v21\n\t"	\
			  "    jno 12f\n\t"				\
			  "    vstrczfs %%v22,%%v19,%%v20,%%v21\n\t"	\
			  "    jno 13f\n\t"				\
			  /* Shorten to byte values.  */		\
			  "    vpkf %%v16,%%v16,%%v17\n\t"		\
			  "    vpkf %%v18,%%v18,%%v19\n\t"		\
			  "    vpkh %%v16,%%v16,%%v18\n\t"		\
			  /* Store 16bytes to buf.  */			\
			  "    vst %%v16,0(%[R_I],%[R_BUF])\n\t"	\
			  /* Loop until all blocks are processed.  */	\
			  "    la %[R_IN],64(%[R_IN])\n\t"		\
			  "    aghi %[R_I],16\n\t"			\
			  "    brct %[R_LI],0b\n\t"			\
			  "    j 20f\n\t"				\
			  /* Found error ch >= max or ch == 0. */	\
			  "13: aghi %[R_TMP],4\n\t"			\
			  "12: aghi %[R_TMP],4\n\t"			\
			  "11: aghi %[R_TMP],4\n\t"			\
			  "10: vlgvb %[R_I],%%v22,7\n\t"		\
			  "    srlg %[R_I],%[R_I],2\n\t"		\
			  "    agr %[R_I],%[R_TMP]\n\t"			\
			  "20: \n\t"					\
			  ".machine pop"				\
			  : /* outputs */ [R_IN] "+a" (inptr)		\
			    , [R_I] "=&a" (index)			\
			    , [R_TMP] "=d" (tmp)			\
			    , [R_LI] "+d" (remaining_loop_count)	\
			  : /* inputs */ [R_BUF] "a" (buf)		\
			    , [R_MAX] "d" (max)				\
			  : /* clobber list*/ "memory", "cc"		\
			    ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17") \
			    ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19") \
			    ASM_CLOBBER_VR ("v20") ASM_CLOBBER_VR ("v21") \
			    ASM_CLOBBER_VR ("v22")			\
			  );						\
	/* Error occured in step 1? An error (ch >= max || ch == 0)	\
	   occured, if remaining_loop_count > 0. The error occured	\
	   at character-index (index) after already processed blocks.  */ \
	loop_count -= remaining_loop_count;				\
	if (loop_count > 0)						\
	  {								\
	    /* Step 2: Translate already processed blocks in buf and	\
	       check for errors (from_ucs4[ch] == 0).  */		\
	    __asm__ volatile (".machine push\n\t"			\
			      ".machine \"z13\"\n\t"			\
			      ".machinemode \"zarch_nohighgprs\"\n\t"	\
			      "    sllk %[R_I],%[R_LI],4\n\t"		\
			      "    ahi %[R_I],-1\n\t"			\
			      /* Execute tr with correct len.  */	\
			      "    exrl %[R_I],21f\n\t"			\
			      /* Post-processing.  */			\
			      "    lghi %[R_I],0\n\t"			\
			      "0:  \n\t"				\
			      /* Find invalid character - value == 0.  */ \
			      "    vl %%v16,0(%[R_I],%[R_BUF])\n\t"	\
			      "    vfenezbs %%v17,%%v16,%%v16\n\t"	\
			      "    je 10f\n\t"				\
			      /* Store 16bytes to buf_out.  */		\
			      "    vst %%v16,0(%[R_I],%[R_OUT])\n\t"	\
			      "    aghi %[R_I],16\n\t"			\
			      "    brct %[R_LI],0b\n\t"			\
			      "    la %[R_OUT],0(%[R_I],%[R_OUT])\n\t"	\
			      "    j 20f\n\t"				\
			      "21: tr 0(1,%[R_BUF]),0(%[R_TBL])\n\t"	\
			      /* Found an error: from_ucs4[ch] == 0.  */ \
			      "10: la %[R_OUT],0(%[R_I],%[R_OUT])\n\t"	\
			      "    vlgvb %[R_I],%%v17,7\n\t"		\
			      "20: \n\t"				\
			      ".machine pop"				\
			      : /* outputs */ [R_OUT] "+a" (outptr)	\
				, [R_I] "=&a" (tmp)			\
				, [R_LI] "+d" (loop_count)		\
			      : /* inputs */ [R_BUF] "a" (buf)		\
				, [R_TBL] "a" (from_ucs4)		\
			      : /* clobber list*/ "memory", "cc"	\
				ASM_CLOBBER_VR ("v16")			\
				ASM_CLOBBER_VR ("v17")			\
			      );					\
	    /* Error occured in processed bytes of step 2?		\
	       Thus possible error in step 1 is obselete.*/		\
	    if (tmp < 16)						\
	      {								\
		index = tmp;						\
		inptr -= loop_count * 64;				\
	      }								\
	  }								\
	/* Error occured in step 1/2?  */				\
	if (index < 16)							\
	  {								\
	    /* Found an invalid character (see step 2) or zero		\
	       (see step 1) at index! Convert the chars before index	\
	       manually. If there is a zero at index detected by step 1, \
	       there could be invalid characters before this zero.  */	\
	    int i;							\
	    uint32_t ch;						\
	    for (i = 0; i < index; i++)					\
	      {								\
		ch = get32 (inptr);					\
		if (__builtin_expect (from_ucs4[ch], '\1') == '\0')     \
		  break;						\
		*outptr++ = from_ucs4[ch];				\
		inptr += 4;						\
	      }								\
	    if (i == index)						\
	      {								\
		ch = get32 (inptr);					\
		if (ch == 0)						\
		  {							\
		    /* This is no error, but handled differently.  */	\
		    *outptr++ = from_ucs4[ch];				\
		    inptr += 4;						\
		    continue;						\
		  }							\
	      }								\
									\
	    /* iconv/loop.c disables -Wmaybe-uninitialized for a false	\
	       positive warning in this code with -Os and has a		\
	       comment referencing this code accordingly.  Updates in	\
	       one place may require updates in the other.  */		\
	    UNICODE_TAG_HANDLER (ch, 4);				\
									\
	    /* This is an illegal character.  */			\
	    STANDARD_TO_LOOP_ERR_HANDLER (4);				\
	  }								\
      }									\
  }

# define LOOP_NEED_FLAGS
# include <iconv/loop.c>


/* Generate ifunc'ed loop function.  */
s390_libc_ifunc_expr (__from_generic_c, __from_generic,
		      (sizeof (from_ucs4) / sizeof (from_ucs4[0]) <= 256
		       && hwcap & HWCAP_S390_VX)
		      ? __from_generic_vx
		      : __from_generic_c);

s390_libc_ifunc_expr (__to_generic_c, __to_generic,
		      (sizeof (from_ucs4) / sizeof (from_ucs4[0]) <= 256
		       && hwcap & HWCAP_S390_VX)
		      ? __to_generic_vx
		      : __to_generic_c);

strong_alias (__to_generic_c_single, __to_generic_single)

# undef FROM_LOOP
# undef TO_LOOP
# define FROM_LOOP		__from_generic
# define TO_LOOP		__to_generic
# include <iconv/skeleton.c>

#else
/* Generate this module without ifunc if build environment lacks vector
   support.  Instead the common 8bit-generic.c is used.  */
# include "iconvdata/8bit-generic.c"
#endif /* !defined HAVE_S390_VX_ASM_SUPPORT */
