/* Conversion between UTF-16 and UTF-32 BE/internal.

   This module uses the Z9-109 variants of the Convert Unicode
   instructions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.

   Author: Andreas Krebbel  <Andreas.Krebbel@de.ibm.com>
   Based on the work by Ulrich Drepper  <drepper@cygnus.com>, 1997.

   Thanks to Daniel Appich who covered the relevant performance work
   in his diploma thesis.

   This is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <dlfcn.h>
#include <stdint.h>
#include <unistd.h>
#include <gconv.h>
#include <string.h>

/* Select which versions should be defined depending on support
   for multiarch, vector and used minimum architecture level.  */
#define HAVE_FROM_C		1
#define FROM_LOOP_DEFAULT	FROM_LOOP_C
#define HAVE_TO_C		1
#define TO_LOOP_DEFAULT		TO_LOOP_C

#if defined HAVE_S390_VX_ASM_SUPPORT && defined USE_MULTIARCH
# define HAVE_FROM_VX		1
# define HAVE_FROM_VX_CU	1
# define HAVE_TO_VX		1
# define HAVE_TO_VX_CU		1
#else
# define HAVE_FROM_VX		0
# define HAVE_FROM_VX_CU	0
# define HAVE_TO_VX		0
# define HAVE_TO_VX_CU		0
#endif

#if defined HAVE_S390_VX_GCC_SUPPORT
# define ASM_CLOBBER_VR(NR) , NR
#else
# define ASM_CLOBBER_VR(NR)
#endif

#if defined __s390x__
# define CONVERT_32BIT_SIZE_T(REG)
#else
# define CONVERT_32BIT_SIZE_T(REG) "llgfr %" #REG ",%" #REG "\n\t"
#endif

/* UTF-32 big endian byte order mark.  */
#define BOM_UTF32               0x0000feffu

/* UTF-16 big endian byte order mark.  */
#define BOM_UTF16               0xfeff

#define DEFINE_INIT		0
#define DEFINE_FINI		0
#define MIN_NEEDED_FROM		2
#define MAX_NEEDED_FROM		4
#define MIN_NEEDED_TO		4
#define FROM_LOOP		FROM_LOOP_DEFAULT
#define TO_LOOP			TO_LOOP_DEFAULT
#define FROM_DIRECTION		(dir == from_utf16)
#define ONE_DIRECTION           0

/* Direction of the transformation.  */
enum direction
{
  illegal_dir,
  to_utf16,
  from_utf16
};

struct utf16_data
{
  enum direction dir;
  int emit_bom;
};


extern int gconv_init (struct __gconv_step *step);
int
gconv_init (struct __gconv_step *step)
{
  /* Determine which direction.  */
  struct utf16_data *new_data;
  enum direction dir = illegal_dir;
  int emit_bom;
  int result;

  emit_bom = (__strcasecmp (step->__to_name, "UTF-32//") == 0
	      || __strcasecmp (step->__to_name, "UTF-16//") == 0);

  if (__strcasecmp (step->__from_name, "UTF-16BE//") == 0
      && (__strcasecmp (step->__to_name, "UTF-32//") == 0
	  || __strcasecmp (step->__to_name, "UTF-32BE//") == 0
	  || __strcasecmp (step->__to_name, "INTERNAL") == 0))
    {
      dir = from_utf16;
    }
  else if ((__strcasecmp (step->__to_name, "UTF-16//") == 0
	    || __strcasecmp (step->__to_name, "UTF-16BE//") == 0)
	   && (__strcasecmp (step->__from_name, "UTF-32BE//") == 0
	       || __strcasecmp (step->__from_name, "INTERNAL") == 0))
    {
      dir = to_utf16;
    }

  result = __GCONV_NOCONV;
  if (dir != illegal_dir)
    {
      new_data = (struct utf16_data *) malloc (sizeof (struct utf16_data));

      result = __GCONV_NOMEM;
      if (new_data != NULL)
	{
	  new_data->dir = dir;
	  new_data->emit_bom = emit_bom;
	  step->__data = new_data;

	  if (dir == from_utf16)
	    {
	      step->__min_needed_from = MIN_NEEDED_FROM;
	      step->__max_needed_from = MIN_NEEDED_FROM;
	      step->__min_needed_to = MIN_NEEDED_TO;
	      step->__max_needed_to = MIN_NEEDED_TO;
	    }
	  else
	    {
	      step->__min_needed_from = MIN_NEEDED_TO;
	      step->__max_needed_from = MIN_NEEDED_TO;
	      step->__min_needed_to = MIN_NEEDED_FROM;
	      step->__max_needed_to = MIN_NEEDED_FROM;
	    }

	  step->__stateful = 0;

	  result = __GCONV_OK;
	}
    }

  return result;
}


extern void gconv_end (struct __gconv_step *data);
void
gconv_end (struct __gconv_step *data)
{
  free (data->__data);
}

#define PREPARE_LOOP							\
  enum direction dir = ((struct utf16_data *) step->__data)->dir;	\
  int emit_bom = ((struct utf16_data *) step->__data)->emit_bom;	\
									\
  if (emit_bom && !data->__internal_use					\
      && data->__invocation_counter == 0)				\
    {									\
      if (dir == to_utf16)						\
	{								\
	  /* Emit the UTF-16 Byte Order Mark.  */			\
	  if (__glibc_unlikely (outbuf + 2 > outend))			\
	    return __GCONV_FULL_OUTPUT;					\
									\
	  put16u (outbuf, BOM_UTF16);					\
	  outbuf += 2;							\
	}								\
      else								\
	{								\
	  /* Emit the UTF-32 Byte Order Mark.  */			\
	  if (__glibc_unlikely (outbuf + 4 > outend))			\
	    return __GCONV_FULL_OUTPUT;					\
									\
	  put32u (outbuf, BOM_UTF32);					\
	  outbuf += 4;							\
	}								\
    }

/* Conversion function from UTF-16 to UTF-32 internal/BE.  */

#if HAVE_FROM_C == 1
/* The software routine is copied from utf-16.c (minus bytes
   swapping).  */
# define BODY_FROM_C							\
  {									\
    uint16_t u1 = get16 (inptr);					\
									\
    if (__builtin_expect (u1 < 0xd800, 1) || u1 > 0xdfff)		\
      {									\
	/* No surrogate.  */						\
	put32 (outptr, u1);						\
	inptr += 2;							\
      }									\
    else								\
      {									\
	/* An isolated low-surrogate was found.  This has to be         \
	   considered ill-formed.  */					\
	if (__glibc_unlikely (u1 >= 0xdc00))				\
	  {								\
	    STANDARD_FROM_LOOP_ERR_HANDLER (2);				\
	  }								\
	/* It's a surrogate character.  At least the first word says	\
	   it is.  */							\
	if (__glibc_unlikely (inptr + 4 > inend))			\
	  {								\
	    /* We don't have enough input for another complete input	\
	       character.  */						\
	    result = __GCONV_INCOMPLETE_INPUT;				\
	    break;							\
	  }								\
									\
	inptr += 2;							\
	uint16_t u2 = get16 (inptr);					\
	if (__builtin_expect (u2 < 0xdc00, 0)				\
	    || __builtin_expect (u2 > 0xdfff, 0))			\
	  {								\
	    /* This is no valid second word for a surrogate.  */	\
	    inptr -= 2;							\
	    STANDARD_FROM_LOOP_ERR_HANDLER (2);				\
	  }								\
									\
	put32 (outptr, ((u1 - 0xd7c0) << 10) + (u2 - 0xdc00));		\
	inptr += 2;							\
      }									\
    outptr += 4;							\
  }


/* Generate loop-function with software routing.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define FROM_LOOP_C		__from_utf16_loop_c
# define LOOPFCT		FROM_LOOP_C
# define LOOP_NEED_FLAGS
# define BODY			BODY_FROM_C
# include <iconv/loop.c>
#else
# define FROM_LOOP_C		NULL
#endif /* HAVE_FROM_C != 1  */

#if HAVE_FROM_VX == 1
# define BODY_FROM_VX							\
  {									\
    size_t inlen = inend - inptr;					\
    size_t outlen = outend - outptr;					\
    unsigned long tmp, tmp2, tmp3;					\
    asm volatile (".machine push\n\t"					\
		  ".machine \"z13\"\n\t"				\
		  ".machinemode \"zarch_nohighgprs\"\n\t"		\
		  /* Setup to check for surrogates.  */			\
		  "    larl %[R_TMP],9f\n\t"				\
		  "    vlm %%v30,%%v31,0(%[R_TMP])\n\t"			\
		  CONVERT_32BIT_SIZE_T ([R_INLEN])			\
		  CONVERT_32BIT_SIZE_T ([R_OUTLEN])			\
		  /* Loop which handles UTF-16 chars <0xd800, >0xdfff.  */ \
		  "0:  clgijl %[R_INLEN],16,2f\n\t"			\
		  "    clgijl %[R_OUTLEN],32,2f\n\t"			\
		  "1:  vl %%v16,0(%[R_IN])\n\t"				\
		  /* Check for surrogate chars.  */			\
		  "    vstrchs %%v19,%%v16,%%v30,%%v31\n\t"		\
		  "    jno 10f\n\t"					\
		  /* Enlarge to UTF-32.  */				\
		  "    vuplhh %%v17,%%v16\n\t"				\
		  "    la %[R_IN],16(%[R_IN])\n\t"			\
		  "    vupllh %%v18,%%v16\n\t"				\
		  "    aghi %[R_INLEN],-16\n\t"				\
		  /* Store 32 bytes to buf_out.  */			\
		  "    vstm %%v17,%%v18,0(%[R_OUT])\n\t"		\
		  "    aghi %[R_OUTLEN],-32\n\t"			\
		  "    la %[R_OUT],32(%[R_OUT])\n\t"			\
		  "    clgijl %[R_INLEN],16,2f\n\t"			\
		  "    clgijl %[R_OUTLEN],32,2f\n\t"			\
		  "    j 1b\n\t"					\
		  /* Setup to check for ch >= 0xd800 && ch <= 0xdfff. (v30, v31)  */ \
		  "9:  .short 0xd800,0xdfff,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		  "    .short 0xa000,0xc000,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		  /* At least one uint16_t is in range of surrogates.	\
		     Store the preceding chars.  */			\
		  "10: vlgvb %[R_TMP],%%v19,7\n\t"			\
		  "    vuplhh %%v17,%%v16\n\t"				\
		  "    sllg %[R_TMP3],%[R_TMP],1\n\t" /* Number of out bytes.  */ \
		  "    ahik %[R_TMP2],%[R_TMP3],-1\n\t" /* Highest index to store.  */ \
		  "    jl 12f\n\t"					\
		  "    vstl %%v17,%[R_TMP2],0(%[R_OUT])\n\t"		\
		  "    vupllh %%v18,%%v16\n\t"				\
		  "    ahi %[R_TMP2],-16\n\t"				\
		  "    jl 11f\n\t"					\
		  "    vstl %%v18,%[R_TMP2],16(%[R_OUT])\n\t"		\
		  "11: \n\t" /* Update pointers.  */			\
		  "    la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		  "    slgr %[R_INLEN],%[R_TMP]\n\t"			\
		  "    la %[R_OUT],0(%[R_TMP3],%[R_OUT])\n\t"		\
		  "    slgr %[R_OUTLEN],%[R_TMP3]\n\t"			\
		  /* Calculate remaining uint16_t values in loaded vrs.  */ \
		  "12: lghi %[R_TMP2],16\n\t"				\
		  "    slgr %[R_TMP2],%[R_TMP]\n\t"			\
		  "    srl %[R_TMP2],1\n\t"				\
		  "    llh %[R_TMP],0(%[R_IN])\n\t"			\
		  "    aghi %[R_OUTLEN],-4\n\t"				\
		  "    j 16f\n\t"					\
		  /* Handle remaining bytes.  */			\
		  "2:  \n\t"						\
		  /* Zero, one or more bytes available?  */		\
		  "    clgfi %[R_INLEN],1\n\t"				\
		  "    je 97f\n\t" /* Only one byte available.  */	\
		  "    jl 99f\n\t" /* End if no bytes available.  */	\
		  /* Calculate remaining uint16_t values in inptr.  */	\
		  "    srlg %[R_TMP2],%[R_INLEN],1\n\t"			\
		  /* Handle remaining uint16_t values.  */		\
		  "13: llh %[R_TMP],0(%[R_IN])\n\t"			\
		  "    slgfi %[R_OUTLEN],4\n\t"				\
		  "    jl 96f \n\t"					\
		  "    clfi %[R_TMP],0xd800\n\t"			\
		  "    jhe 15f\n\t"					\
		  "14: st %[R_TMP],0(%[R_OUT])\n\t"			\
		  "    la %[R_IN],2(%[R_IN])\n\t"			\
		  "    aghi %[R_INLEN],-2\n\t"				\
		  "    la %[R_OUT],4(%[R_OUT])\n\t"			\
		  "    brctg %[R_TMP2],13b\n\t"				\
		  "    j 0b\n\t" /* Switch to vx-loop.  */		\
		  /* Handle UTF-16 surrogate pair.  */			\
		  "15: clfi %[R_TMP],0xdfff\n\t"			\
		  "    jh 14b\n\t" /* Jump away if ch > 0xdfff.  */	\
		  "16: clfi %[R_TMP],0xdc00\n\t"			\
		  "    jhe 98f\n\t" /* Jump away in case of low-surrogate.  */ \
		  "    slgfi %[R_INLEN],4\n\t"				\
		  "    jl 97f\n\t" /* Big enough input?  */		\
		  "    llh %[R_TMP3],2(%[R_IN])\n\t" /* Load low surrogate.  */ \
		  "    slfi %[R_TMP],0xd7c0\n\t"			\
		  "    sll %[R_TMP],10\n\t"				\
		  "    risbgn %[R_TMP],%[R_TMP3],54,63,0\n\t" /* Insert klmnopqrst.  */ \
		  "    nilf %[R_TMP3],0xfc00\n\t"			\
		  "    clfi %[R_TMP3],0xdc00\n\t" /* Check if it starts with 0xdc00.  */ \
		  "    jne 98f\n\t"					\
		  "    st %[R_TMP],0(%[R_OUT])\n\t"			\
		  "    la %[R_IN],4(%[R_IN])\n\t"			\
		  "    la %[R_OUT],4(%[R_OUT])\n\t"			\
		  "    aghi %[R_TMP2],-2\n\t"				\
		  "    jh 13b\n\t" /* Handle remaining uint16_t values.  */ \
		  "    j 0b\n\t" /* Switch to vx-loop.  */		\
		  "96: \n\t" /* Return full output.  */			\
		  "    lghi %[R_RES],%[RES_OUT_FULL]\n\t"		\
		  "    j 99f\n\t"					\
		  "97: \n\t" /* Return incomplete input.  */		\
		  "    lghi %[R_RES],%[RES_IN_FULL]\n\t"		\
		  "    j 99f\n\t"					\
		  "98:\n\t" /* Return Illegal character.  */		\
		  "    lghi %[R_RES],%[RES_IN_ILL]\n\t"			\
		  "99:\n\t"						\
		  ".machine pop"					\
		  : /* outputs */ [R_IN] "+a" (inptr)			\
		    , [R_INLEN] "+d" (inlen), [R_OUT] "+a" (outptr)	\
		    , [R_OUTLEN] "+d" (outlen), [R_TMP] "=a" (tmp)	\
		    , [R_TMP2] "=d" (tmp2), [R_TMP3] "=a" (tmp3)	\
		    , [R_RES] "+d" (result)				\
		  : /* inputs */					\
		    [RES_OUT_FULL] "i" (__GCONV_FULL_OUTPUT)		\
		    , [RES_IN_ILL] "i" (__GCONV_ILLEGAL_INPUT)		\
		    , [RES_IN_FULL] "i" (__GCONV_INCOMPLETE_INPUT)	\
		  : /* clobber list */ "memory", "cc"			\
		    ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")	\
		    ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")	\
		    ASM_CLOBBER_VR ("v30") ASM_CLOBBER_VR ("v31")	\
		  );							\
    if (__glibc_likely (inptr == inend)					\
	|| result != __GCONV_ILLEGAL_INPUT)				\
      break;								\
									\
    STANDARD_FROM_LOOP_ERR_HANDLER (2);					\
  }

/* Generate loop-function with hardware vector instructions.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define FROM_LOOP_VX		__from_utf16_loop_vx
# define LOOPFCT		FROM_LOOP_VX
# define LOOP_NEED_FLAGS
# define BODY			BODY_FROM_VX
# include <iconv/loop.c>
#else
# define FROM_LOOP_VX		NULL
#endif /* HAVE_FROM_VX != 1  */

#if HAVE_FROM_VX_CU == 1
#define BODY_FROM_VX_CU							\
  {									\
    register const unsigned char* pInput asm ("8") = inptr;		\
    register size_t inlen asm ("9") = inend - inptr;			\
    register unsigned char* pOutput asm ("10") = outptr;		\
    register size_t outlen asm ("11") = outend - outptr;		\
    unsigned long tmp, tmp2, tmp3;					\
    asm volatile (".machine push\n\t"					\
		  ".machine \"z13\"\n\t"				\
		  ".machinemode \"zarch_nohighgprs\"\n\t"		\
		  /* Setup to check for surrogates.  */			\
		  "    larl %[R_TMP],9f\n\t"				\
		  "    vlm %%v30,%%v31,0(%[R_TMP])\n\t"			\
		  CONVERT_32BIT_SIZE_T ([R_INLEN])			\
		  CONVERT_32BIT_SIZE_T ([R_OUTLEN])			\
		  /* Loop which handles UTF-16 chars <0xd800, >0xdfff.  */ \
		  "0:  clgijl %[R_INLEN],16,20f\n\t"			\
		  "    clgijl %[R_OUTLEN],32,20f\n\t"			\
		  "1:  vl %%v16,0(%[R_IN])\n\t"				\
		  /* Check for surrogate chars.  */			\
		  "    vstrchs %%v19,%%v16,%%v30,%%v31\n\t"		\
		  "    jno 10f\n\t"					\
		  /* Enlarge to UTF-32.  */				\
		  "    vuplhh %%v17,%%v16\n\t"				\
		  "    la %[R_IN],16(%[R_IN])\n\t"			\
		  "    vupllh %%v18,%%v16\n\t"				\
		  "    aghi %[R_INLEN],-16\n\t"				\
		  /* Store 32 bytes to buf_out.  */			\
		  "    vstm %%v17,%%v18,0(%[R_OUT])\n\t"		\
		  "    aghi %[R_OUTLEN],-32\n\t"			\
		  "    la %[R_OUT],32(%[R_OUT])\n\t"			\
		  "    clgijl %[R_INLEN],16,20f\n\t"			\
		  "    clgijl %[R_OUTLEN],32,20f\n\t"			\
		  "    j 1b\n\t"					\
		  /* Setup to check for ch >= 0xd800 && ch <= 0xdfff. (v30, v31)  */ \
		  "9:  .short 0xd800,0xdfff,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		  "    .short 0xa000,0xc000,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		  /* At least one uint16_t is in range of surrogates.	\
		     Store the preceding chars.  */			\
		  "10: vlgvb %[R_TMP],%%v19,7\n\t"			\
		  "    vuplhh %%v17,%%v16\n\t"				\
		  "    sllg %[R_TMP3],%[R_TMP],1\n\t" /* Number of out bytes.  */ \
		  "    ahik %[R_TMP2],%[R_TMP3],-1\n\t" /* Highest index to store.  */ \
		  "    jl 20f\n\t"					\
		  "    vstl %%v17,%[R_TMP2],0(%[R_OUT])\n\t"		\
		  "    vupllh %%v18,%%v16\n\t"				\
		  "    ahi %[R_TMP2],-16\n\t"				\
		  "    jl 11f\n\t"					\
		  "    vstl %%v18,%[R_TMP2],16(%[R_OUT])\n\t"		\
		  "11: \n\t" /* Update pointers.  */			\
		  "    la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		  "    slgr %[R_INLEN],%[R_TMP]\n\t"			\
		  "    la %[R_OUT],0(%[R_TMP3],%[R_OUT])\n\t"		\
		  "    slgr %[R_OUTLEN],%[R_TMP3]\n\t"			\
		  /* Handles UTF16 surrogates with convert instruction.  */ \
		  "20: cu24 %[R_OUT],%[R_IN],1\n\t"			\
		  "    jo 0b\n\t" /* Try vector implemenation again.  */ \
		  "    lochil %[R_RES],%[RES_OUT_FULL]\n\t" /* cc == 1.  */ \
		  "    lochih %[R_RES],%[RES_IN_ILL]\n\t" /* cc == 2.  */ \
		  ".machine pop"					\
		  : /* outputs */ [R_IN] "+a" (pInput)			\
		    , [R_INLEN] "+d" (inlen), [R_OUT] "+a" (pOutput)	\
		    , [R_OUTLEN] "+d" (outlen), [R_TMP] "=a" (tmp)	\
		    , [R_TMP2] "=d" (tmp2), [R_TMP3] "=a" (tmp3)	\
		    , [R_RES] "+d" (result)				\
		  : /* inputs */					\
		    [RES_OUT_FULL] "i" (__GCONV_FULL_OUTPUT)		\
		    , [RES_IN_ILL] "i" (__GCONV_ILLEGAL_INPUT)		\
		  : /* clobber list */ "memory", "cc"			\
		    ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")	\
		    ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")	\
		    ASM_CLOBBER_VR ("v30") ASM_CLOBBER_VR ("v31")	\
		  );							\
    inptr = pInput;							\
    outptr = pOutput;							\
									\
    if (__glibc_likely (inlen == 0)					\
	|| result == __GCONV_FULL_OUTPUT)				\
      break;								\
    if (inlen == 1)							\
      {									\
	/* Input does not contain a complete utf16 character.  */	\
	result = __GCONV_INCOMPLETE_INPUT;				\
	break;								\
      }									\
    else if (result != __GCONV_ILLEGAL_INPUT)				\
      {									\
	/* Input is >= 2 and < 4 bytes (as cu24 would have processed	\
	   a possible next utf16 character) and not illegal.		\
	   => we have a single high surrogate at end of input.  */	\
	result = __GCONV_INCOMPLETE_INPUT;				\
	break;								\
      }									\
									\
    STANDARD_FROM_LOOP_ERR_HANDLER (2);					\
  }

/* Generate loop-function with hardware vector and utf-convert instructions.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define FROM_LOOP_VX_CU	__from_utf16_loop_vx_cu
# define LOOPFCT		FROM_LOOP_VX_CU
# define LOOP_NEED_FLAGS
# define BODY			BODY_FROM_VX_CU
# include <iconv/loop.c>
#else
# define FROM_LOOP_VX_CU	NULL
#endif /* HAVE_FROM_VX_CU != 1  */

/* Conversion from UTF-32 internal/BE to UTF-16.  */

#if HAVE_TO_C == 1
/* The software routine is copied from utf-16.c (minus bytes
   swapping).  */
# define BODY_TO_C							\
  {									\
    uint32_t c = get32 (inptr);						\
									\
    if (__builtin_expect (c <= 0xd7ff, 1)				\
	|| (c > 0xdfff && c <= 0xffff))					\
      {									\
	/* Two UTF-16 chars.  */					\
	put16 (outptr, c);						\
      }									\
    else if (__builtin_expect (c >= 0x10000, 1)				\
	     && __builtin_expect (c <= 0x10ffff, 1))			\
      {									\
	/* Four UTF-16 chars.  */					\
	uint16_t zabcd = ((c & 0x1f0000) >> 16) - 1;			\
	uint16_t out;							\
									\
	/* Generate a surrogate character.  */				\
	if (__glibc_unlikely (outptr + 4 > outend))			\
	  {								\
	    /* Overflow in the output buffer.  */			\
	    result = __GCONV_FULL_OUTPUT;				\
	    break;							\
	  }								\
									\
	out = 0xd800;							\
	out |= (zabcd & 0xff) << 6;					\
	out |= (c >> 10) & 0x3f;					\
	put16 (outptr, out);						\
	outptr += 2;							\
									\
	out = 0xdc00;							\
	out |= c & 0x3ff;						\
	put16 (outptr, out);						\
      }									\
    else								\
      {									\
	STANDARD_TO_LOOP_ERR_HANDLER (4);				\
      }									\
    outptr += 2;							\
    inptr += 4;								\
  }

/* Generate loop-function with software routing.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_TO
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
# define TO_LOOP_C		__to_utf16_loop_c
# define LOOPFCT		TO_LOOP_C
# define LOOP_NEED_FLAGS
# define BODY			BODY_TO_C
# include <iconv/loop.c>
#else
# define TO_LOOP_C		NULL
#endif /* HAVE_TO_C != 1  */

#if HAVE_TO_VX == 1
# define BODY_TO_VX							\
  {									\
    size_t inlen = inend - inptr;					\
    size_t outlen = outend - outptr;					\
    unsigned long tmp, tmp2, tmp3;					\
    asm volatile (".machine push\n\t"					\
		  ".machine \"z13\"\n\t"				\
		  ".machinemode \"zarch_nohighgprs\"\n\t"		\
		  /* Setup to check for surrogates.  */			\
		  "    larl %[R_TMP],9f\n\t"				\
		  "    vlm %%v30,%%v31,0(%[R_TMP])\n\t"			\
		  CONVERT_32BIT_SIZE_T ([R_INLEN])			\
		  CONVERT_32BIT_SIZE_T ([R_OUTLEN])			\
		  /* Loop which handles UTF-32 chars			\
		     ch < 0xd800 || (ch > 0xdfff && ch < 0x10000).  */	\
		  "0:  clgijl %[R_INLEN],32,2f\n\t"			\
		  "    clgijl %[R_OUTLEN],16,2f\n\t"			\
		  "1:  vlm %%v16,%%v17,0(%[R_IN])\n\t"			\
		  "    lghi %[R_TMP2],0\n\t"				\
		  /* Shorten to UTF-16.  */				\
		  "    vpkf %%v18,%%v16,%%v17\n\t"			\
		  /* Check for surrogate chars.  */			\
		  "    vstrcfs %%v19,%%v16,%%v30,%%v31\n\t"		\
		  "    jno 10f\n\t"					\
		  "    vstrcfs %%v19,%%v17,%%v30,%%v31\n\t"		\
		  "    jno 11f\n\t"					\
		  /* Store 16 bytes to buf_out.  */			\
		  "    vst %%v18,0(%[R_OUT])\n\t"			\
		  "    la %[R_IN],32(%[R_IN])\n\t"			\
		  "    aghi %[R_INLEN],-32\n\t"				\
		  "    aghi %[R_OUTLEN],-16\n\t"			\
		  "    la %[R_OUT],16(%[R_OUT])\n\t"			\
		  "    clgijl %[R_INLEN],32,2f\n\t"			\
		  "    clgijl %[R_OUTLEN],16,2f\n\t"			\
		  "    j 1b\n\t"					\
		  /* Calculate remaining uint32_t values in inptr.  */	\
		  "2:  \n\t"						\
		  "    clgije %[R_INLEN],0,99f\n\t"			\
		  "    clgijl %[R_INLEN],4,92f\n\t"			\
		  "    srlg %[R_TMP2],%[R_INLEN],2\n\t"			\
		  "    j 20f\n\t"					\
		  /* Setup to check for ch >= 0xd800 && ch <= 0xdfff	\
		     and check for ch >= 0x10000. (v30, v31)  */	\
		  "9:  .long 0xd800,0xdfff,0x10000,0x10000\n\t"		\
		  "    .long 0xa0000000,0xc0000000, 0xa0000000,0xa0000000\n\t" \
		  /* At least on UTF32 char is in range of surrogates.	\
		     Store the preceding characters.  */		\
		  "11: ahi %[R_TMP2],16\n\t"				\
		  "10: vlgvb %[R_TMP],%%v19,7\n\t"			\
		  "    agr %[R_TMP],%[R_TMP2]\n\t"			\
		  "    srlg %[R_TMP3],%[R_TMP],1\n\t" /* Number of out bytes.  */ \
		  "    ahik %[R_TMP2],%[R_TMP3],-1\n\t" /* Highest index to store.  */ \
		  "    jl 12f\n\t"					\
		  "    vstl %%v18,%[R_TMP2],0(%[R_OUT])\n\t"		\
		  /* Update pointers.  */				\
		  "    la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		  "    slgr %[R_INLEN],%[R_TMP]\n\t"			\
		  "    la %[R_OUT],0(%[R_TMP3],%[R_OUT])\n\t"		\
		  "    slgr %[R_OUTLEN],%[R_TMP3]\n\t"			\
		  /* Calculate remaining uint32_t values in vrs.  */	\
		  "12: lghi %[R_TMP2],8\n\t"				\
		  "    srlg %[R_TMP3],%[R_TMP3],1\n\t"			\
		  "    slgr %[R_TMP2],%[R_TMP3]\n\t"			\
		  /* Handle remaining UTF-32 characters.  */		\
		  "20: l %[R_TMP],0(%[R_IN])\n\t"			\
		  "    aghi %[R_INLEN],-4\n\t"				\
		  /* Test if ch is 2byte UTF-16 char. */		\
		  "    clfi %[R_TMP],0xffff\n\t"			\
		  "    jh 21f\n\t"					\
		  /* Handle 2 byte UTF16 char.  */			\
		  "    lgr %[R_TMP3],%[R_TMP]\n\t"			\
		  "    nilf %[R_TMP],0xf800\n\t"			\
		  "    clfi %[R_TMP],0xd800\n\t"			\
		  "    je 91f\n\t" /* Do not accept UTF-16 surrogates.  */ \
		  "    slgfi %[R_OUTLEN],2\n\t"				\
		  "    jl 90f \n\t"					\
		  "    sth %[R_TMP3],0(%[R_OUT])\n\t"			\
		  "    la %[R_IN],4(%[R_IN])\n\t"			\
		  "    la %[R_OUT],2(%[R_OUT])\n\t"			\
		  "    brctg %[R_TMP2],20b\n\t"				\
		  "    j 0b\n\t" /* Switch to vx-loop.  */		\
		  /* Test if ch is 4byte UTF-16 char. */		\
		  "21: clfi %[R_TMP],0x10ffff\n\t"			\
		  "    jh 91f\n\t" /* ch > 0x10ffff is not allowed!  */	\
		  /* Handle 4 byte UTF16 char.  */			\
		  "    slgfi %[R_OUTLEN],4\n\t"				\
		  "    jl 90f \n\t"					\
		  "    slfi %[R_TMP],0x10000\n\t" /* zabcd = uvwxy - 1.  */ \
		  "    llilf %[R_TMP3],0xd800dc00\n\t"			\
		  "    la %[R_IN],4(%[R_IN])\n\t"			\
		  "    risbgn %[R_TMP3],%[R_TMP],38,47,6\n\t" /* High surrogate.  */ \
		  "    risbgn %[R_TMP3],%[R_TMP],54,63,0\n\t" /* Low surrogate.  */ \
		  "    st %[R_TMP3],0(%[R_OUT])\n\t"			\
		  "    la %[R_OUT],4(%[R_OUT])\n\t"			\
		  "    brctg %[R_TMP2],20b\n\t"				\
		  "    j 0b\n\t" /* Switch to vx-loop.  */		\
		  "92: lghi %[R_RES],%[RES_IN_FULL]\n\t"		\
		  "    j 99f\n\t"					\
		  "91: lghi %[R_RES],%[RES_IN_ILL]\n\t"			\
		  "    j 99f\n\t"					\
		  "90: lghi %[R_RES],%[RES_OUT_FULL]\n\t"		\
		  "99: \n\t"						\
		  ".machine pop"					\
		  : /* outputs */ [R_IN] "+a" (inptr)			\
		    , [R_INLEN] "+d" (inlen), [R_OUT] "+a" (outptr)	\
		    , [R_OUTLEN] "+d" (outlen), [R_TMP] "=a" (tmp)	\
		    , [R_TMP2] "=d" (tmp2), [R_TMP3] "=a" (tmp3)	\
		    , [R_RES] "+d" (result)				\
		  : /* inputs */					\
		    [RES_OUT_FULL] "i" (__GCONV_FULL_OUTPUT)		\
		    , [RES_IN_ILL] "i" (__GCONV_ILLEGAL_INPUT)		\
		    , [RES_IN_FULL] "i" (__GCONV_INCOMPLETE_INPUT)	\
		  : /* clobber list */ "memory", "cc"			\
		    ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")	\
		    ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")	\
		    ASM_CLOBBER_VR ("v30") ASM_CLOBBER_VR ("v31")	\
		  );							\
    if (__glibc_likely (inptr == inend)					\
	|| result != __GCONV_ILLEGAL_INPUT)				\
      break;								\
									\
    STANDARD_TO_LOOP_ERR_HANDLER (4);					\
  }

/* Generate loop-function with hardware vector instructions.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_TO
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
# define TO_LOOP_VX		__to_utf16_loop_vx
# define LOOPFCT		TO_LOOP_VX
# define LOOP_NEED_FLAGS
# define BODY			BODY_TO_VX
# include <iconv/loop.c>
#else
# define TO_LOOP_VX		NULL
#endif /* HAVE_TO_VX != 1  */

#if HAVE_TO_VX_CU == 1
#define BODY_TO_VX_CU							\
  {									\
    register const unsigned char* pInput asm ("8") = inptr;		\
    register size_t inlen asm ("9") = inend - inptr;			\
    register unsigned char* pOutput asm ("10") = outptr;		\
    register size_t outlen asm ("11") = outend - outptr;		\
    unsigned long tmp, tmp2, tmp3;					\
    asm volatile (".machine push\n\t"					\
		  ".machine \"z13\"\n\t"				\
		  ".machinemode \"zarch_nohighgprs\"\n\t"		\
		  /* Setup to check for surrogates.  */			\
		  "    larl %[R_TMP],9f\n\t"				\
		  "    vlm %%v30,%%v31,0(%[R_TMP])\n\t"			\
		  CONVERT_32BIT_SIZE_T ([R_INLEN])			\
		  CONVERT_32BIT_SIZE_T ([R_OUTLEN])			\
		  /* Loop which handles UTF-32 chars			\
		     ch < 0xd800 || (ch > 0xdfff && ch < 0x10000).  */	\
		  "0:  clgijl %[R_INLEN],32,20f\n\t"			\
		  "    clgijl %[R_OUTLEN],16,20f\n\t"			\
		  "1:  vlm %%v16,%%v17,0(%[R_IN])\n\t"			\
		  "    lghi %[R_TMP2],0\n\t"				\
		  /* Shorten to UTF-16.  */				\
		  "    vpkf %%v18,%%v16,%%v17\n\t"			\
		  /* Check for surrogate chars.  */			\
		  "    vstrcfs %%v19,%%v16,%%v30,%%v31\n\t"		\
		  "    jno 10f\n\t"					\
		  "    vstrcfs %%v19,%%v17,%%v30,%%v31\n\t"		\
		  "    jno 11f\n\t"					\
		  /* Store 16 bytes to buf_out.  */			\
		  "    vst %%v18,0(%[R_OUT])\n\t"			\
		  "    la %[R_IN],32(%[R_IN])\n\t"			\
		  "    aghi %[R_INLEN],-32\n\t"				\
		  "    aghi %[R_OUTLEN],-16\n\t"			\
		  "    la %[R_OUT],16(%[R_OUT])\n\t"			\
		  "    clgijl %[R_INLEN],32,20f\n\t"			\
		  "    clgijl %[R_OUTLEN],16,20f\n\t"			\
		  "    j 1b\n\t"					\
		  /* Setup to check for ch >= 0xd800 && ch <= 0xdfff	\
		     and check for ch >= 0x10000. (v30, v31)  */	\
		  "9:  .long 0xd800,0xdfff,0x10000,0x10000\n\t"		\
		  "    .long 0xa0000000,0xc0000000, 0xa0000000,0xa0000000\n\t" \
		  /* At least one UTF32 char is in range of surrogates.	\
		     Store the preceding characters.  */		\
		  "11: ahi %[R_TMP2],16\n\t"				\
		  "10: vlgvb %[R_TMP],%%v19,7\n\t"			\
		  "    agr %[R_TMP],%[R_TMP2]\n\t"			\
		  "    srlg %[R_TMP3],%[R_TMP],1\n\t" /* Number of out bytes.  */ \
		  "    ahik %[R_TMP2],%[R_TMP3],-1\n\t" /* Highest index to store.  */ \
		  "    jl 20f\n\t"					\
		  "    vstl %%v18,%[R_TMP2],0(%[R_OUT])\n\t"		\
		  /* Update pointers.  */				\
		  "    la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		  "    slgr %[R_INLEN],%[R_TMP]\n\t"			\
		  "    la %[R_OUT],0(%[R_TMP3],%[R_OUT])\n\t"		\
		  "    slgr %[R_OUTLEN],%[R_TMP3]\n\t"			\
		  /* Handles UTF16 surrogates with convert instruction.  */ \
		  "20: cu42 %[R_OUT],%[R_IN]\n\t"			\
		  "    jo 0b\n\t" /* Try vector implemenation again.  */ \
		  "    lochil %[R_RES],%[RES_OUT_FULL]\n\t" /* cc == 1.  */ \
		  "    lochih %[R_RES],%[RES_IN_ILL]\n\t" /* cc == 2.  */ \
		  ".machine pop"					\
		  : /* outputs */ [R_IN] "+a" (pInput)			\
		    , [R_INLEN] "+d" (inlen), [R_OUT] "+a" (pOutput)	\
		    , [R_OUTLEN] "+d" (outlen), [R_TMP] "=a" (tmp)	\
		    , [R_TMP2] "=d" (tmp2), [R_TMP3] "=a" (tmp3)	\
		    , [R_RES] "+d" (result)				\
		  : /* inputs */					\
		    [RES_OUT_FULL] "i" (__GCONV_FULL_OUTPUT)		\
		    , [RES_IN_ILL] "i" (__GCONV_ILLEGAL_INPUT)		\
		  : /* clobber list */ "memory", "cc"			\
		    ASM_CLOBBER_VR ("v16") ASM_CLOBBER_VR ("v17")	\
		    ASM_CLOBBER_VR ("v18") ASM_CLOBBER_VR ("v19")	\
		    ASM_CLOBBER_VR ("v30") ASM_CLOBBER_VR ("v31")	\
		  );							\
    inptr = pInput;							\
    outptr = pOutput;							\
									\
    if (__glibc_likely (inlen == 0)					\
	|| result == __GCONV_FULL_OUTPUT)				\
      break;								\
    if (inlen < 4)							\
      {									\
	result = __GCONV_INCOMPLETE_INPUT;				\
	break;								\
      }									\
									\
    STANDARD_TO_LOOP_ERR_HANDLER (4);					\
  }

/* Generate loop-function with hardware vector and utf-convert instructions.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_TO
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
# define TO_LOOP_VX_CU		__to_utf16_loop_vx_cu
# define LOOPFCT		TO_LOOP_VX_CU
# define LOOP_NEED_FLAGS
# define BODY			BODY_TO_VX_CU
# include <iconv/loop.c>
#else
# define TO_LOOP_VX_CU		NULL
#endif /* HAVE_TO_VX_CU != 1  */

/* This file also exists in sysdeps/s390/multiarch/ which
   generates ifunc resolvers for FROM/TO_LOOP functions
   and includes iconv/skeleton.c afterwards.  */
#if ! defined USE_MULTIARCH
# include <iconv/skeleton.c>
#endif
