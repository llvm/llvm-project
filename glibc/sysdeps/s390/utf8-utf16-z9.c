/* Conversion between UTF-8 and UTF-16 - s390 version.

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
#ifdef HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT
# define HAVE_FROM_C		0
# define FROM_LOOP_DEFAULT	FROM_LOOP_CU
#else
# define HAVE_FROM_C		1
# define FROM_LOOP_DEFAULT	FROM_LOOP_C
#endif

#define HAVE_TO_C		1
#define TO_LOOP_DEFAULT		TO_LOOP_C

#if defined HAVE_S390_MIN_Z196_ZARCH_ASM_SUPPORT || defined USE_MULTIARCH
# define HAVE_FROM_CU		1
#else
# define HAVE_FROM_CU		0
#endif

#if defined HAVE_S390_VX_ASM_SUPPORT && defined USE_MULTIARCH
# define HAVE_FROM_VX		1
# define HAVE_TO_VX		1
# define HAVE_TO_VX_CU		1
#else
# define HAVE_FROM_VX		0
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

/* Defines for skeleton.c.  */
#define DEFINE_INIT		0
#define DEFINE_FINI		0
#define MIN_NEEDED_FROM		1
#define MAX_NEEDED_FROM		4
#define MIN_NEEDED_TO		2
#define MAX_NEEDED_TO		4
#define FROM_LOOP		FROM_LOOP_DEFAULT
#define TO_LOOP			TO_LOOP_DEFAULT
#define FROM_DIRECTION		(dir == from_utf8)
#define ONE_DIRECTION           0


/* UTF-16 big endian byte order mark.  */
#define BOM_UTF16	0xfeff

/* Direction of the transformation.  */
enum direction
{
  illegal_dir,
  to_utf8,
  from_utf8
};

struct utf8_data
{
  enum direction dir;
  int emit_bom;
};


extern int gconv_init (struct __gconv_step *step);
int
gconv_init (struct __gconv_step *step)
{
  /* Determine which direction.  */
  struct utf8_data *new_data;
  enum direction dir = illegal_dir;
  int emit_bom;
  int result;

  emit_bom = (__strcasecmp (step->__to_name, "UTF-16//") == 0);

  if (__strcasecmp (step->__from_name, "ISO-10646/UTF8/") == 0
      && (__strcasecmp (step->__to_name, "UTF-16//") == 0
	  || __strcasecmp (step->__to_name, "UTF-16BE//") == 0))
    {
      dir = from_utf8;
    }
  else if (__strcasecmp (step->__from_name, "UTF-16BE//") == 0
	   && __strcasecmp (step->__to_name, "ISO-10646/UTF8/") == 0)
    {
      dir = to_utf8;
    }

  result = __GCONV_NOCONV;
  if (dir != illegal_dir)
    {
      new_data = (struct utf8_data *) malloc (sizeof (struct utf8_data));

      result = __GCONV_NOMEM;
      if (new_data != NULL)
	{
	  new_data->dir = dir;
	  new_data->emit_bom = emit_bom;
	  step->__data = new_data;

	  if (dir == from_utf8)
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

/* The macro for the hardware loop.  This is used for both
   directions.  */
#define HARDWARE_CONVERT(INSTRUCTION)					\
  {									\
    register const unsigned char* pInput __asm__ ("8") = inptr;		\
    register size_t inlen __asm__ ("9") = inend - inptr;		\
    register unsigned char* pOutput __asm__ ("10") = outptr;		\
    register size_t outlen __asm__("11") = outend - outptr;		\
    unsigned long cc = 0;						\
									\
    __asm__ __volatile__ (".machine push       \n\t"			\
			  ".machine \"z9-109\" \n\t"			\
			  ".machinemode \"zarch_nohighgprs\"\n\t"	\
			  "0: " INSTRUCTION "  \n\t"			\
			  ".machine pop        \n\t"			\
			  "   jo     0b        \n\t"			\
			  "   ipm    %2        \n"			\
			  : "+a" (pOutput), "+a" (pInput), "+d" (cc),	\
			    "+d" (outlen), "+d" (inlen)			\
			  :						\
			  : "cc", "memory");				\
									\
    inptr = pInput;							\
    outptr = pOutput;							\
    cc >>= 28;								\
									\
    if (cc == 1)							\
      {									\
	result = __GCONV_FULL_OUTPUT;					\
      }									\
    else if (cc == 2)							\
      {									\
	result = __GCONV_ILLEGAL_INPUT;					\
      }									\
  }

#define PREPARE_LOOP							\
  enum direction dir = ((struct utf8_data *) step->__data)->dir;	\
  int emit_bom = ((struct utf8_data *) step->__data)->emit_bom;		\
									\
  if (emit_bom && !data->__internal_use					\
      && data->__invocation_counter == 0)				\
    {									\
      /* Emit the UTF-16 Byte Order Mark.  */				\
      if (__glibc_unlikely (outbuf + 2 > outend))			\
	return __GCONV_FULL_OUTPUT;					\
									\
      put16u (outbuf, BOM_UTF16);					\
      outbuf += 2;							\
    }

/* Conversion function from UTF-8 to UTF-16.  */
#define BODY_FROM_HW(ASM)						\
  {									\
    ASM;								\
    if (__glibc_likely (inptr == inend)					\
	|| result == __GCONV_FULL_OUTPUT)				\
      break;								\
									\
    int i;								\
    for (i = 1; inptr + i < inend && i < 5; ++i)			\
      if ((inptr[i] & 0xc0) != 0x80)					\
	break;								\
									\
    if (__glibc_likely (inptr + i == inend				\
			&& result == __GCONV_EMPTY_INPUT))		\
      {									\
	result = __GCONV_INCOMPLETE_INPUT;				\
	break;								\
      }									\
    STANDARD_FROM_LOOP_ERR_HANDLER (i);					\
  }

#if HAVE_FROM_VX == 1
# define HW_FROM_VX							\
  {									\
    register const unsigned char* pInput asm ("8") = inptr;		\
    register size_t inlen asm ("9") = inend - inptr;			\
    register unsigned char* pOutput asm ("10") = outptr;		\
    register size_t outlen asm("11") = outend - outptr;			\
    unsigned long tmp, tmp2, tmp3;					\
    asm volatile (".machine push\n\t"					\
		  ".machine \"z13\"\n\t"				\
		  ".machinemode \"zarch_nohighgprs\"\n\t"		\
		  "    vrepib %%v30,0x7f\n\t" /* For compare > 0x7f.  */ \
		  "    vrepib %%v31,0x20\n\t"				\
		  CONVERT_32BIT_SIZE_T ([R_INLEN])			\
		  CONVERT_32BIT_SIZE_T ([R_OUTLEN])			\
		  /* Loop which handles UTF-8 chars <=0x7f.  */		\
		  "0:  clgijl %[R_INLEN],16,20f\n\t"			\
		  "    clgijl %[R_OUTLEN],32,20f\n\t"			\
		  "1:  vl %%v16,0(%[R_IN])\n\t"				\
		  "    vstrcbs %%v17,%%v16,%%v30,%%v31\n\t"		\
		  "    jno 10f\n\t" /* Jump away if not all bytes are 1byte \
				       UTF8 chars.  */			\
		  /* Enlarge to UTF-16.  */				\
		  "    vuplhb %%v18,%%v16\n\t"				\
		  "    la %[R_IN],16(%[R_IN])\n\t"			\
		  "    vupllb %%v19,%%v16\n\t"				\
		  "    aghi %[R_INLEN],-16\n\t"				\
		  /* Store 32 bytes to buf_out.  */			\
		  "    vstm %%v18,%%v19,0(%[R_OUT])\n\t"		\
		  "    aghi %[R_OUTLEN],-32\n\t"			\
		  "    la %[R_OUT],32(%[R_OUT])\n\t"			\
		  "    clgijl %[R_INLEN],16,20f\n\t"			\
		  "    clgijl %[R_OUTLEN],32,20f\n\t"			\
		  "    j 1b\n\t"					\
		  "10:\n\t"						\
		  /* At least one byte is > 0x7f.			\
		     Store the preceding 1-byte chars.  */		\
		  "    vlgvb %[R_TMP],%%v17,7\n\t"			\
		  "    sllk %[R_TMP2],%[R_TMP],1\n\t" /* Compute highest \
							 index to store. */ \
		  "    llgfr %[R_TMP3],%[R_TMP2]\n\t"			\
		  "    ahi %[R_TMP2],-1\n\t"				\
		  "    jl 20f\n\t"					\
		  "    vuplhb %%v18,%%v16\n\t"				\
		  "    vstl %%v18,%[R_TMP2],0(%[R_OUT])\n\t"		\
		  "    ahi %[R_TMP2],-16\n\t"				\
		  "    jl 11f\n\t"					\
		  "    vupllb %%v19,%%v16\n\t"				\
		  "    vstl %%v19,%[R_TMP2],16(%[R_OUT])\n\t"		\
		  "11: \n\t" /* Update pointers.  */			\
		  "    la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		  "    slgr %[R_INLEN],%[R_TMP]\n\t"			\
		  "    la %[R_OUT],0(%[R_TMP3],%[R_OUT])\n\t"		\
		  "    slgr %[R_OUTLEN],%[R_TMP3]\n\t"			\
		  /* Handle multibyte utf8-char with convert instruction. */ \
		  "20: cu12 %[R_OUT],%[R_IN],1\n\t"			\
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
  }
# define BODY_FROM_VX BODY_FROM_HW (HW_FROM_VX)

/* Generate loop-function with hardware vector and utf-convert instructions.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_TO
# define FROM_LOOP_VX		__from_utf8_loop_vx
# define LOOPFCT		FROM_LOOP_VX
# define LOOP_NEED_FLAGS
# define BODY			BODY_FROM_VX
# include <iconv/loop.c>
#else
# define FROM_LOOP_VX		NULL
#endif /* HAVE_FROM_VX != 1  */

#if HAVE_FROM_CU == 1
# define BODY_FROM_ETF3EH BODY_FROM_HW (HARDWARE_CONVERT ("cu12 %0, %1, 1"))

/* Generate loop-function with hardware utf-convert instruction.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_TO
# define FROM_LOOP_CU		__from_utf8_loop_etf3eh
# define LOOPFCT		FROM_LOOP_CU
# define LOOP_NEED_FLAGS
# define BODY			BODY_FROM_ETF3EH
# include <iconv/loop.c>
#else
# define FROM_LOOP_CU		NULL
#endif /* HAVE_FROM_CU != 1  */

#if HAVE_FROM_C == 1
/* The software implementation is based on the code in gconv_simple.c.  */
# define BODY_FROM_C							\
  {									\
    /* Next input byte.  */						\
    uint16_t ch = *inptr;						\
									\
    if (__glibc_likely (ch < 0x80))					\
      {									\
	/* One byte sequence.  */					\
	++inptr;							\
      }									\
    else								\
      {									\
	uint_fast32_t cnt;						\
	uint_fast32_t i;						\
									\
	if (ch >= 0xc2 && ch < 0xe0)					\
	  {								\
	    /* We expect two bytes.  The first byte cannot be 0xc0	\
	       or 0xc1, otherwise the wide character could have been	\
	       represented using a single byte.  */			\
	    cnt = 2;							\
	    ch &= 0x1f;							\
	  }								\
	else if (__glibc_likely ((ch & 0xf0) == 0xe0))			\
	  {								\
	    /* We expect three bytes.  */				\
	    cnt = 3;							\
	    ch &= 0x0f;							\
	  }								\
	else if (__glibc_likely ((ch & 0xf8) == 0xf0))			\
	  {								\
	    /* We expect four bytes.  */				\
	    cnt = 4;							\
	    ch &= 0x07;							\
	  }								\
	else								\
	  {								\
	    /* Search the end of this ill-formed UTF-8 character.  This	\
	       is the next byte with (x & 0xc0) != 0x80.  */		\
	    i = 0;							\
	    do								\
	      ++i;							\
	    while (inptr + i < inend					\
		   && (*(inptr + i) & 0xc0) == 0x80			\
		   && i < 5);						\
									\
	  errout:							\
	    STANDARD_FROM_LOOP_ERR_HANDLER (i);				\
	  }								\
									\
	if (__glibc_unlikely (inptr + cnt > inend))			\
	  {								\
	    /* We don't have enough input.  But before we report	\
	       that check that all the bytes are correct.  */		\
	    for (i = 1; inptr + i < inend; ++i)				\
	      if ((inptr[i] & 0xc0) != 0x80)				\
		break;							\
									\
	    if (__glibc_likely (inptr + i == inend))			\
	      {								\
		result = __GCONV_INCOMPLETE_INPUT;			\
		break;							\
	      }								\
									\
	    goto errout;						\
	  }								\
									\
	if (cnt == 4)							\
	  {								\
	    /* For 4 byte UTF-8 chars two UTF-16 chars (high and	\
	       low) are needed.  */					\
	    uint16_t zabcd, high, low;					\
									\
	    if (__glibc_unlikely (outptr + 4 > outend))			\
	      {								\
		/* Overflow in the output buffer.  */			\
		result = __GCONV_FULL_OUTPUT;				\
		break;							\
	      }								\
									\
	    /* Check if tail-bytes >= 0x80, < 0xc0.  */			\
	    for (i = 1; i < cnt; ++i)					\
	      {								\
		if ((inptr[i] & 0xc0) != 0x80)				\
		  /* This is an illegal encoding.  */			\
		  goto errout;						\
	      }								\
									\
	    /* See Principles of Operations cu12.  */			\
	    zabcd = (((inptr[0] & 0x7) << 2)				\
		     | ((inptr[1] & 0x30) >> 4)) - 1;			\
									\
	    /* z-bit must be zero after subtracting 1.  */		\
	    if (zabcd & 0x10)						\
	      STANDARD_FROM_LOOP_ERR_HANDLER (4)			\
									\
	    high = (uint16_t)(0xd8 << 8);       /* high surrogate id */ \
	    high |= zabcd << 6;                         /* abcd bits */	\
	    high |= (inptr[1] & 0xf) << 2;              /* efgh bits */	\
	    high |= (inptr[2] & 0x30) >> 4;               /* ij bits */	\
									\
	    low = (uint16_t)(0xdc << 8);         /* low surrogate id */ \
	    low |= ((uint16_t)inptr[2] & 0xc) << 6;       /* kl bits */	\
	    low |= (inptr[2] & 0x3) << 6;                 /* mn bits */	\
	    low |= inptr[3] & 0x3f;                   /* opqrst bits */	\
									\
	    put16 (outptr, high);					\
	    outptr += 2;						\
	    put16 (outptr, low);					\
	    outptr += 2;						\
	    inptr += 4;							\
	    continue;							\
	  }								\
	else								\
	  {								\
	    /* Read the possible remaining bytes.  */			\
	    for (i = 1; i < cnt; ++i)					\
	      {								\
		uint16_t byte = inptr[i];				\
									\
		if ((byte & 0xc0) != 0x80)				\
		  /* This is an illegal encoding.  */			\
		  break;						\
									\
		ch <<= 6;						\
		ch |= byte & 0x3f;					\
	      }								\
									\
	    /* If i < cnt, some trail byte was not >= 0x80, < 0xc0.	\
	       If cnt > 2 and ch < 2^(5*cnt-4), the wide character ch could \
	       have been represented with fewer than cnt bytes.  */	\
	    if (i < cnt || (cnt > 2 && (ch >> (5 * cnt - 4)) == 0)	\
		/* Do not accept UTF-16 surrogates.  */			\
		|| (ch >= 0xd800 && ch <= 0xdfff))			\
	      {								\
		/* This is an illegal encoding.  */			\
		goto errout;						\
	      }								\
									\
	    inptr += cnt;						\
	  }								\
      }									\
    /* Now adjust the pointers and store the result.  */		\
    *((uint16_t *) outptr) = ch;					\
    outptr += sizeof (uint16_t);					\
  }

/* Generate loop-function with software implementation.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_TO
# define FROM_LOOP_C		__from_utf8_loop_c
# define LOOPFCT		FROM_LOOP_C
# define LOOP_NEED_FLAGS
# define BODY			BODY_FROM_C
# include <iconv/loop.c>
#else
# define FROM_LOOP_C		NULL
#endif /* HAVE_FROM_C != 1  */

/* Conversion from UTF-16 to UTF-8.  */

#if HAVE_TO_C == 1
/* The software routine is based on the functionality of the S/390
   hardware instruction (cu21) as described in the Principles of
   Operation.  */
# define BODY_TO_C							\
  {									\
    uint16_t c = get16 (inptr);						\
									\
    if (__glibc_likely (c <= 0x007f))					\
      {									\
	/* Single byte UTF-8 char.  */					\
	*outptr = c & 0xff;						\
	outptr++;							\
      }									\
    else if (c >= 0x0080 && c <= 0x07ff)				\
      {									\
	/* Two byte UTF-8 char.  */					\
									\
	if (__glibc_unlikely (outptr + 2 > outend))			\
	  {								\
	    /* Overflow in the output buffer.  */			\
	    result = __GCONV_FULL_OUTPUT;				\
	    break;							\
	  }								\
									\
	outptr[0] = 0xc0;						\
	outptr[0] |= c >> 6;						\
									\
	outptr[1] = 0x80;						\
	outptr[1] |= c & 0x3f;						\
									\
	outptr += 2;							\
      }									\
    else if ((c >= 0x0800 && c <= 0xd7ff) || c > 0xdfff)		\
      {									\
	/* Three byte UTF-8 char.  */					\
									\
	if (__glibc_unlikely (outptr + 3 > outend))			\
	  {								\
	    /* Overflow in the output buffer.  */			\
	    result = __GCONV_FULL_OUTPUT;				\
	    break;							\
	  }								\
	outptr[0] = 0xe0;						\
	outptr[0] |= c >> 12;						\
									\
	outptr[1] = 0x80;						\
	outptr[1] |= (c >> 6) & 0x3f;					\
									\
	outptr[2] = 0x80;						\
	outptr[2] |= c & 0x3f;						\
									\
	outptr += 3;							\
      }									\
    else if (c >= 0xd800 && c <= 0xdbff)				\
      {									\
	/* Four byte UTF-8 char.  */					\
	uint16_t low, uvwxy;						\
									\
	if (__glibc_unlikely (outptr + 4 > outend))			\
	  {								\
	    /* Overflow in the output buffer.  */			\
	    result = __GCONV_FULL_OUTPUT;				\
	    break;							\
	  }								\
	if (__glibc_unlikely (inptr + 4 > inend))			\
	  {								\
	    result = __GCONV_INCOMPLETE_INPUT;				\
	    break;							\
	  }								\
									\
	inptr += 2;							\
	low = get16 (inptr);						\
									\
	if ((low & 0xfc00) != 0xdc00)					\
	  {								\
	    inptr -= 2;							\
	    STANDARD_TO_LOOP_ERR_HANDLER (2);				\
	  }								\
	uvwxy = ((c >> 6) & 0xf) + 1;					\
	outptr[0] = 0xf0;						\
	outptr[0] |= uvwxy >> 2;					\
									\
	outptr[1] = 0x80;						\
	outptr[1] |= (uvwxy << 4) & 0x30;				\
	outptr[1] |= (c >> 2) & 0x0f;					\
									\
	outptr[2] = 0x80;						\
	outptr[2] |= (c & 0x03) << 4;					\
	outptr[2] |= (low >> 6) & 0x0f;					\
									\
	outptr[3] = 0x80;						\
	outptr[3] |= low & 0x3f;					\
									\
	outptr += 4;							\
      }									\
    else								\
      {									\
	STANDARD_TO_LOOP_ERR_HANDLER (2);				\
      }									\
    inptr += 2;								\
  }

/* Generate loop-function with software implementation.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_TO
# define MAX_NEEDED_INPUT	MAX_NEEDED_TO
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
# define TO_LOOP_C		__to_utf8_loop_c
# define LOOPFCT		TO_LOOP_C
# define BODY                   BODY_TO_C
# define LOOP_NEED_FLAGS
# include <iconv/loop.c>
#else
# define TO_LOOP_C		NULL
#endif /* HAVE_TO_C != 1  */

#if HAVE_TO_VX == 1
# define BODY_TO_VX							\
  {									\
    size_t inlen  = inend - inptr;					\
    size_t outlen  = outend - outptr;					\
    unsigned long tmp, tmp2, tmp3;					\
    asm volatile (".machine push\n\t"					\
		  ".machine \"z13\"\n\t"				\
		  ".machinemode \"zarch_nohighgprs\"\n\t"		\
		  /* Setup to check for values <= 0x7f.  */		\
		  "    larl %[R_TMP],9f\n\t"				\
		  "    vlm %%v30,%%v31,0(%[R_TMP])\n\t"			\
		  CONVERT_32BIT_SIZE_T ([R_INLEN])			\
		  CONVERT_32BIT_SIZE_T ([R_OUTLEN])			\
		  /* Loop which handles UTF-16 chars <=0x7f.  */	\
		  "0:  clgijl %[R_INLEN],32,2f\n\t"			\
		  "    clgijl %[R_OUTLEN],16,2f\n\t"			\
		  "1:  vlm %%v16,%%v17,0(%[R_IN])\n\t"			\
		  "    lghi %[R_TMP2],0\n\t"				\
		  /* Check for > 1byte UTF-8 chars.  */			\
		  "    vstrchs %%v19,%%v16,%%v30,%%v31\n\t"		\
		  "    jno 10f\n\t" /* Jump away if not all bytes are 1byte \
				       UTF8 chars.  */			\
		  "    vstrchs %%v19,%%v17,%%v30,%%v31\n\t"		\
		  "    jno 11f\n\t" /* Jump away if not all bytes are 1byte \
				       UTF8 chars.  */			\
		  /* Shorten to UTF-8.  */				\
		  "    vpkh %%v18,%%v16,%%v17\n\t"			\
		  "    la %[R_IN],32(%[R_IN])\n\t"			\
		  "    aghi %[R_INLEN],-32\n\t"				\
		  /* Store 16 bytes to buf_out.  */			\
		  "    vst %%v18,0(%[R_OUT])\n\t"			\
		  "    aghi %[R_OUTLEN],-16\n\t"			\
		  "    la %[R_OUT],16(%[R_OUT])\n\t"			\
		  "    clgijl %[R_INLEN],32,2f\n\t"			\
		  "    clgijl %[R_OUTLEN],16,2f\n\t"			\
		  "    j 1b\n\t"					\
		  /* Setup to check for ch > 0x7f. (v30, v31)  */	\
		  "9:  .short 0x7f,0x7f,0x0,0x0,0x0,0x0,0x0,0x0\n\t"	\
		  "    .short 0x2000,0x2000,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		  /* At least one byte is > 0x7f.			\
		     Store the preceding 1-byte chars.  */		\
		  "11: lghi %[R_TMP2],16\n\t" /* match was found in v17.  */ \
		  "10:\n\t"						\
		  "    vlgvb %[R_TMP],%%v19,7\n\t"			\
		  /* Shorten to UTF-8.  */				\
		  "    vpkh %%v18,%%v16,%%v17\n\t"			\
		  "    ar %[R_TMP],%[R_TMP2]\n\t" /* Number of in bytes.  */ \
		  "    srlg %[R_TMP3],%[R_TMP],1\n\t" /* Number of out bytes.  */ \
		  "    ahik %[R_TMP2],%[R_TMP3],-1\n\t" /* Highest index to store.  */ \
		  "    jl 13f\n\t"					\
		  "    vstl %%v18,%[R_TMP2],0(%[R_OUT])\n\t"		\
		  /* Update pointers.  */				\
		  "    la %[R_IN],0(%[R_TMP],%[R_IN])\n\t"		\
		  "    slgr %[R_INLEN],%[R_TMP]\n\t"			\
		  "    la %[R_OUT],0(%[R_TMP3],%[R_OUT])\n\t"		\
		  "    slgr %[R_OUTLEN],%[R_TMP3]\n\t"			\
		  "13: \n\t"						\
		  /* Calculate remaining uint16_t values in loaded vrs.  */ \
		  "    lghi %[R_TMP2],16\n\t"				\
		  "    slgr %[R_TMP2],%[R_TMP3]\n\t"			\
		  "    llh %[R_TMP],0(%[R_IN])\n\t"			\
		  "    aghi %[R_INLEN],-2\n\t"				\
		  "    j 22f\n\t"					\
		  /* Handle remaining bytes.  */			\
		  "2:  \n\t"						\
		  /* Zero, one or more bytes available?  */		\
		  "    clgfi %[R_INLEN],1\n\t"				\
		  "    locghie %[R_RES],%[RES_IN_FULL]\n\t" /* Only one byte.  */ \
		  "    jle 99f\n\t" /* End if less than two bytes.  */	\
		  /* Calculate remaining uint16_t values in inptr.  */	\
		  "    srlg %[R_TMP2],%[R_INLEN],1\n\t"			\
		  /* Handle multibyte utf8-char. */			\
		  "20: llh %[R_TMP],0(%[R_IN])\n\t"			\
		  "    aghi %[R_INLEN],-2\n\t"				\
		  /* Test if ch is 1-byte UTF-8 char.  */		\
		  "21: clijh %[R_TMP],0x7f,22f\n\t"			\
		  /* Handle 1-byte UTF-8 char.  */			\
		  "31: slgfi %[R_OUTLEN],1\n\t"				\
		  "    jl 90f \n\t"					\
		  "    stc %[R_TMP],0(%[R_OUT])\n\t"			\
		  "    la %[R_IN],2(%[R_IN])\n\t"			\
		  "    la %[R_OUT],1(%[R_OUT])\n\t"			\
		  "    brctg %[R_TMP2],20b\n\t"				\
		  "    j 0b\n\t" /* Switch to vx-loop.  */		\
		  /* Test if ch is 2-byte UTF-8 char.  */		\
		  "22: clfi %[R_TMP],0x7ff\n\t"				\
		  "    jh 23f\n\t"					\
		  /* Handle 2-byte UTF-8 char.  */			\
		  "32: slgfi %[R_OUTLEN],2\n\t"				\
		  "    jl 90f \n\t"					\
		  "    llill %[R_TMP3],0xc080\n\t"			\
		  "    la %[R_IN],2(%[R_IN])\n\t"			\
		  "    risbgn %[R_TMP3],%[R_TMP],51,55,2\n\t" /* 1. byte.   */ \
		  "    risbgn %[R_TMP3],%[R_TMP],58,63,0\n\t" /* 2. byte.   */ \
		  "    sth %[R_TMP3],0(%[R_OUT])\n\t"			\
		  "    la %[R_OUT],2(%[R_OUT])\n\t"			\
		  "    brctg %[R_TMP2],20b\n\t"				\
		  "    j 0b\n\t" /* Switch to vx-loop.  */		\
		  /* Test if ch is 3-byte UTF-8 char.  */		\
		  "23: clfi %[R_TMP],0xd7ff\n\t"			\
		  "    jh 24f\n\t"					\
		  /* Handle 3-byte UTF-8 char.  */			\
		  "33: slgfi %[R_OUTLEN],3\n\t"				\
		  "    jl 90f \n\t"					\
		  "    llilf %[R_TMP3],0xe08080\n\t"			\
		  "    la %[R_IN],2(%[R_IN])\n\t"			\
		  "    risbgn %[R_TMP3],%[R_TMP],44,47,4\n\t" /* 1. byte.  */ \
		  "    risbgn %[R_TMP3],%[R_TMP],50,55,2\n\t" /* 2. byte.  */ \
		  "    risbgn %[R_TMP3],%[R_TMP],58,63,0\n\t" /* 3. byte.  */ \
		  "    stcm %[R_TMP3],7,0(%[R_OUT])\n\t"		\
		  "    la %[R_OUT],3(%[R_OUT])\n\t"			\
		  "    brctg %[R_TMP2],20b\n\t"				\
		  "    j 0b\n\t" /* Switch to vx-loop.  */		\
		  /* Test if ch is 4-byte UTF-8 char.  */		\
		  "24: clfi %[R_TMP],0xdfff\n\t"			\
		  "    jh 33b\n\t" /* Handle this 3-byte UTF-8 char.  */ \
		  "    clfi %[R_TMP],0xdbff\n\t"			\
		  "    locghih %[R_RES],%[RES_IN_ILL]\n\t"		\
		  "    jh 99f\n\t" /* Jump away if this is a low surrogate \
				      without a preceding high surrogate.  */ \
		  /* Handle 4-byte UTF-8 char.  */			\
		  "34: slgfi %[R_OUTLEN],4\n\t"				\
		  "    jl 90f \n\t"					\
		  "    slgfi %[R_INLEN],2\n\t"				\
		  "    locghil %[R_RES],%[RES_IN_FULL]\n\t"		\
		  "    jl 99f\n\t" /* Jump away if low surrogate is missing.  */ \
		  "    llilf %[R_TMP3],0xf0808080\n\t"			\
		  "    aghi %[R_TMP],0x40\n\t"				\
		  "    risbgn %[R_TMP3],%[R_TMP],37,39,16\n\t" /* 1. byte: uvw  */ \
		  "    risbgn %[R_TMP3],%[R_TMP],42,43,14\n\t" /* 2. byte: xy  */ \
		  "    risbgn %[R_TMP3],%[R_TMP],44,47,14\n\t" /* 2. byte: efgh  */ \
		  "    risbgn %[R_TMP3],%[R_TMP],50,51,12\n\t" /* 3. byte: ij */ \
		  "    llh %[R_TMP],2(%[R_IN])\n\t" /* Load low surrogate.  */ \
		  "    risbgn %[R_TMP3],%[R_TMP],52,55,2\n\t" /* 3. byte: klmn  */ \
		  "    risbgn %[R_TMP3],%[R_TMP],58,63,0\n\t" /* 4. byte: opqrst  */ \
		  "    nilf %[R_TMP],0xfc00\n\t"			\
		  "    clfi %[R_TMP],0xdc00\n\t" /* Check if it starts with 0xdc00.  */ \
		  "    locghine %[R_RES],%[RES_IN_ILL]\n\t"		\
		  "    jne 99f\n\t" /* Jump away if low surrogate is invalid.  */ \
		  "    st %[R_TMP3],0(%[R_OUT])\n\t"			\
		  "    la %[R_IN],4(%[R_IN])\n\t"			\
		  "    la %[R_OUT],4(%[R_OUT])\n\t"			\
		  "    aghi %[R_TMP2],-2\n\t"				\
		  "    jh 20b\n\t"					\
		  "    j 0b\n\t" /* Switch to vx-loop.  */		\
		  /* Exit with __GCONV_FULL_OUTPUT.  */			\
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
    STANDARD_TO_LOOP_ERR_HANDLER (2);					\
  }

/* Generate loop-function with vector implementation.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_TO
# define MAX_NEEDED_INPUT	MAX_NEEDED_TO
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
# define TO_LOOP_VX		__to_utf8_loop_vx
# define LOOPFCT		TO_LOOP_VX
# define BODY                   BODY_TO_VX
# define LOOP_NEED_FLAGS
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
		  /* Setup to check for values <= 0x7f.  */		\
		  "    larl %[R_TMP],9f\n\t"				\
		  "    vlm %%v30,%%v31,0(%[R_TMP])\n\t"			\
		  CONVERT_32BIT_SIZE_T ([R_INLEN])			\
		  CONVERT_32BIT_SIZE_T ([R_OUTLEN])			\
		  /* Loop which handles UTF-16 chars <=0x7f.  */	\
		  "0:  clgijl %[R_INLEN],32,20f\n\t"			\
		  "    clgijl %[R_OUTLEN],16,20f\n\t"			\
		  "1:  vlm %%v16,%%v17,0(%[R_IN])\n\t"			\
		  "    lghi %[R_TMP2],0\n\t"				\
		  /* Check for > 1byte UTF-8 chars.  */			\
		  "    vstrchs %%v19,%%v16,%%v30,%%v31\n\t"		\
		  "    jno 10f\n\t" /* Jump away if not all bytes are 1byte \
				       UTF8 chars.  */			\
		  "    vstrchs %%v19,%%v17,%%v30,%%v31\n\t"		\
		  "    jno 11f\n\t" /* Jump away if not all bytes are 1byte \
				       UTF8 chars.  */			\
		  /* Shorten to UTF-8.  */				\
		  "    vpkh %%v18,%%v16,%%v17\n\t"			\
		  "    la %[R_IN],32(%[R_IN])\n\t"			\
		  "    aghi %[R_INLEN],-32\n\t"				\
		  /* Store 16 bytes to buf_out.  */			\
		  "    vst %%v18,0(%[R_OUT])\n\t"			\
		  "    aghi %[R_OUTLEN],-16\n\t"			\
		  "    la %[R_OUT],16(%[R_OUT])\n\t"			\
		  "    clgijl %[R_INLEN],32,20f\n\t"			\
		  "    clgijl %[R_OUTLEN],16,20f\n\t"			\
		  "    j 1b\n\t"					\
		  /* Setup to check for ch > 0x7f. (v30, v31)  */	\
		  "9:  .short 0x7f,0x7f,0x0,0x0,0x0,0x0,0x0,0x0\n\t"	\
		  "    .short 0x2000,0x2000,0x0,0x0,0x0,0x0,0x0,0x0\n\t" \
		  /* At least one byte is > 0x7f.			\
		     Store the preceding 1-byte chars.  */		\
		  "11: lghi %[R_TMP2],16\n\t" /* match was found in v17.  */ \
		  "10: vlgvb %[R_TMP],%%v19,7\n\t"			\
		  /* Shorten to UTF-8.  */				\
		  "    vpkh %%v18,%%v16,%%v17\n\t"			\
		  "    ar %[R_TMP],%[R_TMP2]\n\t" /* Number of in bytes.  */ \
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
		  "20: cu21 %[R_OUT],%[R_IN],1\n\t"			\
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
	/* Input is >= 2 and < 4 bytes (as cu21 would have processed	\
	   a possible next utf16 character) and not illegal.		\
	   => we have a single high surrogate at end of input.  */	\
	result = __GCONV_INCOMPLETE_INPUT;				\
	break;								\
      }									\
									\
    STANDARD_TO_LOOP_ERR_HANDLER (2);					\
  }

/* Generate loop-function with vector and utf-convert instructions.  */
# define MIN_NEEDED_INPUT	MIN_NEEDED_TO
# define MAX_NEEDED_INPUT	MAX_NEEDED_TO
# define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
# define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
# define TO_LOOP_VX_CU		__to_utf8_loop_vx_cu
# define LOOPFCT		TO_LOOP_VX_CU
# define BODY                   BODY_TO_VX_CU
# define LOOP_NEED_FLAGS
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
