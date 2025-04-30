/* Conversion loop frame work.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

/* This file provides a frame for the reader loop in all conversion modules.
   The actual code must (of course) be provided in the actual module source
   code but certain actions can be written down generically, with some
   customization options which are these:

     MIN_NEEDED_INPUT	minimal number of input bytes needed for the next
			conversion.
     MIN_NEEDED_OUTPUT	minimal number of bytes produced by the next round
			of conversion.

     MAX_NEEDED_INPUT	you guess it, this is the maximal number of input
			bytes needed.  It defaults to MIN_NEEDED_INPUT
     MAX_NEEDED_OUTPUT	likewise for output bytes.

     LOOPFCT		name of the function created.  If not specified
			the name is `loop' but this prevents the use
			of multiple functions in the same file.

     BODY		this is supposed to expand to the body of the loop.
			The user must provide this.

     EXTRA_LOOP_DECLS	extra arguments passed from conversion loop call.

     INIT_PARAMS	code to define and initialize variables from params.
     UPDATE_PARAMS	code to store result in params.

     ONEBYTE_BODY	body of the specialized conversion function for a
			single byte from the current character set to INTERNAL.
*/

#include <assert.h>
#include <endian.h>
#include <iconv/gconv_int.h>
#include <stdint.h>
#include <string.h>
#include <wchar.h>
#include <sys/param.h>		/* For MIN.  */
#define __need_size_t
#include <stddef.h>
#include <libc-diag.h>

/* We have to provide support for machines which are not able to handled
   unaligned memory accesses.  Some of the character encodings have
   representations with a fixed width of 2 or 4 bytes.  But if we cannot
   access unaligned memory we still have to read byte-wise.  */
#undef FCTNAME2
#if _STRING_ARCH_unaligned || !defined DEFINE_UNALIGNED
/* We can handle unaligned memory access.  */
# define get16(addr) *((const uint16_t *) (addr))
# define get32(addr) *((const uint32_t *) (addr))

/* We need no special support for writing values either.  */
# define put16(addr, val) *((uint16_t *) (addr)) = (val)
# define put32(addr, val) *((uint32_t *) (addr)) = (val)

# define FCTNAME2(name) name
#else
/* Distinguish between big endian and little endian.  */
# if __BYTE_ORDER == __LITTLE_ENDIAN
#  define get16(addr) \
     (((const unsigned char *) (addr))[1] << 8				      \
      | ((const unsigned char *) (addr))[0])
#  define get32(addr) \
     (((((const unsigned char *) (addr))[3] << 8			      \
	| ((const unsigned char *) (addr))[2]) << 8			      \
       | ((const unsigned char *) (addr))[1]) << 8			      \
      | ((const unsigned char *) (addr))[0])

#  define put16(addr, val) \
     ({ uint16_t __val = (val);						      \
	((unsigned char *) (addr))[0] = __val;				      \
	((unsigned char *) (addr))[1] = __val >> 8;			      \
	(void) 0; })
#  define put32(addr, val) \
     ({ uint32_t __val = (val);						      \
	((unsigned char *) (addr))[0] = __val;				      \
	__val >>= 8;							      \
	((unsigned char *) (addr))[1] = __val;				      \
	__val >>= 8;							      \
	((unsigned char *) (addr))[2] = __val;				      \
	__val >>= 8;							      \
	((unsigned char *) (addr))[3] = __val;				      \
	(void) 0; })
# else
#  define get16(addr) \
     (((const unsigned char *) (addr))[0] << 8				      \
      | ((const unsigned char *) (addr))[1])
#  define get32(addr) \
     (((((const unsigned char *) (addr))[0] << 8			      \
	| ((const unsigned char *) (addr))[1]) << 8			      \
       | ((const unsigned char *) (addr))[2]) << 8			      \
      | ((const unsigned char *) (addr))[3])

#  define put16(addr, val) \
     ({ uint16_t __val = (val);						      \
	((unsigned char *) (addr))[1] = __val;				      \
	((unsigned char *) (addr))[0] = __val >> 8;			      \
	(void) 0; })
#  define put32(addr, val) \
     ({ uint32_t __val = (val);						      \
	((unsigned char *) (addr))[3] = __val;				      \
	__val >>= 8;							      \
	((unsigned char *) (addr))[2] = __val;				      \
	__val >>= 8;							      \
	((unsigned char *) (addr))[1] = __val;				      \
	__val >>= 8;							      \
	((unsigned char *) (addr))[0] = __val;				      \
	(void) 0; })
# endif

# define FCTNAME2(name) name##_unaligned
#endif
#define FCTNAME(name) FCTNAME2(name)


/* We need at least one byte for the next round.  */
#ifndef MIN_NEEDED_INPUT
# error "MIN_NEEDED_INPUT definition missing"
#elif MIN_NEEDED_INPUT < 1
# error "MIN_NEEDED_INPUT must be >= 1"
#endif

/* Let's see how many bytes we produce.  */
#ifndef MAX_NEEDED_INPUT
# define MAX_NEEDED_INPUT	MIN_NEEDED_INPUT
#endif

/* We produce at least one byte in the next round.  */
#ifndef MIN_NEEDED_OUTPUT
# error "MIN_NEEDED_OUTPUT definition missing"
#elif MIN_NEEDED_OUTPUT < 1
# error "MIN_NEEDED_OUTPUT must be >= 1"
#endif

/* Let's see how many bytes we produce.  */
#ifndef MAX_NEEDED_OUTPUT
# define MAX_NEEDED_OUTPUT	MIN_NEEDED_OUTPUT
#endif

/* Default name for the function.  */
#ifndef LOOPFCT
# define LOOPFCT		loop
#endif

/* Make sure we have a loop body.  */
#ifndef BODY
# error "Definition of BODY missing for function" LOOPFCT
#endif


/* If no arguments have to passed to the loop function define the macro
   as empty.  */
#ifndef EXTRA_LOOP_DECLS
# define EXTRA_LOOP_DECLS
#endif

/* Allow using UPDATE_PARAMS in macros where #ifdef UPDATE_PARAMS test
   isn't possible.  */
#ifndef UPDATE_PARAMS
# define UPDATE_PARAMS do { } while (0)
#endif
#ifndef REINIT_PARAMS
# define REINIT_PARAMS do { } while (0)
#endif


/* To make it easier for the writers of the modules, we define a macro
   to test whether we have to ignore errors.  */
#define ignore_errors_p() \
  (irreversible != NULL && (flags & __GCONV_IGNORE_ERRORS))


/* Error handling for the FROM_LOOP direction, with ignoring of errors.
   Note that we cannot use the do while (0) trick since `break' and
   `continue' must reach certain points.  */
#define STANDARD_FROM_LOOP_ERR_HANDLER(Incr) \
  {									      \
    result = __GCONV_ILLEGAL_INPUT;					      \
									      \
    if (! ignore_errors_p ())						      \
      break;								      \
									      \
    /* We ignore the invalid input byte sequence.  */			      \
    inptr += (Incr);							      \
    ++*irreversible;							      \
    /* But we keep result == __GCONV_ILLEGAL_INPUT, because of the constraint \
       that "iconv -c" must give the same exitcode as "iconv".  */	      \
    continue;								      \
  }

/* Error handling for the TO_LOOP direction, with use of transliteration/
   transcription functions and ignoring of errors.  Note that we cannot use
   the do while (0) trick since `break' and `continue' must reach certain
   points.  */
#define STANDARD_TO_LOOP_ERR_HANDLER(Incr) \
  {									      \
    result = __GCONV_ILLEGAL_INPUT;					      \
									      \
    if (irreversible == NULL)						      \
      /* This means we are in call from __gconv_transliterate.  In this	      \
	 case we are not doing any error recovery outself.  */		      \
      break;								      \
									      \
    /* If needed, flush any conversion state, so that __gconv_transliterate   \
       starts with current shift state.  */				      \
    UPDATE_PARAMS;							      \
									      \
    /* First try the transliteration methods.  */			      \
    if ((step_data->__flags & __GCONV_TRANSLIT) != 0)			      \
      result = __gconv_transliterate					      \
	(step, step_data, *inptrp,					      \
	 &inptr, inend, &outptr, irreversible);			      \
									      \
    REINIT_PARAMS;							      \
									      \
    /* If any of them recognized the input continue with the loop.  */	      \
    if (result != __GCONV_ILLEGAL_INPUT)				      \
      {									      \
	if (__glibc_unlikely (result == __GCONV_FULL_OUTPUT))		      \
	  break;							      \
									      \
	continue;							      \
      }									      \
									      \
    /* Next see whether we have to ignore the error.  If not, stop.  */	      \
    if (! ignore_errors_p ())						      \
      break;								      \
									      \
    /* When we come here it means we ignore the character.  */		      \
    ++*irreversible;							      \
    inptr += Incr;							      \
    /* But we keep result == __GCONV_ILLEGAL_INPUT, because of the constraint \
       that "iconv -c" must give the same exitcode as "iconv".  */	      \
    continue;								      \
  }


/* With GCC 7 when compiling with -Os for 32-bit s390 the compiler
   warns that the variable 'ch', in the definition of BODY in
   sysdeps/s390/multiarch/8bit-generic.c, may be used uninitialized in
   the call to UNICODE_TAG_HANDLER in that macro.  This variable is
   actually always initialized before use, in the prior loop if INDEX
   is nonzero and in the following 'if' if INDEX is zero.  That code
   has a comment referencing this diagnostic disabling; updates in one
   place may require updates in the other.  */
DIAG_PUSH_NEEDS_COMMENT;
DIAG_IGNORE_Os_NEEDS_COMMENT (7, "-Wmaybe-uninitialized");
/* Handling of Unicode 3.1 TAG characters.  Unicode recommends
   "If language codes are not relevant to the particular processing
    operation, then they should be ignored."  This macro is usually
   called right before  STANDARD_TO_LOOP_ERR_HANDLER (Incr).  */
#define UNICODE_TAG_HANDLER(Character, Incr) \
  {									      \
    /* TAG characters are those in the range U+E0000..U+E007F.  */	      \
    if (((Character) >> 7) == (0xe0000 >> 7))				      \
      {									      \
	inptr += Incr;							      \
	continue;							      \
      }									      \
  }
DIAG_POP_NEEDS_COMMENT;


/* The function returns the status, as defined in gconv.h.  */
static inline int
__attribute ((always_inline))
FCTNAME (LOOPFCT) (struct __gconv_step *step,
		   struct __gconv_step_data *step_data,
		   const unsigned char **inptrp, const unsigned char *inend,
		   unsigned char **outptrp, const unsigned char *outend,
		   size_t *irreversible EXTRA_LOOP_DECLS)
{
#ifdef LOOP_NEED_STATE
  mbstate_t *state = step_data->__statep;
#endif
#ifdef LOOP_NEED_FLAGS
  int flags = step_data->__flags;
#endif
#ifdef LOOP_NEED_DATA
  void *data = step->__data;
#endif
  int result = __GCONV_EMPTY_INPUT;
  const unsigned char *inptr = *inptrp;
  unsigned char *outptr = *outptrp;

#ifdef INIT_PARAMS
  INIT_PARAMS;
#endif

  while (inptr != inend)
    {
      /* `if' cases for MIN_NEEDED_OUTPUT ==/!= 1 is made to help the
	 compiler generating better code.  They will be optimized away
	 since MIN_NEEDED_OUTPUT is always a constant.  */
      if (MIN_NEEDED_INPUT > 1
	  && __builtin_expect (inptr + MIN_NEEDED_INPUT > inend, 0))
	{
	  /* We don't have enough input for another complete input
	     character.  */
	  result = __GCONV_INCOMPLETE_INPUT;
	  break;
	}
      if ((MIN_NEEDED_OUTPUT != 1
	   && __builtin_expect (outptr + MIN_NEEDED_OUTPUT > outend, 0))
	  || (MIN_NEEDED_OUTPUT == 1
	      && __builtin_expect (outptr >= outend, 0)))
	{
	  /* Overflow in the output buffer.  */
	  result = __GCONV_FULL_OUTPUT;
	  break;
	}

      /* Here comes the body the user provides.  It can stop with
	 RESULT set to GCONV_INCOMPLETE_INPUT (if the size of the
	 input characters vary in size), GCONV_ILLEGAL_INPUT, or
	 GCONV_FULL_OUTPUT (if the output characters vary in size).  */
      BODY
    }

  /* Update the pointers pointed to by the parameters.  */
  *inptrp = inptr;
  *outptrp = outptr;
  UPDATE_PARAMS;

  return result;
}


/* Include the file a second time to define the function to handle
   unaligned access.  */
#if !defined DEFINE_UNALIGNED && !_STRING_ARCH_unaligned \
    && MIN_NEEDED_INPUT != 1 && MAX_NEEDED_INPUT % MIN_NEEDED_INPUT == 0 \
    && MIN_NEEDED_OUTPUT != 1 && MAX_NEEDED_OUTPUT % MIN_NEEDED_OUTPUT == 0
# undef get16
# undef get32
# undef put16
# undef put32
# undef unaligned

# define DEFINE_UNALIGNED
# include "loop.c"
# undef DEFINE_UNALIGNED
#else
# if MAX_NEEDED_INPUT > 1
#  define SINGLE(fct) SINGLE2 (fct)
#  define SINGLE2(fct) fct##_single
static inline int
__attribute ((always_inline))
SINGLE(LOOPFCT) (struct __gconv_step *step,
		 struct __gconv_step_data *step_data,
		 const unsigned char **inptrp, const unsigned char *inend,
		 unsigned char **outptrp, unsigned char *outend,
		 size_t *irreversible EXTRA_LOOP_DECLS)
{
  mbstate_t *state = step_data->__statep;
#  ifdef LOOP_NEED_FLAGS
  int flags = step_data->__flags;
#  endif
#  ifdef LOOP_NEED_DATA
  void *data = step->__data;
#  endif
  int result = __GCONV_OK;
  unsigned char bytebuf[MAX_NEEDED_INPUT];
  const unsigned char *inptr = *inptrp;
  unsigned char *outptr = *outptrp;
  size_t inlen;

#  ifdef INIT_PARAMS
  INIT_PARAMS;
#  endif

#  ifdef UNPACK_BYTES
  UNPACK_BYTES
#  else
  /* Add the bytes from the state to the input buffer.  */
  assert ((state->__count & 7) <= sizeof (state->__value));
  for (inlen = 0; inlen < (size_t) (state->__count & 7); ++inlen)
    bytebuf[inlen] = state->__value.__wchb[inlen];
#  endif

  /* Are there enough bytes in the input buffer?  */
  if (MIN_NEEDED_INPUT > 1
      && __builtin_expect (inptr + (MIN_NEEDED_INPUT - inlen) > inend, 0))
    {
      *inptrp = inend;
#  ifdef STORE_REST

      /* Building with -O3 GCC emits a `array subscript is above array
	 bounds' warning.  GCC BZ #64739 has been opened for this.  */
      DIAG_PUSH_NEEDS_COMMENT;
      DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Warray-bounds");
      while (inptr < inend)
	bytebuf[inlen++] = *inptr++;
      DIAG_POP_NEEDS_COMMENT;

      inptr = bytebuf;
      inptrp = &inptr;
      inend = &bytebuf[inlen];

      STORE_REST
#  else
      /* We don't have enough input for another complete input
	 character.  */
      size_t inlen_after = inlen + (inend - inptr);
      assert (inlen_after <= sizeof (state->__value.__wchb));
      for (; inlen < inlen_after; inlen++)
	state->__value.__wchb[inlen] = *inptr++;
#  endif

      return __GCONV_INCOMPLETE_INPUT;
    }

  /* Enough space in output buffer.  */
  if ((MIN_NEEDED_OUTPUT != 1 && outptr + MIN_NEEDED_OUTPUT > outend)
      || (MIN_NEEDED_OUTPUT == 1 && outptr >= outend))
    /* Overflow in the output buffer.  */
    return __GCONV_FULL_OUTPUT;

  /*  Now add characters from the normal input buffer.  */
  do
    bytebuf[inlen++] = *inptr++;
  while (inlen < MAX_NEEDED_INPUT && inptr < inend);

  inptr = bytebuf;
  inend = &bytebuf[inlen];

  do
    {
      BODY
    }
  while (0);

  /* Now we either have produced an output character and consumed all the
     bytes from the state and at least one more, or the character is still
     incomplete, or we have some other error (like illegal input character,
     no space in output buffer).  */
  if (__glibc_likely (inptr != bytebuf))
    {
      /* We found a new character.  */
      assert (inptr - bytebuf > (state->__count & 7));

      *inptrp += inptr - bytebuf - (state->__count & 7);
      *outptrp = outptr;

      result = __GCONV_OK;

      /* Clear the state buffer.  */
#  ifdef CLEAR_STATE
      CLEAR_STATE;
#  else
      state->__count &= ~7;
#  endif
    }
  else if (result == __GCONV_INCOMPLETE_INPUT)
    {
      /* This can only happen if we have less than MAX_NEEDED_INPUT bytes
	 available.  */
      assert (inend != &bytebuf[MAX_NEEDED_INPUT]);

      *inptrp += inend - bytebuf - (state->__count & 7);
#  ifdef STORE_REST
      inptrp = &inptr;

      STORE_REST
#  else
      /* We don't have enough input for another complete input
	 character.  */
      assert (inend - inptr > (state->__count & ~7));
      assert (inend - inptr <= sizeof (state->__value.__wchb));
      state->__count = (state->__count & ~7) | (inend - inptr);
      for (inlen = 0; inlen < inend - inptr; inlen++)
	state->__value.__wchb[inlen] = inptr[inlen];
      inptr = inend;
#  endif
    }

  return result;
}
#  undef SINGLE
#  undef SINGLE2
# endif


# ifdef ONEBYTE_BODY
/* Define the shortcut function for btowc.  */
static wint_t
gconv_btowc (struct __gconv_step *step, unsigned char c)
  ONEBYTE_BODY
#  define FROM_ONEBYTE gconv_btowc
# endif

#endif

/* We remove the macro definitions so that we can include this file again
   for the definition of another function.  */
#undef MIN_NEEDED_INPUT
#undef MAX_NEEDED_INPUT
#undef MIN_NEEDED_OUTPUT
#undef MAX_NEEDED_OUTPUT
#undef LOOPFCT
#undef BODY
#undef LOOPFCT
#undef EXTRA_LOOP_DECLS
#undef INIT_PARAMS
#undef UPDATE_PARAMS
#undef REINIT_PARAMS
#undef ONEBYTE_BODY
#undef UNPACK_BYTES
#undef CLEAR_STATE
#undef LOOP_NEED_STATE
#undef LOOP_NEED_FLAGS
#undef LOOP_NEED_DATA
#undef get16
#undef get32
#undef put16
#undef put32
#undef unaligned
