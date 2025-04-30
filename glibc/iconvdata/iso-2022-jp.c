/* Conversion module for ISO-2022-JP and ISO-2022-JP-2.
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

#include <assert.h>
#include <dlfcn.h>
#include <gconv.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "jis0201.h"
#include "jis0208.h"
#include "jis0212.h"
#include "gb2312.h"
#include "ksc5601.h"

struct gap
{
  uint16_t start;
  uint16_t end;
  int32_t idx;
};

#include "iso8859-7jp.h"

/* This makes obvious what everybody knows: 0x1b is the Esc character.  */
#define ESC 0x1b

/* We provide our own initialization and destructor function.  */
#define DEFINE_INIT	0
#define DEFINE_FINI	0

/* Definitions used in the body of the `gconv' function.  */
#define FROM_LOOP		from_iso2022jp_loop
#define TO_LOOP			to_iso2022jp_loop
#define ONE_DIRECTION			0
#define FROM_LOOP_MIN_NEEDED_FROM	1
#define FROM_LOOP_MAX_NEEDED_FROM	4
#define FROM_LOOP_MIN_NEEDED_TO		4
#define FROM_LOOP_MAX_NEEDED_TO		4
#define TO_LOOP_MIN_NEEDED_FROM		4
#define TO_LOOP_MAX_NEEDED_FROM		4
#define TO_LOOP_MIN_NEEDED_TO		1
#define TO_LOOP_MAX_NEEDED_TO		6
#define FROM_DIRECTION		(dir == from_iso2022jp)
#define PREPARE_LOOP \
  enum direction dir = ((struct iso2022jp_data *) step->__data)->dir;	      \
  enum variant var = ((struct iso2022jp_data *) step->__data)->var;	      \
  int save_set;								      \
  int *setp = &data->__statep->__count;
#define EXTRA_LOOP_ARGS		, var, setp


/* Direction of the transformation.  */
enum direction
{
  illegal_dir,
  to_iso2022jp,
  from_iso2022jp
};

/* We handle ISO-2022-jp and ISO-2022-JP-2 here.  */
enum variant
{
  illegal_var,
  iso2022jp,
  iso2022jp2
};


struct iso2022jp_data
{
  enum direction dir;
  enum variant var;
};


/* The COUNT element of the state keeps track of the currently selected
   character set.  The possible values are:  */
enum
{
  ASCII_set = 0,
  JISX0208_1978_set = 1 << 3,
  JISX0208_1983_set = 2 << 3,
  JISX0201_Roman_set = 3 << 3,
  JISX0201_Kana_set = 4 << 3,
  GB2312_set = 5 << 3,
  KSC5601_set = 6 << 3,
  JISX0212_set = 7 << 3,
  CURRENT_SEL_MASK = 7 << 3
};

/* The second value stored is the designation of the G2 set.  The following
   values are possible:  */
enum
{
  UNSPECIFIED_set = 0,
  ISO88591_set = 1 << 6,
  ISO88597_set = 2 << 6,
  CURRENT_ASSIGN_MASK = 3 << 6
};

/* The third value, only used during conversion from Unicode to ISO-2022-JP-2,
   describes the language tag parsing status.  The possible values are as
   follows.  Values >= TAG_language are temporary tag parsing states.  */
enum
{
  TAG_none = 0,
  TAG_language = 4 << 8,
  TAG_language_j = 5 << 8,
  TAG_language_ja = 1 << 8,
  TAG_language_k = 6 << 8,
  TAG_language_ko = 2 << 8,
  TAG_language_z = 7 << 8,
  TAG_language_zh = 3 << 8,
  CURRENT_TAG_MASK = 7 << 8
};


extern int gconv_init (struct __gconv_step *step);
int
gconv_init (struct __gconv_step *step)
{
  /* Determine which direction.  */
  struct iso2022jp_data *new_data;
  enum direction dir = illegal_dir;
  enum variant var = illegal_var;
  int result;

  if (__strcasecmp (step->__from_name, "ISO-2022-JP//") == 0)
    {
      dir = from_iso2022jp;
      var = iso2022jp;
    }
  else if (__strcasecmp (step->__to_name, "ISO-2022-JP//") == 0)
    {
      dir = to_iso2022jp;
      var = iso2022jp;
    }
  else if (__strcasecmp (step->__from_name, "ISO-2022-JP-2//") == 0)
    {
      dir = from_iso2022jp;
      var = iso2022jp2;
    }
  else if (__strcasecmp (step->__to_name, "ISO-2022-JP-2//") == 0)
    {
      dir = to_iso2022jp;
      var = iso2022jp2;
    }

  result = __GCONV_NOCONV;
  if (__builtin_expect (dir, from_iso2022jp) != illegal_dir)
    {
      new_data
	= (struct iso2022jp_data *) malloc (sizeof (struct iso2022jp_data));

      result = __GCONV_NOMEM;
      if (new_data != NULL)
	{
	  new_data->dir = dir;
	  new_data->var = var;
	  step->__data = new_data;

	  if (dir == from_iso2022jp)
	    {
	      step->__min_needed_from = FROM_LOOP_MIN_NEEDED_FROM;
	      step->__max_needed_from = FROM_LOOP_MAX_NEEDED_FROM;
	      step->__min_needed_to = FROM_LOOP_MIN_NEEDED_TO;
	      step->__max_needed_to = FROM_LOOP_MAX_NEEDED_TO;
	    }
	  else
	    {
	      step->__min_needed_from = TO_LOOP_MIN_NEEDED_FROM;
	      step->__max_needed_from = TO_LOOP_MAX_NEEDED_FROM;
	      step->__min_needed_to = TO_LOOP_MIN_NEEDED_TO;
	      step->__max_needed_to = TO_LOOP_MAX_NEEDED_TO;
	    }

	  /* Yes, this is a stateful encoding.  */
	  step->__stateful = 1;

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


/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  /* Avoid warning about unused variable 'var'.  */			      \
  (void) var;								      \
									      \
  if ((data->__statep->__count & ~7) != ASCII_set)			      \
    {									      \
      if (dir == from_iso2022jp						      \
	  || (data->__statep->__count & CURRENT_SEL_MASK) == ASCII_set)	      \
	{								      \
	  /* It's easy, we don't have to emit anything, we just reset the     \
	     state for the input.  Note that this also clears the G2	      \
	     designation.  */						      \
	  data->__statep->__count &= 7;					      \
	  data->__statep->__count |= ASCII_set;				      \
	}								      \
      else								      \
	{								      \
	  /* We are not in the initial state.  To switch back we have	      \
	     to emit the sequence `Esc ( B'.  */			      \
	  if (__glibc_unlikely (outbuf + 3 > outend))			      \
	    /* We don't have enough room in the output buffer.  */	      \
	    status = __GCONV_FULL_OUTPUT;				      \
	  else								      \
	    {								      \
	      /* Write out the shift sequence.  */			      \
	      *outbuf++ = ESC;						      \
	      *outbuf++ = '(';						      \
	      *outbuf++ = 'B';						      \
	      /* Note that this also clears the G2 designation.  */	      \
	      data->__statep->__count &= 7;				      \
	      data->__statep->__count |= ASCII_set;			      \
	    }								      \
	}								      \
    }


/* Since we might have to reset input pointer we must be able to save
   and retore the state.  */
#define SAVE_RESET_STATE(Save) \
  if (Save)								      \
    save_set = *setp;							      \
  else									      \
    *setp = save_set


/* First define the conversion function from ISO-2022-JP to UCS4.  */
#define MIN_NEEDED_INPUT	FROM_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	FROM_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	FROM_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	FROM_LOOP_MAX_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
									      \
    /* Recognize escape sequences.  */					      \
    if (__builtin_expect (ch, 0) == ESC)				      \
      {									      \
	/* We now must be prepared to read two to three more		      \
	   characters.  If we have a match in the first character but	      \
	   then the input buffer ends we terminate with an error since	      \
	   we must not risk missing an escape sequence just because it	      \
	   is not entirely in the current input buffer.  */		      \
	if (__builtin_expect (inptr + 2 >= inend, 0)			      \
	    || (var == iso2022jp2 && inptr[1] == '$' && inptr[2] == '('	      \
		&& __builtin_expect (inptr + 3 >= inend, 0)))		      \
	  {								      \
	    /* Not enough input available.  */				      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
									      \
	if (inptr[1] == '(')						      \
	  {								      \
	    if (inptr[2] == 'B')					      \
	      {								      \
		/* ASCII selected.  */					      \
		set = ASCII_set;					      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (inptr[2] == 'J')					      \
	      {								      \
		/* JIS X 0201 selected.  */				      \
		set = JISX0201_Roman_set;				      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (var == iso2022jp2 && inptr[2] == 'I')		      \
	      {								      \
		/* JIS X 0201 selected.  */				      \
		set = JISX0201_Kana_set;				      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	  }								      \
	else if (inptr[1] == '$')					      \
	  {								      \
	    if (inptr[2] == '@')					      \
	      {								      \
		/* JIS X 0208-1978 selected.  */			      \
		set = JISX0208_1978_set;				      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (inptr[2] == 'B')					      \
	      {								      \
		/* JIS X 0208-1983 selected.  */			      \
		set = JISX0208_1983_set;				      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (var == iso2022jp2)					      \
	      {								      \
		if (inptr[2] == 'A')					      \
		  {							      \
		    /* GB 2312-1980 selected.  */			      \
		    set = GB2312_set;					      \
		    inptr += 3;						      \
		    continue;						      \
		  }							      \
		else if (inptr[2] == '(')				      \
		  {							      \
		    if (inptr[3] == 'C')				      \
		      {							      \
			/* KSC 5601-1987 selected.  */			      \
			set = KSC5601_set;				      \
			inptr += 4;					      \
			continue;					      \
		      }							      \
		    else if (inptr[3] == 'D')				      \
		      {							      \
			/* JIS X 0212-1990 selected.  */		      \
			set = JISX0212_set;				      \
			inptr += 4;					      \
			continue;					      \
		      }							      \
		  }							      \
	      }								      \
	  }								      \
	else if (var == iso2022jp2 && inptr[1] == '.')			      \
	  {								      \
	    if (inptr[2] == 'A')					      \
	      {								      \
		/* ISO 8859-1-GR selected.  */				      \
		set2 = ISO88591_set;					      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (inptr[2] == 'F')					      \
	      {								      \
		/* ISO 8859-7-GR selected.  */				      \
		set2 = ISO88597_set;					      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	  }								      \
      }									      \
									      \
    if (ch == ESC && var == iso2022jp2 && inptr[1] == 'N')		      \
      {									      \
	if (set2 == ISO88591_set)					      \
	  {								      \
	    ch = inptr[2] | 0x80;					      \
	    inptr += 3;							      \
	  }								      \
	else if (__builtin_expect (set2, ISO88597_set) == ISO88597_set)	      \
	  {								      \
	    /* We use the table from the ISO 8859-7 module.  */		      \
	    if (inptr[2] < 0x20 || inptr[2] >= 0x80)			      \
	      STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
	    ch = iso88597_to_ucs4[inptr[2] - 0x20];			      \
	    if (ch == 0)						      \
	      STANDARD_FROM_LOOP_ERR_HANDLER (3);			      \
	    inptr += 3;							      \
	  }								      \
	else								      \
	  {								      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
      }									      \
    else if (ch >= 0x80)						      \
      {									      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
    else if (set == ASCII_set || (ch < 0x21 || ch == 0x7f))		      \
      /* Almost done, just advance the input pointer.  */		      \
      ++inptr;								      \
    else if (set == JISX0201_Roman_set)					      \
      {									      \
	/* Use the JIS X 0201 table.  */				      \
	ch = jisx0201_to_ucs4 (ch);					      \
	if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	  STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	++inptr;							      \
      }									      \
    else if (set == JISX0201_Kana_set)					      \
      {									      \
	/* Use the JIS X 0201 table.  */				      \
	ch = jisx0201_to_ucs4 (ch + 0x80);				      \
	if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	  STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	++inptr;							      \
      }									      \
    else								      \
      {									      \
	if (set == JISX0208_1978_set || set == JISX0208_1983_set)	      \
	  /* XXX I don't have the tables for these two old variants of	      \
	     JIS X 0208.  Therefore I'm using the tables for JIS X	      \
	     0208-1990.  If somebody has problems with this please	      \
	     provide the appropriate tables.  */			      \
	  ch = jisx0208_to_ucs4 (&inptr, inend - inptr, 0);		      \
	else if (set == JISX0212_set)					      \
	  /* Use the JIS X 0212 table.  */				      \
	  ch = jisx0212_to_ucs4 (&inptr, inend - inptr, 0);		      \
	else if (set == GB2312_set)					      \
	  /* Use the GB 2312 table.  */					      \
	  ch = gb2312_to_ucs4 (&inptr, inend - inptr, 0);		      \
	else								      \
	  {								      \
	    assert (set == KSC5601_set);				      \
									      \
	    /* Use the KSC 5601 table.  */				      \
	    ch = ksc5601_to_ucs4 (&inptr, inend - inptr, 0);		      \
	  }								      \
									      \
	if (__glibc_unlikely (ch == 0))					      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	else if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	  {								      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, enum variant var, int *setp
#define INIT_PARAMS		int set = *setp & CURRENT_SEL_MASK;	      \
				int set2 = *setp & CURRENT_ASSIGN_MASK
#define UPDATE_PARAMS		*setp = set | set2
#include <iconv/loop.c>


/* Next, define the other direction.  */

enum conversion { none = 0, european, japanese, chinese, korean, other };

/* A datatype for conversion lists.  */
typedef unsigned int cvlist_t;
#define CVLIST(cv1, cv2, cv3, cv4, cv5) \
  ((cv1) + ((cv2) << 3) + ((cv3) << 6) + ((cv4) << 9) + ((cv5) << 12))
#define CVLIST_FIRST(cvl) ((cvl) & ((1 << 3) - 1))
#define CVLIST_REST(cvl) ((cvl) >> 3)
static const cvlist_t conversion_lists[4] =
  {
    /* TAG_none */        CVLIST (japanese, european, chinese, korean, other),
    /* TAG_language_ja */ CVLIST (japanese, european, chinese, korean, other),
    /* TAG_language_ko */ CVLIST (korean, european, japanese, chinese, other),
    /* TAG_language_zh */ CVLIST (chinese, european, japanese, korean, other)
  };

#define MIN_NEEDED_INPUT	TO_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	TO_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	TO_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	TO_LOOP_MAX_NEEDED_TO
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch;							      \
    size_t written;							      \
									      \
    ch = get32 (inptr);							      \
									      \
    if (var == iso2022jp2)						      \
      {									      \
	/* Handle Unicode tag characters (range U+E0000..U+E007F).  */	      \
	if (__glibc_unlikely ((ch >> 7) == (0xe0000 >> 7)))		      \
	  {								      \
	    ch &= 0x7f;							      \
	    if (ch >= 'A' && ch <= 'Z')					      \
	      ch += 'a' - 'A';						      \
	    if (ch == 0x01)						      \
	      tag = TAG_language;					      \
	    else if (ch == 'j' && tag == TAG_language)			      \
	      tag = TAG_language_j;					      \
	    else if (ch == 'a' && tag == TAG_language_j)		      \
	      tag = TAG_language_ja;					      \
	    else if (ch == 'k' && tag == TAG_language)			      \
	      tag = TAG_language_k;					      \
	    else if (ch == 'o' && tag == TAG_language_k)		      \
	      tag = TAG_language_ko;					      \
	    else if (ch == 'z' && tag == TAG_language)			      \
	      tag = TAG_language_z;					      \
	    else if (ch == 'h' && tag == TAG_language_z)		      \
	      tag = TAG_language_zh;					      \
	    else if (ch == 0x7f)					      \
	      tag = TAG_none;						      \
	    else							      \
	      {								      \
		/* Other tag characters reset the tag parsing state (if the   \
		   current state is a temporary state) or are ignored (if     \
		   the current state is a stable one).  */		      \
		if (tag >= TAG_language)				      \
		  tag = TAG_none;					      \
	      }								      \
									      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
									      \
	/* Non-tag characters reset the tag parsing state, if the current     \
	   state is a temporary state.  */				      \
	if (__glibc_unlikely (tag >= TAG_language))			      \
	  tag = TAG_none;						      \
      }									      \
									      \
    /* First see whether we can write the character using the currently	      \
       selected character set.  But ignore the selected character set if      \
       the current language tag shows different preferences.  */	      \
    if (set == ASCII_set)						      \
      {									      \
	/* Please note that the NUL byte is *not* matched if we are not	      \
	   currently using the ASCII charset.  This is because we must	      \
	   switch to the initial state whenever a NUL byte is written.  */    \
	if (ch <= 0x7f)							      \
	  {								      \
	    *outptr++ = ch;						      \
	    written = 1;						      \
									      \
	    /* At the beginning of a line, G2 designation is cleared.  */     \
	    if (var == iso2022jp2 && ch == 0x0a)			      \
	      set2 = UNSPECIFIED_set;					      \
	  }								      \
	else								      \
	  written = __UNKNOWN_10646_CHAR;				      \
      }									      \
    /* ISO-2022-JP recommends to encode the newline character always in	      \
       ASCII since this allows a context-free interpretation of the	      \
       characters at the beginning of the next line.  Otherwise it would      \
       have to be known whether the last line ended using ASCII or	      \
       JIS X 0201.  */							      \
    else if (set == JISX0201_Roman_set					      \
	     && (__builtin_expect (tag == TAG_none, 1)			      \
		 || tag == TAG_language_ja))				      \
      {									      \
	unsigned char buf[1];						      \
	written = ucs4_to_jisx0201 (ch, buf);				      \
	if (written != __UNKNOWN_10646_CHAR)				      \
	  {								      \
	    if (buf[0] > 0x20 && buf[0] < 0x80)				      \
	      {								      \
		*outptr++ = buf[0];					      \
		written = 1;						      \
	      }								      \
	    else							      \
	      written = __UNKNOWN_10646_CHAR;				      \
	  }								      \
      }									      \
    else if (set == JISX0201_Kana_set					      \
	     && (__builtin_expect (tag == TAG_none, 1)			      \
		 || tag == TAG_language_ja))				      \
      {									      \
	unsigned char buf[1];						      \
	written = ucs4_to_jisx0201 (ch, buf);				      \
	if (written != __UNKNOWN_10646_CHAR)				      \
	  {								      \
	    if (buf[0] > 0xa0 && buf[0] < 0xe0)				      \
	      {								      \
		*outptr++ = buf[0] - 0x80;				      \
		written = 1;						      \
	      }								      \
	    else							      \
	      written = __UNKNOWN_10646_CHAR;				      \
	  }								      \
      }									      \
    else								      \
      {									      \
	if ((set == JISX0208_1978_set || set == JISX0208_1983_set)	      \
	    && (__builtin_expect (tag == TAG_none, 1)			      \
		|| tag == TAG_language_ja))				      \
	  written = ucs4_to_jisx0208 (ch, outptr, outend - outptr);	      \
	else if (set == JISX0212_set					      \
		 && (__builtin_expect (tag == TAG_none, 1)		      \
		     || tag == TAG_language_ja))			      \
	  written = ucs4_to_jisx0212 (ch, outptr, outend - outptr);	      \
	else if (set == GB2312_set					      \
		 && (__builtin_expect (tag == TAG_none, 1)		      \
		     || tag == TAG_language_zh))			      \
	  written = ucs4_to_gb2312 (ch, outptr, outend - outptr);	      \
	else if (set == KSC5601_set					      \
		 && (__builtin_expect (tag == TAG_none, 1)		      \
		     || tag == TAG_language_ko))			      \
	  written = ucs4_to_ksc5601 (ch, outptr, outend - outptr);	      \
	else								      \
	  written = __UNKNOWN_10646_CHAR;				      \
									      \
	if (__glibc_unlikely (written == 0))				      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	else if (written != __UNKNOWN_10646_CHAR)			      \
	  outptr += written;						      \
      }									      \
									      \
    if (written == __UNKNOWN_10646_CHAR					      \
	&& __builtin_expect (tag == TAG_none, 1))			      \
      {									      \
	if (set2 == ISO88591_set)					      \
	  {								      \
	    if (ch >= 0x80 && ch <= 0xff)				      \
	      {								      \
		if (__glibc_unlikely (outptr + 3 > outend))		      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
									      \
		*outptr++ = ESC;					      \
		*outptr++ = 'N';					      \
		*outptr++ = ch & 0x7f;					      \
		written = 3;						      \
	      }								      \
	  }								      \
	else if (set2 == ISO88597_set)					      \
	  {								      \
	    if (__glibc_likely (ch < 0xffff))				      \
	      {								      \
		const struct gap *rp = from_idx;			      \
									      \
		while (ch > rp->end)					      \
		  ++rp;							      \
		if (ch >= rp->start)					      \
		  {							      \
		    unsigned char res =					      \
		      iso88597_from_ucs4[ch - 0xa0 + rp->idx];		      \
		    if (res != '\0')					      \
		      {							      \
			if (__glibc_unlikely (outptr + 3 > outend))	      \
			  {						      \
			    result = __GCONV_FULL_OUTPUT;		      \
			    break;					      \
			  }						      \
									      \
			*outptr++ = ESC;				      \
			*outptr++ = 'N';				      \
			*outptr++ = res & 0x7f;				      \
			written = 3;					      \
		      }							      \
		  }							      \
	      }								      \
	  }								      \
      }									      \
									      \
    if (written == __UNKNOWN_10646_CHAR)				      \
      {									      \
	/* The attempts to use the currently selected character set	      \
	   failed, either because the language tag changed, or because	      \
	   the character requires a different character set, or because	      \
	   the character is unknown.					      \
	   The CJK character sets partially overlap when seen as subsets      \
	   of ISO 10646; therefore there is no single correct result.	      \
	   We use a preferrence order which depends on the language tag.  */  \
									      \
	if (ch <= 0x7f)							      \
	  {								      \
	    /* We must encode using ASCII.  First write out the		      \
	       escape sequence.  */					      \
	    if (__glibc_unlikely (outptr + 3 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    *outptr++ = ESC;						      \
	    *outptr++ = '(';						      \
	    *outptr++ = 'B';						      \
	    set = ASCII_set;						      \
									      \
	    if (__glibc_unlikely (outptr + 1 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    *outptr++ = ch;						      \
									      \
	    /* At the beginning of a line, G2 designation is cleared.  */     \
	    if (var == iso2022jp2 && ch == 0x0a)			      \
	      set2 = UNSPECIFIED_set;					      \
	  }								      \
	else								      \
	  {								      \
	    /* Now it becomes difficult.  We must search the other	      \
	       character sets one by one.  Use an ordered conversion	      \
	       list that depends on the current language tag.  */	      \
	    cvlist_t conversion_list;					      \
	    unsigned char buf[2];					      \
	    int res = __GCONV_ILLEGAL_INPUT;				      \
									      \
	    if (var == iso2022jp2)					      \
	      conversion_list = conversion_lists[tag >> 8];		      \
	    else							      \
	      conversion_list = CVLIST (japanese, 0, 0, 0, 0);		      \
									      \
	    do								      \
	      switch (CVLIST_FIRST (conversion_list))			      \
		{							      \
		case european:						      \
									      \
		  /* Try ISO 8859-1 upper half.   */			      \
		  if (ch >= 0x80 && ch <= 0xff)				      \
		    {							      \
		      if (set2 != ISO88591_set)				      \
			{						      \
			  if (__builtin_expect (outptr + 3 > outend, 0))      \
			    {						      \
			      res = __GCONV_FULL_OUTPUT;		      \
			      break;					      \
			    }						      \
			  *outptr++ = ESC;				      \
			  *outptr++ = '.';				      \
			  *outptr++ = 'A';				      \
			  set2 = ISO88591_set;				      \
			}						      \
									      \
		      if (__glibc_unlikely (outptr + 3 > outend))	      \
			{						      \
			  res = __GCONV_FULL_OUTPUT;			      \
			  break;					      \
			}						      \
		      *outptr++ = ESC;					      \
		      *outptr++ = 'N';					      \
		      *outptr++ = ch - 0x80;				      \
		      res = __GCONV_OK;					      \
		      break;						      \
		    }							      \
									      \
		  /* Try ISO 8859-7 upper half.  */			      \
		  if (__glibc_likely (ch < 0xffff))			      \
		    {							      \
		      const struct gap *rp = from_idx;			      \
									      \
		      while (ch > rp->end)				      \
			++rp;						      \
		      if (ch >= rp->start)				      \
			{						      \
			  unsigned char ch2 =				      \
			    iso88597_from_ucs4[ch - 0xa0 + rp->idx];	      \
			  if (ch2 != '\0')				      \
			    {						      \
			      if (set2 != ISO88597_set)			      \
				{					      \
				  if (__builtin_expect (outptr + 3 > outend,  \
							0))		      \
				    {					      \
				      res = __GCONV_FULL_OUTPUT;	      \
				      break;				      \
				    }					      \
				  *outptr++ = ESC;			      \
				  *outptr++ = '.';			      \
				  *outptr++ = 'F';			      \
				  set2 = ISO88597_set;			      \
				}					      \
									      \
			      if (__builtin_expect (outptr + 3 > outend, 0))  \
				{					      \
				  res = __GCONV_FULL_OUTPUT;		      \
				  break;				      \
				}					      \
			      *outptr++ = ESC;				      \
			      *outptr++ = 'N';				      \
			      *outptr++ = ch2 - 0x80;			      \
			      res = __GCONV_OK;				      \
			      break;					      \
			    }						      \
			}						      \
		    }							      \
									      \
		  break;						      \
									      \
		case japanese:						      \
									      \
		  /* Try JIS X 0201 Roman.  */				      \
		  written = ucs4_to_jisx0201 (ch, buf);			      \
		  if (written != __UNKNOWN_10646_CHAR			      \
		      && buf[0] > 0x20 && buf[0] < 0x80)		      \
		    {							      \
		      if (set != JISX0201_Roman_set)			      \
			{						      \
			  if (__builtin_expect (outptr + 3 > outend, 0))      \
			    {						      \
			      res = __GCONV_FULL_OUTPUT;		      \
			      break;					      \
			    }						      \
			  *outptr++ = ESC;				      \
			  *outptr++ = '(';				      \
			  *outptr++ = 'J';				      \
			  set = JISX0201_Roman_set;			      \
			}						      \
									      \
		      if (__glibc_unlikely (outptr + 1 > outend))	      \
			{						      \
			  res = __GCONV_FULL_OUTPUT;			      \
			  break;					      \
			}						      \
		      *outptr++ = buf[0];				      \
		      res = __GCONV_OK;					      \
		      break;						      \
		    }							      \
									      \
		  /* Try JIS X 0208.  */				      \
		  written = ucs4_to_jisx0208 (ch, buf, 2);		      \
		  if (written != __UNKNOWN_10646_CHAR)			      \
		    {							      \
		      if (set != JISX0208_1983_set)			      \
			{						      \
			  if (__builtin_expect (outptr + 3 > outend, 0))      \
			    {						      \
			      res = __GCONV_FULL_OUTPUT;		      \
			      break;					      \
			    }						      \
			  *outptr++ = ESC;				      \
			  *outptr++ = '$';				      \
			  *outptr++ = 'B';				      \
			  set = JISX0208_1983_set;			      \
			}						      \
									      \
		      if (__glibc_unlikely (outptr + 2 > outend))	      \
			{						      \
			  res = __GCONV_FULL_OUTPUT;			      \
			  break;					      \
			}						      \
		      *outptr++ = buf[0];				      \
		      *outptr++ = buf[1];				      \
		      res = __GCONV_OK;					      \
		      break;						      \
		    }							      \
									      \
		  if (__glibc_unlikely (var == iso2022jp))		      \
		    /* Don't use the other Japanese character sets.  */	      \
		    break;						      \
									      \
		  /* Try JIS X 0212.  */				      \
		  written = ucs4_to_jisx0212 (ch, buf, 2);		      \
		  if (written != __UNKNOWN_10646_CHAR)			      \
		    {							      \
		      if (set != JISX0212_set)				      \
			{						      \
			  if (__builtin_expect (outptr + 4 > outend, 0))      \
			    {						      \
			      res = __GCONV_FULL_OUTPUT;		      \
			      break;					      \
			    }						      \
			  *outptr++ = ESC;				      \
			  *outptr++ = '$';				      \
			  *outptr++ = '(';				      \
			  *outptr++ = 'D';				      \
			  set = JISX0212_set;				      \
			}						      \
									      \
		      if (__glibc_unlikely (outptr + 2 > outend))	      \
			{						      \
			  res = __GCONV_FULL_OUTPUT;			      \
			  break;					      \
			}						      \
		      *outptr++ = buf[0];				      \
		      *outptr++ = buf[1];				      \
		      res = __GCONV_OK;					      \
		      break;						      \
		    }							      \
									      \
		  break;						      \
									      \
		case chinese:						      \
		  assert (var == iso2022jp2);				      \
									      \
		  /* Try GB 2312.  */					      \
		  written = ucs4_to_gb2312 (ch, buf, 2);		      \
		  if (written != __UNKNOWN_10646_CHAR)			      \
		    {							      \
		      if (set != GB2312_set)				      \
			{						      \
			  if (__builtin_expect (outptr + 3 > outend, 0))      \
			    {						      \
			      res = __GCONV_FULL_OUTPUT;		      \
			      break;					      \
			    }						      \
			  *outptr++ = ESC;				      \
			  *outptr++ = '$';				      \
			  *outptr++ = 'A';				      \
			  set = GB2312_set;				      \
			}						      \
									      \
		      if (__glibc_unlikely (outptr + 2 > outend))	      \
			{						      \
			  res = __GCONV_FULL_OUTPUT;			      \
			  break;					      \
			}						      \
		      *outptr++ = buf[0];				      \
		      *outptr++ = buf[1];				      \
		      res = __GCONV_OK;					      \
		      break;						      \
		    }							      \
									      \
		  break;						      \
									      \
		case korean:						      \
		  assert (var == iso2022jp2);				      \
									      \
		  /* Try KSC 5601.  */					      \
		  written = ucs4_to_ksc5601 (ch, buf, 2);		      \
		  if (written != __UNKNOWN_10646_CHAR)			      \
		    {							      \
		      if (set != KSC5601_set)				      \
			{						      \
			  if (__builtin_expect (outptr + 4 > outend, 0))      \
			    {						      \
			      res = __GCONV_FULL_OUTPUT;		      \
			      break;					      \
			    }						      \
			  *outptr++ = ESC;				      \
			  *outptr++ = '$';				      \
			  *outptr++ = '(';				      \
			  *outptr++ = 'C';				      \
			  set = KSC5601_set;				      \
			}						      \
									      \
		      if (__glibc_unlikely (outptr + 2 > outend))	      \
			{						      \
			  res = __GCONV_FULL_OUTPUT;			      \
			  break;					      \
			}						      \
		      *outptr++ = buf[0];				      \
		      *outptr++ = buf[1];				      \
		      res = __GCONV_OK;					      \
		      break;						      \
		    }							      \
									      \
		  break;						      \
									      \
		case other:						      \
		  assert (var == iso2022jp2);				      \
									      \
		  /* Try JIS X 0201 Kana.  This is not officially part	      \
		     of ISO-2022-JP-2, according to RFC 1554.  Therefore      \
		     we try this only after all other attempts.  */	      \
		  written = ucs4_to_jisx0201 (ch, buf);			      \
		  if (written != __UNKNOWN_10646_CHAR && buf[0] >= 0x80)      \
		    {							      \
		      if (set != JISX0201_Kana_set)			      \
			{						      \
			  if (__builtin_expect (outptr + 3 > outend, 0))      \
			    {						      \
			      res = __GCONV_FULL_OUTPUT;		      \
			      break;					      \
			    }						      \
			  *outptr++ = ESC;				      \
			  *outptr++ = '(';				      \
			  *outptr++ = 'I';				      \
			  set = JISX0201_Kana_set;			      \
			}						      \
									      \
		      if (__glibc_unlikely (outptr + 1 > outend))	      \
			{						      \
			  res = __GCONV_FULL_OUTPUT;			      \
			  break;					      \
			}						      \
		      *outptr++ = buf[0] - 0x80;			      \
		      res = __GCONV_OK;					      \
		      break;						      \
		    }							      \
									      \
		  break;						      \
									      \
		default:						      \
		  abort ();						      \
		}							      \
	    while (res == __GCONV_ILLEGAL_INPUT				      \
		   && (conversion_list = CVLIST_REST (conversion_list)) != 0);\
									      \
	    if (res == __GCONV_FULL_OUTPUT)				      \
	      {								      \
		result = res;						      \
		break;							      \
	      }								      \
									      \
	    if (res == __GCONV_ILLEGAL_INPUT)				      \
	      {								      \
		STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
	      }								      \
	  }								      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, enum variant var, int *setp
#define INIT_PARAMS		int set = *setp & CURRENT_SEL_MASK;	      \
				int set2 = *setp & CURRENT_ASSIGN_MASK;	      \
				int tag = *setp & CURRENT_TAG_MASK;
#define REINIT_PARAMS		do					      \
				  {					      \
				    set = *setp & CURRENT_SEL_MASK;	      \
				    set2 = *setp & CURRENT_ASSIGN_MASK;	      \
				    tag = *setp & CURRENT_TAG_MASK;	      \
				  }					      \
				while (0)
#define UPDATE_PARAMS		*setp = set | set2 | tag
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
