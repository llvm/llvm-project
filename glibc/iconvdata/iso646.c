/* Conversion to and from the various ISO 646 CCS.
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

/* The implementation of the conversion which can be performed by this
   module are not very sophisticated and not tuned at all.  There are
   zillions of ISO 646 derivates and supporting them all in a separate
   module is overkill since these coded character sets are hardly ever
   used anymore (except ANSI_X3.4-1968 == ASCII, which is compatible
   with ISO 8859-1).  The European variants are superceded by the
   various ISO 8859-? standards and the Asian variants are embedded in
   larger character sets.  Therefore this implementation is simply
   here to make it possible to do the conversion if it is necessary.
   The cost in the gconv-modules file is set to `2' and therefore
   allows one to easily provide a tuned implementation in case this
   proofs to be necessary.  */

#include <dlfcn.h>
#include <gconv.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Definitions used in the body of the `gconv' function.  */
#define FROM_LOOP		from_ascii
#define TO_LOOP			to_ascii
#define DEFINE_INIT		0
#define DEFINE_FINI		0
#define MIN_NEEDED_FROM		1
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0

#define FROM_DIRECTION		(dir == from_iso646)
#define PREPARE_LOOP \
  enum direction dir = ((struct iso646_data *) step->__data)->dir;	      \
  enum variant var = ((struct iso646_data *) step->__data)->var;
#define EXTRA_LOOP_ARGS		, var


/* Direction of the transformation.  */
enum direction
{
  illegal_dir,
  to_iso646,
  from_iso646
};

/* See names below, must be in the same order.  */
enum variant
{
  GB,		/* BS_4730 */
  CA,		/* CSA_Z243.4-1985-1 */
  CA2,		/* CSA_Z243.4-1985-2 */
  DE,		/* DIN_66003 */
  DK,		/* DS_2089 */
  ES,		/* ES */
  ES2,		/* ES2 */
  CN,		/* GB_1988-80 */
  IT,		/* IT */
  JP,		/* JIS_C6220-1969-RO */
  JP_OCR_B,	/* JIS_C6229-1984-B */
  YU,		/* JUS_I.B1.002 */
  KR,		/* KSC5636 */
  HU,		/* MSZ_7795.3 */
  CU,		/* NC_NC00-10 */
  FR,		/* NF_Z_62-010 */
  FR1,		/* NF_Z_62-010_(1973) */
  NO,		/* NS_4551-1 */
  NO2,		/* NS_4551-2 */
  PT,		/* PT */
  PT2,		/* PT2 */
  SE,		/* SEN_850200_B */
  SE2		/* SEN_850200_C */
};

/* Must be in the same order as enum variant above.  */
static const char names[] =
  "BS_4730//\0"
  "CSA_Z243.4-1985-1//\0"
  "CSA_Z243.4-1985-2//\0"
  "DIN_66003//\0"
  "DS_2089//\0"
  "ES//\0"
  "ES2//\0"
  "GB_1988-80//\0"
  "IT//\0"
  "JIS_C6220-1969-RO//\0"
  "JIS_C6229-1984-B//\0"
  "JUS_I.B1.002//\0"
  "KSC5636//\0"
  "MSZ_7795.3//\0"
  "NC_NC00-10//\0"
  "NF_Z_62-010//\0"
  "NF_Z_62-010_1973//\0" /* Note that we don't have the parenthesis in
			    the name.  */
  "NS_4551-1//\0"
  "NS_4551-2//\0"
  "PT//\0"
  "PT2//\0"
  "SEN_850200_B//\0"
  "SEN_850200_C//\0"
  "\0";

struct iso646_data
{
  enum direction dir;
  enum variant var;
};


extern int gconv_init (struct __gconv_step *step);
int
gconv_init (struct __gconv_step *step)
{
  /* Determine which direction.  */
  struct iso646_data *new_data;
  enum direction dir = illegal_dir;
  int result;

  enum variant var = 0;
  for (const char *name = names; *name != '\0';
       name = __rawmemchr (name, '\0') + 1)
    {
      if (__strcasecmp (step->__from_name, name) == 0)
	{
	  dir = from_iso646;
	  break;
	}
      else if (__strcasecmp (step->__to_name, name) == 0)
	{
	  dir = to_iso646;
	  break;
	}
      ++var;
    }

  result = __GCONV_NOCONV;
  if (__builtin_expect (dir, from_iso646) != illegal_dir)
    {
      new_data = (struct iso646_data *) malloc (sizeof (struct iso646_data));

      result = __GCONV_NOMEM;
      if (new_data != NULL)
	{
	  new_data->dir = dir;
	  new_data->var = var;
	  step->__data = new_data;

	  if (dir == from_iso646)
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


/* First define the conversion function from ASCII to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch;							      \
    int failure = __GCONV_OK;						      \
									      \
    ch = *inptr;							      \
    switch (ch)								      \
      {									      \
      case 0x23:							      \
	if (var == GB || var == ES || var == IT || var == FR || var == FR1)   \
	  ch = 0xa3;							      \
	else if (var == NO2)						      \
	  ch = 0xa7;							      \
	break;								      \
      case 0x24:							      \
	if (var == CN)							      \
	  ch = 0xa5;							      \
	else if (var == HU || var == CU || var == SE || var == SE2)	      \
	  ch = 0xa4;							      \
	break;								      \
      case 0x40:							      \
	if (var == CA || var == CA2 || var == FR || var == FR1)		      \
	  ch = 0xe0;							      \
	else if (var == DE || var == ES || var == IT || var == PT)	      \
	  ch = 0xa7;							      \
	else if (var == ES2)						      \
	  ch = 0x2022;							      \
	else if (var == YU)						      \
	  ch = 0x17d;							      \
	else if (var == HU)						      \
	  ch = 0xc1;							      \
	else if (var == PT2)						      \
	  ch = 0xb4;							      \
	else if (var == SE2)						      \
	  ch = 0xc9;							      \
	break;								      \
      case 0x5b:							      \
	if (var == CA || var == CA2)					      \
	  ch = 0xe2;							      \
	else if (var == DE || var == SE || var == SE2)			      \
	  ch = 0xc4;							      \
	else if (var == DK || var == NO || var == NO2)			      \
	  ch = 0xc6;							      \
	else if (var == ES || var == ES2 || var == CU)			      \
	  ch = 0xa1;							      \
	else if (var == IT || var == FR || var == FR1)			      \
	  ch = 0xb0;							      \
	else if (var == JP_OCR_B)					      \
	  ch = 0x2329;							      \
	else if (var == YU)						      \
	  ch = 0x160;							      \
	else if (var == HU)						      \
	  ch = 0xc9;							      \
	else if (var == PT || var == PT2)				      \
	  ch = 0xc3;							      \
	break;								      \
      case 0x5c:							      \
	if (var == CA || var == CA2 || var == IT || var == FR || var == FR1)  \
	  ch = 0xe7;							      \
	else if (var == DE || var == HU || var == SE || var == SE2)	      \
	  ch = 0xd6;							      \
	else if (var == DK || var == NO || var == NO2)			      \
	  ch = 0xd8;							      \
	else if (var == ES || var == ES2 || var == CU)			      \
	  ch = 0xd1;							      \
	else if (var == JP || var == JP_OCR_B)				      \
	  ch = 0xa5;							      \
	else if (var == YU)						      \
	  ch = 0x110;							      \
	else if (var == KR)						      \
	  ch = 0x20a9;							      \
	else if (var == PT || var == PT2)				      \
	  ch = 0xc7;							      \
	break;								      \
      case 0x5d:							      \
	if (var == CA || var == CA2)					      \
	  ch = 0xea;							      \
	else if (var == DE || var == HU)				      \
	  ch = 0xdc;							      \
	else if (var == DK || var == NO || var == NO2 || var == SE	      \
		 || var == SE2)						      \
	  ch = 0xc5;							      \
	else if (var == ES)						      \
	  ch = 0xbf;							      \
	else if (var == ES2)						      \
	  ch = 0xc7;							      \
	else if (var == IT)						      \
	  ch = 0xe9;							      \
	else if (var == JP_OCR_B)					      \
	  ch = 0x232a;							      \
	else if (var == YU)						      \
	  ch = 0x106;							      \
	else if (var == FR || var == FR1)				      \
	  ch = 0xa7;							      \
	else if (var == PT || var == PT2)				      \
	  ch = 0xd5;							      \
	break;								      \
      case 0x5e:							      \
	if (var == CA)							      \
	  ch = 0xee;							      \
	else if (var == CA2)						      \
	  ch = 0xc9;							      \
	else if (var == ES2 || var == CU)				      \
	  ch = 0xbf;							      \
	else if (var == YU)						      \
	  ch = 0x10c;							      \
	else if (var == SE2)						      \
	  ch = 0xdc;							      \
	break;								      \
      case 0x60:							      \
	if (var == CA || var == CA2)					      \
	  ch = 0xf4;							      \
	else if (var == IT)						      \
	  ch = 0xf9;							      \
	else if (var == JP_OCR_B)					      \
	  /* Illegal character.  */					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	else if (var == YU)						      \
	  ch = 0x17e;							      \
	else if (var == HU)						      \
	  ch = 0xe1;							      \
	else if (var == FR)						      \
	  ch = 0xb5;							      \
	else if (var == SE2)						      \
	  ch = 0xe9;							      \
	break;								      \
      case 0x7b:							      \
	if (var == CA || var == CA2 || var == HU || var == FR || var == FR1)  \
	  ch = 0xe9;							      \
	else if (var == DE || var == SE || var == SE2)			      \
	  ch = 0xe4;							      \
	else if (var == DK || var == NO || var == NO2)			      \
	  ch = 0xe6;							      \
	else if (var == ES)						      \
	  ch = 0xb0;							      \
	else if (var == ES2 || var == CU)				      \
	  ch = 0xb4;							      \
	else if (var == IT)						      \
	  ch = 0xe0;							      \
	else if (var == YU)						      \
	  ch = 0x161;							      \
	else if (var == PT || var == PT2)				      \
	  ch = 0xe3;							      \
	break;								      \
      case 0x7c:							      \
	if (var == CA || var == CA2 || var == FR || var == FR1)		      \
	  ch = 0xf9;							      \
	else if (var == DE || var == HU || var == SE || var == SE2)	      \
	  ch = 0xf6;							      \
	else if (var == DK || var == NO || var == NO2)			      \
	  ch = 0xf8;							      \
	else if (var == ES || var == ES2 || var == CU)			      \
	  ch = 0xf1;							      \
	else if (var == IT)						      \
	  ch = 0xf2;							      \
	else if (var == YU)						      \
	  ch = 0x111;							      \
	else if (var == PT || var == PT2)				      \
	  ch = 0xe7;							      \
	break;								      \
      case 0x7d:							      \
	if (var == CA || var == CA2 || var == IT || var == FR || var == FR1)  \
	  ch = 0xe8;							      \
	else if (var == DE || var == HU)				      \
	  ch = 0xfc;							      \
	else if (var == DK || var == NO || var == NO2 || var == SE	      \
		 || var == SE2)						      \
	  ch = 0xe5;							      \
	else if (var == ES || var == ES2)				      \
	  ch = 0xe7;							      \
	else if (var == YU)						      \
	  ch = 0x107;							      \
	else if (var == CU)						      \
	  ch = 0x5b;							      \
	else if (var == PT || var == PT2)				      \
	  ch = 0xf5;							      \
	break;								      \
      case 0x7e:							      \
	if (var == GB || var == CN || var == JP || var == NO || var == SE)    \
	  ch = 0x203e;							      \
	else if (var == CA || var == CA2)				      \
	  ch = 0xfb;							      \
	else if (var == DE)						      \
	  ch = 0xdf;							      \
	else if (var == ES2 || var == CU || var == FR || var == FR1)	      \
	  ch = 0xa8;							      \
	else if (var == IT)						      \
	  ch = 0xec;							      \
	else if (var == JP_OCR_B)					      \
	  /* Illegal character.  */					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	else if (var == YU)						      \
	  ch = 0x10d;							      \
	else if (var == HU)						      \
	  ch = 0x2dd;							      \
	else if (var == NO2)						      \
	  ch = 0x7c;							      \
	else if (var == PT)						      \
	  ch = 0xb0;							      \
	else if (var == SE2)						      \
	  ch = 0xfc;							      \
	break;								      \
      default:								      \
	break;								      \
      case 0x80 ... 0xff:						      \
	/* Illegal character.  */					      \
	failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      }									      \
									      \
    /* Hopefully gcc can recognize that the following `if' is only true	      \
       when we reach the default case in the `switch' statement.  */	      \
    if (__builtin_expect (failure, __GCONV_OK) == __GCONV_ILLEGAL_INPUT)      \
      {									      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
    else								      \
      {									      \
	put32 (outptr, ch);						      \
	outptr += 4;							      \
      }									      \
    ++inptr;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, enum variant var
#include <iconv/loop.c>


/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    unsigned int ch;							      \
    int failure = __GCONV_OK;						      \
									      \
    ch = get32 (inptr);							      \
    switch (ch)								      \
      {									      \
      case 0x23:							      \
	if (var == GB || var == ES || var == IT || var == FR || var == FR1    \
	    || var == NO2)						      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x24:							      \
	if (var == CN || var == HU || var == CU || var == SE || var == SE2)   \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x40:							      \
	if (var == CA || var == CA2 || var == DE || var == ES || var == ES2   \
	    || var == IT || var == YU || var == HU || var == FR || var == FR1 \
	    || var == PT || var == PT2 || var == SE2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x5b:							      \
	if (var == CA || var == CA2 || var == DE || var == DK || var == ES    \
	    || var == ES2 || var == IT || var == JP_OCR_B || var == YU	      \
	    || var == HU || var == FR || var == FR1 || var == NO	      \
	    || var == NO2 || var == PT || var == PT2 || var == SE	      \
	    || var == SE2)						      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	else if (var == CU)						      \
	  ch = 0x7d;							      \
	break;								      \
      case 0x5c:							      \
	if (var == CA || var == CA2 || var == DE || var == DK || var == ES    \
	    || var == ES2 || var == IT || var == JP || var == JP_OCR_B	      \
	    || var == YU || var == KR || var == HU || var == CU || var == FR  \
	    || var == FR1 || var == NO || var == NO2 || var == PT	      \
	    || var == PT2 || var == SE || var == SE2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x5d:							      \
	if (var == CA || var == CA2 || var == DE || var == DK || var == ES    \
	    || var == ES2 || var == IT || var == JP_OCR_B || var == YU	      \
	    || var == HU || var == FR || var == FR1 || var == NO	      \
	    || var == NO2 || var == PT || var == PT2 || var == SE	      \
	    || var == SE2)						      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x5e:							      \
	if (var == CA || var == CA2 || var == ES2 || var == YU || var == CU   \
	    || var == SE2)						      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x60:							      \
	if (var == CA || var == CA2 || var == IT || var == JP_OCR_B	      \
	    || var == YU || var == HU || var == FR || var == SE2)	      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x7b:							      \
	if (var == CA || var == CA2 || var == DE || var == DK || var == ES    \
	    || var == ES2 || var == IT || var == YU || var == HU	      \
	    || var == CU || var == FR || var == FR1 || var == NO	      \
	    || var == NO2 || var == PT || var == PT2 || var == SE	      \
	    || var == SE2)						      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x7c:							      \
	if (var == CA || var == CA2 || var == DE || var == DK || var == ES    \
	    || var == ES2 || var == IT || var == YU || var == HU || var == CU \
	    || var == FR || var == FR1 || var == NO || var == PT	      \
	    || var == PT2 || var == SE || var == SE2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	else if (var == NO2)						      \
	  ch = 0x7e;							      \
	break;								      \
      case 0x7d:							      \
	if (var == CA || var == CA2 || var == DE || var == DK || var == ES    \
	    || var == ES2 || var == IT || var == YU || var == HU || var == CU \
	    || var == FR || var == FR1 || var == NO || var == NO2	      \
	    || var == PT || var == PT2 || var == SE || var == SE2)	      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x7e:							      \
	if (var == GB || var == CA || var == CA2 || var == DE || var == ES2   \
	    || var == CN || var == IT || var == JP || var == JP_OCR_B	      \
	    || var == YU || var == HU || var == CU || var == FR || var == FR1 \
	    || var == NO || var == NO2 || var == PT || var == SE	      \
	    || var == SE2)						      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xa1:							      \
	if (var != ES && var != ES2 && var != CU)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5b;							      \
	break;								      \
      case 0xa3:							      \
	if (var != GB && var != ES && var != IT && var != FR && var != FR1)   \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x23;							      \
	break;								      \
      case 0xa4:							      \
	if (var != HU && var != CU && var != SE && var != SE2)		      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x24;							      \
	break;								      \
      case 0xa5:							      \
	if (var == CN)							      \
	  ch = 0x24;							      \
	else if (var == JP || var == JP_OCR_B)				      \
	  ch = 0x5c;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xa7:							      \
	if (var == DE || var == ES || var == IT || var == PT)		      \
	  ch = 0x40;							      \
	else if (var == FR || var == FR1)				      \
	  ch = 0x5d;							      \
	else if (var == NO2)						      \
	  ch = 0x23;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xa8:							      \
	if (var != ES2 && var != CU && var != FR && var != FR1)		      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7e;							      \
	break;								      \
      case 0xb0:							      \
	if (var == ES)							      \
	  ch = 0x7b;							      \
	else if (var == IT || var == FR || var == FR1)			      \
	  ch = 0x5b;							      \
	else if (var == PT)						      \
	  ch = 0x7e;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xb4:							      \
	if (var == ES2 || var == CU)					      \
	  ch = 0x7b;							      \
	else if (var == PT2)						      \
	  ch = 0x40;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xb5:							      \
	if (var != FR)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x60;							      \
	break;								      \
      case 0xbf:							      \
	if (var == ES)							      \
	  ch = 0x5d;							      \
	else if (var == ES2 || var == CU)				      \
	  ch = 0x5e;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xc1:							      \
	if (var != HU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x40;							      \
	break;								      \
      case 0xc3:							      \
	if (var != PT && var != PT2)					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5b;							      \
	break;								      \
      case 0xc4:							      \
	if (var != DE && var != SE && var != SE2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5b;							      \
	break;								      \
      case 0xc5:							      \
	if (var != DK && var != NO && var != NO2 && var != SE && var != SE2)  \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5d;							      \
	break;								      \
      case 0xc6:							      \
	if (var != DK && var != NO && var != NO2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5b;							      \
	break;								      \
      case 0xc7:							      \
	if (var == ES2)							      \
	  ch = 0x5d;							      \
	else if (var == PT || var == PT2)				      \
	  ch = 0x5c;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xc9:							      \
	if (var == CA2)							      \
	  ch = 0x5e;							      \
	else if (var == HU)						      \
	  ch = 0x5b;							      \
	else if (var == SE2)						      \
	  ch = 0x40;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xd1:							      \
	if (var != ES && var != ES2 && var != CU)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5c;							      \
	break;								      \
      case 0xd5:							      \
	if (var != PT && var != PT2)					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5d;							      \
	break;								      \
      case 0xd6:							      \
	if (var != DE && var != HU && var != SE && var != SE2)		      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5c;							      \
	break;								      \
      case 0xd8:							      \
	if (var != DK && var != NO && var != NO2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5c;							      \
	break;								      \
      case 0xdc:							      \
	if (var == DE || var == HU)					      \
	  ch = 0x5d;							      \
	else if (var == SE2)						      \
	  ch = 0x5e;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xdf:							      \
	if (var != DE)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7e;							      \
	break;								      \
      case 0xe0:							      \
	if (var == CA || var == CA2 || var == FR || var == FR1)		      \
	  ch = 0x40;							      \
	else if (var == IT)						      \
	  ch = 0x7b;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xe1:							      \
	if (var != HU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x60;							      \
	break;								      \
      case 0xe2:							      \
	if (var != CA && var != CA2)					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5b;							      \
	break;								      \
      case 0xe3:							      \
	if (var != PT && var != PT2)					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7b;							      \
	break;								      \
      case 0xe4:							      \
	if (var != DE && var != SE && var != SE2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7b;							      \
	break;								      \
      case 0xe5:							      \
	if (var != DK && var != NO && var != NO2 && var != SE && var != SE2)  \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7d;							      \
	break;								      \
      case 0xe6:							      \
	if (var != DK && var != NO && var != NO2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7b;							      \
	break;								      \
      case 0xe7:							      \
	if (var == CA || var == CA2 || var == IT || var == FR || var == FR1)  \
	  ch = 0x5c;							      \
	else if (var == ES || var == ES2)				      \
	  ch = 0x7d;							      \
	else if (var == PT || var == PT2)				      \
	  ch = 0x7c;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xe8:							      \
	if (var != CA && var != CA2 && var != IT && var != FR && var != FR1)  \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7d;							      \
	break;								      \
      case 0xe9:							      \
	if (var == CA || var == CA2 || var == HU || var == FR || var == FR1)  \
	  ch = 0x7b;							      \
	else if (var == IT)						      \
	  ch = 0x5d;							      \
	else if (var == SE2)						      \
	  ch = 0x60;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xea:							      \
	if (var != CA && var != CA2)					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5d;							      \
	break;								      \
      case 0xec:							      \
	if (var != IT)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7e;							      \
	break;								      \
      case 0xee:							      \
	if (var != CA)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5e;							      \
	break;								      \
      case 0xf1:							      \
	if (var != ES && var != ES2 && var != CU)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7c;							      \
	break;								      \
      case 0xf2:							      \
	if (var != IT)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7c;							      \
	break;								      \
      case 0xf4:							      \
	if (var != CA && var != CA2)					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x60;							      \
	break;								      \
      case 0xf5:							      \
	if (var != PT && var != PT2)					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7d;							      \
	break;								      \
      case 0xf6:							      \
	if (var != DE && var != HU && var != SE && var != SE2)		      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7c;							      \
	break;								      \
      case 0xf8:							      \
	if (var != DK && var != NO && var != NO2)			      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7c;							      \
	break;								      \
      case 0xf9:							      \
	if (var == CA || var == CA2 || var == FR || var == FR1)		      \
	  ch = 0x7c;							      \
	else if (var == IT)						      \
	  ch = 0x60;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0xfb:							      \
	if (var != CA && var != CA2)					      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7e;							      \
	break;								      \
      case 0xfc:							      \
	if (var == DE || var == HU)					      \
	  ch = 0x7d;							      \
	else if (var == SE2)						      \
	  ch = 0x7e;							      \
	else								      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	break;								      \
      case 0x160:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5b;							      \
	break;								      \
      case 0x106:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5d;							      \
	break;								      \
      case 0x107:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7d;							      \
	break;								      \
      case 0x10c:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5e;							      \
	break;								      \
      case 0x10d:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7e;							      \
	break;								      \
      case 0x110:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5c;							      \
	break;								      \
      case 0x111:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7c;							      \
	break;								      \
      case 0x161:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7b;							      \
	break;								      \
      case 0x17d:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x40;							      \
	break;								      \
      case 0x17e:							      \
	if (var != YU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x60;							      \
	break;								      \
      case 0x2dd:							      \
	if (var != HU)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7e;							      \
	break;								      \
      case 0x2022:							      \
	if (var != ES2)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x40;							      \
	break;								      \
      case 0x203e:							      \
	if (var != GB && var != CN && var != JP && var != NO && var != SE)    \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x7e;							      \
	break;								      \
      case 0x20a9:							      \
	if (var != KR)							      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5c;							      \
	break;								      \
      case 0x2329:							      \
	if (var != JP_OCR_B)						      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5b;							      \
	break;								      \
      case 0x232a:							      \
	if (var != JP_OCR_B)						      \
	  failure = __GCONV_ILLEGAL_INPUT;				      \
	ch = 0x5d;							      \
	break;								      \
      default:								      \
	if (__glibc_unlikely (ch > 0x7f))				      \
	  {								      \
	    UNICODE_TAG_HANDLER (ch, 4);				      \
	    failure = __GCONV_ILLEGAL_INPUT;				      \
	  }								      \
	break;								      \
      }									      \
									      \
    if (__builtin_expect (failure, __GCONV_OK) == __GCONV_ILLEGAL_INPUT)      \
      {									      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
									      \
    *outptr++ = (unsigned char) ch;					      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, enum variant var
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
