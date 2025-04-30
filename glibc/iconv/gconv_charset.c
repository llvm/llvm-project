/* Charset name normalization.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */


#include <stdlib.h>
#include <ctype.h>
#include <locale.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include "gconv_int.h"
#include "gconv_charset.h"


/* This function returns a pointer to the last suffix in a conversion code
   string.  Valid suffixes matched by this function are of the form: '/' or ','
   followed by arbitrary text that doesn't contain '/' or ','.  It does not
   edit the string in any way.  The caller is expected to parse the suffix and
   remove it (by e.g. truncating the string) before the next call.  */
static char *
find_suffix (char *s)
{
  /* The conversion code is in the form of a triplet, separated by '/' chars.
     The third component of the triplet contains suffixes. If we don't have two
     slashes, we don't have a suffix.  */

  int slash_count = 0;
  char *suffix_term = NULL;

  for (int i = 0; s[i] != '\0'; i++)
    switch (s[i])
      {
        case '/':
          slash_count++;
          /* Fallthrough */
        case ',':
          suffix_term = &s[i];
      }

  if (slash_count >= 2)
    return suffix_term;

  return NULL;
}


struct gconv_parsed_code
{
  char *code;
  bool translit;
  bool ignore;
};


/* This function parses an iconv_open encoding PC.CODE, strips any suffixes
   (such as TRANSLIT or IGNORE) from it and sets corresponding flags in it.  */
static void
gconv_parse_code (struct gconv_parsed_code *pc)
{
  pc->translit = false;
  pc->ignore = false;

  while (1)
    {
      /* First drop any trailing whitespaces and separators.  */
      size_t len = strlen (pc->code);
      while ((len > 0)
             && (isspace (pc->code[len - 1])
                 || pc->code[len - 1] == ','
                 || pc->code[len - 1] == '/'))
        len--;

      pc->code[len] = '\0';

      if (len == 0)
        return;

      char * suffix = find_suffix (pc->code);
      if (suffix == NULL)
        {
          /* At this point, we have processed and removed all suffixes from the
             code and what remains of the code is suffix free.  */
          return;
        }
      else
        {
          /* A suffix is processed from the end of the code array going
             backwards, one suffix at a time.  The suffix is an index into the
             code character array and points to: one past the end of the code
             and any unprocessed suffixes, and to the beginning of the suffix
             currently being processed during this iteration.  We must process
             this suffix and then drop it from the code by terminating the
             preceding text with NULL.

             We want to allow and recognize suffixes such as:

             "/TRANSLIT"         i.e. single suffix
             "//TRANSLIT"        i.e. single suffix and multiple separators
             "//TRANSLIT/IGNORE" i.e. suffixes separated by "/"
             "/TRANSLIT//IGNORE" i.e. suffixes separated by "//"
             "//IGNORE,TRANSLIT" i.e. suffixes separated by ","
             "//IGNORE,"         i.e. trailing ","
             "//TRANSLIT/"       i.e. trailing "/"
             "//TRANSLIT//"      i.e. trailing "//"
             "/"                 i.e. empty suffix.

             Unknown suffixes are silently discarded and ignored.  */

          if ((__strcasecmp_l (suffix,
                               GCONV_TRIPLE_SEPARATOR
                               GCONV_TRANSLIT_SUFFIX,
                               _nl_C_locobj_ptr) == 0)
              || (__strcasecmp_l (suffix,
                                  GCONV_SUFFIX_SEPARATOR
                                  GCONV_TRANSLIT_SUFFIX,
                                  _nl_C_locobj_ptr) == 0))
            pc->translit = true;

          if ((__strcasecmp_l (suffix,
                               GCONV_TRIPLE_SEPARATOR
                               GCONV_IGNORE_ERRORS_SUFFIX,
                               _nl_C_locobj_ptr) == 0)
              || (__strcasecmp_l (suffix,
                                  GCONV_SUFFIX_SEPARATOR
                                  GCONV_IGNORE_ERRORS_SUFFIX,
                                  _nl_C_locobj_ptr) == 0))
            pc->ignore = true;

          /* We just processed this suffix.  We can now drop it from the
             code string by truncating it at the suffix's position.  */
          suffix[0] = '\0';
        }
    }
}


/* This function accepts the charset names of the source and destination of the
   conversion and populates *conv_spec with an equivalent conversion
   specification that may later be used by __gconv_open.  The charset names
   might contain options in the form of suffixes that alter the conversion,
   e.g. "ISO-10646/UTF-8/TRANSLIT".  It processes the charset names, ignoring
   and truncating any suffix options in fromcode, and processing and truncating
   any suffix options in tocode.  Supported suffix options ("TRANSLIT" or
   "IGNORE") when found in tocode lead to the corresponding flag in *conv_spec
   to be set to true.  Unrecognized suffix options are silently discarded.  If
   the function succeeds, it returns conv_spec back to the caller.  It returns
   NULL upon failure.  conv_spec must be allocated and freed by the caller.  */
struct gconv_spec *
__gconv_create_spec (struct gconv_spec *conv_spec, const char *fromcode,
                   const char *tocode)
{
  struct gconv_parsed_code pfc, ptc;
  struct gconv_spec *ret = NULL;

  pfc.code = __strdup (fromcode);
  ptc.code = __strdup (tocode);

  if ((pfc.code == NULL)
      || (ptc.code == NULL))
    goto out;

  gconv_parse_code (&pfc);
  gconv_parse_code (&ptc);

  /* We ignore suffixes in the fromcode because that is how the current
     implementation has always handled them.  Only suffixes in the tocode are
     processed and handled.  The reality is that invalid input in the input
     character set should only be ignored if the fromcode specifies IGNORE.
     The current implementation ignores invalid intput in the input character
     set if the tocode contains IGNORE.  We preserve this behavior for
     backwards compatibility.  In the future we may split the handling of
     IGNORE to allow a finer grained specification of ignorning invalid input
     and/or ignoring invalid output.  */
  conv_spec->translit = ptc.translit;
  conv_spec->ignore = ptc.ignore;

  /* 3 extra bytes because 1 extra for '\0', and 2 extra so strip might
     be able to add one or two trailing '/' characters if necessary.  */
  conv_spec->fromcode = malloc (strlen (fromcode) + 3);
  if (conv_spec->fromcode == NULL)
    goto out;

  conv_spec->tocode = malloc (strlen (tocode) + 3);
  if (conv_spec->tocode == NULL)
    {
      free (conv_spec->fromcode);
      conv_spec->fromcode = NULL;
      goto out;
    }

  /* Strip unrecognized characters and ensure that the code has two '/'
     characters as per conversion code triplet specification.  */
  strip (conv_spec->fromcode, pfc.code);
  strip (conv_spec->tocode, ptc.code);
  ret = conv_spec;

out:
  free (pfc.code);
  free (ptc.code);

  return ret;
}
libc_hidden_def (__gconv_create_spec)


void
__gconv_destroy_spec (struct gconv_spec *conv_spec)
{
  free (conv_spec->fromcode);
  free (conv_spec->tocode);
  return;
}
libc_hidden_def (__gconv_destroy_spec)
