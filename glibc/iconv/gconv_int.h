/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#ifndef _GCONV_INT_H
#define _GCONV_INT_H	1

#include "gconv.h"
#include <stdlib.h>		/* For alloca used in macro below.  */
#include <ctype.h>		/* For __toupper_l used in macro below.  */
#include <string.h>		/* For strlen et al used in macro below.  */
#include <libc-lock.h>

__BEGIN_DECLS


/* Structure for alias definition.  Simply two strings.  */
struct gconv_alias
{
  char *fromname;
  char *toname;
};


/* Structure describing one loaded shared object.  This normally are
   objects to perform conversation but as a special case the db shared
   object is also handled.  */
struct __gconv_loaded_object
{
  /* Name of the object.  It must be the first structure element.  */
  const char *name;

  /* Reference counter for the db functionality.  If no conversion is
     needed we unload the db library.  */
  int counter;

  /* The handle for the shared object.  */
  void *handle;

  /* Pointer to the functions the module defines.  */
  __gconv_fct fct;
  __gconv_init_fct init_fct;
  __gconv_end_fct end_fct;
};


/* Description for an available conversion module.  */
struct gconv_module
{
  const char *from_string;
  const char *to_string;

  int cost_hi;
  int cost_lo;

  const char *module_name;

  struct gconv_module *left;	/* Prefix smaller.  */
  struct gconv_module *same;	/* List of entries with identical prefix.  */
  struct gconv_module *right;	/* Prefix larger.  */
};


/* The specification of the conversion that needs to be performed.  */
struct gconv_spec
{
  char *fromcode;
  char *tocode;
  bool translit;
  bool ignore;
};

/* Flags for `gconv_open'.  */
enum
{
  GCONV_AVOID_NOCONV = 1 << 0
};

/* When GCONV_AVOID_NOCONV is set and no conversion is needed,
   __GCONV_NULCONV should be returned.  */
enum
{
  __GCONV_NULCONV = -1
};

/* Global variables.  */

/* Database of alias names.  */
extern void *__gconv_alias_db attribute_hidden;

/* Array with available modules.  */
extern struct gconv_module *__gconv_modules_db attribute_hidden;

/* Value of the GCONV_PATH environment variable.  */
extern const char *__gconv_path_envvar attribute_hidden;

/* Lock for the conversion database content.  */
__libc_lock_define (extern, __gconv_lock attribute_hidden)


/* The gconv functions expects the name to be in upper case and complete,
   including the trailing slashes if necessary.  */
#define norm_add_slashes(str,suffix) \
  ({									      \
    const char *cp = (str);						      \
    char *result;							      \
    char *tmp;								      \
    size_t cnt = 0;							      \
    const size_t suffix_len = strlen (suffix);				      \
									      \
    while (*cp != '\0')							      \
      if (*cp++ == '/')							      \
	++cnt;								      \
									      \
    tmp = result = __alloca (cp - (str) + 3 + suffix_len);		      \
    cp = (str);								      \
    while (*cp != '\0')							      \
      *tmp++ = __toupper_l (*cp++, _nl_C_locobj_ptr);			      \
    if (cnt < 2)							      \
      {									      \
	*tmp++ = '/';							      \
	if (cnt < 1)							      \
	  {								      \
	    *tmp++ = '/';						      \
	    if (suffix_len != 0)					      \
	      tmp = __mempcpy (tmp, suffix, suffix_len);		      \
	  }								      \
      }									      \
    *tmp = '\0';							      \
    result;								      \
  })


/* Return in *HANDLE, a decriptor for the transformation.  The function expects
   the specification of the transformation in the structure pointed to by
   CONV_SPEC.  It only reads *CONV_SPEC and does not take ownership of it.  */
extern int __gconv_open (struct gconv_spec *conv_spec,
                         __gconv_t *handle, int flags);
libc_hidden_proto (__gconv_open)

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
   NULL upon failure.  */
extern struct gconv_spec *
__gconv_create_spec (struct gconv_spec *conv_spec, const char *fromcode,
                     const char *tocode);
libc_hidden_proto (__gconv_create_spec)

/* This function frees all heap memory allocated by __gconv_create_spec.  */
extern void
__gconv_destroy_spec (struct gconv_spec *conv_spec);
libc_hidden_proto (__gconv_destroy_spec)

/* Free resources associated with transformation descriptor CD.  */
extern int __gconv_close (__gconv_t cd)
     attribute_hidden;

/* Transform at most *INBYTESLEFT bytes from buffer starting at *INBUF
   according to rules described by CD and place up to *OUTBYTESLEFT
   bytes in buffer starting at *OUTBUF.  Return number of non-identical
   conversions in *IRREVERSIBLE if this pointer is not null.  */
extern int __gconv (__gconv_t cd, const unsigned char **inbuf,
		    const unsigned char *inbufend, unsigned char **outbuf,
		    unsigned char *outbufend, size_t *irreversible)
     attribute_hidden;

/* Return in *HANDLE a pointer to an array with *NSTEPS elements describing
   the single steps necessary for transformation from FROMSET to TOSET.  */
extern int __gconv_find_transform (const char *toset, const char *fromset,
				   struct __gconv_step **handle,
				   size_t *nsteps, int flags)
     attribute_hidden;

/* Search for transformation in cache data.  */
extern int __gconv_lookup_cache (const char *toset, const char *fromset,
				 struct __gconv_step **handle, size_t *nsteps,
				 int flags)
     attribute_hidden;

/* Compare the two name for whether they are after alias expansion the
   same.  This function uses the cache and fails if none is
   loaded.  */
extern int __gconv_compare_alias_cache (const char *name1, const char *name2,
					int *result)
     attribute_hidden;

/* Free data associated with a step's structure.  */
extern void __gconv_release_step (struct __gconv_step *step)
     attribute_hidden;

/* Read all the configuration data and cache it if not done so already.  */
extern void __gconv_load_conf (void) attribute_hidden;

/* Try to read module cache file.  */
extern int __gconv_load_cache (void) attribute_hidden;

/* Retrieve pointer to internal cache.  */
extern void *__gconv_get_cache (void);

/* Retrieve pointer to internal module database.  */
extern struct gconv_module *__gconv_get_modules_db (void);

/* Retrieve pointer to internal alias database.  */
extern void *__gconv_get_alias_db (void);

/* Comparison function to search alias.  */
extern int __gconv_alias_compare (const void *p1, const void *p2)
     attribute_hidden;

/* Clear reference to transformation step implementations which might
   cause the code to be unloaded.  */
extern int __gconv_close_transform (struct __gconv_step *steps,
				    size_t nsteps)
     attribute_hidden;

/* Free all resources allocated for the transformation record when
   using the cache.  */
extern void __gconv_release_cache (struct __gconv_step *steps, size_t nsteps)
     attribute_hidden;

/* Load shared object named by NAME.  If already loaded increment reference
   count.  */
extern struct __gconv_loaded_object *__gconv_find_shlib (const char *name)
     attribute_hidden;

/* Release shared object.  If no further reference is available unload
   the object.  */
extern void __gconv_release_shlib (struct __gconv_loaded_object *handle)
     attribute_hidden;

/* Fill STEP with information about builtin module with NAME.  */
extern void __gconv_get_builtin_trans (const char *name,
				       struct __gconv_step *step)
     attribute_hidden;

/* Transliteration using the locale's data.  */
extern int __gconv_transliterate (struct __gconv_step *step,
                                  struct __gconv_step_data *step_data,
                                  const unsigned char *inbufstart,
                                  const unsigned char **inbufp,
                                  const unsigned char *inbufend,
                                  unsigned char **outbufstart,
                                  size_t *irreversible);
libc_hidden_proto (__gconv_transliterate)

/* If NAME is an codeset alias expand it.  */
extern int __gconv_compare_alias (const char *name1, const char *name2)
     attribute_hidden;


/* Builtin transformations.  */
#ifdef _LIBC
# define __BUILTIN_TRANSFORM(Name) \
  extern int Name (struct __gconv_step *step,				      \
		   struct __gconv_step_data *data,			      \
		   const unsigned char **inbuf,				      \
		   const unsigned char *inbufend,			      \
		   unsigned char **outbufstart, size_t *irreversible,	      \
		   int do_flush, int consume_incomplete)

__BUILTIN_TRANSFORM (__gconv_transform_ascii_internal);
__BUILTIN_TRANSFORM (__gconv_transform_internal_ascii);
__BUILTIN_TRANSFORM (__gconv_transform_utf8_internal);
__BUILTIN_TRANSFORM (__gconv_transform_internal_utf8);
__BUILTIN_TRANSFORM (__gconv_transform_ucs2_internal);
__BUILTIN_TRANSFORM (__gconv_transform_internal_ucs2);
__BUILTIN_TRANSFORM (__gconv_transform_ucs2reverse_internal);
__BUILTIN_TRANSFORM (__gconv_transform_internal_ucs2reverse);
__BUILTIN_TRANSFORM (__gconv_transform_internal_ucs4);
__BUILTIN_TRANSFORM (__gconv_transform_ucs4_internal);
__BUILTIN_TRANSFORM (__gconv_transform_internal_ucs4le);
__BUILTIN_TRANSFORM (__gconv_transform_ucs4le_internal);
__BUILTIN_TRANSFORM (__gconv_transform_internal_utf16);
__BUILTIN_TRANSFORM (__gconv_transform_utf16_internal);
# undef __BUITLIN_TRANSFORM

/* Specialized conversion function for a single byte to INTERNAL, recognizing
   only ASCII characters.  */
extern wint_t __gconv_btwoc_ascii (struct __gconv_step *step, unsigned char c);

#endif

__END_DECLS

#endif /* gconv_int.h */
