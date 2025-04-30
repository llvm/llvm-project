/* IDNA functions, forwarding to implementations in libidn2.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <allocate_once.h>
#include <dlfcn.h>
#include <inet/net-internal.h>
#include <netdb.h>
#include <stdbool.h>

/* Use the soname and version to locate libidn2, to ensure a
   compatible ABI.  */
#define LIBIDN2_SONAME "libidn2.so.0"
#define LIBIDN2_VERSION "IDN2_0.0.0"

/* Return codes from libidn2.  */
enum
  {
    IDN2_OK = 0,
    IDN2_MALLOC = -100,
  };

/* Functions from libidn2.  */
struct functions
{
  void *handle;
  int (*lookup_ul) (const char *src, char **result, int flags);
  int (*to_unicode_lzlz) (const char *name, char **result, int flags);
};

static void *
functions_allocate (void *closure)
{
  struct functions *result = malloc (sizeof (*result));
  if (result == NULL)
    return NULL;

  void *handle = __libc_dlopen (LIBIDN2_SONAME);
  if (handle == NULL)
    /* Do not cache open failures.  The library may appear
       later.  */
    {
      free (result);
      return NULL;
    }

  void *ptr_lookup_ul
    = __libc_dlvsym (handle, "idn2_lookup_ul", LIBIDN2_VERSION);
  void *ptr_to_unicode_lzlz
    = __libc_dlvsym (handle, "idn2_to_unicode_lzlz", LIBIDN2_VERSION);
  if (ptr_lookup_ul == NULL || ptr_to_unicode_lzlz == NULL)
    {
      __libc_dlclose (handle);
      free (result);
      return NULL;
    }

  result->handle = handle;
  result->lookup_ul = ptr_lookup_ul;
  result->to_unicode_lzlz = ptr_to_unicode_lzlz;
#ifdef PTR_MANGLE
  PTR_MANGLE (result->lookup_ul);
  PTR_MANGLE (result->to_unicode_lzlz);
#endif

  return result;
}

static void
functions_deallocate (void *closure, void *ptr)
{
  struct functions *functions = ptr;
  __libc_dlclose (functions->handle);
  free (functions);
}

/* Ensure that *functions is initialized and return the value of the
   pointer.  If the library cannot be loaded, return NULL.  */
static inline struct functions *
get_functions (void)
{
  static void *functions;
  return allocate_once (&functions, functions_allocate, functions_deallocate,
                        NULL);
}

/* strdup with an EAI_* error code.  */
static int
gai_strdup (const char *name, char **result)
{
  char *ptr = __strdup (name);
  if (ptr == NULL)
    return EAI_MEMORY;
  *result = ptr;
  return 0;
}

int
__idna_to_dns_encoding (const char *name, char **result)
{
  switch (__idna_name_classify (name))
    {
    case idna_name_ascii:
      /* Nothing to convert.  */
      return gai_strdup (name, result);
    case idna_name_nonascii:
      /* Encoding needed.  Handled below.  */
      break;
    case idna_name_nonascii_backslash:
    case idna_name_encoding_error:
      return EAI_IDN_ENCODE;
    case idna_name_memory_error:
      return EAI_MEMORY;
    case idna_name_error:
      return EAI_SYSTEM;
    }

  struct functions *functions = get_functions ();
  if (functions == NULL)
    /* We report this as an encoding error (assuming that libidn2 is
       not installed), although the root cause may be a temporary
       error condition due to resource shortage.  */
    return EAI_IDN_ENCODE;
  char *ptr = NULL;
  __typeof__ (functions->lookup_ul) fptr = functions->lookup_ul;
#ifdef PTR_DEMANGLE
  PTR_DEMANGLE (fptr);
#endif
  int ret = fptr (name, &ptr, 0);
  if (ret == 0)
    {
      /* Assume that idn2_free is equivalent to free.  */
      *result = ptr;
      return 0;
    }
  else if (ret == IDN2_MALLOC)
    return EAI_MEMORY;
  else
    return EAI_IDN_ENCODE;
}
libc_hidden_def (__idna_to_dns_encoding)

int
__idna_from_dns_encoding (const char *name, char **result)
{
  struct functions *functions = get_functions ();
  if (functions == NULL)
    /* Simply use the encoded name, assuming that it is not punycode
       (but even a punycode name would be syntactically valid).  */
    return gai_strdup (name, result);
  char *ptr = NULL;
  __typeof__ (functions->to_unicode_lzlz) fptr = functions->to_unicode_lzlz;
#ifdef PTR_DEMANGLE
  PTR_DEMANGLE (fptr);
#endif
  int ret = fptr (name, &ptr, 0);
  if (ret == 0)
    {
      /* Assume that idn2_free is equivalent to free.  */
      *result = ptr;
      return 0;
    }
  else if (ret == IDN2_MALLOC)
    return EAI_MEMORY;
  else
    return EAI_IDN_ENCODE;
}
libc_hidden_def (__idna_from_dns_encoding)
