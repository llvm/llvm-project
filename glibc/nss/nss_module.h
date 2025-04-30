/* Global list of NSS service modules.
   Copyright (c) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _NSS_MODULE_H
#define _NSS_MODULE_H

#include <nss.h>
#include <stdbool.h>

/* See nss_database.h for a summary of how this relates.  */

/* Typed function pointers for all functions that can be defined by a
   service module.  */
struct nss_module_functions
{
#undef DEFINE_NSS_FUNCTION
#define DEFINE_NSS_FUNCTION(f) nss_##f *f;
#include "function.def"
};

/* Number of elements of the nss_module_functions_untyped array.  */
enum
  {
    nss_module_functions_count = (sizeof (struct nss_module_functions)
                                  / sizeof (void *))
  };

/* Untyped version of struct nss_module_functions, for consistent
   processing purposes.  */
typedef void *nss_module_functions_untyped[nss_module_functions_count];

/* Locate the nss_files functions, as if by dlopen/dlsym.  */
void __nss_files_functions (nss_module_functions_untyped pointers)
  attribute_hidden;

/* Initialization state of a NSS module.  */
enum nss_module_state
{
  nss_module_uninitialized,
  nss_module_loaded,
  nss_module_failed,
};

/* A NSS service module (potentially unloaded).  Client code should
   use the functions below.  */
struct nss_module
{
  /* Actual type is enum nss_module_state.  Use int due to atomic
     access.  Used in a double-checked locking idiom.  */
  int state;

  /* The function pointers in the module.  */
  union
  {
    struct nss_module_functions typed;
    nss_module_functions_untyped untyped;
  } functions;

  /* Only used for __libc_freeres unloading.  */
  void *handle;

  /* The next module in the list. */
  struct nss_module *next;

  /* The name of the module (as it appears in /etc/nsswitch.conf).  */
  char name[];
};

/* Allocates the NSS module NAME (of NAME_LENGTH bytes) and places it
   into the global list.  If it already exists in the list, return the
   pre-existing module.  This does not actually load the module.
   Returns NULL on memory allocation failure.  */
struct nss_module *__nss_module_allocate (const char *name,
                                          size_t name_length) attribute_hidden;

/* Ensures that MODULE is in a loaded or failed state.  */
bool __nss_module_load (struct nss_module *module) attribute_hidden;

/* Ensures that MODULE is loaded and returns a pointer to the function
   NAME defined in it.  Returns NULL if MODULE could not be loaded, or
   if the function NAME is not defined in the module.  */
void *__nss_module_get_function (struct nss_module *module, const char *name)
  attribute_hidden;

/* Block attempts to dlopen any module we haven't already opened.  */
void __nss_module_disable_loading (void);

/* Called from __libc_freeres.  */
void __nss_module_freeres (void) attribute_hidden;

#endif /* NSS_MODULE_H */
