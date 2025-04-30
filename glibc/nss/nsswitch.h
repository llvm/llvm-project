/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef _NSSWITCH_H
#define _NSSWITCH_H	1

/* This is an *internal* header.  */

#include <arpa/nameser.h>
#include <netinet/in.h>
#include <nss.h>
#include <resolv.h>
#include <search.h>
#include <dlfcn.h>
#include <stdbool.h>

/* Actions performed after lookup finished.  */
typedef enum
{
  NSS_ACTION_CONTINUE,
  NSS_ACTION_RETURN,
  NSS_ACTION_MERGE
} lookup_actions;

struct nss_action;

typedef struct service_library
{
  /* Name of service (`files', `dns', `nis', ...).  */
  const char *name;
  /* Pointer to the loaded shared library.  */
  void *lib_handle;
  /* And the link to the next entry.  */
  struct service_library *next;
} service_library;


/* For mapping a function name to a function pointer.  It is known in
   nsswitch.c:nss_lookup_function that a string pointer for the lookup key
   is the first member.  */
typedef struct
{
  const char *fct_name;
  void *fct_ptr;
} known_function;


/* To access the action based on the status value use this macro.  */
#define nss_next_action(ni, status) nss_action_get (ni, status)


#ifdef USE_NSCD
/* Indices into DATABASES in nsswitch.c and __NSS_DATABASE_CUSTOM.  */
enum
  {
# define DEFINE_DATABASE(arg) NSS_DBSIDX_##arg,
# include "databases.def"
# undef DEFINE_DATABASE
    NSS_DBSIDX_max
  };

/* Flags whether custom rules for database is set.  */
extern bool __nss_database_custom[NSS_DBSIDX_max] attribute_hidden;
#endif

/* Warning for NSS functions, which don't require dlopen if glibc
   was built with --enable-static-nss.  */
#ifdef DO_STATIC_NSS
# define nss_interface_function(name)
#else
# define nss_interface_function(name) static_link_warning (name)
#endif


/* Interface functions for NSS.  */

/* Put first function with name FCT_NAME for SERVICE in FCTP.  The
   position is remembered in NI.  The function returns a value < 0 if
   an error occurred or no such function exists.  */
extern int __nss_lookup (struct nss_action **ni, const char *fct_name,
			 const char *fct2_name, void **fctp);
libc_hidden_proto (__nss_lookup)

/* Determine the next step in the lookup process according to the
   result STATUS of the call to the last function returned by
   `__nss_lookup' or `__nss_next'.  NI specifies the last function
   examined.  The function return a value > 0 if the process should
   stop with the last result of the last function call to be the
   result of the entire lookup.  The returned value is 0 if there is
   another function to use and < 0 if an error occurred.

   If ALL_VALUES is nonzero, the return value will not be > 0 as long as
   there is a possibility the lookup process can ever use following
   services.  In other words, only if all four lookup results have
   the action RETURN associated the lookup process stops before the
   natural end.  */
extern int __nss_next2 (struct nss_action **ni, const char *fct_name,
			const char *fct2_name, void **fctp, int status,
			int all_values) attribute_hidden;
libc_hidden_proto (__nss_next2)
extern int __nss_next (struct nss_action **ni, const char *fct_name, void **fctp,
		       int status, int all_values);

/* Search for the service described in NI for a function named FCT_NAME
   and return a pointer to this function if successful.  */
extern void *__nss_lookup_function (struct nss_action *ni, const char *fct_name);
libc_hidden_proto (__nss_lookup_function)


/* Called by NSCD to disable recursive calls and enable special handling
   when used in nscd.  */
struct traced_file;
extern void __nss_disable_nscd (void (*) (size_t, struct traced_file *));


typedef int (*db_lookup_function) (struct nss_action **, const char *, const char *,
				   void **);
typedef enum nss_status (*setent_function) (int);
typedef enum nss_status (*endent_function) (void);
typedef enum nss_status (*getent_function) (void *, char *, size_t,
					    int *, int *);
typedef int (*getent_r_function) (void *, char *, size_t,
				  void **result, int *);

extern void __nss_setent (const char *func_name,
			  db_lookup_function lookup_fct,
			  struct nss_action **nip, struct nss_action **startp,
			  struct nss_action **last_nip, int stayon,
			  int *stayon_tmp, int res)
     attribute_hidden;
extern void __nss_endent (const char *func_name,
			  db_lookup_function lookup_fct,
			  struct nss_action **nip, struct nss_action **startp,
			  struct nss_action **last_nip, int res)
     attribute_hidden;
extern int __nss_getent_r (const char *getent_func_name,
			   const char *setent_func_name,
			   db_lookup_function lookup_fct,
			   struct nss_action **nip, struct nss_action **startp,
			   struct nss_action **last_nip, int *stayon_tmp,
			   int res,
			   void *resbuf, char *buffer, size_t buflen,
			   void **result, int *h_errnop)
     attribute_hidden;
extern void *__nss_getent (getent_r_function func,
			   void **resbuf, char **buffer, size_t buflen,
			   size_t *buffer_size, int *h_errnop)
     attribute_hidden;
struct resolv_context;
struct hostent;
extern int __nss_hostname_digits_dots_context (struct resolv_context *,
					       const char *name,
					       struct hostent *resbuf,
					       char **buffer,
					       size_t *buffer_size,
					       size_t buflen,
					       struct hostent **result,
					       enum nss_status *status, int af,
					       int *h_errnop) attribute_hidden;
extern int __nss_hostname_digits_dots (const char *name,
				       struct hostent *resbuf, char **buffer,
				       size_t *buffer_size, size_t buflen,
				       struct hostent **result,
				       enum nss_status *status, int af,
				       int *h_errnop);
libc_hidden_proto (__nss_hostname_digits_dots)

/* Maximum number of aliases we allow.  */
#define MAX_NR_ALIASES  48
#define MAX_NR_ADDRS    48

/* Prototypes for __nss_*_lookup2 functions.  */
#define DEFINE_DATABASE(arg)						      \
  extern struct nss_action *__nss_##arg##_database attribute_hidden;		      \
  int __nss_##arg##_lookup2 (struct nss_action **, const char *,		      \
			     const char *, void **);			      \
  libc_hidden_proto (__nss_##arg##_lookup2)
#include "databases.def"
#undef DEFINE_DATABASE

#include <nss/nss_module.h>
#include <nss/nss_action.h>
#include <nss/nss_database.h>

#endif	/* nsswitch.h */
