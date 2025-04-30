/*
 * UFC-crypt: ultra fast crypt(3) implementation
 *
 * Copyright (C) 1991-2021 Free Software Foundation, Inc.
 *
 * The GNU C Library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * The GNU C Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with the GNU C Library; if not, see
 * <https://www.gnu.org/licenses/>.
 *
 * @(#)crypt-private.h	1.4 12/20/96
 */

/* Prototypes for local functions in libcrypt.a.  */

#ifndef CRYPT_PRIVATE_H
#define CRYPT_PRIVATE_H	1

#include <features.h>
#include <stdbool.h>

#ifndef DOS
#include "ufc-crypt.h"
#else
/*
 * Thanks to greg%wind@plains.NoDak.edu (Greg W. Wettstein)
 * for DOS patches
 */
#include "pl.h"
#include "ufc.h"
#endif
#include "crypt.h"

/* crypt.c */
extern void _ufc_doit_r (ufc_long itr, struct crypt_data * __restrict __data,
			 ufc_long *res);


/* crypt_util.c */
extern void __init_des_r (struct crypt_data * __restrict __data);
extern void __init_des (void);

extern bool _ufc_setup_salt_r (const char *s,
			       struct crypt_data * __restrict __data);
extern void _ufc_mk_keytab_r (const char *key,
			      struct crypt_data * __restrict __data);
extern void _ufc_dofinalperm_r (ufc_long *res,
				struct crypt_data * __restrict __data);
extern void _ufc_output_conversion_r (ufc_long v1, ufc_long v2,
				      const char *salt,
				      struct crypt_data * __restrict __data);

extern void __setkey_r (const char *__key,
			     struct crypt_data * __restrict __data);
extern void __encrypt_r (char * __restrict __block, int __edflag,
			      struct crypt_data * __restrict __data);

/* crypt-entry.c */
extern char *__crypt_r (const char *__key, const char *__salt,
			     struct crypt_data * __restrict __data);
extern char *fcrypt (const char *key, const char *salt);

extern void __b64_from_24bit (char **cp, int *buflen,
			      unsigned int b2, unsigned int b1, unsigned int b0,
			      int n);

#endif  /* crypt-private.h */
