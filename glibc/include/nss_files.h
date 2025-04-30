/* Internal routines for nss_files.
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
   <https://www.gnu.org/licenses/>.  */

#ifndef _NSS_FILES_H
#define _NSS_FILES_H

#include <nss.h>
#include <stdio.h>
#if IS_IN (libc)
#include <libc-lock.h>
#endif

/* Open PATH for reading, as a data source for nss_files.  */
FILE *__nss_files_fopen (const char *path);
libc_hidden_proto (__nss_files_fopen)

/* Read a line from FP, storing it BUF.  Strip leading blanks and skip
   comments.  Sets errno and returns error code on failure.  Special
   failure: ERANGE means the buffer is too small.  The function writes
   the original offset to *POFFSET (which can be negative in the case
   of non-seekable input).  */
int __nss_readline (FILE *fp, char *buf, size_t len, off64_t *poffset);
libc_hidden_proto (__nss_readline)

/* Seek FP to OFFSET.  Sets errno and returns error code on failure.
   On success, sets errno to ERANGE and returns ERANGE (to indicate
   re-reading of the same input line to the caller).  If OFFSET is
   negative, fail with ESPIPE without seeking.  Intended to be used
   after parsing data read by __nss_readline failed with ERANGE.  */
int __nss_readline_seek (FILE *fp, off64_t offset) attribute_hidden;

/* Handles the result of a parse_line call (as defined by
   nss/nss_files/files-parse.c).  Adjusts the file offset of FP as
   necessary.  Returns 0 on success, and updates errno on failure (and
   returns that error code).  */
int __nss_parse_line_result (FILE *fp, off64_t offset, int parse_line_result);
libc_hidden_proto (__nss_parse_line_result)

/* Per-file data.  Used by the *ent functions that need to preserve
   state across calls.  */
struct nss_files_per_file_data
{
  FILE *stream;
#if IS_IN (libc)
  /* The size of locks changes between libc and nss_files, so this
     member must be last and is only available in libc.  */
  __libc_lock_define (, lock);
#endif
};

/* File index for __nss_files_data_get.  */
enum nss_files_file
  {
    nss_file_aliasent,
    nss_file_etherent,
    nss_file_grent,
    nss_file_hostent,
    nss_file_netent,
    nss_file_protoent,
    nss_file_pwent,
    nss_file_rpcent,
    nss_file_servent,
    nss_file_sgent,
    nss_file_spent,

    nss_file_count
  };

/* Obtains a pointer to the per-file data for FILE, which is written
   to *PDATA, and tries to open the file at PATH for it.  On success,
   returns NSS_STATUS_SUCCESS, and the caller must later call
   __nss_files_data_put.  On failure, NSS_STATUS_TRYAGAIN is returned,
   and *ERRNOP and *HERRNOP are updated if these pointers are not
   null.  */
enum nss_status __nss_files_data_open (struct nss_files_per_file_data **pdata,
                                       enum nss_files_file file,
                                       const char *path,
                                       int *errnop, int *herrnop);
libc_hidden_proto (__nss_files_data_open)

/* Unlock the per-file data, previously obtained by
   __nss_files_data_open.  */
void __nss_files_data_put (struct nss_files_per_file_data *data);
libc_hidden_proto (__nss_files_data_put)

/* Performs the set*ent operation for FILE.  PATH is the file to
   open.  */
enum nss_status __nss_files_data_setent (enum nss_files_file file,
                                           const char *path);
libc_hidden_proto (__nss_files_data_setent)

/* Performs the end*ent operation for FILE.  */
enum nss_status __nss_files_data_endent (enum nss_files_file file);
libc_hidden_proto (__nss_files_data_endent)

struct parser_data;

/* Instances of the parse_line function from
   nss/nss_files/files-parse.c.  */
typedef int nss_files_parse_line (char *line, void *result,
                                  struct parser_data *data,
                                  size_t datalen, int *errnop);
extern nss_files_parse_line _nss_files_parse_etherent;
extern nss_files_parse_line _nss_files_parse_grent;
extern nss_files_parse_line _nss_files_parse_netent;
extern nss_files_parse_line _nss_files_parse_protoent;
extern nss_files_parse_line _nss_files_parse_pwent;
extern nss_files_parse_line _nss_files_parse_rpcent;
extern nss_files_parse_line _nss_files_parse_servent;
extern nss_files_parse_line _nss_files_parse_sgent;
extern nss_files_parse_line _nss_files_parse_spent;

libc_hidden_proto (_nss_files_parse_etherent)
libc_hidden_proto (_nss_files_parse_grent)
libc_hidden_proto (_nss_files_parse_netent)
libc_hidden_proto (_nss_files_parse_protoent)
libc_hidden_proto (_nss_files_parse_pwent)
libc_hidden_proto (_nss_files_parse_rpcent)
libc_hidden_proto (_nss_files_parse_servent)
libc_hidden_proto (_nss_files_parse_sgent)
libc_hidden_proto (_nss_files_parse_spent)

NSS_DECLARE_MODULE_FUNCTIONS (files)
#undef DEFINE_NSS_FUNCTION
#define DEFINE_NSS_FUNCTION(x) libc_hidden_proto (_nss_files_##x)
#include <nss/function.def>
#undef DEFINE_NSS_FUNCTION

void _nss_files_init (void (*cb) (size_t, struct traced_file *));
libc_hidden_proto (_nss_files_init)

/* Generic implementation of fget*ent_r.  Reads lines from FP until
   EOF or a successful parse into *RESULT using PARSER.  Returns 0 on
   success, ENOENT on EOF, ERANGE on too-small buffer.  */
int __nss_fgetent_r (FILE *fp, void *result,
                     char *buffer, size_t buffer_length,
                     nss_files_parse_line parser) attribute_hidden;

#endif /* _NSS_FILES_H */
