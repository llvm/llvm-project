/* Simple library for printing JSON data.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef __JSON_LIB_H__
#define __JSON_LIB_H__

#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>

struct json_ctx
{
  FILE *fp;
  unsigned int indent_level;
  bool first_element;
};

typedef struct json_ctx json_ctx_t;

void json_init (json_ctx_t *ctx, unsigned int indent_level, FILE *fp);
void json_document_begin (json_ctx_t *ctx);
void json_document_end (json_ctx_t *ctx);
void json_attr_object_begin (json_ctx_t *ctx, const char *name);
void json_attr_object_end (json_ctx_t *ctx);
void json_attr_string (json_ctx_t *ctx, const char *name, const char *s);
void json_attr_int (json_ctx_t *ctx, const char *name, int64_t d);
void json_attr_uint (json_ctx_t *ctx, const char *name, uint64_t d);
void json_attr_double (json_ctx_t *ctx, const char *name, double d);
void json_array_begin (json_ctx_t *ctx, const char *name);
void json_array_end (json_ctx_t *ctx);
void json_element_string (json_ctx_t *ctx, const char *s);
void json_element_int (json_ctx_t *ctx, int64_t d);
void json_element_uint (json_ctx_t *ctx, uint64_t d);
void json_element_double (json_ctx_t *ctx, double d);
void json_element_object_begin (json_ctx_t *ctx);
void json_element_object_end (json_ctx_t *ctx);

#endif
