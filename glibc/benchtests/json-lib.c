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

#include <string.h>

#include "json-lib.h"

void
json_init (json_ctx_t *ctx, unsigned int indent_level, FILE *fp)
{
  ctx->indent_level = indent_level;
  ctx->fp = fp;
  ctx->first_element = true;
}

static void
do_indent (json_ctx_t *ctx)
{
  char indent_buf[ctx->indent_level + 1];

  memset (indent_buf, ' ', ctx->indent_level + 1);
  indent_buf[ctx->indent_level] = '\0';

  fputs (indent_buf, ctx->fp);
}

void
json_document_begin (json_ctx_t *ctx)
{
  do_indent (ctx);

  fputs ("{\n", ctx->fp);

  ctx->indent_level++;
  ctx->first_element = true;
}

void
json_document_end (json_ctx_t *ctx)
{
  ctx->indent_level--;

  do_indent (ctx);

  fputs ("\n}", ctx->fp);
}

void
json_attr_object_begin (json_ctx_t *ctx, const char *name)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ",\n");

  do_indent (ctx);

  fprintf (ctx->fp, "\"%s\": {\n", name);

  ctx->indent_level++;
  ctx->first_element = true;
}

void
json_attr_object_end (json_ctx_t *ctx)
{
  ctx->indent_level--;
  ctx->first_element = false;

  fputs ("\n", ctx->fp);

  do_indent (ctx);

  fputs ("}", ctx->fp);
}

void
json_attr_string (json_ctx_t *ctx, const char *name, const char *s)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ",\n");
  else
    ctx->first_element = false;

  do_indent (ctx);

  fprintf (ctx->fp, "\"%s\": \"%s\"", name, s);
}

void
json_attr_uint (json_ctx_t *ctx, const char *name, uint64_t d)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ",\n");
  else
    ctx->first_element = false;

  do_indent (ctx);

  fprintf (ctx->fp, "\"%s\": %" PRIu64 , name, d);
}

void
json_attr_int (json_ctx_t *ctx, const char *name, int64_t d)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ",\n");
  else
    ctx->first_element = false;

  do_indent (ctx);

  fprintf (ctx->fp, "\"%s\": %" PRId64 , name, d);
}

void
json_attr_double (json_ctx_t *ctx, const char *name, double d)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ",\n");
  else
    ctx->first_element = false;

  do_indent (ctx);

  fprintf (ctx->fp, "\"%s\": %g", name, d);
}

void
json_array_begin (json_ctx_t *ctx, const char *name)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ",\n");

  do_indent (ctx);

  fprintf (ctx->fp, "\"%s\": [", name);

  ctx->indent_level++;
  ctx->first_element = true;
}

void
json_array_end (json_ctx_t *ctx)
{
  ctx->indent_level--;
  ctx->first_element = false;

  fputs ("]", ctx->fp);
}

void
json_element_string (json_ctx_t *ctx, const char *s)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ", \"%s\"", s);
  else
    {
      fprintf (ctx->fp, "\"%s\"", s);
      ctx->first_element = false;
    }
}

void
json_element_uint (json_ctx_t *ctx, uint64_t d)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ", %" PRIu64, d);
  else
    {
      fprintf (ctx->fp, "%" PRIu64, d);
      ctx->first_element = false;
    }
}

void
json_element_int (json_ctx_t *ctx, int64_t d)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ", %" PRId64, d);
  else
    {
      fprintf (ctx->fp, "%" PRId64, d);
      ctx->first_element = false;
    }
}

void
json_element_double (json_ctx_t *ctx, double d)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ", %g", d);
  else
    {
      fprintf (ctx->fp, "%g", d);
      ctx->first_element = false;
    }
}

void
json_element_object_begin (json_ctx_t *ctx)
{
  if (!ctx->first_element)
    fprintf (ctx->fp, ",");

  fputs ("\n", ctx->fp);

  do_indent (ctx);

  fputs ("{\n", ctx->fp);

  ctx->indent_level++;
  ctx->first_element = true;
}

void
json_element_object_end (json_ctx_t *ctx)
{
  ctx->indent_level--;
  ctx->first_element = false;

  fputs ("\n", ctx->fp);

  do_indent (ctx);

  fputs ("}", ctx->fp);
}
