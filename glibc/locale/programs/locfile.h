/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1996.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifndef _LOCFILE_H
#define _LOCFILE_H	1

#include <byteswap.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/uio.h>

#include "obstack.h"
#include "linereader.h"
#include "localedef.h"

/* Structure for storing the contents of a category file.  */
struct locale_file
{
  size_t n_elements, next_element;
  uint32_t *offsets;
  struct obstack data;
  int structure_stage;
};


/* Macros used in the parser.  */
#define SYNTAX_ERROR(string, args...) \
  do									      \
    {									      \
      lr_error (ldfile, string, ## args);				      \
      lr_ignore_rest (ldfile, 0);					      \
    }									      \
  while (0)


/* General handling of `copy'.  */
extern void handle_copy (struct linereader *ldfile,
			 const struct charmap_t *charmap,
			 const char *repertoire_name,
			 struct localedef_t *result, enum token_t token,
			 int locale, const char *locale_name,
			 int ignore_content);

/* Found in locfile.c.  */
extern int locfile_read (struct localedef_t *result,
			 const struct charmap_t *charmap);

/* Check validity of all the locale data.  */
extern void check_all_categories (struct localedef_t *definitions,
				  const struct charmap_t *charmap);

/* Write out all locale categories.  */
extern void write_all_categories (struct localedef_t *definitions,
				  const struct charmap_t *charmap,
				  const char *locname,
				  const char *output_path);

extern bool swap_endianness_p;

/* Change the output to be big-endian if BIG_ENDIAN is true and
   little-endian otherwise.  */
static inline void
set_big_endian (bool big_endian)
{
  swap_endianness_p = (big_endian != (__BYTE_ORDER == __BIG_ENDIAN));
}

/* Munge VALUE so that, when stored, it has the correct byte order
   for the output files.  */
static uint32_t
__attribute__ ((unused))
maybe_swap_uint32 (uint32_t value)
{
  return swap_endianness_p ? bswap_32 (value) : value;
}

/* Likewise, but munge an array of N uint32_ts starting at ARRAY.  */
static inline void
maybe_swap_uint32_array (uint32_t *array, size_t n)
{
  if (swap_endianness_p)
    while (n-- > 0)
      array[n] = bswap_32 (array[n]);
}

/* Like maybe_swap_uint32_array, but the array of N elements is at
   the end of OBSTACK's current object.  */
static inline void
maybe_swap_uint32_obstack (struct obstack *obstack, size_t n)
{
  maybe_swap_uint32_array ((uint32_t *) obstack_next_free (obstack) - n, n);
}

/* Write out the data.  */
extern void init_locale_data (struct locale_file *file, size_t n_elements);
extern void align_locale_data (struct locale_file *file, size_t boundary);
extern void add_locale_empty (struct locale_file *file);
extern void add_locale_raw_data (struct locale_file *file, const void *data,
				 size_t size);
extern void add_locale_raw_obstack (struct locale_file *file,
				    struct obstack *obstack);
extern void add_locale_string (struct locale_file *file, const char *string);
extern void add_locale_wstring (struct locale_file *file,
				const uint32_t *string);
extern void add_locale_uint32 (struct locale_file *file, uint32_t value);
extern void add_locale_uint32_array (struct locale_file *file,
				     const uint32_t *data, size_t n_elems);
extern void add_locale_char (struct locale_file *file, char value);
extern void start_locale_structure (struct locale_file *file);
extern void end_locale_structure (struct locale_file *file);
extern void start_locale_prelude (struct locale_file *file);
extern void end_locale_prelude (struct locale_file *file);
extern void write_locale_data (const char *output_path, int catidx,
			       const char *category, struct locale_file *file);


/* Entrypoints for the parsers of the individual categories.  */

/* Handle LC_CTYPE category.  */
extern void ctype_read (struct linereader *ldfile,
			struct localedef_t *result,
			const struct charmap_t *charmap,
			const char *repertoire_name,
			int ignore_content);
extern void ctype_finish (struct localedef_t *locale,
			  const struct charmap_t *charmap);
extern void ctype_output (struct localedef_t *locale,
			  const struct charmap_t *charmap,
			  const char *output_path);
extern uint32_t *find_translit (struct localedef_t *locale,
				const struct charmap_t *charmap, uint32_t wch);

/* Handle LC_COLLATE category.  */
extern void collate_read (struct linereader *ldfile,
			  struct localedef_t *result,
			  const struct charmap_t *charmap,
			  const char *repertoire_name,
			  int ignore_content);
extern void collate_finish (struct localedef_t *locale,
			    const struct charmap_t *charmap);
extern void collate_output (struct localedef_t *locale,
			    const struct charmap_t *charmap,
			    const char *output_path);

/* Handle LC_MONETARY category.  */
extern void monetary_read (struct linereader *ldfile,
			   struct localedef_t *result,
			   const struct charmap_t *charmap,
			   const char *repertoire_name,
			   int ignore_content);
extern void monetary_finish (struct localedef_t *locale,
			     const struct charmap_t *charmap);
extern void monetary_output (struct localedef_t *locale,
			     const struct charmap_t *charmap,
			     const char *output_path);

/* Handle LC_NUMERIC category.  */
extern void numeric_read (struct linereader *ldfile,
			  struct localedef_t *result,
			  const struct charmap_t *charmap,
			  const char *repertoire_name,
			  int ignore_content);
extern void numeric_finish (struct localedef_t *locale,
			    const struct charmap_t *charmap);
extern void numeric_output (struct localedef_t *locale,
			    const struct charmap_t *charmap,
			    const char *output_path);

/* Handle LC_MESSAGES category.  */
extern void messages_read (struct linereader *ldfile,
			   struct localedef_t *result,
			   const struct charmap_t *charmap,
			   const char *repertoire_name,
			   int ignore_content);
extern void messages_finish (struct localedef_t *locale,
			     const struct charmap_t *charmap);
extern void messages_output (struct localedef_t *locale,
			     const struct charmap_t *charmap,
			     const char *output_path);

/* Handle LC_TIME category.  */
extern void time_read (struct linereader *ldfile,
		       struct localedef_t *result,
		       const struct charmap_t *charmap,
		       const char *repertoire_name,
		       int ignore_content);
extern void time_finish (struct localedef_t *locale,
			 const struct charmap_t *charmap);
extern void time_output (struct localedef_t *locale,
			 const struct charmap_t *charmap,
			 const char *output_path);

/* Handle LC_PAPER category.  */
extern void paper_read (struct linereader *ldfile,
			struct localedef_t *result,
			const struct charmap_t *charmap,
			const char *repertoire_name,
			int ignore_content);
extern void paper_finish (struct localedef_t *locale,
			  const struct charmap_t *charmap);
extern void paper_output (struct localedef_t *locale,
			  const struct charmap_t *charmap,
			  const char *output_path);

/* Handle LC_NAME category.  */
extern void name_read (struct linereader *ldfile,
		       struct localedef_t *result,
		       const struct charmap_t *charmap,
		       const char *repertoire_name,
		       int ignore_content);
extern void name_finish (struct localedef_t *locale,
			 const struct charmap_t *charmap);
extern void name_output (struct localedef_t *locale,
			 const struct charmap_t *charmap,
			 const char *output_path);

/* Handle LC_ADDRESS category.  */
extern void address_read (struct linereader *ldfile,
			  struct localedef_t *result,
			  const struct charmap_t *charmap,
			  const char *repertoire_name,
			  int ignore_content);
extern void address_finish (struct localedef_t *locale,
			    const struct charmap_t *charmap);
extern void address_output (struct localedef_t *locale,
			    const struct charmap_t *charmap,
			    const char *output_path);

/* Handle LC_TELEPHONE category.  */
extern void telephone_read (struct linereader *ldfile,
			    struct localedef_t *result,
			    const struct charmap_t *charmap,
			    const char *repertoire_name,
			    int ignore_content);
extern void telephone_finish (struct localedef_t *locale,
			      const struct charmap_t *charmap);
extern void telephone_output (struct localedef_t *locale,
			      const struct charmap_t *charmap,
			      const char *output_path);

/* Handle LC_MEASUREMENT category.  */
extern void measurement_read (struct linereader *ldfile,
			      struct localedef_t *result,
			      const struct charmap_t *charmap,
			      const char *repertoire_name,
			      int ignore_content);
extern void measurement_finish (struct localedef_t *locale,
				const struct charmap_t *charmap);
extern void measurement_output (struct localedef_t *locale,
				const struct charmap_t *charmap,
				const char *output_path);

/* Handle LC_IDENTIFICATION category.  */
extern void identification_read (struct linereader *ldfile,
				 struct localedef_t *result,
				 const struct charmap_t *charmap,
				 const char *repertoire_name,
				 int ignore_content);
extern void identification_finish (struct localedef_t *locale,
				   const struct charmap_t *charmap);
extern void identification_output (struct localedef_t *locale,
				   const struct charmap_t *charmap,
				   const char *output_path);

#endif /* locfile.h */
