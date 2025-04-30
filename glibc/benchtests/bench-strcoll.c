/* Measure strcoll execution time in different locales.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <fcntl.h>
#include <assert.h>
#include <stdlib.h>
#include <locale.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json-lib.h"
#include "bench-timing.h"
#include <string.h>

/* Many thanks to http://generator.lorem-ipsum.info/  */
#define INPUT_PREFIX "strcoll-inputs/"

static const char *const input_files[] = {
  "filelist#C",
  "filelist#en_US.UTF-8",
  "lorem_ipsum#vi_VN.UTF-8",
  "lorem_ipsum#ar_SA.UTF-8",
  "lorem_ipsum#en_US.UTF-8",
  "lorem_ipsum#zh_CN.UTF-8",
  "lorem_ipsum#cs_CZ.UTF-8",
  "lorem_ipsum#en_GB.UTF-8",
  "lorem_ipsum#da_DK.UTF-8",
  "lorem_ipsum#pl_PL.UTF-8",
  "lorem_ipsum#fr_FR.UTF-8",
  "lorem_ipsum#pt_PT.UTF-8",
  "lorem_ipsum#el_GR.UTF-8",
  "lorem_ipsum#ru_RU.UTF-8",
  "lorem_ipsum#he_IL.UTF-8",
  "lorem_ipsum#es_ES.UTF-8",
  "lorem_ipsum#hi_IN.UTF-8",
  "lorem_ipsum#sv_SE.UTF-8",
  "lorem_ipsum#hu_HU.UTF-8",
  "lorem_ipsum#tr_TR.UTF-8",
  "lorem_ipsum#is_IS.UTF-8",
  "lorem_ipsum#it_IT.UTF-8",
  "lorem_ipsum#sr_RS.UTF-8",
  "lorem_ipsum#ja_JP.UTF-8"
};

#define TEXTFILE_DELIMITER " \n\r\t.,?!"

static char *
read_file (const char *filename)
{
  struct stat stats;
  char *buffer = NULL;
  int fd = open (filename, O_CLOEXEC);

  if (fd >= 0)
    {
      if (fstat (fd, &stats) == 0)
	{
	  buffer = malloc (stats.st_size + 1);
	  if (buffer)
	    {
	      if (read (fd, buffer, stats.st_size) == stats.st_size)
		buffer[stats.st_size] = '\0';
	      else
		{
		  free (buffer);
		  buffer = NULL;
		}
	    }
	}
      close (fd);
    }

  return buffer;
}

static size_t
count_words (const char *text, const char *delim)
{
  size_t wordcount = 0;
  char *tmp = strdup (text);

  char *token = strtok (tmp, delim);
  while (token != NULL)
    {
      if (*token != '\0')
	wordcount++;
      token = strtok (NULL, delim);
    }

  free (tmp);
  return wordcount;
}

typedef struct
{
  size_t size;
  char **words;
} word_list;

static word_list *
new_word_list (size_t size)
{
  word_list *list = malloc (sizeof (word_list));
  assert (list != NULL);
  list->size = size;
  list->words = malloc (size * sizeof (char *));
  assert (list->words != NULL);
  return list;
}

static word_list *
str_word_list (const char *str, const char *delim)
{
  size_t n = 0;
  word_list *list = new_word_list (count_words (str, delim));

  char *toks = strdup (str);
  char *word = strtok (toks, delim);
  while (word != NULL && n < list->size)
    {
      if (*word != '\0')
	list->words[n++] = strdup (word);
      word = strtok (NULL, delim);
    }

  free (toks);
  return list;
}

static word_list *
copy_word_list (const word_list *list)
{
  size_t i;
  word_list *copy = new_word_list (list->size);

  for (i = 0; i < list->size; i++)
    copy->words[i] = strdup (list->words[i]);

  return copy;
}

static void
free_word_list (word_list *list)
{
  size_t i;
  for (i = 0; i < list->size; i++)
    free (list->words[i]);

  free (list->words);
  free (list);
}

static int
compare_words (const void *a, const void *b)
{
  const char *s1 = *(char **) a;
  const char *s2 = *(char **) b;
  return strcoll (s1, s2);
}

#undef INNER_LOOP_ITERS
#define INNER_LOOP_ITERS 16

static void
bench_list (json_ctx_t *json_ctx, word_list *list)
{
  size_t i;
  timing_t start, stop, cur;

  word_list **tests = malloc (INNER_LOOP_ITERS * sizeof (word_list *));
  assert (tests != NULL);
  for (i = 0; i < INNER_LOOP_ITERS; i++)
    tests[i] = copy_word_list (list);

  TIMING_NOW (start);
  for (i = 0; i < INNER_LOOP_ITERS; i++)
    qsort (tests[i]->words, tests[i]->size, sizeof (char *), compare_words);
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);
  setlocale (LC_ALL, "en_US.UTF-8");
  json_attr_double (json_ctx, "duration", cur);
  json_attr_double (json_ctx, "iterations", i);
  json_attr_double (json_ctx, "mean", (double) cur / i);

  for (i = 0; i < INNER_LOOP_ITERS; i++)
    free_word_list (tests[i]);
  free (tests);
}

typedef enum
{
  OK,
  ERROR_FILENAME,
  ERROR_LOCALE,
  ERROR_IO
} result_t;

static result_t
bench_file (json_ctx_t *json_ctx, const char *testname, const char *filename,
	    const char *locale)
{
  if (setlocale (LC_ALL, locale) == NULL)
    return ERROR_LOCALE;

  char *text = read_file (filename);
  if (text == NULL)
    return ERROR_IO;

  word_list *list = str_word_list (text, TEXTFILE_DELIMITER);

  json_attr_object_begin (json_ctx, testname);
  bench_list (json_ctx, list);
  json_attr_object_end (json_ctx);

  free_word_list (list);
  free (text);
  return OK;
}

int
main (void)
{
  json_ctx_t *json_ctx = malloc (sizeof (json_ctx_t));
  assert (json_ctx != NULL);
  json_init (json_ctx, 2, stdout);
  json_attr_object_begin (json_ctx, "strcoll");

  size_t i;
  result_t result = OK;
  for (i = 0; i < (sizeof (input_files) / sizeof (input_files[0])); i++)
    {
      char *locale = strchr (input_files[i], '#');
      if (locale == NULL)
	{
	  printf ("Failed to get locale from filename %s, aborting!\n",
		  input_files[i]);
	  return ERROR_FILENAME;
	}

      char *filename;
      asprintf (&filename, INPUT_PREFIX "%s", input_files[i]);
      result = bench_file (json_ctx, input_files[i], filename, locale + 1);

      if (result != OK)
	{
	  if (result == ERROR_LOCALE)
	    printf ("Failed to set locale %s, aborting!\n", locale);
	  else if (result == ERROR_IO)
	    printf ("Failed to read file %s, aborting!\n", filename);
	  free (filename);
	  goto out;
	}
      free (filename);
    }

out:
  json_attr_object_end (json_ctx);
  free (json_ctx);
  return result;
}
