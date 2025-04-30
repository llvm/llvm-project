/* Test ns_name-related functions.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* This test program processes the tst-ns_name.data file.  */

#include <ctype.h>
#include <resolv.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xstdio.h>

/* A byte buffer and its length.  */
struct buffer
{
  unsigned char *data;
  size_t length;
};

/* Convert a base64-encoded string to its binary representation.  */
static bool
base64_to_buffer (const char *base64, struct buffer *result)
{
  /* "-" denotes an empty input.  */
  if (strcmp (base64, "-") == 0)
    {
      result->data = xmalloc (1);
      result->length = 0;
      return true;
    }

  size_t size = strlen (base64);
  unsigned char *data = xmalloc (size);
  int ret = b64_pton (base64, data, size);
  if (ret < 0 || ret > size)
    return false;
  result->data = xrealloc (data, ret);
  result->length = ret;
  return true;
}

/* A test case for ns_name_unpack and ns_name_ntop.  */
struct test_case
{
  char *path;
  size_t lineno;
  struct buffer input;
  size_t input_offset;
  int unpack_result;
  struct buffer unpack_output;
  int ntop_result;
  char *ntop_text;
};

/* Deallocate the buffers associated with the test case.  */
static void
free_test_case (struct test_case *t)
{
  free (t->path);
  free (t->input.data);
  free (t->unpack_output.data);
  free (t->ntop_text);
}

/* Extract the test case information from a test file line.  */
static bool
parse_test_case (const char *path, size_t lineno, const char *line,
                 struct test_case *result)
{
  memset (result, 0, sizeof (*result));
  result->path = xstrdup (path);
  result->lineno = lineno;
  result->ntop_result = -1;
  char *input = NULL;
  char *unpack_output = NULL;
  int ret = sscanf (line, "%ms %zu %d %ms %d %ms",
                    &input, &result->input_offset,
                    &result->unpack_result, &unpack_output,
                    &result->ntop_result, &result->ntop_text);
  if (ret < 3)
    {
      printf ("%s:%zu: error: missing input fields\n", path, lineno);
      free (input);
      return false;
    }
  if (!base64_to_buffer (input, &result->input))
    {
      printf ("%s:%zu: error: malformed base64 input data\n", path, lineno);
      free (input);
      free (unpack_output);
      free (result->ntop_text);
      return false;
    }
  free (input);

  if (unpack_output == NULL)
    result->unpack_output = (struct buffer) { NULL, 0 };
  else if (!base64_to_buffer (unpack_output, &result->unpack_output))
    {
      printf ("%s:%zu: error: malformed base64 unpack data\n", path, lineno);
      free (result->input.data);
      free (unpack_output);
      free (result->ntop_text);
      return false;
    }
  free (unpack_output);

  /* At this point, all allocated buffers have been transferred to
     *result.  */

  if (result->input_offset > result->input.length)
    {
      printf ("%s:%zu: error: input offset %zu exceeds buffer size %zu\n",
              path, lineno, result->input_offset, result->input.length);
      free_test_case (result);
      return false;
    }
  if (result->unpack_result < -1)
    {
      printf ("%s:%zu: error: invalid unpack result %d\n",
              path, lineno, result->unpack_result);
      free_test_case (result);
      return false;
    }
  if (result->ntop_result < -1)
    {
      printf ("%s:%zu: error: invalid ntop result %d\n",
              path, lineno, result->ntop_result);
      free_test_case (result);
      return false;
    }

  bool fields_consistent;
  switch (ret)
    {
    case 3:
      fields_consistent = result->unpack_result == -1;
      break;
    case 5:
      fields_consistent = result->unpack_result != -1
        && result->ntop_result == -1;
      break;
    case 6:
      fields_consistent = result->unpack_result != -1
        && result->ntop_result != -1;
      break;
    default:
      fields_consistent = false;
    }
  if (!fields_consistent)
    {
      printf ("%s:%zu: error: wrong number of fields: %d\n",
              path, lineno, ret);
      free_test_case (result);
      return false;
    }
  return true;
}

/* Format the buffer as a hexadecimal string and write it to standard
   output.  */
static void
print_hex (const char *label, struct buffer buffer)
{
  printf ("  %s ", label);
  unsigned char *p = buffer.data;
  unsigned char *end = p + buffer.length;
  while (p < end)
    {
      printf ("%02X", *p & 0xFF);
      ++p;
    }
  putchar ('\n');
}

/* Run the test case specified in *T.  */
static void
run_test_case (struct test_case *t)
{
  /* Test ns_name_unpack.  */
  unsigned char *unpacked = xmalloc (NS_MAXCDNAME);
  int consumed = ns_name_unpack
    (t->input.data, t->input.data + t->input.length,
     t->input.data + t->input_offset,
     unpacked, NS_MAXCDNAME);
  if (consumed != t->unpack_result)
    {
      support_record_failure ();
      printf ("%s:%zu: error: wrong result from ns_name_unpack\n"
              "  expected: %d\n"
              "  actual:   %d\n",
              t->path, t->lineno, t->unpack_result, consumed);
      return;
    }
  if (consumed != -1)
    {
      if (memcmp (unpacked, t->unpack_output.data,
                  t->unpack_output.length) != 0)
        {
          support_record_failure ();
          printf ("%s:%zu: error: wrong data from ns_name_unpack\n",
                  t->path, t->lineno);
          print_hex ("expected:", t->unpack_output);
          print_hex ("actual:  ",
                     (struct buffer) { unpacked, t->unpack_output.length });
          return;
        }

      /* Test ns_name_ntop.  */
      char *text = xmalloc (NS_MAXDNAME);
      int ret = ns_name_ntop (unpacked, text, NS_MAXDNAME);
      if (ret != t->ntop_result)
        {
          support_record_failure ();
          printf ("%s:%zu: error: wrong result from ns_name_top\n"
                  "  expected: %d\n"
                  "  actual:   %d\n",
                  t->path, t->lineno, t->ntop_result, ret);
          return;
        }
      if (ret != -1)
        {
          if (strcmp (text, t->ntop_text) != 0)
            {
              support_record_failure ();
              printf ("%s:%zu: error: wrong data from ns_name_ntop\n",
                      t->path, t->lineno);
              printf ("  expected: \"%s\"\n", t->ntop_text);
              printf ("  actual:   \"%s\"\n", text);
              return;
            }

          /* Test ns_name_pton.  Unpacking does not check the
             NS_MAXCDNAME limit, but packing does, so we need to
             adjust the expected result.  */
          int expected;
          if (t->unpack_output.length > NS_MAXCDNAME)
            expected = -1;
          else if (strcmp (text, ".") == 0)
            /* The root domain is fully qualified.  */
            expected = 1;
          else
            /* The domain name is never fully qualified.  */
            expected = 0;
          unsigned char *repacked = xmalloc (NS_MAXCDNAME);
          ret = ns_name_pton (text, repacked, NS_MAXCDNAME);
          if (ret != expected)
            {
              support_record_failure ();
              printf ("%s:%zu: error: wrong result from ns_name_pton\n"
                      "  expected: %d\n"
                      "  actual:   %d\n",
                      t->path, t->lineno, expected, ret);
              return;
            }
          if (ret >= 0
              && memcmp (repacked, unpacked, t->unpack_output.length) != 0)
            {
              support_record_failure ();
              printf ("%s:%zu: error: wrong data from ns_name_pton\n",
                      t->path, t->lineno);
              print_hex ("expected:", t->unpack_output);
              print_hex ("actual:  ",
                         (struct buffer) { repacked, t->unpack_output.length });
              return;
            }

          /* Test ns_name_compress, no compression case.  */
          if (t->unpack_output.length > NS_MAXCDNAME)
            expected = -1;
          else
            expected = t->unpack_output.length;
          memset (repacked, '$', NS_MAXCDNAME);
          {
            enum { ptr_count = 5 };
            const unsigned char *dnptrs[ptr_count] = { repacked, };
            ret = ns_name_compress (text, repacked, NS_MAXCDNAME,
                                    dnptrs, dnptrs + ptr_count);
            if (ret != expected)
              {
                support_record_failure ();
                printf ("%s:%zu: error: wrong result from ns_name_compress\n"
                        "  expected: %d\n"
                        "  actual:   %d\n",
                        t->path, t->lineno, expected, ret);
                return;
              }
            if (ret < 0)
              {
                TEST_VERIFY (dnptrs[0] == repacked);
                TEST_VERIFY (dnptrs[1] == NULL);
              }
            else
              {
                if (memcmp (repacked, unpacked, t->unpack_output.length) != 0)
                  {
                    support_record_failure ();
                    printf ("%s:%zu: error: wrong data from ns_name_compress\n",
                            t->path, t->lineno);
                    print_hex ("expected:", t->unpack_output);
                    print_hex ("actual:  ", (struct buffer) { repacked, ret });
                    return;
                  }
                TEST_VERIFY (dnptrs[0] == repacked);
                if (unpacked[0] == '\0')
                  /* The root domain is not a compression target.  */
                  TEST_VERIFY (dnptrs[1] == NULL);
                else
                  {
                    TEST_VERIFY (dnptrs[1] == repacked);
                    TEST_VERIFY (dnptrs[2] == NULL);
                  }
              }
          }

          /* Test ns_name_compress, full compression case.  Skip this
             test for invalid names and the root domain.  */
          if (expected >= 0 && unpacked[0] != '\0')
            {
              /* The destination buffer needs additional room for the
                 offset, the initial name, and the compression
                 reference.  */
              enum { name_offset = 259 };
              size_t target_offset = name_offset + t->unpack_output.length;
              size_t repacked_size = target_offset + 2;
              repacked = xrealloc (repacked, repacked_size);
              memset (repacked, '@', repacked_size);
              memcpy (repacked + name_offset,
                      t->unpack_output.data, t->unpack_output.length);
              enum { ptr_count = 5 };
              const unsigned char *dnptrs[ptr_count]
                = { repacked, repacked + name_offset, };
              ret = ns_name_compress
                (text, repacked + target_offset, NS_MAXCDNAME,
                 dnptrs, dnptrs + ptr_count);
              if (ret != 2)
                {
                  support_record_failure ();
                  printf ("%s:%zu: error: wrong result from ns_name_compress"
                          " (2)\n"
                          "  expected: 2\n"
                          "  actual:   %d\n",
                          t->path, t->lineno, ret);
                  return;
                }
              if (memcmp (repacked + target_offset, "\xc1\x03", 2) != 0)
                {
                  support_record_failure ();
                  printf ("%s:%zu: error: wrong data from ns_name_compress"
                          " (2)\n"
                          "  expected: C103\n",
                          t->path, t->lineno);
                  print_hex ("actual:  ",
                             (struct buffer) { repacked + target_offset, ret });
                  return;
                }
              TEST_VERIFY (dnptrs[0] == repacked);
              TEST_VERIFY (dnptrs[1] == repacked + name_offset);
              TEST_VERIFY (dnptrs[2] == NULL);
            }

          free (repacked);
        }
      free (text);
    }
  free (unpacked);
}

/* Open the file at PATH, parse the test cases contained in it, and
   run them.  */
static void
run_test_file (const char *path)
{
  FILE *fp = xfopen (path, "re");
  char *line = NULL;
  size_t line_allocated = 0;
  size_t lineno = 0;

  while (true)
    {
      ssize_t ret = getline (&line, &line_allocated, fp);
      if (ret < 0)
        {
          if (ferror (fp))
            {
              printf ("%s: error reading file: %m\n", path);
              exit (1);
            }
          TEST_VERIFY (feof (fp));
          break;
        }

      ++lineno;
      char *p = line;
      while (isspace (*p))
        ++p;
      if (*p == '\0' || *p == '#')
        continue;

      struct test_case test_case;
      if (!parse_test_case (path, lineno, line, &test_case))
        {
          support_record_failure ();
          continue;
        }
      run_test_case (&test_case);
      free_test_case (&test_case);
    }
  free (line);
  xfclose (fp);
}

static int
do_test (void)
{
  run_test_file ("tst-ns_name.data");
  return 0;
}

#include <support/test-driver.c>
