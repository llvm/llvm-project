#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>


static int
do_test (void)
{
  size_t size;
  wchar_t *buf;
  FILE *fp = open_wmemstream (&buf, &size);
  if (fp == NULL)
    {
      puts ("open_wmemstream failed");
      return 1;
    }

  off64_t off = ftello64 (fp);
  if (off != 0)
    {
      puts ("initial position wrong");
      return 1;
    }

  if (fseek (fp, 32768, SEEK_SET) != 0)
    {
      puts ("fseek failed");
      return 1;
    }

  if (fputws (L"foo", fp) == EOF)
    {
      puts ("fputws failed");
      return 1;
    }

  if (fclose (fp) == EOF)
    {
      puts ("fclose failed");
      return 1;
    }

  if (size != 32768 + 3)
    {
      printf ("expected size %d, got %zu\n", 32768 + 3, size);
      return 1;
    }

  for (int i = 0; i < 32768; ++i)
    if (buf[i] != L'\0')
      {
	printf ("wide character at offset %d is %#x\n",
		i, (unsigned int) buf[i]);
	return 1;
      }

  if (wmemcmp (buf + 32768, L"foo", 3) != 0)
    {
      puts ("written string incorrect");
      return 1;
    }

  /* Mark the buffer.  */
  wmemset (buf, L'A', size);
  free (buf);

  /* Try again, this time with write mode enabled before the seek.  */
  fp = open_wmemstream (&buf, &size);
  if (fp == NULL)
    {
      puts ("2nd open_wmemstream failed");
      return 1;
    }

  off = ftello64 (fp);
  if (off != 0)
    {
      puts ("2nd initial position wrong");
      return 1;
    }

  if (fputws (L"bar", fp) == EOF)
    {
      puts ("2nd fputws failed");
      return 1;
    }

  if (fseek (fp, 32768, SEEK_SET) != 0)
    {
      puts ("2nd fseek failed");
      return 1;
    }

  if (fputws (L"foo", fp) == EOF)
    {
      puts ("3rd fputws failed");
      return 1;
    }

  if (fclose (fp) == EOF)
    {
      puts ("2nd fclose failed");
      return 1;
    }

  if (size != 32768 + 3)
    {
      printf ("2nd expected size %d, got %zu\n", 32768 + 3, size);
      return 1;
    }

  if (wmemcmp (buf, L"bar", 3) != 0)
    {
      puts ("initial string incorrect in 2nd try");
      return 1;
    }

  for (int i = 3; i < 32768; ++i)
    if (buf[i] != L'\0')
      {
	printf ("wide character at offset %d is %#x in 2nd try\n",
		i, (unsigned int) buf[i]);
	return 1;
      }

  if (wmemcmp (buf + 32768, L"foo", 3) != 0)
    {
      puts ("written string incorrect in 2nd try");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
