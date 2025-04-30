/* Tests for the getentropy, getrandom functions.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/random.h>

/* Set to true if any errors are encountered.  */
static bool errors;

/* Test getrandom with a single buffer length.  NB: The passed-in
   buffer must have room for four extra bytes after the specified
   length, which are used to test that getrandom leaves those bytes
   unchanged.  */
static void
test_length (char *buffer, size_t length, unsigned int flags)
{
  memset (buffer, 0, length);
  strcpy (buffer + length, "123");
  ssize_t ret = getrandom (buffer, length, flags);
  if (ret < 0)
    {
      /* EAGAIN is an expected error with GRND_RANDOM and
         GRND_NONBLOCK.  */
      if ((flags & GRND_RANDOM)
          && (flags & GRND_NONBLOCK)
          && errno == EAGAIN)
        return;
      printf ("error: getrandom (%zu, 0x%x): %m\n", length, flags);
      errors = true;
      return;
    }
 if (ret != length)
    {
      if (flags & GRND_RANDOM)
        {
          if (ret == 0 || ret > length)
            {
              printf ("error: getrandom (%zu, 0x%x) returned %zd\n",
                      length, flags, ret);
              errors = true;
            }
        }
      else
        {
          printf ("error: getrandom (%zu, 0x%x) returned %zd\n",
                  length, flags, ret);
          errors = true;
        }
    }
  if (length >= 7)
    {
      /* One spurious test failure in 2**56 is sufficiently
         unlikely.  */
      int non_null = 0;
      for (int i = 0; i < length; ++i)
        non_null += buffer[i] != 0;
      if (non_null == 0)
        {
          printf ("error: getrandom (%zu, 0x%x) returned all-zero bytes\n",
                  length, flags);
          errors = true;
        }
    }
  if (memcmp (buffer + length, "123", 4) != 0)
    {
      printf ("error: getrandom (%zu, 0x%x) wrote spurious bytes\n",
              length, flags);
      errors = true;
    }
}

/* Call getrandom repeatedly to fill the buffer.  */
static bool
getrandom_full (char *buffer, size_t length, unsigned int flags)
{
  char *end = buffer + length;
  while (buffer < end)
    {
      ssize_t ret = getrandom (buffer, end - buffer, flags);
      if (ret < 0)
        {
          printf ("error: getrandom (%zu, 0x%x): %m\n", length, flags);
          errors = true;
          return false;
        }
      buffer += ret;
    }

  return true;
}

static void
test_flags (unsigned int flags)
{
  /* Test various lengths, but only for !GRND_RANDOM, to conserve
     entropy.  */
  {
    enum { max_length = 300 };
    char buffer[max_length + 4];
    if (flags & GRND_RANDOM)
      test_length (buffer, 0, flags);
    else
      {
        for (int length = 0; length <= 9; ++length)
          test_length (buffer, length, flags);
        test_length (buffer, 16, flags);
        test_length (buffer, max_length, flags);
      }
  }

  /* Test that getrandom returns different data.  */
  if (!(flags & GRND_NONBLOCK))
    {
      char buffer1[8];
      memset (buffer1, 0, sizeof (buffer1));

      char buffer2[8];
      memset (buffer2, 0, sizeof (buffer2));

      if (getrandom_full (buffer1, sizeof (buffer1), flags)
          && getrandom_full (buffer2, sizeof (buffer2), flags))
        {
          /* The probability that these two 8-byte buffers are equal
             is very small (assuming that two subsequent calls to
             getrandom result are independent, uniformly distributed
             random variables).  */
          if (memcmp (buffer1, buffer2, sizeof (buffer1)) == 0)
            {
              printf ("error: getrandom returns constant value\n");
              errors = true;
            }
        }
    }
}

static void
test_getentropy (void)
{
  char buf[16];
  memset (buf, '@', sizeof (buf));
  if (getentropy (buf, 0) != 0)
    {
      printf ("error: getentropy zero length: %m\n");
      errors = true;
      return;
    }
  for (size_t i = 0; i < sizeof (buf); ++i)
    if (buf[i] != '@')
      {
        printf ("error: getentropy modified zero-length buffer\n");
        errors = true;
        return;
      }

  if (getentropy (buf, sizeof (buf)) != 0)
    {
      printf ("error: getentropy buf: %m\n");
      errors = true;
      return;
    }

  char buf2[256];
  _Static_assert (sizeof (buf) < sizeof (buf2), "buf and buf2 compatible");
  memset (buf2, '@', sizeof (buf2));
  if (getentropy (buf2, sizeof (buf)) != 0)
    {
      printf ("error: getentropy buf2: %m\n");
      errors = true;
      return;
    }

  /* The probability that these two buffers are equal is very
     small. */
  if (memcmp (buf, buf2, sizeof (buf) == 0))
    {
      printf ("error: getentropy appears to return constant bytes\n");
      errors = true;
      return;
    }

  for (size_t i = sizeof (buf); i < sizeof (buf2); ++i)
    if (buf2[i] != '@')
      {
        printf ("error: getentropy wrote beyond the end of the buffer\n");
        errors = true;
        return;
      }

  char buf3[257];
  if (getentropy (buf3, sizeof (buf3)) == 0)
    {
      printf ("error: getentropy successful for 257 byte buffer\n");
      errors = true;
      return;
    }
  if (errno != EIO)
    {
      printf ("error: getentropy wrong error for 257 byte buffer: %m\n");
      errors = true;
      return;
    }
}

static int
do_test (void)
{
  /* Check if getrandom is not supported by this system.  */
  if (getrandom (NULL, 0, 0) == -1 && errno == ENOSYS)
    return 77;

  for (int use_random = 0; use_random < 2; ++use_random)
    for (int use_nonblock = 0; use_nonblock < 2; ++use_nonblock)
      {
        unsigned int flags = 0;
        if (use_random)
          flags |= GRND_RANDOM;
        if (use_nonblock)
          flags |= GRND_NONBLOCK;
        test_flags (flags);
      }

  test_getentropy ();

  return errors;
}

#include <support/test-driver.c>
