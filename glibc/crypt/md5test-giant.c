/* Testcase for https://sourceware.org/bugzilla/show_bug.cgi?id=14090.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "md5.h"

/* This test will not work with 32-bit size_t, so let it succeed
   there.  */
#if SIZE_MAX <= UINT32_MAX
static int
do_test (void)
{
  return 0;
}
#else

# define CONST_2G  0x080000000
# define CONST_10G 0x280000000

/* MD5 sum values of zero-filled blocks of specified sizes.  */
static const struct test_data_s
{
  const char ref[16];
  size_t len;
} test_data[] =
  {
    { "\xd4\x1d\x8c\xd9\x8f\x00\xb2\x04\xe9\x80\x09\x98\xec\xf8\x42\x7e",
      0x000000000 },
    { "\xa9\x81\x13\x0c\xf2\xb7\xe0\x9f\x46\x86\xdc\x27\x3c\xf7\x18\x7e",
      0x080000000 },
    { "\xc9\xa5\xa6\x87\x8d\x97\xb4\x8c\xc9\x65\xc1\xe4\x18\x59\xf0\x34",
      0x100000000 },
    { "\x58\xcf\x63\x8a\x73\x3f\x91\x90\x07\xb4\x28\x7c\xf5\x39\x6d\x0c",
      0x180000000 },
    { "\xb7\x70\x35\x1f\xad\xae\x5a\x96\xbb\xaf\x97\x02\xed\x97\xd2\x8d",
      0x200000000 },
    { "\x2d\xd2\x6c\x4d\x47\x99\xeb\xd2\x9f\xa3\x1e\x48\xd4\x9e\x8e\x53",
      0x280000000 },
};

static int
report (const char *id, const char *md5, size_t len, const char *ref)
{
  if (memcmp (md5, ref, 16))
    {
      printf ("test %s with size %zd failed\n", id, len);
      return 1;
    }
  return 0;
}

/* Test md5 in a single md5_process_bytes call.  */
static int
test_single (void *buf, size_t len, const char *ref)
{
  char sum[16];
  struct md5_ctx ctx;

  __md5_init_ctx (&ctx);
  __md5_process_bytes (buf, len, &ctx);
  __md5_finish_ctx (&ctx, sum);

  return report ("single", sum, len, ref);
}

/* Test md5 with two md5_process_bytes calls to trigger a
   different path in md5_process_block for sizes > 2 GB.  */
static int
test_double (void *buf, size_t len, const char *ref)
{
  char sum[16];
  struct md5_ctx ctx;

  __md5_init_ctx (&ctx);
  if (len >= CONST_2G)
    {
      __md5_process_bytes (buf, CONST_2G, &ctx);
      __md5_process_bytes (buf + CONST_2G, len - CONST_2G, &ctx);
    }
  else
    __md5_process_bytes (buf, len, &ctx);

  __md5_finish_ctx (&ctx, sum);

  return report ("double", sum, len, ref);
}


static int
do_test (void)
{
  void *buf;
  unsigned int j;
  int result = 0;

  buf = mmap64 (0, CONST_10G, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (buf == MAP_FAILED)
    {
      puts ("Could not allocate 10 GB via mmap, skipping test.");
      return 0;
    }

  for (j = 0; j < sizeof (test_data) / sizeof (struct test_data_s); j++)
    {
      if (test_single (buf, test_data[j].len, test_data[j].ref))
	result = 1;
      if (test_double (buf, test_data[j].len, test_data[j].ref))
	result = 1;
    }

  return result;
}
#endif

/* This needs on a fast machine 90s.  */
#define TIMEOUT 480
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
