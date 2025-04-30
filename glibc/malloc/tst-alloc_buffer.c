/* Tests for struct alloc_buffer.
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

#include <arpa/inet.h>
#include <alloc_buffer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>

/* Return true if PTR is sufficiently aligned for TYPE.  */
#define IS_ALIGNED(ptr, type) \
  ((((uintptr_t) ptr) & (__alloc_buffer_assert_align (__alignof (type)) - 1)) \
   == 0)

/* Structure with non-power-of-two size.  */
struct twelve
{
  uint32_t buffer[3] __attribute__ ((aligned (4)));
};
_Static_assert (sizeof (struct twelve) == 12, "struct twelve");
_Static_assert (__alignof__ (struct twelve) == 4, "struct twelve");

/* Check for success obtaining empty arrays.  Does not assume the
   buffer is empty.  */
static void
test_empty_array (struct alloc_buffer refbuf)
{
  bool refbuf_failed = alloc_buffer_has_failed (&refbuf);
  if (test_verbose)
    printf ("info: %s: current=0x%llx end=0x%llx refbuf_failed=%d\n",
            __func__, (unsigned long long) refbuf.__alloc_buffer_current,
            (unsigned long long) refbuf.__alloc_buffer_end, refbuf_failed);
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY ((alloc_buffer_alloc_bytes (&buf, 0) == NULL)
                 == refbuf_failed);
    TEST_VERIFY (alloc_buffer_has_failed (&buf) == refbuf_failed);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY ((alloc_buffer_alloc_array (&buf, char, 0) == NULL)
                 == refbuf_failed);
    TEST_VERIFY (alloc_buffer_has_failed (&buf) == refbuf_failed);
  }
  /* The following tests can fail due to the need for aligning the
     returned pointer.  */
  {
    struct alloc_buffer buf = refbuf;
    bool expect_failure = refbuf_failed
      || !IS_ALIGNED (alloc_buffer_next (&buf, void), double);
    double *ptr = alloc_buffer_alloc_array (&buf, double, 0);
    TEST_VERIFY (IS_ALIGNED (ptr, double));
    TEST_VERIFY ((ptr == NULL) == expect_failure);
    TEST_VERIFY (alloc_buffer_has_failed (&buf) == expect_failure);
  }
  {
    struct alloc_buffer buf = refbuf;
    bool expect_failure = refbuf_failed
      || !IS_ALIGNED (alloc_buffer_next (&buf, void), struct twelve);
    struct twelve *ptr = alloc_buffer_alloc_array (&buf, struct twelve, 0);
    TEST_VERIFY (IS_ALIGNED (ptr, struct twelve));
    TEST_VERIFY ((ptr == NULL) == expect_failure);
    TEST_VERIFY (alloc_buffer_has_failed (&buf) == expect_failure);
  }
}

/* Test allocation of impossibly large arrays.  */
static void
test_impossible_array (struct alloc_buffer refbuf)
{
  if (test_verbose)
    printf ("info: %s: current=0x%llx end=0x%llx\n",
            __func__, (unsigned long long) refbuf.__alloc_buffer_current,
            (unsigned long long) refbuf.__alloc_buffer_end);
  static const size_t counts[] =
    { SIZE_MAX, SIZE_MAX - 1, SIZE_MAX - 2, SIZE_MAX - 3, SIZE_MAX - 4,
      SIZE_MAX / 2, SIZE_MAX / 2 + 1, SIZE_MAX / 2 - 1, 0};

  for (int i = 0; counts[i] != 0; ++i)
    {
      size_t count = counts[i];
      if (test_verbose)
        printf ("info: %s: count=%zu\n", __func__, count);
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_bytes (&buf, count) == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_array (&buf, char, count) == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_array (&buf, short, count) == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_array (&buf, double, count) == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_array (&buf, struct twelve, count)
                     == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
    }
}

/* Check for failure to obtain anything from a failed buffer.  */
static void
test_after_failure (struct alloc_buffer refbuf)
{
  if (test_verbose)
    printf ("info: %s: current=0x%llx end=0x%llx\n",
            __func__, (unsigned long long) refbuf.__alloc_buffer_current,
            (unsigned long long) refbuf.__alloc_buffer_end);
  TEST_VERIFY (alloc_buffer_has_failed (&refbuf));
  {
    struct alloc_buffer buf = refbuf;
    alloc_buffer_add_byte (&buf, 17);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, char) == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, double) == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, struct twelve) == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
  }

  test_impossible_array (refbuf);
  for (int count = 0; count <= 4; ++count)
    {
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_bytes (&buf, count) == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_array (&buf, char, count) == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_array (&buf, double, count) == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
      {
        struct alloc_buffer buf = refbuf;
        TEST_VERIFY (alloc_buffer_alloc_array (&buf, struct twelve, count)
                     == NULL);
        TEST_VERIFY (alloc_buffer_has_failed (&buf));
      }
    }
}

static void
test_empty (struct alloc_buffer refbuf)
{
  TEST_VERIFY (alloc_buffer_size (&refbuf) == 0);
  if (alloc_buffer_next (&refbuf, void) != NULL)
    TEST_VERIFY (!alloc_buffer_has_failed (&refbuf));
  test_empty_array (refbuf);
  test_impossible_array (refbuf);

  /* Failure to obtain non-empty objects.  */
  {
    struct alloc_buffer buf = refbuf;
    alloc_buffer_add_byte (&buf, 17);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, char) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, double) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, struct twelve) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, char, 1) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, double, 1) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, struct twelve, 1) == NULL);
    test_after_failure (buf);
  }
}

static void
test_size_1 (struct alloc_buffer refbuf)
{
  TEST_VERIFY (!alloc_buffer_has_failed (&refbuf));
  TEST_VERIFY (alloc_buffer_size (&refbuf) == 1);
  test_empty_array (refbuf);
  test_impossible_array (refbuf);

  /* Success adding a single byte.  */
  {
    struct alloc_buffer buf = refbuf;
    alloc_buffer_add_byte (&buf, 17);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    test_empty (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "\x11", 1) == 0);
  {
    struct alloc_buffer buf = refbuf;
    signed char *ptr = alloc_buffer_alloc (&buf, signed char);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    *ptr = 126;
    test_empty (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "\176", 1) == 0);
  {
    struct alloc_buffer buf = refbuf;
    char *ptr = alloc_buffer_alloc_array (&buf, char, 1);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    *ptr = (char) 253;
    test_empty (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "\xfd", 1) == 0);

  /* Failure with larger objects.  */
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, short) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, double) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, struct twelve) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, short, 1) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, double, 1) == NULL);
    test_after_failure (buf);
  }
  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, struct twelve, 1) == NULL);
    test_after_failure (buf);
  }
}

static void
test_size_2 (struct alloc_buffer refbuf)
{
  TEST_VERIFY (!alloc_buffer_has_failed (&refbuf));
  TEST_VERIFY (alloc_buffer_size (&refbuf) == 2);
  TEST_VERIFY (IS_ALIGNED (alloc_buffer_next (&refbuf, void), short));
  test_empty_array (refbuf);
  test_impossible_array (refbuf);

  /* Success adding two bytes.  */
  {
    struct alloc_buffer buf = refbuf;
    alloc_buffer_add_byte (&buf, '@');
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    test_size_1 (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "@\xfd", 2) == 0);
  {
    struct alloc_buffer buf = refbuf;
    signed char *ptr = alloc_buffer_alloc (&buf, signed char);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    *ptr = 'A';
    test_size_1 (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "A\xfd", 2) == 0);
  {
    struct alloc_buffer buf = refbuf;
    char *ptr = alloc_buffer_alloc_array (&buf, char, 1);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    *ptr = 'B';
    test_size_1 (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "B\xfd", 2) == 0);
  {
    struct alloc_buffer buf = refbuf;
    unsigned short *ptr = alloc_buffer_alloc (&buf, unsigned short);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (IS_ALIGNED (ptr, unsigned short));
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    *ptr = htons (0x12f4);
    test_empty (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "\x12\xf4", 2) == 0);
  {
    struct alloc_buffer buf = refbuf;
    unsigned short *ptr = alloc_buffer_alloc_array (&buf, unsigned short, 1);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (IS_ALIGNED (ptr, unsigned short));
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    *ptr = htons (0x13f5);
    test_empty (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "\x13\xf5", 2) == 0);
  {
    struct alloc_buffer buf = refbuf;
    char *ptr = alloc_buffer_alloc_array (&buf, char, 2);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    memcpy (ptr, "12", 2);
    test_empty (buf);
  }
  TEST_VERIFY (memcmp (alloc_buffer_next (&refbuf, void), "12", 2) == 0);
}

static void
test_misaligned (char pad)
{
  enum { SIZE = 23 };
  char *backing = xmalloc (SIZE + 2);
  backing[0] = ~pad;
  backing[SIZE + 1] = pad;
  struct alloc_buffer refbuf = alloc_buffer_create (backing + 1, SIZE);

  {
    struct alloc_buffer buf = refbuf;
    short *ptr = alloc_buffer_alloc_array (&buf, short, SIZE / sizeof (short));
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (IS_ALIGNED (ptr, short));
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    for (int i = 0; i < SIZE / sizeof (short); ++i)
      ptr[i] = htons (0xff01 + i);
    TEST_VERIFY (memcmp (ptr,
                         "\xff\x01\xff\x02\xff\x03\xff\x04"
                         "\xff\x05\xff\x06\xff\x07\xff\x08"
                         "\xff\x09\xff\x0a\xff\x0b", 22) == 0);
  }
  {
    struct alloc_buffer buf = refbuf;
    uint32_t *ptr = alloc_buffer_alloc_array
      (&buf, uint32_t, SIZE / sizeof (uint32_t));
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (IS_ALIGNED (ptr, uint32_t));
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    for (int i = 0; i < SIZE / sizeof (uint32_t); ++i)
      ptr[i] = htonl (0xf1e2d301 + i);
    TEST_VERIFY (memcmp (ptr,
                         "\xf1\xe2\xd3\x01\xf1\xe2\xd3\x02"
                         "\xf1\xe2\xd3\x03\xf1\xe2\xd3\x04"
                         "\xf1\xe2\xd3\x05", 20) == 0);
  }
  {
    struct alloc_buffer buf = refbuf;
    struct twelve *ptr = alloc_buffer_alloc (&buf, struct twelve);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (IS_ALIGNED (ptr, struct twelve));
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    ptr->buffer[0] = htonl (0x11223344);
    ptr->buffer[1] = htonl (0x55667788);
    ptr->buffer[2] = htonl (0x99aabbcc);
    TEST_VERIFY (memcmp (ptr,
                         "\x11\x22\x33\x44"
                         "\x55\x66\x77\x88"
                         "\x99\xaa\xbb\xcc", 12) == 0);
  }
  {
    static const double nums[] = { 1, 2 };
    struct alloc_buffer buf = refbuf;
    double *ptr = alloc_buffer_alloc_array (&buf, double, 2);
    TEST_VERIFY_EXIT (ptr != NULL);
    TEST_VERIFY (IS_ALIGNED (ptr, double));
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    ptr[0] = nums[0];
    ptr[1] = nums[1];
    TEST_VERIFY (memcmp (ptr, nums, sizeof (nums)) == 0);
  }

  /* Verify that padding was not overwritten.  */
  TEST_VERIFY (backing[0] == (char) ~pad);
  TEST_VERIFY (backing[SIZE + 1] == pad);
  free (backing);
}

/* Check that overflow during alignment is handled properly.  */
static void
test_large_misaligned (void)
{
  uintptr_t minus1 = -1;
  uintptr_t start = minus1 & ~0xfe;
  struct alloc_buffer refbuf = alloc_buffer_create ((void *) start, 16);
  TEST_VERIFY (!alloc_buffer_has_failed (&refbuf));

  struct __attribute__ ((aligned (256))) align256
  {
    int dymmy;
  };

  {
    struct alloc_buffer buf = refbuf;
    TEST_VERIFY (alloc_buffer_alloc (&buf, struct align256) == NULL);
    test_after_failure (buf);
  }
  for (int count = 0; count < 3; ++count)
    {
      struct alloc_buffer buf = refbuf;
      TEST_VERIFY (alloc_buffer_alloc_array (&buf, struct align256, count)
                   == NULL);
      test_after_failure (buf);
    }
}

/* Check behavior of large allocations.  */
static void
test_large (void)
{
  {
    /* Allocation which wraps around.  */
    struct alloc_buffer buf = { 1, SIZE_MAX };
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, char, SIZE_MAX) == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
  }

  {
    /* Successful very large allocation.  */
    struct alloc_buffer buf = { 1, SIZE_MAX };
    uintptr_t val = (uintptr_t) alloc_buffer_alloc_array
      (&buf, char, SIZE_MAX - 1);
    TEST_VERIFY (val == 1);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    test_empty (buf);
  }

  {
    typedef char __attribute__ ((aligned (2))) char2;

    /* Overflow in array size computation.   */
    struct alloc_buffer buf = { 1, SIZE_MAX };
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, char2, SIZE_MAX - 1) == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));

    /* Successful allocation after alignment.  */
    buf = (struct alloc_buffer) { 1, SIZE_MAX };
    uintptr_t val = (uintptr_t) alloc_buffer_alloc_array
      (&buf, char2, SIZE_MAX - 2);
    TEST_VERIFY (val == 2);
    test_empty (buf);

    /* Alignment behavior near the top of the address space.  */
    buf = (struct alloc_buffer) { SIZE_MAX, SIZE_MAX };
    TEST_VERIFY (alloc_buffer_next (&buf, char2) == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
    buf = (struct alloc_buffer) { SIZE_MAX, SIZE_MAX };
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, char2, 0) == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
  }

  {
    typedef short __attribute__ ((aligned (2))) short2;

    /* Test overflow in size computation.  */
    struct alloc_buffer buf = { 1, SIZE_MAX };
    TEST_VERIFY (alloc_buffer_alloc_array (&buf, short2, SIZE_MAX / 2)
                 == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));

    /* A slightly smaller array fits within the allocation.  */
    buf = (struct alloc_buffer) { 2, SIZE_MAX - 1 };
    uintptr_t val = (uintptr_t) alloc_buffer_alloc_array
      (&buf, short2, SIZE_MAX / 2 - 1);
    TEST_VERIFY (val == 2);
    test_empty (buf);
  }
}

static void
test_copy_bytes (void)
{
  char backing[4];
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    alloc_buffer_copy_bytes (&buf, "1", 1);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    TEST_VERIFY (alloc_buffer_size (&buf) == 3);
    TEST_VERIFY (memcmp (backing, "1@@@", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    alloc_buffer_copy_bytes (&buf, "12", 3);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    TEST_VERIFY (alloc_buffer_size (&buf) == 1);
    TEST_VERIFY (memcmp (backing, "12\0@", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    alloc_buffer_copy_bytes (&buf, "1234", 4);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    TEST_VERIFY (alloc_buffer_size (&buf) == 0);
    TEST_VERIFY (memcmp (backing, "1234", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    alloc_buffer_copy_bytes (&buf, "1234", 5);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
    TEST_VERIFY (memcmp (backing, "@@@@", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    alloc_buffer_copy_bytes (&buf, "1234", -1);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
    TEST_VERIFY (memcmp (backing, "@@@@", 4) == 0);
  }
}

static void
test_copy_string (void)
{
  char backing[4];
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    const char *p = alloc_buffer_copy_string (&buf, "");
    TEST_VERIFY (p == backing);
    TEST_VERIFY (strcmp (p, "") == 0);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    TEST_VERIFY (alloc_buffer_size (&buf) == 3);
    TEST_VERIFY (memcmp (backing, "\0@@@", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    const char *p = alloc_buffer_copy_string (&buf, "1");
    TEST_VERIFY (p == backing);
    TEST_VERIFY (strcmp (p, "1") == 0);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    TEST_VERIFY (alloc_buffer_size (&buf) == 2);
    TEST_VERIFY (memcmp (backing, "1\0@@", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    const char *p = alloc_buffer_copy_string (&buf, "12");
    TEST_VERIFY (p == backing);
    TEST_VERIFY (strcmp (p, "12") == 0);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    TEST_VERIFY (alloc_buffer_size (&buf) == 1);
    TEST_VERIFY (memcmp (backing, "12\0@", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    const char *p = alloc_buffer_copy_string (&buf, "123");
    TEST_VERIFY (p == backing);
    TEST_VERIFY (strcmp (p, "123") == 0);
    TEST_VERIFY (!alloc_buffer_has_failed (&buf));
    TEST_VERIFY (alloc_buffer_size (&buf) == 0);
    TEST_VERIFY (memcmp (backing, "123", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    TEST_VERIFY (alloc_buffer_copy_string (&buf, "1234") == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
    TEST_VERIFY (memcmp (backing, "@@@@", 4) == 0);
  }
  {
    memset (backing, '@', sizeof (backing));
    struct alloc_buffer buf = alloc_buffer_create (backing, sizeof (backing));
    TEST_VERIFY (alloc_buffer_copy_string (&buf, "12345") == NULL);
    TEST_VERIFY (alloc_buffer_has_failed (&buf));
    TEST_VERIFY (memcmp (backing, "@@@@", 4) == 0);
  }
}

static int
do_test (void)
{
  test_empty (alloc_buffer_create (NULL, 0));
  test_empty (alloc_buffer_create ((char *) "", 0));
  test_empty (alloc_buffer_create ((void *) 1, 0));

  {
    void *ptr = (void *) "";    /* Cannot be freed. */
    struct alloc_buffer buf = alloc_buffer_allocate (1, &ptr);
    test_size_1 (buf);
    free (ptr);                 /* Should have been overwritten.  */
  }

  {
    void *ptr= (void *) "";     /* Cannot be freed.  */
    struct alloc_buffer buf = alloc_buffer_allocate (2, &ptr);
    test_size_2 (buf);
    free (ptr);                 /* Should have been overwritten.  */
  }

  test_misaligned (0);
  test_misaligned (0xc7);
  test_misaligned (0xff);

  test_large_misaligned ();
  test_large ();
  test_copy_bytes ();
  test_copy_string ();

  return 0;
}

#include <support/test-driver.c>
