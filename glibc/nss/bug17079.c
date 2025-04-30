/* Test for bug 17079: heap overflow in NSS with small buffers.
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

#include <errno.h>
#include <nss.h>
#include <pwd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <support/support.h>

/* Check if two passwd structs contain the same data.  */
static bool
equal (const struct passwd *a, const struct passwd *b)
{
  return strcmp (a->pw_name, b->pw_name) == 0
    && strcmp (a->pw_passwd, b->pw_passwd) == 0
    && a->pw_uid == b->pw_uid
    && a->pw_gid == b->pw_gid
    && strcmp (a->pw_gecos, b->pw_gecos) == 0
    && strcmp (a->pw_dir, b->pw_dir) == 0
    && strcmp (a->pw_shell, b->pw_shell) == 0;
}

enum { MAX_TEST_ITEMS = 10 };
static struct passwd test_items[MAX_TEST_ITEMS];
static int test_count;

/* Initialize test_items and test_count above, with data from the
   passwd database.  */
static bool
init_test_items (void)
{
  setpwent ();
  do
    {
      struct passwd *pwd = getpwent ();
      if (pwd == NULL)
        break;
      struct passwd *target = test_items + test_count;
      target->pw_name = xstrdup (pwd->pw_name);
      target->pw_passwd = xstrdup (pwd->pw_passwd);
      target->pw_uid = pwd->pw_uid;
      target->pw_gid = pwd->pw_gid;
      target->pw_gecos = xstrdup (pwd->pw_gecos);
      target->pw_dir = xstrdup (pwd->pw_dir);
      target->pw_shell = xstrdup (pwd->pw_shell);
    }
  while (++test_count < MAX_TEST_ITEMS);
  endpwent ();

  /* Filter out those test items which cannot be looked up by name or
     UID.  */
  bool found = false;
  for (int i = 0; i < test_count; ++i)
    {
      struct passwd *pwd1 = getpwnam (test_items[i].pw_name);
      struct passwd *pwd2 = getpwuid (test_items[i].pw_uid);
      if (pwd1 == NULL || !equal (pwd1, test_items + i)
          || pwd2 == NULL || !equal (pwd2, test_items + i))
        {
          printf ("info: skipping user \"%s\", UID %ld due to inconsistency\n",
                  test_items[i].pw_name, (long) test_items[i].pw_uid);
          test_items[i].pw_name = NULL;
        }
      else
        found = true;
    }

  if (!found)
    puts ("error: no accounts found which can be looked up by name and UID.");
  return found;
}

/* Set to true if an error is encountered.  */
static bool errors;

/* Return true if the padding has not been tampered with.  */
static bool
check_padding (char *buffer, size_t size, char pad)
{
  char *end = buffer + size;
  while (buffer < end)
    {
      if (*buffer != pad)
        return false;
      ++buffer;
    }
  return true;
}

/* Test one buffer size and padding combination.  */
static void
test_one (const struct passwd *item, size_t buffer_size,
           char pad, size_t padding_size)
{
  char *buffer = xmalloc (buffer_size + padding_size);

  struct passwd pwd;
  struct passwd *result;
  int ret;

  /* Test getpwname_r.  */
  memset (buffer, pad, buffer_size + padding_size);
  pwd = (struct passwd) {};
  ret = getpwnam_r (item->pw_name, &pwd, buffer, buffer_size, &result);
  if (!check_padding (buffer + buffer_size, padding_size, pad))
    {
      printf ("error: padding change: "
              "name \"%s\", buffer size %zu, padding size %zu, pad 0x%02x\n",
              item->pw_name, buffer_size, padding_size, (unsigned char) pad);
      errors = true;
    }
  if (ret == 0)
    {
      if (result == NULL)
        {
          printf ("error: no data: name \"%s\", buffer size %zu\n",
                  item->pw_name, buffer_size);
          errors = true;
        }
      else if (!equal (item, result))
        {
          printf ("error: lookup mismatch: name \"%s\", buffer size %zu\n",
                  item->pw_name, buffer_size);
          errors = true;
        }
    }
  else if (ret != ERANGE)
    {
      errno = ret;
      printf ("error: lookup failure for name \"%s\": %m (%d)\n",
              item->pw_name, ret);
      errors = true;
    }

  /* Test getpwuid_r.  */
  memset (buffer, pad, buffer_size + padding_size);
  pwd = (struct passwd) {};
  ret = getpwuid_r (item->pw_uid, &pwd, buffer, buffer_size, &result);
  if (!check_padding (buffer + buffer_size, padding_size, pad))
    {
      printf ("error: padding change: "
              "UID %ld, buffer size %zu, padding size %zu, pad 0x%02x\n",
              (long) item->pw_uid, buffer_size, padding_size,
              (unsigned char) pad);
      errors = true;
    }
  if (ret == 0)
    {
      if (result == NULL)
        {
          printf ("error: no data: UID %ld, buffer size %zu\n",
                  (long) item->pw_uid, buffer_size);
          errors = true;
        }
      else if (!equal (item, result))
        {
          printf ("error: lookup mismatch: UID %ld, buffer size %zu\n",
                  (long) item->pw_uid, buffer_size);
          errors = true;
        }
    }
  else if (ret != ERANGE)
    {
      errno = ret;
      printf ("error: lookup failure for UID \"%ld\": %m (%d)\n",
              (long) item->pw_uid, ret);
      errors = true;
    }

  free (buffer);
}

/* Test one buffer size with different paddings.  */
static void
test_buffer_size (size_t buffer_size)
{
  for (int i = 0; i < test_count; ++i)
    for (size_t padding_size = 0; padding_size < 3; ++padding_size)
      {
        /* Skip entries with inconsistent name/UID lookups.  */
        if (test_items[i].pw_name == NULL)
          continue;

        test_one (test_items + i, buffer_size, '\0', padding_size);
        if (padding_size > 0)
          {
            test_one (test_items + i, buffer_size, ':', padding_size);
            test_one (test_items + i, buffer_size, '\n', padding_size);
            test_one (test_items + i, buffer_size, '\xff', padding_size);
            test_one (test_items + i, buffer_size, '@', padding_size);
          }
      }
}

int
do_test (void)
{
  __nss_configure_lookup ("passwd", "files");

  if (!init_test_items ())
    return 1;
  printf ("info: %d test items\n", test_count);

  for (size_t buffer_size = 0; buffer_size <= 65; ++buffer_size)
    test_buffer_size (buffer_size);
  for (size_t buffer_size = 64 + 4; buffer_size < 256; buffer_size += 4)
    test_buffer_size (buffer_size);
  test_buffer_size (255);
  test_buffer_size (257);
  for (size_t buffer_size = 256; buffer_size < 512; buffer_size += 8)
    test_buffer_size (buffer_size);
  test_buffer_size (511);
  test_buffer_size (513);
  test_buffer_size (1024);
  test_buffer_size (2048);

  if (errors)
    return 1;
  else
    return 0;
}

#include <support/test-driver.c>
