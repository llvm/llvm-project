/* Check for file descriptor leak in alias :include: processing (bug 23521).
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <aliases.h>
#include <array_length.h>
#include <dlfcn.h>
#include <errno.h>
#include <gnu/lib-names.h>
#include <nss.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xstdio.h>
#include <support/xunistd.h>

static struct support_chroot *chroot_env;

/* Number of the aliases for the "many" user.  This must be large
   enough to trigger reallocation for the pointer array, but result in
   answers below the maximum size tried in do_test.  */
enum { many_aliases = 30 };

static void
prepare (int argc, char **argv)
{
  chroot_env = support_chroot_create
    ((struct support_chroot_configuration) { } );

  char *path = xasprintf ("%s/etc/aliases", chroot_env->path_chroot);
  add_temp_file (path);
  support_write_file_string
    (path,
     "user1: :include:/etc/aliases.user1\n"
     "user2: :include:/etc/aliases.user2\n"
     "comment: comment1, :include:/etc/aliases.comment\n"
     "many: :include:/etc/aliases.many\n");
  free (path);

  path = xasprintf ("%s/etc/aliases.user1", chroot_env->path_chroot);
  add_temp_file (path);
  support_write_file_string (path, "alias1\n");
  free (path);

  path = xasprintf ("%s/etc/aliases.user2", chroot_env->path_chroot);
  add_temp_file (path);
  support_write_file_string (path, "alias1a, alias2\n");
  free (path);

  path = xasprintf ("%s/etc/aliases.comment", chroot_env->path_chroot);
  add_temp_file (path);
  support_write_file_string
    (path,
     /* The line must be longer than the line with the :include:
        directive in /etc/aliases.  */
     "# Long line.  ##############################################\n"
     "comment2\n");
  free (path);

  path = xasprintf ("%s/etc/aliases.many", chroot_env->path_chroot);
  add_temp_file (path);
  FILE *fp = xfopen (path, "w");
  for (int i = 0; i < many_aliases; ++i)
    fprintf (fp, "a%d\n", i);
  TEST_VERIFY_EXIT (! ferror (fp));
  xfclose (fp);
  free (path);
}

/* The names of the users to test.  */
static const char *users[] = { "user1", "user2", "comment", "many" };

static void
check_aliases (int id, const struct aliasent *e)
{
  TEST_VERIFY_EXIT (id >= 0 || id < array_length (users));
  const char *name = users[id];
  TEST_COMPARE_BLOB (e->alias_name, strlen (e->alias_name),
                     name, strlen (name));

  switch (id)
    {
    case 0:
      TEST_COMPARE (e->alias_members_len, 1);
      TEST_COMPARE_BLOB (e->alias_members[0], strlen (e->alias_members[0]),
                         "alias1", strlen ("alias1"));
      break;

    case 1:
      TEST_COMPARE (e->alias_members_len, 2);
      TEST_COMPARE_BLOB (e->alias_members[0], strlen (e->alias_members[0]),
                         "alias1a", strlen ("alias1a"));
      TEST_COMPARE_BLOB (e->alias_members[1], strlen (e->alias_members[1]),
                         "alias2", strlen ("alias2"));
      break;

    case 2:
      TEST_COMPARE (e->alias_members_len, 2);
      TEST_COMPARE_BLOB (e->alias_members[0], strlen (e->alias_members[0]),
                         "comment1", strlen ("comment1"));
      TEST_COMPARE_BLOB (e->alias_members[1], strlen (e->alias_members[1]),
                         "comment2", strlen ("comment2"));
      break;

    case 3:
      TEST_COMPARE (e->alias_members_len, many_aliases);
      for (int i = 0; i < e->alias_members_len; ++i)
        {
          char alias[30];
          int len = snprintf (alias, sizeof (alias), "a%d", i);
          TEST_VERIFY_EXIT (len > 0);
          TEST_COMPARE_BLOB (e->alias_members[i], strlen (e->alias_members[i]),
                             alias, len);
        }
      break;
    }
}

static int
do_test (void)
{
  /* Make sure we don't try to load the module in the chroot.  */
  if (dlopen (LIBNSS_FILES_SO, RTLD_NOW) == NULL)
    FAIL_EXIT1 ("could not load " LIBNSS_FILES_SO ": %s", dlerror ());

  /* Some of these descriptors will become unavailable if there is a
     file descriptor leak.  10 is chosen somewhat arbitrarily.  The
     array must be longer than the number of files opened by nss_files
     at the same time (currently that number is 2).  */
  int next_descriptors[10];
  for (size_t i = 0; i < array_length (next_descriptors); ++i)
    {
      next_descriptors[i] = dup (0);
      TEST_VERIFY_EXIT (next_descriptors[i] > 0);
    }
  for (size_t i = 0; i < array_length (next_descriptors); ++i)
    xclose (next_descriptors[i]);

  support_become_root ();
  if (!support_can_chroot ())
    return EXIT_UNSUPPORTED;

  __nss_configure_lookup ("aliases", "files");

  xchroot (chroot_env->path_chroot);

  /* Attempt various buffer sizes.  If the operation succeeds, we
     expect correct data.  */
  for (int id = 0; id < array_length (users); ++id)
    {
      bool found = false;
      for (size_t size = 1; size <= 1000; ++size)
        {
          void *buffer = malloc (size);
          struct aliasent result;
          struct aliasent *res;
          errno = EINVAL;
          int ret = getaliasbyname_r (users[id], &result, buffer, size, &res);
          if (ret == 0)
            {
              if (res != NULL)
                {
                  found = true;
                  check_aliases (id, res);
                }
              else
                {
                  support_record_failure ();
                  printf ("error: failed lookup for user \"%s\", size %zu\n",
                          users[id], size);
                }
            }
          else if (ret != ERANGE)
            {
              support_record_failure ();
              printf ("error: invalid return code %d (user \"%s\", size %zu)\n",
                      ret, users[id], size);
            }
          free (buffer);

          /* Make sure that we did not have a file descriptor leak.  */
          for (size_t i = 0; i < array_length (next_descriptors); ++i)
            {
              int new_fd = dup (0);
              if (new_fd != next_descriptors[i])
                {
                  support_record_failure ();
                  printf ("error: descriptor %d at index %zu leaked"
                          " (user \"%s\", size %zu)\n",
                          next_descriptors[i], i, users[id], size);

                  /* Close unexpected descriptor, the leak probing
                     descriptors, and the leaked descriptor
                     next_descriptors[i].  */
                  xclose (new_fd);
                  for (size_t j = 0; j <= i; ++j)
                    xclose (next_descriptors[j]);
                  goto next_size;
                }
            }
          for (size_t i = 0; i < array_length (next_descriptors); ++i)
            xclose (next_descriptors[i]);

        next_size:
          ;
        }
      if (!found)
        {
          support_record_failure ();
          printf ("error: user %s not found\n", users[id]);
        }
    }

  support_chroot_free (chroot_env);
  return 0;
}

#define PREPARE prepare
#include <support/test-driver.c>
