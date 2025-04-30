/* Test autofs "ignore" filtering for getment_r.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <array_length.h>
#include <errno.h>
#include <mntent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/temp_file.h>
#include <support/xstdio.h>
#include <support/xunistd.h>

struct test_case
{
  const char *line;
  struct
  {
    /* Like struct mntent, but with const pointers.  */
    const char *mnt_fsname;
    const char *mnt_dir;
    const char *mnt_type;
    const char *mnt_opts;
    int mnt_freq;
    int mnt_passno;
  } expected;
};

static struct test_case test_cases[] =
  {
    { "/etc/auto.direct /mnt/auto/1 autofs defaults 0 0",
      { "/etc/auto.direct", "/mnt/auto/1", "autofs", "defaults", 0, 0 } },

    /* These entries are filtered out.  */
    { "/etc/auto.2 /mnt/auto/2 autofs ignore 0 0", { NULL, } },
    { "/etc/auto.3 /mnt/auto/3 autofs ignore,other 1 2", { NULL, } },
    { "/etc/auto.4 /mnt/auto/4 autofs other,ignore 3 4", { NULL, } },
    { "/etc/auto.5 /mnt/auto/5 autofs opt1,ignore,opt2 5 6", { NULL, } },

    /* Dummy entry to make the desynchronization more obvious.  */
    { "/dev/sda1 / xfs defaults 0 0",
      { "/dev/sda1", "/", "xfs", "defaults", 0, 0 } },

    /* These are not filtered because the file system is not autofs.  */
    { "/etc/auto.direct /mnt/auto/6 autofs1 ignore 0 0",
      { "/etc/auto.direct", "/mnt/auto/6", "autofs1", "ignore", 0, 0 } },
    { "/etc/auto.direct /mnt/auto/7 autofs1 ignore,other 0 0",
      { "/etc/auto.direct", "/mnt/auto/7", "autofs1", "ignore,other", 0, 0 } },
    { "/etc/auto.direct /mnt/auto/8 autofs1 other,ignore 0 0",
      { "/etc/auto.direct", "/mnt/auto/8", "autofs1", "other,ignore", 0, 0 } },
    { "/etc/auto.direct /mnt/auto/9 autofs1 opt1,ignore,opt2 0 0",
      { "/etc/auto.direct", "/mnt/auto/9", "autofs1", "opt1,ignore,opt2", } },

    /* These are not filtered because the string "ignore" is not an
       option name.  */
    { "/etc/auto.direct /mnt/auto/10 autofs noignore 1 2",
      { "/etc/auto.direct", "/mnt/auto/10", "autofs", "noignore", 1, 2 } },
    { "/etc/auto.direct /mnt/auto/11 autofs noignore,other 0 0",
      { "/etc/auto.direct", "/mnt/auto/11", "autofs", "noignore,other", } },
    { "/etc/auto.direct /mnt/auto/12 autofs other,noignore 0 0",
      { "/etc/auto.direct", "/mnt/auto/12", "autofs", "other,noignore", } },
    { "/etc/auto.direct /mnt/auto/13 autofs errors=ignore 0 0",
      { "/etc/auto.direct", "/mnt/auto/13", "autofs", "errors=ignore", } },
    { "/etc/auto.direct /mnt/auto/14 autofs errors=ignore,other 0 0",
      { "/etc/auto.direct", "/mnt/auto/14", "autofs",
        "errors=ignore,other", } },
    { "/etc/auto.direct /mnt/auto/15 autofs other,errors=ignore 0 0",
      { "/etc/auto.direct", "/mnt/auto/15", "autofs",
        "other,errors=ignore", } },

    /* These are not filtered because the string is escaped.  '\151'
       is 'i', but it is not actually decoded by the parser.  */
    { "/etc/auto.\\151gnore /mnt/auto/16 autofs \\151gnore 0 0",
      { "/etc/auto.\\151gnore", "/mnt/auto/16", "autofs",
        "\\151gnore", } },
  };

static int
do_test (void)
{
  char *path;
  xclose (create_temp_file ("tst-mntent-autofs-", &path));

  /* Write the test file.  */
  FILE *fp = xfopen (path, "w");
  for (size_t i = 0; i < array_length (test_cases); ++i)
    fprintf (fp, "%s\n", test_cases[i].line);
  xfclose (fp);

  /* Open the test file again, this time for parsing.  */
  fp = setmntent (path, "r");
  TEST_VERIFY_EXIT (fp != NULL);
  char buffer[512];
  struct mntent me;

  for (size_t i = 0; i < array_length (test_cases); ++i)
    {
      if (test_cases[i].expected.mnt_type == NULL)
        continue;

      memset (buffer, 0xcc, sizeof (buffer));
      memset (&me, 0xcc, sizeof (me));
      struct mntent *pme = getmntent_r (fp, &me, buffer, sizeof (buffer));
      TEST_VERIFY_EXIT (pme != NULL);
      TEST_VERIFY (pme == &me);
      TEST_COMPARE_STRING (test_cases[i].expected.mnt_fsname, me.mnt_fsname);
      TEST_COMPARE_STRING (test_cases[i].expected.mnt_dir, me.mnt_dir);
      TEST_COMPARE_STRING (test_cases[i].expected.mnt_type, me.mnt_type);
      TEST_COMPARE_STRING (test_cases[i].expected.mnt_opts, me.mnt_opts);
      TEST_COMPARE (test_cases[i].expected.mnt_freq, me.mnt_freq);
      TEST_COMPARE (test_cases[i].expected.mnt_passno, me.mnt_passno);
    }

  TEST_VERIFY (getmntent_r (fp, &me, buffer, sizeof (buffer)) == NULL);

  TEST_COMPARE (feof (fp), 1);
  TEST_COMPARE (ferror (fp), 0);
  errno = 0;
  TEST_COMPARE (endmntent (fp), 1);
  TEST_COMPARE (errno, 0);
  free (path);
  return 0;
}

#include <support/test-driver.c>
