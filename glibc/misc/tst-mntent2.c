#include <mntent.h>
#include <stdio.h>
#include <string.h>


static int
do_test (void)
{
  int result = 0;
  struct mntent mef;

  mef.mnt_fsname = strdupa ("/dev/sdf6");
  mef.mnt_dir = strdupa ("/some dir");
  mef.mnt_type = strdupa ("ext3");
  mef.mnt_opts = strdupa ("opt1,opt2,noopt=6,rw,norw,brw");
  mef.mnt_freq = 1;
  mef.mnt_passno = 2;

#define TEST(opt, found) \
  if ((!!hasmntopt (&mef, (opt))) != (found))				\
    {									\
      printf ("Option %s was %sfound\n", (opt), (found) ? "not " : "");	\
      result = 1;							\
    }

  TEST ("opt1", 1)
  TEST ("opt2", 1)
  TEST ("noopt", 1)
  TEST ("rw", 1)
  TEST ("norw", 1)
  TEST ("brw", 1)
  TEST ("opt", 0)
  TEST ("oopt", 0)
  TEST ("w", 0)
  TEST ("r", 0)
  TEST ("br", 0)
  TEST ("nor", 0)
  TEST ("or", 0)

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
