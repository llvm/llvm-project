#include <stdio.h>
#include <sys/statvfs.h>


/* This test cannot detect many errors.  But it will fail if the
   statvfs is completely hosed and it'll detect a missing export.  So
   it is better than nothing.  */
static int
do_test (int argc, char *argv[])
{
  for (int i = 1; i < argc; ++i)
    {
      struct statvfs st;
      if (statvfs (argv[i], &st) != 0)
        printf ("%s: failed (%m)\n", argv[i]);
      else
        printf ("%s: free: %llu, mandatory: %s\n", argv[i],
                (unsigned long long int) st.f_bfree,
#ifdef ST_MANDLOCK
                (st.f_flag & ST_MANDLOCK) ? "yes" : "no"
#else
                "no"
#endif
                );
    }
  return 0;
}

#define TEST_FUNCTION do_test (argc, argv)
#include "../test-skeleton.c"
