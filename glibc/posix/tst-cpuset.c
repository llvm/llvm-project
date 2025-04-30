#include <sched.h>
#include <stdio.h>

static int
do_test (void)
{
  int result = 0;

  cpu_set_t s1;
  cpu_set_t s2;
  cpu_set_t s3;

  CPU_ZERO (&s1);
  CPU_SET (0, &s1);

  CPU_ZERO (&s2);
  CPU_SET (0, &s2);
  CPU_SET (1, &s2);

  CPU_AND (&s3, &s1, &s2);
  if (! CPU_EQUAL (&s3, &s1))
    {
      puts ("result of CPU_AND wrong");
      result = 1;
    }

  CPU_OR (&s3, &s1, &s2);
  if (! CPU_EQUAL (&s3, &s2))
    {
      puts ("result of CPU_OR wrong");
      result = 1;
    }

  CPU_XOR (&s3, &s1, &s2);
  if (CPU_COUNT (&s3) != 1)
    {
      puts ("result of CPU_XOR wrong");
      result = 1;
    }

  cpu_set_t *vs1 = CPU_ALLOC (2048);
  cpu_set_t *vs2 = CPU_ALLOC (2048);
  cpu_set_t *vs3 = CPU_ALLOC (2048);
  size_t vssize = CPU_ALLOC_SIZE (2048);

  CPU_ZERO_S (vssize, vs1);
  CPU_SET_S (0, vssize, vs1);

  CPU_ZERO_S (vssize, vs2);
  CPU_SET_S (0, vssize, vs2);
  CPU_SET_S (2047, vssize, vs2);

  CPU_AND_S (vssize, vs3, vs1, vs2);
  if (! CPU_EQUAL_S (vssize, vs3, vs1))
    {
      puts ("result of CPU_AND_S wrong");
      result = 1;
    }

  CPU_OR_S (vssize, vs3, vs1, vs2);
  if (! CPU_EQUAL_S (vssize, vs3, vs2))
    {
      puts ("result of CPU_OR_S wrong");
      result = 1;
    }

  CPU_XOR_S (vssize, vs3, vs1, vs2);
  if (CPU_COUNT_S (vssize, vs3) != 1)
    {
      puts ("result of CPU_XOR_S wrong");
      result = 1;
    }

  CPU_FREE (vs1);
  CPU_FREE (vs2);
  CPU_FREE (vs3);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
