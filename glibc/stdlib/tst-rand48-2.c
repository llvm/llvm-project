#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int
do_test (void)
{
  time_t t = time (NULL);
  int i, ret = 0;
  double d;
  long int l;
  struct drand48_data data;
  unsigned short int buf[3];

  srand48 ((long int) t);
  for (i = 0; i < 50; i++)
    if ((d = drand48 ()) < 0.0 || d >= 1.0)
      {
        printf ("drand48 %d %g\n", i, d);
        ret = 1;
      }

  srand48_r ((long int) t, &data);
  for (i = 0; i < 50; i++)
    if (drand48_r (&data, &d) != 0 || d < 0.0 || d >= 1.0)
      {
        printf ("drand48_r %d %g\n", i, d);
        ret = 1;
      }

  buf[2] = (t & 0xffff0000) >> 16; buf[1] = (t & 0xffff); buf[0] = 0x330e;
  for (i = 0; i < 50; i++)
    if ((d = erand48 (buf)) < 0.0 || d >= 1.0)
      {
        printf ("erand48 %d %g\n", i, d);
        ret = 1;
      }

  buf[2] = (t & 0xffff0000) >> 16; buf[1] = (t & 0xffff); buf[0] = 0x330e;
  for (i = 0; i < 50; i++)
    if (erand48_r (buf, &data, &d) != 0 || d < 0.0 || d >= 1.0)
      {
        printf ("erand48_r %d %g\n", i, d);
        ret = 1;
      }

  srand48 ((long int) t);
  for (i = 0; i < 50; i++)
    if ((l = lrand48 ()) < 0 || l > INT32_MAX)
      {
        printf ("lrand48 %d %ld\n", i, l);
        ret = 1;
      }

  srand48_r ((long int) t, &data);
  for (i = 0; i < 50; i++)
    if (lrand48_r (&data, &l) != 0 || l < 0 || l > INT32_MAX)
      {
        printf ("lrand48_r %d %ld\n", i, l);
        ret = 1;
      }

  buf[2] = (t & 0xffff0000) >> 16; buf[1] = (t & 0xffff); buf[0] = 0x330e;
  for (i = 0; i < 50; i++)
    if ((l = nrand48 (buf)) < 0 || l > INT32_MAX)
      {
        printf ("nrand48 %d %ld\n", i, l);
        ret = 1;
      }

  buf[2] = (t & 0xffff0000) >> 16; buf[1] = (t & 0xffff); buf[0] = 0x330e;
  for (i = 0; i < 50; i++)
    if (nrand48_r (buf, &data, &l) != 0 || l < 0 || l > INT32_MAX)
      {
        printf ("nrand48_r %d %ld\n", i, l);
        ret = 1;
      }

  srand48 ((long int) t);
  for (i = 0; i < 50; i++)
    if ((l = mrand48 ()) < INT32_MIN || l > INT32_MAX)
      {
        printf ("mrand48 %d %ld\n", i, l);
        ret = 1;
      }

  srand48_r ((long int) t, &data);
  for (i = 0; i < 50; i++)
    if (mrand48_r (&data, &l) != 0 || l < INT32_MIN || l > INT32_MAX)
      {
        printf ("mrand48_r %d %ld\n", i, l);
        ret = 1;
      }

  buf[2] = (t & 0xffff0000) >> 16; buf[1] = (t & 0xffff); buf[0] = 0x330e;
  for (i = 0; i < 50; i++)
    if ((l = jrand48 (buf)) < INT32_MIN || l > INT32_MAX)
      {
        printf ("jrand48 %d %ld\n", i, l);
        ret = 1;
      }

  buf[2] = (t & 0xffff0000) >> 16; buf[1] = (t & 0xffff); buf[0] = 0x330e;
  for (i = 0; i < 50; i++)
    if (jrand48_r (buf, &data, &l) != 0 || l < INT32_MIN || l > INT32_MAX)
      {
        printf ("jrand48_r %d %ld\n", i, l);
        ret = 1;
      }

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
