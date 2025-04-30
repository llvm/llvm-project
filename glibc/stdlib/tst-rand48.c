#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DECIMAL_DIG
# define DECIMAL_DIG	21
#endif


static int
do_test (void)
{
  unsigned short int xs[3] = { 0x0001, 0x0012, 0x0123 };
  unsigned short int lxs[7];
  unsigned short int *xsp;
  int result = 0;
  long int l;
  double d;
  double e;

  /* Test srand48.  */
  srand48 (0x98765432);
  /* Get the values of the internal Xi array.  */
  xsp = seed48 (xs);
  if (xsp[0] != 0x330e || xsp[1] != 0x5432 || xsp[2] != 0x9876)
    {
      puts ("srand48(0x98765432) didn't set correct value");
      printf ("  expected: { %04hx, %04hx, %04hx }\n", 0x330e, 0x5432, 0x9876);
      printf ("  seen:     { %04hx, %04hx, %04hx }\n", xsp[0], xsp[1], xsp[2]);
      result = 1;
    }
  /* Put the values back.  */
  memcpy (xs, xsp, sizeof (xs));
  (void) seed48 (xs);

  /* See whether the correct values are installed.  */
  l = lrand48 ();
  if (l != 0x2fed1413l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x2fed1413l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != -0x5d73effdl)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, -0x5d73effdl, l);
      result = 1;
    }

  l = lrand48 ();
  if (l != 0x585fcfb7l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x585fcfb7l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != -0x61770b8cl)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, -0x61770b8cl, l);
      result = 1;
    }

  /* Test seed48.  The previous call should have install the values in
     the initialization of `xs' above.  */
  xs[0] = 0x1234;
  xs[1] = 0x5678;
  xs[2] = 0x9012;
  xsp = seed48 (xs);
  if (xsp[0] != 0x62f2 || xsp[1] != 0xf474 || xsp[2] != 0x9e88)
    {
      puts ("seed48() did not install the values correctly");
      printf ("  expected: { %04hx, %04hx, %04hx }\n", 0x62f2, 0xf474, 0x9e88);
      printf ("  seen:     { %04hx, %04hx, %04hx }\n", xsp[0], xsp[1], xsp[2]);
      result = 1;
    }

  /* Test lrand48 and mrand48.  We continue from the seed established
     above.  */
  l = lrand48 ();
  if (l != 0x017e48b5l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x017e48b5l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != -0x1485e05dl)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, -0x1485e05dl, l);
      result = 1;
    }

  l = lrand48 ();
  if (l != 0x6b6a3f95l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x6b6a3f95l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != 0x175c0d6fl)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x175c0d6fl, l);
      result = 1;
    }

  /* Test lcong48.  */
  lxs[0] = 0x4567;
  lxs[1] = 0x6789;
  lxs[2] = 0x8901;
  lxs[3] = 0x0123;
  lxs[4] = 0x2345;
  lxs[5] = 0x1111;
  lxs[6] = 0x2222;
  lcong48 (lxs);

  /* See whether the correct values are installed.  */
  l = lrand48 ();
  if (l != 0x6df63d66l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x6df63d66l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != 0x2f92c8e1l)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x2f92c8e1l, l);
      result = 1;
    }

  l = lrand48 ();
  if (l != 0x3b4869ffl)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x3b4869ffl, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != 0x5cd4cc3el)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x5cd4cc3el, l);
      result = 1;
    }

  /* Check whether srand48() restores the A and C parameters.  */
  srand48 (0x98765432);

  /* See whether the correct values are installed.  */
  l = lrand48 ();
  if (l != 0x2fed1413l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x2fed1413l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != -0x5d73effdl)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, -0x5d73effdl, l);
      result = 1;
    }

  l = lrand48 ();
  if (l != 0x585fcfb7l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x585fcfb7l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != -0x61770b8cl)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, -0x61770b8cl, l);
      result = 1;
    }

  /* And again to see whether seed48() does the same.  */
  lcong48 (lxs);

  /* See whether lxs wasn't modified.  */
  l = lrand48 ();
  if (l != 0x6df63d66l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x6df63d66l, l);
      result = 1;
    }

  /* Test seed48.  The previous call should have install the values in
     the initialization of `xs' above.  */
  xs[0] = 0x1234;
  xs[1] = 0x5678;
  xs[2] = 0x9012;
  xsp = seed48 (xs);
  if (xsp[0] != 0x0637 || xsp[1] != 0x7acd || xsp[2] != 0xdbec)
    {
      puts ("seed48() did not install the values correctly");
      printf ("  expected: { %04hx, %04hx, %04hx }\n", 0x0637, 0x7acd, 0xdbec);
      printf ("  seen:     { %04hx, %04hx, %04hx }\n", xsp[0], xsp[1], xsp[2]);
      result = 1;
    }

  /* Test lrand48 and mrand48.  We continue from the seed established
     above.  */
  l = lrand48 ();
  if (l != 0x017e48b5l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x017e48b5l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != -0x1485e05dl)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, -0x1485e05dl, l);
      result = 1;
    }

  l = lrand48 ();
  if (l != 0x6b6a3f95l)
    {
      printf ("lrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x6b6a3f95l, l);
      result = 1;
    }

  l = mrand48 ();
  if (l != 0x175c0d6fl)
    {
      printf ("mrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x175c0d6fl, l);
      result = 1;
    }

  /* Test drand48.  */
  d = drand48 ();
  if (d != 0.0908832261858485424)
    {
      printf ("drand48() in line %d failed: expected %.*g, seen %.*g\n",
	      __LINE__ - 4, DECIMAL_DIG, 0.0908832261858485424,
	      DECIMAL_DIG, d);
      result = 1;
    }

  d = drand48 ();
  if (d != 0.943149381730059133133)
    {
      printf ("drand48() in line %d failed: expected %.*g, seen %.*g\n",
	      __LINE__ - 4, DECIMAL_DIG, 0.943149381730059133133,
	      DECIMAL_DIG, d);
      result = 1;
    }

  /* Now the functions which get the Xis passed.  */
  xs[0] = 0x3849;
  xs[1] = 0x5061;
  xs[2] = 0x7283;

  l = nrand48 (xs);
  if (l != 0x1efe61a1l)
    {
      printf ("nrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x1efe61a1l, l);
      result = 1;
    }

  l = jrand48 (xs);
  if (l != -0xa973860l)
    {
      printf ("jrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, -0xa973860l, l);
      result = 1;
    }

  l = nrand48 (xs);
  if (l != 0x2a5e57fel)
    {
      printf ("nrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x2a5e57fel, l);
      result = 1;
    }

  l = jrand48 (xs);
  if (l != 0x71a779a8l)
    {
      printf ("jrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x71a779a8l, l);
      result = 1;
    }

  /* Test whether the global A and C are used.  */
  lcong48 (lxs);

  l = nrand48 (xs);
  if (l != 0x32beee9fl)
    {
      printf ("nrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x32beee9fl, l);
      result = 1;
    }

  l = jrand48 (xs);
  if (l != 0x7bddf3bal)
    {
      printf ("jrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x7bddf3bal, l);
      result = 1;
    }

  l = nrand48 (xs);
  if (l != 0x85bdf28l)
    {
      printf ("nrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x85bdf28l, l);
      result = 1;
    }

  l = jrand48 (xs);
  if (l != 0x7b433e47l)
    {
      printf ("jrand48() in line %d failed: expected %lx, seen %lx\n",
	      __LINE__ - 4, 0x7b433e47l, l);
      result = 1;
    }

  /* Test erand48.  Also compare with the drand48 results.  */
  (void) seed48 (xs);

  d = drand48 ();
  e = erand48 (xs);
  if (d != e)
    {
      printf ("\
drand48() and erand48 in lines %d and %d produce different results\n",
	      __LINE__ - 6, __LINE__ - 5);
      printf ("  drand48() = %g, erand48() = %g\n", d, e);
      result = 1;
    }
  else if (e != 0.640650904452755298735)
    {
      printf ("erand48() in line %d failed: expected %.*g, seen %.*g\n",
	      __LINE__ - 4, DECIMAL_DIG, 0.640650904452755298735,
	      DECIMAL_DIG, e);
      result = 1;

    }

  d = drand48 ();
  e = erand48 (xs);
  if (d != e)
    {
      printf ("\
drand48() and erand48 in lines %d and %d produce different results\n",
	      __LINE__ - 6, __LINE__ - 5);
      printf ("  drand48() = %g, erand48() = %g\n", d, e);
      result = 1;
    }
  else if (e != 0.115372323508150742555)
    {
      printf ("erand48() in line %d failed: expected %.*g, seen %.*g\n",
	      __LINE__ - 4, DECIMAL_DIG, 0.0115372323508150742555,
	      DECIMAL_DIG, e);
      result = 1;

    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
