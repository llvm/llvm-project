/* Test program for random(), srandom(), initstate(), setstate()
   Written by Michael J. Fischer, August 21, 2000
   Placed in the public domain.  */

/* This program primarily tests the correct functioning of srandom()
   and setstate().  The strategy is generate and store a set of random
   sequences, each with a specified starting seed.  Then each sequence
   is regenerated twice and checked against the stored values.

   First they are regenerated one sequence at a time, using srandom()
   to set the initial state.  A discrepency here would suggest that
   srandom() was failing to completely initialize the random number
   generator.

   Second the sequences are regenerated in an interleaved order.
   A state vector is created for each sequence using initstate().
   setstate() is used to switch from sequence to sequence during
   the interleaved generation.  A discrepency here would suggest
   a problem with either initstate() failing to initialize the
   random number generator properly, or the failure of setstate()
   to correctly save and restore state information.  Also, each
   time setstate() is called, the returned value is checked for
   correctness (since we know what it should be).

   Note:  We use default state vector for sequence 0 and our own
   state vectors for the remaining sequences.  This is to give a check
   that the value returned by initstate() is valid and can indeed be
   used in the future.  */

/* Strategy:
   1.  Use srandom() followed by calls on random to generate a set of
       sequences of values.
   2.  Regenerate and check the sequences.
   3.  Use initstate() to create new states.
   4.  Regenerate the sequences in an interleaved manner and check.
*/

#include <stdlib.h>
#include <stdio.h>

const int degree = 128;		/* random number generator degree (should
				   be one of 8, 16, 32, 64, 128, 256) */
const int nseq = 3;		/* number of test sequences */
const int nrnd = 50;		/* length of each test sequence */
const unsigned int seed[3] = { 0x12344321U, 0xEE11DD22U, 0xFEDCBA98 };

void fail (const char *msg, int s, int i) __attribute__ ((__noreturn__));

static int
do_test (void)
{
  long int rnd[nseq][nrnd];	/* pseudorandom numbers */
  char* state[nseq];		/* state for PRNG */
  char* oldstate[nseq];		/* old PRNG state */
  int s;			/* sequence index */
  int i;			/* element index */

  printf ("Begining random package test using %d sequences of length %d.\n",
	  nseq, nrnd);

  /* 1. Generate and store the sequences.  */
  printf ("Generating random sequences.\n");
  for (s = 0; s < nseq; ++s)
    {
      srandom ( seed[s] );
      for (i = 0; i < nrnd; ++i)
	rnd[s][i] = random ();
    }

  /* 2. Regenerate and check.  */
  printf ("Regenerating and checking sequences.\n");
  for (s = 0; s < nseq; ++s)
    {
      srandom (seed[s]);
      for (i = 0; i < nrnd; ++i)
	if (rnd[s][i] != random ())
	  fail ("first regenerate test", s, i);
    }

  /* 3. Create state vector, one for each sequence.
	First state is random's internal state; others are malloced.  */
  printf ("Creating and checking state vector for each sequence.\n");
  srandom (seed[0]);			/* reseed with first seed */
  for (s = 1; s < nseq; ++s)
    {
      state[s] = (char*) malloc (degree);
      oldstate[s] = initstate (seed[s], state[s], degree);
    }
  state[0] = oldstate[1];

  /* Check returned values.  */
  for (s = 1; s < nseq - 1; ++s)
    if (state[s] != oldstate[s + 1])
      fail ("bad initstate() return value", s, i);

  /* 4. Regenerate sequences interleaved and check.  */
  printf ("Regenerating and checking sequences in interleaved order.\n");
  for (i = 0; i < nrnd; ++i)
    {
      for (s = 0; s < nseq; ++s)
	{
	  char *oldstate = (char *) setstate (state[s]);
	  if (oldstate != state[(s + nseq - 1) % nseq])
	    fail ("bad setstate() return value", s, i);
	  if (rnd[s][i] != random ())
	    fail ("bad value generated in interleave test", s, i);
	}
    }
  printf ("All tests passed!\n");
  return 0;
}

void
fail (const char *msg, int s, int i)
{
  printf ("\nTest FAILED: ");
  printf ("%s (seq %d, pos %d).\n", msg, s, i);
  exit (1);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
