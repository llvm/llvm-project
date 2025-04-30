#include <crypt.h>
#include <stdio.h>
#include <string.h>

static const struct
{
  const char *salt;
  const char *input;
  const char *expected;
} tests[] =
{
  { "$5$saltstring", "Hello world!",
    "$5$saltstring$5B8vYYiY.CVt1RlTTf8KbXBH3hsxY/GNooZaBBGWEc5" },
  { "$5$rounds=10000$saltstringsaltstring", "Hello world!",
    "$5$rounds=10000$saltstringsaltst$3xv.VbSHBb41AL9AvLeujZkZRBAwqFMz2."
    "opqey6IcA" },
  { "$5$rounds=5000$toolongsaltstring", "This is just a test",
    "$5$rounds=5000$toolongsaltstrin$Un/5jzAHMgOGZ5.mWJpuVolil07guHPvOW8"
    "mGRcvxa5" },
  { "$5$rounds=1400$anotherlongsaltstring",
    "a very much longer text to encrypt.  This one even stretches over more"
    "than one line.",
    "$5$rounds=1400$anotherlongsalts$Rx.j8H.h8HjEDGomFU8bDkXm3XIUnzyxf12"
    "oP84Bnq1" },
  { "$5$rounds=77777$short",
    "we have a short salt string but not a short password",
    "$5$rounds=77777$short$JiO1O3ZpDAxGJeaDIuqCoEFysAe1mZNJRs3pw0KQRd/" },
  { "$5$rounds=123456$asaltof16chars..", "a short string",
    "$5$rounds=123456$asaltof16chars..$gP3VQ/6X7UUEW3HkBn2w1/Ptq2jxPyzV/"
    "cZKmF/wJvD" },
  { "$5$rounds=10$roundstoolow", "the minimum number is still observed",
    "$5$rounds=1000$roundstoolow$yfvwcWrQ8l/K0DAWyuPMDNHpIVlTQebY9l/gL97"
    "2bIC" },
};
#define ntests (sizeof (tests) / sizeof (tests[0]))



static int
do_test (void)
{
  int result = 0;
  int i;

  for (i = 0; i < ntests; ++i)
    {
      char *cp = crypt (tests[i].input, tests[i].salt);

      if (strcmp (cp, tests[i].expected) != 0)
	{
	  printf ("test %d: expected \"%s\", got \"%s\"\n",
		  i, tests[i].expected, cp);
	  result = 1;
	}
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
