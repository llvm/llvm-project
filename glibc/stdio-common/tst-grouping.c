/* BZ 12394, test by Bruno Haible.  */
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int
do_test (void)
{
  char buf1[1000];
  char buf2[1000];
  int result = 0;

  if (setlocale (LC_NUMERIC, "de_DE.UTF-8") == NULL)
    return 1;

  sprintf (buf1, "%'.2f",  999.996);
  sprintf (buf2, "%'.2f", 1000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 2;

  sprintf (buf1, "%'.2f",  999999.996);
  sprintf (buf2, "%'.2f", 1000000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 2;

  sprintf (buf1, "%'.2f",  999999999.996);
  sprintf (buf2, "%'.2f", 1000000000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 2;

  sprintf (buf1, "%'.2f",  999999999999.996);
  sprintf (buf2, "%'.2f", 1000000000000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 2;

  sprintf (buf1, "%'.2f",  999999999999999.996);
  sprintf (buf2, "%'.2f", 1000000000000000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 2;

  sprintf (buf1, "%'.5g",  999.996);
  sprintf (buf2, "%'.5g", 1000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 4;

  sprintf (buf1, "%'.4g",  9999.996);
  sprintf (buf2, "%'.4g", 10000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 8;

  sprintf (buf1, "%'.5g",  99999.996);
  sprintf (buf2, "%'.5g", 100000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 8;

  sprintf (buf1, "%'.6g",  999999.996);
  sprintf (buf2, "%'.6g", 1000000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 8;

  sprintf (buf1, "%'.7g",  9999999.996);
  sprintf (buf2, "%'.7g", 10000000.004);
  printf ("%d: \"%s\" vs \"%s\"\n", __LINE__, buf1, buf2);
  if (strcmp (buf1, buf2) != 0)
    result |= 8;

  return result;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
