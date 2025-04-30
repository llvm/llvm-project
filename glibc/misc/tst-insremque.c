#include <search.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond) \
  do									\
    if (! (cond))							\
      {									\
	printf ("Condition " #cond " not true on line %d\n", __LINE__);	\
	ret = 1;							\
      }									\
  while (0)

static int
do_test (void)
{
  struct qelem elements[4];
  int ret = 0;

  /* Linear list.  */
  memset (elements, 0xff, sizeof (elements));
  insque (&elements[0], NULL);
  remque (&elements[0]);
  insque (&elements[0], NULL);
  insque (&elements[2], &elements[0]);
  insque (&elements[1], &elements[0]);
  insque (&elements[3], &elements[2]);
  remque (&elements[2]);
  insque (&elements[2], &elements[0]);
  CHECK (elements[0].q_back == NULL);
  CHECK (elements[0].q_forw == &elements[2]);
  CHECK (elements[1].q_back == &elements[2]);
  CHECK (elements[1].q_forw == &elements[3]);
  CHECK (elements[2].q_back == &elements[0]);
  CHECK (elements[2].q_forw == &elements[1]);
  CHECK (elements[3].q_back == &elements[1]);
  CHECK (elements[3].q_forw == NULL);

  /* Circular list.  */
  memset (elements, 0xff, sizeof (elements));
  elements[0].q_back = &elements[0];
  elements[0].q_forw = &elements[0];
  insque (&elements[2], &elements[0]);
  insque (&elements[1], &elements[0]);
  insque (&elements[3], &elements[2]);
  remque (&elements[2]);
  insque (&elements[2], &elements[0]);
  CHECK (elements[0].q_back == &elements[3]);
  CHECK (elements[0].q_forw == &elements[2]);
  CHECK (elements[1].q_back == &elements[2]);
  CHECK (elements[1].q_forw == &elements[3]);
  CHECK (elements[2].q_back == &elements[0]);
  CHECK (elements[2].q_forw == &elements[1]);
  CHECK (elements[3].q_back == &elements[1]);
  CHECK (elements[3].q_forw == &elements[0]);

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
