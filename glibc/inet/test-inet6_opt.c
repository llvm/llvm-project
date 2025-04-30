#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define OPT_X	42
#define OPT_Y	43
#define OPT_Z	44

static void *
encode_inet6_opt (socklen_t *elp)
{
  void *eb = NULL;
  socklen_t el;
  int cl;
  void *db;
  int offset;
  uint8_t val1;
  uint16_t val2;
  uint32_t val4;
  uint64_t val8;

  *elp = 0;
#define CHECK() \
  if (cl == -1)						\
    {							\
      printf ("cl == -1 on line %d\n", __LINE__);	\
      free (eb);					\
      return NULL;					\
    }

  /* Estimate the length */
  cl = inet6_opt_init (NULL, 0);
  CHECK ();
  cl = inet6_opt_append (NULL, 0, cl, OPT_X, 12, 8, NULL);
  CHECK ();
  cl = inet6_opt_append (NULL, 0, cl, OPT_Y, 7, 4, NULL);
  CHECK ();
  cl = inet6_opt_append (NULL, 0, cl, OPT_Z, 7, 1, NULL);
  CHECK ();
  cl = inet6_opt_finish (NULL, 0, cl);
  CHECK ();
  el = cl;

  eb = malloc (el + 8);
  if (eb == NULL)
    {
      puts ("malloc failed");
      return NULL;
    }
  /* Canary.  */
  memcpy (eb + el, "deadbeef", 8);

  cl = inet6_opt_init (eb, el);
  CHECK ();

  cl = inet6_opt_append (eb, el, cl, OPT_X, 12, 8, &db);
  CHECK ();
  val4 = 0x12345678;
  offset = inet6_opt_set_val (db, 0, &val4, sizeof  (val4));
  val8 = 0x0102030405060708LL;
  inet6_opt_set_val (db, offset, &val8, sizeof  (val8));

  cl = inet6_opt_append (eb, el, cl, OPT_Y, 7, 4, &db);
  CHECK ();
  val1 = 0x01;
  offset = inet6_opt_set_val (db, 0, &val1, sizeof  (val1));
  val2 = 0x1331;
  offset = inet6_opt_set_val (db, offset, &val2, sizeof  (val2));
  val4 = 0x01020304;
  inet6_opt_set_val (db, offset, &val4, sizeof  (val4));

  cl = inet6_opt_append (eb, el, cl, OPT_Z, 7, 1, &db);
  CHECK ();
  inet6_opt_set_val (db, 0, (void *) "abcdefg", 7);

  cl = inet6_opt_finish (eb, el, cl);
  CHECK ();

  if (memcmp (eb + el, "deadbeef", 8) != 0)
    {
      puts ("Canary corrupted");
      free (eb);
      return NULL;
    }
  *elp = el;
  return eb;
}

int
decode_inet6_opt (void *eb, socklen_t el)
{
  int ret = 0;
  int seq = 0;
  int cl = 0;
  int offset;
  uint8_t type;
  socklen_t len;
  uint8_t val1;
  uint16_t val2;
  uint32_t val4;
  uint64_t val8;
  void *db;
  char buf[8];

  while ((cl = inet6_opt_next (eb, el, cl, &type, &len, &db)) != -1)
    switch (type)
      {
      case OPT_X:
	if (seq++ != 0)
	  {
	    puts ("OPT_X is not first");
	    ret = 1;
	  }
	if (len != 12)
	  {
	    printf ("OPT_X's length %d != 12\n", len);
	    ret = 1;
	  }
	offset = inet6_opt_get_val (db, 0, &val4, sizeof (val4));
	if (val4 != 0x12345678)
	  {
	    printf ("OPT_X's val4 %x != 0x12345678\n", val4);
	    ret = 1;
	  }
	offset = inet6_opt_get_val (db, offset, &val8, sizeof (val8));
	if (offset != len || val8 != 0x0102030405060708LL)
	  {
	    printf ("OPT_X's val8 %llx != 0x0102030405060708\n",
		    (long long) val8);
	    ret = 1;
	  }
	break;
      case OPT_Y:
	if (seq++ != 1)
	  {
	    puts ("OPT_Y is not second");
	    ret = 1;
	  }
	if (len != 7)
	  {
	    printf ("OPT_Y's length %d != 7\n", len);
	    ret = 1;
	  }
	offset = inet6_opt_get_val (db, 0, &val1, sizeof (val1));
	if (val1 != 0x01)
	  {
	    printf ("OPT_Y's val1 %x != 0x01\n", val1);
	    ret = 1;
	  }
	offset = inet6_opt_get_val (db, offset, &val2, sizeof (val2));
	if (val2 != 0x1331)
	  {
	    printf ("OPT_Y's val2 %x != 0x1331\n", val2);
	    ret = 1;
	  }
	offset = inet6_opt_get_val (db, offset, &val4, sizeof (val4));
	if (offset != len || val4 != 0x01020304)
	  {
	    printf ("OPT_Y's val4 %x != 0x01020304\n", val4);
	    ret = 1;
	  }
	break;
      case OPT_Z:
	if (seq++ != 2)
	  {
	    puts ("OPT_Z is not third");
	    ret = 1;
	  }
	if (len != 7)
	  {
	    printf ("OPT_Z's length %d != 7\n", len);
	    ret = 1;
	  }
	offset = inet6_opt_get_val (db, 0, buf, 7);
	if (offset != len || memcmp (buf, "abcdefg", 7) != 0)
	  {
	    buf[7] = '\0';
	    printf ("OPT_Z's buf \"%s\" != \"abcdefg\"\n", buf);
	    ret = 1;
	  }
	break;
      default:
	printf ("Unknown option %d\n", type);
	ret = 1;
	break;
      }
  if (seq != 3)
    {
      puts ("Didn't see all of OPT_X, OPT_Y and OPT_Z");
      ret = 1;
    }
  return ret;
}

static int
do_test (void)
{
  void *eb;
  socklen_t el;
  eb = encode_inet6_opt (&el);
  if (eb == NULL)
    return 1;
  if (decode_inet6_opt (eb, el))
    return 1;
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
