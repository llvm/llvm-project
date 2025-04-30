#include <mcheck.h>
#include <obstack.h>
#include <stdio.h>
#include <stdlib.h>


static int
do_test (void)
{
  struct obstack ob;
  int n;

  mcheck_pedantic (NULL);

#define obstack_chunk_alloc malloc
#define obstack_chunk_free free

  obstack_init (&ob);

  for (n = 0; n < 40000; ++n)
    {
      mcheck_check_all ();
      obstack_printf (&ob, "%.*s%05d", 1 + n % 7, "foobarbaz", n);
      if (n % 777 == 0)
	obstack_finish (&ob);
    }

  /* Another loop where we finish all objects, each of size 1.  This will
     manage to call `obstack_print' with all possible positions inside
     an obstack chunk.  */
  for (n = 0; n < 40000; ++n)
    {
      mcheck_check_all ();
      obstack_printf (&ob, "%c", 'a' + n % 26);
      obstack_finish (&ob);
    }

  /* And a final check.  */
  mcheck_check_all ();

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
