/* Test case by Alexandre Duret-Lutz <duret_g@epita.fr>.  */
#include <obstack.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define obstack_chunk_alloc verbose_malloc
#define obstack_chunk_free verbose_free
#define ALIGN_BOUNDARY 64
#define ALIGN_MASK (ALIGN_BOUNDARY - 1)
#define OBJECT_SIZE 1000

static void *
verbose_malloc (size_t size)
{
  void *buf = malloc (size);
  printf ("malloc (%zu) => %p\n", size, buf);
  return buf;
}

static void
verbose_free (void *buf)
{
  free (buf);
  printf ("free (%p)\n", buf);
}

static int
do_test (void)
{
  int result = 0;
  int align = 2;

  while (align <= 64)
    {
      struct obstack obs;
      int i;
      int align_mask = align - 1;

      printf ("\n Alignment mask: %d\n", align_mask);

      obstack_init (&obs);
      obstack_alignment_mask (&obs) = align_mask;
      /* finish an empty object to take alignment into account */
      obstack_finish (&obs);

      /* let's allocate some objects and print their addresses */
      for (i = 15; i > 0; --i)
	{
	  void *obj = obstack_alloc (&obs, OBJECT_SIZE);

	  printf ("obstack_alloc (%u) => %p \t%s\n", OBJECT_SIZE, obj,
		  ((uintptr_t) obj & align_mask) ? "(not aligned)" : "");
	  result |= ((uintptr_t) obj & align_mask) != 0;
	}

      /* clean up */
      obstack_free (&obs, 0);

      align <<= 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
