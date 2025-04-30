/* Emulate Emacs heap dumping to test malloc_set_state.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Wolfram Gloger <wg@malloc.de>, 2001.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <libc-symbols.h>
#include <shlib-compat.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>

#include "malloc.h"

/* Make the compatibility symbols availabile to this test case.  */
void *malloc_get_state (void);
compat_symbol_reference (libc, malloc_get_state, malloc_get_state, GLIBC_2_0);
int malloc_set_state (void *);
compat_symbol_reference (libc, malloc_set_state, malloc_set_state, GLIBC_2_0);

/* Maximum object size in the fake heap.  */
enum { max_size = 64 };

/* Allocation actions.  These are randomized actions executed on the
   dumped heap (see allocation_tasks below).  They are interspersed
   with operations on the new heap (see heap_activity).  */
enum allocation_action
  {
    action_free,                /* Dumped and freed.  */
    action_realloc,             /* Dumped and realloc'ed.  */
    action_realloc_same,        /* Dumped and realloc'ed, same size.  */
    action_realloc_smaller,     /* Dumped and realloc'ed, shrinked.  */
    action_count
  };

/* Dumped heap.  Initialize it, so that the object is placed into the
   .data section, for increased realism.  The size is an upper bound;
   we use about half of the space.  */
static size_t dumped_heap[action_count * max_size * max_size
                          / sizeof (size_t)] = {1};

/* Next free space in the dumped heap.  Also top of the heap at the
   end of the initialization procedure.  */
static size_t *next_heap_chunk;

/* Copied from malloc.c and hooks.c.  The version is deliberately
   lower than the final version of malloc_set_state.  */
# define NBINS 128
# define MALLOC_STATE_MAGIC   0x444c4541l
# define MALLOC_STATE_VERSION (0 * 0x100l + 4l)
static struct
{
  long magic;
  long version;
  void *av[NBINS * 2 + 2];
  char *sbrk_base;
  int sbrked_mem_bytes;
  unsigned long trim_threshold;
  unsigned long top_pad;
  unsigned int n_mmaps_max;
  unsigned long mmap_threshold;
  int check_action;
  unsigned long max_sbrked_mem;
  unsigned long max_total_mem;
  unsigned int n_mmaps;
  unsigned int max_n_mmaps;
  unsigned long mmapped_mem;
  unsigned long max_mmapped_mem;
  int using_malloc_checking;
  unsigned long max_fast;
  unsigned long arena_test;
  unsigned long arena_max;
  unsigned long narenas;
} save_state =
  {
    .magic = MALLOC_STATE_MAGIC,
    .version = MALLOC_STATE_VERSION,
  };

/* Allocate a blob in the fake heap.  */
static void *
dumped_heap_alloc (size_t length)
{
  /* malloc needs three state bits in the size field, so the minimum
     alignment is 8 even on 32-bit architectures.  malloc_set_state
     should be compatible with such heaps even if it currently
     provides more alignment to applications.  */
  enum
  {
    heap_alignment = 8,
    heap_alignment_mask = heap_alignment - 1
  };
  _Static_assert (sizeof (size_t) <= heap_alignment,
                  "size_t compatible with heap alignment");

  /* Need at least this many bytes for metadata and application
     data. */
  size_t chunk_size = sizeof (size_t) + length;
  /* Round up the allocation size to the heap alignment.  */
  chunk_size += heap_alignment_mask;
  chunk_size &= ~heap_alignment_mask;
  TEST_VERIFY_EXIT ((chunk_size & 3) == 0);
  if (next_heap_chunk == NULL)
    /* Initialize the top of the heap.  Add one word of zero padding,
       to match existing practice.  */
    {
      dumped_heap[0] = 0;
      next_heap_chunk = dumped_heap + 1;
    }
  else
    /* The previous chunk is allocated. */
    chunk_size |= 1;
  *next_heap_chunk = chunk_size;

  /* User data starts after the chunk header.  */
  void *result = next_heap_chunk + 1;
  next_heap_chunk += chunk_size / sizeof (size_t);

  /* Mark the previous chunk as used.   */
  *next_heap_chunk = 1;
  return result;
}

/* Global seed variable for the random number generator.  */
static unsigned long long global_seed;

/* Simple random number generator.  The numbers are in the range from
   0 to UINT_MAX (inclusive).  */
static unsigned int
rand_next (unsigned long long *seed)
{
  /* Linear congruential generated as used for MMIX.  */
  *seed = *seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return *seed >> 32;
}

/* Fill LENGTH bytes at BUFFER with random contents, as determined by
   SEED.  */
static void
randomize_buffer (unsigned char *buffer, size_t length,
                  unsigned long long seed)
{
  for (size_t i = 0; i < length; ++i)
    buffer[i] = rand_next (&seed);
}

/* Dumps the buffer to standard output,  in hexadecimal.  */
static void
dump_hex (unsigned char *buffer, size_t length)
{
  for (int i = 0; i < length; ++i)
    printf (" %02X", buffer[i]);
}

/* Set to true if an error is encountered.  */
static bool errors = false;

/* Keep track of object allocations.  */
struct allocation
{
  unsigned char *data;
  unsigned int size;
  unsigned int seed;
};

/* Check that the allocation task allocation has the expected
   contents.  */
static void
check_allocation (const struct allocation *alloc, int index)
{
  size_t size = alloc->size;
  if (alloc->data == NULL)
    {
      printf ("error: NULL pointer for allocation of size %zu at %d, seed %u\n",
              size, index, alloc->seed);
      errors = true;
      return;
    }

  unsigned char expected[4096];
  if (size > sizeof (expected))
    {
      printf ("error: invalid allocation size %zu at %d, seed %u\n",
              size, index, alloc->seed);
      errors = true;
      return;
    }
  randomize_buffer (expected, size, alloc->seed);
  if (memcmp (alloc->data, expected, size) != 0)
    {
      printf ("error: allocation %d data mismatch, size %zu, seed %u\n",
              index, size, alloc->seed);
      printf ("  expected:");
      dump_hex (expected, size);
      putc ('\n', stdout);
      printf ("    actual:");
      dump_hex (alloc->data, size);
      putc ('\n', stdout);
      errors = true;
    }
}

/* A heap allocation combined with pending actions on it.  */
struct allocation_task
{
  struct allocation allocation;
  enum allocation_action action;
};

/* Allocation tasks.  Initialized by init_allocation_tasks and used by
   perform_allocations.  */
enum { allocation_task_count = action_count * max_size };
static struct allocation_task allocation_tasks[allocation_task_count];

/* Fisher-Yates shuffle of allocation_tasks.  */
static void
shuffle_allocation_tasks (void)
{
  for (int i = 0; i < allocation_task_count - 1; ++i)
    {
      /* Pick pair in the tail of the array.  */
      int j = i + (rand_next (&global_seed)
                   % ((unsigned) (allocation_task_count - i)));
      TEST_VERIFY_EXIT (j >= 0 && j < allocation_task_count);
      /* Exchange. */
      struct allocation_task tmp = allocation_tasks[i];
      allocation_tasks[i] = allocation_tasks[j];
      allocation_tasks[j] = tmp;
    }
}

/* Set up the allocation tasks and the dumped heap.  */
static void
initial_allocations (void)
{
  /* Initialize in a position-dependent way.  */
  for (int i = 0; i < allocation_task_count; ++i)
    allocation_tasks[i] = (struct allocation_task)
      {
        .allocation =
          {
            .size = 1 + (i / action_count),
            .seed = i,
          },
        .action = i % action_count
      };

  /* Execute the tasks in a random order.  */
  shuffle_allocation_tasks ();

  /* Initialize the contents of the dumped heap.   */
  for (int i = 0; i < allocation_task_count; ++i)
    {
      struct allocation_task *task = allocation_tasks + i;
      task->allocation.data = dumped_heap_alloc (task->allocation.size);
      randomize_buffer (task->allocation.data, task->allocation.size,
                        task->allocation.seed);
    }

  for (int i = 0; i < allocation_task_count; ++i)
    check_allocation (&allocation_tasks[i].allocation, i);
}

/* Indicates whether init_heap has run.  This variable needs to be
   volatile because malloc is declared __THROW, which implies it is a
   leaf function, but we expect it to run our hooks.  */
static volatile bool heap_initialized;

/* Executed by glibc malloc, through __malloc_initialize_hook
   below.  */
static void
init_heap (void)
{
  if (test_verbose)
    printf ("info: performing heap initialization\n");
  heap_initialized = true;

  /* Populate the dumped heap.  */
  initial_allocations ();

  /* Complete initialization of the saved heap data structure.  */
  save_state.sbrk_base = (void *) dumped_heap;
  save_state.sbrked_mem_bytes = sizeof (dumped_heap);
  /* Top pointer.  Adjust so that it points to the start of struct
     malloc_chunk.  */
  save_state.av[2] = (void *) (next_heap_chunk - 1);

  /* Integrate the dumped heap into the process heap.  */
  TEST_VERIFY_EXIT (malloc_set_state (&save_state) == 0);
}

/* Interpose the initialization callback.  */
void (*volatile __malloc_initialize_hook) (void) = init_heap;
compat_symbol_reference (libc, __malloc_initialize_hook,
                         __malloc_initialize_hook, GLIBC_2_0);

/* Simulate occasional unrelated heap activity in the non-dumped
   heap.  */
enum { heap_activity_allocations_count = 32 };
static struct allocation heap_activity_allocations
  [heap_activity_allocations_count] = {};
static int heap_activity_seed_counter = 1000 * 1000;

static void
heap_activity (void)
{
  /* Only do this from time to time.  */
  if ((rand_next (&global_seed) % 4) == 0)
    {
      int slot = rand_next (&global_seed) % heap_activity_allocations_count;
      struct allocation *alloc = heap_activity_allocations + slot;
      if (alloc->data == NULL)
        {
          alloc->size = rand_next (&global_seed) % (4096U + 1);
          alloc->data = xmalloc (alloc->size);
          alloc->seed = heap_activity_seed_counter++;
          randomize_buffer (alloc->data, alloc->size, alloc->seed);
          check_allocation (alloc, 1000 + slot);
        }
      else
        {
          check_allocation (alloc, 1000 + slot);
          free (alloc->data);
          alloc->data = NULL;
        }
    }
}

static void
heap_activity_deallocate (void)
{
  for (int i = 0; i < heap_activity_allocations_count; ++i)
    free (heap_activity_allocations[i].data);
}

/* Perform a full heap check across the dumped heap allocation tasks,
   and the simulated heap activity directly above.  */
static void
full_heap_check (void)
{
  /* Dumped heap.  */
  for (int i = 0; i < allocation_task_count; ++i)
    if (allocation_tasks[i].allocation.data != NULL)
      check_allocation (&allocation_tasks[i].allocation, i);

  /* Heap activity allocations.  */
  for (int i = 0; i < heap_activity_allocations_count; ++i)
    if (heap_activity_allocations[i].data != NULL)
      check_allocation (heap_activity_allocations + i, i);
}

/* Used as an optimization barrier to force a heap allocation.  */
__attribute__ ((noinline, noclone))
static void
my_free (void *ptr)
{
  free (ptr);
}

static int
do_test (void)
{
  my_free (malloc (1));
  TEST_VERIFY_EXIT (heap_initialized);

  /* The first pass performs the randomly generated allocation
     tasks.  */
  if (test_verbose)
    printf ("info: first pass through allocation tasks\n");
  full_heap_check ();

  /* Execute the post-undump tasks in a random order.  */
  shuffle_allocation_tasks ();

  for (int i = 0; i < allocation_task_count; ++i)
    {
      heap_activity ();
      struct allocation_task *task = allocation_tasks + i;
      switch (task->action)
        {
        case action_free:
          check_allocation (&task->allocation, i);
          free (task->allocation.data);
          task->allocation.data = NULL;
          break;

        case action_realloc:
          check_allocation (&task->allocation, i);
          task->allocation.data = xrealloc
            (task->allocation.data, task->allocation.size + max_size);
          check_allocation (&task->allocation, i);
          break;

        case action_realloc_same:
          check_allocation (&task->allocation, i);
          task->allocation.data = xrealloc
            (task->allocation.data, task->allocation.size);
          check_allocation (&task->allocation, i);
          break;

        case action_realloc_smaller:
          check_allocation (&task->allocation, i);
          size_t new_size = task->allocation.size - 1;
          task->allocation.data = xrealloc (task->allocation.data, new_size);
          if (new_size == 0)
            {
              if (task->allocation.data != NULL)
                {
                  printf ("error: realloc with size zero did not deallocate\n");
                  errors = true;
                }
              /* No further action on this task.  */
              task->action = action_free;
            }
          else
            {
              task->allocation.size = new_size;
              check_allocation (&task->allocation, i);
            }
          break;

        case action_count:
          FAIL_EXIT1 ("task->action should never be action_count");
        }
      full_heap_check ();
    }

  /* The second pass frees the objects which were allocated during the
     first pass.  */
  if (test_verbose)
    printf ("info: second pass through allocation tasks\n");

  shuffle_allocation_tasks ();
  for (int i = 0; i < allocation_task_count; ++i)
    {
      heap_activity ();
      struct allocation_task *task = allocation_tasks + i;
      switch (task->action)
        {
        case action_free:
          /* Already freed, nothing to do.  */
          break;

        case action_realloc:
        case action_realloc_same:
        case action_realloc_smaller:
          check_allocation (&task->allocation, i);
          free (task->allocation.data);
          task->allocation.data = NULL;
          break;

        case action_count:
          FAIL_EXIT1 ("task->action should never be action_count");
        }
      full_heap_check ();
    }

  heap_activity_deallocate ();

  /* Check that the malloc_get_state stub behaves in the intended
     way.  */
  errno = 0;
  if (malloc_get_state () != NULL)
    {
      printf ("error: malloc_get_state succeeded\n");
      errors = true;
    }
  if (errno != ENOSYS)
    {
      printf ("error: malloc_get_state: %m\n");
      errors = true;
    }

  return errors;
}

#include <support/test-driver.c>
