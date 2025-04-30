#include <stdlib.h>
#include <string.h>

#include "hurdmalloc.h"		/* XXX see that file */

#include <mach.h>
#include <mach/spin-lock.h>
#define vm_allocate __vm_allocate
#define vm_page_size __vm_page_size

/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 * (pre-GNU) HISTORY
 *
 * Revision 2.7  91/05/14  17:57:34  mrt
 * 	Correcting copyright
 *
 * Revision 2.6  91/02/14  14:20:26  mrt
 * 	Added new Mach copyright
 * 	[91/02/13  12:41:21  mrt]
 *
 * Revision 2.5  90/11/05  14:37:33  rpd
 * 	Added malloc_fork* code.
 * 	[90/11/02            rwd]
 *
 * 	Add spin_lock_t.
 * 	[90/10/31            rwd]
 *
 * Revision 2.4  90/08/07  14:31:28  rpd
 * 	Removed RCS keyword nonsense.
 *
 * Revision 2.3  90/06/02  15:14:00  rpd
 * 	Converted to new IPC.
 * 	[90/03/20  20:56:57  rpd]
 *
 * Revision 2.2  89/12/08  19:53:59  rwd
 * 	Removed conditionals.
 * 	[89/10/23            rwd]
 *
 * Revision 2.1  89/08/03  17:09:46  rwd
 * Created.
 *
 *
 * 13-Sep-88  Eric Cooper (ecc) at Carnegie Mellon University
 *	Changed realloc() to copy min(old size, new size) bytes.
 *	Bug found by Mike Kupfer at Olivetti.
 */
/*
 * 	File: 	malloc.c
 *	Author: Eric Cooper, Carnegie Mellon University
 *	Date:	July, 1988
 *
 * 	Memory allocator for use with multiple threads.
 */


#include <assert.h>

#define MCHECK

/*
 * Structure of memory block header.
 * When free, next points to next block on free list.
 * When allocated, fl points to free list.
 * Size of header is 4 bytes, so minimum usable block size is 8 bytes.
 */

#define CHECK_BUSY  0x8a3c743e
#define CHECK_FREE  0x66688b92

#ifdef MCHECK

typedef struct header {
  long check;
  union {
    struct header *next;
    struct free_list *fl;
  } u;
} *header_t;

#define HEADER_SIZE sizeof (struct header)
#define HEADER_NEXT(h) ((h)->u.next)
#define HEADER_FREE(h) ((h)->u.fl)
#define HEADER_CHECK(h) ((h)->check)
#define MIN_SIZE	16
#define LOG2_MIN_SIZE	4

#else /* ! MCHECK */

typedef union header {
	union header *next;
	struct free_list *fl;
} *header_t;

#define HEADER_SIZE sizeof (union header)
#define HEADER_NEXT(h) ((h)->next)
#define HEADER_FREE(h) ((h)->fl)
#define MIN_SIZE	8	/* minimum block size */
#define LOG2_MIN_SIZE	3

#endif /* MCHECK */

typedef struct free_list {
	spin_lock_t lock;	/* spin lock for mutual exclusion */
	header_t head;		/* head of free list for this size */
#ifdef	DEBUG
	int in_use;		/* # mallocs - # frees */
#endif	/* DEBUG */
} *free_list_t;

/*
 * Free list with index i contains blocks of size 2 ^ (i + LOG2_MIN_SIZE)
 * including header.  Smallest block size is MIN_SIZE, with MIN_SIZE -
 * HEADER_SIZE bytes available to user.  Size argument to malloc is a signed
 * integer for sanity checking, so largest block size is 2^31.
 */
#define NBUCKETS	29

static struct free_list malloc_free_list[NBUCKETS];

/* Initialization just sets everything to zero, but might be necessary on a
   machine where spin_lock_init does otherwise, and is necessary when
   running an executable that was written by something like Emacs's unexec.
   It preserves the values of data variables like malloc_free_list, but
   does not save the vm_allocate'd space allocated by this malloc.  */

static void
malloc_init (void)
{
  int i;
  for (i = 0; i < NBUCKETS; ++i)
    {
      spin_lock_init (&malloc_free_list[i].lock);
      malloc_free_list[i].head = NULL;
#ifdef DEBUG
      malloc_free_list[i].in_use = 0;
#endif
    }

  /* This not only suppresses a `defined but not used' warning,
     but it is ABSOLUTELY NECESSARY to avoid the hyperclever
     compiler from "optimizing out" the entire function!  */
  (void) &malloc_init;
}

static void
more_memory(int size, free_list_t fl)
{
	int amount;
	int n;
	vm_address_t where;
	header_t h;
	kern_return_t r;

	if (size <= vm_page_size) {
		amount = vm_page_size;
		n = vm_page_size / size;
		/* We lose vm_page_size - n*size bytes here.  */
	} else {
		amount = size;
		n = 1;
	}

	r = vm_allocate(mach_task_self(), &where, (vm_size_t) amount, TRUE);
	assert_perror (r);

	h = (header_t) where;
	do {
		HEADER_NEXT (h) = fl->head;
#ifdef MCHECK
		HEADER_CHECK (h) = CHECK_FREE;
#endif
		fl->head = h;
		h = (header_t) ((char *) h + size);
	} while (--n != 0);
}

/* Declaration changed to standard one for GNU.  */
void *
malloc (size_t size)
{
	int i, n;
	free_list_t fl;
	header_t h;

	if ((int) size < 0)		/* sanity check */
		return 0;
	size += HEADER_SIZE;
	/*
	 * Find smallest power-of-two block size
	 * big enough to hold requested size plus header.
	 */
	i = 0;
	n = MIN_SIZE;
	while (n < size) {
		i += 1;
		n <<= 1;
	}
	assert(i < NBUCKETS);
	fl = &malloc_free_list[i];
	spin_lock(&fl->lock);
	h = fl->head;
	if (h == 0) {
		/*
		 * Free list is empty;
		 * allocate more blocks.
		 */
		more_memory(n, fl);
		h = fl->head;
		if (h == 0) {
			/*
			 * Allocation failed.
			 */
			spin_unlock(&fl->lock);
			return 0;
		}
	}
	/*
	 * Pop block from free list.
	 */
	fl->head = HEADER_NEXT (h);

#ifdef MCHECK
	assert (HEADER_CHECK (h) == CHECK_FREE);
	HEADER_CHECK (h) = CHECK_BUSY;
#endif

#ifdef	DEBUG
	fl->in_use += 1;
#endif	/* DEBUG */
	spin_unlock(&fl->lock);
	/*
	 * Store free list pointer in block header
	 * so we can figure out where it goes
	 * at free() time.
	 */
	HEADER_FREE (h) = fl;
	/*
	 * Return pointer past the block header.
	 */
	return ((char *) h) + HEADER_SIZE;
}

/* Declaration changed to standard one for GNU.  */
void
free (void *base)
{
	header_t h;
	free_list_t fl;
	int i;

	if (base == 0)
		return;
	/*
	 * Find free list for block.
	 */
	h = (header_t) (base - HEADER_SIZE);

#ifdef MCHECK
	assert (HEADER_CHECK (h) == CHECK_BUSY);
#endif

	fl = HEADER_FREE (h);
	i = fl - malloc_free_list;
	/*
	 * Sanity checks.
	 */
	if (i < 0 || i >= NBUCKETS) {
		assert(0 <= i && i < NBUCKETS);
		return;
	}
	if (fl != &malloc_free_list[i]) {
		assert(fl == &malloc_free_list[i]);
		return;
	}
	/*
	 * Push block on free list.
	 */
	spin_lock(&fl->lock);
	HEADER_NEXT (h) = fl->head;
#ifdef MCHECK
	HEADER_CHECK (h) = CHECK_FREE;
#endif
	fl->head = h;
#ifdef	DEBUG
	fl->in_use -= 1;
#endif	/* DEBUG */
	spin_unlock(&fl->lock);
	return;
}

/* Declaration changed to standard one for GNU.  */
void *
realloc (void *old_base, size_t new_size)
{
	header_t h;
	free_list_t fl;
	int i;
	unsigned int old_size;
	char *new_base;

	if (old_base == 0)
	  return malloc (new_size);

	/*
	 * Find size of old block.
	 */
	h = (header_t) (old_base - HEADER_SIZE);
#ifdef MCHECK
	assert (HEADER_CHECK (h) == CHECK_BUSY);
#endif
	fl = HEADER_FREE (h);
	i = fl - malloc_free_list;
	/*
	 * Sanity checks.
	 */
	if (i < 0 || i >= NBUCKETS) {
		assert(0 <= i && i < NBUCKETS);
		return 0;
	}
	if (fl != &malloc_free_list[i]) {
		assert(fl == &malloc_free_list[i]);
		return 0;
	}
	/*
	 * Free list with index i contains blocks of size
	 * 2 ^ (i + * LOG2_MIN_SIZE) including header.
	 */
	old_size = (1 << (i + LOG2_MIN_SIZE)) - HEADER_SIZE;

	if (new_size <= old_size
	    && new_size > (((old_size + HEADER_SIZE) >> 1) - HEADER_SIZE))
	  /* The new size still fits in the same block, and wouldn't fit in
	     the next smaller block!  */
	  return old_base;

	/*
	 * Allocate new block, copy old bytes, and free old block.
	 */
	new_base = malloc(new_size);
	if (new_base)
	  memcpy (new_base, old_base,
		  (int) (old_size < new_size ? old_size : new_size));

	if (new_base || new_size == 0)
	  /* Free OLD_BASE, but only if the malloc didn't fail.  */
	  free (old_base);

	return new_base;
}

#ifdef	DEBUG
void
print_malloc_free_list (void)
{
	int i, size;
	free_list_t fl;
	int n;
	header_t h;
	int total_used = 0;
	int total_free = 0;

	fprintf(stderr, "      Size     In Use       Free      Total\n");
	for (i = 0, size = MIN_SIZE, fl = malloc_free_list;
	     i < NBUCKETS;
	     i += 1, size <<= 1, fl += 1) {
		spin_lock(&fl->lock);
		if (fl->in_use != 0 || fl->head != 0) {
			total_used += fl->in_use * size;
			for (n = 0, h = fl->head; h != 0; h = HEADER_NEXT (h), n += 1)
				;
			total_free += n * size;
			fprintf(stderr, "%10d %10d %10d %10d\n",
				size, fl->in_use, n, fl->in_use + n);
		}
		spin_unlock(&fl->lock);
	}
	fprintf(stderr, " all sizes %10d %10d %10d\n",
		total_used, total_free, total_used + total_free);
}
#endif	/* DEBUG */

void
_hurd_malloc_fork_prepare(void)
/*
 * Prepare the malloc module for a fork by insuring that no thread is in a
 * malloc critical section.
 */
{
    int i;

    for (i = 0; i < NBUCKETS; i++) {
	spin_lock(&malloc_free_list[i].lock);
    }
}

void
_hurd_malloc_fork_parent(void)
/*
 * Called in the parent process after a fork() to resume normal operation.
 */
{
    int i;

    for (i = NBUCKETS-1; i >= 0; i--) {
	spin_unlock(&malloc_free_list[i].lock);
    }
}

void
_hurd_malloc_fork_child(void)
/*
 * Called in the child process after a fork() to resume normal operation.
 */
{
    int i;

    for (i = NBUCKETS-1; i >= 0; i--) {
	spin_unlock(&malloc_free_list[i].lock);
    }
}


text_set_element (_hurd_preinit_hook, malloc_init);
