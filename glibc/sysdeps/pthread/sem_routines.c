/* Helper code for POSIX semaphore implementation.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <search.h>
#include <semaphoreP.h>
#include <sys/mman.h>
#include <sem_routines.h>

/* Keeping track of currently used mappings.  */
struct inuse_sem
{
  dev_t dev;
  ino_t ino;
  int refcnt;
  sem_t *sem;
  char name[];
};

struct search_sem
{
  dev_t dev;
  ino_t ino;
  int refcnt;
  sem_t *sem;
  char name[NAME_MAX + 1];
};

/* Comparison function for search of existing mapping.  */
static int
sem_search (const void *a, const void *b)
{
  const struct inuse_sem *as = (const struct inuse_sem *) a;
  const struct inuse_sem *bs = (const struct inuse_sem *) b;

  if (as->ino != bs->ino)
    /* Cannot return the difference the type is larger than int.  */
    return as->ino < bs->ino ? -1 : (as->ino == bs->ino ? 0 : 1);

  if (as->dev != bs->dev)
    /* Cannot return the difference the type is larger than int.  */
    return as->dev < bs->dev ? -1 : (as->dev == bs->dev ? 0 : 1);

  return strcmp (as->name, bs->name);
}

/* The search tree for existing mappings.  */
static void *sem_mappings;

/* Lock to protect the search tree.  */
static int sem_mappings_lock = LLL_LOCK_INITIALIZER;


/* Search for existing mapping and if possible add the one provided.  */
sem_t *
__sem_check_add_mapping (const char *name, int fd, sem_t *existing)
{
  size_t namelen = strlen (name);
  if (namelen > NAME_MAX)
    return SEM_FAILED;
  namelen += 1;

  sem_t *result = SEM_FAILED;

  /* Get the information about the file.  */
  struct stat64 st;
  if (__fstat64 (fd, &st) == 0)
    {
      /* Get the lock.  */
      lll_lock (sem_mappings_lock, LLL_PRIVATE);

      /* Search for an existing mapping given the information we have.  */
      struct search_sem fake;
      memcpy (fake.name, name, namelen);
      fake.dev = st.st_dev;
      fake.ino = st.st_ino;

      struct inuse_sem **foundp = __tfind (&fake, &sem_mappings, sem_search);
      if (foundp != NULL)
	{
	  /* There is already a mapping.  Use it.  */
	  result = (*foundp)->sem;
	  ++(*foundp)->refcnt;
	}
      else
	{
	  /* We haven't found a mapping.  Install ione.  */
	  struct inuse_sem *newp;

	  newp = (struct inuse_sem *) malloc (sizeof (*newp) + namelen);
	  if (newp != NULL)
	    {
	      /* If the caller hasn't provided any map it now.  */
	      if (existing == SEM_FAILED)
		existing = (sem_t *) __mmap (NULL, sizeof (sem_t),
					     PROT_READ | PROT_WRITE,
					     MAP_SHARED, fd, 0);

	      newp->dev = st.st_dev;
	      newp->ino = st.st_ino;
	      newp->refcnt = 1;
	      newp->sem = existing;
	      memcpy (newp->name, name, namelen);

	      /* Insert the new value.  */
	      if (existing != MAP_FAILED
		  && __tsearch (newp, &sem_mappings, sem_search) != NULL)
		/* Successful.  */
		result = existing;
	      else
		/* Something went wrong while inserting the new
		   value.  We fail completely.  */
		free (newp);
	    }
	}

      /* Release the lock.  */
      lll_unlock (sem_mappings_lock, LLL_PRIVATE);
    }

  if (result != existing && existing != SEM_FAILED && existing != MAP_FAILED)
    {
      /* Do not disturb errno.  */
      int save = errno;
      __munmap (existing, sizeof (sem_t));
      errno = save;
    }

  return result;
}

struct walk_closure
{
  sem_t *the_sem;
  struct inuse_sem *rec;
};

static void
walker (const void *inodep, VISIT which, void *closure0)
{
  struct walk_closure *closure = closure0;
  struct inuse_sem *nodep = *(struct inuse_sem **) inodep;

  if (nodep->sem == closure->the_sem)
    closure->rec = nodep;
}

bool
__sem_remove_mapping (sem_t *sem)
{
  bool ret = true;

  /* Get the lock.  */
  lll_lock (sem_mappings_lock, LLL_PRIVATE);

  /* Locate the entry for the mapping the caller provided.  */
  struct inuse_sem *rec;
  {
    struct walk_closure closure = { .the_sem = sem, .rec = NULL };
    __twalk_r (sem_mappings, walker, &closure);
    rec = closure.rec;
  }
  if (rec != NULL)
    {
      /* Check the reference counter.  If it is going to be zero, free
	 all the resources.  */
      if (--rec->refcnt == 0)
	{
	  /* Remove the record from the tree.  */
	  __tdelete (rec, &sem_mappings, sem_search);

	  if (__munmap (rec->sem, sizeof (sem_t)) == -1)
	    ret = false;

	  free (rec);
	}
    }
  else
    ret = false;

  /* Release the lock.  */
  lll_unlock (sem_mappings_lock, LLL_PRIVATE);

  return ret;
}
