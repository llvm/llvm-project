/* C string table handling.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Written by Ulrich Drepper <drepper@redhat.com>, 2000.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/cdefs.h>
#include <sys/param.h>


struct Strent
{
  const char *string;
  size_t len;
  struct Strent *next;
  struct Strent *left;
  struct Strent *right;
  size_t offset;
  char reverse[0];
};


struct memoryblock
{
  struct memoryblock *next;
  char memory[0];
};


struct Strtab
{
  struct Strent *root;
  struct memoryblock *memory;
  char *backp;
  size_t left;
  size_t total;

  struct Strent null;
};


/* Cache for the pagesize.  We correct this value a bit so that `malloc'
   is not allocating more than a page.  */
static size_t ps;


#include <programs/xmalloc.h>

/* Prototypes for our functions that are used from iconvconfig.c.  If
   you change these, change also iconvconfig.c.  */
/* Create new C string table object in memory.  */
extern struct Strtab *strtabinit (void);

/* Free resources allocated for C string table ST.  */
extern void strtabfree (struct Strtab *st);

/* Add string STR (length LEN is != 0) to C string table ST.  */
extern struct Strent *strtabadd (struct Strtab *st, const char *str,
				 size_t len);

/* Finalize string table ST and store size in *SIZE and return a pointer.  */
extern void *strtabfinalize (struct Strtab *st, size_t *size);

/* Get offset in string table for string associated with SE.  */
extern size_t strtaboffset (struct Strent *se);


struct Strtab *
strtabinit (void)
{
  struct Strtab *ret;

  if (ps == 0)
    {
      ps = sysconf (_SC_PAGESIZE) - 2 * sizeof (void *);
      assert (sizeof (struct memoryblock) < ps);
    }

  ret = (struct Strtab *) calloc (1, sizeof (struct Strtab));
  if (ret != NULL)
    {
      ret->null.len = 1;
      ret->null.string = "";
    }
  return ret;
}


static void
morememory (struct Strtab *st, size_t len)
{
  struct memoryblock *newmem;

  if (len < ps)
    len = ps;
  newmem = (struct memoryblock *) malloc (len);
  if (newmem == NULL)
    abort ();

  newmem->next = st->memory;
  st->memory = newmem;
  st->backp = newmem->memory;
  st->left = len - offsetof (struct memoryblock, memory);
}


void
strtabfree (struct Strtab *st)
{
  struct memoryblock *mb = st->memory;

  while (mb != NULL)
    {
      void *old = mb;
      mb = mb->next;
      free (old);
    }

  free (st);
}


static struct Strent *
newstring (struct Strtab *st, const char *str, size_t len)
{
  struct Strent *newstr;
  size_t align;
  int i;

  /* Compute the amount of padding needed to make the structure aligned.  */
  align = ((__alignof__ (struct Strent)
	    - (((uintptr_t) st->backp)
	       & (__alignof__ (struct Strent) - 1)))
	   & (__alignof__ (struct Strent) - 1));

  /* Make sure there is enough room in the memory block.  */
  if (st->left < align + sizeof (struct Strent) + len)
    {
      morememory (st, sizeof (struct Strent) + len);
      align = 0;
    }

  /* Create the reserved string.  */
  newstr = (struct Strent *) (st->backp + align);
  newstr->string = str;
  newstr->len = len;
  newstr->next = NULL;
  newstr->left = NULL;
  newstr->right = NULL;
  newstr->offset = 0;
  for (i = len - 2; i >= 0; --i)
    newstr->reverse[i] = str[len - 2 - i];
  newstr->reverse[len - 1] = '\0';
  st->backp += align + sizeof (struct Strent) + len;
  st->left -= align + sizeof (struct Strent) + len;

  return newstr;
}


/* XXX This function should definitely be rewritten to use a balancing
   tree algorithm (AVL, red-black trees).  For now a simple, correct
   implementation is enough.  */
static struct Strent **
searchstring (struct Strent **sep, struct Strent *newstr)
{
  int cmpres;

  /* More strings?  */
  if (*sep == NULL)
    {
      *sep = newstr;
      return sep;
    }

  /* Compare the strings.  */
  cmpres = memcmp ((*sep)->reverse, newstr->reverse,
		   MIN ((*sep)->len, newstr->len) - 1);
  if (cmpres == 0)
    /* We found a matching string.  */
    return sep;
  else if (cmpres > 0)
    return searchstring (&(*sep)->left, newstr);
  else
    return searchstring (&(*sep)->right, newstr);
}


/* Add new string.  The actual string is assumed to be permanent.  */
struct Strent *
strtabadd (struct Strtab *st, const char *str, size_t len)
{
  struct Strent *newstr;
  struct Strent **sep;

  /* Compute the string length if the caller doesn't know it.  */
  if (len == 0)
    len = strlen (str) + 1;

  /* Make sure all "" strings get offset 0.  */
  if (len == 1)
    return &st->null;

  /* Allocate memory for the new string and its associated information.  */
  newstr = newstring (st, str, len);

  /* Search in the array for the place to insert the string.  If there
     is no string with matching prefix and no string with matching
     leading substring, create a new entry.  */
  sep = searchstring (&st->root, newstr);
  if (*sep != newstr)
    {
      /* This is not the same entry.  This means we have a prefix match.  */
      if ((*sep)->len > newstr->len)
	{
	  struct Strent *subs;

	  for (subs = (*sep)->next; subs; subs = subs->next)
	    if (subs->len == newstr->len)
	      {
		/* We have an exact match with a substring.  Free the memory
		   we allocated.  */
		st->left += st->backp - (char *) newstr;
		st->backp = (char *) newstr;

		return subs;
	      }

	  /* We have a new substring.  This means we don't need the reverse
	     string of this entry anymore.  */
	  st->backp -= newstr->len;
	  st->left += newstr->len;

	  newstr->next = (*sep)->next;
	  (*sep)->next = newstr;
	}
      else if ((*sep)->len != newstr->len)
	{
	  /* When we get here it means that the string we are about to
	     add has a common prefix with a string we already have but
	     it is longer.  In this case we have to put it first.  */
	  st->total += newstr->len - (*sep)->len;
	  newstr->next = *sep;
	  newstr->left = (*sep)->left;
	  newstr->right = (*sep)->right;
	  *sep = newstr;
	}
      else
	{
	  /* We have an exact match.  Free the memory we allocated.  */
	  st->left += st->backp - (char *) newstr;
	  st->backp = (char *) newstr;

	  newstr = *sep;
	}
    }
  else
    st->total += newstr->len;

  return newstr;
}


static void
copystrings (struct Strent *nodep, char **freep, size_t *offsetp)
{
  struct Strent *subs;

  if (nodep->left != NULL)
    copystrings (nodep->left, freep, offsetp);

  /* Process the current node.  */
  nodep->offset = *offsetp;
  *freep = (char *) mempcpy (*freep, nodep->string, nodep->len);
  *offsetp += nodep->len;

  for (subs = nodep->next; subs != NULL; subs = subs->next)
    {
      assert (subs->len < nodep->len);
      subs->offset = nodep->offset + nodep->len - subs->len;
    }

  if (nodep->right != NULL)
    copystrings (nodep->right, freep, offsetp);
}


void *
strtabfinalize (struct Strtab *st, size_t *size)
{
  size_t copylen;
  char *endp;
  char *retval;

  /* Fill in the information.  */
  endp = retval = (char *) xmalloc (st->total + 1);

  /* Always put an empty string at the beginning so that a zero offset
     can mean error.  */
  *endp++ = '\0';

  /* Now run through the tree and add all the string while also updating
     the offset members of the elfstrent records.  */
  copylen = 1;
  copystrings (st->root, &endp, &copylen);
  assert (copylen == st->total + 1);
  assert (endp == retval + st->total + 1);
  *size = copylen;

  return retval;
}


size_t
strtaboffset (struct Strent *se)
{
  return se->offset;
}
