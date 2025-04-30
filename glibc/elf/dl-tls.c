/* Thread-local storage handling in the ELF dynamic linker.  Generic version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <errno.h>
#include <libintl.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/param.h>
#include <atomic.h>

#include <tls.h>
#include <dl-tls.h>
#include <ldsodefs.h>

#if THREAD_GSCOPE_IN_TCB
# include <list.h>
#endif

#define TUNABLE_NAMESPACE rtld
#include <dl-tunables.h>

/* Surplus static TLS, GLRO(dl_tls_static_surplus), is used for

   - IE TLS in libc.so for all dlmopen namespaces except in the initial
     one where libc.so is not loaded dynamically but at startup time,
   - IE TLS in other libraries which may be dynamically loaded even in the
     initial namespace,
   - and optionally for optimizing dynamic TLS access.

   The maximum number of namespaces is DL_NNS, but to support that many
   namespaces correctly the static TLS allocation should be significantly
   increased, which may cause problems with small thread stacks due to the
   way static TLS is accounted (bug 11787).

   So there is a rtld.nns tunable limit on the number of supported namespaces
   that affects the size of the static TLS and by default it's small enough
   not to cause problems with existing applications. The limit is not
   enforced or checked: it is the user's responsibility to increase rtld.nns
   if more dlmopen namespaces are used.

   Audit modules use their own namespaces, they are not included in rtld.nns,
   but come on top when computing the number of namespaces.  */

/* Size of initial-exec TLS in libc.so.  This should be the maximum of
   observed PT_GNU_TLS sizes across all architectures.  Some
   architectures have lower values due to differences in type sizes
   and link editor capabilities.  */
#define LIBC_IE_TLS 144

/* Size of initial-exec TLS in libraries other than libc.so.
   This should be large enough to cover runtime libraries of the
   compiler such as libgomp and libraries in libc other than libc.so.  */
#define OTHER_IE_TLS 144

/* Default number of namespaces.  */
#define DEFAULT_NNS 4

/* Default for dl_tls_static_optional.  */
#define OPTIONAL_TLS 512

/* Compute the static TLS surplus based on the namespace count and the
   TLS space that can be used for optimizations.  */
static inline int
tls_static_surplus (int nns, int opt_tls)
{
  return (nns - 1) * LIBC_IE_TLS + nns * OTHER_IE_TLS + opt_tls;
}

/* This value is chosen so that with default values for the tunables,
   the computation of dl_tls_static_surplus in
   _dl_tls_static_surplus_init yields the historic value 1664, for
   backwards compatibility.  */
#define LEGACY_TLS (1664 - tls_static_surplus (DEFAULT_NNS, OPTIONAL_TLS))

/* Calculate the size of the static TLS surplus, when the given
   number of audit modules are loaded.  Must be called after the
   number of audit modules is known and before static TLS allocation.  */
void
_dl_tls_static_surplus_init (size_t naudit)
{
  size_t nns, opt_tls;

#if HAVE_TUNABLES
  nns = TUNABLE_GET (nns, size_t, NULL);
  opt_tls = TUNABLE_GET (optional_static_tls, size_t, NULL);
#else
  /* Default values of the tunables.  */
  nns = DEFAULT_NNS;
  opt_tls = OPTIONAL_TLS;
#endif
  if (nns > DL_NNS)
    nns = DL_NNS;
  if (DL_NNS - nns < naudit)
    _dl_fatal_printf ("Failed loading %lu audit modules, %lu are supported.\n",
		      (unsigned long) naudit, (unsigned long) (DL_NNS - nns));
  nns += naudit;

  GL(dl_tls_static_optional) = opt_tls;
  assert (LEGACY_TLS >= 0);
  GLRO(dl_tls_static_surplus) = tls_static_surplus (nns, opt_tls) + LEGACY_TLS;
}

/* Out-of-memory handler.  */
static void
__attribute__ ((__noreturn__))
oom (void)
{
  _dl_fatal_printf ("cannot allocate memory for thread-local data: ABORT\n");
}


void
_dl_assign_tls_modid (struct link_map *l)
{
  size_t result;

  if (__builtin_expect (GL(dl_tls_dtv_gaps), false))
    {
      size_t disp = 0;
      struct dtv_slotinfo_list *runp = GL(dl_tls_dtv_slotinfo_list);

      /* Note that this branch will never be executed during program
	 start since there are no gaps at that time.  Therefore it
	 does not matter that the dl_tls_dtv_slotinfo is not allocated
	 yet when the function is called for the first times.

	 NB: the offset +1 is due to the fact that DTV[0] is used
	 for something else.  */
      result = GL(dl_tls_static_nelem) + 1;
      if (result <= GL(dl_tls_max_dtv_idx))
	do
	  {
	    while (result - disp < runp->len)
	      {
		if (runp->slotinfo[result - disp].map == NULL)
		  break;

		++result;
		assert (result <= GL(dl_tls_max_dtv_idx) + 1);
	      }

	    if (result - disp < runp->len)
	      {
		/* Mark the entry as used, so any dependency see it.  */
		atomic_store_relaxed (&runp->slotinfo[result - disp].map, l);
		break;
	      }

	    disp += runp->len;
	  }
	while ((runp = runp->next) != NULL);

      if (result > GL(dl_tls_max_dtv_idx))
	{
	  /* The new index must indeed be exactly one higher than the
	     previous high.  */
	  assert (result == GL(dl_tls_max_dtv_idx) + 1);
	  /* There is no gap anymore.  */
	  GL(dl_tls_dtv_gaps) = false;

	  goto nogaps;
	}
    }
  else
    {
      /* No gaps, allocate a new entry.  */
    nogaps:

      result = GL(dl_tls_max_dtv_idx) + 1;
      /* Can be read concurrently.  */
      atomic_store_relaxed (&GL(dl_tls_max_dtv_idx), result);
    }

  l->l_tls_modid = result;
}


size_t
_dl_count_modids (void)
{
  /* The count is the max unless dlclose or failed dlopen created gaps.  */
  if (__glibc_likely (!GL(dl_tls_dtv_gaps)))
    return GL(dl_tls_max_dtv_idx);

  /* We have gaps and are forced to count the non-NULL entries.  */
  size_t n = 0;
  struct dtv_slotinfo_list *runp = GL(dl_tls_dtv_slotinfo_list);
  while (runp != NULL)
    {
      for (size_t i = 0; i < runp->len; ++i)
	if (runp->slotinfo[i].map != NULL)
	  ++n;

      runp = runp->next;
    }

  return n;
}


#ifdef SHARED
void
_dl_determine_tlsoffset (void)
{
  size_t max_align = TLS_TCB_ALIGN;
  size_t freetop = 0;
  size_t freebottom = 0;

  /* The first element of the dtv slot info list is allocated.  */
  assert (GL(dl_tls_dtv_slotinfo_list) != NULL);
  /* There is at this point only one element in the
     dl_tls_dtv_slotinfo_list list.  */
  assert (GL(dl_tls_dtv_slotinfo_list)->next == NULL);

  struct dtv_slotinfo *slotinfo = GL(dl_tls_dtv_slotinfo_list)->slotinfo;

  /* Determining the offset of the various parts of the static TLS
     block has several dependencies.  In addition we have to work
     around bugs in some toolchains.

     Each TLS block from the objects available at link time has a size
     and an alignment requirement.  The GNU ld computes the alignment
     requirements for the data at the positions *in the file*, though.
     I.e, it is not simply possible to allocate a block with the size
     of the TLS program header entry.  The data is layed out assuming
     that the first byte of the TLS block fulfills

       p_vaddr mod p_align == &TLS_BLOCK mod p_align

     This means we have to add artificial padding at the beginning of
     the TLS block.  These bytes are never used for the TLS data in
     this module but the first byte allocated must be aligned
     according to mod p_align == 0 so that the first byte of the TLS
     block is aligned according to p_vaddr mod p_align.  This is ugly
     and the linker can help by computing the offsets in the TLS block
     assuming the first byte of the TLS block is aligned according to
     p_align.

     The extra space which might be allocated before the first byte of
     the TLS block need not go unused.  The code below tries to use
     that memory for the next TLS block.  This can work if the total
     memory requirement for the next TLS block is smaller than the
     gap.  */

#if TLS_TCB_AT_TP
  /* We simply start with zero.  */
  size_t offset = 0;

  for (size_t cnt = 0; slotinfo[cnt].map != NULL; ++cnt)
    {
      assert (cnt < GL(dl_tls_dtv_slotinfo_list)->len);

      size_t firstbyte = (-slotinfo[cnt].map->l_tls_firstbyte_offset
			  & (slotinfo[cnt].map->l_tls_align - 1));
      size_t off;
      max_align = MAX (max_align, slotinfo[cnt].map->l_tls_align);

      if (freebottom - freetop >= slotinfo[cnt].map->l_tls_blocksize)
	{
	  off = roundup (freetop + slotinfo[cnt].map->l_tls_blocksize
			 - firstbyte, slotinfo[cnt].map->l_tls_align)
		+ firstbyte;
	  if (off <= freebottom)
	    {
	      freetop = off;

	      /* XXX For some architectures we perhaps should store the
		 negative offset.  */
	      slotinfo[cnt].map->l_tls_offset = off;
	      continue;
	    }
	}

      off = roundup (offset + slotinfo[cnt].map->l_tls_blocksize - firstbyte,
		     slotinfo[cnt].map->l_tls_align) + firstbyte;
      if (off > offset + slotinfo[cnt].map->l_tls_blocksize
		+ (freebottom - freetop))
	{
	  freetop = offset;
	  freebottom = off - slotinfo[cnt].map->l_tls_blocksize;
	}
      offset = off;

      /* XXX For some architectures we perhaps should store the
	 negative offset.  */
      slotinfo[cnt].map->l_tls_offset = off;
    }

  GL(dl_tls_static_used) = offset;
  GLRO (dl_tls_static_size) = (roundup (offset + GLRO(dl_tls_static_surplus),
					max_align)
			       + TLS_TCB_SIZE);
#elif TLS_DTV_AT_TP
  /* The TLS blocks start right after the TCB.  */
  size_t offset = TLS_TCB_SIZE;

  for (size_t cnt = 0; slotinfo[cnt].map != NULL; ++cnt)
    {
      assert (cnt < GL(dl_tls_dtv_slotinfo_list)->len);

      size_t firstbyte = (-slotinfo[cnt].map->l_tls_firstbyte_offset
			  & (slotinfo[cnt].map->l_tls_align - 1));
      size_t off;
      max_align = MAX (max_align, slotinfo[cnt].map->l_tls_align);

      if (slotinfo[cnt].map->l_tls_blocksize <= freetop - freebottom)
	{
	  off = roundup (freebottom, slotinfo[cnt].map->l_tls_align);
	  if (off - freebottom < firstbyte)
	    off += slotinfo[cnt].map->l_tls_align;
	  if (off + slotinfo[cnt].map->l_tls_blocksize - firstbyte <= freetop)
	    {
	      slotinfo[cnt].map->l_tls_offset = off - firstbyte;
	      freebottom = (off + slotinfo[cnt].map->l_tls_blocksize
			    - firstbyte);
	      continue;
	    }
	}

      off = roundup (offset, slotinfo[cnt].map->l_tls_align);
      if (off - offset < firstbyte)
	off += slotinfo[cnt].map->l_tls_align;

      slotinfo[cnt].map->l_tls_offset = off - firstbyte;
      if (off - firstbyte - offset > freetop - freebottom)
	{
	  freebottom = offset;
	  freetop = off - firstbyte;
	}

      offset = off + slotinfo[cnt].map->l_tls_blocksize - firstbyte;
    }

  GL(dl_tls_static_used) = offset;
  GLRO (dl_tls_static_size) = roundup (offset + GLRO(dl_tls_static_surplus),
				       TLS_TCB_ALIGN);
#else
# error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
#endif

  /* The alignment requirement for the static TLS block.  */
  GLRO (dl_tls_static_align) = max_align;
}
#endif /* SHARED */

static void *
allocate_dtv (void *result)
{
  dtv_t *dtv;
  size_t dtv_length;

  /* Relaxed MO, because the dtv size is later rechecked, not relied on.  */
  size_t max_modid = atomic_load_relaxed (&GL(dl_tls_max_dtv_idx));
  /* We allocate a few more elements in the dtv than are needed for the
     initial set of modules.  This should avoid in most cases expansions
     of the dtv.  */
  dtv_length = max_modid + DTV_SURPLUS;
  dtv = calloc (dtv_length + 2, sizeof (dtv_t));
  if (dtv != NULL)
    {
      /* This is the initial length of the dtv.  */
      dtv[0].counter = dtv_length;

      /* The rest of the dtv (including the generation counter) is
	 Initialize with zero to indicate nothing there.  */

      /* Add the dtv to the thread data structures.  */
      INSTALL_DTV (result, dtv);
    }
  else
    result = NULL;

  return result;
}

/* Get size and alignment requirements of the static TLS block.  This
   function is no longer used by glibc itself, but the GCC sanitizers
   use it despite the GLIBC_PRIVATE status.  */
void
_dl_get_tls_static_info (size_t *sizep, size_t *alignp)
{
  *sizep = GLRO (dl_tls_static_size);
  *alignp = GLRO (dl_tls_static_align);
}

/* Derive the location of the pointer to the start of the original
   allocation (before alignment) from the pointer to the TCB.  */
static inline void **
tcb_to_pointer_to_free_location (void *tcb)
{
#if TLS_TCB_AT_TP
  /* The TCB follows the TLS blocks, and the pointer to the front
     follows the TCB.  */
  void **original_pointer_location = tcb + TLS_TCB_SIZE;
#elif TLS_DTV_AT_TP
  /* The TCB comes first, preceded by the pre-TCB, and the pointer is
     before that.  */
  void **original_pointer_location = tcb - TLS_PRE_TCB_SIZE - sizeof (void *);
#endif
  return original_pointer_location;
}

void *
_dl_allocate_tls_storage (void)
{
  void *result;
  size_t size = GLRO (dl_tls_static_size);

#if TLS_DTV_AT_TP
  /* Memory layout is:
     [ TLS_PRE_TCB_SIZE ] [ TLS_TCB_SIZE ] [ TLS blocks ]
			  ^ This should be returned.  */
  size += TLS_PRE_TCB_SIZE;
#endif

  /* Perform the allocation.  Reserve space for the required alignment
     and the pointer to the original allocation.  */
  size_t alignment = GLRO (dl_tls_static_align);
  void *allocated = malloc (size + alignment + sizeof (void *));
  if (__glibc_unlikely (allocated == NULL))
    return NULL;

  /* Perform alignment and allocate the DTV.  */
#if TLS_TCB_AT_TP
  /* The TCB follows the TLS blocks, which determine the alignment.
     (TCB alignment requirements have been taken into account when
     calculating GLRO (dl_tls_static_align).)  */
  void *aligned = (void *) roundup ((uintptr_t) allocated, alignment);
  result = aligned + size - TLS_TCB_SIZE;

  /* Clear the TCB data structure.  We can't ask the caller (i.e.
     libpthread) to do it, because we will initialize the DTV et al.  */
  memset (result, '\0', TLS_TCB_SIZE);
#elif TLS_DTV_AT_TP
  /* Pre-TCB and TCB come before the TLS blocks.  The layout computed
     in _dl_determine_tlsoffset assumes that the TCB is aligned to the
     TLS block alignment, and not just the TLS blocks after it.  This
     can leave an unused alignment gap between the TCB and the TLS
     blocks.  */
  result = (void *) roundup
    (sizeof (void *) + TLS_PRE_TCB_SIZE + (uintptr_t) allocated,
     alignment);

  /* Clear the TCB data structure and TLS_PRE_TCB_SIZE bytes before
     it.  We can't ask the caller (i.e. libpthread) to do it, because
     we will initialize the DTV et al.  */
  memset (result - TLS_PRE_TCB_SIZE, '\0', TLS_PRE_TCB_SIZE + TLS_TCB_SIZE);
#endif

  /* Record the value of the original pointer for later
     deallocation.  */
  *tcb_to_pointer_to_free_location (result) = allocated;

  result = allocate_dtv (result);
  if (result == NULL)
    free (allocated);
  return result;
}


#ifndef SHARED
extern dtv_t _dl_static_dtv[];
# define _dl_initial_dtv (&_dl_static_dtv[1])
#endif

static dtv_t *
_dl_resize_dtv (dtv_t *dtv, size_t max_modid)
{
  /* Resize the dtv.  */
  dtv_t *newp;
  size_t newsize = max_modid + DTV_SURPLUS;
  size_t oldsize = dtv[-1].counter;

  if (dtv == GL(dl_initial_dtv))
    {
      /* This is the initial dtv that was either statically allocated in
	 __libc_setup_tls or allocated during rtld startup using the
	 dl-minimal.c malloc instead of the real malloc.  We can't free
	 it, we have to abandon the old storage.  */

      newp = malloc ((2 + newsize) * sizeof (dtv_t));
      if (newp == NULL)
	oom ();
      memcpy (newp, &dtv[-1], (2 + oldsize) * sizeof (dtv_t));
    }
  else
    {
      newp = realloc (&dtv[-1],
		      (2 + newsize) * sizeof (dtv_t));
      if (newp == NULL)
	oom ();
    }

  newp[0].counter = newsize;

  /* Clear the newly allocated part.  */
  memset (newp + 2 + oldsize, '\0',
	  (newsize - oldsize) * sizeof (dtv_t));

  /* Return the generation counter.  */
  return &newp[1];
}


void *
_dl_allocate_tls_init (void *result)
{
  if (result == NULL)
    /* The memory allocation failed.  */
    return NULL;

  dtv_t *dtv = GET_DTV (result);
  struct dtv_slotinfo_list *listp;
  size_t total = 0;
  size_t maxgen = 0;

  /* Protects global dynamic TLS related state.  */
  __rtld_lock_lock_recursive (GL(dl_load_lock));

  /* Check if the current dtv is big enough.   */
  if (dtv[-1].counter < GL(dl_tls_max_dtv_idx))
    {
      /* Resize the dtv.  */
      dtv = _dl_resize_dtv (dtv, GL(dl_tls_max_dtv_idx));

      /* Install this new dtv in the thread data structures.  */
      INSTALL_DTV (result, &dtv[-1]);
    }

  /* We have to prepare the dtv for all currently loaded modules using
     TLS.  For those which are dynamically loaded we add the values
     indicating deferred allocation.  */
  listp = GL(dl_tls_dtv_slotinfo_list);
  while (1)
    {
      size_t cnt;

      for (cnt = total == 0 ? 1 : 0; cnt < listp->len; ++cnt)
	{
	  struct link_map *map;
	  void *dest;

	  /* Check for the total number of used slots.  */
	  if (total + cnt > GL(dl_tls_max_dtv_idx))
	    break;

	  map = listp->slotinfo[cnt].map;
	  if (map == NULL)
	    /* Unused entry.  */
	    continue;

	  /* Keep track of the maximum generation number.  This might
	     not be the generation counter.  */
	  assert (listp->slotinfo[cnt].gen <= GL(dl_tls_generation));
	  maxgen = MAX (maxgen, listp->slotinfo[cnt].gen);

	  dtv[map->l_tls_modid].pointer.val = TLS_DTV_UNALLOCATED;
	  dtv[map->l_tls_modid].pointer.to_free = NULL;

	  if (map->l_tls_offset == NO_TLS_OFFSET
	      || map->l_tls_offset == FORCED_DYNAMIC_TLS_OFFSET)
	    continue;

	  assert (map->l_tls_modid == total + cnt);
	  assert (map->l_tls_blocksize >= map->l_tls_initimage_size);
#if TLS_TCB_AT_TP
	  assert ((size_t) map->l_tls_offset >= map->l_tls_blocksize);
	  dest = (char *) result - map->l_tls_offset;
#elif TLS_DTV_AT_TP
	  dest = (char *) result + map->l_tls_offset;
#else
# error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
#endif

	  /* Set up the DTV entry.  The simplified __tls_get_addr that
	     some platforms use in static programs requires it.  */
	  dtv[map->l_tls_modid].pointer.val = dest;

	  /* Copy the initialization image and clear the BSS part.  */
	  memset (__mempcpy (dest, map->l_tls_initimage,
			     map->l_tls_initimage_size), '\0',
		  map->l_tls_blocksize - map->l_tls_initimage_size);
	}

      total += cnt;
      if (total > GL(dl_tls_max_dtv_idx))
	break;

      listp = listp->next;
      assert (listp != NULL);
    }
  __rtld_lock_unlock_recursive (GL(dl_load_lock));

  /* The DTV version is up-to-date now.  */
  dtv[0].counter = maxgen;

  return result;
}
rtld_hidden_def (_dl_allocate_tls_init)

void *
_dl_allocate_tls (void *mem)
{
  return _dl_allocate_tls_init (mem == NULL
				? _dl_allocate_tls_storage ()
				: allocate_dtv (mem));
}
rtld_hidden_def (_dl_allocate_tls)


void
_dl_deallocate_tls (void *tcb, bool dealloc_tcb)
{
  dtv_t *dtv = GET_DTV (tcb);

  /* We need to free the memory allocated for non-static TLS.  */
  for (size_t cnt = 0; cnt < dtv[-1].counter; ++cnt)
    free (dtv[1 + cnt].pointer.to_free);

  /* The array starts with dtv[-1].  */
  if (dtv != GL(dl_initial_dtv))
    free (dtv - 1);

  if (dealloc_tcb)
    free (*tcb_to_pointer_to_free_location (tcb));
}
rtld_hidden_def (_dl_deallocate_tls)


#ifdef SHARED
/* The __tls_get_addr function has two basic forms which differ in the
   arguments.  The IA-64 form takes two parameters, the module ID and
   offset.  The form used, among others, on IA-32 takes a reference to
   a special structure which contain the same information.  The second
   form seems to be more often used (in the moment) so we default to
   it.  Users of the IA-64 form have to provide adequate definitions
   of the following macros.  */
# ifndef GET_ADDR_ARGS
#  define GET_ADDR_ARGS tls_index *ti
#  define GET_ADDR_PARAM ti
# endif
# ifndef GET_ADDR_MODULE
#  define GET_ADDR_MODULE ti->ti_module
# endif
# ifndef GET_ADDR_OFFSET
#  define GET_ADDR_OFFSET ti->ti_offset
# endif

/* Allocate one DTV entry.  */
static struct dtv_pointer
allocate_dtv_entry (size_t alignment, size_t size)
{
  if (powerof2 (alignment) && alignment <= _Alignof (max_align_t))
    {
      /* The alignment is supported by malloc.  */
      void *ptr = malloc (size);
      return (struct dtv_pointer) { ptr, ptr };
    }

  /* Emulate memalign to by manually aligning a pointer returned by
     malloc.  First compute the size with an overflow check.  */
  size_t alloc_size = size + alignment;
  if (alloc_size < size)
    return (struct dtv_pointer) {};

  /* Perform the allocation.  This is the pointer we need to free
     later.  */
  void *start = malloc (alloc_size);
  if (start == NULL)
    return (struct dtv_pointer) {};

  /* Find the aligned position within the larger allocation.  */
  void *aligned = (void *) roundup ((uintptr_t) start, alignment);

  return (struct dtv_pointer) { .val = aligned, .to_free = start };
}

static struct dtv_pointer
allocate_and_init (struct link_map *map)
{
  struct dtv_pointer result = allocate_dtv_entry
    (map->l_tls_align, map->l_tls_blocksize);
  if (result.val == NULL)
    oom ();

  /* Initialize the memory.  */
  memset (__mempcpy (result.val, map->l_tls_initimage,
		     map->l_tls_initimage_size),
	  '\0', map->l_tls_blocksize - map->l_tls_initimage_size);

  return result;
}


struct link_map *
_dl_update_slotinfo (unsigned long int req_modid)
{
  struct link_map *the_map = NULL;
  dtv_t *dtv = THREAD_DTV ();

  /* The global dl_tls_dtv_slotinfo array contains for each module
     index the generation counter current when the entry was created.
     This array never shrinks so that all module indices which were
     valid at some time can be used to access it.  Before the first
     use of a new module index in this function the array was extended
     appropriately.  Access also does not have to be guarded against
     modifications of the array.  It is assumed that pointer-size
     values can be read atomically even in SMP environments.  It is
     possible that other threads at the same time dynamically load
     code and therefore add to the slotinfo list.  This is a problem
     since we must not pick up any information about incomplete work.
     The solution to this is to ignore all dtv slots which were
     created after the one we are currently interested.  We know that
     dynamic loading for this module is completed and this is the last
     load operation we know finished.  */
  unsigned long int idx = req_modid;
  struct dtv_slotinfo_list *listp = GL(dl_tls_dtv_slotinfo_list);

  while (idx >= listp->len)
    {
      idx -= listp->len;
      listp = listp->next;
    }

  if (dtv[0].counter < listp->slotinfo[idx].gen)
    {
      /* CONCURRENCY NOTES:

	 Here the dtv needs to be updated to new_gen generation count.

	 This code may be called during TLS access when GL(dl_load_lock)
	 is not held.  In that case the user code has to synchronize with
	 dlopen and dlclose calls of relevant modules.  A module m is
	 relevant if the generation of m <= new_gen and dlclose of m is
	 synchronized: a memory access here happens after the dlopen and
	 before the dlclose of relevant modules.  The dtv entries for
	 relevant modules need to be updated, other entries can be
	 arbitrary.

	 This e.g. means that the first part of the slotinfo list can be
	 accessed race free, but the tail may be concurrently extended.
	 Similarly relevant slotinfo entries can be read race free, but
	 other entries are racy.  However updating a non-relevant dtv
	 entry does not affect correctness.  For a relevant module m,
	 max_modid >= modid of m.  */
      size_t new_gen = listp->slotinfo[idx].gen;
      size_t total = 0;
      size_t max_modid  = atomic_load_relaxed (&GL(dl_tls_max_dtv_idx));
      assert (max_modid >= req_modid);

      /* We have to look through the entire dtv slotinfo list.  */
      listp =  GL(dl_tls_dtv_slotinfo_list);
      do
	{
	  for (size_t cnt = total == 0 ? 1 : 0; cnt < listp->len; ++cnt)
	    {
	      size_t modid = total + cnt;

	      /* Later entries are not relevant.  */
	      if (modid > max_modid)
		break;

	      size_t gen = atomic_load_relaxed (&listp->slotinfo[cnt].gen);

	      if (gen > new_gen)
		/* Not relevant.  */
		continue;

	      /* If the entry is older than the current dtv layout we
		 know we don't have to handle it.  */
	      if (gen <= dtv[0].counter)
		continue;

	      /* If there is no map this means the entry is empty.  */
	      struct link_map *map
		= atomic_load_relaxed (&listp->slotinfo[cnt].map);
	      /* Check whether the current dtv array is large enough.  */
	      if (dtv[-1].counter < modid)
		{
		  if (map == NULL)
		    continue;

		  /* Resize the dtv.  */
		  dtv = _dl_resize_dtv (dtv, max_modid);

		  assert (modid <= dtv[-1].counter);

		  /* Install this new dtv in the thread data
		     structures.  */
		  INSTALL_NEW_DTV (dtv);
		}

	      /* If there is currently memory allocate for this
		 dtv entry free it.  */
	      /* XXX Ideally we will at some point create a memory
		 pool.  */
	      free (dtv[modid].pointer.to_free);
	      dtv[modid].pointer.val = TLS_DTV_UNALLOCATED;
	      dtv[modid].pointer.to_free = NULL;

	      if (modid == req_modid)
		the_map = map;
	    }

	  total += listp->len;
	  if (total > max_modid)
	    break;

	  /* Synchronize with _dl_add_to_slotinfo.  Ideally this would
	     be consume MO since we only need to order the accesses to
	     the next node after the read of the address and on most
	     hardware (other than alpha) a normal load would do that
	     because of the address dependency.  */
	  listp = atomic_load_acquire (&listp->next);
	}
      while (listp != NULL);

      /* This will be the new maximum generation counter.  */
      dtv[0].counter = new_gen;
    }

  return the_map;
}


static void *
__attribute_noinline__
tls_get_addr_tail (GET_ADDR_ARGS, dtv_t *dtv, struct link_map *the_map)
{
  /* The allocation was deferred.  Do it now.  */
  if (the_map == NULL)
    {
      /* Find the link map for this module.  */
      size_t idx = GET_ADDR_MODULE;
      struct dtv_slotinfo_list *listp = GL(dl_tls_dtv_slotinfo_list);

      while (idx >= listp->len)
	{
	  idx -= listp->len;
	  listp = listp->next;
	}

      the_map = listp->slotinfo[idx].map;
    }

  /* Make sure that, if a dlopen running in parallel forces the
     variable into static storage, we'll wait until the address in the
     static TLS block is set up, and use that.  If we're undecided
     yet, make sure we make the decision holding the lock as well.  */
  if (__glibc_unlikely (the_map->l_tls_offset
			!= FORCED_DYNAMIC_TLS_OFFSET))
    {
      __rtld_lock_lock_recursive (GL(dl_load_lock));
      if (__glibc_likely (the_map->l_tls_offset == NO_TLS_OFFSET))
	{
	  the_map->l_tls_offset = FORCED_DYNAMIC_TLS_OFFSET;
	  __rtld_lock_unlock_recursive (GL(dl_load_lock));
	}
      else if (__glibc_likely (the_map->l_tls_offset
			       != FORCED_DYNAMIC_TLS_OFFSET))
	{
#if TLS_TCB_AT_TP
	  void *p = (char *) THREAD_SELF - the_map->l_tls_offset;
#elif TLS_DTV_AT_TP
	  void *p = (char *) THREAD_SELF + the_map->l_tls_offset + TLS_PRE_TCB_SIZE;
#else
# error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
#endif
	  __rtld_lock_unlock_recursive (GL(dl_load_lock));

	  dtv[GET_ADDR_MODULE].pointer.to_free = NULL;
	  dtv[GET_ADDR_MODULE].pointer.val = p;

	  return (char *) p + GET_ADDR_OFFSET;
	}
      else
	__rtld_lock_unlock_recursive (GL(dl_load_lock));
    }
  struct dtv_pointer result = allocate_and_init (the_map);
  dtv[GET_ADDR_MODULE].pointer = result;
  assert (result.to_free != NULL);

  return (char *) result.val + GET_ADDR_OFFSET;
}


static struct link_map *
__attribute_noinline__
update_get_addr (GET_ADDR_ARGS)
{
  struct link_map *the_map = _dl_update_slotinfo (GET_ADDR_MODULE);
  dtv_t *dtv = THREAD_DTV ();

  void *p = dtv[GET_ADDR_MODULE].pointer.val;

  if (__glibc_unlikely (p == TLS_DTV_UNALLOCATED))
    return tls_get_addr_tail (GET_ADDR_PARAM, dtv, the_map);

  return (void *) p + GET_ADDR_OFFSET;
}

/* For all machines that have a non-macro version of __tls_get_addr, we
   want to use rtld_hidden_proto/rtld_hidden_def in order to call the
   internal alias for __tls_get_addr from ld.so. This avoids a PLT entry
   in ld.so for __tls_get_addr.  */

#ifndef __tls_get_addr
extern void * __tls_get_addr (GET_ADDR_ARGS);
rtld_hidden_proto (__tls_get_addr)
rtld_hidden_def (__tls_get_addr)
#endif

/* The generic dynamic and local dynamic model cannot be used in
   statically linked applications.  */
void *
__tls_get_addr (GET_ADDR_ARGS)
{
  dtv_t *dtv = THREAD_DTV ();

  /* Update is needed if dtv[0].counter < the generation of the accessed
     module.  The global generation counter is used here as it is easier
     to check.  Synchronization for the relaxed MO access is guaranteed
     by user code, see CONCURRENCY NOTES in _dl_update_slotinfo.  */
  size_t gen = atomic_load_relaxed (&GL(dl_tls_generation));
  if (__glibc_unlikely (dtv[0].counter != gen))
    return update_get_addr (GET_ADDR_PARAM);

  void *p = dtv[GET_ADDR_MODULE].pointer.val;

  if (__glibc_unlikely (p == TLS_DTV_UNALLOCATED))
    return tls_get_addr_tail (GET_ADDR_PARAM, dtv, NULL);

  return (char *) p + GET_ADDR_OFFSET;
}
#endif


/* Look up the module's TLS block as for __tls_get_addr,
   but never touch anything.  Return null if it's not allocated yet.  */
void *
_dl_tls_get_addr_soft (struct link_map *l, struct pthread *pd)
{
  if (__glibc_unlikely (l->l_tls_modid == 0))
    /* This module has no TLS segment.  */
    return NULL;

  dtv_t *dtv = GET_DTV (TLS_TPADJ (pd));
  /* This may be called without holding the GL(dl_load_lock).  Reading
     arbitrary gen value is fine since this is best effort code.  */
  size_t gen = atomic_load_relaxed (&GL(dl_tls_generation));
  if (__glibc_unlikely (dtv[0].counter != gen))
    {
      /* This thread's DTV is not completely current,
	 but it might already cover this module.  */

      if (l->l_tls_modid >= dtv[-1].counter)
	/* Nope.  */
	return NULL;

      size_t idx = l->l_tls_modid;
      struct dtv_slotinfo_list *listp = GL(dl_tls_dtv_slotinfo_list);
      while (idx >= listp->len)
	{
	  idx -= listp->len;
	  listp = listp->next;
	}

      /* We've reached the slot for this module.
	 If its generation counter is higher than the DTV's,
	 this thread does not know about this module yet.  */
      if (dtv[0].counter < listp->slotinfo[idx].gen)
	return NULL;
    }

  void *data = dtv[l->l_tls_modid].pointer.val;
  if (__glibc_unlikely (data == TLS_DTV_UNALLOCATED))
    /* The DTV is current, but this thread has not yet needed
       to allocate this module's segment.  */
    data = NULL;

  return data;
}


void
_dl_add_to_slotinfo (struct link_map *l, bool do_add)
{
  /* Now that we know the object is loaded successfully add
     modules containing TLS data to the dtv info table.  We
     might have to increase its size.  */
  struct dtv_slotinfo_list *listp;
  struct dtv_slotinfo_list *prevp;
  size_t idx = l->l_tls_modid;

  /* Find the place in the dtv slotinfo list.  */
  listp = GL(dl_tls_dtv_slotinfo_list);
  prevp = NULL;		/* Needed to shut up gcc.  */
  do
    {
      /* Does it fit in the array of this list element?  */
      if (idx < listp->len)
	break;
      idx -= listp->len;
      prevp = listp;
      listp = listp->next;
    }
  while (listp != NULL);

  if (listp == NULL)
    {
      /* When we come here it means we have to add a new element
	 to the slotinfo list.  And the new module must be in
	 the first slot.  */
      assert (idx == 0);

      listp = (struct dtv_slotinfo_list *)
	malloc (sizeof (struct dtv_slotinfo_list)
		+ TLS_SLOTINFO_SURPLUS * sizeof (struct dtv_slotinfo));
      if (listp == NULL)
	{
	  /* We ran out of memory while resizing the dtv slotinfo list.  */
	  _dl_signal_error (ENOMEM, "dlopen", NULL, N_("\
cannot create TLS data structures"));
	}

      listp->len = TLS_SLOTINFO_SURPLUS;
      listp->next = NULL;
      memset (listp->slotinfo, '\0',
	      TLS_SLOTINFO_SURPLUS * sizeof (struct dtv_slotinfo));
      /* Synchronize with _dl_update_slotinfo.  */
      atomic_store_release (&prevp->next, listp);
    }

  /* Add the information into the slotinfo data structure.  */
  if (do_add)
    {
      /* Can be read concurrently.  See _dl_update_slotinfo.  */
      atomic_store_relaxed (&listp->slotinfo[idx].map, l);
      atomic_store_relaxed (&listp->slotinfo[idx].gen,
			    GL(dl_tls_generation) + 1);
    }
}

#if THREAD_GSCOPE_IN_TCB
static inline void __attribute__((always_inline))
init_one_static_tls (struct pthread *curp, struct link_map *map)
{
# if TLS_TCB_AT_TP
  void *dest = (char *) curp - map->l_tls_offset;
# elif TLS_DTV_AT_TP
  void *dest = (char *) curp + map->l_tls_offset + TLS_PRE_TCB_SIZE;
# else
#  error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
# endif

  /* Initialize the memory.  */
  memset (__mempcpy (dest, map->l_tls_initimage, map->l_tls_initimage_size),
	  '\0', map->l_tls_blocksize - map->l_tls_initimage_size);
}

void
_dl_init_static_tls (struct link_map *map)
{
  lll_lock (GL (dl_stack_cache_lock), LLL_PRIVATE);

  /* Iterate over the list with system-allocated threads first.  */
  list_t *runp;
  list_for_each (runp, &GL (dl_stack_used))
    init_one_static_tls (list_entry (runp, struct pthread, list), map);

  /* Now the list with threads using user-allocated stacks.  */
  list_for_each (runp, &GL (dl_stack_user))
    init_one_static_tls (list_entry (runp, struct pthread, list), map);

  lll_unlock (GL (dl_stack_cache_lock), LLL_PRIVATE);
}
#endif /* THREAD_GSCOPE_IN_TCB */
