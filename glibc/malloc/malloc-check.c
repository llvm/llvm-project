/* glibc.malloc.check implementation.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Wolfram Gloger <wg@malloc.de>, 2001.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#define __mremap mremap
#include "malloc.c"

/* When memory is tagged, the checking data is stored in the user part
   of the chunk.  We can't rely on the user not having modified the
   tags, so fetch the tag at each location before dereferencing
   it.  */
#define SAFE_CHAR_OFFSET(p,offset) \
  ((unsigned char *) tag_at (((unsigned char *) p) + offset))

/* A simple, standard set of debugging hooks.  Overhead is `only' one
   byte per chunk; still this will catch most cases of double frees or
   overruns.  The goal here is to avoid obscure crashes due to invalid
   usage, unlike in the MALLOC_DEBUG code. */

static unsigned char
magicbyte (const void *p)
{
  unsigned char magic;

  magic = (((uintptr_t) p >> 3) ^ ((uintptr_t) p >> 11)) & 0xFF;
  /* Do not return 1.  See the comment in mem2mem_check().  */
  if (magic == 1)
    ++magic;
  return magic;
}

/* Visualize the chunk as being partitioned into blocks of 255 bytes from the
   highest address of the chunk, downwards.  The end of each block tells
   us the size of that block, up to the actual size of the requested
   memory.  Our magic byte is right at the end of the requested size, so we
   must reach it with this iteration, otherwise we have witnessed a memory
   corruption.  */
static size_t
malloc_check_get_size (void *mem)
{
  size_t size;
  unsigned char c;
  mchunkptr p = mem2chunk (mem);
  unsigned char magic = magicbyte (p);

  for (size = CHUNK_HDR_SZ + memsize (p) - 1;
       (c = *SAFE_CHAR_OFFSET (p, size)) != magic;
       size -= c)
    {
      if (c <= 0 || size < (c + CHUNK_HDR_SZ))
	malloc_printerr ("malloc_check_get_size: memory corruption");
    }

  /* chunk2mem size.  */
  return size - CHUNK_HDR_SZ;
}

/* Instrument a chunk with overrun detector byte(s) and convert it
   into a user pointer with requested size req_sz. */

static void *
mem2mem_check (void *ptr, size_t req_sz)
{
  mchunkptr p;
  unsigned char *m_ptr = ptr;
  size_t max_sz, block_sz, i;
  unsigned char magic;

  if (!ptr)
    return ptr;

  p = mem2chunk (ptr);
  magic = magicbyte (p);
  max_sz = memsize (p);

  for (i = max_sz - 1; i > req_sz; i -= block_sz)
    {
      block_sz = MIN (i - req_sz, 0xff);
      /* Don't allow the magic byte to appear in the chain of length bytes.
         For the following to work, magicbyte cannot return 0x01.  */
      if (block_sz == magic)
        --block_sz;

      *SAFE_CHAR_OFFSET (m_ptr, i) = block_sz;
    }
  *SAFE_CHAR_OFFSET (m_ptr, req_sz) = magic;
  return (void *) m_ptr;
}

/* Convert a pointer to be free()d or realloc()ed to a valid chunk
   pointer.  If the provided pointer is not valid, return NULL. */

static mchunkptr
mem2chunk_check (void *mem, unsigned char **magic_p)
{
  mchunkptr p;
  INTERNAL_SIZE_T sz, c;
  unsigned char magic;

  if (!aligned_OK (mem))
    return NULL;

  p = mem2chunk (mem);
  sz = chunksize (p);
  magic = magicbyte (p);
  if (!chunk_is_mmapped (p))
    {
      /* Must be a chunk in conventional heap memory. */
      int contig = contiguous (&main_arena);
      if ((contig &&
           ((char *) p < mp_.sbrk_base ||
            ((char *) p + sz) >= (mp_.sbrk_base + main_arena.system_mem))) ||
          sz < MINSIZE || sz & MALLOC_ALIGN_MASK || !inuse (p) ||
          (!prev_inuse (p) && ((prev_size (p) & MALLOC_ALIGN_MASK) != 0 ||
                               (contig && (char *) prev_chunk (p) < mp_.sbrk_base) ||
                               next_chunk (prev_chunk (p)) != p)))
        return NULL;

      for (sz = CHUNK_HDR_SZ + memsize (p) - 1;
	   (c = *SAFE_CHAR_OFFSET (p, sz)) != magic;
	   sz -= c)
        {
          if (c == 0 || sz < (c + CHUNK_HDR_SZ))
            return NULL;
        }
    }
  else
    {
      unsigned long offset, page_mask = GLRO (dl_pagesize) - 1;

      /* mmap()ed chunks have MALLOC_ALIGNMENT or higher power-of-two
         alignment relative to the beginning of a page.  Check this
         first. */
      offset = (unsigned long) mem & page_mask;
      if ((offset != MALLOC_ALIGNMENT && offset != 0 && offset != 0x10 &&
           offset != 0x20 && offset != 0x40 && offset != 0x80 && offset != 0x100 &&
           offset != 0x200 && offset != 0x400 && offset != 0x800 && offset != 0x1000 &&
           offset < 0x2000) ||
          !chunk_is_mmapped (p) || prev_inuse (p) ||
          ((((unsigned long) p - prev_size (p)) & page_mask) != 0) ||
          ((prev_size (p) + sz) & page_mask) != 0)
        return NULL;

      for (sz = CHUNK_HDR_SZ + memsize (p) - 1;
	   (c = *SAFE_CHAR_OFFSET (p, sz)) != magic;
	   sz -= c)
        {
          if (c == 0 || sz < (c + CHUNK_HDR_SZ))
            return NULL;
        }
    }

  unsigned char* safe_p = SAFE_CHAR_OFFSET (p, sz);
  *safe_p ^= 0xFF;
  if (magic_p)
    *magic_p = safe_p;
  return p;
}

/* Check for corruption of the top chunk.  */
static void
top_check (void)
{
  mchunkptr t = top (&main_arena);

  if (t == initial_top (&main_arena) ||
      (!chunk_is_mmapped (t) &&
       chunksize (t) >= MINSIZE &&
       prev_inuse (t) &&
       (!contiguous (&main_arena) ||
        (char *) t + chunksize (t) == mp_.sbrk_base + main_arena.system_mem)))
    return;

  malloc_printerr ("malloc: top chunk is corrupt");
}

static void *
malloc_check (size_t sz)
{
  void *victim;
  size_t nb;

  if (__builtin_add_overflow (sz, 1, &nb))
    {
      __set_errno (ENOMEM);
      return NULL;
    }

  __libc_lock_lock (main_arena.mutex);
  top_check ();
  victim = _int_malloc (&main_arena, nb);
  __libc_lock_unlock (main_arena.mutex);
  return mem2mem_check (tag_new_usable (victim), sz);
}

static void
free_check (void *mem)
{
  mchunkptr p;

  if (!mem)
    return;

  int err = errno;

  /* Quickly check that the freed pointer matches the tag for the memory.
     This gives a useful double-free detection.  */
  if (__glibc_unlikely (mtag_enabled))
    *(volatile char *)mem;

  __libc_lock_lock (main_arena.mutex);
  p = mem2chunk_check (mem, NULL);
  if (!p)
    malloc_printerr ("free(): invalid pointer");
  if (chunk_is_mmapped (p))
    {
      __libc_lock_unlock (main_arena.mutex);
      munmap_chunk (p);
    }
  else
    {
      /* Mark the chunk as belonging to the library again.  */
      (void)tag_region (chunk2mem (p), memsize (p));
      _int_free (&main_arena, p, 1);
      __libc_lock_unlock (main_arena.mutex);
    }
  __set_errno (err);
}

static void *
realloc_check (void *oldmem, size_t bytes)
{
  INTERNAL_SIZE_T chnb;
  void *newmem = 0;
  unsigned char *magic_p;
  size_t rb;

  if (__builtin_add_overflow (bytes, 1, &rb))
    {
      __set_errno (ENOMEM);
      return NULL;
    }
  if (oldmem == 0)
    return malloc_check (bytes);

  if (bytes == 0)
    {
      free_check (oldmem);
      return NULL;
    }

  /* Quickly check that the freed pointer matches the tag for the memory.
     This gives a useful double-free detection.  */
  if (__glibc_unlikely (mtag_enabled))
    *(volatile char *)oldmem;

  __libc_lock_lock (main_arena.mutex);
  const mchunkptr oldp = mem2chunk_check (oldmem, &magic_p);
  __libc_lock_unlock (main_arena.mutex);
  if (!oldp)
    malloc_printerr ("realloc(): invalid pointer");
  const INTERNAL_SIZE_T oldsize = chunksize (oldp);

  if (!checked_request2size (rb, &chnb))
    {
      __set_errno (ENOMEM);
      goto invert;
    }

  __libc_lock_lock (main_arena.mutex);

  if (chunk_is_mmapped (oldp))
    {
#if HAVE_MREMAP
      mchunkptr newp = mremap_chunk (oldp, chnb);
      if (newp)
        newmem = chunk2mem_tag (newp);
      else
#endif
      {
	/* Note the extra SIZE_SZ overhead. */
        if (oldsize - SIZE_SZ >= chnb)
          newmem = oldmem; /* do nothing */
        else
          {
            /* Must alloc, copy, free. */
	    top_check ();
	    newmem = _int_malloc (&main_arena, rb);
            if (newmem)
              {
                memcpy (newmem, oldmem, oldsize - CHUNK_HDR_SZ);
                munmap_chunk (oldp);
              }
          }
      }
    }
  else
    {
      top_check ();
      newmem = _int_realloc (&main_arena, oldp, oldsize, chnb);
    }

  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* GCC 7 warns about magic_p may be used uninitialized.  But we never
     reach here if magic_p is uninitialized.  */
  DIAG_IGNORE_NEEDS_COMMENT (7, "-Wmaybe-uninitialized");
#endif
  /* mem2chunk_check changed the magic byte in the old chunk.
     If newmem is NULL, then the old chunk will still be used though,
     so we need to invert that change here.  */
invert:
  if (newmem == NULL)
    *magic_p ^= 0xFF;
  DIAG_POP_NEEDS_COMMENT;

  __libc_lock_unlock (main_arena.mutex);

  return mem2mem_check (tag_new_usable (newmem), bytes);
}

static void *
memalign_check (size_t alignment, size_t bytes)
{
  void *mem;

  if (alignment <= MALLOC_ALIGNMENT)
    return malloc_check (bytes);

  if (alignment < MINSIZE)
    alignment = MINSIZE;

  /* If the alignment is greater than SIZE_MAX / 2 + 1 it cannot be a
     power of 2 and will cause overflow in the check below.  */
  if (alignment > SIZE_MAX / 2 + 1)
    {
      __set_errno (EINVAL);
      return NULL;
    }

  /* Check for overflow.  */
  if (bytes > SIZE_MAX - alignment - MINSIZE)
    {
      __set_errno (ENOMEM);
      return NULL;
    }

  /* Make sure alignment is power of 2.  */
  if (!powerof2 (alignment))
    {
      size_t a = MALLOC_ALIGNMENT * 2;
      while (a < alignment)
        a <<= 1;
      alignment = a;
    }

  __libc_lock_lock (main_arena.mutex);
  top_check ();
  mem = _int_memalign (&main_arena, alignment, bytes + 1);
  __libc_lock_unlock (main_arena.mutex);
  return mem2mem_check (tag_new_usable (mem), bytes);
}

#if HAVE_TUNABLES
static void
TUNABLE_CALLBACK (set_mallopt_check) (tunable_val_t *valp)
{
  int32_t value = (int32_t) valp->numval;
  if (value != 0)
    __malloc_debug_enable (MALLOC_CHECK_HOOK);
}
#endif

static bool
initialize_malloc_check (void)
{
  /* This is the copy of the malloc initializer that we pulled in along with
     malloc-check.  This does not affect any of the libc malloc structures.  */
  ptmalloc_init ();
#if HAVE_TUNABLES
  TUNABLE_GET (check, int32_t, TUNABLE_CALLBACK (set_mallopt_check));
#else
  const char *s = secure_getenv ("MALLOC_CHECK_");
  if (s && s[0] != '\0' && s[0] != '0')
    __malloc_debug_enable (MALLOC_CHECK_HOOK);
#endif
  return __is_malloc_debug_enabled (MALLOC_CHECK_HOOK);
}
