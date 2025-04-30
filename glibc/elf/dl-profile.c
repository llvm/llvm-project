/* Profiling of shared libraries.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.
   Based on the BSD mcount implementation.

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
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <ldsodefs.h>
#include <sys/gmon.h>
#include <sys/gmon_out.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <atomic.h>
#include <not-cancel.h>

/* The LD_PROFILE feature has to be implemented different to the
   normal profiling using the gmon/ functions.  The problem is that an
   arbitrary amount of processes simulataneously can be run using
   profiling and all write the results in the same file.  To provide
   this mechanism one could implement a complicated mechanism to merge
   the content of two profiling runs or one could extend the file
   format to allow more than one data set.  For the second solution we
   would have the problem that the file can grow in size beyond any
   limit and both solutions have the problem that the concurrency of
   writing the results is a big problem.

   Another much simpler method is to use mmap to map the same file in
   all using programs and modify the data in the mmap'ed area and so
   also automatically on the disk.  Using the MAP_SHARED option of
   mmap(2) this can be done without big problems in more than one
   file.

   This approach is very different from the normal profiling.  We have
   to use the profiling data in exactly the way they are expected to
   be written to disk.  But the normal format used by gprof is not usable
   to do this.  It is optimized for size.  It writes the tags as single
   bytes but this means that the following 32/64 bit values are
   unaligned.

   Therefore we use a new format.  This will look like this

					0  1  2  3	<- byte is 32 bit word
	0000				g  m  o  n
	0004				*version*	<- GMON_SHOBJ_VERSION
	0008				00 00 00 00
	000c				00 00 00 00
	0010				00 00 00 00

	0014				*tag*		<- GMON_TAG_TIME_HIST
	0018				?? ?? ?? ??
					?? ?? ?? ??	<- 32/64 bit LowPC
	0018+A				?? ?? ?? ??
					?? ?? ?? ??	<- 32/64 bit HighPC
	0018+2*A			*histsize*
	001c+2*A			*profrate*
	0020+2*A			s  e  c  o
	0024+2*A			n  d  s  \0
	0028+2*A			\0 \0 \0 \0
	002c+2*A			\0 \0 \0
	002f+2*A			s

	0030+2*A			?? ?? ?? ??	<- Count data
	...				...
	0030+2*A+K			?? ?? ?? ??

	0030+2*A+K			*tag*		<- GMON_TAG_CG_ARC
	0034+2*A+K			*lastused*
	0038+2*A+K			?? ?? ?? ??
					?? ?? ?? ??	<- FromPC#1
	0038+3*A+K			?? ?? ?? ??
					?? ?? ?? ??	<- ToPC#1
	0038+4*A+K			?? ?? ?? ??	<- Count#1
	...				...		   ...
	0038+(2*(CN-1)+2)*A+(CN-1)*4+K	?? ?? ?? ??
					?? ?? ?? ??	<- FromPC#CGN
	0038+(2*(CN-1)+3)*A+(CN-1)*4+K	?? ?? ?? ??
					?? ?? ?? ??	<- ToPC#CGN
	0038+(2*CN+2)*A+(CN-1)*4+K	?? ?? ?? ??	<- Count#CGN

   We put (for now?) no basic block information in the file since this would
   introduce rase conditions among all the processes who want to write them.

   `K' is the number of count entries which is computed as

 		textsize / HISTFRACTION

   `CG' in the above table is the number of call graph arcs.  Normally,
   the table is sparse and the profiling code writes out only the those
   entries which are really used in the program run.  But since we must
   not extend this table (the profiling file) we'll keep them all here.
   So CN can be executed in advance as

		MINARCS <= textsize*(ARCDENSITY/100) <= MAXARCS

   Now the remaining question is: how to build the data structures we can
   work with from this data.  We need the from set and must associate the
   froms with all the associated tos.  We will do this by constructing this
   data structures at the program start.  To do this we'll simply visit all
   entries in the call graph table and add it to the appropriate list.  */

extern int __profile_frequency (void);
libc_hidden_proto (__profile_frequency)

/* We define a special type to address the elements of the arc table.
   This is basically the `gmon_cg_arc_record' format but it includes
   the room for the tag and it uses real types.  */
struct here_cg_arc_record
  {
    uintptr_t from_pc;
    uintptr_t self_pc;
    /* The count field is atomically incremented in _dl_mcount, which
       requires it to be properly aligned for its type, and for this
       alignment to be visible to the compiler.  The amount of data
       before an array of this structure is calculated as
       expected_size in _dl_start_profile.  Everything in that
       calculation is a multiple of 4 bytes (in the case of
       kcountsize, because it is derived from a subtraction of
       page-aligned values, and the corresponding calculation in
       __monstartup also ensures it is at least a multiple of the size
       of u_long), so all copies of this field do in fact have the
       appropriate alignment.  */
    uint32_t count __attribute__ ((aligned (__alignof__ (uint32_t))));
  } __attribute__ ((packed));

static struct here_cg_arc_record *data;

/* Nonzero if profiling is under way.  */
static int running;

/* This is the number of entry which have been incorporated in the toset.  */
static uint32_t narcs;
/* This is a pointer to the object representing the number of entries
   currently in the mmaped file.  At no point of time this has to be the
   same as NARCS.  If it is equal all entries from the file are in our
   lists.  */
#ifdef __clang__
static volatile unsigned long *narcsp;
#else
static volatile uint32_t *narcsp;
#endif

struct here_fromstruct
  {
    struct here_cg_arc_record volatile *here;
    uint16_t link;
  };

static volatile uint16_t *tos;

static struct here_fromstruct *froms;
static uint32_t fromlimit;
#ifdef __clang__
static volatile unsigned long fromidx;
#else
static volatile uint32_t fromidx;
#endif

static uintptr_t lowpc;
static size_t textsize;
static unsigned int log_hashfraction;



/* Set up profiling data to profile object desribed by MAP.  The output
   file is found (or created) in OUTPUT_DIR.  */
void
_dl_start_profile (void)
{
  char *filename;
  int fd;
  struct __stat64_t64 st;
  const ElfW(Phdr) *ph;
  ElfW(Addr) mapstart = ~((ElfW(Addr)) 0);
  ElfW(Addr) mapend = 0;
  char *hist, *cp;
  size_t idx;
  size_t tossize;
  size_t fromssize;
  uintptr_t highpc;
  uint16_t *kcount;
  size_t kcountsize;
  struct gmon_hdr *addr = NULL;
  off_t expected_size;
  /* See profil(2) where this is described.  */
  int s_scale;
#define SCALE_1_TO_1	0x10000L
  const char *errstr = NULL;

  /* Compute the size of the sections which contain program code.  */
  for (ph = GL(dl_profile_map)->l_phdr;
       ph < &GL(dl_profile_map)->l_phdr[GL(dl_profile_map)->l_phnum]; ++ph)
    if (ph->p_type == PT_LOAD && (ph->p_flags & PF_X))
      {
	ElfW(Addr) start = (ph->p_vaddr & ~(GLRO(dl_pagesize) - 1));
	ElfW(Addr) end = ((ph->p_vaddr + ph->p_memsz + GLRO(dl_pagesize) - 1)
			  & ~(GLRO(dl_pagesize) - 1));

	if (start < mapstart)
	  mapstart = start;
	if (end > mapend)
	  mapend = end;
      }

  /* Now we can compute the size of the profiling data.  This is done
     with the same formulars as in `monstartup' (see gmon.c).  */
  running = 0;
  lowpc = ROUNDDOWN (mapstart + GL(dl_profile_map)->l_addr,
		     HISTFRACTION * sizeof (HISTCOUNTER));
  highpc = ROUNDUP (mapend + GL(dl_profile_map)->l_addr,
		    HISTFRACTION * sizeof (HISTCOUNTER));
  textsize = highpc - lowpc;
  kcountsize = textsize / HISTFRACTION;
  if ((HASHFRACTION & (HASHFRACTION - 1)) == 0)
    {
      /* If HASHFRACTION is a power of two, mcount can use shifting
	 instead of integer division.  Precompute shift amount.

	 This is a constant but the compiler cannot compile the
	 expression away since the __ffs implementation is not known
	 to the compiler.  Help the compiler by precomputing the
	 usual cases.  */
      assert (HASHFRACTION == 2);

      if (sizeof (*froms) == 8)
	log_hashfraction = 4;
      else if (sizeof (*froms) == 16)
	log_hashfraction = 5;
      else
	log_hashfraction = __ffs (HASHFRACTION * sizeof (*froms)) - 1;
    }
  else
    log_hashfraction = -1;
  tossize = textsize / HASHFRACTION;
  fromlimit = textsize * ARCDENSITY / 100;
  if (fromlimit < MINARCS)
    fromlimit = MINARCS;
  if (fromlimit > MAXARCS)
    fromlimit = MAXARCS;
  fromssize = fromlimit * sizeof (struct here_fromstruct);

  expected_size = (sizeof (struct gmon_hdr)
		   + 4 + sizeof (struct gmon_hist_hdr) + kcountsize
		   + 4 + 4 + fromssize * sizeof (struct here_cg_arc_record));

  /* Create the gmon_hdr we expect or write.  */
  struct real_gmon_hdr
  {
    char cookie[4];
    int32_t version;
    char spare[3 * 4];
  } gmon_hdr;
  if (sizeof (gmon_hdr) != sizeof (struct gmon_hdr)
      || (offsetof (struct real_gmon_hdr, cookie)
	  != offsetof (struct gmon_hdr, cookie))
      || (offsetof (struct real_gmon_hdr, version)
	  != offsetof (struct gmon_hdr, version)))
    abort ();

  memcpy (&gmon_hdr.cookie[0], GMON_MAGIC, sizeof (gmon_hdr.cookie));
  gmon_hdr.version = GMON_SHOBJ_VERSION;
  memset (gmon_hdr.spare, '\0', sizeof (gmon_hdr.spare));

  /* Create the hist_hdr we expect or write.  */
  struct real_gmon_hist_hdr
  {
    char *low_pc;
    char *high_pc;
    int32_t hist_size;
    int32_t prof_rate;
    char dimen[15];
    char dimen_abbrev;
  } hist_hdr;
  if (sizeof (hist_hdr) != sizeof (struct gmon_hist_hdr)
      || (offsetof (struct real_gmon_hist_hdr, low_pc)
	  != offsetof (struct gmon_hist_hdr, low_pc))
      || (offsetof (struct real_gmon_hist_hdr, high_pc)
	  != offsetof (struct gmon_hist_hdr, high_pc))
      || (offsetof (struct real_gmon_hist_hdr, hist_size)
	  != offsetof (struct gmon_hist_hdr, hist_size))
      || (offsetof (struct real_gmon_hist_hdr, prof_rate)
	  != offsetof (struct gmon_hist_hdr, prof_rate))
      || (offsetof (struct real_gmon_hist_hdr, dimen)
	  != offsetof (struct gmon_hist_hdr, dimen))
      || (offsetof (struct real_gmon_hist_hdr, dimen_abbrev)
	  != offsetof (struct gmon_hist_hdr, dimen_abbrev)))
    abort ();

  hist_hdr.low_pc = (char *) mapstart;
  hist_hdr.high_pc = (char *) mapend;
  hist_hdr.hist_size = kcountsize / sizeof (HISTCOUNTER);
  hist_hdr.prof_rate = __profile_frequency ();
  if (sizeof (hist_hdr.dimen) >= sizeof ("seconds"))
    {
      memcpy (hist_hdr.dimen, "seconds", sizeof ("seconds"));
      memset (hist_hdr.dimen + sizeof ("seconds"), '\0',
	      sizeof (hist_hdr.dimen) - sizeof ("seconds"));
    }
  else
    strncpy (hist_hdr.dimen, "seconds", sizeof (hist_hdr.dimen));
  hist_hdr.dimen_abbrev = 's';

  /* First determine the output name.  We write in the directory
     OUTPUT_DIR and the name is composed from the shared objects
     soname (or the file name) and the ending ".profile".  */
  filename = (char *) alloca (strlen (GLRO(dl_profile_output)) + 1
			      + strlen (GLRO(dl_profile)) + sizeof ".profile");
  cp = __stpcpy (filename, GLRO(dl_profile_output));
  *cp++ = '/';
  __stpcpy (__stpcpy (cp, GLRO(dl_profile)), ".profile");

  fd = __open64_nocancel (filename, O_RDWR|O_CREAT|O_NOFOLLOW, DEFFILEMODE);
  if (fd == -1)
    {
      char buf[400];
      int errnum;

      /* We cannot write the profiling data so don't do anything.  */
      errstr = "%s: cannot open file: %s\n";
    print_error:
      errnum = errno;
      if (fd != -1)
	__close_nocancel (fd);
      _dl_error_printf (errstr, filename,
			__strerror_r (errnum, buf, sizeof buf));
      return;
    }

  if (__fstat64_time64 (fd, &st) < 0 || !S_ISREG (st.st_mode))
    {
      /* Not stat'able or not a regular file => don't use it.  */
      errstr = "%s: cannot stat file: %s\n";
      goto print_error;
    }

  /* Test the size.  If it does not match what we expect from the size
     values in the map MAP we don't use it and warn the user.  */
  if (st.st_size == 0)
    {
      /* We have to create the file.  */
      char buf[GLRO(dl_pagesize)];

      memset (buf, '\0', GLRO(dl_pagesize));

      if (__lseek (fd, expected_size & ~(GLRO(dl_pagesize) - 1), SEEK_SET) == -1)
	{
	cannot_create:
	  errstr = "%s: cannot create file: %s\n";
	  goto print_error;
	}

      if (TEMP_FAILURE_RETRY
	  (__write_nocancel (fd, buf, (expected_size & (GLRO(dl_pagesize) - 1))))
	  < 0)
	goto cannot_create;
    }
  else if (st.st_size != expected_size)
    {
      __close_nocancel (fd);
    wrong_format:

      if (addr != NULL)
	__munmap ((void *) addr, expected_size);

      _dl_error_printf ("%s: file is no correct profile data file for `%s'\n",
			filename, GLRO(dl_profile));
      return;
    }

  addr = (struct gmon_hdr *) __mmap (NULL, expected_size, PROT_READ|PROT_WRITE,
				     MAP_SHARED|MAP_FILE, fd, 0);
  if (addr == (struct gmon_hdr *) MAP_FAILED)
    {
      errstr = "%s: cannot map file: %s\n";
      goto print_error;
    }

  /* We don't need the file descriptor anymore.  */
  __close_nocancel (fd);

  /* Pointer to data after the header.  */
  hist = (char *) (addr + 1);
  kcount = (uint16_t *) ((char *) hist + sizeof (uint32_t)
			 + sizeof (struct gmon_hist_hdr));

  /* Compute pointer to array of the arc information.  */
#ifdef __clang__
  narcsp = (unsigned long *) ((char *) kcount + kcountsize + sizeof (uint32_t));
#else
  narcsp = (uint32_t *) ((char *) kcount + kcountsize + sizeof (uint32_t));
#endif
  data = (struct here_cg_arc_record *) ((char *) narcsp + sizeof (uint32_t));

  if (st.st_size == 0)
    {
      /* Create the signature.  */
      memcpy (addr, &gmon_hdr, sizeof (struct gmon_hdr));

      *(uint32_t *) hist = GMON_TAG_TIME_HIST;
      memcpy (hist + sizeof (uint32_t), &hist_hdr,
	      sizeof (struct gmon_hist_hdr));

      narcsp[-1] = GMON_TAG_CG_ARC;
    }
  else
    {
      /* Test the signature in the file.  */
      if (memcmp (addr, &gmon_hdr, sizeof (struct gmon_hdr)) != 0
	  || *(uint32_t *) hist != GMON_TAG_TIME_HIST
	  || memcmp (hist + sizeof (uint32_t), &hist_hdr,
		     sizeof (struct gmon_hist_hdr)) != 0
	  || narcsp[-1] != GMON_TAG_CG_ARC)
	goto wrong_format;
    }

  /* Allocate memory for the froms data and the pointer to the tos records.  */
  tos = (uint16_t *) calloc (tossize + fromssize, 1);
  if (tos == NULL)
    {
      __munmap ((void *) addr, expected_size);
      _dl_fatal_printf ("Out of memory while initializing profiler\n");
      /* NOTREACHED */
    }

  froms = (struct here_fromstruct *) ((char *) tos + tossize);
  fromidx = 0;

  /* Now we have to process all the arc count entries.  BTW: it is
     not critical whether the *NARCSP value changes meanwhile.  Before
     we enter a new entry in to toset we will check that everything is
     available in TOS.  This happens in _dl_mcount.

     Loading the entries in reverse order should help to get the most
     frequently used entries at the front of the list.  */
  for (idx = narcs = MIN (*narcsp, fromlimit); idx > 0; )
    {
      size_t to_index;
      size_t newfromidx;
      --idx;
      to_index = (data[idx].self_pc / (HASHFRACTION * sizeof (*tos)));
      newfromidx = fromidx++;
      froms[newfromidx].here = &data[idx];
      froms[newfromidx].link = tos[to_index];
      tos[to_index] = newfromidx;
    }

  /* Setup counting data.  */
  if (kcountsize < highpc - lowpc)
    {
#if 0
      s_scale = ((double) kcountsize / (highpc - lowpc)) * SCALE_1_TO_1;
#else
      size_t range = highpc - lowpc;
      size_t quot = range / kcountsize;

      if (quot >= SCALE_1_TO_1)
	s_scale = 1;
      else if (quot >= SCALE_1_TO_1 / 256)
	s_scale = SCALE_1_TO_1 / quot;
      else if (range > ULONG_MAX / 256)
	s_scale = (SCALE_1_TO_1 * 256) / (range / (kcountsize / 256));
      else
	s_scale = (SCALE_1_TO_1 * 256) / ((range * 256) / kcountsize);
#endif
    }
  else
    s_scale = SCALE_1_TO_1;

  /* Start the profiler.  */
  __profil ((void *) kcount, kcountsize, lowpc, s_scale);

  /* Turn on profiling.  */
  running = 1;
}


void
_dl_mcount (ElfW(Addr) frompc, ElfW(Addr) selfpc)
{
  volatile uint16_t *topcindex;
  size_t i, fromindex;
  struct here_fromstruct *fromp;

  if (! running)
    return;

  /* Compute relative addresses.  The shared object can be loaded at
     any address.  The value of frompc could be anything.  We cannot
     restrict it in any way, just set to a fixed value (0) in case it
     is outside the allowed range.  These calls show up as calls from
     <external> in the gprof output.  */
  frompc -= lowpc;
  if (frompc >= textsize)
    frompc = 0;
  selfpc -= lowpc;
  if (selfpc >= textsize)
    goto done;

  /* Getting here we now have to find out whether the location was
     already used.  If yes we are lucky and only have to increment a
     counter (this also has to be atomic).  If the entry is new things
     are getting complicated...  */

  /* Avoid integer divide if possible.  */
  if ((HASHFRACTION & (HASHFRACTION - 1)) == 0)
    i = selfpc >> log_hashfraction;
  else
    i = selfpc / (HASHFRACTION * sizeof (*tos));

  topcindex = &tos[i];
  fromindex = *topcindex;

  if (fromindex == 0)
    goto check_new_or_add;

  fromp = &froms[fromindex];

  /* We have to look through the chain of arcs whether there is already
     an entry for our arc.  */
  while (fromp->here->from_pc != frompc)
    {
      if (fromp->link != 0)
	do
	  fromp = &froms[fromp->link];
	while (fromp->link != 0 && fromp->here->from_pc != frompc);

      if (fromp->here->from_pc != frompc)
	{
	  topcindex = &fromp->link;

	check_new_or_add:
	  /* Our entry is not among the entries we read so far from the
	     data file.  Now see whether we have to update the list.  */
	  while (narcs != *narcsp && narcs < fromlimit)
	    {
	      size_t to_index;
	      size_t newfromidx;
	      to_index = (data[narcs].self_pc
			  / (HASHFRACTION * sizeof (*tos)));
	      newfromidx = catomic_exchange_and_add (&fromidx, 1) + 1;
	      froms[newfromidx].here = &data[narcs];
	      froms[newfromidx].link = tos[to_index];
	      tos[to_index] = newfromidx;
	      catomic_increment (&narcs);
	    }

	  /* If we still have no entry stop searching and insert.  */
	  if (*topcindex == 0)
	    {
	      uint_fast32_t newarc = catomic_exchange_and_add (narcsp, 1);

	      /* In rare cases it could happen that all entries in FROMS are
		 occupied.  So we cannot count this anymore.  */
	      if (newarc >= fromlimit)
		goto done;

	      *topcindex = catomic_exchange_and_add (&fromidx, 1) + 1;
	      fromp = &froms[*topcindex];

	      fromp->here = &data[newarc];
	      data[newarc].from_pc = frompc;
	      data[newarc].self_pc = selfpc;
	      data[newarc].count = 0;
	      fromp->link = 0;
	      catomic_increment (&narcs);

	      break;
	    }

	  fromp = &froms[*topcindex];
	}
      else
	/* Found in.  */
	break;
    }

  /* Increment the counter.  */
  catomic_increment (&fromp->here->count);

 done:
  ;
}
rtld_hidden_def (_dl_mcount)
