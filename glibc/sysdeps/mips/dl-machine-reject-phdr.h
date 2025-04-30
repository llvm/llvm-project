/* Machine-dependent program header inspection for the ELF loader.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef _DL_MACHINE_REJECT_PHDR_H
#define _DL_MACHINE_REJECT_PHDR_H 1

#include <unistd.h>
#include <sys/prctl.h>

#if defined PR_GET_FP_MODE && defined PR_SET_FP_MODE
# define HAVE_PRCTL_FP_MODE 1
#else
# define HAVE_PRCTL_FP_MODE 0
#endif

/* Reject an object with a debug message.  */
#define REJECT(str, args...)						      \
  {									      \
    if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_LIBS))		      \
      _dl_debug_printf (str, ##args);					      \
    return true;							      \
  }

/* Search the program headers for the ABI Flags.  */

static inline const ElfW(Phdr) *
find_mips_abiflags (const ElfW(Phdr) *phdr, ElfW(Half) phnum)
{
  const ElfW(Phdr) *ph;

  for (ph = phdr; ph < &phdr[phnum]; ++ph)
    if (ph->p_type == PT_MIPS_ABIFLAGS)
      return ph;
  return NULL;
}

/* Cache the FP ABI value from the PT_MIPS_ABIFLAGS program header.  */

static bool
cached_fpabi_reject_phdr_p (struct link_map *l)
{
  if (l->l_mach.fpabi == 0)
    {
      const ElfW(Phdr) *ph = find_mips_abiflags (l->l_phdr, l->l_phnum);

      if (ph)
	{
	  Elf_MIPS_ABIFlags_v0 * mips_abiflags;
	  if (ph->p_filesz < sizeof (Elf_MIPS_ABIFlags_v0))
	    REJECT ("   %s: malformed PT_MIPS_ABIFLAGS found\n", l->l_name);

	  mips_abiflags = (Elf_MIPS_ABIFlags_v0 *) (l->l_addr + ph->p_vaddr);

	  if (__glibc_unlikely (mips_abiflags->flags2 != 0))
	    REJECT ("   %s: unknown MIPS.abiflags flags2: %u\n", l->l_name,
		    mips_abiflags->flags2);

	  l->l_mach.fpabi = mips_abiflags->fp_abi;
	  l->l_mach.odd_spreg = (mips_abiflags->flags1
				 & MIPS_AFL_FLAGS1_ODDSPREG) != 0;
	}
      else
	{
	  l->l_mach.fpabi = -1;
	  l->l_mach.odd_spreg = true;
	}
    }
  return false;
}

/* Return a description of the specified floating-point ABI.  */

static const char *
fpabi_string (int fpabi)
{
  switch (fpabi)
    {
    case Val_GNU_MIPS_ABI_FP_ANY:
      return "Hard or soft float";
    case Val_GNU_MIPS_ABI_FP_DOUBLE:
      return "Hard float (double precision)";
    case Val_GNU_MIPS_ABI_FP_SINGLE:
      return "Hard float (single precision)";
    case Val_GNU_MIPS_ABI_FP_SOFT:
      return "Soft float";
    case Val_GNU_MIPS_ABI_FP_OLD_64:
      return "Unsupported FP64";
    case Val_GNU_MIPS_ABI_FP_XX:
      return "Hard float (32-bit CPU, Any FPU)";
    case Val_GNU_MIPS_ABI_FP_64:
      return "Hard float (32-bit CPU, 64-bit FPU)";
    case Val_GNU_MIPS_ABI_FP_64A:
      return "Hard float compat (32-bit CPU, 64-bit FPU)";
    case -1:
      return "Double precision, single precision or soft float";
    default:
      return "Unknown FP ABI";
    }
}

/* A structure to describe the requirements of each FP ABI extension.
   Each field says whether the ABI can be executed in that mode.  The FR0 field
   is actually overloaded and means 'default' FR mode for the ABI.  I.e. For
   O32 it is FR0 and for N32/N64 it is actually FR1.  Since this logic is
   focussed on the intricacies of mode management for O32 we call the field
   FR0.  */

struct abi_req
{
  bool single;
  bool soft;
  bool fr0;
  bool fr1;
  bool fre;
};

/* FP ABI requirements for all Val_GNU_MIPS_ABI_FP_* values.  */

static const struct abi_req reqs[Val_GNU_MIPS_ABI_FP_MAX + 1] =
    {{true,  true,  true,  true,  true},  /* Any */
     {false, false, true,  false, true},  /* Double-float */
     {true,  false, false, false, false}, /* Single-float */
     {false, true,  false, false, false}, /* Soft-float */
     {false, false, false, false, false}, /* old-FP64 */
     {false, false, true,  true,  true},  /* FPXX */
     {false, false, false, true,  false}, /* FP64 */
     {false, false, false, true,  true}}; /* FP64A */

/* FP ABI requirements for objects without a PT_MIPS_ABIFLAGS segment.  */

static const struct abi_req none_req = { true, true, true, false, true };

/* Return true iff ELF program headers are incompatible with the running
   host.  This verifies that floating-point ABIs are compatible and
   re-configures the hardware mode if necessary.  This code handles both the
   DT_NEEDED libraries and the dlopen'ed libraries.  It also accounts for the
   impact of dlclose.  */

static bool __attribute_used__
elf_machine_reject_phdr_p (const ElfW(Phdr) *phdr, uint_fast16_t phnum,
			   const char *buf, size_t len, struct link_map *map,
			   int fd)
{
  const ElfW(Phdr) *ph = find_mips_abiflags (phdr, phnum);
  struct link_map *l;
  Lmid_t nsid;
  int in_abi = -1;
  struct abi_req in_req;
  Elf_MIPS_ABIFlags_v0 *mips_abiflags = NULL;
  bool perfect_match = false;
#if _MIPS_SIM == _ABIO32
  unsigned int cur_mode = -1;
# if HAVE_PRCTL_FP_MODE
  bool cannot_mode_switch = false;

  /* Get the current hardware mode.  */
  cur_mode = __prctl (PR_GET_FP_MODE);
# endif
#endif

  /* Read the attributes section.  */
  if (ph != NULL)
    {
      ElfW(Addr) size = ph->p_filesz;

      if (ph->p_offset + size <= len)
	mips_abiflags = (Elf_MIPS_ABIFlags_v0 *) (buf + ph->p_offset);
      else
	{
	  mips_abiflags = alloca (size);
	  __lseek (fd, ph->p_offset, SEEK_SET);
	  if (__libc_read (fd, (void *) mips_abiflags, size) != size)
	    REJECT ("   unable to read PT_MIPS_ABIFLAGS\n");
	}

      if (size < sizeof (Elf_MIPS_ABIFlags_v0))
	REJECT ("   contains malformed PT_MIPS_ABIFLAGS\n");

      if (__glibc_unlikely (mips_abiflags->flags2 != 0))
	REJECT ("   unknown MIPS.abiflags flags2: %u\n", mips_abiflags->flags2);

      in_abi = mips_abiflags->fp_abi;
    }

  /* ANY is compatible with anything.  */
  perfect_match |= (in_abi == Val_GNU_MIPS_ABI_FP_ANY);

  /* Unknown ABIs are rejected.  */
  if (in_abi != -1 && in_abi > Val_GNU_MIPS_ABI_FP_MAX)
    REJECT ("   uses unknown FP ABI: %u\n", in_abi);

  /* Obtain the initial requirements.  */
  in_req = (in_abi == -1) ? none_req : reqs[in_abi];

  /* Check that the new requirement does not conflict with any currently
     loaded object.  */
  for (nsid = 0; nsid < DL_NNS; ++nsid)
    for (l = GL(dl_ns)[nsid]._ns_loaded; l != NULL; l = l->l_next)
      {
	struct abi_req existing_req;

	if (cached_fpabi_reject_phdr_p (l))
	  return true;

#if _MIPS_SIM == _ABIO32
	/* A special case arises for O32 FP64 and FP64A where the kernel
	   pre-dates PT_MIPS_ABIFLAGS.  These ABIs will be blindly loaded even
	   if the hardware mode is unavailable or disabled.  In this
	   circumstance the prctl call to obtain the current mode will fail.
	   Detect this situation here and reject everything.  This will
	   effectively prevent dynamically linked applications from failing in
	   unusual ways but there is nothing we can do to help static
	   applications.  */
	if ((l->l_mach.fpabi == Val_GNU_MIPS_ABI_FP_64A
	     || l->l_mach.fpabi == Val_GNU_MIPS_ABI_FP_64)
	    && cur_mode == -1)
	  REJECT ("   found %s running in the wrong mode\n",
		  fpabi_string (l->l_mach.fpabi));
#endif

	/* Found a perfect match, success.  */
	perfect_match |= (in_abi == l->l_mach.fpabi);

	/* Unknown ABIs are rejected.  */
	if (l->l_mach.fpabi != -1 && l->l_mach.fpabi > Val_GNU_MIPS_ABI_FP_MAX)
	  REJECT ("   found unknown FP ABI: %u\n", l->l_mach.fpabi);

	existing_req = (l->l_mach.fpabi == -1 ? none_req
			: reqs[l->l_mach.fpabi]);

	/* Merge requirements.  */
	in_req.soft &= existing_req.soft;
	in_req.single &= existing_req.single;
	in_req.fr0 &= existing_req.fr0;
	in_req.fr1 &= existing_req.fr1;
	in_req.fre &= existing_req.fre;

	/* If there is at least one mode which is still usable then the new
	   object can be loaded.  */
	if (in_req.single || in_req.soft || in_req.fr1 || in_req.fr0
	    || in_req.fre)
	  {
#if _MIPS_SIM == _ABIO32 && HAVE_PRCTL_FP_MODE
	    /* Account for loaded ABIs which prohibit mode switching.  */
	    if (l->l_mach.fpabi == Val_GNU_MIPS_ABI_FP_XX)
	      cannot_mode_switch |= l->l_mach.odd_spreg;
#endif
	  }
	else
	  REJECT ("   uses %s, already loaded %s\n",
		  fpabi_string (in_abi),
		  fpabi_string (l->l_mach.fpabi));
      }

#if _MIPS_SIM == _ABIO32
  /* At this point we know that the newly loaded object is compatible with all
     existing objects but the hardware mode may not be correct.  */
  if ((in_req.fr1 || in_req.fre || in_req.fr0)
      && !perfect_match)
    {
      if (__glibc_unlikely (GLRO(dl_debug_mask) & DL_DEBUG_LIBS))
	_dl_debug_printf ("   needs %s%s mode\n", in_req.fr0 ? "FR0 or " : "",
			  (in_req.fre && !in_req.fr1) ? "FRE" : "FR1");

      /* If the PR_GET_FP_MODE is not supported then only FR0 is available.
	 If the overall requirements cannot be met by FR0 then reject the
	 object.  */
      if (cur_mode == -1)
	return !in_req.fr0;

# if HAVE_PRCTL_FP_MODE
      {
	unsigned int fr1_mode = PR_FP_MODE_FR;

	/* It is not possible to change the mode of a thread which may be
	   executing FPXX code with odd-singles.  If an FPXX object with
	   odd-singles is loaded then just check the current mode is OK. This
	   can be either the FR1 mode or FR0 if the requirements are met by
	   FR0.  */
	if (cannot_mode_switch)
	  return (!(in_req.fre && cur_mode == (PR_FP_MODE_FR | PR_FP_MODE_FRE))
		  && !(in_req.fr1 && cur_mode == PR_FP_MODE_FR)
		  && !(in_req.fr0 && cur_mode == 0));

	/* If the overall requirements can be satisfied by FRE but not FR1 then
	   fr1_mode must become FRE.  */
	if (in_req.fre && !in_req.fr1)
	  fr1_mode |= PR_FP_MODE_FRE;

	/* Set the new mode.  Use fr1_mode if the requirements cannot be met by
	   FR0.  */
	if (!in_req.fr0)
	  return __prctl (PR_SET_FP_MODE, fr1_mode) != 0;
	else if (__prctl (PR_SET_FP_MODE, /* fr0_mode */ 0) != 0)
	  {
	    /* Setting FR0 can validly fail on an R6 core so retry with the FR1
	       mode as a fall back.  */
	    if (errno != ENOTSUP)
	      return true;

	    return __prctl (PR_SET_FP_MODE, fr1_mode) != 0;
	  }
      }
# endif /* HAVE_PRCTL_FP_MODE */
    }
#endif /* _MIPS_SIM == _ABIO32 */

  return false;
}

#endif /* dl-machine-reject-phdr.h */
