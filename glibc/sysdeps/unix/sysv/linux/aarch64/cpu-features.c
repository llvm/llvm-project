/* Initialize CPU feature data.  AArch64 version.
   This file is part of the GNU C Library.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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

#include <cpu-features.h>
#include <sys/auxv.h>
#include <elf/dl-hwcaps.h>
#include <sys/prctl.h>

#define DCZID_DZP_MASK (1 << 4)
#define DCZID_BS_MASK (0xf)

/* The maximal set of permitted tags that the MTE random tag generation
   instruction may use.  We exclude tag 0 because a) we want to reserve
   that for the libc heap structures and b) because it makes it easier
   to see when pointer have been correctly tagged.  */
#define MTE_ALLOWED_TAGS (0xfffe << PR_MTE_TAG_SHIFT)

#if HAVE_TUNABLES
struct cpu_list
{
  const char *name;
  uint64_t midr;
};

static struct cpu_list cpu_list[] = {
      {"falkor",	 0x510FC000},
      {"thunderxt88",	 0x430F0A10},
      {"thunderx2t99",   0x431F0AF0},
      {"thunderx2t99p1", 0x420F5160},
      {"phecda",	 0x680F0000},
      {"ares",		 0x411FD0C0},
      {"emag",		 0x503F0001},
      {"kunpeng920", 	 0x481FD010},
      {"a64fx",		 0x460F0010},
      {"generic", 	 0x0}
};

static uint64_t
get_midr_from_mcpu (const char *mcpu)
{
  for (int i = 0; i < sizeof (cpu_list) / sizeof (struct cpu_list); i++)
    if (strcmp (mcpu, cpu_list[i].name) == 0)
      return cpu_list[i].midr;

  return UINT64_MAX;
}
#endif

static inline void
init_cpu_features (struct cpu_features *cpu_features)
{
  register uint64_t midr = UINT64_MAX;

#if HAVE_TUNABLES
  /* Get the tunable override.  */
  const char *mcpu = TUNABLE_GET (glibc, cpu, name, const char *, NULL);
  if (mcpu != NULL)
    midr = get_midr_from_mcpu (mcpu);
#endif

  /* If there was no useful tunable override, query the MIDR if the kernel
     allows it.  */
  if (midr == UINT64_MAX)
    {
      if (GLRO (dl_hwcap) & HWCAP_CPUID)
	asm volatile ("mrs %0, midr_el1" : "=r"(midr));
      else
	midr = 0;
    }

  cpu_features->midr_el1 = midr;

  /* Check if ZVA is enabled.  */
  unsigned dczid;
  asm volatile ("mrs %0, dczid_el0" : "=r"(dczid));

  if ((dczid & DCZID_DZP_MASK) == 0)
    cpu_features->zva_size = 4 << (dczid & DCZID_BS_MASK);

  /* Check if BTI is supported.  */
  cpu_features->bti = GLRO (dl_hwcap2) & HWCAP2_BTI;

  /* Setup memory tagging support if the HW and kernel support it, and if
     the user has requested it.  */
  cpu_features->mte_state = 0;

#ifdef USE_MTAG
# if HAVE_TUNABLES
  int mte_state = TUNABLE_GET (glibc, mem, tagging, unsigned, 0);
  cpu_features->mte_state = (GLRO (dl_hwcap2) & HWCAP2_MTE) ? mte_state : 0;
  /* If we lack the MTE feature, disable the tunable, since it will
     otherwise cause instructions that won't run on this CPU to be used.  */
  TUNABLE_SET (glibc, mem, tagging, cpu_features->mte_state);
# endif

  if (cpu_features->mte_state & 2)
    __prctl (PR_SET_TAGGED_ADDR_CTRL,
	     (PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_SYNC | MTE_ALLOWED_TAGS),
	     0, 0, 0);
  else if (cpu_features->mte_state)
    __prctl (PR_SET_TAGGED_ADDR_CTRL,
	     (PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_ASYNC | MTE_ALLOWED_TAGS),
	     0, 0, 0);
#endif

  /* Check if SVE is supported.  */
  cpu_features->sve = GLRO (dl_hwcap) & HWCAP_SVE;
}
