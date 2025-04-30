/* Initialize x86 cache info.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

static const struct intel_02_cache_info
{
  unsigned char idx;
  unsigned char assoc;
  unsigned char linesize;
  unsigned char rel_name;
  unsigned int size;
} intel_02_known [] =
  {
#define M(sc) ((sc) - _SC_LEVEL1_ICACHE_SIZE)
    { 0x06,  4, 32, M(_SC_LEVEL1_ICACHE_SIZE),    8192 },
    { 0x08,  4, 32, M(_SC_LEVEL1_ICACHE_SIZE),   16384 },
    { 0x09,  4, 32, M(_SC_LEVEL1_ICACHE_SIZE),   32768 },
    { 0x0a,  2, 32, M(_SC_LEVEL1_DCACHE_SIZE),    8192 },
    { 0x0c,  4, 32, M(_SC_LEVEL1_DCACHE_SIZE),   16384 },
    { 0x0d,  4, 64, M(_SC_LEVEL1_DCACHE_SIZE),   16384 },
    { 0x0e,  6, 64, M(_SC_LEVEL1_DCACHE_SIZE),   24576 },
    { 0x21,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),   262144 },
    { 0x22,  4, 64, M(_SC_LEVEL3_CACHE_SIZE),   524288 },
    { 0x23,  8, 64, M(_SC_LEVEL3_CACHE_SIZE),  1048576 },
    { 0x25,  8, 64, M(_SC_LEVEL3_CACHE_SIZE),  2097152 },
    { 0x29,  8, 64, M(_SC_LEVEL3_CACHE_SIZE),  4194304 },
    { 0x2c,  8, 64, M(_SC_LEVEL1_DCACHE_SIZE),   32768 },
    { 0x30,  8, 64, M(_SC_LEVEL1_ICACHE_SIZE),   32768 },
    { 0x39,  4, 64, M(_SC_LEVEL2_CACHE_SIZE),   131072 },
    { 0x3a,  6, 64, M(_SC_LEVEL2_CACHE_SIZE),   196608 },
    { 0x3b,  2, 64, M(_SC_LEVEL2_CACHE_SIZE),   131072 },
    { 0x3c,  4, 64, M(_SC_LEVEL2_CACHE_SIZE),   262144 },
    { 0x3d,  6, 64, M(_SC_LEVEL2_CACHE_SIZE),   393216 },
    { 0x3e,  4, 64, M(_SC_LEVEL2_CACHE_SIZE),   524288 },
    { 0x3f,  2, 64, M(_SC_LEVEL2_CACHE_SIZE),   262144 },
    { 0x41,  4, 32, M(_SC_LEVEL2_CACHE_SIZE),   131072 },
    { 0x42,  4, 32, M(_SC_LEVEL2_CACHE_SIZE),   262144 },
    { 0x43,  4, 32, M(_SC_LEVEL2_CACHE_SIZE),   524288 },
    { 0x44,  4, 32, M(_SC_LEVEL2_CACHE_SIZE),  1048576 },
    { 0x45,  4, 32, M(_SC_LEVEL2_CACHE_SIZE),  2097152 },
    { 0x46,  4, 64, M(_SC_LEVEL3_CACHE_SIZE),  4194304 },
    { 0x47,  8, 64, M(_SC_LEVEL3_CACHE_SIZE),  8388608 },
    { 0x48, 12, 64, M(_SC_LEVEL2_CACHE_SIZE),  3145728 },
    { 0x49, 16, 64, M(_SC_LEVEL2_CACHE_SIZE),  4194304 },
    { 0x4a, 12, 64, M(_SC_LEVEL3_CACHE_SIZE),  6291456 },
    { 0x4b, 16, 64, M(_SC_LEVEL3_CACHE_SIZE),  8388608 },
    { 0x4c, 12, 64, M(_SC_LEVEL3_CACHE_SIZE), 12582912 },
    { 0x4d, 16, 64, M(_SC_LEVEL3_CACHE_SIZE), 16777216 },
    { 0x4e, 24, 64, M(_SC_LEVEL2_CACHE_SIZE),  6291456 },
    { 0x60,  8, 64, M(_SC_LEVEL1_DCACHE_SIZE),   16384 },
    { 0x66,  4, 64, M(_SC_LEVEL1_DCACHE_SIZE),    8192 },
    { 0x67,  4, 64, M(_SC_LEVEL1_DCACHE_SIZE),   16384 },
    { 0x68,  4, 64, M(_SC_LEVEL1_DCACHE_SIZE),   32768 },
    { 0x78,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),  1048576 },
    { 0x79,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),   131072 },
    { 0x7a,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),   262144 },
    { 0x7b,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),   524288 },
    { 0x7c,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),  1048576 },
    { 0x7d,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),  2097152 },
    { 0x7f,  2, 64, M(_SC_LEVEL2_CACHE_SIZE),   524288 },
    { 0x80,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),   524288 },
    { 0x82,  8, 32, M(_SC_LEVEL2_CACHE_SIZE),   262144 },
    { 0x83,  8, 32, M(_SC_LEVEL2_CACHE_SIZE),   524288 },
    { 0x84,  8, 32, M(_SC_LEVEL2_CACHE_SIZE),  1048576 },
    { 0x85,  8, 32, M(_SC_LEVEL2_CACHE_SIZE),  2097152 },
    { 0x86,  4, 64, M(_SC_LEVEL2_CACHE_SIZE),   524288 },
    { 0x87,  8, 64, M(_SC_LEVEL2_CACHE_SIZE),  1048576 },
    { 0xd0,  4, 64, M(_SC_LEVEL3_CACHE_SIZE),   524288 },
    { 0xd1,  4, 64, M(_SC_LEVEL3_CACHE_SIZE),  1048576 },
    { 0xd2,  4, 64, M(_SC_LEVEL3_CACHE_SIZE),  2097152 },
    { 0xd6,  8, 64, M(_SC_LEVEL3_CACHE_SIZE),  1048576 },
    { 0xd7,  8, 64, M(_SC_LEVEL3_CACHE_SIZE),  2097152 },
    { 0xd8,  8, 64, M(_SC_LEVEL3_CACHE_SIZE),  4194304 },
    { 0xdc, 12, 64, M(_SC_LEVEL3_CACHE_SIZE),  2097152 },
    { 0xdd, 12, 64, M(_SC_LEVEL3_CACHE_SIZE),  4194304 },
    { 0xde, 12, 64, M(_SC_LEVEL3_CACHE_SIZE),  8388608 },
    { 0xe2, 16, 64, M(_SC_LEVEL3_CACHE_SIZE),  2097152 },
    { 0xe3, 16, 64, M(_SC_LEVEL3_CACHE_SIZE),  4194304 },
    { 0xe4, 16, 64, M(_SC_LEVEL3_CACHE_SIZE),  8388608 },
    { 0xea, 24, 64, M(_SC_LEVEL3_CACHE_SIZE), 12582912 },
    { 0xeb, 24, 64, M(_SC_LEVEL3_CACHE_SIZE), 18874368 },
    { 0xec, 24, 64, M(_SC_LEVEL3_CACHE_SIZE), 25165824 },
  };

#define nintel_02_known (sizeof (intel_02_known) / sizeof (intel_02_known [0]))

static int
intel_02_known_compare (const void *p1, const void *p2)
{
  const struct intel_02_cache_info *i1;
  const struct intel_02_cache_info *i2;

  i1 = (const struct intel_02_cache_info *) p1;
  i2 = (const struct intel_02_cache_info *) p2;

  if (i1->idx == i2->idx)
    return 0;

  return i1->idx < i2->idx ? -1 : 1;
}


static long int
__attribute__ ((noinline))
intel_check_word (int name, unsigned int value, bool *has_level_2,
		  bool *no_level_2_or_3,
		  const struct cpu_features *cpu_features)
{
  if ((value & 0x80000000) != 0)
    /* The register value is reserved.  */
    return 0;

  /* Fold the name.  The _SC_ constants are always in the order SIZE,
     ASSOC, LINESIZE.  */
  int folded_rel_name = (M(name) / 3) * 3;

  while (value != 0)
    {
      unsigned int byte = value & 0xff;

      if (byte == 0x40)
	{
	  *no_level_2_or_3 = true;

	  if (folded_rel_name == M(_SC_LEVEL3_CACHE_SIZE))
	    /* No need to look further.  */
	    break;
	}
      else if (byte == 0xff)
	{
	  /* CPUID leaf 0x4 contains all the information.  We need to
	     iterate over it.  */
	  unsigned int eax;
	  unsigned int ebx;
	  unsigned int ecx;
	  unsigned int edx;

	  unsigned int round = 0;
	  while (1)
	    {
	      __cpuid_count (4, round, eax, ebx, ecx, edx);

	      enum { null = 0, data = 1, inst = 2, uni = 3 } type = eax & 0x1f;
	      if (type == null)
		/* That was the end.  */
		break;

	      unsigned int level = (eax >> 5) & 0x7;

	      if ((level == 1 && type == data
		   && folded_rel_name == M(_SC_LEVEL1_DCACHE_SIZE))
		  || (level == 1 && type == inst
		      && folded_rel_name == M(_SC_LEVEL1_ICACHE_SIZE))
		  || (level == 2 && folded_rel_name == M(_SC_LEVEL2_CACHE_SIZE))
		  || (level == 3 && folded_rel_name == M(_SC_LEVEL3_CACHE_SIZE))
		  || (level == 4 && folded_rel_name == M(_SC_LEVEL4_CACHE_SIZE)))
		{
		  unsigned int offset = M(name) - folded_rel_name;

		  if (offset == 0)
		    /* Cache size.  */
		    return (((ebx >> 22) + 1)
			    * (((ebx >> 12) & 0x3ff) + 1)
			    * ((ebx & 0xfff) + 1)
			    * (ecx + 1));
		  if (offset == 1)
		    return (ebx >> 22) + 1;

		  assert (offset == 2);
		  return (ebx & 0xfff) + 1;
		}

	      ++round;
	    }
	  /* There is no other cache information anywhere else.  */
	  break;
	}
      else
	{
	  if (byte == 0x49 && folded_rel_name == M(_SC_LEVEL3_CACHE_SIZE))
	    {
	      /* Intel reused this value.  For family 15, model 6 it
		 specifies the 3rd level cache.  Otherwise the 2nd
		 level cache.  */
	      unsigned int family = cpu_features->basic.family;
	      unsigned int model = cpu_features->basic.model;

	      if (family == 15 && model == 6)
		{
		  /* The level 3 cache is encoded for this model like
		     the level 2 cache is for other models.  Pretend
		     the caller asked for the level 2 cache.  */
		  name = (_SC_LEVEL2_CACHE_SIZE
			  + (name - _SC_LEVEL3_CACHE_SIZE));
		  folded_rel_name = M(_SC_LEVEL2_CACHE_SIZE);
		}
	    }

	  struct intel_02_cache_info *found;
	  struct intel_02_cache_info search;

	  search.idx = byte;
	  found = bsearch (&search, intel_02_known, nintel_02_known,
			   sizeof (intel_02_known[0]), intel_02_known_compare);
	  if (found != NULL)
	    {
	      if (found->rel_name == folded_rel_name)
		{
		  unsigned int offset = M(name) - folded_rel_name;

		  if (offset == 0)
		    /* Cache size.  */
		    return found->size;
		  if (offset == 1)
		    return found->assoc;

		  assert (offset == 2);
		  return found->linesize;
		}

	      if (found->rel_name == M(_SC_LEVEL2_CACHE_SIZE))
		*has_level_2 = true;
	    }
	}

      /* Next byte for the next round.  */
      value >>= 8;
    }

  /* Nothing found.  */
  return 0;
}


static long int __attribute__ ((noinline))
handle_intel (int name, const struct cpu_features *cpu_features)
{
  unsigned int maxidx = cpu_features->basic.max_cpuid;

  /* Return -1 for older CPUs.  */
  if (maxidx < 2)
    return -1;

  /* OK, we can use the CPUID instruction to get all info about the
     caches.  */
  unsigned int cnt = 0;
  unsigned int max = 1;
  long int result = 0;
  bool no_level_2_or_3 = false;
  bool has_level_2 = false;

  while (cnt++ < max)
    {
      unsigned int eax;
      unsigned int ebx;
      unsigned int ecx;
      unsigned int edx;
      __cpuid (2, eax, ebx, ecx, edx);

      /* The low byte of EAX in the first round contain the number of
	 rounds we have to make.  At least one, the one we are already
	 doing.  */
      if (cnt == 1)
	{
	  max = eax & 0xff;
	  eax &= 0xffffff00;
	}

      /* Process the individual registers' value.  */
      result = intel_check_word (name, eax, &has_level_2,
				 &no_level_2_or_3, cpu_features);
      if (result != 0)
	return result;

      result = intel_check_word (name, ebx, &has_level_2,
				 &no_level_2_or_3, cpu_features);
      if (result != 0)
	return result;

      result = intel_check_word (name, ecx, &has_level_2,
				 &no_level_2_or_3, cpu_features);
      if (result != 0)
	return result;

      result = intel_check_word (name, edx, &has_level_2,
				 &no_level_2_or_3, cpu_features);
      if (result != 0)
	return result;
    }

  if (name >= _SC_LEVEL2_CACHE_SIZE && name <= _SC_LEVEL3_CACHE_LINESIZE
      && no_level_2_or_3)
    return -1;

  return 0;
}


static long int __attribute__ ((noinline))
handle_amd (int name)
{
  unsigned int eax;
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;
  __cpuid (0x80000000, eax, ebx, ecx, edx);

  /* No level 4 cache (yet).  */
  if (name > _SC_LEVEL3_CACHE_LINESIZE)
    return 0;

  unsigned int fn = 0x80000005 + (name >= _SC_LEVEL2_CACHE_SIZE);
  if (eax < fn)
    return 0;

  __cpuid (fn, eax, ebx, ecx, edx);

  if (name < _SC_LEVEL1_DCACHE_SIZE)
    {
      name += _SC_LEVEL1_DCACHE_SIZE - _SC_LEVEL1_ICACHE_SIZE;
      ecx = edx;
    }

  switch (name)
    {
    case _SC_LEVEL1_DCACHE_SIZE:
      return (ecx >> 14) & 0x3fc00;

    case _SC_LEVEL1_DCACHE_ASSOC:
      ecx >>= 16;
      if ((ecx & 0xff) == 0xff)
	/* Fully associative.  */
	return (ecx << 2) & 0x3fc00;
      return ecx & 0xff;

    case _SC_LEVEL1_DCACHE_LINESIZE:
      return ecx & 0xff;

    case _SC_LEVEL2_CACHE_SIZE:
      return (ecx & 0xf000) == 0 ? 0 : (ecx >> 6) & 0x3fffc00;

    case _SC_LEVEL2_CACHE_ASSOC:
      switch ((ecx >> 12) & 0xf)
	{
	case 0:
	case 1:
	case 2:
	case 4:
	  return (ecx >> 12) & 0xf;
	case 6:
	  return 8;
	case 8:
	  return 16;
	case 10:
	  return 32;
	case 11:
	  return 48;
	case 12:
	  return 64;
	case 13:
	  return 96;
	case 14:
	  return 128;
	case 15:
	  return ((ecx >> 6) & 0x3fffc00) / (ecx & 0xff);
	default:
	  return 0;
	}
      /* NOTREACHED */

    case _SC_LEVEL2_CACHE_LINESIZE:
      return (ecx & 0xf000) == 0 ? 0 : ecx & 0xff;

    case _SC_LEVEL3_CACHE_SIZE:
      return (edx & 0xf000) == 0 ? 0 : (edx & 0x3ffc0000) << 1;

    case _SC_LEVEL3_CACHE_ASSOC:
      switch ((edx >> 12) & 0xf)
	{
	case 0:
	case 1:
	case 2:
	case 4:
	  return (edx >> 12) & 0xf;
	case 6:
	  return 8;
	case 8:
	  return 16;
	case 10:
	  return 32;
	case 11:
	  return 48;
	case 12:
	  return 64;
	case 13:
	  return 96;
	case 14:
	  return 128;
	case 15:
	  return ((edx & 0x3ffc0000) << 1) / (edx & 0xff);
	default:
	  return 0;
	}
      /* NOTREACHED */

    case _SC_LEVEL3_CACHE_LINESIZE:
      return (edx & 0xf000) == 0 ? 0 : edx & 0xff;

    default:
      assert (! "cannot happen");
    }
  return -1;
}


static long int __attribute__ ((noinline))
handle_zhaoxin (int name)
{
  unsigned int eax;
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;

  int folded_rel_name = (M(name) / 3) * 3;

  unsigned int round = 0;
  while (1)
    {
      __cpuid_count (4, round, eax, ebx, ecx, edx);

      enum { null = 0, data = 1, inst = 2, uni = 3 } type = eax & 0x1f;
      if (type == null)
        break;

      unsigned int level = (eax >> 5) & 0x7;

      if ((level == 1 && type == data
        && folded_rel_name == M(_SC_LEVEL1_DCACHE_SIZE))
        || (level == 1 && type == inst
            && folded_rel_name == M(_SC_LEVEL1_ICACHE_SIZE))
        || (level == 2 && folded_rel_name == M(_SC_LEVEL2_CACHE_SIZE))
        || (level == 3 && folded_rel_name == M(_SC_LEVEL3_CACHE_SIZE)))
        {
          unsigned int offset = M(name) - folded_rel_name;

          if (offset == 0)
            /* Cache size.  */
            return (((ebx >> 22) + 1)
                * (((ebx >> 12) & 0x3ff) + 1)
                * ((ebx & 0xfff) + 1)
                * (ecx + 1));
          if (offset == 1)
            return (ebx >> 22) + 1;

          assert (offset == 2);
          return (ebx & 0xfff) + 1;
        }

      ++round;
    }

  /* Nothing found.  */
  return 0;
}

static void
get_common_cache_info (long int *shared_ptr, unsigned int *threads_ptr,
                long int core)
{
  unsigned int eax;
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;

  /* Number of logical processors sharing L2 cache.  */
  int threads_l2;

  /* Number of logical processors sharing L3 cache.  */
  int threads_l3;

  const struct cpu_features *cpu_features = __get_cpu_features ();
  int max_cpuid = cpu_features->basic.max_cpuid;
  unsigned int family = cpu_features->basic.family;
  unsigned int model = cpu_features->basic.model;
  long int shared = *shared_ptr;
  unsigned int threads = *threads_ptr;
  bool inclusive_cache = true;
  bool support_count_mask = true;

  /* Try L3 first.  */
  unsigned int level = 3;

  if (cpu_features->basic.kind == arch_kind_zhaoxin && family == 6)
    support_count_mask = false;

  if (shared <= 0)
    {
      /* Try L2 otherwise.  */
      level  = 2;
      shared = core;
      threads_l2 = 0;
      threads_l3 = -1;
    }
  else
    {
      threads_l2 = 0;
      threads_l3 = 0;
    }

  /* A value of 0 for the HTT bit indicates there is only a single
     logical processor.  */
  if (HAS_CPU_FEATURE (HTT))
    {
      /* Figure out the number of logical threads that share the
         highest cache level.  */
      if (max_cpuid >= 4)
        {
          int i = 0;

          /* Query until cache level 2 and 3 are enumerated.  */
          int check = 0x1 | (threads_l3 == 0) << 1;
          do
            {
              __cpuid_count (4, i++, eax, ebx, ecx, edx);

              /* There seems to be a bug in at least some Pentium Ds
                 which sometimes fail to iterate all cache parameters.
                 Do not loop indefinitely here, stop in this case and
                 assume there is no such information.  */
              if (cpu_features->basic.kind == arch_kind_intel
                  && (eax & 0x1f) == 0 )
                goto intel_bug_no_cache_info;

              switch ((eax >> 5) & 0x7)
                {
                  default:
                    break;
                  case 2:
                    if ((check & 0x1))
                      {
                        /* Get maximum number of logical processors
                           sharing L2 cache.  */
                        threads_l2 = (eax >> 14) & 0x3ff;
                        check &= ~0x1;
                      }
                    break;
                  case 3:
                    if ((check & (0x1 << 1)))
                      {
                        /* Get maximum number of logical processors
                           sharing L3 cache.  */
                        threads_l3 = (eax >> 14) & 0x3ff;

                        /* Check if L2 and L3 caches are inclusive.  */
                        inclusive_cache = (edx & 0x2) != 0;
                        check &= ~(0x1 << 1);
                      }
                    break;
                }
            }
          while (check);

          /* If max_cpuid >= 11, THREADS_L2/THREADS_L3 are the maximum
             numbers of addressable IDs for logical processors sharing
             the cache, instead of the maximum number of threads
             sharing the cache.  */
          if (max_cpuid >= 11 && support_count_mask)
            {
              /* Find the number of logical processors shipped in
                 one core and apply count mask.  */
              i = 0;

              /* Count SMT only if there is L3 cache.  Always count
                 core if there is no L3 cache.  */
              int count = ((threads_l2 > 0 && level == 3)
                           | ((threads_l3 > 0
                               || (threads_l2 > 0 && level == 2)) << 1));

              while (count)
                {
                  __cpuid_count (11, i++, eax, ebx, ecx, edx);

                  int shipped = ebx & 0xff;
                  int type = ecx & 0xff00;
                  if (shipped == 0 || type == 0)
                    break;
                  else if (type == 0x100)
                    {
                      /* Count SMT.  */
                      if ((count & 0x1))
                        {
                          int count_mask;

                          /* Compute count mask.  */
                          asm ("bsr %1, %0"
                               : "=r" (count_mask) : "g" (threads_l2));
                          count_mask = ~(-1 << (count_mask + 1));
                          threads_l2 = (shipped - 1) & count_mask;
                          count &= ~0x1;
                        }
                    }
                  else if (type == 0x200)
                    {
                      /* Count core.  */
                      if ((count & (0x1 << 1)))
                        {
                          int count_mask;
                          int threads_core
                            = (level == 2 ? threads_l2 : threads_l3);

                          /* Compute count mask.  */
                          asm ("bsr %1, %0"
                               : "=r" (count_mask) : "g" (threads_core));
                          count_mask = ~(-1 << (count_mask + 1));
                          threads_core = (shipped - 1) & count_mask;
                          if (level == 2)
                            threads_l2 = threads_core;
                          else
                            threads_l3 = threads_core;
                          count &= ~(0x1 << 1);
                        }
                    }
                }
            }
          if (threads_l2 > 0)
            threads_l2 += 1;
          if (threads_l3 > 0)
            threads_l3 += 1;
          if (level == 2)
            {
              if (threads_l2)
                {
                  threads = threads_l2;
                  if (cpu_features->basic.kind == arch_kind_intel
                      && threads > 2
                      && family == 6)
                    switch (model)
                      {
                        case 0x37:
                        case 0x4a:
                        case 0x4d:
                        case 0x5a:
                        case 0x5d:
                          /* Silvermont has L2 cache shared by 2 cores.  */
                          threads = 2;
                          break;
                        default:
                          break;
                      }
                }
            }
          else if (threads_l3)
            threads = threads_l3;
        }
      else
        {
intel_bug_no_cache_info:
          /* Assume that all logical threads share the highest cache
             level.  */
          threads
            = ((cpu_features->features[CPUID_INDEX_1].cpuid.ebx >> 16)
	       & 0xff);
        }

        /* Cap usage of highest cache level to the number of supported
           threads.  */
        if (shared > 0 && threads > 0)
          shared /= threads;
    }

  /* Account for non-inclusive L2 and L3 caches.  */
  if (!inclusive_cache)
    {
      if (threads_l2 > 0)
        core /= threads_l2;
      shared += core;
    }

  *shared_ptr = shared;
  *threads_ptr = threads;
}

static void
dl_init_cacheinfo (struct cpu_features *cpu_features)
{
  /* Find out what brand of processor.  */
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;
  int max_cpuid_ex;
  long int data = -1;
  long int shared = -1;
  long int core = -1;
  unsigned int threads = 0;
  unsigned long int level1_icache_size = -1;
  unsigned long int level1_icache_linesize = -1;
  unsigned long int level1_dcache_size = -1;
  unsigned long int level1_dcache_assoc = -1;
  unsigned long int level1_dcache_linesize = -1;
  unsigned long int level2_cache_size = -1;
  unsigned long int level2_cache_assoc = -1;
  unsigned long int level2_cache_linesize = -1;
  unsigned long int level3_cache_size = -1;
  unsigned long int level3_cache_assoc = -1;
  unsigned long int level3_cache_linesize = -1;
  unsigned long int level4_cache_size = -1;

  if (cpu_features->basic.kind == arch_kind_intel)
    {
      data = handle_intel (_SC_LEVEL1_DCACHE_SIZE, cpu_features);
      core = handle_intel (_SC_LEVEL2_CACHE_SIZE, cpu_features);
      shared = handle_intel (_SC_LEVEL3_CACHE_SIZE, cpu_features);

      level1_icache_size
	= handle_intel (_SC_LEVEL1_ICACHE_SIZE, cpu_features);
      level1_icache_linesize
	= handle_intel (_SC_LEVEL1_ICACHE_LINESIZE, cpu_features);
      level1_dcache_size = data;
      level1_dcache_assoc
	= handle_intel (_SC_LEVEL1_DCACHE_ASSOC, cpu_features);
      level1_dcache_linesize
	= handle_intel (_SC_LEVEL1_DCACHE_LINESIZE, cpu_features);
      level2_cache_size = core;
      level2_cache_assoc
	= handle_intel (_SC_LEVEL2_CACHE_ASSOC, cpu_features);
      level2_cache_linesize
	= handle_intel (_SC_LEVEL2_CACHE_LINESIZE, cpu_features);
      level3_cache_size = shared;
      level3_cache_assoc
	= handle_intel (_SC_LEVEL3_CACHE_ASSOC, cpu_features);
      level3_cache_linesize
	= handle_intel (_SC_LEVEL3_CACHE_LINESIZE, cpu_features);
      level4_cache_size
	= handle_intel (_SC_LEVEL4_CACHE_SIZE, cpu_features);

      get_common_cache_info (&shared, &threads, core);
    }
  else if (cpu_features->basic.kind == arch_kind_zhaoxin)
    {
      data = handle_zhaoxin (_SC_LEVEL1_DCACHE_SIZE);
      core = handle_zhaoxin (_SC_LEVEL2_CACHE_SIZE);
      shared = handle_zhaoxin (_SC_LEVEL3_CACHE_SIZE);

      level1_icache_size = handle_zhaoxin (_SC_LEVEL1_ICACHE_SIZE);
      level1_icache_linesize = handle_zhaoxin (_SC_LEVEL1_ICACHE_LINESIZE);
      level1_dcache_size = data;
      level1_dcache_assoc = handle_zhaoxin (_SC_LEVEL1_DCACHE_ASSOC);
      level1_dcache_linesize = handle_zhaoxin (_SC_LEVEL1_DCACHE_LINESIZE);
      level2_cache_size = core;
      level2_cache_assoc = handle_zhaoxin (_SC_LEVEL2_CACHE_ASSOC);
      level2_cache_linesize = handle_zhaoxin (_SC_LEVEL2_CACHE_LINESIZE);
      level3_cache_size = shared;
      level3_cache_assoc = handle_zhaoxin (_SC_LEVEL3_CACHE_ASSOC);
      level3_cache_linesize = handle_zhaoxin (_SC_LEVEL3_CACHE_LINESIZE);

      get_common_cache_info (&shared, &threads, core);
    }
  else if (cpu_features->basic.kind == arch_kind_amd)
    {
      data  = handle_amd (_SC_LEVEL1_DCACHE_SIZE);
      core = handle_amd (_SC_LEVEL2_CACHE_SIZE);
      shared = handle_amd (_SC_LEVEL3_CACHE_SIZE);

      level1_icache_size = handle_amd (_SC_LEVEL1_ICACHE_SIZE);
      level1_icache_linesize = handle_amd (_SC_LEVEL1_ICACHE_LINESIZE);
      level1_dcache_size = data;
      level1_dcache_assoc = handle_amd (_SC_LEVEL1_DCACHE_ASSOC);
      level1_dcache_linesize = handle_amd (_SC_LEVEL1_DCACHE_LINESIZE);
      level2_cache_size = core;
      level2_cache_assoc = handle_amd (_SC_LEVEL2_CACHE_ASSOC);
      level2_cache_linesize = handle_amd (_SC_LEVEL2_CACHE_LINESIZE);
      level3_cache_size = shared;
      level3_cache_assoc = handle_amd (_SC_LEVEL3_CACHE_ASSOC);
      level3_cache_linesize = handle_amd (_SC_LEVEL3_CACHE_LINESIZE);

      /* Get maximum extended function. */
      __cpuid (0x80000000, max_cpuid_ex, ebx, ecx, edx);

      if (shared <= 0)
	/* No shared L3 cache.  All we have is the L2 cache.  */
	shared = core;
      else
	{
	  /* Figure out the number of logical threads that share L3.  */
	  if (max_cpuid_ex >= 0x80000008)
	    {
	      /* Get width of APIC ID.  */
	      __cpuid (0x80000008, max_cpuid_ex, ebx, ecx, edx);
	      threads = 1 << ((ecx >> 12) & 0x0f);
	    }

	  if (threads == 0 || cpu_features->basic.family >= 0x17)
	    {
	      /* If APIC ID width is not available, use logical
		 processor count.  */
	      __cpuid (0x00000001, max_cpuid_ex, ebx, ecx, edx);

	      if ((edx & (1 << 28)) != 0)
		threads = (ebx >> 16) & 0xff;
	    }

	  /* Cap usage of highest cache level to the number of
	     supported threads.  */
	  if (threads > 0)
	    shared /= threads;

	  /* Get shared cache per ccx for Zen architectures.  */
	  if (cpu_features->basic.family >= 0x17)
	    {
	      unsigned int eax;

	      /* Get number of threads share the L3 cache in CCX.  */
	      __cpuid_count (0x8000001D, 0x3, eax, ebx, ecx, edx);

	      unsigned int threads_per_ccx = ((eax >> 14) & 0xfff) + 1;
	      shared *= threads_per_ccx;
	    }
	  else
	    {
	      /* Account for exclusive L2 and L3 caches.  */
	      shared += core;
            }
	}
    }

  cpu_features->level1_icache_size = level1_icache_size;
  cpu_features->level1_icache_linesize = level1_icache_linesize;
  cpu_features->level1_dcache_size = level1_dcache_size;
  cpu_features->level1_dcache_assoc = level1_dcache_assoc;
  cpu_features->level1_dcache_linesize = level1_dcache_linesize;
  cpu_features->level2_cache_size = level2_cache_size;
  cpu_features->level2_cache_assoc = level2_cache_assoc;
  cpu_features->level2_cache_linesize = level2_cache_linesize;
  cpu_features->level3_cache_size = level3_cache_size;
  cpu_features->level3_cache_assoc = level3_cache_assoc;
  cpu_features->level3_cache_linesize = level3_cache_linesize;
  cpu_features->level4_cache_size = level4_cache_size;

  /* The default setting for the non_temporal threshold is 3/4 of one
     thread's share of the chip's cache. For most Intel and AMD processors
     with an initial release date between 2017 and 2020, a thread's typical
     share of the cache is from 500 KBytes to 2 MBytes. Using the 3/4
     threshold leaves 125 KBytes to 500 KBytes of the thread's data
     in cache after a maximum temporal copy, which will maintain
     in cache a reasonable portion of the thread's stack and other
     active data. If the threshold is set higher than one thread's
     share of the cache, it has a substantial risk of negatively
     impacting the performance of other threads running on the chip. */
  unsigned long int non_temporal_threshold = shared * 3 / 4;

#if HAVE_TUNABLES
  /* NB: The REP MOVSB threshold must be greater than VEC_SIZE * 8.  */
  unsigned int minimum_rep_movsb_threshold;
#endif
  /* NB: The default REP MOVSB threshold is 2048 * (VEC_SIZE / 16).  */
  unsigned int rep_movsb_threshold;
  if (CPU_FEATURE_USABLE_P (cpu_features, AVX512F)
      && !CPU_FEATURE_PREFERRED_P (cpu_features, Prefer_No_AVX512))
    {
      rep_movsb_threshold = 2048 * (64 / 16);
#if HAVE_TUNABLES
      minimum_rep_movsb_threshold = 64 * 8;
#endif
    }
  else if (CPU_FEATURE_PREFERRED_P (cpu_features,
				    AVX_Fast_Unaligned_Load))
    {
      rep_movsb_threshold = 2048 * (32 / 16);
#if HAVE_TUNABLES
      minimum_rep_movsb_threshold = 32 * 8;
#endif
    }
  else
    {
      rep_movsb_threshold = 2048 * (16 / 16);
#if HAVE_TUNABLES
      minimum_rep_movsb_threshold = 16 * 8;
#endif
    }
  /* NB: The default REP MOVSB threshold is 2112 on processors with fast
     short REP MOVSB (FSRM).  */
  if (CPU_FEATURE_USABLE_P (cpu_features, FSRM))
    rep_movsb_threshold = 2112;

  unsigned long int rep_movsb_stop_threshold;
  /* ERMS feature is implemented from AMD Zen3 architecture and it is
     performing poorly for data above L2 cache size. Henceforth, adding
     an upper bound threshold parameter to limit the usage of Enhanced
     REP MOVSB operations and setting its value to L2 cache size.  */
  if (cpu_features->basic.kind == arch_kind_amd)
    rep_movsb_stop_threshold = core;
  /* Setting the upper bound of ERMS to the computed value of
     non-temporal threshold for architectures other than AMD.  */
  else
    rep_movsb_stop_threshold = non_temporal_threshold;

  /* The default threshold to use Enhanced REP STOSB.  */
  unsigned long int rep_stosb_threshold = 2048;

#if HAVE_TUNABLES
  long int tunable_size;

  tunable_size = TUNABLE_GET (x86_data_cache_size, long int, NULL);
  /* NB: Ignore the default value 0.  */
  if (tunable_size != 0)
    data = tunable_size;

  tunable_size = TUNABLE_GET (x86_shared_cache_size, long int, NULL);
  /* NB: Ignore the default value 0.  */
  if (tunable_size != 0)
    shared = tunable_size;

  tunable_size = TUNABLE_GET (x86_non_temporal_threshold, long int, NULL);
  /* NB: Ignore the default value 0.  */
  if (tunable_size != 0)
    non_temporal_threshold = tunable_size;

  tunable_size = TUNABLE_GET (x86_rep_movsb_threshold, long int, NULL);
  if (tunable_size > minimum_rep_movsb_threshold)
    rep_movsb_threshold = tunable_size;

  /* NB: The default value of the x86_rep_stosb_threshold tunable is the
     same as the default value of __x86_rep_stosb_threshold and the
     minimum value is fixed.  */
  rep_stosb_threshold = TUNABLE_GET (x86_rep_stosb_threshold,
				     long int, NULL);

  TUNABLE_SET_WITH_BOUNDS (x86_data_cache_size, data, 0, SIZE_MAX);
  TUNABLE_SET_WITH_BOUNDS (x86_shared_cache_size, shared, 0, SIZE_MAX);
  TUNABLE_SET_WITH_BOUNDS (x86_non_temporal_threshold, non_temporal_threshold,
			   0, SIZE_MAX);
  TUNABLE_SET_WITH_BOUNDS (x86_rep_movsb_threshold, rep_movsb_threshold,
			   minimum_rep_movsb_threshold, SIZE_MAX);
  TUNABLE_SET_WITH_BOUNDS (x86_rep_stosb_threshold, rep_stosb_threshold, 1,
			   SIZE_MAX);
#endif

  cpu_features->data_cache_size = data;
  cpu_features->shared_cache_size = shared;
  cpu_features->non_temporal_threshold = non_temporal_threshold;
  cpu_features->rep_movsb_threshold = rep_movsb_threshold;
  cpu_features->rep_stosb_threshold = rep_stosb_threshold;
  cpu_features->rep_movsb_stop_threshold = rep_movsb_stop_threshold;
}
