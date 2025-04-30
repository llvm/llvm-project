/* x86 CPU feature tuning.
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

#if HAVE_TUNABLES
# define TUNABLE_NAMESPACE cpu
# include <stdbool.h>
# include <stdint.h>
# include <unistd.h>		/* Get STDOUT_FILENO for _dl_printf.  */
# include <elf/dl-tunables.h>
# include <string.h>
# include <cpu-features.h>
# include <ldsodefs.h>

/* We can't use IFUNC memcmp nor strlen in init_cpu_features from libc.a
   since IFUNC must be set up by init_cpu_features.  */
# if defined USE_MULTIARCH && !defined SHARED
#  ifdef __x86_64__
#   define DEFAULT_MEMCMP	__memcmp_sse2
#  else
#   define DEFAULT_MEMCMP	__memcmp_ia32
#  endif
extern __typeof (memcmp) DEFAULT_MEMCMP;
# else
#  define DEFAULT_MEMCMP	memcmp
# endif

# define CHECK_GLIBC_IFUNC_CPU_OFF(f, cpu_features, name, len)		\
  _Static_assert (sizeof (#name) - 1 == len, #name " != " #len);	\
  if (!DEFAULT_MEMCMP (f, #name, len))					\
    {									\
      CPU_FEATURE_UNSET (cpu_features, name)				\
      break;								\
    }

/* Disable a preferred feature NAME.  We don't enable a preferred feature
   which isn't available.  */
# define CHECK_GLIBC_IFUNC_PREFERRED_OFF(f, cpu_features, name, len)	\
  _Static_assert (sizeof (#name) - 1 == len, #name " != " #len);	\
  if (!DEFAULT_MEMCMP (f, #name, len))					\
    {									\
      cpu_features->preferred[index_arch_##name]			\
	&= ~bit_arch_##name;						\
      break;								\
    }

/* Enable/disable a preferred feature NAME.  */
# define CHECK_GLIBC_IFUNC_PREFERRED_BOTH(f, cpu_features, name,	\
					  disable, len)			\
  _Static_assert (sizeof (#name) - 1 == len, #name " != " #len);	\
  if (!DEFAULT_MEMCMP (f, #name, len))					\
    {									\
      if (disable)							\
	cpu_features->preferred[index_arch_##name] &= ~bit_arch_##name;	\
      else								\
	cpu_features->preferred[index_arch_##name] |= bit_arch_##name;	\
      break;								\
    }

/* Enable/disable a preferred feature NAME.  Enable a preferred feature
   only if the feature NEED is usable.  */
# define CHECK_GLIBC_IFUNC_PREFERRED_NEED_BOTH(f, cpu_features, name,	\
					       need, disable, len)	\
  _Static_assert (sizeof (#name) - 1 == len, #name " != " #len);	\
  if (!DEFAULT_MEMCMP (f, #name, len))					\
    {									\
      if (disable)							\
	cpu_features->preferred[index_arch_##name] &= ~bit_arch_##name;	\
      else if (CPU_FEATURE_USABLE_P (cpu_features, need))		\
	cpu_features->preferred[index_arch_##name] |= bit_arch_##name;	\
      break;								\
    }

attribute_hidden
void
TUNABLE_CALLBACK (set_hwcaps) (tunable_val_t *valp)
{
  /* The current IFUNC selection is based on microbenchmarks in glibc.
     It should give the best performance for most workloads.  But other
     choices may have better performance for a particular workload or on
     the hardware which wasn't available when the selection was made.
     The environment variable:

     GLIBC_TUNABLES=glibc.cpu.hwcaps=-xxx,yyy,-zzz,....

     can be used to enable CPU/ARCH feature yyy, disable CPU/ARCH feature
     yyy and zzz, where the feature name is case-sensitive and has to
     match the ones in cpu-features.h.  It can be used by glibc developers
     to tune for a new processor or override the IFUNC selection to
     improve performance for a particular workload.

     NOTE: the IFUNC selection may change over time.  Please check all
     multiarch implementations when experimenting.  */

  const char *p = valp->strval;
  struct cpu_features *cpu_features = &GLRO(dl_x86_cpu_features);
  size_t len;

  do
    {
      const char *c, *n;
      bool disable;
      size_t nl;

      for (c = p; *c != ','; c++)
	if (*c == '\0')
	  break;

      len = c - p;
      disable = *p == '-';
      if (disable)
	{
	  n = p + 1;
	  nl = len - 1;
	}
      else
	{
	  n = p;
	  nl = len;
	}
      switch (nl)
	{
	default:
	  break;
	case 3:
	  if (disable)
	    {
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX, 3);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, CX8, 3);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, FMA, 3);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, HTT, 3);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, IBT, 3);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, RTM, 3);
	    }
	  break;
	case 4:
	  if (disable)
	    {
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX2, 4);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, BMI1, 4);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, BMI2, 4);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, CMOV, 4);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, ERMS, 4);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, FMA4, 4);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, SSE2, 4);
	      CHECK_GLIBC_IFUNC_PREFERRED_OFF (n, cpu_features, I586, 4);
	      CHECK_GLIBC_IFUNC_PREFERRED_OFF (n, cpu_features, I686, 4);
	    }
	  break;
	case 5:
	  if (disable)
	    {
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, LZCNT, 5);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, MOVBE, 5);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, SHSTK, 5);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, SSSE3, 5);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, XSAVE, 5);
	    }
	  break;
	case 6:
	  if (disable)
	    {
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, POPCNT, 6);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, SSE4_1, 6);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, SSE4_2, 6);
	      if (!DEFAULT_MEMCMP (n, "XSAVEC", 6))
		{
		  /* Update xsave_state_size to XSAVE state size.  */
		  cpu_features->xsave_state_size
		    = cpu_features->xsave_state_full_size;
		  CPU_FEATURE_UNSET (cpu_features, XSAVEC);
		}
	    }
	  break;
	case 7:
	  if (disable)
	    {
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX512F, 7);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, OSXSAVE, 7);
	    }
	  break;
	case 8:
	  if (disable)
	    {
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX512CD, 8);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX512BW, 8);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX512DQ, 8);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX512ER, 8);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX512PF, 8);
	      CHECK_GLIBC_IFUNC_CPU_OFF (n, cpu_features, AVX512VL, 8);
	    }
	  CHECK_GLIBC_IFUNC_PREFERRED_BOTH (n, cpu_features, Slow_BSF,
					    disable, 8);
	  break;
	case 11:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_BOTH (n, cpu_features,
						Prefer_ERMS,
						disable, 11);
	      CHECK_GLIBC_IFUNC_PREFERRED_BOTH (n, cpu_features,
						Prefer_FSRM,
						disable, 11);
	      CHECK_GLIBC_IFUNC_PREFERRED_NEED_BOTH (n, cpu_features,
						     Slow_SSE4_2,
						     SSE4_2,
						     disable, 11);
	    }
	  break;
	case 15:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_BOTH (n, cpu_features,
						Fast_Rep_String,
						disable, 15);
	    }
	  break;
	case 16:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_NEED_BOTH
		(n, cpu_features, Prefer_No_AVX512, AVX512F,
		 disable, 16);
	    }
	  break;
	case 18:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_BOTH (n, cpu_features,
						Fast_Copy_Backward,
						disable, 18);
	      CHECK_GLIBC_IFUNC_PREFERRED_NEED_BOTH
		(n, cpu_features, Prefer_AVX2_STRCMP, AVX2, disable, 18);
	    }
	  break;
	case 19:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_BOTH (n, cpu_features,
						Fast_Unaligned_Load,
						disable, 19);
	      CHECK_GLIBC_IFUNC_PREFERRED_BOTH (n, cpu_features,
						Fast_Unaligned_Copy,
						disable, 19);
	    }
	  break;
	case 20:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_NEED_BOTH
		(n, cpu_features, Prefer_No_VZEROUPPER, AVX, disable,
		 20);
	    }
	  break;
	case 21:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_BOTH (n, cpu_features,
						Prefer_MAP_32BIT_EXEC,
						disable, 21);
	    }
	  break;
	case 23:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_NEED_BOTH
		(n, cpu_features, AVX_Fast_Unaligned_Load, AVX,
		 disable, 23);
	    }
	  break;
	case 24:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_NEED_BOTH
		(n, cpu_features, MathVec_Prefer_No_AVX512, AVX512F,
		 disable, 24);
	    }
	  break;
	case 26:
	    {
	      CHECK_GLIBC_IFUNC_PREFERRED_NEED_BOTH
		(n, cpu_features, Prefer_PMINUB_for_stringop, SSE2,
		 disable, 26);
	    }
	  break;
	}
      p += len + 1;
    }
  while (*p != '\0');
}

# if CET_ENABLED

attribute_hidden
void
TUNABLE_CALLBACK (set_x86_ibt) (tunable_val_t *valp)
{
  if (DEFAULT_MEMCMP (valp->strval, "on", sizeof ("on")) == 0)
    GL(dl_x86_feature_control).ibt = cet_always_on;
  else if (DEFAULT_MEMCMP (valp->strval, "off", sizeof ("off")) == 0)
    GL(dl_x86_feature_control).ibt = cet_always_off;
  else if (DEFAULT_MEMCMP (valp->strval, "permissive",
			   sizeof ("permissive")) == 0)
    GL(dl_x86_feature_control).ibt = cet_permissive;
}

attribute_hidden
void
TUNABLE_CALLBACK (set_x86_shstk) (tunable_val_t *valp)
{
  if (DEFAULT_MEMCMP (valp->strval, "on", sizeof ("on")) == 0)
    GL(dl_x86_feature_control).shstk = cet_always_on;
  else if (DEFAULT_MEMCMP (valp->strval, "off", sizeof ("off")) == 0)
    GL(dl_x86_feature_control).shstk = cet_always_off;
  else if (DEFAULT_MEMCMP (valp->strval, "permissive",
			   sizeof ("permissive")) == 0)
    GL(dl_x86_feature_control).shstk = cet_permissive;
}
# endif
#endif
