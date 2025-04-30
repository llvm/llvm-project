/* Initialize CPU feature data.
   This file is part of the GNU C Library.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.

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

#include <dl-hwcap.h>
#include <libc-pointer-arith.h>
#include <get-isa-level.h>
#include <cacheinfo.h>
#include <dl-cacheinfo.h>
#include <dl-minsigstacksize.h>

#if HAVE_TUNABLES
extern void TUNABLE_CALLBACK (set_hwcaps) (tunable_val_t *)
  attribute_hidden;

# if CET_ENABLED
extern void TUNABLE_CALLBACK (set_x86_ibt) (tunable_val_t *)
  attribute_hidden;
extern void TUNABLE_CALLBACK (set_x86_shstk) (tunable_val_t *)
  attribute_hidden;
# endif
#endif

#if CET_ENABLED
# include <dl-cet.h>
#endif

static void
update_active (struct cpu_features *cpu_features)
{
  /* Copy the cpuid bits to active bits for CPU featuress whose usability
     in user space can be detected without additonal OS support.  */
  CPU_FEATURE_SET_ACTIVE (cpu_features, SSE3);
  CPU_FEATURE_SET_ACTIVE (cpu_features, PCLMULQDQ);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SSSE3);
  CPU_FEATURE_SET_ACTIVE (cpu_features, CMPXCHG16B);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SSE4_1);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SSE4_2);
  CPU_FEATURE_SET_ACTIVE (cpu_features, MOVBE);
  CPU_FEATURE_SET_ACTIVE (cpu_features, POPCNT);
  CPU_FEATURE_SET_ACTIVE (cpu_features, AES);
  CPU_FEATURE_SET_ACTIVE (cpu_features, OSXSAVE);
  CPU_FEATURE_SET_ACTIVE (cpu_features, TSC);
  CPU_FEATURE_SET_ACTIVE (cpu_features, CX8);
  CPU_FEATURE_SET_ACTIVE (cpu_features, CMOV);
  CPU_FEATURE_SET_ACTIVE (cpu_features, CLFSH);
  CPU_FEATURE_SET_ACTIVE (cpu_features, MMX);
  CPU_FEATURE_SET_ACTIVE (cpu_features, FXSR);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SSE);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SSE2);
  CPU_FEATURE_SET_ACTIVE (cpu_features, HTT);
  CPU_FEATURE_SET_ACTIVE (cpu_features, BMI1);
  CPU_FEATURE_SET_ACTIVE (cpu_features, HLE);
  CPU_FEATURE_SET_ACTIVE (cpu_features, BMI2);
  CPU_FEATURE_SET_ACTIVE (cpu_features, ERMS);
  CPU_FEATURE_SET_ACTIVE (cpu_features, RDSEED);
  CPU_FEATURE_SET_ACTIVE (cpu_features, ADX);
  CPU_FEATURE_SET_ACTIVE (cpu_features, CLFLUSHOPT);
  CPU_FEATURE_SET_ACTIVE (cpu_features, CLWB);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SHA);
  CPU_FEATURE_SET_ACTIVE (cpu_features, PREFETCHWT1);
  CPU_FEATURE_SET_ACTIVE (cpu_features, OSPKE);
  CPU_FEATURE_SET_ACTIVE (cpu_features, WAITPKG);
  CPU_FEATURE_SET_ACTIVE (cpu_features, GFNI);
  CPU_FEATURE_SET_ACTIVE (cpu_features, RDPID);
  CPU_FEATURE_SET_ACTIVE (cpu_features, RDRAND);
  CPU_FEATURE_SET_ACTIVE (cpu_features, CLDEMOTE);
  CPU_FEATURE_SET_ACTIVE (cpu_features, MOVDIRI);
  CPU_FEATURE_SET_ACTIVE (cpu_features, MOVDIR64B);
  CPU_FEATURE_SET_ACTIVE (cpu_features, FSRM);
  CPU_FEATURE_SET_ACTIVE (cpu_features, RTM_ALWAYS_ABORT);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SERIALIZE);
  CPU_FEATURE_SET_ACTIVE (cpu_features, TSXLDTRK);
  CPU_FEATURE_SET_ACTIVE (cpu_features, LAHF64_SAHF64);
  CPU_FEATURE_SET_ACTIVE (cpu_features, LZCNT);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SSE4A);
  CPU_FEATURE_SET_ACTIVE (cpu_features, PREFETCHW);
  CPU_FEATURE_SET_ACTIVE (cpu_features, TBM);
  CPU_FEATURE_SET_ACTIVE (cpu_features, RDTSCP);
  CPU_FEATURE_SET_ACTIVE (cpu_features, WBNOINVD);
  CPU_FEATURE_SET_ACTIVE (cpu_features, FZLRM);
  CPU_FEATURE_SET_ACTIVE (cpu_features, FSRS);
  CPU_FEATURE_SET_ACTIVE (cpu_features, FSRCS);
  CPU_FEATURE_SET_ACTIVE (cpu_features, PTWRITE);

  if (!CPU_FEATURES_CPU_P (cpu_features, RTM_ALWAYS_ABORT))
    CPU_FEATURE_SET_ACTIVE (cpu_features, RTM);

#if CET_ENABLED
  CPU_FEATURE_SET_ACTIVE (cpu_features, IBT);
  CPU_FEATURE_SET_ACTIVE (cpu_features, SHSTK);
#endif

  /* Can we call xgetbv?  */
  if (CPU_FEATURES_CPU_P (cpu_features, OSXSAVE))
    {
      unsigned int xcrlow;
      unsigned int xcrhigh;
      asm ("xgetbv" : "=a" (xcrlow), "=d" (xcrhigh) : "c" (0));
      /* Is YMM and XMM state usable?  */
      if ((xcrlow & (bit_YMM_state | bit_XMM_state))
	  == (bit_YMM_state | bit_XMM_state))
	{
	  /* Determine if AVX is usable.  */
	  if (CPU_FEATURES_CPU_P (cpu_features, AVX))
	    {
	      CPU_FEATURE_SET (cpu_features, AVX);
	      /* The following features depend on AVX being usable.  */
	      /* Determine if AVX2 is usable.  */
	      if (CPU_FEATURES_CPU_P (cpu_features, AVX2))
		{
		  CPU_FEATURE_SET (cpu_features, AVX2);

		  /* Unaligned load with 256-bit AVX registers are faster
		     on Intel/AMD processors with AVX2.  */
		  cpu_features->preferred[index_arch_AVX_Fast_Unaligned_Load]
		    |= bit_arch_AVX_Fast_Unaligned_Load;
		}
	      /* Determine if AVX-VNNI is usable.  */
	      CPU_FEATURE_SET_ACTIVE (cpu_features, AVX_VNNI);
	      /* Determine if FMA is usable.  */
	      CPU_FEATURE_SET_ACTIVE (cpu_features, FMA);
	      /* Determine if VAES is usable.  */
	      CPU_FEATURE_SET_ACTIVE (cpu_features, VAES);
	      /* Determine if VPCLMULQDQ is usable.  */
	      CPU_FEATURE_SET_ACTIVE (cpu_features, VPCLMULQDQ);
	      /* Determine if XOP is usable.  */
	      CPU_FEATURE_SET_ACTIVE (cpu_features, XOP);
	      /* Determine if F16C is usable.  */
	      CPU_FEATURE_SET_ACTIVE (cpu_features, F16C);
	    }

	  /* Check if OPMASK state, upper 256-bit of ZMM0-ZMM15 and
	     ZMM16-ZMM31 state are enabled.  */
	  if ((xcrlow & (bit_Opmask_state | bit_ZMM0_15_state
			 | bit_ZMM16_31_state))
	      == (bit_Opmask_state | bit_ZMM0_15_state | bit_ZMM16_31_state))
	    {
	      /* Determine if AVX512F is usable.  */
	      if (CPU_FEATURES_CPU_P (cpu_features, AVX512F))
		{
		  CPU_FEATURE_SET (cpu_features, AVX512F);
		  /* Determine if AVX512CD is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512CD);
		  /* Determine if AVX512ER is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512ER);
		  /* Determine if AVX512PF is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512PF);
		  /* Determine if AVX512VL is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512VL);
		  /* Determine if AVX512DQ is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512DQ);
		  /* Determine if AVX512BW is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512BW);
		  /* Determine if AVX512_4FMAPS is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_4FMAPS);
		  /* Determine if AVX512_4VNNIW is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_4VNNIW);
		  /* Determine if AVX512_BITALG is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_BITALG);
		  /* Determine if AVX512_IFMA is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_IFMA);
		  /* Determine if AVX512_VBMI is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_VBMI);
		  /* Determine if AVX512_VBMI2 is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_VBMI2);
		  /* Determine if is AVX512_VNNI usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_VNNI);
		  /* Determine if AVX512_VPOPCNTDQ is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features,
					  AVX512_VPOPCNTDQ);
		  /* Determine if AVX512_VP2INTERSECT is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features,
					  AVX512_VP2INTERSECT);
		  /* Determine if AVX512_BF16 is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_BF16);
		  /* Determine if AVX512_FP16 is usable.  */
		  CPU_FEATURE_SET_ACTIVE (cpu_features, AVX512_FP16);
		}
	    }
	}

      /* Are XTILECFG and XTILEDATA states usable?  */
      if ((xcrlow & (bit_XTILECFG_state | bit_XTILEDATA_state))
	  == (bit_XTILECFG_state | bit_XTILEDATA_state))
	{
	  /* Determine if AMX_BF16 is usable.  */
	  CPU_FEATURE_SET_ACTIVE (cpu_features, AMX_BF16);
	  /* Determine if AMX_TILE is usable.  */
	  CPU_FEATURE_SET_ACTIVE (cpu_features, AMX_TILE);
	  /* Determine if AMX_INT8 is usable.  */
	  CPU_FEATURE_SET_ACTIVE (cpu_features, AMX_INT8);
	}

      /* These features are usable only when OSXSAVE is enabled.  */
      CPU_FEATURE_SET (cpu_features, XSAVE);
      CPU_FEATURE_SET_ACTIVE (cpu_features, XSAVEOPT);
      CPU_FEATURE_SET_ACTIVE (cpu_features, XSAVEC);
      CPU_FEATURE_SET_ACTIVE (cpu_features, XGETBV_ECX_1);
      CPU_FEATURE_SET_ACTIVE (cpu_features, XFD);

      /* For _dl_runtime_resolve, set xsave_state_size to xsave area
	 size + integer register save size and align it to 64 bytes.  */
      if (cpu_features->basic.max_cpuid >= 0xd)
	{
	  unsigned int eax, ebx, ecx, edx;

	  __cpuid_count (0xd, 0, eax, ebx, ecx, edx);
	  if (ebx != 0)
	    {
	      unsigned int xsave_state_full_size
		= ALIGN_UP (ebx + STATE_SAVE_OFFSET, 64);

	      cpu_features->xsave_state_size
		= xsave_state_full_size;
	      cpu_features->xsave_state_full_size
		= xsave_state_full_size;

	      /* Check if XSAVEC is available.  */
	      if (CPU_FEATURES_CPU_P (cpu_features, XSAVEC))
		{
		  unsigned int xstate_comp_offsets[32];
		  unsigned int xstate_comp_sizes[32];
		  unsigned int i;

		  xstate_comp_offsets[0] = 0;
		  xstate_comp_offsets[1] = 160;
		  xstate_comp_offsets[2] = 576;
		  xstate_comp_sizes[0] = 160;
		  xstate_comp_sizes[1] = 256;

		  for (i = 2; i < 32; i++)
		    {
		      if ((STATE_SAVE_MASK & (1 << i)) != 0)
			{
			  __cpuid_count (0xd, i, eax, ebx, ecx, edx);
			  xstate_comp_sizes[i] = eax;
			}
		      else
			{
			  ecx = 0;
			  xstate_comp_sizes[i] = 0;
			}

		      if (i > 2)
			{
			  xstate_comp_offsets[i]
			    = (xstate_comp_offsets[i - 1]
			       + xstate_comp_sizes[i -1]);
			  if ((ecx & (1 << 1)) != 0)
			    xstate_comp_offsets[i]
			      = ALIGN_UP (xstate_comp_offsets[i], 64);
			}
		    }

		  /* Use XSAVEC.  */
		  unsigned int size
		    = xstate_comp_offsets[31] + xstate_comp_sizes[31];
		  if (size)
		    {
		      cpu_features->xsave_state_size
			= ALIGN_UP (size + STATE_SAVE_OFFSET, 64);
		      CPU_FEATURE_SET (cpu_features, XSAVEC);
		    }
		}
	    }
	}
    }

  /* Determine if PKU is usable.  */
  if (CPU_FEATURES_CPU_P (cpu_features, OSPKE))
    CPU_FEATURE_SET (cpu_features, PKU);

  /* Determine if Key Locker instructions are usable.  */
  if (CPU_FEATURES_CPU_P (cpu_features, AESKLE))
    {
      CPU_FEATURE_SET (cpu_features, AESKLE);
      CPU_FEATURE_SET_ACTIVE (cpu_features, KL);
      CPU_FEATURE_SET_ACTIVE (cpu_features, WIDE_KL);
    }

  cpu_features->isa_1 = get_isa_level (cpu_features);
}

static void
get_extended_indices (struct cpu_features *cpu_features)
{
  unsigned int eax, ebx, ecx, edx;
  __cpuid (0x80000000, eax, ebx, ecx, edx);
  if (eax >= 0x80000001)
    __cpuid (0x80000001,
	     cpu_features->features[CPUID_INDEX_80000001].cpuid.eax,
	     cpu_features->features[CPUID_INDEX_80000001].cpuid.ebx,
	     cpu_features->features[CPUID_INDEX_80000001].cpuid.ecx,
	     cpu_features->features[CPUID_INDEX_80000001].cpuid.edx);
  if (eax >= 0x80000007)
    __cpuid (0x80000007,
	     cpu_features->features[CPUID_INDEX_80000007].cpuid.eax,
	     cpu_features->features[CPUID_INDEX_80000007].cpuid.ebx,
	     cpu_features->features[CPUID_INDEX_80000007].cpuid.ecx,
	     cpu_features->features[CPUID_INDEX_80000007].cpuid.edx);
  if (eax >= 0x80000008)
    __cpuid (0x80000008,
	     cpu_features->features[CPUID_INDEX_80000008].cpuid.eax,
	     cpu_features->features[CPUID_INDEX_80000008].cpuid.ebx,
	     cpu_features->features[CPUID_INDEX_80000008].cpuid.ecx,
	     cpu_features->features[CPUID_INDEX_80000008].cpuid.edx);
}

static void
get_common_indices (struct cpu_features *cpu_features,
		    unsigned int *family, unsigned int *model,
		    unsigned int *extended_model, unsigned int *stepping)
{
  if (family)
    {
      unsigned int eax;
      __cpuid (1, eax,
	       cpu_features->features[CPUID_INDEX_1].cpuid.ebx,
	       cpu_features->features[CPUID_INDEX_1].cpuid.ecx,
	       cpu_features->features[CPUID_INDEX_1].cpuid.edx);
      cpu_features->features[CPUID_INDEX_1].cpuid.eax = eax;
      *family = (eax >> 8) & 0x0f;
      *model = (eax >> 4) & 0x0f;
      *extended_model = (eax >> 12) & 0xf0;
      *stepping = eax & 0x0f;
      if (*family == 0x0f)
	{
	  *family += (eax >> 20) & 0xff;
	  *model += *extended_model;
	}
    }

  if (cpu_features->basic.max_cpuid >= 7)
    {
      __cpuid_count (7, 0,
		     cpu_features->features[CPUID_INDEX_7].cpuid.eax,
		     cpu_features->features[CPUID_INDEX_7].cpuid.ebx,
		     cpu_features->features[CPUID_INDEX_7].cpuid.ecx,
		     cpu_features->features[CPUID_INDEX_7].cpuid.edx);
      __cpuid_count (7, 1,
		     cpu_features->features[CPUID_INDEX_7_ECX_1].cpuid.eax,
		     cpu_features->features[CPUID_INDEX_7_ECX_1].cpuid.ebx,
		     cpu_features->features[CPUID_INDEX_7_ECX_1].cpuid.ecx,
		     cpu_features->features[CPUID_INDEX_7_ECX_1].cpuid.edx);
    }

  if (cpu_features->basic.max_cpuid >= 0xd)
    __cpuid_count (0xd, 1,
		   cpu_features->features[CPUID_INDEX_D_ECX_1].cpuid.eax,
		   cpu_features->features[CPUID_INDEX_D_ECX_1].cpuid.ebx,
		   cpu_features->features[CPUID_INDEX_D_ECX_1].cpuid.ecx,
		   cpu_features->features[CPUID_INDEX_D_ECX_1].cpuid.edx);

  if (cpu_features->basic.max_cpuid >= 0x14)
    __cpuid_count (0x14, 0,
		   cpu_features->features[CPUID_INDEX_14_ECX_0].cpuid.eax,
		   cpu_features->features[CPUID_INDEX_14_ECX_0].cpuid.ebx,
		   cpu_features->features[CPUID_INDEX_14_ECX_0].cpuid.ecx,
		   cpu_features->features[CPUID_INDEX_14_ECX_0].cpuid.edx);

  if (cpu_features->basic.max_cpuid >= 0x19)
    __cpuid_count (0x19, 0,
		   cpu_features->features[CPUID_INDEX_19].cpuid.eax,
		   cpu_features->features[CPUID_INDEX_19].cpuid.ebx,
		   cpu_features->features[CPUID_INDEX_19].cpuid.ecx,
		   cpu_features->features[CPUID_INDEX_19].cpuid.edx);

  dl_check_minsigstacksize (cpu_features);
}

_Static_assert (((index_arch_Fast_Unaligned_Load
		  == index_arch_Fast_Unaligned_Copy)
		 && (index_arch_Fast_Unaligned_Load
		     == index_arch_Prefer_PMINUB_for_stringop)
		 && (index_arch_Fast_Unaligned_Load
		     == index_arch_Slow_SSE4_2)
		 && (index_arch_Fast_Unaligned_Load
		     == index_arch_Fast_Rep_String)
		 && (index_arch_Fast_Unaligned_Load
		     == index_arch_Fast_Copy_Backward)),
		"Incorrect index_arch_Fast_Unaligned_Load");

static inline void
init_cpu_features (struct cpu_features *cpu_features)
{
  unsigned int ebx, ecx, edx;
  unsigned int family = 0;
  unsigned int model = 0;
  unsigned int stepping = 0;
  enum cpu_features_kind kind;

#if !HAS_CPUID
  if (__get_cpuid_max (0, 0) == 0)
    {
      kind = arch_kind_other;
      goto no_cpuid;
    }
#endif

  __cpuid (0, cpu_features->basic.max_cpuid, ebx, ecx, edx);

  /* This spells out "GenuineIntel".  */
  if (ebx == 0x756e6547 && ecx == 0x6c65746e && edx == 0x49656e69)
    {
      unsigned int extended_model;

      kind = arch_kind_intel;

      get_common_indices (cpu_features, &family, &model, &extended_model,
			  &stepping);

      get_extended_indices (cpu_features);

      update_active (cpu_features);

      if (family == 0x06)
	{
	  model += extended_model;
	  switch (model)
	    {
	    case 0x1c:
	    case 0x26:
	      /* BSF is slow on Atom.  */
	      cpu_features->preferred[index_arch_Slow_BSF]
		|= bit_arch_Slow_BSF;
	      break;

	    case 0x57:
	      /* Knights Landing.  Enable Silvermont optimizations.  */

	    case 0x7a:
	      /* Unaligned load versions are faster than SSSE3
		 on Goldmont Plus.  */

	    case 0x5c:
	    case 0x5f:
	      /* Unaligned load versions are faster than SSSE3
		 on Goldmont.  */

	    case 0x4c:
	    case 0x5a:
	    case 0x75:
	      /* Airmont is a die shrink of Silvermont.  */

	    case 0x37:
	    case 0x4a:
	    case 0x4d:
	    case 0x5d:
	      /* Unaligned load versions are faster than SSSE3
		 on Silvermont.  */
	      cpu_features->preferred[index_arch_Fast_Unaligned_Load]
		|= (bit_arch_Fast_Unaligned_Load
		    | bit_arch_Fast_Unaligned_Copy
		    | bit_arch_Prefer_PMINUB_for_stringop
		    | bit_arch_Slow_SSE4_2);
	      break;

	    case 0x86:
	    case 0x96:
	    case 0x9c:
	      /* Enable rep string instructions, unaligned load, unaligned
	         copy, pminub and avoid SSE 4.2 on Tremont.  */
	      cpu_features->preferred[index_arch_Fast_Rep_String]
		|= (bit_arch_Fast_Rep_String
		    | bit_arch_Fast_Unaligned_Load
		    | bit_arch_Fast_Unaligned_Copy
		    | bit_arch_Prefer_PMINUB_for_stringop
		    | bit_arch_Slow_SSE4_2);
	      break;

	    default:
	      /* Unknown family 0x06 processors.  Assuming this is one
		 of Core i3/i5/i7 processors if AVX is available.  */
	      if (!CPU_FEATURES_CPU_P (cpu_features, AVX))
		break;
	      /* Fall through.  */

	    case 0x1a:
	    case 0x1e:
	    case 0x1f:
	    case 0x25:
	    case 0x2c:
	    case 0x2e:
	    case 0x2f:
	      /* Rep string instructions, unaligned load, unaligned copy,
		 and pminub are fast on Intel Core i3, i5 and i7.  */
	      cpu_features->preferred[index_arch_Fast_Rep_String]
		|= (bit_arch_Fast_Rep_String
		    | bit_arch_Fast_Unaligned_Load
		    | bit_arch_Fast_Unaligned_Copy
		    | bit_arch_Prefer_PMINUB_for_stringop);
	      break;
	    }

	 /* Disable TSX on some Haswell processors to avoid TSX on kernels that
	    weren't updated with the latest microcode package (which disables
	    broken feature by default).  */
	 switch (model)
	    {
	    case 0x3f:
	      /* Xeon E7 v3 with stepping >= 4 has working TSX.  */
	      if (stepping >= 4)
		break;
	      /* Fall through.  */
	    case 0x3c:
	    case 0x45:
	    case 0x46:
	      /* Disable Intel TSX on Haswell processors (except Xeon E7 v3
		 with stepping >= 4) to avoid TSX on kernels that weren't
		 updated with the latest microcode package (which disables
		 broken feature by default).  */
	      CPU_FEATURE_UNSET (cpu_features, RTM);
	      break;
	    }
	}


      /* Since AVX512ER is unique to Xeon Phi, set Prefer_No_VZEROUPPER
         if AVX512ER is available.  Don't use AVX512 to avoid lower CPU
	 frequency if AVX512ER isn't available.  */
      if (CPU_FEATURES_CPU_P (cpu_features, AVX512ER))
	cpu_features->preferred[index_arch_Prefer_No_VZEROUPPER]
	  |= bit_arch_Prefer_No_VZEROUPPER;
      else
	{
	  cpu_features->preferred[index_arch_Prefer_No_AVX512]
	    |= bit_arch_Prefer_No_AVX512;

	  /* Avoid RTM abort triggered by VZEROUPPER inside a
	     transactionally executing RTM region.  */
	  if (CPU_FEATURE_USABLE_P (cpu_features, RTM))
	    cpu_features->preferred[index_arch_Prefer_No_VZEROUPPER]
	      |= bit_arch_Prefer_No_VZEROUPPER;

	  /* Since to compare 2 32-byte strings, 256-bit EVEX strcmp
	     requires 2 loads, 3 VPCMPs and 2 KORDs while AVX2 strcmp
	     requires 1 load, 2 VPCMPEQs, 1 VPMINU and 1 VPMOVMSKB,
	     AVX2 strcmp is faster than EVEX strcmp.  */
	  if (CPU_FEATURE_USABLE_P (cpu_features, AVX2))
	    cpu_features->preferred[index_arch_Prefer_AVX2_STRCMP]
	      |= bit_arch_Prefer_AVX2_STRCMP;
	}

      /* Avoid avoid short distance REP MOVSB on processor with FSRM.  */
      if (CPU_FEATURES_CPU_P (cpu_features, FSRM))
	cpu_features->preferred[index_arch_Avoid_Short_Distance_REP_MOVSB]
	  |= bit_arch_Avoid_Short_Distance_REP_MOVSB;
    }
  /* This spells out "AuthenticAMD" or "HygonGenuine".  */
  else if ((ebx == 0x68747541 && ecx == 0x444d4163 && edx == 0x69746e65)
	   || (ebx == 0x6f677948 && ecx == 0x656e6975 && edx == 0x6e65476e))
    {
      unsigned int extended_model;

      kind = arch_kind_amd;

      get_common_indices (cpu_features, &family, &model, &extended_model,
			  &stepping);

      get_extended_indices (cpu_features);

      update_active (cpu_features);

      ecx = cpu_features->features[CPUID_INDEX_1].cpuid.ecx;

      if (CPU_FEATURE_USABLE_P (cpu_features, AVX))
	{
	  /* Since the FMA4 bit is in CPUID_INDEX_80000001 and
	     FMA4 requires AVX, determine if FMA4 is usable here.  */
	  CPU_FEATURE_SET_ACTIVE (cpu_features, FMA4);
	}

      if (family == 0x15)
	{
	  /* "Excavator"   */
	  if (model >= 0x60 && model <= 0x7f)
	  {
	    cpu_features->preferred[index_arch_Fast_Unaligned_Load]
	      |= (bit_arch_Fast_Unaligned_Load
		  | bit_arch_Fast_Copy_Backward);

	    /* Unaligned AVX loads are slower.*/
	    cpu_features->preferred[index_arch_AVX_Fast_Unaligned_Load]
	      &= ~bit_arch_AVX_Fast_Unaligned_Load;
	  }
	}
    }
  /* This spells out "CentaurHauls" or " Shanghai ".  */
  else if ((ebx == 0x746e6543 && ecx == 0x736c7561 && edx == 0x48727561)
	   || (ebx == 0x68532020 && ecx == 0x20206961 && edx == 0x68676e61))
    {
      unsigned int extended_model, stepping;

      kind = arch_kind_zhaoxin;

      get_common_indices (cpu_features, &family, &model, &extended_model,
			  &stepping);

      get_extended_indices (cpu_features);

      update_active (cpu_features);

      model += extended_model;
      if (family == 0x6)
        {
          if (model == 0xf || model == 0x19)
            {
	      CPU_FEATURE_UNSET (cpu_features, AVX);
	      CPU_FEATURE_UNSET (cpu_features, AVX2);

              cpu_features->preferred[index_arch_Slow_SSE4_2]
                |= bit_arch_Slow_SSE4_2;

	      cpu_features->preferred[index_arch_AVX_Fast_Unaligned_Load]
		&= ~bit_arch_AVX_Fast_Unaligned_Load;
            }
        }
      else if (family == 0x7)
        {
	  if (model == 0x1b)
	    {
	      CPU_FEATURE_UNSET (cpu_features, AVX);
	      CPU_FEATURE_UNSET (cpu_features, AVX2);

	      cpu_features->preferred[index_arch_Slow_SSE4_2]
		|= bit_arch_Slow_SSE4_2;

	      cpu_features->preferred[index_arch_AVX_Fast_Unaligned_Load]
		&= ~bit_arch_AVX_Fast_Unaligned_Load;
	    }
	  else if (model == 0x3b)
	    {
	      CPU_FEATURE_UNSET (cpu_features, AVX);
	      CPU_FEATURE_UNSET (cpu_features, AVX2);

	      cpu_features->preferred[index_arch_AVX_Fast_Unaligned_Load]
		&= ~bit_arch_AVX_Fast_Unaligned_Load;
	    }
	}
    }
  else
    {
      kind = arch_kind_other;
      get_common_indices (cpu_features, NULL, NULL, NULL, NULL);
      update_active (cpu_features);
    }

  /* Support i586 if CX8 is available.  */
  if (CPU_FEATURES_CPU_P (cpu_features, CX8))
    cpu_features->preferred[index_arch_I586] |= bit_arch_I586;

  /* Support i686 if CMOV is available.  */
  if (CPU_FEATURES_CPU_P (cpu_features, CMOV))
    cpu_features->preferred[index_arch_I686] |= bit_arch_I686;

#if !HAS_CPUID
no_cpuid:
#endif

  cpu_features->basic.kind = kind;
  cpu_features->basic.family = family;
  cpu_features->basic.model = model;
  cpu_features->basic.stepping = stepping;

  dl_init_cacheinfo (cpu_features);

#if HAVE_TUNABLES
  TUNABLE_GET (hwcaps, tunable_val_t *, TUNABLE_CALLBACK (set_hwcaps));

  bool disable_xsave_features = false;

  if (!CPU_FEATURE_USABLE_P (cpu_features, OSXSAVE))
    {
      /* These features are usable only if OSXSAVE is usable.  */
      CPU_FEATURE_UNSET (cpu_features, XSAVE);
      CPU_FEATURE_UNSET (cpu_features, XSAVEOPT);
      CPU_FEATURE_UNSET (cpu_features, XSAVEC);
      CPU_FEATURE_UNSET (cpu_features, XGETBV_ECX_1);
      CPU_FEATURE_UNSET (cpu_features, XFD);

      disable_xsave_features = true;
    }

  if (disable_xsave_features
      || (!CPU_FEATURE_USABLE_P (cpu_features, XSAVE)
	  && !CPU_FEATURE_USABLE_P (cpu_features, XSAVEC)))
    {
      /* Clear xsave_state_size if both XSAVE and XSAVEC aren't usable.  */
      cpu_features->xsave_state_size = 0;

      CPU_FEATURE_UNSET (cpu_features, AVX);
      CPU_FEATURE_UNSET (cpu_features, AVX2);
      CPU_FEATURE_UNSET (cpu_features, AVX_VNNI);
      CPU_FEATURE_UNSET (cpu_features, FMA);
      CPU_FEATURE_UNSET (cpu_features, VAES);
      CPU_FEATURE_UNSET (cpu_features, VPCLMULQDQ);
      CPU_FEATURE_UNSET (cpu_features, XOP);
      CPU_FEATURE_UNSET (cpu_features, F16C);
      CPU_FEATURE_UNSET (cpu_features, AVX512F);
      CPU_FEATURE_UNSET (cpu_features, AVX512CD);
      CPU_FEATURE_UNSET (cpu_features, AVX512ER);
      CPU_FEATURE_UNSET (cpu_features, AVX512PF);
      CPU_FEATURE_UNSET (cpu_features, AVX512VL);
      CPU_FEATURE_UNSET (cpu_features, AVX512DQ);
      CPU_FEATURE_UNSET (cpu_features, AVX512BW);
      CPU_FEATURE_UNSET (cpu_features, AVX512_4FMAPS);
      CPU_FEATURE_UNSET (cpu_features, AVX512_4VNNIW);
      CPU_FEATURE_UNSET (cpu_features, AVX512_BITALG);
      CPU_FEATURE_UNSET (cpu_features, AVX512_IFMA);
      CPU_FEATURE_UNSET (cpu_features, AVX512_VBMI);
      CPU_FEATURE_UNSET (cpu_features, AVX512_VBMI2);
      CPU_FEATURE_UNSET (cpu_features, AVX512_VNNI);
      CPU_FEATURE_UNSET (cpu_features, AVX512_VPOPCNTDQ);
      CPU_FEATURE_UNSET (cpu_features, AVX512_VP2INTERSECT);
      CPU_FEATURE_UNSET (cpu_features, AVX512_BF16);
      CPU_FEATURE_UNSET (cpu_features, AVX512_FP16);
      CPU_FEATURE_UNSET (cpu_features, AMX_BF16);
      CPU_FEATURE_UNSET (cpu_features, AMX_TILE);
      CPU_FEATURE_UNSET (cpu_features, AMX_INT8);

      CPU_FEATURE_UNSET (cpu_features, FMA4);
    }

#elif defined SHARED
  /* Reuse dl_platform, dl_hwcap and dl_hwcap_mask for x86.  The
     glibc.cpu.hwcap_mask tunable is initialized already, so no
     need to do this.  */
  GLRO(dl_hwcap_mask) = HWCAP_IMPORTANT;
#endif

#ifdef __x86_64__
  GLRO(dl_hwcap) = HWCAP_X86_64;
  if (cpu_features->basic.kind == arch_kind_intel)
    {
      const char *platform = NULL;

      if (CPU_FEATURE_USABLE_P (cpu_features, AVX512CD))
	{
	  if (CPU_FEATURE_USABLE_P (cpu_features, AVX512ER))
	    {
	      if (CPU_FEATURE_USABLE_P (cpu_features, AVX512PF))
		platform = "xeon_phi";
	    }
	  else
	    {
	      if (CPU_FEATURE_USABLE_P (cpu_features, AVX512BW)
		  && CPU_FEATURE_USABLE_P (cpu_features, AVX512DQ)
		  && CPU_FEATURE_USABLE_P (cpu_features, AVX512VL))
		GLRO(dl_hwcap) |= HWCAP_X86_AVX512_1;
	    }
	}

      if (platform == NULL
	  && CPU_FEATURE_USABLE_P (cpu_features, AVX2)
	  && CPU_FEATURE_USABLE_P (cpu_features, FMA)
	  && CPU_FEATURE_USABLE_P (cpu_features, BMI1)
	  && CPU_FEATURE_USABLE_P (cpu_features, BMI2)
	  && CPU_FEATURE_USABLE_P (cpu_features, LZCNT)
	  && CPU_FEATURE_USABLE_P (cpu_features, MOVBE)
	  && CPU_FEATURE_USABLE_P (cpu_features, POPCNT))
	platform = "haswell";

      if (platform != NULL)
	GLRO(dl_platform) = platform;
    }
#else
  GLRO(dl_hwcap) = 0;
  if (CPU_FEATURE_USABLE_P (cpu_features, SSE2))
    GLRO(dl_hwcap) |= HWCAP_X86_SSE2;

  if (CPU_FEATURES_ARCH_P (cpu_features, I686))
    GLRO(dl_platform) = "i686";
  else if (CPU_FEATURES_ARCH_P (cpu_features, I586))
    GLRO(dl_platform) = "i586";
#endif

#if CET_ENABLED
# if HAVE_TUNABLES
  TUNABLE_GET (x86_ibt, tunable_val_t *,
	       TUNABLE_CALLBACK (set_x86_ibt));
  TUNABLE_GET (x86_shstk, tunable_val_t *,
	       TUNABLE_CALLBACK (set_x86_shstk));
# endif

  /* Check CET status.  */
  unsigned int cet_status = get_cet_status ();

  if ((cet_status & GNU_PROPERTY_X86_FEATURE_1_IBT) == 0)
    CPU_FEATURE_UNSET (cpu_features, IBT)
  if ((cet_status & GNU_PROPERTY_X86_FEATURE_1_SHSTK) == 0)
    CPU_FEATURE_UNSET (cpu_features, SHSTK)

  if (cet_status)
    {
      GL(dl_x86_feature_1) = cet_status;

# ifndef SHARED
      /* Check if IBT and SHSTK are enabled by kernel.  */
      if ((cet_status & GNU_PROPERTY_X86_FEATURE_1_IBT)
	  || (cet_status & GNU_PROPERTY_X86_FEATURE_1_SHSTK))
	{
	  /* Disable IBT and/or SHSTK if they are enabled by kernel, but
	     disabled by environment variable:

	     GLIBC_TUNABLES=glibc.cpu.hwcaps=-IBT,-SHSTK
	   */
	  unsigned int cet_feature = 0;
	  if (!CPU_FEATURE_USABLE (IBT))
	    cet_feature |= GNU_PROPERTY_X86_FEATURE_1_IBT;
	  if (!CPU_FEATURE_USABLE (SHSTK))
	    cet_feature |= GNU_PROPERTY_X86_FEATURE_1_SHSTK;

	  if (cet_feature)
	    {
	      int res = dl_cet_disable_cet (cet_feature);

	      /* Clear the disabled bits in dl_x86_feature_1.  */
	      if (res == 0)
		GL(dl_x86_feature_1) &= ~cet_feature;
	    }

	  /* Lock CET if IBT or SHSTK is enabled in executable.  Don't
	     lock CET if IBT or SHSTK is enabled permissively.  */
	  if (GL(dl_x86_feature_control).ibt != cet_permissive
	      && GL(dl_x86_feature_control).shstk != cet_permissive)
	    dl_cet_lock_cet ();
	}
# endif
    }
#endif

#ifndef SHARED
  /* NB: In libc.a, call init_cacheinfo.  */
  init_cacheinfo ();
#endif
}
