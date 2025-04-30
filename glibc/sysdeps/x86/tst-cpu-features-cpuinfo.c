/* Test CPU feature data against /proc/cpuinfo.
   This file is part of the GNU C Library.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static char *cpu_flags;

/* Search for flags in /proc/cpuinfo and store line
   in cpu_flags.  */
void
get_cpuinfo (void)
{
  FILE *f;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  f = fopen ("/proc/cpuinfo", "r");
  if (f == NULL)
    {
      printf ("cannot open /proc/cpuinfo\n");
      exit (1);
    }

  while ((read = getline (&line, &len, f)) != -1)
    {
      if (strncmp (line, "flags", 5) == 0)
       {
         cpu_flags = strdup (line);
         break;
       }
    }
  fclose (f);
  free (line);
}

int
check_proc (const char *proc_name, const char *search_name, int flag,
	    int active, const char *name)
{
  int found = 0;

  printf ("Checking %s:\n", name);
  printf ("  %s: %d\n", name, flag);
  char *str = strstr (cpu_flags, search_name);
  if (str == NULL)
    {
      /* If searching for " XXX " failed, try " XXX\n".  */
      size_t len = strlen (search_name);
      char buffer[80];
      if (len >= sizeof buffer)
	abort ();
      memcpy (buffer, search_name, len + 1);
      buffer[len - 1] = '\n';
      str = strstr (cpu_flags, buffer);
    }
  if (str != NULL)
    found = 1;
  printf ("  cpuinfo (%s): %d\n", proc_name, found);

  if (found != flag)
    {
      if (found || active)
	printf (" *** failure ***\n");
      else
	{
	  printf (" *** missing in /proc/cpuinfo ***\n");
	  return 0;
	}
    }

  return (found != flag);
}

#define CHECK_PROC(str, name) \
  check_proc (#str, " "#str" ", HAS_CPU_FEATURE (name), \
	      CPU_FEATURE_USABLE (name), \
	      "HAS_CPU_FEATURE (" #name ")")

static int
do_test (int argc, char **argv)
{
  int fails = 0;
  const struct cpu_features *cpu_features = __get_cpu_features ();

  get_cpuinfo ();
  fails += CHECK_PROC (acpi, ACPI);
  fails += CHECK_PROC (adx, ADX);
  fails += CHECK_PROC (apic, APIC);
  fails += CHECK_PROC (aes, AES);
  fails += CHECK_PROC (amx_bf16, AMX_BF16);
  fails += CHECK_PROC (amx_int8, AMX_INT8);
  fails += CHECK_PROC (amx_tile, AMX_TILE);
  fails += CHECK_PROC (arch_capabilities, ARCH_CAPABILITIES);
  fails += CHECK_PROC (avx, AVX);
  fails += CHECK_PROC (avx2, AVX2);
  fails += CHECK_PROC (avx512_4fmaps, AVX512_4FMAPS);
  fails += CHECK_PROC (avx512_4vnniw, AVX512_4VNNIW);
  fails += CHECK_PROC (avx512_bf16, AVX512_BF16);
  fails += CHECK_PROC (avx512_bitalg, AVX512_BITALG);
  fails += CHECK_PROC (avx512ifma, AVX512_IFMA);
  fails += CHECK_PROC (avx512vbmi, AVX512_VBMI);
  fails += CHECK_PROC (avx512_vbmi2, AVX512_VBMI2);
  fails += CHECK_PROC (avx512_vnni, AVX512_VNNI);
  fails += CHECK_PROC (avx512_vp2intersect, AVX512_VP2INTERSECT);
  fails += CHECK_PROC (avx512_vpopcntdq, AVX512_VPOPCNTDQ);
  fails += CHECK_PROC (avx512bw, AVX512BW);
  fails += CHECK_PROC (avx512cd, AVX512CD);
  fails += CHECK_PROC (avx512er, AVX512ER);
  fails += CHECK_PROC (avx512dq, AVX512DQ);
  fails += CHECK_PROC (avx512f, AVX512F);
  fails += CHECK_PROC (avx512pf, AVX512PF);
  fails += CHECK_PROC (avx512vl, AVX512VL);
  fails += CHECK_PROC (bmi1, BMI1);
  fails += CHECK_PROC (bmi2, BMI2);
  fails += CHECK_PROC (cldemote, CLDEMOTE);
  fails += CHECK_PROC (clflushopt, CLFLUSHOPT);
  fails += CHECK_PROC (clflush, CLFSH);
  fails += CHECK_PROC (clwb, CLWB);
  fails += CHECK_PROC (cmov, CMOV);
  fails += CHECK_PROC (cx16, CMPXCHG16B);
  fails += CHECK_PROC (cnxt_id, CNXT_ID);
  fails += CHECK_PROC (core_capabilities, CORE_CAPABILITIES);
  fails += CHECK_PROC (cx8, CX8);
  fails += CHECK_PROC (dca, DCA);
  fails += CHECK_PROC (de, DE);
  fails += CHECK_PROC (zero_fcs_fds, DEPR_FPU_CS_DS);
  fails += CHECK_PROC (dts, DS);
  fails += CHECK_PROC (ds_cpl, DS_CPL);
  fails += CHECK_PROC (dtes64, DTES64);
  fails += CHECK_PROC (est, EIST);
  fails += CHECK_PROC (enqcmd, ENQCMD);
  fails += CHECK_PROC (erms, ERMS);
  fails += CHECK_PROC (f16c, F16C);
  fails += CHECK_PROC (fma, FMA);
  fails += CHECK_PROC (fma4, FMA4);
  fails += CHECK_PROC (fpu, FPU);
  fails += CHECK_PROC (fsgsbase, FSGSBASE);
  fails += CHECK_PROC (fsrm, FSRM);
  fails += CHECK_PROC (fxsr, FXSR);
  fails += CHECK_PROC (gfni, GFNI);
  fails += CHECK_PROC (hle, HLE);
  fails += CHECK_PROC (ht, HTT);
  fails += CHECK_PROC (hybrid, HYBRID);
  if (cpu_features->basic.kind == arch_kind_intel)
    {
      fails += CHECK_PROC (ibrs, IBRS_IBPB);
      fails += CHECK_PROC (stibp, STIBP);
    }
  else if (cpu_features->basic.kind == arch_kind_amd)
    {
      fails += CHECK_PROC (ibpb, AMD_IBPB);
      fails += CHECK_PROC (ibrs, AMD_IBRS);
      fails += CHECK_PROC (stibp, AMD_STIBP);
    }
  fails += CHECK_PROC (ibt, IBT);
  fails += CHECK_PROC (invariant_tsc, INVARIANT_TSC);
  fails += CHECK_PROC (invpcid, INVPCID);
  fails += CHECK_PROC (flush_l1d, L1D_FLUSH);
  fails += CHECK_PROC (lahf_lm, LAHF64_SAHF64);
  fails += CHECK_PROC (lm, LM);
  fails += CHECK_PROC (lwp, LWP);
  fails += CHECK_PROC (abm, LZCNT);
  fails += CHECK_PROC (mca, MCA);
  fails += CHECK_PROC (mce, MCE);
  fails += CHECK_PROC (md_clear, MD_CLEAR);
  fails += CHECK_PROC (mmx, MMX);
  fails += CHECK_PROC (monitor, MONITOR);
  fails += CHECK_PROC (movbe, MOVBE);
  fails += CHECK_PROC (movdiri, MOVDIRI);
  fails += CHECK_PROC (movdir64b, MOVDIR64B);
  fails += CHECK_PROC (mpx, MPX);
  fails += CHECK_PROC (msr, MSR);
  fails += CHECK_PROC (mtrr, MTRR);
  fails += CHECK_PROC (nx, NX);
  fails += CHECK_PROC (ospke, OSPKE);
#if 0
  /* NB: /proc/cpuinfo doesn't report this feature.  */
  fails += CHECK_PROC (osxsave, OSXSAVE);
#endif
  fails += CHECK_PROC (pae, PAE);
  fails += CHECK_PROC (pdpe1gb, PAGE1GB);
  fails += CHECK_PROC (pat, PAT);
  fails += CHECK_PROC (pbe, PBE);
  fails += CHECK_PROC (pcid, PCID);
  fails += CHECK_PROC (pclmulqdq, PCLMULQDQ);
  fails += CHECK_PROC (pconfig, PCONFIG);
  fails += CHECK_PROC (pdcm, PDCM);
  fails += CHECK_PROC (pge, PGE);
  fails += CHECK_PROC (pks, PKS);
  fails += CHECK_PROC (pku, PKU);
  fails += CHECK_PROC (popcnt, POPCNT);
  fails += CHECK_PROC (3dnowprefetch, PREFETCHW);
  fails += CHECK_PROC (prefetchwt1, PREFETCHWT1);
  fails += CHECK_PROC (ptwrite, PTWRITE);
  fails += CHECK_PROC (pse, PSE);
  fails += CHECK_PROC (pse36, PSE_36);
  fails += CHECK_PROC (psn, PSN);
  fails += CHECK_PROC (rdpid, RDPID);
  fails += CHECK_PROC (rdrand, RDRAND);
  fails += CHECK_PROC (rdseed, RDSEED);
  fails += CHECK_PROC (rdt_a, RDT_A);
  fails += CHECK_PROC (cqm, RDT_M);
  fails += CHECK_PROC (rdtscp, RDTSCP);
  fails += CHECK_PROC (rtm, RTM);
  fails += CHECK_PROC (sdbg, SDBG);
  fails += CHECK_PROC (sep, SEP);
  fails += CHECK_PROC (serialize, SERIALIZE);
  fails += CHECK_PROC (sgx, SGX);
  fails += CHECK_PROC (sgx_lc, SGX_LC);
  fails += CHECK_PROC (sha_ni, SHA);
  fails += CHECK_PROC (shstk, SHSTK);
  fails += CHECK_PROC (smap, SMAP);
  fails += CHECK_PROC (smep, SMEP);
  fails += CHECK_PROC (smx, SMX);
  fails += CHECK_PROC (ss, SS);
  if (cpu_features->basic.kind == arch_kind_intel)
    fails += CHECK_PROC (ssbd, SSBD);
  else if (cpu_features->basic.kind == arch_kind_amd)
    {
      /* This feature is implemented in 2 different ways on AMD processors:
	 newer systems provides AMD_SSBD (function 8000_0008, EBX[24]),
	 while older system proviseds AMD_VIRT_SSBD (function 8000_008,
	 EBX[25]).  However for AMD_VIRT_SSBD, kernel shows both 'ssbd'
	 and 'virt_ssbd' on /proc/cpuinfo; while for AMD_SSBD only 'ssbd'
	 is provided.  */
      if (HAS_CPU_FEATURE (AMD_SSBD))
	fails += CHECK_PROC (ssbd, AMD_SSBD);
      else if (HAS_CPU_FEATURE (AMD_VIRT_SSBD))
	fails += CHECK_PROC (virt_ssbd, AMD_VIRT_SSBD);
    }
  fails += CHECK_PROC (sse, SSE);
  fails += CHECK_PROC (sse2, SSE2);
  fails += CHECK_PROC (pni, SSE3);
  fails += CHECK_PROC (sse4_1, SSE4_1);
  fails += CHECK_PROC (sse4_2, SSE4_2);
  fails += CHECK_PROC (sse4a, SSE4A);
  fails += CHECK_PROC (ssse3, SSSE3);
  fails += CHECK_PROC (svm, SVM);
#ifdef __x86_64__
  /* NB: SYSCALL_SYSRET is 64-bit only.  */
  fails += CHECK_PROC (syscall, SYSCALL_SYSRET);
#endif
  fails += CHECK_PROC (tbm, TBM);
  fails += CHECK_PROC (tm, TM);
  fails += CHECK_PROC (tm2, TM2);
  fails += CHECK_PROC (intel_pt, TRACE);
  fails += CHECK_PROC (tsc, TSC);
  fails += CHECK_PROC (tsc_adjust, TSC_ADJUST);
  fails += CHECK_PROC (tsc_deadline_timer, TSC_DEADLINE);
  fails += CHECK_PROC (tsxldtrk, TSXLDTRK);
  fails += CHECK_PROC (umip, UMIP);
  fails += CHECK_PROC (vaes, VAES);
  fails += CHECK_PROC (vme, VME);
  fails += CHECK_PROC (vmx, VMX);
  fails += CHECK_PROC (vpclmulqdq, VPCLMULQDQ);
  fails += CHECK_PROC (waitpkg, WAITPKG);
  fails += CHECK_PROC (wbnoinvd, WBNOINVD);
  fails += CHECK_PROC (x2apic, X2APIC);
  fails += CHECK_PROC (xfd, XFD);
  fails += CHECK_PROC (xgetbv1, XGETBV_ECX_1);
  fails += CHECK_PROC (xop, XOP);
  fails += CHECK_PROC (xsave, XSAVE);
  fails += CHECK_PROC (xsavec, XSAVEC);
  fails += CHECK_PROC (xsaveopt, XSAVEOPT);
  fails += CHECK_PROC (xsaves, XSAVES);
  fails += CHECK_PROC (xtpr, XTPRUPDCTRL);

  printf ("%d differences between /proc/cpuinfo and glibc code.\n", fails);

  return (fails != 0);
}

#include "../../../test-skeleton.c"
