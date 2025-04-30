/* Data structure for x86 CPU features.
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

#ifndef	_PRIVATE_CPU_FEATURES_H
#define	_PRIVATE_CPU_FEATURES_H	1

#ifdef _CPU_FEATURES_H
# error this should be impossible
#endif

/* Get data structures without inline functions.  */
#define _SYS_PLATFORM_X86_H
#include <bits/platform/x86.h>

enum
{
  CPUID_INDEX_MAX = CPUID_INDEX_14_ECX_0 + 1
};

enum
{
  /* The integer bit array index for the first set of preferred feature
     bits.  */
  PREFERRED_FEATURE_INDEX_1 = 0,
  /* The current maximum size of the feature integer bit array.  */
  PREFERRED_FEATURE_INDEX_MAX
};

/* Only used directly in cpu-features.c.  */
#define CPU_FEATURE_SET(ptr, name) \
  ptr->features[index_cpu_##name].active.reg_##name |= bit_cpu_##name;
#define CPU_FEATURE_UNSET(ptr, name) \
  ptr->features[index_cpu_##name].active.reg_##name &= ~bit_cpu_##name;
#define CPU_FEATURE_SET_ACTIVE(ptr, name) \
  ptr->features[index_cpu_##name].active.reg_##name \
     |= ptr->features[index_cpu_##name].cpuid.reg_##name & bit_cpu_##name;
#define CPU_FEATURE_PREFERRED_P(ptr, name) \
  ((ptr->preferred[index_arch_##name] & bit_arch_##name) != 0)

#define CPU_FEATURE_CHECK_P(ptr, name, check) \
  ((ptr->features[index_cpu_##name].check.reg_##name \
    & bit_cpu_##name) != 0)
#define CPU_FEATURE_PRESENT_P(ptr, name) \
  CPU_FEATURE_CHECK_P (ptr, name, cpuid)
#define CPU_FEATURE_ACTIVE_P(ptr, name) \
  CPU_FEATURE_CHECK_P (ptr, name, active)
#define CPU_FEATURE_CPU_P(ptr, name) \
  CPU_FEATURE_PRESENT_P (ptr, name)
#define CPU_FEATURE_USABLE_P(ptr, name) \
  CPU_FEATURE_ACTIVE_P (ptr, name)

/* HAS_CPU_FEATURE evaluates to true if CPU supports the feature.  */
#define HAS_CPU_FEATURE(name) \
  CPU_FEATURE_CPU_P (__get_cpu_features (), name)
/* CPU_FEATURE_USABLE evaluates to true if the feature is usable.  */
#define CPU_FEATURE_USABLE(name) \
  CPU_FEATURE_USABLE_P (__get_cpu_features (), name)
/* CPU_FEATURE_PREFER evaluates to true if we prefer the feature at
   runtime.  */
#define CPU_FEATURE_PREFERRED(name) \
  CPU_FEATURE_PREFERRED_P(__get_cpu_features (), name)

#define CPU_FEATURES_CPU_P(ptr, name) \
  CPU_FEATURE_CPU_P (ptr, name)
#define CPU_FEATURES_ARCH_P(ptr, name) \
  CPU_FEATURE_PREFERRED_P (ptr, name)
#define HAS_ARCH_FEATURE(name) \
  CPU_FEATURE_PREFERRED (name)

/* CPU features.  */

/* CPUID_INDEX_1.  */

/* ECX.  */
#define bit_cpu_SSE3		(1u << 0)
#define bit_cpu_PCLMULQDQ	(1u << 1)
#define bit_cpu_DTES64		(1u << 2)
#define bit_cpu_MONITOR		(1u << 3)
#define bit_cpu_DS_CPL		(1u << 4)
#define bit_cpu_VMX		(1u << 5)
#define bit_cpu_SMX		(1u << 6)
#define bit_cpu_EIST		(1u << 7)
#define bit_cpu_TM2		(1u << 8)
#define bit_cpu_SSSE3		(1u << 9)
#define bit_cpu_CNXT_ID		(1u << 10)
#define bit_cpu_SDBG		(1u << 11)
#define bit_cpu_FMA		(1u << 12)
#define bit_cpu_CMPXCHG16B	(1u << 13)
#define bit_cpu_XTPRUPDCTRL	(1u << 14)
#define bit_cpu_PDCM		(1u << 15)
#define bit_cpu_INDEX_1_ECX_16	(1u << 16)
#define bit_cpu_PCID		(1u << 17)
#define bit_cpu_DCA		(1u << 18)
#define bit_cpu_SSE4_1		(1u << 19)
#define bit_cpu_SSE4_2		(1u << 20)
#define bit_cpu_X2APIC		(1u << 21)
#define bit_cpu_MOVBE		(1u << 22)
#define bit_cpu_POPCNT		(1u << 23)
#define bit_cpu_TSC_DEADLINE	(1u << 24)
#define bit_cpu_AES		(1u << 25)
#define bit_cpu_XSAVE		(1u << 26)
#define bit_cpu_OSXSAVE		(1u << 27)
#define bit_cpu_AVX		(1u << 28)
#define bit_cpu_F16C		(1u << 29)
#define bit_cpu_RDRAND		(1u << 30)
#define bit_cpu_INDEX_1_ECX_31	(1u << 31)

/* EDX.  */
#define bit_cpu_FPU		(1u << 0)
#define bit_cpu_VME		(1u << 1)
#define bit_cpu_DE		(1u << 2)
#define bit_cpu_PSE		(1u << 3)
#define bit_cpu_TSC		(1u << 4)
#define bit_cpu_MSR		(1u << 5)
#define bit_cpu_PAE		(1u << 6)
#define bit_cpu_MCE		(1u << 7)
#define bit_cpu_CX8		(1u << 8)
#define bit_cpu_APIC		(1u << 9)
#define bit_cpu_INDEX_1_EDX_10	(1u << 10)
#define bit_cpu_SEP		(1u << 11)
#define bit_cpu_MTRR		(1u << 12)
#define bit_cpu_PGE		(1u << 13)
#define bit_cpu_MCA		(1u << 14)
#define bit_cpu_CMOV		(1u << 15)
#define bit_cpu_PAT		(1u << 16)
#define bit_cpu_PSE_36		(1u << 17)
#define bit_cpu_PSN		(1u << 18)
#define bit_cpu_CLFSH		(1u << 19)
#define bit_cpu_INDEX_1_EDX_20	(1u << 20)
#define bit_cpu_DS		(1u << 21)
#define bit_cpu_ACPI		(1u << 22)
#define bit_cpu_MMX		(1u << 23)
#define bit_cpu_FXSR		(1u << 24)
#define bit_cpu_SSE		(1u << 25)
#define bit_cpu_SSE2		(1u << 26)
#define bit_cpu_SS		(1u << 27)
#define bit_cpu_HTT		(1u << 28)
#define bit_cpu_TM		(1u << 29)
#define bit_cpu_INDEX_1_EDX_30	(1u << 30)
#define bit_cpu_PBE		(1u << 31)

/* CPUID_INDEX_7.  */

/* EBX.  */
#define bit_cpu_FSGSBASE	(1u << 0)
#define bit_cpu_TSC_ADJUST	(1u << 1)
#define bit_cpu_SGX		(1u << 2)
#define bit_cpu_BMI1		(1u << 3)
#define bit_cpu_HLE		(1u << 4)
#define bit_cpu_AVX2		(1u << 5)
#define bit_cpu_INDEX_7_EBX_6	(1u << 6)
#define bit_cpu_SMEP		(1u << 7)
#define bit_cpu_BMI2		(1u << 8)
#define bit_cpu_ERMS		(1u << 9)
#define bit_cpu_INVPCID		(1u << 10)
#define bit_cpu_RTM		(1u << 11)
#define bit_cpu_RDT_M		(1u << 12)
#define bit_cpu_DEPR_FPU_CS_DS	(1u << 13)
#define bit_cpu_MPX		(1u << 14)
#define bit_cpu_RDT_A		(1u << 15)
#define bit_cpu_AVX512F		(1u << 16)
#define bit_cpu_AVX512DQ	(1u << 17)
#define bit_cpu_RDSEED		(1u << 18)
#define bit_cpu_ADX		(1u << 19)
#define bit_cpu_SMAP		(1u << 20)
#define bit_cpu_AVX512_IFMA	(1u << 21)
#define bit_cpu_INDEX_7_EBX_22	(1u << 22)
#define bit_cpu_CLFLUSHOPT	(1u << 23)
#define bit_cpu_CLWB		(1u << 24)
#define bit_cpu_TRACE		(1u << 25)
#define bit_cpu_AVX512PF	(1u << 26)
#define bit_cpu_AVX512ER	(1u << 27)
#define bit_cpu_AVX512CD	(1u << 28)
#define bit_cpu_SHA		(1u << 29)
#define bit_cpu_AVX512BW	(1u << 30)
#define bit_cpu_AVX512VL	(1u << 31)

/* ECX.  */
#define bit_cpu_PREFETCHWT1	(1u << 0)
#define bit_cpu_AVX512_VBMI	(1u << 1)
#define bit_cpu_UMIP		(1u << 2)
#define bit_cpu_PKU		(1u << 3)
#define bit_cpu_OSPKE		(1u << 4)
#define bit_cpu_WAITPKG		(1u << 5)
#define bit_cpu_AVX512_VBMI2	(1u << 6)
#define bit_cpu_SHSTK		(1u << 7)
#define bit_cpu_GFNI		(1u << 8)
#define bit_cpu_VAES		(1u << 9)
#define bit_cpu_VPCLMULQDQ	(1u << 10)
#define bit_cpu_AVX512_VNNI	(1u << 11)
#define bit_cpu_AVX512_BITALG	(1u << 12)
#define bit_cpu_INDEX_7_ECX_13	(1u << 13)
#define bit_cpu_AVX512_VPOPCNTDQ (1u << 14)
#define bit_cpu_INDEX_7_ECX_15	(1u << 15)
#define bit_cpu_INDEX_7_ECX_16	(1u << 16)
/* Note: Bits 17-21: The value of MAWAU used by the BNDLDX and BNDSTX
   instructions in 64-bit mode.  */
#define bit_cpu_RDPID		(1u << 22)
#define bit_cpu_KL		(1u << 23)
#define bit_cpu_INDEX_7_ECX_24	(1u << 24)
#define bit_cpu_CLDEMOTE	(1u << 25)
#define bit_cpu_INDEX_7_ECX_26	(1u << 26)
#define bit_cpu_MOVDIRI		(1u << 27)
#define bit_cpu_MOVDIR64B	(1u << 28)
#define bit_cpu_ENQCMD		(1u << 29)
#define bit_cpu_SGX_LC		(1u << 30)
#define bit_cpu_PKS		(1u << 31)

/* EDX.  */
#define bit_cpu_INDEX_7_EDX_0	(1u << 0)
#define bit_cpu_INDEX_7_EDX_1	(1u << 1)
#define bit_cpu_AVX512_4VNNIW	(1u << 2)
#define bit_cpu_AVX512_4FMAPS	(1u << 3)
#define bit_cpu_FSRM		(1u << 4)
#define bit_cpu_UINTR		(1u << 5)
#define bit_cpu_INDEX_7_EDX_6	(1u << 6)
#define bit_cpu_INDEX_7_EDX_7	(1u << 7)
#define bit_cpu_AVX512_VP2INTERSECT (1u << 8)
#define bit_cpu_INDEX_7_EDX_9	(1u << 9)
#define bit_cpu_MD_CLEAR	(1u << 10)
#define bit_cpu_RTM_ALWAYS_ABORT (1u << 11)
#define bit_cpu_INDEX_7_EDX_12	(1u << 12)
#define bit_cpu_INDEX_7_EDX_13	(1u << 13)
#define bit_cpu_SERIALIZE	(1u << 14)
#define bit_cpu_HYBRID		(1u << 15)
#define bit_cpu_TSXLDTRK	(1u << 16)
#define bit_cpu_INDEX_7_EDX_17	(1u << 17)
#define bit_cpu_PCONFIG		(1u << 18)
#define bit_cpu_INDEX_7_EDX_19	(1u << 19)
#define bit_cpu_IBT		(1u << 20)
#define bit_cpu_INDEX_7_EDX_21	(1u << 21)
#define bit_cpu_AMX_BF16	(1u << 22)
#define bit_cpu_AVX512_FP16	(1u << 23)
#define bit_cpu_AMX_TILE	(1u << 24)
#define bit_cpu_AMX_INT8	(1u << 25)
#define bit_cpu_IBRS_IBPB	(1u << 26)
#define bit_cpu_STIBP		(1u << 27)
#define bit_cpu_L1D_FLUSH	(1u << 28)
#define bit_cpu_ARCH_CAPABILITIES (1u << 29)
#define bit_cpu_CORE_CAPABILITIES (1u << 30)
#define bit_cpu_SSBD		(1u << 31)

/* CPUID_INDEX_80000001.  */

/* ECX.  */
#define bit_cpu_LAHF64_SAHF64	(1u << 0)
#define bit_cpu_SVM		(1u << 2)
#define bit_cpu_LZCNT		(1u << 5)
#define bit_cpu_SSE4A		(1u << 6)
#define bit_cpu_PREFETCHW	(1u << 8)
#define bit_cpu_XOP		(1u << 11)
#define bit_cpu_LWP		(1u << 15)
#define bit_cpu_FMA4		(1u << 16)
#define bit_cpu_TBM		(1u << 21)

/* EDX.  */
#define bit_cpu_SYSCALL_SYSRET	(1u << 11)
#define bit_cpu_NX		(1u << 20)
#define bit_cpu_PAGE1GB		(1u << 26)
#define bit_cpu_RDTSCP		(1u << 27)
#define bit_cpu_LM		(1u << 29)

/* CPUID_INDEX_D_ECX_1.  */

/* EAX.  */
#define bit_cpu_XSAVEOPT	(1u << 0)
#define bit_cpu_XSAVEC		(1u << 1)
#define bit_cpu_XGETBV_ECX_1	(1u << 2)
#define bit_cpu_XSAVES		(1u << 3)
#define bit_cpu_XFD		(1u << 4)

/* CPUID_INDEX_80000007.  */

/* EDX.  */
#define bit_cpu_INVARIANT_TSC	(1u << 8)

/* CPUID_INDEX_80000008.  */

/* EBX.  */
#define bit_cpu_WBNOINVD	(1u << 9)
#define bit_cpu_AMD_IBPB	(1u << 12)
#define bit_cpu_AMD_IBRS	(1u << 14)
#define bit_cpu_AMD_STIBP	(1u << 15)
#define bit_cpu_AMD_SSBD	(1u << 24)
#define bit_cpu_AMD_VIRT_SSBD	(1u << 25)

/* CPUID_INDEX_7_ECX_1.  */

/* EAX.  */
#define bit_cpu_AVX_VNNI	(1u << 4)
#define bit_cpu_AVX512_BF16	(1u << 5)
#define bit_cpu_FZLRM		(1u << 10)
#define bit_cpu_FSRS		(1u << 11)
#define bit_cpu_FSRCS		(1u << 12)
#define bit_cpu_HRESET		(1u << 22)
#define bit_cpu_LAM		(1u << 26)

/* CPUID_INDEX_19.  */

/* EBX.  */
#define bit_cpu_AESKLE		(1u << 0)
#define bit_cpu_WIDE_KL		(1u << 2)

/* CPUID_INDEX_14_ECX_0.  */

/* EBX.  */
#define bit_cpu_PTWRITE		(1u << 4)

/* CPUID_INDEX_1.  */

/* ECX.  */
#define index_cpu_SSE3		CPUID_INDEX_1
#define index_cpu_PCLMULQDQ	CPUID_INDEX_1
#define index_cpu_DTES64	CPUID_INDEX_1
#define index_cpu_MONITOR	CPUID_INDEX_1
#define index_cpu_DS_CPL	CPUID_INDEX_1
#define index_cpu_VMX		CPUID_INDEX_1
#define index_cpu_SMX		CPUID_INDEX_1
#define index_cpu_EIST		CPUID_INDEX_1
#define index_cpu_TM2		CPUID_INDEX_1
#define index_cpu_SSSE3		CPUID_INDEX_1
#define index_cpu_CNXT_ID	CPUID_INDEX_1
#define index_cpu_SDBG		CPUID_INDEX_1
#define index_cpu_FMA		CPUID_INDEX_1
#define index_cpu_CMPXCHG16B	CPUID_INDEX_1
#define index_cpu_XTPRUPDCTRL	CPUID_INDEX_1
#define index_cpu_PDCM		CPUID_INDEX_1
#define index_cpu_INDEX_1_ECX_16 CPUID_INDEX_1
#define index_cpu_PCID		CPUID_INDEX_1
#define index_cpu_DCA		CPUID_INDEX_1
#define index_cpu_SSE4_1	CPUID_INDEX_1
#define index_cpu_SSE4_2	CPUID_INDEX_1
#define index_cpu_X2APIC	CPUID_INDEX_1
#define index_cpu_MOVBE		CPUID_INDEX_1
#define index_cpu_POPCNT	CPUID_INDEX_1
#define index_cpu_TSC_DEADLINE	CPUID_INDEX_1
#define index_cpu_AES		CPUID_INDEX_1
#define index_cpu_XSAVE		CPUID_INDEX_1
#define index_cpu_OSXSAVE	CPUID_INDEX_1
#define index_cpu_AVX		CPUID_INDEX_1
#define index_cpu_F16C		CPUID_INDEX_1
#define index_cpu_RDRAND	CPUID_INDEX_1
#define index_cpu_INDEX_1_ECX_31 CPUID_INDEX_1

/* ECX.  */
#define index_cpu_FPU		CPUID_INDEX_1
#define index_cpu_VME		CPUID_INDEX_1
#define index_cpu_DE		CPUID_INDEX_1
#define index_cpu_PSE		CPUID_INDEX_1
#define index_cpu_TSC		CPUID_INDEX_1
#define index_cpu_MSR		CPUID_INDEX_1
#define index_cpu_PAE		CPUID_INDEX_1
#define index_cpu_MCE		CPUID_INDEX_1
#define index_cpu_CX8		CPUID_INDEX_1
#define index_cpu_APIC		CPUID_INDEX_1
#define index_cpu_INDEX_1_EDX_10 CPUID_INDEX_1
#define index_cpu_SEP		CPUID_INDEX_1
#define index_cpu_MTRR		CPUID_INDEX_1
#define index_cpu_PGE		CPUID_INDEX_1
#define index_cpu_MCA		CPUID_INDEX_1
#define index_cpu_CMOV		CPUID_INDEX_1
#define index_cpu_PAT		CPUID_INDEX_1
#define index_cpu_PSE_36	CPUID_INDEX_1
#define index_cpu_PSN		CPUID_INDEX_1
#define index_cpu_CLFSH		CPUID_INDEX_1
#define index_cpu_INDEX_1_EDX_20 CPUID_INDEX_1
#define index_cpu_DS		CPUID_INDEX_1
#define index_cpu_ACPI		CPUID_INDEX_1
#define index_cpu_MMX		CPUID_INDEX_1
#define index_cpu_FXSR		CPUID_INDEX_1
#define index_cpu_SSE		CPUID_INDEX_1
#define index_cpu_SSE2		CPUID_INDEX_1
#define index_cpu_SS		CPUID_INDEX_1
#define index_cpu_HTT		CPUID_INDEX_1
#define index_cpu_TM		CPUID_INDEX_1
#define index_cpu_INDEX_1_EDX_30 CPUID_INDEX_1
#define index_cpu_PBE		CPUID_INDEX_1

/* CPUID_INDEX_7.  */

/* EBX.  */
#define index_cpu_FSGSBASE	CPUID_INDEX_7
#define index_cpu_TSC_ADJUST	CPUID_INDEX_7
#define index_cpu_SGX		CPUID_INDEX_7
#define index_cpu_BMI1		CPUID_INDEX_7
#define index_cpu_HLE		CPUID_INDEX_7
#define index_cpu_AVX2		CPUID_INDEX_7
#define index_cpu_INDEX_7_EBX_6	CPUID_INDEX_7
#define index_cpu_SMEP		CPUID_INDEX_7
#define index_cpu_BMI2		CPUID_INDEX_7
#define index_cpu_ERMS		CPUID_INDEX_7
#define index_cpu_INVPCID	CPUID_INDEX_7
#define index_cpu_RTM		CPUID_INDEX_7
#define index_cpu_RDT_M		CPUID_INDEX_7
#define index_cpu_DEPR_FPU_CS_DS CPUID_INDEX_7
#define index_cpu_MPX		CPUID_INDEX_7
#define index_cpu_RDT_A		CPUID_INDEX_7
#define index_cpu_AVX512F	CPUID_INDEX_7
#define index_cpu_AVX512DQ	CPUID_INDEX_7
#define index_cpu_RDSEED	CPUID_INDEX_7
#define index_cpu_ADX		CPUID_INDEX_7
#define index_cpu_SMAP		CPUID_INDEX_7
#define index_cpu_AVX512_IFMA	CPUID_INDEX_7
#define index_cpu_INDEX_7_EBX_22 CPUID_INDEX_7
#define index_cpu_CLFLUSHOPT	CPUID_INDEX_7
#define index_cpu_CLWB		CPUID_INDEX_7
#define index_cpu_TRACE		CPUID_INDEX_7
#define index_cpu_AVX512PF	CPUID_INDEX_7
#define index_cpu_AVX512ER	CPUID_INDEX_7
#define index_cpu_AVX512CD	CPUID_INDEX_7
#define index_cpu_SHA		CPUID_INDEX_7
#define index_cpu_AVX512BW	CPUID_INDEX_7
#define index_cpu_AVX512VL	CPUID_INDEX_7

/* ECX.  */
#define index_cpu_PREFETCHWT1	CPUID_INDEX_7
#define index_cpu_AVX512_VBMI	CPUID_INDEX_7
#define index_cpu_UMIP		CPUID_INDEX_7
#define index_cpu_PKU		CPUID_INDEX_7
#define index_cpu_OSPKE		CPUID_INDEX_7
#define index_cpu_WAITPKG	CPUID_INDEX_7
#define index_cpu_AVX512_VBMI2	CPUID_INDEX_7
#define index_cpu_SHSTK		CPUID_INDEX_7
#define index_cpu_GFNI		CPUID_INDEX_7
#define index_cpu_VAES		CPUID_INDEX_7
#define index_cpu_VPCLMULQDQ	CPUID_INDEX_7
#define index_cpu_AVX512_VNNI	CPUID_INDEX_7
#define index_cpu_AVX512_BITALG CPUID_INDEX_7
#define index_cpu_INDEX_7_ECX_13 CPUID_INDEX_7
#define index_cpu_AVX512_VPOPCNTDQ CPUID_INDEX_7
#define index_cpu_INDEX_7_ECX_15 CPUID_INDEX_7
#define index_cpu_INDEX_7_ECX_16 CPUID_INDEX_7
#define index_cpu_RDPID		CPUID_INDEX_7
#define index_cpu_KL		CPUID_INDEX_7
#define index_cpu_INDEX_7_ECX_24 CPUID_INDEX_7
#define index_cpu_CLDEMOTE	CPUID_INDEX_7
#define index_cpu_INDEX_7_ECX_26 CPUID_INDEX_7
#define index_cpu_MOVDIRI	CPUID_INDEX_7
#define index_cpu_MOVDIR64B	CPUID_INDEX_7
#define index_cpu_ENQCMD	CPUID_INDEX_7
#define index_cpu_SGX_LC	CPUID_INDEX_7
#define index_cpu_PKS		CPUID_INDEX_7

/* EDX.  */
#define index_cpu_INDEX_7_EDX_0	CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_1	CPUID_INDEX_7
#define index_cpu_AVX512_4VNNIW CPUID_INDEX_7
#define index_cpu_AVX512_4FMAPS	CPUID_INDEX_7
#define index_cpu_FSRM		CPUID_INDEX_7
#define index_cpu_UINTR		CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_6	CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_7	CPUID_INDEX_7
#define index_cpu_AVX512_VP2INTERSECT CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_9	CPUID_INDEX_7
#define index_cpu_MD_CLEAR	CPUID_INDEX_7
#define index_cpu_RTM_ALWAYS_ABORT CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_12 CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_13 CPUID_INDEX_7
#define index_cpu_SERIALIZE	CPUID_INDEX_7
#define index_cpu_HYBRID	CPUID_INDEX_7
#define index_cpu_TSXLDTRK	CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_17 CPUID_INDEX_7
#define index_cpu_PCONFIG	CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_19 CPUID_INDEX_7
#define index_cpu_IBT		CPUID_INDEX_7
#define index_cpu_INDEX_7_EDX_21 CPUID_INDEX_7
#define index_cpu_AMX_BF16	CPUID_INDEX_7
#define index_cpu_AVX512_FP16	CPUID_INDEX_7
#define index_cpu_AMX_TILE	CPUID_INDEX_7
#define index_cpu_AMX_INT8	CPUID_INDEX_7
#define index_cpu_IBRS_IBPB	CPUID_INDEX_7
#define index_cpu_STIBP		CPUID_INDEX_7
#define index_cpu_L1D_FLUSH	CPUID_INDEX_7
#define index_cpu_ARCH_CAPABILITIES CPUID_INDEX_7
#define index_cpu_CORE_CAPABILITIES CPUID_INDEX_7
#define index_cpu_SSBD		CPUID_INDEX_7

/* CPUID_INDEX_80000001.  */

/* ECX.  */
#define index_cpu_LAHF64_SAHF64 CPUID_INDEX_80000001
#define index_cpu_SVM		CPUID_INDEX_80000001
#define index_cpu_LZCNT		CPUID_INDEX_80000001
#define index_cpu_SSE4A		CPUID_INDEX_80000001
#define index_cpu_PREFETCHW	CPUID_INDEX_80000001
#define index_cpu_XOP		CPUID_INDEX_80000001
#define index_cpu_LWP		CPUID_INDEX_80000001
#define index_cpu_FMA4		CPUID_INDEX_80000001
#define index_cpu_TBM		CPUID_INDEX_80000001

/* EDX.  */
#define index_cpu_SYSCALL_SYSRET CPUID_INDEX_80000001
#define index_cpu_NX		CPUID_INDEX_80000001
#define index_cpu_PAGE1GB	CPUID_INDEX_80000001
#define index_cpu_RDTSCP	CPUID_INDEX_80000001
#define index_cpu_LM		CPUID_INDEX_80000001

/* CPUID_INDEX_D_ECX_1.  */

/* EAX.  */
#define index_cpu_XSAVEOPT	CPUID_INDEX_D_ECX_1
#define index_cpu_XSAVEC	CPUID_INDEX_D_ECX_1
#define index_cpu_XGETBV_ECX_1	CPUID_INDEX_D_ECX_1
#define index_cpu_XSAVES	CPUID_INDEX_D_ECX_1
#define index_cpu_XFD		CPUID_INDEX_D_ECX_1

/* CPUID_INDEX_80000007.  */

/* EDX.  */
#define index_cpu_INVARIANT_TSC	CPUID_INDEX_80000007

/* CPUID_INDEX_80000008.  */

/* EBX.  */
#define index_cpu_WBNOINVD	CPUID_INDEX_80000008
#define index_cpu_AMD_IBPB	CPUID_INDEX_80000008
#define index_cpu_AMD_IBRS	CPUID_INDEX_80000008
#define index_cpu_AMD_STIBP	CPUID_INDEX_80000008
#define index_cpu_AMD_SSBD	CPUID_INDEX_80000008
#define index_cpu_AMD_VIRT_SSBD	CPUID_INDEX_80000008

/* CPUID_INDEX_7_ECX_1.  */

/* EAX.  */
#define index_cpu_AVX_VNNI	CPUID_INDEX_7_ECX_1
#define index_cpu_AVX512_BF16	CPUID_INDEX_7_ECX_1
#define index_cpu_FZLRM		CPUID_INDEX_7_ECX_1
#define index_cpu_FSRS		CPUID_INDEX_7_ECX_1
#define index_cpu_FSRCS		CPUID_INDEX_7_ECX_1
#define index_cpu_HRESET	CPUID_INDEX_7_ECX_1
#define index_cpu_LAM		CPUID_INDEX_7_ECX_1

/* CPUID_INDEX_19.  */

/* EBX.  */
#define index_cpu_AESKLE	CPUID_INDEX_19
#define index_cpu_WIDE_KL	CPUID_INDEX_19

/* CPUID_INDEX_14_ECX_0.  */

/* EBX.  */
#define index_cpu_PTWRITE	CPUID_INDEX_14_ECX_0

/* CPUID_INDEX_1.  */

/* ECX.  */
#define reg_SSE3		ecx
#define reg_PCLMULQDQ		ecx
#define reg_DTES64		ecx
#define reg_MONITOR		ecx
#define reg_DS_CPL		ecx
#define reg_VMX			ecx
#define reg_SMX			ecx
#define reg_EIST		ecx
#define reg_TM2			ecx
#define reg_SSSE3		ecx
#define reg_CNXT_ID		ecx
#define reg_SDBG		ecx
#define reg_FMA			ecx
#define reg_CMPXCHG16B		ecx
#define reg_XTPRUPDCTRL		ecx
#define reg_PDCM		ecx
#define reg_INDEX_1_ECX_16	ecx
#define reg_PCID		ecx
#define reg_DCA			ecx
#define reg_SSE4_1		ecx
#define reg_SSE4_2		ecx
#define reg_X2APIC		ecx
#define reg_MOVBE		ecx
#define reg_POPCNT		ecx
#define reg_TSC_DEADLINE	ecx
#define reg_AES			ecx
#define reg_XSAVE		ecx
#define reg_OSXSAVE		ecx
#define reg_AVX			ecx
#define reg_F16C		ecx
#define reg_RDRAND		ecx
#define reg_INDEX_1_ECX_31	ecx

/* EDX.  */
#define reg_FPU			edx
#define reg_VME			edx
#define reg_DE			edx
#define reg_PSE			edx
#define reg_TSC			edx
#define reg_MSR			edx
#define reg_PAE			edx
#define reg_MCE			edx
#define reg_CX8			edx
#define reg_APIC		edx
#define reg_INDEX_1_EDX_10	edx
#define reg_SEP			edx
#define reg_MTRR		edx
#define reg_PGE			edx
#define reg_MCA			edx
#define reg_CMOV		edx
#define reg_PAT			edx
#define reg_PSE_36		edx
#define reg_PSN			edx
#define reg_CLFSH		edx
#define reg_INDEX_1_EDX_20	edx
#define reg_DS			edx
#define reg_ACPI		edx
#define reg_MMX			edx
#define reg_FXSR		edx
#define reg_SSE			edx
#define reg_SSE2		edx
#define reg_SS			edx
#define reg_HTT			edx
#define reg_TM			edx
#define reg_INDEX_1_EDX_30	edx
#define reg_PBE			edx

/* CPUID_INDEX_7.  */

/* EBX.  */
#define reg_FSGSBASE		ebx
#define reg_TSC_ADJUST		ebx
#define reg_SGX			ebx
#define reg_BMI1		ebx
#define reg_HLE			ebx
#define reg_BMI2		ebx
#define reg_AVX2		ebx
#define reg_INDEX_7_EBX_6	ebx
#define reg_SMEP		ebx
#define reg_ERMS		ebx
#define reg_INVPCID		ebx
#define reg_RTM			ebx
#define reg_RDT_M		ebx
#define reg_DEPR_FPU_CS_DS	ebx
#define reg_MPX			ebx
#define reg_RDT_A		ebx
#define reg_AVX512F		ebx
#define reg_AVX512DQ		ebx
#define reg_RDSEED		ebx
#define reg_ADX			ebx
#define reg_SMAP		ebx
#define reg_AVX512_IFMA		ebx
#define reg_INDEX_7_EBX_22	ebx
#define reg_CLFLUSHOPT		ebx
#define reg_CLWB		ebx
#define reg_TRACE		ebx
#define reg_AVX512PF		ebx
#define reg_AVX512ER		ebx
#define reg_AVX512CD		ebx
#define reg_SHA			ebx
#define reg_AVX512BW		ebx
#define reg_AVX512VL		ebx

/* ECX.  */
#define reg_PREFETCHWT1		ecx
#define reg_AVX512_VBMI		ecx
#define reg_UMIP		ecx
#define reg_PKU			ecx
#define reg_OSPKE		ecx
#define reg_WAITPKG		ecx
#define reg_AVX512_VBMI2	ecx
#define reg_SHSTK		ecx
#define reg_GFNI		ecx
#define reg_VAES		ecx
#define reg_VPCLMULQDQ		ecx
#define reg_AVX512_VNNI		ecx
#define reg_AVX512_BITALG	ecx
#define reg_INDEX_7_ECX_13	ecx
#define reg_AVX512_VPOPCNTDQ	ecx
#define reg_INDEX_7_ECX_15	ecx
#define reg_INDEX_7_ECX_16	ecx
#define reg_RDPID		ecx
#define reg_KL			ecx
#define reg_INDEX_7_ECX_24	ecx
#define reg_CLDEMOTE		ecx
#define reg_INDEX_7_ECX_26	ecx
#define reg_MOVDIRI		ecx
#define reg_MOVDIR64B		ecx
#define reg_ENQCMD		ecx
#define reg_SGX_LC		ecx
#define reg_PKS			ecx

/* EDX.  */
#define reg_INDEX_7_EDX_0	edx
#define reg_INDEX_7_EDX_1	edx
#define reg_AVX512_4VNNIW	edx
#define reg_AVX512_4FMAPS	edx
#define reg_FSRM		edx
#define reg_UINTR		edx
#define reg_INDEX_7_EDX_6	edx
#define reg_INDEX_7_EDX_7	edx
#define reg_AVX512_VP2INTERSECT	edx
#define reg_INDEX_7_EDX_9	edx
#define reg_MD_CLEAR		edx
#define reg_RTM_ALWAYS_ABORT	edx
#define reg_INDEX_7_EDX_12	edx
#define reg_INDEX_7_EDX_13	edx
#define reg_SERIALIZE		edx
#define reg_HYBRID		edx
#define reg_TSXLDTRK		edx
#define reg_INDEX_7_EDX_17	edx
#define reg_PCONFIG		edx
#define reg_INDEX_7_EDX_19	edx
#define reg_IBT			edx
#define reg_INDEX_7_EDX_21	edx
#define reg_AMX_BF16		edx
#define reg_AVX512_FP16		edx
#define reg_AMX_TILE		edx
#define reg_AMX_INT8		edx
#define reg_IBRS_IBPB		edx
#define reg_STIBP		edx
#define reg_L1D_FLUSH		edx
#define reg_ARCH_CAPABILITIES	edx
#define reg_CORE_CAPABILITIES	edx
#define reg_SSBD		edx

/* CPUID_INDEX_80000001.  */

/* ECX.  */
#define reg_LAHF64_SAHF64	ecx
#define reg_SVM			ecx
#define reg_LZCNT		ecx
#define reg_SSE4A		ecx
#define reg_PREFETCHW		ecx
#define reg_XOP			ecx
#define reg_LWP			ecx
#define reg_FMA4		ecx
#define reg_TBM			ecx

/* EDX.  */
#define reg_SYSCALL_SYSRET	edx
#define reg_NX			edx
#define reg_PAGE1GB		edx
#define reg_RDTSCP		edx
#define reg_LM			edx

/* CPUID_INDEX_D_ECX_1.  */

/* EAX.  */
#define reg_XSAVEOPT		eax
#define reg_XSAVEC		eax
#define reg_XGETBV_ECX_1	eax
#define reg_XSAVES		eax
#define reg_XFD			eax

/* CPUID_INDEX_80000007.  */

/* EDX.  */
#define reg_INVARIANT_TSC	edx

/* CPUID_INDEX_80000008.  */

/* EBX.  */
#define reg_WBNOINVD		ebx
#define reg_AMD_IBPB		ebx
#define reg_AMD_IBRS		ebx
#define reg_AMD_STIBP		ebx
#define reg_AMD_SSBD		ebx
#define reg_AMD_VIRT_SSBD	ebx

/* CPUID_INDEX_7_ECX_1.  */

/* EAX.  */
#define reg_AVX_VNNI		eax
#define reg_AVX512_BF16		eax
#define reg_FZLRM		eax
#define reg_FSRS		eax
#define reg_FSRCS		eax
#define reg_HRESET		eax
#define reg_LAM			eax

/* CPUID_INDEX_19.  */

/* EBX.  */
#define reg_AESKLE		ebx
#define reg_WIDE_KL		ebx

/* CPUID_INDEX_14_ECX_0.  */

/* EBX.  */
#define reg_PTWRITE		ebx

/* PREFERRED_FEATURE_INDEX_1.  First define the bitindex values
   sequentially, then define the bit_arch* and index_arch_* lookup
   constants.  */
enum
  {
#define BIT(x) _bitindex_arch_##x ,
#include "cpu-features-preferred_feature_index_1.def"
#undef BIT
  };
enum
  {
#define BIT(x)					\
    bit_arch_##x = 1u << _bitindex_arch_##x ,	\
    index_arch_##x = PREFERRED_FEATURE_INDEX_1,
#include "cpu-features-preferred_feature_index_1.def"
#undef BIT
  };

/* XCR0 Feature flags.  */
#define bit_XMM_state		(1u << 1)
#define bit_YMM_state		(1u << 2)
#define bit_Opmask_state	(1u << 5)
#define bit_ZMM0_15_state	(1u << 6)
#define bit_ZMM16_31_state	(1u << 7)
#define bit_XTILECFG_state	(1u << 17)
#define bit_XTILEDATA_state	(1u << 18)

enum cpu_features_kind
{
  arch_kind_unknown = 0,
  arch_kind_intel,
  arch_kind_amd,
  arch_kind_zhaoxin,
  arch_kind_other
};

struct cpu_features_basic
{
  enum cpu_features_kind kind;
  int max_cpuid;
  unsigned int family;
  unsigned int model;
  unsigned int stepping;
};

struct cpuid_registers
{
  unsigned int eax;
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;
};

struct cpuid_feature_internal
{
  union
    {
      unsigned int cpuid_array[4];
      struct cpuid_registers cpuid;
    };
  union
    {
      unsigned int active_array[4];
      struct cpuid_registers active;
    };
};

/* NB: When adding new fields, update sysdeps/x86/dl-diagnostics-cpu.c
   to print them.  */
struct cpu_features
{
  struct cpu_features_basic basic;
  struct cpuid_feature_internal features[CPUID_INDEX_MAX];
  unsigned int preferred[PREFERRED_FEATURE_INDEX_MAX];
  /* X86 micro-architecture ISA levels.  */
  unsigned int isa_1;
  /* The state size for XSAVEC or XSAVE.  The type must be unsigned long
     int so that we use

	sub xsave_state_size_offset(%rip) %RSP_LP

     in _dl_runtime_resolve.  */
  unsigned long int xsave_state_size;
  /* The full state size for XSAVE when XSAVEC is disabled by

     GLIBC_TUNABLES=glibc.cpu.hwcaps=-XSAVEC
   */
  unsigned int xsave_state_full_size;
  /* Data cache size for use in memory and string routines, typically
     L1 size.  */
  unsigned long int data_cache_size;
  /* Shared cache size for use in memory and string routines, typically
     L2 or L3 size.  */
  unsigned long int shared_cache_size;
  /* Threshold to use non temporal store.  */
  unsigned long int non_temporal_threshold;
  /* Threshold to use "rep movsb".  */
  unsigned long int rep_movsb_threshold;
  /* Threshold to stop using "rep movsb".  */
  unsigned long int rep_movsb_stop_threshold;
  /* Threshold to use "rep stosb".  */
  unsigned long int rep_stosb_threshold;
  /* _SC_LEVEL1_ICACHE_SIZE.  */
  unsigned long int level1_icache_size;
  /* _SC_LEVEL1_ICACHE_LINESIZE.  */
  unsigned long int level1_icache_linesize;
  /* _SC_LEVEL1_DCACHE_SIZE.  */
  unsigned long int level1_dcache_size;
  /* _SC_LEVEL1_DCACHE_ASSOC.  */
  unsigned long int level1_dcache_assoc;
  /* _SC_LEVEL1_DCACHE_LINESIZE.  */
  unsigned long int level1_dcache_linesize;
  /* _SC_LEVEL2_CACHE_ASSOC.  */
  unsigned long int level2_cache_size;
  /* _SC_LEVEL2_DCACHE_ASSOC.  */
  unsigned long int level2_cache_assoc;
  /* _SC_LEVEL2_CACHE_LINESIZE.  */
  unsigned long int level2_cache_linesize;
  /* /_SC_LEVEL3_CACHE_SIZE.  */
  unsigned long int level3_cache_size;
  /* _SC_LEVEL3_CACHE_ASSOC.  */
  unsigned long int level3_cache_assoc;
  /* _SC_LEVEL3_CACHE_LINESIZE.  */
  unsigned long int level3_cache_linesize;
  /* /_SC_LEVEL4_CACHE_SIZE.  */
  unsigned long int level4_cache_size;
};

/* Get a pointer to the CPU features structure.  */
extern const struct cpu_features *_dl_x86_get_cpu_features (void)
     __attribute__ ((pure));

#define __get_cpu_features() _dl_x86_get_cpu_features()

#if defined (_LIBC) && !IS_IN (nonlib)
/* Unused for x86.  */
# define INIT_ARCH()
# define _dl_x86_get_cpu_features() (&GLRO(dl_x86_cpu_features))
extern void _dl_x86_init_cpu_features (void) attribute_hidden;
#endif

#ifdef __x86_64__
# define HAS_CPUID 1
#elif (defined __i586__ || defined __pentium__	\
	|| defined __geode__ || defined __k6__)
# define HAS_CPUID 1
# define HAS_I586 1
# define HAS_I686 HAS_ARCH_FEATURE (I686)
#elif defined __i486__
# define HAS_CPUID 0
# define HAS_I586 HAS_ARCH_FEATURE (I586)
# define HAS_I686 HAS_ARCH_FEATURE (I686)
#else
# define HAS_CPUID 1
# define HAS_I586 1
# define HAS_I686 1
#endif

#endif /* include/cpu-features.h */
