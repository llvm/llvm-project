/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>

typedef enum    {
    RES4_EAX = 0,         /* Index of eax in res[4] structure */
    RES4_EBX = 1,         /* Index of ebx in res[4] structure */
    RES4_ECX = 2,         /* Index of ecx in res[4] structure */
    RES4_EDX = 3,         /* Index of edx in res[4] structure */
} RES4_E;

extern int __pgi_cpuid_getmax(uint32_t);

/*
 * __pgi_cpuid( uint32_t id, uint32_t *res):
 * __pgcpuid( uint32_t id, uint32_t *res):
 *
 * uint32_t res[4];
 *     res[0] <-- %eax
 *     res[1] <-- %ebx
 *     res[2] <-- %ecx
 *     res[3] <-- %edx
 */

extern int __pgi_cpuid(uint32_t, uint32_t *);
extern int __pgcpuid(uint32_t, uint32_t *);

/*
 * __pgi_cpuid_ecx( uint32_t id, uint32_t *res, uint32_t ecx ):
 *
 * uint32_t res[4];
 *     res[0] <-- %eax
 *     res[1] <-- %ebx
 *     res[2] <-- %ecx
 *     res[3] <-- %edx
 */

extern int __pgi_cpuid_ecx(uint32_t, uint32_t *, uint32_t );

/*
 * __pgi_getbv( uint32_t xcr_reg, uint64_t *xcr_result):
 *     xcr_result[31: 0] <-- %eax
 *     xcr_result[63:32] <-- %edx
 */

extern int __pgi_getbv(uint32_t, uint64_t *);

/*
 ***********************************************************************
 * Non-vendor specific information
 */

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0000h ) - Largest Standard Function Value and Vendor String
 *  eax = value
 *  ebx || edx || ecx gives a 12-character vendor string:
 *   GenuineIntel
 *   AuthenticAMD
 *   UMC UMC UMC
 *   CyrixInstead
 *   NexGenDriven
 *   CentaurHauls
 *   RiseRiseRise
 *   SiS SiS SiS
 *   GenuineTMx86
 *   Geode by NSC
 */

typedef union CPU0 {
  unsigned int i[4];
  struct {
    int largest; /* largest standard function value */
    char vendor[12];
  } b;
} CPU0;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0000h ) - Highest Extended Function Available
 *  eax = highest 8000 00xx function available
 *  ebx = reserved
 *  ecx = reserved
 *  edx = reserved
 */

typedef union CPU80 {
  unsigned int i[4];
  struct {
    unsigned int largest; /* largest standard function value */
    int ebx, ecx, edx;
  } b;
} CPU80;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0002h ) - Processor Name String
 *  eax = chars 0..3
 *  ebx = chars 4..7
 *  ecx = chars 8..11
 *  edx = chars 12..15
 *
 * CPUID( 8000 0003h ) - Processor Name String
 *  eax = chars 16..19
 *  ebx = chars 20..23
 *  ecx = chars 24..27
 *  edx = chars 28..31
 *
 * CPUID( 8000 0004h ) - Processor Name String
 *  eax = chars 32..35
 *  ebx = chars 36..39
 *  ecx = chars 40..43
 *  edx = chars 44..47
 */

typedef union Xname {
  unsigned int i[4];
  char name[16];
} Xname;

/*
 ***********************************************************************
 * Intel-specific information
 */

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0001h ) - Processor Version Information and Feature Flags
 *  eax = Int_Version
 *  ebx = Int_Brand
 *  ecx = Int_Feature2
 *  edx = Int_Feature1
 */

typedef union ICPU1 {
  unsigned int i[4];
  struct {
    struct Int_Version {
      unsigned int stepping : 4; /* processor stepping / revision */
      unsigned int model : 4;    /* processor model */
      unsigned int family : 4;   /* processor family, Pentium 4 is 1111 */
      unsigned int ptype : 2;    /* processor type */
      unsigned int rs : 2;
      unsigned int extmodel : 4;  /* extended model information, if model=F */
      unsigned int extfamily : 8; /* extended family information, if family=F */
      unsigned int rs2 : 4;
    } eax;
    /* Notes:
     *  ptype is
     *   00 - Original OEM processor
     *   01 - Overdrive
     *   10 - dual processor
     *   11 - reserved
     */
    struct Int_Brand {
      unsigned int brandindex : 8; /* 8-bit brand index */
      unsigned int clflush : 8;    /* CLFLUSH size == 8*cache line size */
      unsigned int proccount : 8;  /* maximum logical processor count. The
                                    * nearest power-of-2 integer that is
                                    * not smaller is the number of unique
                                    * APIC IDs; this field is valid only
                                    * if CPUID.1.edx.htt is set.
                                    */
      unsigned int apic : 8;       /* initial local APIC physical ID */
    } ebx;
    /* Notes:
     *  brandindex is:
     *    0 - unsupported
     *    1 - Celeron
     *    2 - Pentium III
     *    3 - Pentium III Xeon
     *    4 - Pentium III
     *    6 - Mobile Pentium III-M
     *    7 - Mobile Celeron
     *    8 - Pentium 4
     *    9 - Pentium 4
     *   10 - Celeron
     *   11 - Xeon (or Xeon MP)
     *   12 - Xeon MP
     *   14 - Mobile Pentium 4-M
     *   15 - Mobile Celeron
     */
    struct Int_Feature2 {
      unsigned int sse3 : 1;      /* 0:SSE3 */
      unsigned int pclmulqdq : 1; /* 1: PCLMULQDQ instruction */
      unsigned int rs2 : 1;
      unsigned int mon : 1; /* 3:monitor/mwait */
      unsigned int cpl : 1; /* 4:CPL qualified debug store */
      unsigned int vmx : 1; /* 5:virtual machine technology */
      unsigned int rs6 : 1;
      unsigned int est : 1;   /* 7:speedstep technology */
      unsigned int tm2 : 1;   /* 8:thermal monitor 2 */
      unsigned int ssse3 : 1; /* 9:Supplemental SSE 4/SSSE3/mni/core2 */
      unsigned int cnxt : 1;  /* 10:L1 context ID */
      unsigned int rs11 : 1;
      unsigned int fma : 1;  /* 12:FMA - FMA extensions in YMM */
      unsigned int cx16 : 1; /* 13:compare/exchange 16-bytes instruction */
      unsigned int xtpr : 1; /* 14:xTPR update control */
      unsigned int pdcm : 1; /* 15:perf/debug capability MSR */
      unsigned int rsx : 2;
      unsigned int dca : 1;   /* 18:DCA - direct cache access */
      unsigned int sse41 : 1; /* 19:SSE 4.1 */
      unsigned int sse42 : 1; /* 20:SSE 4.2 */
      unsigned int apic : 1;  /* 21:x2APIC */
      unsigned int rsy : 1;
      unsigned int popcnt : 1; /* 23:POPCNT instruction */
      unsigned int rsy2 : 1;
      unsigned int aes : 1;     /* 25:AES instruction */
      unsigned int xsave : 1;   /* 26:XSAVE save extended states */
      unsigned int osxsave : 1; /* 27:OSXSAVE - XSAVE supported by OS */
      unsigned int avx : 1;     /* 28:AVX instructions */
      unsigned int f16c: 1;     /* 29:16-bit FP conversion instructions */
      unsigned int rdrand : 1;  /* 30:RDRAND instruction */
      unsigned int rsz : 1;
    } ecx;

    struct Int_Feature1 {
      unsigned int fpu : 1;  /*  0:floating point unit on chip */
      unsigned int vme : 1;  /*  1:virtual mode extension */
      unsigned int de : 1;   /*  2:debugging extension */
      unsigned int pse : 1;  /*  3:page size extension */
      unsigned int tsc : 1;  /*  4:time stamp counter */
      unsigned int msr : 1;  /*  5:model specific registers */
      unsigned int pae : 1;  /*  6:physical address extension */
      unsigned int mce : 1;  /*  7:machine check exception */
      unsigned int cx8 : 1;  /*  8:compare/exchange 8-bytes instruction */
      unsigned int apic : 1; /*  9:on chip APIC hardware */
      unsigned int rs10 : 1;
      unsigned int sep : 1;   /* 11:fast system call */
      unsigned int mtrr : 1;  /* 12:memory type range registers */
      unsigned int pge : 1;   /* 13:page global enable */
      unsigned int mca : 1;   /* 14:machine check architecture */
      unsigned int cmov : 1;  /* 15:conditional move */
      unsigned int pat : 1;   /* 16:page attribute table */
      unsigned int pseg : 1;  /* 17:36-bit page size extensions */
      unsigned int psn : 1;   /* 18:processor serial number */
      unsigned int cflsh : 1; /* 19:cflush */
      unsigned int rs20 : 1;
      unsigned int dtes : 1; /* 21:debug store */
      unsigned int
          acpi : 1; /* 22:thermal monitor and software controlled clock */
      unsigned int mmx : 1;    /* 23:mmx extensions */
      unsigned int fxsr : 1;   /* 24:fast floating point save/restore */
      unsigned int sse : 1;    /* 25:streaming SIMD extensions */
      unsigned int sse2 : 1;   /* 26:streaming SIMD extensions 2 */
      unsigned int slfsnp : 1; /* 27:self-snoop */
      unsigned int htt : 1;    /* 28:hyper-threading technology */
      unsigned int tm : 1;     /* 29:thermal monitor */
      unsigned int rs30 : 1;
      unsigned int ferr : 1; /* 31:FERR signalling change*/
    } edx;
  } u;
} ICPU1;

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0002h ) - Cache and TLB Information
 *  eax = Int_Cache
 *  ebx = Int_Cache
 *  ecx = Int_Cache
 *  edx = Int_Cache
 */

typedef union ICPU2 {
  unsigned int i[4];
  struct Int_Cache {
    unsigned int c1 : 8, c2 : 8, c3 : 8, c4 : 7; /* see below */
    unsigned int invalid : 1; /* if set, no information here */
  } u[4];
} ICPU2;

/* Notes:
 *  c1 for eax is the number of times that CPUID(2) must be called
 *  to get all cache information; it is usually just 1.
 *  Otherwise, * if 'invalid' is not set then the four values in c1/c2/c3/c4
 *  (c2/c3/c4 for eax) may be zero (no information), or may be one of
 *  the following in any order:
 *   00 - no information
 *   01 -   32 entry ITLB 4-way for 4K pages
 *   02 -    2 entry ITLB 2-way for 4M pages
 *   03 -   64 entry DTLB 4-way for 4K pages
 *   04 -    8 entry DTLB 4-way for 4M pages
 *   06 -   8KB L1 Icache 4-way  32b line
 *   08 -  16KB L1 Icache 4-way  32b line
 *   0a -   8KB L1 Dcache 2-way  32b line
 *   0c -  16KB L1 Dcache 4-way  32b line
 *   22 - 512KB L3 cache  4-way  64b line  128b sector
 *   23 -   1MB L3 cache  8-way  64b line  128b sector
 *   25 -   2MB L3 cache  8-way  64b line  128b sector
 *   29 -   4MB L3 cache  8-way  64b line  128b sector
 *   2c -  32KB L1 Dcache 8-way  64b line
 *   30 -  32KB L1 Icache 8-way  64b line
 *   39 - 128KB L2 cache  4-way  64b line  sectored
 *   3b - 128KB L2 cache  2-way  64b line  sectored
 *   3c - 256KB L2 cache  4-way  64b line  sectored
 *   40 - no L3 cache, or no L2 cache if no L2 cache info
 *   41 - 128KB L2 cache  4-way  32b line
 *   42 - 256KB L2 cache  4-way  32b line
 *   43 - 512KB L2 cache  4-way  32b line
 *   44 -   1MB L2 cache  4-way  32b line
 *   45 -   2MB L2 cache  4-way  32b line
 *   50 -   64 entry ITLB for 4K and 2MB/4MB pages
 *   51 -  128 entry ITLB for 4K and 2MB/4MB pages
 *   52 -  256 entry ITLB for 4K and 2MB/4MB pages
 *   5b -   64 entry DTLB for 4K and 2MB/4MB pages
 *   5c -  128 entry DTLB for 4K and 2MB/4MB pages
 *   5d -  256 entry DTLB for 4K and 2MB/4MB pages
 *   60 -  16KB L1 cache  8-way  64b line
 *   66 -   8KB L1 cache  4-way  64b line
 *   67 -  16KB L1 cache  4-way  64b line
 *   68 -  32KB L1 cache  4-way  64b line
 *   70 -  12K uop trace cache, 8-way
 *   71 -  16K uop trace cache, 8-way
 *   72 -  32K uop trace cache, 8-way
 *   79 - 128KB L2 cache  8-way  64b line  128b sector
 *   7a - 256KB L2 cache  8-way  64b line  128b sector
 *   7b - 512KB L2 cache  8-way  64b line  128b sector
 *   7c -   1MB L2 cache  8-way  64b line  128b sector
 *   7d -   2MB L2 cache  8-way  64b line  sectored
 *   7f - 512KB L2 cache  2-way  64b line  sectored
 *   82 - 256KB L2 cache  8-way  32b line
 *   83 - 512KB L2 cache  8-way  32b line
 *   84 -   1MB L2 cache  8-way  33b line
 *   85 -   2MB L2 cache  8-way  32b line
 *   86 - 512KB L2 cache  4-way  64b line
 *   87 -   1MB L2 cache  8-way  64b line
 *   b0 -  128 entry ITLB 4-way for 4K pages
 *   b3 -  128 entry DTLB 4-way for 4K pages
 *   f0 -   64b prefetching
 *   f1 -  128b prefetching
 */

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0003h ) - Reserved
 */

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0004h ) - Deterministic Cache Parameters
 *  eax = Int_Cache_Parms1
 *  ebx = Int_Cache_Parms2
 *  ecx = int - number of sets (-1 which means add one to this value)
 *  edx = Int_Cache_Parms4
 */

typedef union ICPU4 {
  unsigned int i[4];
  struct {
    struct Int_Cache_Parms1 {
      unsigned int cachetype : 5; /* 0-none, 1-data, 2-instruction, 3-unified */
      unsigned int cachelevel : 3; /* 1..n */
      unsigned int selflevel : 1;  /* self-initializing cache level */
      unsigned int fullyassoc : 1; /* fully associative cache */
      unsigned int rs : 4;
      unsigned int nthreads : 12; /* number of threads sharing this cache
                                   * (-1).  The nearest power-of-2 int
                                   * that is not smaller than 1+nthreads
                                   * is the max number of unique APIC IDs
                                   */
      unsigned int ncores : 6;    /* physical cores on the die (-1). The
                                   * nearest power-of-2 int that is not
                                   * smaller than 1+ncores is the max
                                   * number of unique Core_IDs.
                                   */
    } eax;

    struct Int_Cache_Parms2 {
      unsigned int linesize : 12;   /* system coherency line size (-1) */
      unsigned int partitions : 10; /* physical line partitions (-1) */
      unsigned int assoc : 10;      /* associativity */
    } ebx;
    int nsets; /* number of sets */
    struct Int_Cache_Params4 {
      unsigned int
          llcbehavior : 1; /* wbinvd/invd behavior on lower level chaches */
      unsigned int
          llcinclusive : 1; /* cache is inclusive to lower cache levels */
      unsigned int cacheindexing : 1; /* complex cache indexing */
    } edx;
  } u;
} ICPU4;

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0005h ) - Monitor/Mwait
 *  eax = Int_Monitor - smallest
 *  ebx = Int_Monitor - largest
 *  ecx = reserved
 *  edx = reserved
 */

typedef union ICPU5 {
  unsigned int i[4];
  struct {
    struct Int_Monitor {
      unsigned int limit : 16; /* smallest/largest monitor line size in bytes */
      unsigned int rs : 16;
    } smallest;
    struct Int_Monitor largest;
    int ecx;
    int edx;
  } u;
} ICPU5;

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0006h ) - Thermal and Power Management Leaf
 *  eax = Int_Power_Mgmt1
 *  ebx = Int_Power_Mgmt2
 *  ecx = reserved
 *  edx = reserved
 */

typedef union ICPU6 {
  unsigned int i[4];
  struct {
    struct Int_Power_Mgmt1 {
      unsigned int tempsensor : 1;
      unsigned int turboboost : 1;
      unsigned int arat : 1;
      unsigned int rsvd3 : 1;
      unsigned int pln : 1;
      unsigned int ecmd : 1;
      unsigned int ptm : 1;
      unsigned int rsvd : 25;
    } eax;
    struct Int_Power_Mgmt2 {
      unsigned int numinterrupts : 4;
      unsigned int rsvd : 28;
    } ebx;
  } u;
} ICPU6;

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0007h ) - Structured Extended Feature Flags
 *  eax = Int_Monitor - smallest
 *  ebx = Int_Monitoe - largest
 *  ecx = reserved
 *  edx = reserved
 */

typedef union ICPU7 {
  unsigned int i[4];
  struct {
    unsigned int numsubleaves; /* eax */
    struct Int_Feature_7 {
      unsigned int fsgsbase : 1;
      unsigned int ia32_tsc_adjust : 1;
      unsigned int sgx  : 1;
      unsigned int bmi1 : 1;
      unsigned int hle  : 1;
      unsigned int avx2 : 1;
      unsigned int rsv6 : 1;
      unsigned int smep : 1;
      unsigned int bmi2 : 1;
      unsigned int erms : 1;
      unsigned int invpcid : 1;
      unsigned int rtm : 1;
      unsigned int pqm : 1;
      unsigned int depcsds : 1;
      unsigned int memprotect : 1;
      unsigned int pqe : 1;
      unsigned int avx512f : 1;
      unsigned int avx512dq : 1;
      unsigned int rdseed : 1;
      unsigned int adx : 1;
      unsigned int smap : 1;
      unsigned int avx512fma : 1;
      unsigned int rsv22 : 1;
      unsigned int clflushopt : 1;
      unsigned int clwb : 1;
      unsigned int trace : 1;
      unsigned int avx512pf : 1;
      unsigned int avx512er : 1;
      unsigned int avx512cd : 1;
      unsigned int sha : 1;
      unsigned int avx512bw : 1;
      unsigned int avx512vl: 1;
    } ebx;
    struct {
      unsigned int prefetchwt1 : 1;
      unsigned int avx512vbmi : 1;
      unsigned int rsv2 : 1;
      unsigned int pku : 1;
      unsigned int ospke : 1;
      unsigned int rsvd : 27;
    } ecx;
    struct {
      unsigned int rsv0 : 1;
      unsigned int avx512_4vnniw : 1;
      unsigned int avx512_4fmapx : 1;
      unsigned int rsvd : 28;
    } edx;
  } u;
} ICPU7;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0001h ) - Processor Feature Flags
 *  eax = reserved
 *  ebx = reserved
 *  ecx = reserved
 *  edx = Int_XFeature1
 */

typedef union ICPU81 {
  unsigned int i[4];
  struct {
    int eax, ebx, ecx;
    struct Int_XFeature1 {
      unsigned int rs0 : 11;
      unsigned int sep : 1; /* 11:syscall/sysret */
      unsigned int rs12 : 8;
      unsigned int nx : 1; /* 20: no-execute page protection */
      unsigned int rs21 : 8;
      unsigned int lm : 1; /* 29:long mode capable */
      unsigned int rs30 : 2;
    } edx;
  } u;
} ICPU81;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0002h ) - processor name string
 * CPUID( 8000 0003h ) - processor name string continued
 * CPUID( 8000 0004h ) - processor name string continued
 */

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0005h ) - reserved
 *  eax = reserved
 *  ebx = reserved
 *  ecx = reserved
 *  edx = reserved
 */

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0006h ) - Cache information
 *  eax = reserved
 *  ebx = reserved
 *  ecx = Int_Cache_Info
 *  edx = reserved
 */

typedef union ICPU86 {
  unsigned int i[4];
  struct {
    int eax, ebx;
    struct Int_Cache_Info {
      unsigned int linesize : 8; /* cache line size */
      unsigned int rs : 4;
      unsigned int assoc : 4; /* L2 associativity */
      unsigned int size : 16; /* cache size in K */
    } ecx;
    int edx;
  } u;
} ICPU86;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0007h ) - Reserved
 *  eax = reserved
 *  ebx = reserved
 *  ecx = reserved
 *  edx = reserved
 */

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0008h ) - Address Size
 *  eax = Int_Physical
 *  ebx = reserved
 *  ecx = reserved
 *  edx = reserved
 */

typedef union ICPU88 {
  unsigned int i[4];
  struct {
    struct Int_Physical {
      unsigned int physical : 8; /* max physical address width in bits */
      unsigned int virtual : 8;  /* max virtual address width in bits */
      unsigned int rs : 16;
    } eax;
    int ebx, ecx, edx;
  } u;
} ICPU88;

/*
 ***********************************************************************
 * AMD-specific information
 */


/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0001h ) - Processor Version Information and Feature Flags
 *  eax = AMD_Version
 *  ebx = AMD_Brand
 *  ecx = AMD_Feature2
 *  edx = AMD_Feature1
 */

typedef union ACPU1 {
  unsigned int i[4];
  struct {

    struct AMD_Version {
      unsigned int stepping : 4; /* processor stepping / revision */
      unsigned int model : 4;    /* processor model */
      unsigned int family : 4;   /* processor family */
      unsigned int rs1 : 4;      /* reserved */
      unsigned int
          extmodel : 4; /* extended model information, if family == 0 */
      unsigned int
          extfamily : 8;    /* extended family information, if family == 0 */
      unsigned int rs2 : 4; /* reserved */
    } eax;
    /* Notes:
     *  if family==0, Family is extfamily
     *  if family==0, Model is extmodel<<4 + model
     */

    struct AMD_Brand {
      unsigned int
          brandid : 8; /* 8-bit brand ID; 0 means use 12-bit brand ID */
      unsigned int clflush : 8;   /* CLFLUSH size */
      unsigned int proccount : 8; /* logical processor count */
      unsigned int apic : 8;      /* initial local APIC physical ID */
    } ebx;
    /* Notes:
     *  brandid==0 means use the 12-bit brand ID of CPUID( 8000 0001 )
     *  clflush is cache line size in quadwords (8 bytes)
     *    this is only valid if clflush feature bit is set
     *  proccount is valid if cmp_legacy==1 && htt==1, indicates
     *    number of physical cores to legacy software
     *    better to use CPUID( 8000 0008 )
     */

    struct AMD_Feature2 {
      unsigned int sse3 : 1;      /* 0:SSE3 */
      unsigned int pclmulqdq : 1; /* 1: PCLMULQDQ: PCLMULQDQ */
      unsigned int rs1 : 1;
      unsigned int mon : 1; /* 3:monitor/mwait */
      unsigned int rs2 : 5;
      unsigned int ssse3 : 1; /* 9:Supplemental SSE 4/SSSE3/mni/core2 */
      unsigned int rs3 : 2;
      unsigned int fma : 1;  /* 12:FMA - FMA extensions in YMM */
      unsigned int cx16 : 1; /* 13:compare/exchange 16-bytes instruction */
      unsigned int rs14 : 5;
      unsigned int sse41 : 1; /* 19:SSE 4.1 */
      unsigned int sse42 : 1; /* 20:SSE 4.2 */
      unsigned int rs21 : 2;
      unsigned int popcnt : 1; /* 23:POPCNT instruction */
      unsigned int rs24 : 1;
      unsigned int aes : 1;     /* 25:AES instruction */
      unsigned int xsave : 1;   /* 26:XSAVE instruction */
      unsigned int osxsave : 1; /* 27:XSAVE OS */
      unsigned int avx : 1;     /* 28:AVX instructions */
      unsigned int f16c : 1;    /* 29:half-precision convert instruction */
      unsigned int rs30 : 1;
      unsigned int raz : 1; /* 31:reserved for use by hypervisor to indicate
                               guest status */
    } ecx;

    struct AMD_Feature1 {
      unsigned int fpu : 1;  /*  0:floating point unit on chip */
      unsigned int vme : 1;  /*  1:virtual mode extension */
      unsigned int de : 1;   /*  2:debugging extension */
      unsigned int pse : 1;  /*  3:page size extension */
      unsigned int tsc : 1;  /*  4:time stamp counter */
      unsigned int msr : 1;  /*  5:model specific registers (K86 MSR) */
      unsigned int pae : 1;  /*  6:physical address extension */
      unsigned int mce : 1;  /*  7:machine check exception */
      unsigned int cx8 : 1;  /*  8:compare/exchange 8-bytes instruction */
      unsigned int apic : 1; /*  9:on chip APIC hardware */
      unsigned int rs10 : 1;
      unsigned int sep : 1;  /* 11:sysenter/sysexit (>=PIII) */
      unsigned int mtrr : 1; /* 12:memory type range registers */
      unsigned int pge : 1;  /* 13:page global enable */
      unsigned int mca : 1;  /* 14:machine check architecture */
      unsigned int cmov : 1; /* 15:conditional move */
      unsigned int pat : 1;  /* 16:page attribute table */
      unsigned int pseg : 1; /* 17:36-bit page size extensions */
      unsigned int rs18 : 1;
      unsigned int cflsh : 1; /* 19:clflush */
      unsigned int rs20 : 1;
      unsigned int rs21 : 1;
      unsigned int rs22 : 1;
      unsigned int mmx : 1;  /* 23:mmx extensions */
      unsigned int fxsr : 1; /* 24:fast floating point save/restore */
      unsigned int sse : 1;  /* 25:streaming SIMD extensions */
      unsigned int sse2 : 1; /* 26:SSE2 */
      unsigned int rs27 : 1;
      unsigned int htt : 1; /* 28:hyper-threading technology */
      unsigned int rs29 : 1;
      unsigned int rs30 : 1;
      unsigned int rs31 : 1;
    } edx;
  } u;
} ACPU1;

/*
 * ---------------------------------------------------------------------
 * CPUID( 0000 0007h ) - Structured Extended Feature Identifiers
 *  eax = Reserved
 *  ebx = AMD_Brand
 *  ecx = Reserved
 *  edx = Reserved
 */

typedef union ACPU7 {
  unsigned int i[4];
  struct {

    int eax;			/* Reserved */
    struct AMD_Extended_Features {
      unsigned int fsgbase : 1;	/* FS & GS base read/write support */
      unsigned int rs1  : 1;
      unsigned int rs2  : 1;
      unsigned int bmi1 : 1;	/* bit manipluation group 1 */
      unsigned int rs4  : 1;
      unsigned int avx2 : 1;	/* AVX extension support (avx2) */
      unsigned int rs6  : 1;
      unsigned int smep : 1;	/* Supervisor mode execution protection */
      unsigned int bmi2 : 1;	/* bit manipluation group 2 */
      unsigned int rs9  : 1;
      unsigned int rs10 : 1;
      unsigned int rs11 : 1;
      unsigned int rs12 : 1;
      unsigned int rs13 : 1;
      unsigned int rs14 : 1;
      unsigned int rs15 : 1;
      unsigned int rs16 : 1;
      unsigned int rs17 : 1;
      unsigned int rdseed : 1;	/* RDSEED is present */
      unsigned int adx  : 1;	/* ADCX and ADOX are present */
      unsigned int smap : 1;	/* Secure mode access prevention - supported */
      unsigned int rs21 : 1;
      unsigned int pcommit: 1;
      unsigned int clfshopt: 1;
      unsigned int rs24 : 1;
      unsigned int rs25 : 1;
      unsigned int rs26 : 1;
      unsigned int rs27 : 1;
      unsigned int rs29 : 1;
      unsigned int sha  : 1;
      unsigned int rs30 : 1;
      unsigned int rs31 : 1;
    } ebx;
    int ecx, edx;		/* Reserved */

  } u;
} ACPU7;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0001h ) - Processor Version Information and Feature Flags
 *  eax = AMD_Version - same as 0000 0001h
 *  ebx = 12-bit Brand ID
 *  ecx = AMD_XFeature2
 *  edx = AMD_XFeature1
 *
 * Notes:
 *  if the 12-bit Brand ID is zero, use the 8-bit Brand ID
 */

typedef union ACPU81 {
  unsigned int i[4];
  struct {
    struct AMD_Version eax;

    struct AMD_XBrand {
      unsigned int
          brandid : 16; /* 16-bit brand ID; 0 means use 8-bit brand ID */
      unsigned int rs : 16;
    } ebx;
    /* Notes:
     *  brandid==0 means use the 8-bit brand ID of CPUID( 0000 0001 )
     */

    struct AMD_XFeature2 {
      unsigned int ahf : 1;     /* 0:LAHF/SAHF support in long mode */
      unsigned int cmp : 1;     /* 1:CMP_LEGACY */
      unsigned int svm : 1;     /* 2:Secure Virtual Machine */
      unsigned int extapic : 1; /* 3:Extended APIC register space: */
      unsigned int lockmov : 1; /* 4:LOCK MOV CR0 means MOV CR8 */
      unsigned int abm : 1;     /* 5:Advanced Bit Manipulation, POPCNT, LZCNT */
      unsigned int sse4a : 1;   /* 6: EXTRQ,INSERTQ,MOVNT[SS|SD] */
      unsigned int mas : 1;     /* 7:Misaligned SSE mode */
      unsigned int prefetch : 1; /* 8:3DNow prefetch */
      unsigned int osvw : 1;     /*  9:OS visible workaround */
      unsigned int ibs : 1;      /* 10:intstruction based sampling */
      unsigned int xop : 1;      /* 11:extended operation support */
      unsigned int skinit : 1;   /*12: SKINIT & STGI are supported */
      unsigned int wdt : 1;      /* 13:watchdog timer support */
      unsigned int rs14 : 1;
      unsigned int lwp : 1;  /* 15:lightweight profiling support */
      unsigned int fma4 : 1; /* 16:4-operand FMA instructions */
      unsigned int tce : 1;  /* 17:translation cache extension */
      unsigned int rs18 : 1;
      unsigned int
          nodeid : 1; /*19:MSRC001_100C[NodeId,NodesPerProcessor] supported */
      unsigned int rs20 : 1;
      unsigned int tbm : 1;      /* 21:trailing bit manipulation support */
      unsigned int topolext : 1; /*22:topology extensions suppport */
      unsigned int rs23 : 9;
    } ecx;

    struct AMD_XFeature1 {
      unsigned int fpu : 1;  /*  0:floating point unit on chip */
      unsigned int vme : 1;  /*  1:virtual mode extension */
      unsigned int de : 1;   /*  2:debugging extension */
      unsigned int pse : 1;  /*  3:page size extension */
      unsigned int tsc : 1;  /*  4:time stamp counter */
      unsigned int msr : 1;  /*  5:model specific registers (K86 MSR) */
      unsigned int pae : 1;  /*  6:physical address extension */
      unsigned int mce : 1;  /*  7:machine check exception */
      unsigned int cx8 : 1;  /*  8:compare/exchange 8-bytes instruction */
      unsigned int apic : 1; /*  9:on chip APIC hardware */
      unsigned int rs10 : 1;
      unsigned int sep : 1;  /* 11:sysenter/sysexit (>=PIII) */
      unsigned int mtrr : 1; /* 12:memory type range registers */
      unsigned int pge : 1;  /* 13:page global enable */
      unsigned int mca : 1;  /* 14:machine check architecture */
      unsigned int cmov : 1; /* 15:conditional move */
      unsigned int pat : 1;  /* 16:page attribute table */
      unsigned int pseg : 1; /* 17:36-bit page size extensions */
      unsigned int rs18 : 1;
      unsigned int rs19 : 1;
      unsigned int nx : 1; /* 20: no-execute page protection */
      unsigned int rs21 : 1;
      unsigned int ammx : 1;  /* 22:AMD MMX instruction extensions */
      unsigned int mmx : 1;   /* 23:mmx extensions */
      unsigned int fxsr : 1;  /* 24:fast floating point save/restore */
      unsigned int fxsro : 1; /* 25:fxsave/fxrstor optimizations */
      unsigned int rs26 : 1;
      unsigned int rdtscp : 1; /* 27: RDTSCP instruction */
      unsigned int rs28 : 1;
      unsigned int lm : 1;     /* 29:long mode capable */
      unsigned int now3dx : 1; /* 30:3DNow! instructions extensions */
      unsigned int now3d : 1;  /* 31:3DNow! instructions */
    } edx;
  } u;
} ACPU81;
/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0002h ) - processor name string
 * CPUID( 8000 0003h ) - processor name string continued
 * CPUID( 8000 0004h ) - processor name string continued
 */

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0005h ) - L1 Cache and L1 TLB information
 *  eax = AMD_L1_TLB - L1 TLB Large Page Information
 *  ebx = AMD_L1_TLB - L1 TLB 4-Kbyte Page Information
 *  ecx = AMD_L1_Cache - L1 Data Cache Information
 *  edx = AMD_L1_Cache - L1 Instruction Cache Information
 */

typedef union ACPU85 {
  unsigned int i[4];
  struct {
    struct AMD_L1_TLB {
      unsigned int ientries : 8; /* number of entries in instruction TLB */
      unsigned int iassoc : 8;   /* associativity, FF=full in instruction TLB */
      unsigned int dentries : 8; /* number of entries in data TLB */
      unsigned int dassoc : 8;   /* associativity, FF=full in data TLB */
    } tlb_large, tlb_4k;

    struct AMD_L1_Cache {
      unsigned int linesize : 8; /* line size in bytes */
      unsigned int taglines : 8; /* lines per tag */
      unsigned int assoc : 8;    /* associativity */
      unsigned int size : 8;     /* cache size in Kbytes */
    } dcache, icache;
  } u;
} ACPU85;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0006h ) - L1 Cache and L1 TLB information
 *  eax = AMD_L2_TLB - L2 TLB Large Page Information
 *  ebx = AMD_L2_TLB - L2 TLB 4-Kbyte Page Information
 *  ecx = AMD_L2_Cache - L2 (Unified) Cache Information
 *  edx = reserved
 */

typedef union ACPU86 {
  unsigned int i[4];
  struct {
    struct AMD_L2_TLB {
      unsigned int itlb_entries : 12; /* number of entries in instruction TLB */
      unsigned int
          itlb_assoc : 4; /* associativity, FF=full in instruction TLB */
      unsigned int dtlb_entries : 12; /* number of entries in data TLB */
      unsigned int dtlb_assoc : 4;    /* associativity, FF=full in data TLB */
    } tlb_large, tlb_4k;

    struct AMD_L2_Cache {
      unsigned int linesize : 8; /* line size in bytes */
      unsigned int taglines : 4; /* lines per tag */
      unsigned int assoc : 4;    /* associativity */
      unsigned int size : 16;    /* cache size in Kbytes */
    } l2cache;
    struct AMD_L3_Cache {
      unsigned int linesize : 8; /* line size in bytes */
      unsigned int taglines : 4; /* lines per tag */
      unsigned int assoc : 4;    /* associativity */
      unsigned int reserved : 2;
      unsigned int size : 14; /* cache size in half-Mbytes */
    } l3cache;
  } u;
} ACPU86;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0007h ) - Advanced Power Management
 *  eax = reserved
 *  ebx = reserved
 *  ecx = reserved
 *  edx = AMD_Power
 */

typedef union ACPU87 {
  unsigned int i[4];
  struct {
    int eax, ebx, ecx;
    struct AMD_Power {
      unsigned int ts : 1;  /* 0: temperature sensor */
      unsigned int fid : 1; /* 1: frequency ID control */
      unsigned int vid : 1; /* 2: voltage ID control */
      unsigned int ttp : 1; /* 3: thermal trip */
      unsigned int tm : 1;  /* 4: thermal monitoring */
      unsigned int stc : 1; /* 5: software thermal control */
      unsigned int mhz : 1; /* 6: 100 Mhz multiplier control */
      unsigned int rs : 25;
    } edx;
  } u;
} ACPU87;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 0008h ) - Address Size and Physical Core Count
 *  eax = AMD_Physical
 *  ebx = reserved
 *  ecx = AMD_Core_Count
 *  edx = reserved
 */

typedef union ACPU88 {
  unsigned int i[4];
  struct {
    struct AMD_Physical {
      unsigned int physical : 8; /* max physical address width in bits */
      unsigned int virtual : 8;  /* max virtual address width in bits */
      unsigned int rs : 16;
    } eax;

    int ebx;
    struct AMD_Core_Count {
      unsigned int cores : 8; /* number of cores minus one (0 means 1 core) */
      unsigned int rs : 24;
    } ecx;
    int edx;
  } u;
} ACPU88;

/*
 * ---------------------------------------------------------------------
 * CPUID( 8000 001eh ) - Extended APIC / CoreId / NodeId
 *  eax = APIC ID
 *  ebx = Core ID
 *  ecx = Node ID
 *  edx = reserved
 */

typedef union ACPU81e {
  unsigned int i[4];
  struct {
    struct AMD_ExtAPICId {
      unsigned int extendedapicid;
    } eax;

    struct AMD_CoreId {
      unsigned int coreid : 8;
      unsigned int threadspercore : 8; /* number of threads minus one (0 means 1 thread) */
      unsigned int rs : 16;
    } ebx;

    struct AMD_NodeId {
      unsigned int nodeid : 8;
      unsigned int nodesperproc : 3; /* number of threads minus one (0 means 1 thread) */
      unsigned int rs : 21;
    } ecx;
    int edx;
  } u;
} ACPU81e;

#define x80 0x80000000U
