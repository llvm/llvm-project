//===-- cpu_model.c - Support for __cpu_model builtin  ------------*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file is based on LLVM's lib/Support/Host.cpp.
//  It implements the operating system Host concept and builtin
//  __cpu_model for the compiler_rt library, for x86 only.
//
//===----------------------------------------------------------------------===//

#if (defined(__i386__) || defined(_M_IX86) || \
     defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))

#include <assert.h>

#define bool int
#define true 1
#define false 0

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifndef __has_attribute
#define __has_attribute(attr) 0
#endif

enum VendorSignatures {
  SIG_INTEL = 0x756e6547 /* Genu */,
  SIG_AMD = 0x68747541 /* Auth */
};

enum ProcessorVendors {
  VENDOR_INTEL = 1,
  VENDOR_AMD,
  VENDOR_OTHER,
  VENDOR_MAX
};

enum ProcessorTypes {
  INTEL_ATOM = 1,
  INTEL_CORE2,
  INTEL_COREI7,
  AMDFAM10H,
  AMDFAM15H,
  INTEL_i386,
  INTEL_i486,
  INTEL_PENTIUM,
  INTEL_PENTIUM_PRO,
  INTEL_PENTIUM_II,
  INTEL_PENTIUM_III,
  INTEL_PENTIUM_IV,
  INTEL_PENTIUM_M,
  INTEL_CORE_DUO,
  INTEL_XEONPHI,
  INTEL_X86_64,
  INTEL_NOCONA,
  INTEL_PRESCOTT,
  AMD_i486,
  AMDPENTIUM,
  AMDATHLON,
  AMDFAM14H,
  AMDFAM16H,
  CPU_TYPE_MAX
};

enum ProcessorSubtypes {
  INTEL_COREI7_NEHALEM = 1,
  INTEL_COREI7_WESTMERE,
  INTEL_COREI7_SANDYBRIDGE,
  AMDFAM10H_BARCELONA,
  AMDFAM10H_SHANGHAI,
  AMDFAM10H_ISTANBUL,
  AMDFAM15H_BDVER1,
  AMDFAM15H_BDVER2,
  INTEL_PENTIUM_MMX,
  INTEL_CORE2_65,
  INTEL_CORE2_45,
  INTEL_COREI7_IVYBRIDGE,
  INTEL_COREI7_HASWELL,
  INTEL_COREI7_BROADWELL,
  INTEL_COREI7_SKYLAKE,
  INTEL_COREI7_SKYLAKE_AVX512,
  INTEL_ATOM_BONNELL,
  INTEL_ATOM_SILVERMONT,
  INTEL_KNIGHTS_LANDING,
  AMDPENTIUM_K6,
  AMDPENTIUM_K62,
  AMDPENTIUM_K63,
  AMDPENTIUM_GEODE,
  AMDATHLON_TBIRD,
  AMDATHLON_MP,
  AMDATHLON_XP,
  AMDATHLON_K8SSE3,
  AMDATHLON_OPTERON,
  AMDATHLON_FX,
  AMDATHLON_64,
  AMD_BTVER1,
  AMD_BTVER2,
  AMDFAM15H_BDVER3,
  AMDFAM15H_BDVER4,
  CPU_SUBTYPE_MAX
};

enum ProcessorFeatures {
  FEATURE_CMOV = 0,
  FEATURE_MMX,
  FEATURE_POPCNT,
  FEATURE_SSE,
  FEATURE_SSE2,
  FEATURE_SSE3,
  FEATURE_SSSE3,
  FEATURE_SSE4_1,
  FEATURE_SSE4_2,
  FEATURE_AVX,
  FEATURE_AVX2,
  FEATURE_AVX512,
  FEATURE_AVX512SAVE,
  FEATURE_MOVBE,
  FEATURE_ADX,
  FEATURE_EM64T
};

// The check below for i386 was copied from clang's cpuid.h (__get_cpuid_max).
// Check motivated by bug reports for OpenSSL crashing on CPUs without CPUID
// support. Consequently, for i386, the presence of CPUID is checked first
// via the corresponding eflags bit.
static bool isCpuIdSupported() {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__i386__)
  int __cpuid_supported;
  __asm__("  pushfl\n"
          "  popl   %%eax\n"
          "  movl   %%eax,%%ecx\n"
          "  xorl   $0x00200000,%%eax\n"
          "  pushl  %%eax\n"
          "  popfl\n"
          "  pushfl\n"
          "  popl   %%eax\n"
          "  movl   $0,%0\n"
          "  cmpl   %%eax,%%ecx\n"
          "  je     1f\n"
          "  movl   $1,%0\n"
          "1:"
          : "=r"(__cpuid_supported)
          :
          : "eax", "ecx");
  if (!__cpuid_supported)
    return false;
#endif
  return true;
#endif
  return true;
}

// This code is copied from lib/Support/Host.cpp.
// Changes to either file should be mirrored in the other.

/// getX86CpuIDAndInfo - Execute the specified cpuid and return the 4 values in
/// the specified arguments.  If we can't run cpuid on the host, return true.
static void getX86CpuIDAndInfo(unsigned value, unsigned *rEAX, unsigned *rEBX,
                               unsigned *rECX, unsigned *rEDX) {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__)
  // gcc doesn't know cpuid would clobber ebx/rbx. Preseve it manually.
  __asm__("movq\t%%rbx, %%rsi\n\t"
          "cpuid\n\t"
          "xchgq\t%%rbx, %%rsi\n\t"
          : "=a"(*rEAX), "=S"(*rEBX), "=c"(*rECX), "=d"(*rEDX)
          : "a"(value));
#elif defined(__i386__)
  __asm__("movl\t%%ebx, %%esi\n\t"
          "cpuid\n\t"
          "xchgl\t%%ebx, %%esi\n\t"
          : "=a"(*rEAX), "=S"(*rEBX), "=c"(*rECX), "=d"(*rEDX)
          : "a"(value));
// pedantic #else returns to appease -Wunreachable-code (so we don't generate
// postprocessed code that looks like "return true; return false;")
#else
  assert(0 && "This method is defined only for x86.");
#endif
#elif defined(_MSC_VER)
  // The MSVC intrinsic is portable across x86 and x64.
  int registers[4];
  __cpuid(registers, value);
  *rEAX = registers[0];
  *rEBX = registers[1];
  *rECX = registers[2];
  *rEDX = registers[3];
#else
  assert(0 && "This method is defined only for GNUC, Clang or MSVC.");
#endif
}

/// getX86CpuIDAndInfoEx - Execute the specified cpuid with subleaf and return
/// the 4 values in the specified arguments.  If we can't run cpuid on the host,
/// return true.
static void getX86CpuIDAndInfoEx(unsigned value, unsigned subleaf,
                                 unsigned *rEAX, unsigned *rEBX, unsigned *rECX,
                                 unsigned *rEDX) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  // gcc doesn't know cpuid would clobber ebx/rbx. Preserve it manually.
  // FIXME: should we save this for Clang?
  __asm__("movq\t%%rbx, %%rsi\n\t"
          "cpuid\n\t"
          "xchgq\t%%rbx, %%rsi\n\t"
          : "=a"(*rEAX), "=S"(*rEBX), "=c"(*rECX), "=d"(*rEDX)
          : "a"(value), "c"(subleaf));
#elif defined(_MSC_VER)
  int registers[4];
  __cpuidex(registers, value, subleaf);
  *rEAX = registers[0];
  *rEBX = registers[1];
  *rECX = registers[2];
  *rEDX = registers[3];
#else
  assert(0 && "This method is defined only for GNUC, Clang or MSVC.");
#endif
#elif defined(__i386__) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
  __asm__("movl\t%%ebx, %%esi\n\t"
          "cpuid\n\t"
          "xchgl\t%%ebx, %%esi\n\t"
          : "=a"(*rEAX), "=S"(*rEBX), "=c"(*rECX), "=d"(*rEDX)
          : "a"(value), "c"(subleaf));
#elif defined(_MSC_VER)
  __asm {
      mov   eax,value
      mov   ecx,subleaf
      cpuid
      mov   esi,rEAX
      mov   dword ptr [esi],eax
      mov   esi,rEBX
      mov   dword ptr [esi],ebx
      mov   esi,rECX
      mov   dword ptr [esi],ecx
      mov   esi,rEDX
      mov   dword ptr [esi],edx
  }
#else
  assert(0 && "This method is defined only for GNUC, Clang or MSVC.");
#endif
#else
  assert(0 && "This method is defined only for x86.");
#endif
}

// Read control register 0 (XCR0). Used to detect features such as AVX.
static bool getX86XCR0(unsigned *rEAX, unsigned *rEDX) {
#if defined(__GNUC__) || defined(__clang__)
  // Check xgetbv; this uses a .byte sequence instead of the instruction
  // directly because older assemblers do not include support for xgetbv and
  // there is no easy way to conditionally compile based on the assembler used.
  __asm__(".byte 0x0f, 0x01, 0xd0" : "=a"(*rEAX), "=d"(*rEDX) : "c"(0));
  return false;
#elif defined(_MSC_FULL_VER) && defined(_XCR_XFEATURE_ENABLED_MASK)
  unsigned long long Result = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
  *rEAX = Result;
  *rEDX = Result >> 32;
  return false;
#else
  return true;
#endif
}

static void detectX86FamilyModel(unsigned EAX, unsigned *Family,
                                 unsigned *Model) {
  *Family = (EAX >> 8) & 0xf; // Bits 8 - 11
  *Model = (EAX >> 4) & 0xf;  // Bits 4 - 7
  if (*Family == 6 || *Family == 0xf) {
    if (*Family == 0xf)
      // Examine extended family ID if family ID is F.
      *Family += (EAX >> 20) & 0xff; // Bits 20 - 27
    // Examine extended model ID if family ID is 6 or F.
    *Model += ((EAX >> 16) & 0xf) << 4; // Bits 16 - 19
  }
}

static void getIntelProcessorTypeAndSubtype(unsigned int Family,
                                            unsigned int Model,
                                            unsigned int Brand_id,
                                            unsigned int Features,
                                            unsigned *Type, unsigned *Subtype) {
  if (Brand_id != 0)
    return;
  switch (Family) {
  case 3:
    *Type = INTEL_i386;
    break;
  case 4:
    switch (Model) {
    case 0: // Intel486 DX processors
    case 1: // Intel486 DX processors
    case 2: // Intel486 SX processors
    case 3: // Intel487 processors, IntelDX2 OverDrive processors,
            // IntelDX2 processors
    case 4: // Intel486 SL processor
    case 5: // IntelSX2 processors
    case 7: // Write-Back Enhanced IntelDX2 processors
    case 8: // IntelDX4 OverDrive processors, IntelDX4 processors
    default:
      *Type = INTEL_i486;
      break;
    }
  case 5:
    switch (Model) {
    case 1: // Pentium OverDrive processor for Pentium processor (60, 66),
            // Pentium processors (60, 66)
    case 2: // Pentium OverDrive processor for Pentium processor (75, 90,
            // 100, 120, 133), Pentium processors (75, 90, 100, 120, 133,
            // 150, 166, 200)
    case 3: // Pentium OverDrive processors for Intel486 processor-based
            // systems
      *Type = INTEL_PENTIUM;
      break;
    case 4: // Pentium OverDrive processor with MMX technology for Pentium
            // processor (75, 90, 100, 120, 133), Pentium processor with
            // MMX technology (166, 200)
      *Type = INTEL_PENTIUM;
      *Subtype = INTEL_PENTIUM_MMX;
      break;
    default:
      *Type = INTEL_PENTIUM;
      break;
    }
  case 6:
    switch (Model) {
    case 0x01: // Pentium Pro processor
      *Type = INTEL_PENTIUM_PRO;
      break;
    case 0x03: // Intel Pentium II OverDrive processor, Pentium II processor,
               // model 03
    case 0x05: // Pentium II processor, model 05, Pentium II Xeon processor,
               // model 05, and Intel Celeron processor, model 05
    case 0x06: // Celeron processor, model 06
      *Type = INTEL_PENTIUM_II;
      break;
    case 0x07: // Pentium III processor, model 07, and Pentium III Xeon
               // processor, model 07
    case 0x08: // Pentium III processor, model 08, Pentium III Xeon processor,
               // model 08, and Celeron processor, model 08
    case 0x0a: // Pentium III Xeon processor, model 0Ah
    case 0x0b: // Pentium III processor, model 0Bh
      *Type = INTEL_PENTIUM_III;
      break;
    case 0x09: // Intel Pentium M processor, Intel Celeron M processor model 09.
    case 0x0d: // Intel Pentium M processor, Intel Celeron M processor, model
               // 0Dh. All processors are manufactured using the 90 nm process.
    case 0x15: // Intel EP80579 Integrated Processor and Intel EP80579
               // Integrated Processor with Intel QuickAssist Technology
      *Type = INTEL_PENTIUM_M;
      break;
    case 0x0e: // Intel Core Duo processor, Intel Core Solo processor, model
               // 0Eh. All processors are manufactured using the 65 nm process.
      *Type = INTEL_CORE_DUO;
      break;   // yonah
    case 0x0f: // Intel Core 2 Duo processor, Intel Core 2 Duo mobile
               // processor, Intel Core 2 Quad processor, Intel Core 2 Quad
               // mobile processor, Intel Core 2 Extreme processor, Intel
               // Pentium Dual-Core processor, Intel Xeon processor, model
               // 0Fh. All processors are manufactured using the 65 nm process.
    case 0x16: // Intel Celeron processor model 16h. All processors are
               // manufactured using the 65 nm process
      *Type = INTEL_CORE2; // "core2"
      *Subtype = INTEL_CORE2_65;
      break;
    case 0x17: // Intel Core 2 Extreme processor, Intel Xeon processor, model
               // 17h. All processors are manufactured using the 45 nm process.
               //
               // 45nm: Penryn , Wolfdale, Yorkfield (XE)
    case 0x1d: // Intel Xeon processor MP. All processors are manufactured using
               // the 45 nm process.
      *Type = INTEL_CORE2; // "penryn"
      *Subtype = INTEL_CORE2_45;
      break;
    case 0x1a: // Intel Core i7 processor and Intel Xeon processor. All
               // processors are manufactured using the 45 nm process.
    case 0x1e: // Intel(R) Core(TM) i7 CPU         870  @ 2.93GHz.
               // As found in a Summer 2010 model iMac.
    case 0x1f:
    case 0x2e:              // Nehalem EX
      *Type = INTEL_COREI7; // "nehalem"
      *Subtype = INTEL_COREI7_NEHALEM;
      break;
    case 0x25: // Intel Core i7, laptop version.
    case 0x2c: // Intel Core i7 processor and Intel Xeon processor. All
               // processors are manufactured using the 32 nm process.
    case 0x2f: // Westmere EX
      *Type = INTEL_COREI7; // "westmere"
      *Subtype = INTEL_COREI7_WESTMERE;
      break;
    case 0x2a: // Intel Core i7 processor. All processors are manufactured
               // using the 32 nm process.
    case 0x2d:
      *Type = INTEL_COREI7; //"sandybridge"
      *Subtype = INTEL_COREI7_SANDYBRIDGE;
      break;
    case 0x3a:
    case 0x3e:              // Ivy Bridge EP
      *Type = INTEL_COREI7; // "ivybridge"
      *Subtype = INTEL_COREI7_IVYBRIDGE;
      break;

    // Haswell:
    case 0x3c:
    case 0x3f:
    case 0x45:
    case 0x46:
      *Type = INTEL_COREI7; // "haswell"
      *Subtype = INTEL_COREI7_HASWELL;
      break;

    // Broadwell:
    case 0x3d:
    case 0x47:
    case 0x4f:
    case 0x56:
      *Type = INTEL_COREI7; // "broadwell"
      *Subtype = INTEL_COREI7_BROADWELL;
      break;

    // Skylake:
    case 0x4e:
      *Type = INTEL_COREI7; // "skylake-avx512"
      *Subtype = INTEL_COREI7_SKYLAKE_AVX512;
      break;
    case 0x5e:
      *Type = INTEL_COREI7; // "skylake"
      *Subtype = INTEL_COREI7_SKYLAKE;
      break;

    case 0x1c: // Most 45 nm Intel Atom processors
    case 0x26: // 45 nm Atom Lincroft
    case 0x27: // 32 nm Atom Medfield
    case 0x35: // 32 nm Atom Midview
    case 0x36: // 32 nm Atom Midview
      *Type = INTEL_ATOM;
      *Subtype = INTEL_ATOM_BONNELL;
      break; // "bonnell"

    // Atom Silvermont codes from the Intel software optimization guide.
    case 0x37:
    case 0x4a:
    case 0x4d:
    case 0x5a:
    case 0x5d:
    case 0x4c: // really airmont
      *Type = INTEL_ATOM;
      *Subtype = INTEL_ATOM_SILVERMONT;
      break; // "silvermont"

    case 0x57:
      *Type = INTEL_XEONPHI; // knl
      *Subtype = INTEL_KNIGHTS_LANDING;
      break;

    default: // Unknown family 6 CPU, try to guess.
      if (Features & (1 << FEATURE_AVX512)) {
        *Type = INTEL_XEONPHI; // knl
        *Subtype = INTEL_KNIGHTS_LANDING;
        break;
      }
      if (Features & (1 << FEATURE_ADX)) {
        *Type = INTEL_COREI7;
        *Subtype = INTEL_COREI7_BROADWELL;
        break;
      }
      if (Features & (1 << FEATURE_AVX2)) {
        *Type = INTEL_COREI7;
        *Subtype = INTEL_COREI7_HASWELL;
        break;
      }
      if (Features & (1 << FEATURE_AVX)) {
        *Type = INTEL_COREI7;
        *Subtype = INTEL_COREI7_SANDYBRIDGE;
        break;
      }
      if (Features & (1 << FEATURE_SSE4_2)) {
        if (Features & (1 << FEATURE_MOVBE)) {
          *Type = INTEL_ATOM;
          *Subtype = INTEL_ATOM_SILVERMONT;
        } else {
          *Type = INTEL_COREI7;
          *Subtype = INTEL_COREI7_NEHALEM;
        }
        break;
      }
      if (Features & (1 << FEATURE_SSE4_1)) {
        *Type = INTEL_CORE2; // "penryn"
        *Subtype = INTEL_CORE2_45;
        break;
      }
      if (Features & (1 << FEATURE_SSSE3)) {
        if (Features & (1 << FEATURE_MOVBE)) {
          *Type = INTEL_ATOM;
          *Subtype = INTEL_ATOM_BONNELL; // "bonnell"
        } else {
          *Type = INTEL_CORE2; // "core2"
          *Subtype = INTEL_CORE2_65;
        }
        break;
      }
      if (Features & (1 << FEATURE_EM64T)) {
        *Type = INTEL_X86_64;
        break; // x86-64
      }
      if (Features & (1 << FEATURE_SSE2)) {
        *Type = INTEL_PENTIUM_M;
        break;
      }
      if (Features & (1 << FEATURE_SSE)) {
        *Type = INTEL_PENTIUM_III;
        break;
      }
      if (Features & (1 << FEATURE_MMX)) {
        *Type = INTEL_PENTIUM_II;
        break;
      }
      *Type = INTEL_PENTIUM_PRO;
      break;
    }
  case 15: {
    switch (Model) {
    case 0: // Pentium 4 processor, Intel Xeon processor. All processors are
            // model 00h and manufactured using the 0.18 micron process.
    case 1: // Pentium 4 processor, Intel Xeon processor, Intel Xeon
            // processor MP, and Intel Celeron processor. All processors are
            // model 01h and manufactured using the 0.18 micron process.
    case 2: // Pentium 4 processor, Mobile Intel Pentium 4 processor - M,
            // Intel Xeon processor, Intel Xeon processor MP, Intel Celeron
            // processor, and Mobile Intel Celeron processor. All processors
            // are model 02h and manufactured using the 0.13 micron process.
      *Type =
          ((Features & (1 << FEATURE_EM64T)) ? INTEL_X86_64 : INTEL_PENTIUM_IV);
      break;

    case 3: // Pentium 4 processor, Intel Xeon processor, Intel Celeron D
            // processor. All processors are model 03h and manufactured using
            // the 90 nm process.
    case 4: // Pentium 4 processor, Pentium 4 processor Extreme Edition,
            // Pentium D processor, Intel Xeon processor, Intel Xeon
            // processor MP, Intel Celeron D processor. All processors are
            // model 04h and manufactured using the 90 nm process.
    case 6: // Pentium 4 processor, Pentium D processor, Pentium processor
            // Extreme Edition, Intel Xeon processor, Intel Xeon processor
            // MP, Intel Celeron D processor. All processors are model 06h
            // and manufactured using the 65 nm process.
      *Type =
          ((Features & (1 << FEATURE_EM64T)) ? INTEL_NOCONA : INTEL_PRESCOTT);
      break;

    default:
      *Type =
          ((Features & (1 << FEATURE_EM64T)) ? INTEL_X86_64 : INTEL_PENTIUM_IV);
      break;
    }
  }
  default:
    break; /*"generic"*/
  }
}

static void getAMDProcessorTypeAndSubtype(unsigned int Family,
                                          unsigned int Model,
                                          unsigned int Features, unsigned *Type,
                                          unsigned *Subtype) {
  // FIXME: this poorly matches the generated SubtargetFeatureKV table.  There
  // appears to be no way to generate the wide variety of AMD-specific targets
  // from the information returned from CPUID.
  switch (Family) {
  case 4:
    *Type = AMD_i486;
  case 5:
    *Type = AMDPENTIUM;
    switch (Model) {
    case 6:
    case 7:
      *Subtype = AMDPENTIUM_K6;
      break; // "k6"
    case 8:
      *Subtype = AMDPENTIUM_K62;
      break; // "k6-2"
    case 9:
    case 13:
      *Subtype = AMDPENTIUM_K63;
      break; // "k6-3"
    case 10:
      *Subtype = AMDPENTIUM_GEODE;
      break; // "geode"
    default:
      break;
    }
  case 6:
    *Type = AMDATHLON;
    switch (Model) {
    case 4:
      *Subtype = AMDATHLON_TBIRD;
      break; // "athlon-tbird"
    case 6:
    case 7:
    case 8:
      *Subtype = AMDATHLON_MP;
      break; // "athlon-mp"
    case 10:
      *Subtype = AMDATHLON_XP;
      break; // "athlon-xp"
    default:
      break;
    }
  case 15:
    *Type = AMDATHLON;
    if (Features & (1 << FEATURE_SSE3)) {
      *Subtype = AMDATHLON_K8SSE3;
      break; // "k8-sse3"
    }
    switch (Model) {
    case 1:
      *Subtype = AMDATHLON_OPTERON;
      break; // "opteron"
    case 5:
      *Subtype = AMDATHLON_FX;
      break; // "athlon-fx"; also opteron
    default:
      *Subtype = AMDATHLON_64;
      break; // "athlon64"
    }
  case 16:
    *Type = AMDFAM10H; // "amdfam10"
    switch (Model) {
    case 2:
      *Subtype = AMDFAM10H_BARCELONA;
      break;
    case 4:
      *Subtype = AMDFAM10H_SHANGHAI;
      break;
    case 8:
      *Subtype = AMDFAM10H_ISTANBUL;
      break;
    default:
      break;
    }
  case 20:
    *Type = AMDFAM14H;
    *Subtype = AMD_BTVER1;
    break; // "btver1";
  case 21:
    *Type = AMDFAM15H;
    if (!(Features &
          (1 << FEATURE_AVX))) { // If no AVX support, provide a sane fallback.
      *Subtype = AMD_BTVER1;
      break; // "btver1"
    }
    if (Model >= 0x50 && Model <= 0x6f) {
      *Subtype = AMDFAM15H_BDVER4;
      break; // "bdver4"; 50h-6Fh: Excavator
    }
    if (Model >= 0x30 && Model <= 0x3f) {
      *Subtype = AMDFAM15H_BDVER3;
      break; // "bdver3"; 30h-3Fh: Steamroller
    }
    if (Model >= 0x10 && Model <= 0x1f) {
      *Subtype = AMDFAM15H_BDVER2;
      break; // "bdver2"; 10h-1Fh: Piledriver
    }
    if (Model <= 0x0f) {
      *Subtype = AMDFAM15H_BDVER1;
      break; // "bdver1"; 00h-0Fh: Bulldozer
    }
    break;
  case 22:
    *Type = AMDFAM16H;
    if (!(Features &
          (1 << FEATURE_AVX))) { // If no AVX support provide a sane fallback.
      *Subtype = AMD_BTVER1;
      break; // "btver1";
    }
    *Subtype = AMD_BTVER2;
    break; // "btver2"
  default:
    break; // "generic"
  }
}

static unsigned getAvailableFeatures(unsigned int ECX, unsigned int EDX,
                                     unsigned MaxLeaf) {
  unsigned Features = 0;
  unsigned int EAX, EBX;
  Features |= (((EDX >> 23) & 1) << FEATURE_MMX);
  Features |= (((EDX >> 25) & 1) << FEATURE_SSE);
  Features |= (((EDX >> 26) & 1) << FEATURE_SSE2);
  Features |= (((ECX >> 0) & 1) << FEATURE_SSE3);
  Features |= (((ECX >> 9) & 1) << FEATURE_SSSE3);
  Features |= (((ECX >> 19) & 1) << FEATURE_SSE4_1);
  Features |= (((ECX >> 20) & 1) << FEATURE_SSE4_2);
  Features |= (((ECX >> 22) & 1) << FEATURE_MOVBE);

  // If CPUID indicates support for XSAVE, XRESTORE and AVX, and XGETBV
  // indicates that the AVX registers will be saved and restored on context
  // switch, then we have full AVX support.
  const unsigned AVXBits = (1 << 27) | (1 << 28);
  bool HasAVX = ((ECX & AVXBits) == AVXBits) && !getX86XCR0(&EAX, &EDX) &&
                ((EAX & 0x6) == 0x6);
  bool HasAVX512Save = HasAVX && ((EAX & 0xe0) == 0xe0);
  bool HasLeaf7 = MaxLeaf >= 0x7;
  getX86CpuIDAndInfoEx(0x7, 0x0, &EAX, &EBX, &ECX, &EDX);
  bool HasADX = HasLeaf7 && ((EBX >> 19) & 1);
  bool HasAVX2 = HasAVX && HasLeaf7 && (EBX & 0x20);
  bool HasAVX512 = HasLeaf7 && HasAVX512Save && ((EBX >> 16) & 1);
  Features |= (HasAVX << FEATURE_AVX);
  Features |= (HasAVX2 << FEATURE_AVX2);
  Features |= (HasAVX512 << FEATURE_AVX512);
  Features |= (HasAVX512Save << FEATURE_AVX512SAVE);
  Features |= (HasADX << FEATURE_ADX);

  getX86CpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
  Features |= (((EDX >> 29) & 0x1) << FEATURE_EM64T);
  return Features;
}

#if defined(HAVE_INIT_PRIORITY)
#define CONSTRUCTOR_ATTRIBUTE __attribute__((__constructor__ 101))
#elif __has_attribute(__constructor__)
#define CONSTRUCTOR_ATTRIBUTE __attribute__((__constructor__))
#else
// FIXME: For MSVC, we should make a function pointer global in .CRT$X?? so that
// this runs during initialization.
#define CONSTRUCTOR_ATTRIBUTE
#endif

int __cpu_indicator_init(void) CONSTRUCTOR_ATTRIBUTE;

struct __processor_model {
  unsigned int __cpu_vendor;
  unsigned int __cpu_type;
  unsigned int __cpu_subtype;
  unsigned int __cpu_features[1];
} __cpu_model = {0, 0, 0, {0}};

/* A constructor function that is sets __cpu_model and __cpu_features with
   the right values.  This needs to run only once.  This constructor is
   given the highest priority and it should run before constructors without
   the priority set.  However, it still runs after ifunc initializers and
   needs to be called explicitly there.  */

int CONSTRUCTOR_ATTRIBUTE
__cpu_indicator_init(void) {
  unsigned int EAX, EBX, ECX, EDX;
  unsigned int MaxLeaf = 5;
  unsigned int Vendor;
  unsigned int Model, Family, Brand_id;
  unsigned int Features = 0;

  /* This function needs to run just once.  */
  if (__cpu_model.__cpu_vendor)
    return 0;

  if (!isCpuIdSupported())
    return -1;

  /* Assume cpuid insn present. Run in level 0 to get vendor id. */
  getX86CpuIDAndInfo(0, &MaxLeaf, &Vendor, &ECX, &EDX);

  if (MaxLeaf < 1) {
    __cpu_model.__cpu_vendor = VENDOR_OTHER;
    return -1;
  }
  getX86CpuIDAndInfo(1, &EAX, &EBX, &ECX, &EDX);
  detectX86FamilyModel(EAX, &Family, &Model);
  Brand_id = EBX & 0xff;

  /* Find available features. */
  Features = getAvailableFeatures(ECX, EDX, MaxLeaf);
  __cpu_model.__cpu_features[0] = Features;

  if (Vendor == SIG_INTEL) {
    /* Get CPU type.  */
    getIntelProcessorTypeAndSubtype(Family, Model, Brand_id, Features,
                                    &(__cpu_model.__cpu_type),
                                    &(__cpu_model.__cpu_subtype));
    __cpu_model.__cpu_vendor = VENDOR_INTEL;
  } else if (Vendor == SIG_AMD) {
    /* Get CPU type.  */
    getAMDProcessorTypeAndSubtype(Family, Model, Features,
                                  &(__cpu_model.__cpu_type),
                                  &(__cpu_model.__cpu_subtype));
    __cpu_model.__cpu_vendor = VENDOR_AMD;
  } else
    __cpu_model.__cpu_vendor = VENDOR_OTHER;

  assert(__cpu_model.__cpu_vendor < VENDOR_MAX);
  assert(__cpu_model.__cpu_type < CPU_TYPE_MAX);
  assert(__cpu_model.__cpu_subtype < CPU_SUBTYPE_MAX);

  return 0;
}

#endif
