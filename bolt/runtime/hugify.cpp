//===- bolt/runtime/hugify.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#if (defined(__x86_64__) || defined(__aarch64__) || defined(__arm64__)) &&     \
    !defined(__APPLE__)

#include "common.h"

#pragma GCC visibility push(hidden)

// Enables a very verbose logging to stderr useful when debugging
// #define ENABLE_DEBUG

#ifdef ENABLE_DEBUG
#define DEBUG(X)                                                               \
  { X; }
#else
#define DEBUG(X)                                                               \
  {}
#endif

// Function constrains trampoline to _start,
// so we can resume regular execution of the function that we hooked.
extern void __bolt_hugify_start_program();
extern "C" void __bolt_hugify_self_impl();

// The __hot_start and __hot_end symbols set by Bolt. We use them to figure
// out the rage for marking huge pages.
extern uint64_t __hot_start;
extern uint64_t __hot_end;

// Runtime option emitted by BOLT. When non-zero, anonymously remap all
// file-backed executable VMAs for THP instead of only the hot text range.
extern uint32_t __bolt_hugify_all_text;
extern uint32_t __bolt_hugify_all_text_stats;

static constexpr int MADV_HUGEPAGE = 14;
static constexpr size_t HugePageBytes = 2L * 1024 * 1024;
static constexpr size_t MaxMapsBytes = 4L * 1024 * 1024;
static constexpr size_t MaxSmapsBytes = 16L * 1024 * 1024;

static bool hasEnvironmentVariable(const char *Name) {
  int FD = __open("/proc/self/environ", O_RDONLY, 0);
  if (FD < 0)
    return false;

  const uint32_t NameSize = strLen(Name);
  char Buf[256];
  uint32_t NameOffset = 0;
  bool Matches = true;
  while (true) {
    const int64_t ReadSize = static_cast<int64_t>(__read(FD, Buf, sizeof(Buf)));
    if (ReadSize <= 0)
      break;

    for (int64_t I = 0; I < ReadSize; ++I) {
      const char C = Buf[I];
      if (C == '\0') {
        NameOffset = 0;
        Matches = true;
        continue;
      }
      if (!Matches)
        continue;
      if (NameOffset < NameSize) {
        Matches = C == Name[NameOffset++];
        continue;
      }
      if (C == '=') {
        __close(FD);
        return true;
      }
      Matches = false;
    }
  }

  __close(FD);
  return false;
}

static bool getKernelVersion(uint32_t *Val) {
  // release should be in the format: %d.%d.%d
  // major, minor, release
  struct UtsNameTy UtsName;
  if (__uname(&UtsName) < 0)
    return false;
  const char *Buf = UtsName.release;
  const char *End = Buf + strLen(Buf);
  const char Delims[2][2] = {".", "."};

  for (int i = 0; i < 3; ++i) {
    if (!scanUInt32(Buf, End, Val[i]))
      return false;
    if (i < sizeof(Delims) / sizeof(Delims[0])) {
      const char *Ptr = Delims[i];
      while (*Ptr != '\0') {
        if (Buf == End || *Ptr != *Buf)
          return false;
        ++Ptr;
        ++Buf;
      }
    }
  }
  return true;
}

static bool isTHPEnabled() {
  char Buf[64];

  int FD = __open("/sys/kernel/mm/transparent_hugepage/enabled",
                  0 /* O_RDONLY */, 0);
  if (FD < 0)
    return false;

  memset(Buf, 0, sizeof(Buf));
  const int64_t Res = static_cast<int64_t>(__read(FD, Buf, sizeof(Buf) - 1));
  __close(FD);
  if (Res <= 0)
    return false;

  return strStr(Buf, "[always]") || strStr(Buf, "[madvise]");
}

/// File-backed THP is only used on kernels where BOLT knows it is supported.
static bool hasPagecacheTHPSupport() {
  struct KernelVersionTy {
    uint32_t major;
    uint32_t minor;
    uint32_t release;
  } KernelVersion{};

  if (!getKernelVersion((uint32_t *)&KernelVersion))
    return false;
  if (KernelVersion.major >= 6 ||
      (KernelVersion.major == 5 && KernelVersion.minor >= 10))
    return true;

  return false;
}

static bool isHexDigit(char C) {
  return ('0' <= C && C <= '9') || ('a' <= C && C <= 'f') ||
         ('A' <= C && C <= 'F');
}

static const char *parseHex(const char *Buf, const char *End, uint64_t &Val) {
  uint64_t Res = 0;
  const char *Start = Buf;
  while (Buf < End && isHexDigit(*Buf)) {
    Res <<= 4;
    if ('0' <= *Buf && *Buf <= '9')
      Res += *Buf - '0';
    else if ('a' <= *Buf && *Buf <= 'f')
      Res += *Buf - 'a' + 10;
    else
      Res += *Buf - 'A' + 10;
    ++Buf;
  }

  if (Buf == Start)
    return nullptr;
  Val = Res;
  return Buf;
}

static const char *parseDecimal(const char *Buf, const char *End,
                                uint64_t &Val) {
  uint64_t Result = 0;
  const char *Start = Buf;
  while (Buf < End && '0' <= *Buf && *Buf <= '9') {
    Result = Result * 10 + *Buf - '0';
    ++Buf;
  }
  if (Buf == Start)
    return nullptr;
  Val = Result;
  return Buf;
}

struct HugePageStats {
  uint64_t AnonHugeKB{0};
  uint64_t FilePmdKB{0};
  uint64_t VMAs{0};
};

static const char *findLineEnd(const char *Buf, const char *End) {
  while (Buf < End && *Buf != '\n')
    ++Buf;
  return Buf;
}

static void printHugePageStats(const HugePageStats &Stats,
                               uint64_t DeferredKB) {
  char Buf[256];
  char *Ptr = Buf;
  const uint64_t HugeKB = Stats.AnonHugeKB + Stats.FilePmdKB;
  Ptr = strCopy(Ptr, "[hugify] section=text huge_kb=");
  Ptr = intToStr(Ptr, HugeKB, 10);
  Ptr = strCopy(Ptr, " pages_2mb=");
  Ptr = intToStr(Ptr, HugeKB / 2048, 10);
  Ptr = strCopy(Ptr, " anon_kb=");
  Ptr = intToStr(Ptr, Stats.AnonHugeKB, 10);
  Ptr = strCopy(Ptr, " file_kb=");
  Ptr = intToStr(Ptr, Stats.FilePmdKB, 10);
  Ptr = strCopy(Ptr, " vmas=");
  Ptr = intToStr(Ptr, Stats.VMAs, 10);
  Ptr = strCopy(Ptr, " deferred_kb=");
  Ptr = intToStr(Ptr, DeferredKB, 10);
  Ptr = strCopy(Ptr, "\n");
  __write(1, Buf, Ptr - Buf);
}

static const char *skipLine(const char *Buf, const char *End) {
  Buf = findLineEnd(Buf, End);
  if (Buf < End)
    ++Buf;
  return Buf;
}

struct MapsEntry {
  uint64_t Start{0};
  uint64_t End{0};
  char Perms[4]{};
  bool HasFile{false};
};

struct RemapArgs {
  uint8_t *Target;
  const uint8_t *Copy;
  uint64_t Size;
  uint64_t FinalProt;
};

struct DeferredRemap {
  uint8_t *From{nullptr};
  uint8_t *To{nullptr};
  int FinalProt{PROT_NONE};
};

static_assert(offsetof(RemapArgs, Target) == 0);
static_assert(offsetof(RemapArgs, Copy) == 8);
static_assert(offsetof(RemapArgs, Size) == 16);
static_assert(offsetof(RemapArgs, FinalProt) == 24);

extern "C" uint8_t __bolt_hugify_remap_stub_start[];
extern "C" uint8_t __bolt_hugify_remap_stub_end[];

extern "C" __attribute((naked)) bool __bolt_hugify_remap_stub(RemapArgs *Args) {
#if defined(__x86_64__)
  __asm__ __volatile__(".global __bolt_hugify_remap_stub_start\n"
                       "__bolt_hugify_remap_stub_start:\n"
                       "push %rbx\n"
                       "mov %rdi, %rbx\n"
                       "mov $9, %rax\n"
                       "mov 0(%rbx), %rdi\n"
                       "mov 16(%rbx), %rsi\n"
                       "mov 24(%rbx), %rdx\n"
                       "or $3, %rdx\n"
                       "mov $0x32, %r10\n"
                       "mov $-1, %r8\n"
                       "xor %r9d, %r9d\n"
                       "syscall\n"
                       "cmp $-4095, %rax\n"
                       "jae 2f\n"
                       "mov $28, %rax\n"
                       "mov 0(%rbx), %rdi\n"
                       "mov 16(%rbx), %rsi\n"
                       "mov $14, %rdx\n"
                       "syscall\n"
                       "mov 0(%rbx), %rdi\n"
                       "mov 8(%rbx), %rsi\n"
                       "mov 16(%rbx), %rcx\n"
                       "rep movsb\n"
                       "mov $10, %rax\n"
                       "mov 0(%rbx), %rdi\n"
                       "mov 16(%rbx), %rsi\n"
                       "mov 24(%rbx), %rdx\n"
                       "syscall\n"
                       "mov $1, %eax\n"
                       "pop %rbx\n"
                       "ret\n"
                       "2:\n"
                       "xor %eax, %eax\n"
                       "pop %rbx\n"
                       "ret\n"
                       ".global __bolt_hugify_remap_stub_end\n"
                       "__bolt_hugify_remap_stub_end:\n");
#elif defined(__aarch64__) || defined(__arm64__)
  __asm__ __volatile__(".global __bolt_hugify_remap_stub_start\n"
                       "__bolt_hugify_remap_stub_start:\n"
                       "mov x9, x0\n"
                       "ldr x0, [x9, #0]\n"
                       "ldr x1, [x9, #16]\n"
                       "ldr x2, [x9, #24]\n"
                       "orr x2, x2, #3\n"
                       "mov x3, #0x32\n"
                       "mov x4, #-1\n"
                       "mov x5, #0\n"
                       "mov x8, #222\n"
                       "svc #0\n"
                       "cmp x0, #0\n"
                       "b.lt 2f\n"
                       "ldr x0, [x9, #0]\n"
                       "ldr x1, [x9, #16]\n"
                       "mov x2, #14\n"
                       "mov x8, #233\n"
                       "svc #0\n"
                       "ldr x10, [x9, #0]\n"
                       "ldr x11, [x9, #8]\n"
                       "ldr x12, [x9, #16]\n"
                       "1:\n"
                       "ldr x13, [x11], #8\n"
                       "str x13, [x10], #8\n"
                       "subs x12, x12, #8\n"
                       "b.ne 1b\n"
                       "mrs x14, ctr_el0\n"
                       "ubfx x15, x14, #16, #4\n"
                       "mov x17, #4\n"
                       "lsl x17, x17, x15\n"
                       "sub x15, x17, #1\n"
                       "ldr x10, [x9, #0]\n"
                       "bic x10, x10, x15\n"
                       "ldr x12, [x9, #0]\n"
                       "ldr x13, [x9, #16]\n"
                       "add x12, x12, x13\n"
                       "3:\n"
                       "dc cvau, x10\n"
                       "add x10, x10, x17\n"
                       "cmp x10, x12\n"
                       "b.lo 3b\n"
                       "dsb ish\n"
                       "ubfx x15, x14, #0, #4\n"
                       "mov x17, #4\n"
                       "lsl x17, x17, x15\n"
                       "sub x15, x17, #1\n"
                       "ldr x10, [x9, #0]\n"
                       "bic x10, x10, x15\n"
                       "4:\n"
                       "ic ivau, x10\n"
                       "add x10, x10, x17\n"
                       "cmp x10, x12\n"
                       "b.lo 4b\n"
                       "dsb ish\n"
                       "isb\n"
                       "ldr x0, [x9, #0]\n"
                       "ldr x1, [x9, #16]\n"
                       "ldr x2, [x9, #24]\n"
                       "mov x8, #226\n"
                       "svc #0\n"
                       "mov x0, #1\n"
                       "ret\n"
                       "2:\n"
                       "mov x0, #0\n"
                       "ret\n"
                       ".global __bolt_hugify_remap_stub_end\n"
                       "__bolt_hugify_remap_stub_end:\n");
#else
  __exit(1);
#endif
}

static const char *skipToken(const char *Buf, const char *End) {
  while (Buf < End && *Buf != ' ')
    ++Buf;
  while (Buf < End && *Buf == ' ')
    ++Buf;
  return Buf;
}

static bool parseMapsLine(const char *Line, const char *LineEnd,
                          MapsEntry &Entry) {
  uint64_t Start;
  const char *Buf = parseHex(Line, LineEnd, Start);
  if (!Buf || Buf >= LineEnd || *Buf != '-')
    return false;
  ++Buf;
  Buf = parseHex(Buf, LineEnd, Entry.End);
  if (!Buf || Buf >= LineEnd || *Buf != ' ')
    return false;

  Entry.Start = Start;
  if (Entry.Start >= Entry.End)
    return false;

  Buf = skipToken(Buf, LineEnd);
  if (LineEnd - Buf < 4)
    return false;
  Entry.Perms[0] = Buf[0];
  Entry.Perms[1] = Buf[1];
  Entry.Perms[2] = Buf[2];
  Entry.Perms[3] = Buf[3];

  // Skip permissions and offset.
  Buf = skipToken(Buf, LineEnd);
  Buf = skipToken(Buf, LineEnd);

  // Skip device.
  Buf = skipToken(Buf, LineEnd);

  uint64_t Inode;
  Buf = parseDecimal(Buf, LineEnd, Inode);
  if (!Buf)
    return false;
  Entry.HasFile = Inode != 0;
  return true;
}

static bool isExecutable(const MapsEntry &Entry) {
  return Entry.Perms[2] == 'x';
}

static bool isWritable(const MapsEntry &Entry) { return Entry.Perms[1] == 'w'; }

static bool isReadable(const MapsEntry &Entry) { return Entry.Perms[0] == 'r'; }

static bool isPrivate(const MapsEntry &Entry) { return Entry.Perms[3] == 'p'; }

static bool isExecutableVMA(const MapsEntry &Entry) {
  return isPrivate(Entry) && isReadable(Entry) && isExecutable(Entry);
}

static bool isTextVMA(const MapsEntry &Entry) {
  return Entry.HasFile && isExecutableVMA(Entry);
}

static bool isTextAddress(const char *Buf, const char *End, uint64_t Address) {
  const char *Cur = Buf;
  while (Cur < End) {
    const char *LineEnd = findLineEnd(Cur, End);
    MapsEntry Entry;
    if (parseMapsLine(Cur, LineEnd, Entry) && Entry.Start <= Address &&
        Address < Entry.End)
      return isTextVMA(Entry);
    Cur = skipLine(Cur, End);
  }
  return false;
}

static bool parseKBValue(const char *Line, const char *LineEnd,
                         const char *Prefix, uint64_t &Value) {
  const uint32_t PrefixSize = strLen(Prefix);
  if (LineEnd - Line < PrefixSize || strnCmp(Line, Prefix, PrefixSize) != 0)
    return false;
  const char *Ptr = Line + PrefixSize;
  while (Ptr < LineEnd && *Ptr == ' ')
    ++Ptr;
  return parseDecimal(Ptr, LineEnd, Value) != nullptr;
}

static bool isSyscallError(const void *Address);

static bool collectHugePageStats(const char *MapsBuf, const char *MapsEnd,
                                 HugePageStats &Stats) {
  char *SmapsBuf =
      reinterpret_cast<char *>(__mmap(0, MaxSmapsBytes, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (isSyscallError(SmapsBuf))
    return false;
  int FD = __open("/proc/self/smaps", O_RDONLY, 0);
  if (FD < 0) {
    __munmap(SmapsBuf, MaxSmapsBytes);
    return false;
  }

  size_t Bytes = 0;
  while (Bytes + 1 < MaxSmapsBytes) {
    const int64_t ReadSize = static_cast<int64_t>(
        __read(FD, SmapsBuf + Bytes, MaxSmapsBytes - Bytes - 1));
    if (ReadSize <= 0)
      break;
    Bytes += ReadSize;
  }
  __close(FD);
  if (!Bytes) {
    __munmap(SmapsBuf, MaxSmapsBytes);
    return false;
  }
  SmapsBuf[Bytes] = '\0';

  bool IsText = false;
  uint64_t CurrentAnonKB = 0;
  uint64_t CurrentFileKB = 0;
  auto Flush = [&]() {
    if (!IsText || (!CurrentAnonKB && !CurrentFileKB))
      return;
    Stats.AnonHugeKB += CurrentAnonKB;
    Stats.FilePmdKB += CurrentFileKB;
    ++Stats.VMAs;
  };

  const char *Cur = SmapsBuf;
  const char *End = SmapsBuf + Bytes;
  while (Cur < End) {
    const char *LineEnd = findLineEnd(Cur, End);
    MapsEntry Entry;
    if (parseMapsLine(Cur, LineEnd, Entry)) {
      Flush();
      IsText = isTextAddress(MapsBuf, MapsEnd, Entry.Start);
      CurrentAnonKB = 0;
      CurrentFileKB = 0;
    } else {
      uint64_t Value;
      if (parseKBValue(Cur, LineEnd, "AnonHugePages:", Value))
        CurrentAnonKB = Value;
      else if (parseKBValue(Cur, LineEnd, "FilePmdMapped:", Value))
        CurrentFileKB = Value;
    }
    Cur = skipLine(Cur, End);
  }
  Flush();

  __munmap(SmapsBuf, MaxSmapsBytes);
  return true;
}

static bool isSyscallError(const void *Address) {
  return reinterpret_cast<uint64_t>(Address) >= static_cast<uint64_t>(-4095LL);
}

static int getProtection(const MapsEntry &Entry) {
  int Prot = PROT_NONE;
  if (isReadable(Entry))
    Prot |= PROT_READ;
  if (isWritable(Entry))
    Prot |= PROT_WRITE;
  if (isExecutable(Entry))
    Prot |= PROT_EXEC;
  return Prot;
}

static bool adviseRange(uint8_t *From, uint8_t *To) {
  if (From >= To)
    return true;
  return __madvise(From, To - From, MADV_HUGEPAGE) != -1;
}

static void syncInstructionCache(uint8_t *From, uint8_t *To) {
#if defined(__aarch64__) || defined(__arm64__)
  uint64_t CacheType;
  __asm__ __volatile__("mrs %0, ctr_el0" : "=r"(CacheType));

  const uint64_t DCacheLineSize = 4 << ((CacheType >> 16) & 15);
  const uint64_t ICacheLineSize = 4 << (CacheType & 15);
  uint64_t Address = reinterpret_cast<uint64_t>(From) & ~(DCacheLineSize - 1);
  const uint64_t End = reinterpret_cast<uint64_t>(To);

  for (; Address < End; Address += DCacheLineSize)
    __asm__ __volatile__("dc cvau, %0" : : "r"(Address) : "memory");
  __asm__ __volatile__("dsb ish" : : : "memory");

  Address = reinterpret_cast<uint64_t>(From) & ~(ICacheLineSize - 1);
  for (; Address < End; Address += ICacheLineSize)
    __asm__ __volatile__("ic ivau, %0" : : "r"(Address) : "memory");
  __asm__ __volatile__("dsb ish\nisb" : : : "memory");
#else
  (void)From;
  (void)To;
#endif
}

using RemapStubTy = bool (*)(RemapArgs *);

static bool remapAnonymousRange(uint8_t *From, uint8_t *To, int FinalProt,
                                RemapStubTy RemapStub) {
  bool RemappedAny = false;
  while (From < To) {
    const uint64_t NextBoundary =
        alignTo(reinterpret_cast<uint64_t>(From) + 1, HugePageBytes);
    uint8_t *ChunkEnd = reinterpret_cast<uint8_t *>(NextBoundary);
    if (ChunkEnd > To)
      ChunkEnd = To;
    const size_t Size = ChunkEnd - From;

    uint8_t *Copy = reinterpret_cast<uint8_t *>(__mmap(
        0, Size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (isSyscallError(Copy))
      return RemappedAny;

    memcpy(Copy, From, Size);
    RemapArgs Args{From, Copy, Size, static_cast<uint64_t>(FinalProt)};
    if (!RemapStub(&Args)) {
      __munmap(Copy, Size);
      return RemappedAny;
    }

    __munmap(Copy, Size);
    RemappedAny = true;
    From = ChunkEnd;
  }
  return RemappedAny;
}

static bool processVMA(const MapsEntry &Entry, uint64_t RuntimeAddress,
                       RemapStubTy RemapStub, DeferredRemap &Deferred) {
  uint8_t *From = reinterpret_cast<uint8_t *>(Entry.Start);
  uint8_t *To = reinterpret_cast<uint8_t *>(Entry.End);
  const int FinalProt = getProtection(Entry);
  if (Entry.Start <= RuntimeAddress && RuntimeAddress < Entry.End) {
    const uint64_t RuntimeChunkStart = RuntimeAddress & ~(HugePageBytes - 1);
    const uint64_t RuntimeChunkEnd = RuntimeChunkStart + HugePageBytes;
    uint8_t *DeferredFrom = reinterpret_cast<uint8_t *>(
        RuntimeChunkStart > Entry.Start ? RuntimeChunkStart : Entry.Start);
    uint8_t *DeferredTo = reinterpret_cast<uint8_t *>(
        RuntimeChunkEnd < Entry.End ? RuntimeChunkEnd : Entry.End);

    // Anonymous THP does not depend on the original ELF file offset being
    // huge-page aligned or on CONFIG_READ_ONLY_THP_FOR_FS.
    bool RemappedAny = false;
    RemappedAny |=
        remapAnonymousRange(From, DeferredFrom, FinalProt, RemapStub);
    RemappedAny |= remapAnonymousRange(DeferredTo, To, FinalProt, RemapStub);
    Deferred = {DeferredFrom, DeferredTo, FinalProt};
    return RemappedAny;
  }

  return remapAnonymousRange(From, To, FinalProt, RemapStub);
}

static bool processTextVMAs(const char *Buf, const char *End,
                            uint64_t RuntimeAddress, RemapStubTy RemapStub,
                            DeferredRemap &Deferred) {
  bool RemappedAny = false;

  const char *Cur = Buf;
  while (Cur < End) {
    const char *LineEnd = findLineEnd(Cur, End);
    MapsEntry Entry;
    if (parseMapsLine(Cur, LineEnd, Entry)) {
      if (isTextVMA(Entry))
        RemappedAny |= processVMA(Entry, RuntimeAddress, RemapStub, Deferred);
    }
    Cur = skipLine(Cur, End);
  }

  return RemappedAny;
}

static bool processAllTextVMAs() {
  // To safely remap the executable text segment (which may include the BOLT
  // runtime code itself) without crashing, we must execute the actual
  // mmap/madvise syscalls from a safe, isolated memory region. We allocate an
  // anonymous page and copy the pure assembly remap stub into it. This prevents
  // the code from unmapping the very instructions it is executing.
  const size_t StubSize =
      __bolt_hugify_remap_stub_end - __bolt_hugify_remap_stub_start;
  const size_t StubMappingSize = alignTo(StubSize, 4096);
  uint8_t *StubMapping = reinterpret_cast<uint8_t *>(
      __mmap(0, StubMappingSize, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (isSyscallError(StubMapping))
    return false;
  memcpy(StubMapping, __bolt_hugify_remap_stub_start, StubSize);
  syncInstructionCache(StubMapping, StubMapping + StubSize);
  if (__mprotect(StubMapping, StubMappingSize, PROT_READ | PROT_EXEC) < 0) {
    __munmap(StubMapping, StubMappingSize);
    return false;
  }
  RemapStubTy RemapStub = reinterpret_cast<RemapStubTy>(StubMapping);

  char *Buf =
      reinterpret_cast<char *>(__mmap(0, MaxMapsBytes, PROT_READ | PROT_WRITE,
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (isSyscallError(Buf)) {
    __munmap(StubMapping, StubMappingSize);
    return false;
  }

  int FD = __open("/proc/self/maps", O_RDONLY, 0);
  if (FD < 0) {
    __munmap(Buf, MaxMapsBytes);
    __munmap(StubMapping, StubMappingSize);
    return false;
  }

  size_t Bytes = 0;
  while (Bytes < MaxMapsBytes) {
    const int64_t ReadSize =
        static_cast<int64_t>(__read(FD, Buf + Bytes, MaxMapsBytes - Bytes));
    if (ReadSize <= 0)
      break;
    Bytes += ReadSize;
  }
  __close(FD);
  if (!Bytes) {
    __munmap(Buf, MaxMapsBytes);
    __munmap(StubMapping, StubMappingSize);
    return false;
  }

  const char *End = Buf + Bytes;

  // Identify the BOLT runtime address, carve out a 2MB-aligned chunk
  // surrounding it, remap the VMA regions before and after it, and store
  // this skipped chunk's boundaries into `Deferred`.
  DeferredRemap Deferred;

  const uint64_t RuntimeAddress =
      reinterpret_cast<uint64_t>(&__bolt_hugify_self_impl);
  bool ProcessedAny =
      processTextVMAs(Buf, End, RuntimeAddress, RemapStub, Deferred);

  if (__bolt_hugify_all_text_stats) {
    HugePageStats Stats;
    const uint64_t DeferredKB =
        Deferred.From ? (Deferred.To - Deferred.From) / 1024 : 0;
    if (collectHugePageStats(Buf, End, Stats))
      printHugePageStats(Stats, DeferredKB);
    else
      __write(1, "[hugify] smaps_stats=unavailable\n",
              sizeof("[hugify] smaps_stats=unavailable\n") - 1);
  }

  __munmap(Buf, MaxMapsBytes);
  // Remapped the deferred chunk at the final step.
  if (Deferred.From)
    ProcessedAny |= remapAnonymousRange(Deferred.From, Deferred.To,
                                        Deferred.FinalProt, RemapStub);
  __munmap(StubMapping, StubMappingSize);
  return ProcessedAny;
}

static void hugifyForOldKernel(uint8_t *From, uint8_t *To) {
  __prctl(41 /* PR_SET_THP_DISABLE */, 0, 0, 0, 0);
  if (!remapAnonymousRange(From, To, PROT_READ | PROT_EXEC,
                           __bolt_hugify_remap_stub)) {
    char Msg[] = "[hugify] failed to remap hot text anonymously\n";
    reportError(Msg, sizeof(Msg));
  }
}

extern "C" void __bolt_hugify_self_impl() {
  uint8_t *HotStart = (uint8_t *)&__hot_start;
  uint8_t *HotEnd = (uint8_t *)&__hot_end;
  // Make sure the start and end are aligned with huge page address
  uint8_t *From = HotStart - ((intptr_t)HotStart & (HugePageBytes - 1));
  uint8_t *To = HotEnd + (HugePageBytes - 1);
  To -= (intptr_t)To & (HugePageBytes - 1);

  DEBUG(reportNumber("[hugify] hot start: ", (uint64_t)HotStart, 16);)
  DEBUG(reportNumber("[hugify] hot end: ", (uint64_t)HotEnd, 16);)
  DEBUG(reportNumber("[hugify] aligned huge page from: ", (uint64_t)From, 16);)
  DEBUG(reportNumber("[hugify] aligned huge page to: ", (uint64_t)To, 16);)

  const bool THPEnabled = isTHPEnabled();
  const bool HasPagecacheTHP = THPEnabled && hasPagecacheTHPSupport();
  if (__bolt_hugify_all_text &&
      !hasEnvironmentVariable("DISABLE_BOLT_HUGIFY_ALL_TEXT")) {
    if (!THPEnabled)
      return;
    __prctl(41 /* PR_SET_THP_DISABLE */, 0, 0, 0, 0);
    processAllTextVMAs();
    return;
  }

  if (!HasPagecacheTHP) {
    DEBUG(report(
              "[hugify] workaround with memory alignment for kernel < 5.10\n");)
    hugifyForOldKernel(From, To);
    return;
  }

  if (!adviseRange(From, To)) {
    char Msg[] = "[hugify] failed to allocate large page\n";
    // TODO: allow user to control the failure behavior.
    reportError(Msg, sizeof(Msg));
  }
}

/// This is hooking ELF's entry, it needs to save all machine state.
extern "C" __attribute((naked)) void __bolt_hugify_self() {
  // clang-format off
#if defined(__x86_64__)
  __asm__ __volatile__(SAVE_ALL "call __bolt_hugify_self_impl\n" RESTORE_ALL
                                "jmp __bolt_hugify_start_program\n"
                                :::);
#elif defined(__aarch64__) || defined(__arm64__)
  __asm__ __volatile__(SAVE_ALL "bl __bolt_hugify_self_impl\n" RESTORE_ALL
                                "adrp x16, __bolt_hugify_start_program\n"
                                "add x16, x16, #:lo12:__bolt_hugify_start_program\n"
                                "br x16\n"
                                :::);
#else
  __exit(1);
#endif
  // clang-format on
}
#endif
