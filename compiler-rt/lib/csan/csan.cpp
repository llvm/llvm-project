//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Watchpoint-based host data race detector inspired by KCSAN. The host and GPU
/// runtimes share a configurable packed-watchpoint implementation.
///
//===----------------------------------------------------------------------===//

#include "csan.h"
#include "csan_watch.h"

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

#define INTERFACE extern "C" SANITIZER_INTERFACE_ATTRIBUTE

static constexpr u32 kHostWatchpointEntries = 64;
static constexpr u32 kHostCheckAdjacent = 1;
static constexpr uptr kHostSlotRange = 4096;
static constexpr uptr kHostMaxAccessSize =
    kHostSlotRange * (1 + kHostCheckAdjacent);
static_assert(__atomic_always_lock_free(sizeof(u64), nullptr),
              "host watchpoints must be lock-free");

using HostWatchpointTable =
    __csan::WatchpointTable<kHostMaxAccessSize, kHostCheckAdjacent>;
static u64 HostWatchpoints[kHostWatchpointEntries +
                           HostWatchpointTable::OverflowEntries];
static_assert(HostWatchpointTable::AddressBits == 48,
              "host watchpoints require 48-bit pointers");

static HostWatchpointTable GetHostWatchpoints() {
  return HostWatchpointTable(HostWatchpoints);
}

static u32 HostWatchpointSlot(uptr Address) {
  return Address / kHostSlotRange % kHostWatchpointEntries;
}

namespace __csan {

Flags flags_data;

static void Initialize();

static atomic_uint64_t NumDataRaces;
static THREADLOCAL u32 DisableCount;
static THREADLOCAL s32 Skip;
static THREADLOCAL u32 RandState;

namespace {
struct ScopedDisable {
  ScopedDisable() { ++DisableCount; }
  ~ScopedDisable() { --DisableCount; }
};
} // namespace

void RecordDataRace() {
  atomic_fetch_add(&NumDataRaces, 1, memory_order_relaxed);
}

void Flags::SetDefaults() {
#define CSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "csan_flags.inc"
#undef CSAN_FLAG
}

static void RegisterCsanFlags(FlagParser *Parser, Flags *F) {
#define CSAN_FLAG(Type, Name, DefaultValue, Description)                       \
  RegisterFlag(Parser, #Name, Description, &F->Name);
#include "csan_flags.inc"
#undef CSAN_FLAG
}

void InitializeFlags() {
  SetCommonFlagsDefaults();
  {
    CommonFlags CF;
    CF.CopyFrom(*common_flags());
    CF.external_symbolizer_path = GetEnv("CSAN_SYMBOLIZER_PATH");
    OverrideCommonFlags(CF);
  }

  flags()->SetDefaults();

  FlagParser Parser;
  RegisterCommonFlags(&Parser);
  RegisterCsanFlags(&Parser, flags());
  Parser.ParseString(__csan_default_options());
  Parser.ParseStringFromEnv("CSAN_OPTIONS");
  InitializeCommonFlags();
  if (Verbosity())
    ReportUnrecognizedFlags();
  if (common_flags()->help)
    Parser.PrintFlagDescriptions();
}

static u32 Random(u32 EpRo) {
  if (EpRo <= 1)
    return 0;
  u32 State = RandState;
  if (!State)
    State = (u32)__builtin_readcyclecounter() | 1u;
  State = 1664525u * State + 1013904223u;
  RandState = State;
  return State % EpRo;
}

static void ResetSkip() {
  s32 Count = flags()->skip_watch;
  if (Count < 0)
    Count = 0;
  if (Count)
    Count -= (s32)Random((u32)Count);
  Skip = Count;
}

static bool ShouldWatch(int Type) {
  if (Type & CSAN_ACCESS_ATOMIC)
    return false;
  if (--Skip >= 0)
    return false;
  return true;
}

static void DelayAccess(int Type) {
  s32 Delay = flags()->udelay;
  if (Delay < 0)
    Delay = 0;
  if (Delay) {
    u32 Skew = (Type & CSAN_ACCESS_COMPOUND) ? 1u : 0u;
    u32 Span = (u32)Delay >> Skew;
    if (!Span)
      Span = (u32)Delay;
    Delay -= (s32)Random(Span);
  }
  if (Delay)
    internal_usleep((u64)Delay);
}

static u64 ReadRange(const volatile u8 *Bytes, uptr Size) {
  u64 Sum = 0xcbf29ce484222325ull;
  uptr I = 0;
  for (; I < Size && ((uptr)(Bytes + I) & 7u); ++I)
    Sum = (Sum ^ Bytes[I]) * 0x100000001b3ull;
  for (; I + 8 <= Size; I += 8)
    Sum = (Sum ^ *(const volatile u64 *)(Bytes + I)) * 0x100000001b3ull;
  for (; I < Size; ++I)
    Sum = (Sum ^ Bytes[I]) * 0x100000001b3ull;
  return Sum;
}

static u64 ReadInstrumented(const volatile void *Ptr, uptr Size) {
  switch (Size) {
  case 1:
    return *(const volatile u8 *)Ptr;
  case 2:
    return *(const volatile u16 *)Ptr;
  case 4:
    return *(const volatile u32 *)Ptr;
  case 8:
    return *(const volatile u64 *)Ptr;
  default:
    return ReadRange((const volatile u8 *)Ptr, Size);
  }
}

static AccessInfo MakeAccessInfo(const volatile void *Ptr, uptr Size, int Type,
                                 uptr PC, uptr BP) {
  AccessInfo AI;
  AI.ptr = Ptr;
  AI.size = Size;
  AI.access_type = Type;
  AI.tid = (u32)GetTid();
  AI.pc = PC;
  AI.bp = BP;
  return AI;
}

NOINLINE static void FoundWatchpoint(const volatile void *Ptr, uptr Size,
                                     int Type, uptr PC, uptr, u64 *WP,
                                     u64 Encoded) {
  ScopedDisable Disable;
  GetHostWatchpoints().TryConsume(WP, Encoded, PC,
                                  (Type & CSAN_ACCESS_WRITE) != 0, Size);
}

NOINLINE static void SetupWatchpoint(const volatile void *Ptr, uptr Size,
                                     int Type, uptr PC, uptr BP) {
  ScopedDisable Disable;
  ResetSkip();

  if ((uptr)Ptr < GetPageSizeCached())
    return;

  u64 *WP = GetHostWatchpoints().Insert((uptr)Ptr, Size,
                                        (Type & CSAN_ACCESS_WRITE) != 0,
                                        HostWatchpointSlot((uptr)Ptr));
  if (!WP)
    return;

  const u64 Old = ReadInstrumented(Ptr, Size);
  DelayAccess(Type);
  const u64 New = ReadInstrumented(Ptr, Size);

  ValueChange VC = Old != New ? kValueChangeTrue : kValueChangeMaybe;

  const AccessInfo AI = MakeAccessInfo(Ptr, Size, Type, PC, BP);
  void *Peer;
  int PeerAccess;
  u32 PeerSize;
  if (!GetHostWatchpoints().Consume(WP, Peer, PeerAccess, PeerSize)) {
    ReportKnownOrigin(AI, VC, (uptr)Peer, PeerAccess, PeerSize, Old, New);
  } else if (VC == kValueChangeTrue) {
    ReportUnknownOrigin(AI, Old, New);
  }

  GetHostWatchpoints().Remove(WP);
}

ALWAYS_INLINE static void CheckAccess(const volatile void *Ptr, uptr Size,
                                      int Type, uptr PC, uptr BP) {
  if (UNLIKELY(!Ptr || !Size))
    return;
  if (Size > kHostMaxAccessSize)
    Size = kHostMaxAccessSize;
  if (UNLIKELY(uptr(Ptr) & ~HostWatchpointTable::AddressMask))
    return;
  Initialize();
  if (UNLIKELY(DisableCount))
    return;

  u64 Encoded;
  u64 *WP =
      GetHostWatchpoints().Find((uptr)Ptr, Size, !(Type & CSAN_ACCESS_WRITE),
                                HostWatchpointSlot((uptr)Ptr), Encoded);
  if (UNLIKELY(WP != nullptr))
    FoundWatchpoint(Ptr, Size, Type, PC, BP, WP, Encoded);
  else if (UNLIKELY(ShouldWatch(Type)))
    SetupWatchpoint(Ptr, Size, Type, PC, BP);
}

static StaticSpinMutex InitMutex;
static atomic_uint8_t Initialized;

void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;
  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;
  SanitizerToolName = "ConcurrencySanitizer";
  CacheBinaryName();
  InitializeFlags();
  atomic_store(&Initialized, 1, memory_order_release);
  Symbolizer::LateInitialize();
}

} // namespace __csan

SANITIZER_INTERFACE_WEAK_DEF(const char *, __csan_default_options, void) {
  return "";
}

INTERFACE u64 __csan_get_num_data_races() {
  return atomic_load(&__csan::NumDataRaces, memory_order_relaxed);
}

INTERFACE void __csan_init() { __csan::Initialize(); }

INTERFACE void __csan_func_entry(void *) {}
INTERFACE void __csan_func_exit() {}
INTERFACE void __csan_ignore_thread_begin() { ++__csan::DisableCount; }
INTERFACE void __csan_ignore_thread_end() {
  if (__csan::DisableCount)
    --__csan::DisableCount;
}

static int AccessFlags(int Flags, bool IsWrite) {
  return Flags | (IsWrite ? CSAN_ACCESS_WRITE : 0);
}

#define CSAN_PROBE(name, N, IsWrite)                                           \
  INTERFACE void name(void *Addr, int Flags) {                                 \
    GET_CALLER_PC_BP;                                                          \
    __csan::CheckAccess(Addr, N, AccessFlags(Flags, IsWrite), pc, bp);         \
  }

#define CSAN_ACCESS(N)                                                         \
  CSAN_PROBE(__csan_read##N, N, false)                                         \
  CSAN_PROBE(__csan_unaligned_read##N, N, false)                               \
  CSAN_PROBE(__csan_volatile_read##N, N, false)                                \
  CSAN_PROBE(__csan_unaligned_volatile_read##N, N, false)                      \
  CSAN_PROBE(__csan_write##N, N, true)                                         \
  CSAN_PROBE(__csan_unaligned_write##N, N, true)                               \
  CSAN_PROBE(__csan_volatile_write##N, N, true)                                \
  CSAN_PROBE(__csan_unaligned_volatile_write##N, N, true)                      \
  CSAN_PROBE(__csan_read_write##N, N, true)                                    \
  CSAN_PROBE(__csan_unaligned_read_write##N, N, true)

CSAN_ACCESS(1)
CSAN_ACCESS(2)
CSAN_ACCESS(4)
CSAN_ACCESS(8)
CSAN_ACCESS(16)

INTERFACE void __csan_read_range(void *Addr, uptr Size, int Flags) {
  GET_CALLER_PC_BP;
  __csan::CheckAccess(Addr, Size, AccessFlags(Flags, false), pc, bp);
}

INTERFACE void __csan_write_range(void *Addr, uptr Size, int Flags) {
  GET_CALLER_PC_BP;
  __csan::CheckAccess(Addr, Size, AccessFlags(Flags, true), pc, bp);
}

// TODO: Handle thread reordering checks like KCSAN.
INTERFACE void __csan_atomic_thread_fence(int) {}
INTERFACE void __csan_atomic_signal_fence(int) {}
