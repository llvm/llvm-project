//=-- lsan_common.h -------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Private LSan header.
//
//===----------------------------------------------------------------------===//

#ifndef LSAN_COMMON_H
#define LSAN_COMMON_H

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_stoptheworld.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

// LeakSanitizer relies on some Glibc's internals (e.g. TLS machinery) thus
// supported for Linux only. Also, LSan doesn't like 32 bit architectures
// because of "small" (4 bytes) pointer size that leads to high false negative
// ratio on large leaks. But we still want to have it for some 32 bit arches
// (e.g. x86), see https://github.com/google/sanitizers/issues/403.
// To enable LeakSanitizer on new architecture, one need to implement
// internal_clone function as well as (probably) adjust TLS machinery for
// new architecture inside sanitizer library.
#if (SANITIZER_LINUX && !SANITIZER_ANDROID || CAN_SANITIZE_LEAKS_MAC) \
     && (SANITIZER_WORDSIZE == 64) && (defined(__x86_64__) \
     ||  defined(__mips64) ||  defined(__aarch64__))
#define CAN_SANITIZE_LEAKS 1
#elif SANITIZER_LINUX && !SANITIZER_ANDROID && defined(__i386__)
#define CAN_SANITIZE_LEAKS 1
#else
#define CAN_SANITIZE_LEAKS 0
#endif

namespace __sanitizer {
class FlagParser;
struct DTLS;
}

namespace __lsan {

// Chunk tags.
enum ChunkTag {
  kDirectlyLeaked = 0,  // default
  kIndirectlyLeaked = 1,
  kReachable = 2,
  kIgnored = 3
};

struct Flags {
#define LSAN_FLAG(Type, Name, DefaultValue, Description) Type Name;
#include "lsan_flags.inc"
#undef LSAN_FLAG

  void SetDefaults();
  uptr pointer_alignment() const {
    return use_unaligned ? 1 : sizeof(uptr);
  }
};

extern Flags lsan_flags;
inline Flags *flags() { return &lsan_flags; }
void RegisterLsanFlags(FlagParser *parser, Flags *f);

struct Leak {
  u32 id;
  uptr hit_count;
  uptr total_size;
  u32 stack_trace_id;
  bool is_directly_leaked;
  bool is_suppressed;
};

struct LeakedObject {
  u32 leak_id;
  uptr addr;
  uptr size;
};

// Aggregates leaks by stack trace prefix.
class LeakReport {
 public:
  LeakReport() : next_id_(0), leaks_(1), leaked_objects_(1) {}
  void AddLeakedChunk(uptr chunk, u32 stack_trace_id, uptr leaked_size,
                      ChunkTag tag);
  void ReportTopLeaks(uptr max_leaks);
  void PrintSummary();
  void ApplySuppressions();
  uptr UnsuppressedLeakCount();


 private:
  void PrintReportForLeak(uptr index);
  void PrintLeakedObjectsForLeak(uptr index);

  u32 next_id_;
  InternalMmapVector<Leak> leaks_;
  InternalMmapVector<LeakedObject> leaked_objects_;
};

typedef InternalMmapVector<uptr> Frontier;

// Platform-specific functions.
void InitializePlatformSpecificModules();
void ProcessGlobalRegions(Frontier *frontier);
void ProcessPlatformSpecificAllocations(Frontier *frontier);
// Run stoptheworld while holding any platform-specific locks.
void DoStopTheWorld(StopTheWorldCallback callback, void* argument);

void ScanRangeForPointers(uptr begin, uptr end,
                          Frontier *frontier,
                          const char *region_type, ChunkTag tag);

enum IgnoreObjectResult {
  kIgnoreObjectSuccess,
  kIgnoreObjectAlreadyIgnored,
  kIgnoreObjectInvalid
};

// Functions called from the parent tool.
void InitCommonLsan();
void DoLeakCheck();
void DisableCounterUnderflow();
bool DisabledInThisThread();

// Used to implement __lsan::ScopedDisabler.
void DisableInThisThread();
void EnableInThisThread();
// Can be used to ignore memory allocated by an intercepted
// function.
struct ScopedInterceptorDisabler {
  ScopedInterceptorDisabler() { DisableInThisThread(); }
  ~ScopedInterceptorDisabler() { EnableInThisThread(); }
};

// Special case for "new T[0]" where T is a type with DTOR.
// new T[0] will allocate one word for the array size (0) and store a pointer
// to the end of allocated chunk.
inline bool IsSpecialCaseOfOperatorNew0(uptr chunk_beg, uptr chunk_size,
                                        uptr addr) {
  return chunk_size == sizeof(uptr) && chunk_beg + chunk_size == addr &&
         *reinterpret_cast<uptr *>(chunk_beg) == 0;
}

// The following must be implemented in the parent tool.

void ForEachChunk(ForEachChunkCallback callback, void *arg);
// Returns the address range occupied by the global allocator object.
void GetAllocatorGlobalRange(uptr *begin, uptr *end);
// Wrappers for allocator's ForceLock()/ForceUnlock().
void LockAllocator();
void UnlockAllocator();
// Returns true if [addr, addr + sizeof(void *)) is poisoned.
bool WordIsPoisoned(uptr addr);
// Wrappers for ThreadRegistry access.
void LockThreadRegistry();
void UnlockThreadRegistry();
bool GetThreadRangesLocked(uptr os_id, uptr *stack_begin, uptr *stack_end,
                           uptr *tls_begin, uptr *tls_end, uptr *cache_begin,
                           uptr *cache_end, DTLS **dtls);
void ForEachExtraStackRange(uptr os_id, RangeIteratorCallback callback,
                            void *arg);
// If called from the main thread, updates the main thread's TID in the thread
// registry. We need this to handle processes that fork() without a subsequent
// exec(), which invalidates the recorded TID. To update it, we must call
// gettid() from the main thread. Our solution is to call this function before
// leak checking and also before every call to pthread_create() (to handle cases
// where leak checking is initiated from a non-main thread).
void EnsureMainThreadIDIsCorrect();
// If p points into a chunk that has been allocated to the user, returns its
// user-visible address. Otherwise, returns 0.
uptr PointsIntoChunk(void *p);
// Returns address of user-visible chunk contained in this allocator chunk.
uptr GetUserBegin(uptr chunk);
// Helper for __lsan_ignore_object().
IgnoreObjectResult IgnoreObjectLocked(const void *p);
// Wrapper for chunk metadata operations.
class LsanMetadata {
 public:
  // Constructor accepts address of user-visible chunk.
  explicit LsanMetadata(uptr chunk);
  bool allocated() const;
  ChunkTag tag() const;
  void set_tag(ChunkTag value);
  uptr requested_size() const;
  u32 stack_trace_id() const;
 private:
  void *metadata_;
};

}  // namespace __lsan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
int __lsan_is_turned_off();

SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char *__lsan_default_suppressions();
}  // extern "C"

#endif  // LSAN_COMMON_H
