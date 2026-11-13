//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Host ConcurrencySanitizer report generation.
///
//===----------------------------------------------------------------------===//

#include "csan.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

namespace __sanitizer {
void BufferedStackTrace::UnwindImpl(uptr pc, uptr bp, void *context,
                                    bool request_fast, u32 max_depth) {
  uptr top = 0;
  uptr bottom = 0;
  GetThreadStackTopAndBottom(false, &top, &bottom);
  bool fast = StackTrace::WillUseFastUnwind(request_fast);
  Unwind(max_depth, pc, bp, context, top, bottom, fast);
}
} // namespace __sanitizer

namespace __csan {
namespace {

class Decorator : public SanitizerCommonDecorator {
public:
  const char *Access() { return Blue(); }
  const char *Location() { return Green(); }
};

// A pair of racing program counter values to deduplicate and check.
struct RacyPcs {
  uptr pc[2];

  bool operator==(const RacyPcs &Other) const {
    if (pc[0] == Other.pc[0] && pc[1] == Other.pc[1])
      return true;
    return pc[0] == Other.pc[1] && pc[1] == Other.pc[0];
  }
};
InternalMmapVectorNoCtor<RacyPcs> RacyPcsSeen;

struct PeerInfo {
  uptr pc;
  uptr size;
  int access_type;
};

bool HandleRacyPcs(uptr PC, uptr PeerPC) {
  const RacyPcs Racy = {{PC, PeerPC}};
  for (uptr I = 0; I < RacyPcsSeen.size(); ++I) {
    if (Racy == RacyPcsSeen[I]) {
      VReport(2, "%s: suppressing report as doubled\n", SanitizerToolName);
      return true;
    }
  }
  RacyPcsSeen.push_back(Racy);
  return false;
}

const char *AccessKind(int Type) {
  const bool Write = Type & CSAN_ACCESS_WRITE;
  const bool Atomic = Type & CSAN_ACCESS_ATOMIC;
  const bool Compound = Type & CSAN_ACCESS_COMPOUND;
  if (Compound && Write)
    return Atomic ? "read-write (atomic)" : "read-write";
  if (Write)
    return Atomic ? "write (atomic)" : "write";
  return Atomic ? "read (atomic)" : "read";
}

const char *MemOpDesc(bool First, int Type) {
  const bool Write = Type & CSAN_ACCESS_WRITE;
  const bool Atomic = Type & CSAN_ACCESS_ATOMIC;
  const bool Compound = Type & CSAN_ACCESS_COMPOUND;
  if (Compound && Write)
    return Atomic ? (First ? "Read-write (atomic)"
                           : "Previous read-write (atomic)")
                  : (First ? "Read-write" : "Previous read-write");
  if (Write)
    return Atomic ? (First ? "Write (atomic)" : "Previous write (atomic)")
                  : (First ? "Write" : "Previous write");
  return Atomic ? (First ? "Read (atomic)" : "Previous read (atomic)")
                : (First ? "Read" : "Previous read");
}

void CaptureStack(uptr PC, uptr BP, uptr *Out, u32 *N) {
  UNINITIALIZED BufferedStackTrace Stack;
  Stack.Unwind(PC, BP, nullptr, common_flags()->fast_unwind_on_fatal,
               kMaxStackFrames);
  *N = Stack.size > kMaxStackFrames ? kMaxStackFrames : Stack.size;
  if (*N)
    internal_memcpy(Out, Stack.trace, *N * sizeof(uptr));
}

SymbolizedStack *SymbolizeFrame(uptr PC) {
  return Symbolizer::GetOrInit()->SymbolizePC(
      StackTrace::GetPreviousInstructionPc(PC));
}

u32 SkipRuntimeFrames(const uptr *PCs, u32 N) {
  for (u32 I = 0; I < N; ++I) {
    SymbolizedStack *Frames = SymbolizeFrame(PCs[I]);
    const bool User = Frames && SkipInternalFrames(Frames);
    if (Frames)
      Frames->ClearAll();
    if (User)
      return I;
  }
  return 0;
}

void CopyFuncName(uptr PC, InternalScopedString *Out) {
  SymbolizedStack *Frames = SymbolizeFrame(PC);
  const SymbolizedStack *User = Frames ? SkipInternalFrames(Frames) : nullptr;
  if (!User)
    User = Frames;
  if (User && User->info.function && User->info.function[0])
    Out->Append(User->info.function);
  else
    Out->AppendF("%p", (void *)PC);
  if (Frames)
    Frames->ClearAll();
}

void PrintFrames(const uptr *PCs, u32 N) {
  if (!N) {
    Printf("    [failed to restore the stack]\n\n");
    return;
  }
  const u32 Skip = SkipRuntimeFrames(PCs, N);
  StackTrace Trace(PCs + Skip, N - Skip);
  Trace.Print();
}

void PrintLocation(uptr Addr) {
  DataInfo Loc;
  if (!Symbolizer::GetOrInit()->SymbolizeData(Addr, &Loc) || !Loc.name ||
      !Loc.name[0] || Loc.name[0] == '?') {
    Loc.Clear();
    return;
  }
  Decorator D;
  Printf("%s", D.Location());
  if (Loc.size)
    Printf("  Location is global '%s' of size %zu at %p\n", Loc.name, Loc.size,
           (void *)Addr);
  else
    Printf("  Location is global '%s' at %p\n", Loc.name, (void *)Addr);
  Printf("%s", D.Default());
  Loc.Clear();
}

void PrintHexValue(u64 V, uptr Size) {
  switch (Size) {
  case 1:
    Printf("0x%02llx", (unsigned long long)V);
    break;
  case 2:
    Printf("0x%04llx", (unsigned long long)V);
    break;
  case 4:
    Printf("0x%08llx", (unsigned long long)V);
    break;
  default:
    Printf("0x%016llx", (unsigned long long)V);
    break;
  }
}

void PrintValueChange(uptr Size, u64 Old, u64 New) {
  if (Size == 0 || Size > 8 || Old == New)
    return;
  Printf("  value changed: ");
  PrintHexValue(Old, Size);
  Printf(" -> ");
  PrintHexValue(New, Size);
  Printf("\n");
}

void PrintReport(const AccessInfo &AI, const PeerInfo *Other, u64 Old,
                 u64 New) {
  if (HandleRacyPcs(AI.pc, Other ? Other->pc : 0))
    return;

  UNINITIALIZED uptr ThisStack[kMaxStackFrames];
  u32 ThisN = 0;
  CaptureStack(AI.pc, AI.bp, ThisStack, &ThisN);

  const u32 ThisSkip = SkipRuntimeFrames(ThisStack, ThisN);
  const uptr ThisFrame = ThisN ? ThisStack[ThisSkip] : AI.pc;
  const uptr OtherFrame = Other ? Other->pc : 0;

  RecordDataRace();

  Decorator D;
  InternalScopedString ThisFn;
  CopyFuncName(ThisFrame, &ThisFn);

  Printf("==================\n");
  Printf("%s", D.Warning());
  if (Other) {
    InternalScopedString OtherFn;
    CopyFuncName(OtherFrame, &OtherFn);
    const int Cmp = internal_strcmp(OtherFn.data(), ThisFn.data());
    Printf("WARNING: ConcurrencySanitizer: data race in %s / %s\n",
           Cmp < 0 ? OtherFn.data() : ThisFn.data(),
           Cmp < 0 ? ThisFn.data() : OtherFn.data());
  } else {
    Printf("WARNING: ConcurrencySanitizer: data race of unknown origin in %s\n",
           ThisFn.data());
  }
  Printf("%s", D.Default());

  Printf("%s", D.Access());
  if (Other) {
    Printf("  %s of size %zu at %p by thread %u:\n",
           MemOpDesc(true, AI.access_type), AI.size, AI.ptr, AI.tid);
    Printf("%s", D.Default());
    PrintFrames(ThisStack, ThisN);

    Printf("%s", D.Access());
    Printf("  %s of size %zu at %p:\n", MemOpDesc(false, Other->access_type),
           Other->size, AI.ptr);
    Printf("%s", D.Default());
    PrintFrames(&Other->pc, 1);
  } else {
    Printf(
        "  race at unknown origin, with %s of size %zu at %p by thread %u:\n",
        AccessKind(AI.access_type), AI.size, AI.ptr, AI.tid);
    Printf("%s", D.Default());
    PrintFrames(ThisStack, ThisN);
  }

  PrintValueChange(AI.size, Old, New);
  PrintLocation((uptr)AI.ptr);

  if (ThisN) {
    StackTrace Summary(ThisStack + ThisSkip, ThisN - ThisSkip);
    ReportErrorSummary(Other ? "data race" : "data race of unknown origin",
                       &Summary);
  }
  Printf("==================\n");
  if (flags()->halt_on_error)
    Die();
}

} // namespace

void ReportKnownOrigin(const AccessInfo &AI, ValueChange VC, uptr PeerPC,
                       int PeerAccess, uptr PeerSize, u64 Old, u64 New) {
  ScopedErrorReportLock L;
  if (VC == kValueChangeFalse)
    return;
  const PeerInfo Peer = {PeerPC, PeerSize, PeerAccess};
  PrintReport(AI, &Peer, Old, New);
}

void ReportUnknownOrigin(const AccessInfo &AI, u64 Old, u64 New) {
  ScopedErrorReportLock L;
  PrintReport(AI, nullptr, Old, New);
}

} // namespace __csan
