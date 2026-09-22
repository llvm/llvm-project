//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Host-side device race report generation.
///
//===----------------------------------------------------------------------===//

#include "csan_offload.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "shared/rpc.h"

using namespace __sanitizer;

namespace __csan {
namespace {

class Decorator : public SanitizerCommonDecorator {
public:
  const char *Access() { return Blue(); }
  const char *Location() { return Green(); }
};

const char *KindName(u32 Kind) {
  switch (Kind) {
  case CSAN_RACE_UNKNOWN_ORIGIN:
    return "data race of unknown origin";
  case CSAN_RACE_INTRA_WAVE:
    return "intra-wave data race";
  default:
    return "data race";
  }
}

const char *MopDesc(bool First, u32 Type) {
  const bool Write = Type & CSAN_ACCESS_WRITE;
  return First ? (Write ? "Write" : "Read")
               : (Write ? "Previous write" : "Previous read");
}

// A pair of PCs that have already been reported as racing. The device samples
// the same static access from thousands of threads, so without this every
// launch buries the user in duplicates of one bug.
struct RacyPcs {
  u64 pc[2];

  bool operator==(const RacyPcs &other) const {
    if (pc[0] == other.pc[0] && pc[1] == other.pc[1])
      return true;
    return pc[0] == other.pc[1] && pc[1] == other.pc[0];
  }
};

Mutex RacyMutex;
InternalMmapVectorNoCtor<RacyPcs> RacyPcsSeen;

bool FindRacyPcs(const RacyPcs &Racy) {
  for (uptr I = 0; I < RacyPcsSeen.size(); ++I) {
    if (Racy == RacyPcsSeen[I]) {
      VReport(2, "%s: suppressing report as doubled\n", SanitizerToolName);
      return true;
    }
  }
  return false;
}

bool HandleRacyPcs(const __csan_gpu_race &R) {
  RacyPcs Racy = {{R.pc, R.peer_pc}};
  {
    ReadLock L(&RacyMutex);
    if (FindRacyPcs(Racy))
      return true;
  }
  Lock L(&RacyMutex);
  if (FindRacyPcs(Racy))
    return true;
  RacyPcsSeen.push_back(Racy);
  return false;
}

void PrintFrames(SymbolizedStack *Frames, u64 PC) {
  if (!Frames) {
    Printf("    #0 (%p)\n", (void *)(uptr)PC);
    return;
  }
  const SymbolizedStack *F = SkipInternalFrames(Frames);
  if (!F)
    F = Frames;
  int N = 0;
  for (; F; F = F->next, ++N) {
    InternalScopedString Res;
    StackTracePrinter::GetOrInit()->RenderFrame(
        &Res, common_flags()->stack_trace_format, N, F->info.address, &F->info,
        common_flags()->symbolize_vs_style, common_flags()->strip_path_prefix);
    Printf("%s\n", Res.data());
  }
}

} // namespace

void PrintOffloadReport(const __csan_gpu_race &R) {
  if (HandleRacyPcs(R))
    return;

  Decorator D;
  Printf("==================\n");
  Printf("%s", D.Warning());
  Printf("WARNING: ConcurrencySanitizer: %s\n", KindName(R.kind));
  Printf("%s", D.Default());

  Printf("%s", D.Access());
  Printf("  %s of size %u at 0x%zx in block (%u,%u,%u) thread (%u,%u,%u) "
         "lane %u:\n",
         MopDesc(true, R.access_type), R.size, (uptr)R.addr, R.block[0],
         R.block[1], R.block[2], R.thread[0], R.thread[1], R.thread[2], R.lane);
  Printf("%s", D.Default());
  SymbolizedStack *This = Offload::Get().Symbolize((uptr)R.pc);
  PrintFrames(This, R.pc);

  Printf("%s", D.Access());
  if (R.peer_pc) {
    Printf("  %s of size %u at 0x%zx:\n", MopDesc(false, R.peer_access_type),
           R.peer_size, (uptr)R.addr);
    Printf("%s", D.Default());
    SymbolizedStack *Peer = Offload::Get().Symbolize((uptr)R.peer_pc);
    PrintFrames(Peer, R.peer_pc);
    if (Peer)
      Peer->ClearAll();
  } else if (R.kind == CSAN_RACE_INTRA_WAVE) {
    Printf("  Previous access by lane %u in the same wave\n", R.peer_lane);
    Printf("%s", D.Default());
  } else {
    Printf("  Previous access of unknown origin\n");
    Printf("%s", D.Default());
  }

  DataInfo Loc;
  if (Offload::Get().SymbolizeData((uptr)R.addr, &Loc)) {
    Printf("%s", D.Location());
    if (Loc.size)
      Printf("  Location is global '%s' of size %zu at 0x%zx\n", Loc.name,
             Loc.size, (uptr)R.addr);
    else
      Printf("  Location is global '%s' at 0x%zx\n", Loc.name, (uptr)R.addr);
    Printf("%s", D.Default());
    Loc.Clear();
  }

  if (This) {
    const SymbolizedStack *User = SkipInternalFrames(This);
    ReportErrorSummary(KindName(R.kind), (User ? User : This)->info);
    This->ClearAll();
  }

  Printf("==================\n");
}

u32 HandleOffloadReport(void *PortPtr, u32) {
  auto &Port = *reinterpret_cast<rpc::Server::Port *>(PortPtr);
  if (Port.get_opcode() != SANITIZER_OFFLOAD_CSAN)
    return rpc::RPC_UNHANDLED_OPCODE;

  Port.recv([&](rpc::Buffer *Buffer, u32) {
    __csan_gpu_race R;
    internal_memcpy(&R, Buffer->data, sizeof(R));
    PrintOffloadReport(R);
  });
  return rpc::RPC_SUCCESS;
}

} // namespace __csan
