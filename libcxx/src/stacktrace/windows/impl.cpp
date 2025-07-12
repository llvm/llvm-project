//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_LIBCPP_WIN32API)
// windows.h must be first
#  include <windows.h>
// other windows-specific headers
#  include <dbghelp.h>
#  define PSAPI_VERSION 1
#  include <psapi.h>

#  include <stacktrace>

#  include "stacktrace/utils/debug.h"
#  include "stacktrace/windows/dll.h"
#  include "stacktrace/windows/impl.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

namespace {

/*
Global objects, shared among all threads and among all stacktrace operations.
The `dbghelp` APIs are not safe to call concurrently (according to their docs)
so we claim a lock in the `WinDebugAPIs` constructor.
*/

// Statically-initialized
DbgHelpDLL dbg;
PSAPIDLL ps;

// Initialized once, in first WinDebugAPIs construction;
// protected by the above mutex.
HANDLE proc;
HMODULE exe;
IMAGE_NT_HEADERS* ntHeaders;
bool globalInitialized{false};

// Globals used across invocations of the functions below.
// Also guarded by the above mutex.
bool symsInitialized{false};
HMODULE moduleHandles[1024];
size_t moduleCount; // 0 IFF module enumeration failed

} // namespace

win_impl::global_init() {
  if (!globalInitialized) {
    // Cannot proceed without these DLLs:
    if (!dbg) {
      return;
    }
    if (!ps) {
      return;
    }
    proc = GetCurrentProcess();
    if (!(exe = GetModuleHandle(nullptr))) {
      return;
    }
    if (!(ntHeaders = (*dbg.ImageNtHeader)(exe))) {
      return;
    }

    globalInitialized = true;
  }

  // Initialize symbol machinery.
  // Presumably the symbols in this process's space can change between
  // stacktraces, so we'll do this each time we take a trace.
  // The final `true` means we want the runtime to enumerate all this
  // process's modules' symbol tables.
  symsInitialized  = (*dbg.SymInitialize)(proc, nullptr, true);
  DWORD symOptions = (*dbg.SymGetOptions)();
  symOptions |= SYMOPT_LOAD_LINES | SYMOPT_UNDNAME;
  (*dbg.SymSetOptions)(symOptions);
}

win_impl::~win_impl() {
  if (symsInitialized) {
    (*dbg.SymCleanup)(proc);
    symsInitialized = false;
  }
}

void win_impl::ident_modules() {
  if (!globalInitialized) {
    return;
  }
  DWORD needBytes;
  auto enumMods = (*ps.EnumProcessModules)(proc, moduleHandles, sizeof(moduleHandles), LPDWORD(&needBytes));
  if (enumMods) {
    moduleCount = needBytes / sizeof(HMODULE);
  } else {
    moduleCount = 0;
    Debug() << "EnumProcessModules failed: " << GetLastError() << '\n';
  }
}

void win_impl::symbolize() {
  // Very long symbols longer than this amount will be truncated.
  static constexpr size_t kMaxSymName = 256;
  if (!globalInitialized) {
    return;
  }

  for (auto& entry : builder_.__entries_) {
    char space[sizeof(IMAGEHLP_SYMBOL64) + kMaxSymName];
    auto* sym          = (IMAGEHLP_SYMBOL64*)space;
    sym->SizeOfStruct  = sizeof(IMAGEHLP_SYMBOL64);
    sym->MaxNameLength = kMaxSymName;
    uint64_t disp{0};
    if ((*dbg.SymGetSymFromAddr64)(proc, entry.__addr_actual_, &disp, sym)) {
      // Copy chars into the destination string which uses the caller-provided allocator.
      ((entry_base&)entry).__desc_ = {sym->Name};
    } else {
      Debug() << "SymGetSymFromAddr64 failed: " << GetLastError() << '\n';
    }
  }
}

void win_impl::resolve_lines() {
  if (!globalInitialized) {
    return;
  }

  for (auto& entry : builder_.__entries_) {
    DWORD disp{0};
    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    if ((*dbg.SymGetLineFromAddr64)(proc, entry.__addr_actual_, &disp, &line)) {
      // Copy chars into the destination string which uses the caller-provided allocator.
      entry.__file_ = line.FileName;
      entry.__line_ = line.LineNumber;
    } else {
      Debug() << "SymGetLineFromAddr64 failed: " << GetLastError() << '\n';
    }
  }
}

/*
Inlining is disabled from here on;
this is to ensure `collect` below doesn't get merged into its caller
and mess around with the top of the stack (making `skip` inaccurate).
*/
#  pragma auto_inline(off)

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void win_impl::collect(size_t skip, size_t max_depth) {
  if (!globalInitialized) {
    return;
  }

  auto thread  = GetCurrentThread();
  auto machine = ntHeaders->FileHeader.Machine;

  CONTEXT ccx;
  RtlCaptureContext(&ccx);

  STACKFRAME64 frame;
  memset(&frame, 0, sizeof(frame));
  frame.AddrPC.Mode      = AddrModeFlat;
  frame.AddrStack.Mode   = AddrModeFlat;
  frame.AddrFrame.Mode   = AddrModeFlat;
  frame.AddrPC.Offset    = ctrace.Rip;
  frame.AddrStack.Offset = ctrace.Rsp;
  frame.AddrFrame.Offset = ctrace.Rbp;

  while (max_depth &&
         (*dbg.StackWalk64)(
             machine,
             proc,
             thread,
             &frame,
             &ccx,
             nullptr,
             dbg.SymFunctionTableAccess64,
             dbg.SymGetModuleBase64,
             nullptr)) {
    if (skip) {
      --skip;
      continue;
    }
    --max_depth;
    auto& entry = builder_.__entries_.emplace_back();
    // We don't need to compute the un-slid addr; windbg only needs the actual addresses.
    // Assume address is of the instruction after a call instruction, since we can't
    // differentiate between a signal, SEH exception handler, or a normal function call.
    entry.__addr_actual_ = frame.AddrPC.Offset - 1; // Back up 1 byte to get into prev insn range
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
