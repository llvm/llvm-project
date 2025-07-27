//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#if defined(_LIBCPP_WIN32API)

// windows.h must be first
#  include <windows.h>
// other windows-specific headers
#  include <dbghelp.h>
#  define PSAPI_VERSION 1
#  include <psapi.h>

#  include "stacktrace/windows/dll.h"
#  include "stacktrace/windows/impl.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

win_impl::~win_impl() {
  auto& dbg = dbghelp_dll::get();
  if (initialized_) {
    (*dbg.SymCleanup)(proc_);
    initialized_ = false;
  }
}

win_impl::win_impl(base& base) : base_(base) {
  std::lock_guard<std::mutex> guard(mutex_);

  auto& dbg = dbghelp_dll::get();
  auto& ps  = psapi_dll::get();

  if (!initialized_) {
    // Cannot proceed without these DLLs:
    if (!dbg) {
      return;
    }
    if (!ps) {
      return;
    }
    proc_ = GetCurrentProcess();
    if (!(exe_ = GetModuleHandle(nullptr))) {
      return;
    }
    if (!(nt_headers_ = (*dbg.ImageNtHeader)(exe_))) {
      return;
    }

    initialized_ = true;
  }

  // The final `true` means we want the runtime to enumerate all this
  // process's modules' symbol tables.
  initialized_     = (*dbg.SymInitialize)(proc_, nullptr, true);
  DWORD symOptions = (*dbg.SymGetOptions)();
  symOptions |= SYMOPT_LOAD_LINES | SYMOPT_UNDNAME;
  (*dbg.SymSetOptions)(symOptions);
}

void win_impl::ident_modules() {
  if (!initialized_) {
    return;
  }

  auto& ps = psapi_dll::get();
  DWORD needBytes;

  auto enumMods = (*ps.EnumProcessModules)(proc_, module_handles_, sizeof(module_handles_), LPDWORD(&needBytes));
  if (enumMods) {
    module_count_ = needBytes / sizeof(HMODULE);
  } else {
    module_count_ = 0;
  }
}

void win_impl::symbolize() {
  if (!initialized_) {
    return;
  }

  // Very long symbols longer than this amount will be truncated.
  static constexpr size_t kMaxSymName = 256;

  auto& dbg = dbghelp_dll::get();

  for (auto& entry : base_.__entries_) {
    char space[sizeof(IMAGEHLP_SYMBOL64) + kMaxSymName];
    auto* sym          = (IMAGEHLP_SYMBOL64*)space;
    sym->SizeOfStruct  = sizeof(IMAGEHLP_SYMBOL64);
    sym->MaxNameLength = kMaxSymName;
    uint64_t disp{0};
    if ((*dbg.SymGetSymFromAddr64)(proc_, entry.__addr_actual_, &disp, sym)) {
      // Copy chars into the destination string which uses the caller-provided allocator.
      ((entry_base&)entry).__desc_ = {sym->Name};
    }
  }
}

void win_impl::resolve_lines() {
  if (!initialized_) {
    return;
  }

  auto& dbg = dbghelp_dll::get();

  for (auto& entry : base_.__entries_) {
    DWORD disp{0};
    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    if ((*dbg.SymGetLineFromAddr64)(proc_, entry.__addr_actual_, &disp, &line)) {
      // Copy chars into the destination string which uses the caller-provided allocator.
      entry.__file_ = line.FileName;
      entry.__line_ = line.LineNumber;
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
  if (!initialized_) {
    return;
  }

  auto& dbg    = dbghelp_dll::get();
  auto thread  = GetCurrentThread();
  auto machine = nt_headers_->FileHeader.Machine;

  CONTEXT ccx;
  RtlCaptureContext(&ccx);

  STACKFRAME64 frame;
  memset(&frame, 0, sizeof(frame));
  frame.AddrPC.Mode      = AddrModeFlat;
  frame.AddrStack.Mode   = AddrModeFlat;
  frame.AddrFrame.Mode   = AddrModeFlat;
  frame.AddrPC.Offset    = ccx.Rip;
  frame.AddrStack.Offset = ccx.Rsp;
  frame.AddrFrame.Offset = ccx.Rbp;

  while (max_depth &&
         (*dbg.StackWalk64)(
             machine,
             proc_,
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
    auto& entry = base_.__entries_.emplace_back();
    // We don't need to compute the un-slid addr; windbg only needs the actual addresses.
    // Assume address is of the instruction after a call instruction, since we can't
    // differentiate between a signal, SEH exception handler, or a normal function call.
    entry.__addr_actual_ = frame.AddrPC.Offset - 1; // Back up 1 byte to get into prev insn range
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
