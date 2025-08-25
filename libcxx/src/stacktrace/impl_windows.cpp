//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32)

#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
//
#  include <dbghelp.h>
#  include <psapi.h>
//
#  include <cstring>
#  include <iostream>
#  include <mutex>
#  include <stacktrace>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

namespace {

struct dll {
  HMODULE module_{};
  bool loaded_{};

  explicit dll(char const* name) : module_(LoadLibraryA(name)) {}

  ~dll() {
    if (module_) {
      FreeLibrary(module_);
    }
  }

  template <typename F>
  bool get_func(F* func, char const* name) {
    *func = (F)GetProcAddress(module_, name);
    return func != nullptr;
  }
};

// clang-format off

struct dbghelp_dll final : dll {
  IMAGE_NT_HEADERS* (*ImageNtHeader)(void*);
  bool    (*StackWalk64)        (DWORD, HANDLE, HANDLE, STACKFRAME64*, void*, void*, void*, void*, void*);
  bool    (*SymCleanup)         (HANDLE);
  void*   (*SymFunctionTableAccess64)(HANDLE, DWORD64);
  bool    (*SymGetLineFromAddr64)(HANDLE, DWORD64, DWORD*, IMAGEHLP_LINE64*);
  DWORD64 (*SymGetModuleBase64) (HANDLE, DWORD64);
  DWORD   (*SymGetOptions)      ();
  bool    (*SymGetSearchPath)   (HANDLE, char const*, DWORD);
  bool    (*SymGetSymFromAddr64)(HANDLE, DWORD64, DWORD64*, IMAGEHLP_SYMBOL64*);
  bool    (*SymInitialize)      (HANDLE, char const*, bool);
  DWORD64 (*SymLoadModule64)    (HANDLE, HANDLE, char const*, char const*, void*, DWORD);
  DWORD   (*SymSetOptions)      (DWORD);
  bool    (*SymSetSearchPath)   (HANDLE, char const*);

  dbghelp_dll() : dll("dbghelp.dll") {
    loaded_ = true
      && get_func(&ImageNtHeader, "ImageNtHeader")
      && get_func(&StackWalk64, "StackWalk64")
      && get_func(&SymCleanup, "SymCleanup")
      && get_func(&SymFunctionTableAccess64, "SymFunctionTableAccess64")
      && get_func(&SymGetLineFromAddr64, "SymGetLineFromAddr64")
      && get_func(&SymGetModuleBase64, "SymGetModuleBase64")
      && get_func(&SymGetOptions, "SymGetOptions")
      && get_func(&SymGetSearchPath, "SymGetSearchPath")
      && get_func(&SymGetSymFromAddr64, "SymGetSymFromAddr64")
      && get_func(&SymInitialize, "SymInitialize")
      && get_func(&SymLoadModule64, "SymLoadModule64")
      && get_func(&SymSetOptions, "SymSetOptions")
      && get_func(&SymSetSearchPath, "SymSetSearchPath")
    ;
  }
};

struct psapi_dll final : dll {
  bool  (*EnumProcessModules)   (HANDLE, HMODULE*, DWORD, DWORD*);
  bool  (*GetModuleInformation) (HANDLE, HMODULE, MODULEINFO*, DWORD);
  DWORD (*GetModuleBaseName)    (HANDLE, HMODULE, char**, DWORD);

  psapi_dll() : dll("psapi.dll") {
    loaded_ = true
      && get_func(&EnumProcessModules, "EnumProcessModules")
      && get_func(&GetModuleInformation, "GetModuleInformation")
      && get_func(&GetModuleBaseName, "GetModuleBaseNameA")
    ;
  }
};

struct sym_init_scope {
  dbghelp_dll& dbghelp_;
  HANDLE proc_;

  sym_init_scope(dbghelp_dll& dbghelp, HANDLE proc)
    : dbghelp_(dbghelp), proc_(proc) {
    (*dbghelp_.SymInitialize)(proc_, nullptr, true);
  }
  ~sym_init_scope() {
    (*dbghelp_.SymCleanup)(proc_);
  }
};

}  // namespace

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI void
base::current_impl(size_t skip, size_t max_depth) {
  if (!max_depth) [[unlikely]] {
    return;
  }

  static psapi_dll psapi;
  static dbghelp_dll dbghelp;
  if (!psapi.loaded_ || !dbghelp.loaded_) { return; }

  // Not thread-safe according to docs
  static std::mutex api_mutex;
  std::lock_guard<std::mutex> api_guard(api_mutex);

  HANDLE proc;
  proc = GetCurrentProcess();

  HMODULE exe;
  if (!(exe = GetModuleHandle(nullptr))) { return; }

  sym_init_scope symscope(dbghelp, proc);

  char sym_path[MAX_PATH * 4]; // arbitrary
  if (!(*dbghelp.SymGetSearchPath)(proc, sym_path, sizeof(sym_path))) { return; }

  char exe_dir[MAX_PATH];
  if (!GetModuleFileNameA(nullptr, exe_dir, sizeof(exe_dir))) { return; }
  size_t exe_dir_len = strlen(exe_dir);
  while (exe_dir_len > 0 && exe_dir[exe_dir_len - 1] != '\\') { exe_dir[--exe_dir_len] = 0; }
  if (exe_dir_len > 0) { exe_dir[--exe_dir_len] = 0; }  // strip last backslash

  if (!strstr(sym_path, exe_dir)) {
    (void) strncat(sym_path, ";", sizeof(sym_path) - 1);
    (void) strncat(sym_path, exe_dir, sizeof(sym_path) - 1);
    if (!(*dbghelp.SymSetSearchPath)(proc, sym_path)) { return; }
  }

  IMAGE_NT_HEADERS* nt_headers;
  if (!(nt_headers = (*dbghelp.ImageNtHeader)(exe))) { return; }

  (*dbghelp.SymSetOptions)(
    (*dbghelp.SymGetOptions)()
    | SYMOPT_LOAD_LINES
    | SYMOPT_UNDNAME);

  auto thread  = GetCurrentThread();
  auto machine = nt_headers->FileHeader.Machine;

  CONTEXT ccx;
  RtlCaptureContext(&ccx);

  STACKFRAME64 frame;
  memset(&frame, 0, sizeof(frame));
  frame.AddrPC.Mode      = AddrModeFlat;
  frame.AddrStack.Mode   = AddrModeFlat;
  frame.AddrFrame.Mode   = AddrModeFlat;
#if defined(_M_IX86)
  frame.AddrPC.Offset    = ccx.Eip;
  frame.AddrStack.Offset = ccx.Esp;
  frame.AddrFrame.Offset = ccx.Ebp;
#elif defined(_M_AMD64)
  frame.AddrPC.Offset    = ccx.Rip;
  frame.AddrStack.Offset = ccx.Rsp;
  frame.AddrFrame.Offset = ccx.Rbp;
#elif defined(_M_ARM)
  frame.AddrPC.Offset    = ccx.Pc;
  frame.AddrStack.Offset = ccx.Sp;
  frame.AddrFrame.Offset = ccx.Fp;
#elif defined(_M_ARM64)
  frame.AddrPC.Offset    = ccx.Pc;
  frame.AddrStack.Offset = ccx.Sp;
  frame.AddrFrame.Offset = ccx.Fp;
#else
#error Unhandled CPU/arch for stacktrace
#endif

  ++skip;  // skip call to this `populate` func
  while (max_depth) {
    if (!(*dbghelp.StackWalk64)(
          machine, proc, thread, &frame, &ccx, nullptr,
          dbghelp.SymFunctionTableAccess64, dbghelp.SymGetModuleBase64,
          nullptr)) {
      break; }

    if (skip && skip--) { continue; }
    if (!frame.AddrPC.Offset) { break; }

    auto& entry = this->__entry_append_();
    // Note: can't differentiate between a signal, SEH exception handler, or a normal function call
    entry.__addr_ = frame.AddrPC.Offset - 1; // Back up 1 byte to get into prev insn range

    --max_depth;
  }

  DWORD need_bytes = 0;
  HMODULE module_handles[1024] {0};
  if (!(*psapi.EnumProcessModules)(
          proc, module_handles, sizeof(module_handles), LPDWORD(&need_bytes))) {
    return;
  }

  // Symbols longer than this will be truncated.
  static constexpr size_t kMaxSymName = 256;

  for (auto& entry : __entry_iters_()) {    
#if defined(_M_ARM64) || defined(_M_AMD64)
    char space[sizeof(IMAGEHLP_SYMBOL64) + kMaxSymName + 1];
    auto* sym          = (IMAGEHLP_SYMBOL64*)space;
    sym->SizeOfStruct  = sizeof(IMAGEHLP_SYMBOL64);
    sym->MaxNameLength = kMaxSymName;
    uint64_t symdisp{0};
    DWORD linedisp{0};
    IMAGEHLP_LINE64 line;
    if ((*dbghelp.SymGetSymFromAddr64)(proc, entry.__addr_, &symdisp, sym)) {
      entry.assign_desc(__strings_.create()).assign(sym->Name);
    }
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    if ((*dbghelp.SymGetLineFromAddr64)(proc, entry.__addr_, &linedisp, &line)) {
      entry.assign_file(__strings_.create()).assign(line.FileName);
      entry.__line_ = line.LineNumber;
    }
#else
    char space[sizeof(IMAGEHLP_SYMBOL) + kMaxSymName + 1];
    auto* sym          = (IMAGEHLP_SYMBOL*)space;
    sym->SizeOfStruct  = sizeof(IMAGEHLP_SYMBOL);
    sym->MaxNameLength = kMaxSymName;
    uint32_t symdisp{0};
    DWORD linedisp{0};
    IMAGEHLP_LINE line;
    if ((*dbghelp.SymGetSymFromAddr)(proc, entry.__addr_, &symdisp, sym)) {
      entry.assign_desc(__strings_.create()).assign(sym->Name);
    }
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE);
    if ((*dbghelp.SymGetLineFromAddr)(proc, entry.__addr_, &linedisp, &line)) {
      entry.assign_file(__strings_.create()).assign(line.FileName);
      entry.__line_ = line.LineNumber;
    }
#endif
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif  // _WIN32
