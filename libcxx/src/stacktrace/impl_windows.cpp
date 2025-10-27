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

struct _DLL {
  HMODULE module_{};
  bool loaded_{};

  explicit _DLL(char const* name) : module_(LoadLibraryA(name)) {}

  ~_DLL() {
    if (module_) {
      FreeLibrary(module_);
    }
  }

  template <typename F>
  bool get_func(F** func, char const* name) {
    return ((*func = (F*)(void*)GetProcAddress(module_, name)) != nullptr);
  }
};

// clang-format off

struct _Dbghelp_DLL final : _DLL {
  IMAGE_NT_HEADERS* (*ImageNtHeader)(void*);
  bool    (WINAPI *SymCleanup)         (HANDLE);
  DWORD   (WINAPI *SymGetOptions)      ();
  bool    (WINAPI *SymGetSearchPath)   (HANDLE, char const*, DWORD);
  bool    (WINAPI *SymInitialize)      (HANDLE, char const*, bool);
  DWORD   (WINAPI *SymSetOptions)      (DWORD);
  bool    (WINAPI *SymSetSearchPath)   (HANDLE, char const*);
  bool    (WINAPI *StackWalk64)        (DWORD, HANDLE, HANDLE, STACKFRAME64*, void*, void*, void*, void*, void*);
  void*   (WINAPI *SymFunctionTableAccess64)(HANDLE, DWORD64);
  bool    (WINAPI *SymGetLineFromAddr64)(HANDLE, DWORD64, DWORD*, IMAGEHLP_LINE64*);
  DWORD64 (WINAPI *SymGetModuleBase64) (HANDLE, DWORD64);
  bool    (WINAPI *SymGetModuleInfo64) (HANDLE, DWORD64, IMAGEHLP_MODULE64*);
  bool    (WINAPI *SymGetSymFromAddr64)(HANDLE, DWORD64, DWORD64*, IMAGEHLP_SYMBOL64*);
  DWORD64 (WINAPI *SymLoadModule64)    (HANDLE, HANDLE, char const*, char const*, void*, DWORD);

  _Dbghelp_DLL() : _DLL("dbghelp.dll") {
    loaded_ = true
      && get_func(&ImageNtHeader, "ImageNtHeader")
      && get_func(&SymCleanup, "SymCleanup")
      && get_func(&SymGetOptions, "SymGetOptions")
      && get_func(&SymGetSearchPath, "SymGetSearchPath")
      && get_func(&SymInitialize, "SymInitialize")
      && get_func(&SymSetOptions, "SymSetOptions")
      && get_func(&SymSetSearchPath, "SymSetSearchPath")
      && get_func(&StackWalk64, "StackWalk64")
      && get_func(&SymFunctionTableAccess64, "SymFunctionTableAccess64")
      && get_func(&SymGetLineFromAddr64, "SymGetLineFromAddr64")
      && get_func(&SymGetModuleBase64, "SymGetModuleBase64")
      && get_func(&SymGetModuleInfo64, "SymGetModuleInfo64")
      && get_func(&SymGetSymFromAddr64, "SymGetSymFromAddr64")
      && get_func(&SymLoadModule64, "SymLoadModule64")
      ;
  }
};

struct _Psapi_DLL final : _DLL {
  bool  (WINAPI *EnumProcessModules)   (HANDLE, HMODULE*, DWORD, DWORD*);
  bool  (WINAPI *GetModuleInformation) (HANDLE, HMODULE, MODULEINFO*, DWORD);
  DWORD (WINAPI *GetModuleBaseName)    (HANDLE, HMODULE, char**, DWORD);

  _Psapi_DLL() : _DLL("psapi.dll") {
    loaded_ = true
      && get_func(&EnumProcessModules, "EnumProcessModules")
      && get_func(&GetModuleInformation, "GetModuleInformation")
      && get_func(&GetModuleBaseName, "GetModuleBaseNameA")
    ;
  }
};

struct _Sym_Init_Scope {
  _Dbghelp_DLL& dbghelp_;
  HANDLE proc_;

  _Sym_Init_Scope(_Dbghelp_DLL& dbghelp, HANDLE proc)
    : dbghelp_(dbghelp), proc_(proc) {
    (*dbghelp_.SymInitialize)(proc_, nullptr, true);
  }
  ~_Sym_Init_Scope() {
    (*dbghelp_.SymCleanup)(proc_);
  }
};

}  // namespace

_LIBCPP_EXPORTED_FROM_ABI void
_Trace::windows_impl(size_t skip, size_t max_depth) {
  if (!max_depth) [[unlikely]] {
    return;
  }

  static _Psapi_DLL psapi;
  static _Dbghelp_DLL dbghelp;
  if (!psapi.loaded_ || !dbghelp.loaded_) { return; }

  // Not thread-safe according to docs
  static std::mutex api_mutex;
  std::lock_guard<std::mutex> api_guard(api_mutex);

  HANDLE proc = GetCurrentProcess();
  HMODULE exe = GetModuleHandle(nullptr);
  if (!exe) { return; }

  _Sym_Init_Scope symscope(dbghelp, proc);

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
#if defined(_M_AMD64)
  frame.AddrPC.Offset    = ccx.Rip;
  frame.AddrStack.Offset = ccx.Rsp;
  frame.AddrFrame.Offset = ccx.Rbp;
#elif defined(_M_ARM64)
  frame.AddrPC.Offset    = ccx.Pc;
  frame.AddrStack.Offset = ccx.Sp;
  frame.AddrFrame.Offset = ccx.Fp;
#elif defined(_M_IX86)
  frame.AddrPC.Offset    = ccx.Eip;
  frame.AddrStack.Offset = ccx.Esp;
  frame.AddrFrame.Offset = ccx.Ebp;
#else
# warning unrecognized architecture; returned stacktraces will be empty
  return;
#endif

  // Skip call to this `current_impl` func
  ++skip;
  
  while (max_depth) {

    if (!(*dbghelp.StackWalk64)(
          machine, proc, thread, &frame, &ccx, nullptr,
          (void*) dbghelp.SymFunctionTableAccess64,
          (void*) dbghelp.SymGetModuleBase64,
          nullptr)) {
      break;
    }

    if (skip && skip--) { continue; }
    if (!frame.AddrPC.Offset) { break; }

    auto& entry = this->__entry_append_();

    // Note: can't differentiate between a signal / exception, or a normal function call.
    // This assumes the more common (presumably) case of normal function calls, so we'll
    // always back up 1 byte to get into the previous (calling) instruction.
    entry.__addr_ = frame.AddrPC.Offset - 1;

    // Get the filename of the module containing this calling instruction, i.e. the program
    // itself or a DLL.  This is used in place of the source filename, if the source filename
    // cannot be found (missing PDB, etc.).  If the source file can be determined this will
    // be overwritten.
    IMAGEHLP_MODULE64 mod_info;
    memset(&mod_info, 0, sizeof(mod_info));
    mod_info.SizeOfStruct = sizeof(mod_info);
    if ((*dbghelp.SymGetModuleInfo64)(proc, frame.AddrPC.Offset, &mod_info)) {
      entry.assign_file(__create_str()).assign(mod_info.LoadedImageName);
    }

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
    char space[sizeof(IMAGEHLP_SYMBOL64) + kMaxSymName + 1];
    auto* sym          = (IMAGEHLP_SYMBOL64*)space;
    sym->SizeOfStruct  = sizeof(IMAGEHLP_SYMBOL64);
    sym->MaxNameLength = kMaxSymName;
    uint64_t symdisp{0};
    DWORD linedisp{0};
    IMAGEHLP_LINE64 line;
    if ((*dbghelp.SymGetSymFromAddr64)(proc, entry.__addr_, &symdisp, sym)) {
      entry.assign_desc(__create_str()).assign(sym->Name);
    }
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    if ((*dbghelp.SymGetLineFromAddr64)(proc, entry.__addr_, &linedisp, &line)) {
      entry.assign_file(__create_str()).assign(line.FileName);
      entry.__line_ = line.LineNumber;
    }
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif  // _WIN32
