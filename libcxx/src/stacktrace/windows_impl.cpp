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
#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>
#  include <cstring>
#  include <mutex>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

namespace {

namespace __dll {

template <typename F>
bool get_func(HMODULE module, F** func, char const* name) {
  return ((*func = reinterpret_cast<F*>(reinterpret_cast<void*>(GetProcAddress(module, name)))) != nullptr);
}

// clang-format off
bool(WINAPI* EnumProcessModules)(HANDLE, HMODULE*, DWORD, DWORD*);
bool(WINAPI* GetModuleInformation)(HANDLE, HMODULE, MODULEINFO*, DWORD);
DWORD(WINAPI* GetModuleBaseName)(HANDLE, HMODULE, char**, DWORD);
IMAGE_NT_HEADERS* (*ImageNtHeader)(void*);
bool(WINAPI* SymCleanup)(HANDLE);
DWORD(WINAPI* SymGetOptions)();
bool(WINAPI* SymGetSearchPath)(HANDLE, char const*, DWORD);
bool(WINAPI* SymInitialize)(HANDLE, char const*, bool);
DWORD(WINAPI* SymSetOptions)(DWORD);
bool(WINAPI* SymSetSearchPath)(HANDLE, char const*);
#  ifdef _WIN64
void*(WINAPI* SymFunctionTableAccess)(HANDLE, DWORD64);
bool(WINAPI* SymGetLineFromAddr)(HANDLE, DWORD64, DWORD*, IMAGEHLP_LINE64*);
DWORD64(WINAPI* SymGetModuleBase)(HANDLE, DWORD64);
bool(WINAPI* SymGetModuleInfo)(HANDLE, DWORD64, IMAGEHLP_MODULE64*);
bool(WINAPI* SymGetSymFromAddr)(HANDLE, DWORD64, DWORD64*, IMAGEHLP_SYMBOL64*);
DWORD64(WINAPI* SymLoadModule)(HANDLE, HANDLE, char const*, char const*, void*, DWORD);
bool(WINAPI* StackWalk)(DWORD, HANDLE, HANDLE, STACKFRAME64*, void*, void*,
                        decltype(SymFunctionTableAccess), decltype(SymGetModuleBase), void*);
#  else
void*(WINAPI* SymFunctionTableAccess)(HANDLE, DWORD);
bool(WINAPI* SymGetLineFromAddr)(HANDLE, DWORD, DWORD*, IMAGEHLP_LINE*);
DWORD(WINAPI* SymGetModuleBase)(HANDLE, DWORD);
bool(WINAPI* SymGetModuleInfo)(HANDLE, DWORD, IMAGEHLP_MODULE*);
bool(WINAPI* SymGetSymFromAddr)(HANDLE, DWORD, DWORD*, IMAGEHLP_SYMBOL*);
DWORD(WINAPI* SymLoadModule)(HANDLE, HANDLE, char const*, char const*, void*, DWORD);
bool(WINAPI* StackWalk)(DWORD, HANDLE, HANDLE, STACKFRAME*, void*, void*,
                        decltype(SymFunctionTableAccess), decltype(SymGetModuleBase), void*);
#  endif
// clang-format on

bool loadFuncs() {
  static bool attempted{false};
  static bool succeeded{false};
  static std::mutex mutex;

  std::lock_guard<std::mutex> g(mutex);

  if (succeeded) {
    return true;
  }
  if (attempted) {
    return false;
  }

  attempted = true;

  HMODULE psapi   = LoadLibraryA("psapi.dll");
  HMODULE dbghelp = LoadLibraryA("dbghelp.dll");

  // clang-format off
  succeeded = true
      && (psapi != nullptr)
      && (dbghelp != nullptr)
      && get_func(psapi, &EnumProcessModules, "EnumProcessModules")
      && get_func(psapi, &GetModuleInformation, "GetModuleInformation")
      && get_func(psapi, &GetModuleBaseName, "GetModuleBaseNameA")
      && get_func(dbghelp, &ImageNtHeader, "ImageNtHeader")
      && get_func(dbghelp, &SymCleanup, "SymCleanup")
      && get_func(dbghelp, &SymGetOptions, "SymGetOptions")
      && get_func(dbghelp, &SymGetSearchPath, "SymGetSearchPath")
      && get_func(dbghelp, &SymInitialize, "SymInitialize")
      && get_func(dbghelp, &SymSetOptions, "SymSetOptions")
      && get_func(dbghelp, &SymSetSearchPath, "SymSetSearchPath")
#ifdef _WIN64
      && get_func(dbghelp, &StackWalk, "StackWalk64")
      && get_func(dbghelp, &SymFunctionTableAccess, "SymFunctionTableAccess64")
      && get_func(dbghelp, &SymGetLineFromAddr, "SymGetLineFromAddr64")
      && get_func(dbghelp, &SymGetModuleBase, "SymGetModuleBase64")
      && get_func(dbghelp, &SymGetModuleInfo, "SymGetModuleInfo64")
      && get_func(dbghelp, &SymGetSymFromAddr, "SymGetSymFromAddr64")
      && get_func(dbghelp, &SymLoadModule, "SymLoadModule64")
#else
      && get_func(dbghelp, &StackWalk, "StackWalk")
      && get_func(dbghelp, &SymFunctionTableAccess, "SymFunctionTableAccess")
      && get_func(dbghelp, &SymGetLineFromAddr, "SymGetLineFromAddr")
      && get_func(dbghelp, &SymGetModuleBase, "SymGetModuleBase")
      && get_func(dbghelp, &SymGetModuleInfo, "SymGetModuleInfo")
      && get_func(dbghelp, &SymGetSymFromAddr, "SymGetSymFromAddr")
      && get_func(dbghelp, &SymLoadModule, "SymLoadModule")
#endif
      ;
  // clang-format on

  return succeeded;
}

} // namespace __dll

using namespace __dll;

struct _Sym_Init_Scope {
  HANDLE proc_;

  explicit _Sym_Init_Scope(HANDLE proc) : proc_(proc) { SymInitialize(proc_, nullptr, true); }
  ~_Sym_Init_Scope() { SymCleanup(proc_); }
};

} // namespace

_LIBCPP_EXPORTED_FROM_ABI void _Trace::windows_impl(size_t skip, size_t max_depth) {
  static bool loadedDLLFuncs = loadFuncs();
  if (!loadedDLLFuncs) {
    return;
  }

  if (!max_depth) {
    return;
  }

  // Use the Windows Debug Help and Process Status libraries to get a stacktrace.
  //   https://learn.microsoft.com/en-us/windows/win32/debug/debug-help-library
  //   https://learn.microsoft.com/en-us/windows/win32/psapi/process-status-helper

  // These APIs are not thread-safe, according to docs.
  static std::mutex api_mutex;
  std::lock_guard<std::mutex> api_guard(api_mutex);

  HANDLE proc = GetCurrentProcess();
  HMODULE exe = GetModuleHandleA(nullptr);
  if (!exe) {
    return;
  }

  _Sym_Init_Scope symscope(proc);

  // Allow space for a handful of paths
  char sym_path[MAX_PATH * 4];
  if (!SymGetSearchPath(proc, sym_path, sizeof(sym_path))) {
    return;
  }

  char exe_dir[MAX_PATH];
  if (!GetModuleFileNameA(nullptr, exe_dir, sizeof(exe_dir))) {
    return;
  }
  size_t exe_dir_len = strlen(exe_dir);
  while (exe_dir_len > 0 && exe_dir[exe_dir_len - 1] != '\\') {
    exe_dir[--exe_dir_len] = 0;
  }
  if (exe_dir_len > 0) {
    exe_dir[--exe_dir_len] = 0;
  } // strip last backslash

  if (!strstr(sym_path, exe_dir)) {
    (void)strncat(sym_path, ";", sizeof(sym_path) - 1);
    (void)strncat(sym_path, exe_dir, sizeof(sym_path) - 1);
    if (!SymSetSearchPath(proc, sym_path)) {
      return;
    }
  }

  IMAGE_NT_HEADERS* nt_headers;
  if (!(nt_headers = ImageNtHeader(exe))) {
    return;
  }

  SymSetOptions(SymGetOptions() | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);

  HANDLE thread = GetCurrentThread();
  int machine   = nt_headers->FileHeader.Machine;

  CONTEXT ccx;
  RtlCaptureContext(&ccx);

  STACKFRAME frame;
  memset(&frame, 0, sizeof(frame));
  frame.AddrPC.Mode    = AddrModeFlat;
  frame.AddrStack.Mode = AddrModeFlat;
  frame.AddrFrame.Mode = AddrModeFlat;
#  if defined(_M_AMD64)
  frame.AddrPC.Offset    = ccx.Rip;
  frame.AddrStack.Offset = ccx.Rsp;
  frame.AddrFrame.Offset = ccx.Rbp;
#  elif defined(_M_ARM64)
  frame.AddrPC.Offset    = ccx.Pc;
  frame.AddrStack.Offset = ccx.Sp;
  frame.AddrFrame.Offset = ccx.Fp;
#  elif defined(_M_IX86)
  frame.AddrPC.Offset    = ccx.Eip;
  frame.AddrStack.Offset = ccx.Esp;
  frame.AddrFrame.Offset = ccx.Ebp;
#  else
#    warning unrecognized architecture; returned stacktraces will be empty
  return;
#  endif

  // Skip call to this `current_impl` func
  ++skip;

  while (max_depth) {
    if (!StackWalk(machine, proc, thread, &frame, &ccx, nullptr, SymFunctionTableAccess, SymGetModuleBase, nullptr)) {
      break;
    }

    if (skip && skip--) {
      continue;
    }
    if (!frame.AddrPC.Offset) {
      break;
    }

    _Entry& entry = this->__entry_append_();

    // Note: can't differentiate between a signal / exception, or a normal function call.
    // This assumes the more common (presumably) case of normal function calls, so we'll
    // always back up 1 byte to get into the previous (calling) instruction.
    entry.__addr_ = frame.AddrPC.Offset - 1;

    // Get the filename of the module containing this calling instruction, i.e. the program
    // itself or a DLL.  This is used in place of the source filename, if the source filename
    // cannot be found (missing PDB, etc.).  If the source file can be determined this will
    // be overwritten.
    IMAGEHLP_MODULE mod_info;
    memset(&mod_info, 0, sizeof(mod_info));
    mod_info.SizeOfStruct = sizeof(mod_info);
    if (SymGetModuleInfo(proc, frame.AddrPC.Offset, &mod_info)) {
      entry.__file_.assign(mod_info.LoadedImageName);
    }

    --max_depth;
  }

  DWORD need_bytes = 0;
  HMODULE module_handles[1024]{0};
  if (!EnumProcessModules(proc, module_handles, sizeof(module_handles), LPDWORD(&need_bytes))) {
    return;
  }

  // https://learn.microsoft.com/en-us/cpp/build/reference/h-restrict-length-of-external-names
  constexpr static size_t __max_sym_len = 2047;

  for (_Entry& entry : __entry_iters_()) {
    char space[sizeof(IMAGEHLP_SYMBOL) + __max_sym_len + 1];
    IMAGEHLP_SYMBOL* sym = reinterpret_cast<IMAGEHLP_SYMBOL*>(space);
    sym->SizeOfStruct    = sizeof(IMAGEHLP_SYMBOL);
    sym->MaxNameLength   = __max_sym_len;

#  if defined(_WIN64)
    DWORD64 symdisp{};
#  else
    DWORD symdisp{};
#  endif
    IMAGEHLP_LINE line;
    if (SymGetSymFromAddr(proc, entry.__addr_, &symdisp, sym)) {
      entry.__desc_.assign(sym->Name);
    }

    DWORD linedisp{};
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE);
    if (SymGetLineFromAddr(proc, entry.__addr_, &linedisp, &line)) {
      entry.__file_.assign(line.FileName);
      entry.__line_ = line.LineNumber;
    }
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _WIN32
