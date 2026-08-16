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
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

namespace {

template <typename F>
bool get_func(HMODULE module, F** func, char const* name) {
  return ((*func = reinterpret_cast<F*>(reinterpret_cast<void*>(GetProcAddress(module, name)))) != nullptr);
}

// clang-format off
BOOL   (WINAPI* EnumProcessModules)(HANDLE, HMODULE*, DWORD, LPDWORD);
BOOL   (WINAPI* GetModuleInformation)(HANDLE, HMODULE, LPMODULEINFO, DWORD);
DWORD  (WINAPI* GetModuleBaseNameW)(HANDLE, HMODULE, LPWSTR, DWORD);
BOOL   (WINAPI* SymCleanup)(HANDLE);
DWORD  (WINAPI* SymGetOptions)();
BOOL   (WINAPI* SymGetSearchPathW)(HANDLE, PWSTR, DWORD);
BOOL   (WINAPI* SymInitialize)(HANDLE, PCSTR, BOOL);
DWORD  (WINAPI* SymSetOptions)(DWORD);
BOOL   (WINAPI* SymSetSearchPathW)(HANDLE, PCWSTR);
#  ifdef _WIN64
BOOL   (WINAPI* SymGetLineFromAddr)(HANDLE, DWORD64, PDWORD, IMAGEHLP_LINE64*);
BOOL   (WINAPI* SymGetModuleInfo)(HANDLE, DWORD64, PIMAGEHLP_MODULE64);
BOOL   (WINAPI* SymGetSymFromAddr)(HANDLE, DWORD64, PDWORD64, PIMAGEHLP_SYMBOL64);
DWORD64(WINAPI* SymLoadModule)(HANDLE, HANDLE, PCSTR, PCSTR, DWORD64, DWORD);
#  else
BOOL   (WINAPI* SymGetLineFromAddr)(HANDLE, DWORD, PDWORD, IMAGEHLP_LINE*);
BOOL   (WINAPI* SymGetModuleInfo)(HANDLE, DWORD, PIMAGEHLP_MODULE);
BOOL   (WINAPI* SymGetSymFromAddr)(HANDLE, DWORD, PDWORD, PIMAGEHLP_SYMBOL);
DWORD  (WINAPI* SymLoadModule)(HANDLE, HANDLE, PCSTR, PCSTR, DWORD, DWORD);
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
  if (attempted /* but not successful */) {
    return false;
  }

  attempted = true;

  HMODULE psapi   = LoadLibraryExW(L"psapi.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  HMODULE dbghelp = LoadLibraryExW(L"dbghelp.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);

  // clang-format off
  succeeded = true
      && (psapi != nullptr)
      && (dbghelp != nullptr)
      && get_func(psapi, &EnumProcessModules, "EnumProcessModules")
      && get_func(psapi, &GetModuleInformation, "GetModuleInformation")
      && get_func(psapi, &GetModuleBaseNameW, "GetModuleBaseNameW")
      && get_func(dbghelp, &SymCleanup, "SymCleanup")
      && get_func(dbghelp, &SymGetOptions, "SymGetOptions")
      && get_func(dbghelp, &SymGetSearchPathW, "SymGetSearchPathW")
      && get_func(dbghelp, &SymInitialize, "SymInitialize")
      && get_func(dbghelp, &SymSetOptions, "SymSetOptions")
      && get_func(dbghelp, &SymSetSearchPathW, "SymSetSearchPathW")
#ifdef _WIN64
      && get_func(dbghelp, &SymGetLineFromAddr, "SymGetLineFromAddr64")
      && get_func(dbghelp, &SymGetModuleInfo, "SymGetModuleInfo64")
      && get_func(dbghelp, &SymGetSymFromAddr, "SymGetSymFromAddr64")
      && get_func(dbghelp, &SymLoadModule, "SymLoadModule64")
#else
      && get_func(dbghelp, &SymGetLineFromAddr, "SymGetLineFromAddr")
      && get_func(dbghelp, &SymGetModuleInfo, "SymGetModuleInfo")
      && get_func(dbghelp, &SymGetSymFromAddr, "SymGetSymFromAddr")
      && get_func(dbghelp, &SymLoadModule, "SymLoadModule")
#endif
      ;
  // clang-format on

  return succeeded;
}

struct SymInitScope {
  HANDLE proc_;

  explicit SymInitScope(HANDLE proc) : proc_(proc) { SymInitialize(proc_, nullptr, true); }
  ~SymInitScope() { SymCleanup(proc_); }
};

} // namespace

void _Trace::__windows_impl(size_t skip, size_t max_depth) {
  static BOOL loadedDLLFuncs = loadFuncs();
  if (!loadedDLLFuncs) {
    return;
  }

  if (!max_depth) {
    return;
  }

  // Use the Windows Debug Help and Process Status libraries to get a
  // stacktrace.
  //   https://learn.microsoft.com/en-us/windows/win32/debug/debug-help-library
  //   https://learn.microsoft.com/en-us/windows/win32/psapi/process-status-helper

  // These APIs are not thread-safe, according to docs.
  static std::mutex api_mutex;
  std::lock_guard<std::mutex> api_guard(api_mutex);

  HANDLE proc = GetCurrentProcess();

  SymInitScope symscope(proc);

  // Allow space for a handful of paths
  wchar_t sym_path[MAX_PATH * 4];
  if (!SymGetSearchPathW(proc, sym_path, sizeof(sym_path) / sizeof(sym_path[0]))) {
    return;
  }

  wchar_t exe_dir[MAX_PATH];
  if (!GetModuleFileNameW(nullptr, exe_dir, sizeof(exe_dir) / sizeof(exe_dir[0]))) {
    return;
  }
  size_t exe_dir_len = wcslen(exe_dir);
  while (exe_dir_len > 0 && exe_dir[exe_dir_len - 1] != '\\') {
    exe_dir[--exe_dir_len] = 0;
  }
  if (exe_dir_len > 0) {
    exe_dir[--exe_dir_len] = 0; // strip last backslash
  }

  if (!wcsstr(sym_path, exe_dir)) {
    (void)wcsncat(sym_path, L";", sizeof(sym_path) - 1);
    (void)wcsncat(sym_path, exe_dir, sizeof(sym_path) - 1);
    if (!SymSetSearchPathW(proc, sym_path)) {
      return;
    }
  }

  SymSetOptions(SymGetOptions() | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);

  // Skip call to this `current_impl` func
  ++skip;

  // CaptureStackBackTrace is implemented directly by the OS (ntdll's
  // RtlCaptureStackBackTrace) rather than through RtlCaptureContext + StackWalk, so it
  // needs no seeded CONTEXT and no per-architecture register mapping.
  // More importantly it also sidesteps a documented bug in RtlCaptureContext's
  // frame-pointer capture on i686, which was causing every capture to crash there.
  PVOID raw_addrs[_Trace::__absolute_max_depth];
  DWORD to_capture = DWORD(max_depth > _Trace::__absolute_max_depth ? _Trace::__absolute_max_depth : max_depth);
  DWORD captured   = CaptureStackBackTrace(DWORD(skip), to_capture, raw_addrs, nullptr);

  for (DWORD i = 0; i < captured; i++) {
    auto raw_addr = reinterpret_cast<uintptr_t>(raw_addrs[i]);
    if (!raw_addr) {
      break;
    }

    _Entry& entry = __entry_append_();

    // Note: can't differentiate between a signal / exception, or a normal
    // function call. This assumes the more common (presumably) case of
    // normal function calls, so we'll always back up 1 byte to get into the
    // previous (calling) instruction.
    entry.__addr_ = raw_addr - 1;

    // Get the filename of the module containing this calling instruction,
    // i.e. the program itself or a DLL.  This is used in place of the source
    // filename, if the source filename cannot be found (missing PDB, etc.).
    // If the source file can be determined this will be overwritten.
    IMAGEHLP_MODULE mod_info{};
    mod_info.SizeOfStruct = sizeof(mod_info);
    if (SymGetModuleInfo(proc, raw_addr, &mod_info)) {
      entry.__file_.__assign(mod_info.LoadedImageName);
    }
  }

  DWORD need_bytes = 0;
  HMODULE module_handles[1024]{};
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
      entry.__desc_.__assign(sym->Name);
    }

    DWORD linedisp{};
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE);
    if (SymGetLineFromAddr(proc, entry.__addr_, &linedisp, &line)) {
      entry.__file_.__assign(line.FileName);
      entry.__line_ = line.LineNumber;
    }
  }
}

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _WIN32
