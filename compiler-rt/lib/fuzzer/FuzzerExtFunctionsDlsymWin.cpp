//===- FuzzerExtFunctionsDlsymWin.cpp - Interface to external functions ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Implementation using dynamic loading for Windows.
//===----------------------------------------------------------------------===//
#include "FuzzerDefs.h"
#if LIBFUZZER_WINDOWS

#include "FuzzerExtFunctions.h"
#include "FuzzerIO.h"
#include "Windows.h"

// This must be included after Windows.h.
#include "Psapi.h"

namespace fuzzer {

ExternalFunctions::ExternalFunctions() {
  HMODULE Modules[1024];
  DWORD BytesNeeded;
  HANDLE CurrentProcess = GetCurrentProcess();

  if (!EnumProcessModules(CurrentProcess, Modules, sizeof(Modules),
                          &BytesNeeded)) {
    Printf("EnumProcessModules failed (error: %d).\n", GetLastError());
    exit(1);
  }

  if (sizeof(Modules) < BytesNeeded) {
    Printf("Error: the array is not big enough to hold all loaded modules.\n");
    exit(1);
  }

  for (size_t i = 0; i < (BytesNeeded / sizeof(HMODULE)); i++)
  {
    FARPROC Fn;
#define EXT_FUNC(NAME, RETURN_TYPE, FUNC_SIG, WARN)                            \
    if (this->NAME == nullptr) {                                               \
      Fn = GetProcAddress(Modules[i], #NAME);                                  \
      if (Fn == nullptr)                                                       \
         Fn = GetProcAddress(Modules[i], #NAME "__dll");                       \
      this->NAME = (decltype(ExternalFunctions::NAME)) Fn;                     \
    }
#include "FuzzerExtFunctions.def"
#undef EXT_FUNC
  }

#define EXT_FUNC(NAME, RETURN_TYPE, FUNC_SIG, WARN)                            \
  if (this->NAME == nullptr && WARN)                                           \
    Printf("WARNING: Failed to find function \"%s\".\n", #NAME);
#include "FuzzerExtFunctions.def"
#undef EXT_FUNC
}

} // namespace fuzzer

#endif // LIBFUZZER_WINDOWS
