//===-- sanitizer_symbolize.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of weak hooks from sanitizer_symbolizer_posix_libcdep.cpp.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <string>

#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"

static llvm::symbolize::LLVMSymbolizer *getDefaultSymbolizer() {
  static llvm::symbolize::LLVMSymbolizer *DefaultSymbolizer =
      new llvm::symbolize::LLVMSymbolizer();
  return DefaultSymbolizer;
}

namespace __sanitizer {
int internal_snprintf(char *buffer, unsigned long length, const char *format,
                      ...);
}  // namespace __sanitizer

extern "C" {

typedef uint64_t u64;

bool __sanitizer_symbolize_code(const char *ModuleName, uint64_t ModuleOffset,
                                char *Buffer, int MaxLength) {
  std::string Result;
  {
    llvm::raw_string_ostream OS(Result);
    llvm::symbolize::DIPrinter Printer(OS);
    // TODO: it is neccessary to set proper SectionIndex here.
    // object::SectionedAddress::UndefSection works for only absolute addresses.
    auto ResOrErr = getDefaultSymbolizer()->symbolizeInlinedCode(
        ModuleName,
        {ModuleOffset, llvm::object::SectionedAddress::UndefSection});
    Printer << (ResOrErr ? ResOrErr.get() : llvm::DIInliningInfo());
  }
  return __sanitizer::internal_snprintf(Buffer, MaxLength, "%s",
                                        Result.c_str()) < MaxLength;
}

bool __sanitizer_symbolize_data(const char *ModuleName, uint64_t ModuleOffset,
                                char *Buffer, int MaxLength) {
  std::string Result;
  {
    llvm::raw_string_ostream OS(Result);
    llvm::symbolize::DIPrinter Printer(OS);
    // TODO: it is neccessary to set proper SectionIndex here.
    // object::SectionedAddress::UndefSection works for only absolute addresses.
    auto ResOrErr = getDefaultSymbolizer()->symbolizeData(
        ModuleName,
        {ModuleOffset, llvm::object::SectionedAddress::UndefSection});
    Printer << (ResOrErr ? ResOrErr.get() : llvm::DIGlobal());
  }
  return __sanitizer::internal_snprintf(Buffer, MaxLength, "%s",
                                        Result.c_str()) < MaxLength;
}

void __sanitizer_symbolize_flush() { getDefaultSymbolizer()->flush(); }

int __sanitizer_symbolize_demangle(const char *Name, char *Buffer,
                                   int MaxLength) {
  std::string Result =
      llvm::symbolize::LLVMSymbolizer::DemangleName(Name, nullptr);
  return __sanitizer::internal_snprintf(Buffer, MaxLength, "%s",
                                        Result.c_str()) < MaxLength
             ? static_cast<int>(Result.size() + 1)
             : 0;
}

}  // extern "C"
