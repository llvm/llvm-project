//===--- CodeGenOptions.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CodeGenOptions.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>
#include <string.h>

namespace Fortran::frontend {

using namespace llvm;

CodeGenOptions::CodeGenOptions() {
#define CODEGENOPT(Name, Bits, Default) Name = Default;
#define ENUM_CODEGENOPT(Name, Type, Bits, Default) set##Name(Default);
#include "flang/Frontend/CodeGenOptions.def"
}

// Check if ASan should use GC-friendly instrumentation for globals.
// First of all, there is no point if -fdata-sections is off (expect for MachO,
// where this is not a factor). Also, on ELF this feature requires an assembler
// extension that only works with -integrated-as at the moment.
bool asanUseGlobalsGC(const Triple &T, const CodeGenOptions &CGOpts) {
  if (!CGOpts.SanitizeAddressGlobalsDeadStripping)
    return false;
  switch (T.getObjectFormat()) {
  case Triple::MachO:
  case Triple::COFF:
    return true;
  case Triple::ELF:
    return !CGOpts.DisableIntegratedAS;
  case Triple::GOFF:
    llvm::report_fatal_error("ASan not implemented for GOFF");
  case Triple::XCOFF:
    llvm::report_fatal_error("ASan not implemented for XCOFF.");
  case Triple::Wasm:
  case Triple::DXContainer:
  case Triple::SPIRV:
  case Triple::UnknownObjectFormat:
    break;
  }
  return false;
}

std::optional<llvm::CodeModel::Model> getCodeModel(llvm::StringRef string) {
  return llvm::StringSwitch<std::optional<llvm::CodeModel::Model>>(string)
      .Case("tiny", llvm::CodeModel::Model::Tiny)
      .Case("small", llvm::CodeModel::Model::Small)
      .Case("kernel", llvm::CodeModel::Model::Kernel)
      .Case("medium", llvm::CodeModel::Model::Medium)
      .Case("large", llvm::CodeModel::Model::Large)
      .Default(std::nullopt);
}

} // end namespace Fortran::frontend
