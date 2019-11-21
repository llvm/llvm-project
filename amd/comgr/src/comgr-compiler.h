/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#ifndef COMGR_COMPILER_H
#define COMGR_COMPILER_H

#include "comgr.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"

namespace COMGR {

class InProcessDriver {
  llvm::raw_ostream &DiagOS;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts;
  clang::TextDiagnosticPrinter *DiagClient;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID;
  clang::DiagnosticsEngine Diags;
  std::unique_ptr<clang::driver::Driver> TheDriver;

  const std::string LinkerJobName = "amdgpu::Linker";

public:
  InProcessDriver(llvm::raw_ostream &DiagOS);
  amd_comgr_status_t execute(llvm::ArrayRef<const char *> Args);
};

/// Manages executing Compiler-related actions.
///
/// @warning No more than one public method should be called on a constructed
/// object before it is destructed.
class AMDGPUCompiler {
  struct AMDGPUCompilerDiagnosticHandler : public llvm::DiagnosticHandler {
    AMDGPUCompiler *Compiler = nullptr;

    AMDGPUCompilerDiagnosticHandler(AMDGPUCompiler *Compiler)
        : Compiler(Compiler) {}

    bool handleDiagnostics(const llvm::DiagnosticInfo &DI) override {
      assert(Compiler && "Compiler cannot be nullptr");
      unsigned Severity = DI.getSeverity();
      switch (Severity) {
      case llvm::DS_Error:
        Compiler->LogS << "ERROR: ";
        break;
      default:
        llvm_unreachable("Only expecting errors");
      }
      llvm::DiagnosticPrinterRawOStream DP(Compiler->LogS);
      DI.print(DP);
      Compiler->LogS << "\n";
      return true;
    }
  };

  DataAction *ActionInfo;
  DataSet *InSet;
  amd_comgr_data_set_t OutSetT;
  /// User supplied target triple.
  std::string Triple;
  /// User supplied target CPU.
  std::string CPU;
  /// User supplied target GPU Arch.
  std::string GPUArch;
  std::string CUDAGPUArch;
  /// HIP and Clang Include Paths
  std::string HIPIncludePath;
  std::string ClangIncludePath;
  /// Perform out-of-process compilation.
  bool CompileOOP = false;
  /// Precompiled header file paths.
  llvm::SmallVector<llvm::SmallString<128>, 2> PrecompiledHeaders;
  /// Arguments common to all driver invocations in the current action.
  llvm::SmallVector<const char *, 128> Args;
  llvm::SmallString<128> TmpDir;
  llvm::SmallString<128> InputDir;
  llvm::SmallString<128> OutputDir;
  llvm::SmallString<128> IncludeDir;
  llvm::raw_ostream &LogS;

  amd_comgr_status_t createTmpDirs();
  amd_comgr_status_t removeTmpDirs();
  amd_comgr_status_t processFile(const char *InputFilePath,
                                 const char *OutputFilePath);
  /// Process each file in @c InSet individually, placing output in @c OutSet.
  amd_comgr_status_t processFiles(amd_comgr_data_kind_t OutputKind,
                                  const char *OutputSuffix);
  amd_comgr_status_t addIncludeFlags();
  amd_comgr_status_t addTargetIdentifierFlags(llvm::StringRef IdentStr,
                                              bool SrcToBC);
  amd_comgr_status_t addCompilationFlags();
  amd_comgr_status_t
  executeOutOfProcessHIPCompilation(llvm::ArrayRef<const char *> Args);

public:
  AMDGPUCompiler(DataAction *ActionInfo, DataSet *InSet, DataSet *OutSet,
                 llvm::raw_ostream &LogS);
  ~AMDGPUCompiler();

  amd_comgr_status_t preprocessToSource();
  amd_comgr_status_t compileToBitcode();
  amd_comgr_status_t linkBitcodeToBitcode();
  amd_comgr_status_t codeGenBitcodeToRelocatable();
  amd_comgr_status_t codeGenBitcodeToAssembly();
  amd_comgr_status_t assembleToRelocatable();
  amd_comgr_status_t linkToRelocatable();
  amd_comgr_status_t linkToExecutable();
  amd_comgr_status_t compileToFatBin();

  amd_comgr_language_t getLanguage() const { return ActionInfo->Language; }
};
} // namespace COMGR

#endif
