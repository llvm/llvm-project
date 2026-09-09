//===--- HIPUtility.cpp - Common HIP Tool Chain Utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIPUtility.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Options/Options.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace clang::driver;
using namespace clang::driver::tools;
using namespace llvm::opt;

#if defined(_WIN32) || defined(_WIN64)
#define NULL_FILE "nul"
#else
#define NULL_FILE "/dev/null"
#endif

namespace {
const unsigned HIPCodeObjectAlign = 4096;
} // namespace

// Constructs a triple string for clang offload bundler.
static std::string normalizeForBundler(const llvm::Triple &OrigT,
                                       StringRef BoundArch) {
  llvm::Triple T(OrigT);
  bool HasTargetID = !BoundArch.empty();

  // FIXME: Short-term hack. The HIP runtime hardcodes the legacy
  // "amdgcn-amd-amdhsa--" prefix when parsing the target IDs embedded in the
  // fatbin bundle, so force it.
  if (HasTargetID && T.isAMDGCN()) {
    return ("amdgcn-" + T.getVendorName() + "-" + T.getOSName() + "-" +
            T.getEnvironmentName())
        .str();
  }

  return HasTargetID ? (T.getArchName() + "-" + T.getVendorName() + "-" +
                        T.getOSName() + "-" + T.getEnvironmentName())
                           .str()
                     : T.normalize(llvm::Triple::CanonicalForm::FOUR_IDENT);
}

// Construct a clang-offload-bundler command to bundle code objects for
// different devices into a HIP fat binary.
void HIP::constructHIPFatbinCommand(Compilation &C, const JobAction &JA,
                                    llvm::StringRef OutputFileName,
                                    const InputInfoList &Inputs,
                                    const llvm::opt::ArgList &Args,
                                    const Tool &T) {
  // Construct clang-offload-bundler command to bundle object files for
  // for different GPU archs.
  ArgStringList BundlerArgs;
  BundlerArgs.push_back(Args.MakeArgString("-type=o"));
  BundlerArgs.push_back(
      Args.MakeArgString("-bundle-align=" + Twine(HIPCodeObjectAlign)));

  // ToDo: Remove the dummy host binary entry which is required by
  // clang-offload-bundler.
  std::string BundlerTargetArg = "-targets=host-x86_64-unknown-linux-gnu";
  // AMDGCN:
  // For code object version 2 and 3, the offload kind in bundle ID is 'hip'
  // for backward compatibility. For code object version 4 and greater, the
  // offload kind in bundle ID is 'hipv4'.
  std::string OffloadKind = "hip";
  if (T.getToolChain().getTriple().isAMDGCN() &&
      getAMDGPUCodeObjectVersion(C.getDriver(), Args) >= 4)
    OffloadKind = OffloadKind + "v4";
  for (const auto &II : Inputs) {
    const auto *A = II.getAction();
    const llvm::Triple &InputTriple = A->getOffloadingToolChain()->getTriple();

    BoundArch BA = A->getOffloadingArch();
    BundlerTargetArg += ',' + OffloadKind + '-';
    if (BA.ArchName == "amdgcnspirv")
      BundlerTargetArg += "spirv64-amd-amdhsa-";
    else
      BundlerTargetArg += normalizeForBundler(InputTriple, BA.ArchName);
    if (BA)
      BundlerTargetArg += '-' + BA.ArchName.str();
  }
  BundlerArgs.push_back(Args.MakeArgString(BundlerTargetArg));

  // Use a NULL file as input for the dummy host binary entry
  std::string BundlerInputArg = "-input=" NULL_FILE;
  BundlerArgs.push_back(Args.MakeArgString(BundlerInputArg));
  for (const auto &II : Inputs) {
    BundlerInputArg = std::string("-input=") + II.getFilename();
    BundlerArgs.push_back(Args.MakeArgString(BundlerInputArg));
  }

  std::string Output = std::string(OutputFileName);
  auto *BundlerOutputArg =
      Args.MakeArgString(std::string("-output=").append(Output));
  BundlerArgs.push_back(BundlerOutputArg);

  addOffloadCompressArgs(Args, BundlerArgs);

  const char *Bundler = Args.MakeArgString(
      T.getToolChain().GetProgramPath("clang-offload-bundler"));
  C.addCommand(std::make_unique<Command>(
      JA, T, ResponseFileSupport::None(), Bundler, BundlerArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(Output))));
}

// Convenience function for creating temporary file for both modes of
// isSaveTempsEnabled().
const char *HIP::getTempFile(Compilation &C, StringRef Prefix,
                             StringRef Extension) {
  if (C.getDriver().isSaveTempsEnabled()) {
    return C.getArgs().MakeArgString(Prefix + "." + Extension);
  }
  auto TmpFile = C.getDriver().GetTemporaryPath(Prefix, Extension);
  return C.addTempFile(C.getArgs().MakeArgString(TmpFile));
}
