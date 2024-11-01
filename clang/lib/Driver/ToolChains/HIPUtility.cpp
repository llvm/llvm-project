//===--- HIPUtility.cpp - Common HIP Tool Chain Utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIPUtility.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"

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
static std::string normalizeForBundler(const llvm::Triple &T,
                                       bool HasTargetID) {
  return HasTargetID ? (T.getArchName() + "-" + T.getVendorName() + "-" +
                        T.getOSName() + "-" + T.getEnvironmentName())
                           .str()
                     : T.normalize();
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
  std::string BundlerTargetArg = "-targets=host-x86_64-unknown-linux";
  // AMDGCN:
  // For code object version 2 and 3, the offload kind in bundle ID is 'hip'
  // for backward compatibility. For code object version 4 and greater, the
  // offload kind in bundle ID is 'hipv4'.
  std::string OffloadKind = "hip";
  auto &TT = T.getToolChain().getTriple();
  if (TT.isAMDGCN() && getAMDGPUCodeObjectVersion(C.getDriver(), Args) >= 4)
    OffloadKind = OffloadKind + "v4";
  for (const auto &II : Inputs) {
    const auto *A = II.getAction();
    auto ArchStr = llvm::StringRef(A->getOffloadingArch());
    BundlerTargetArg +=
        "," + OffloadKind + "-" + normalizeForBundler(TT, !ArchStr.empty());
    if (!ArchStr.empty())
      BundlerTargetArg += "-" + ArchStr.str();
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

  const char *Bundler = Args.MakeArgString(
      T.getToolChain().GetProgramPath("clang-offload-bundler"));
  C.addCommand(std::make_unique<Command>(
      JA, T, ResponseFileSupport::None(), Bundler, BundlerArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(Output))));
}

/// Add Generated HIP Object File which has device images embedded into the
/// host to the argument list for linking. Using MC directives, embed the
/// device code and also define symbols required by the code generation so that
/// the image can be retrieved at runtime.
void HIP::constructGenerateObjFileFromHIPFatBinary(
    Compilation &C, const InputInfo &Output, const InputInfoList &Inputs,
    const ArgList &Args, const JobAction &JA, const Tool &T) {
  const ToolChain &TC = T.getToolChain();
  std::string Name = std::string(llvm::sys::path::stem(Output.getFilename()));

  // Create Temp Object File Generator,
  // Offload Bundled file and Bundled Object file.
  // Keep them if save-temps is enabled.
  const char *McinFile;
  const char *BundleFile;
  if (C.getDriver().isSaveTempsEnabled()) {
    McinFile = C.getArgs().MakeArgString(Name + ".mcin");
    BundleFile = C.getArgs().MakeArgString(Name + ".hipfb");
  } else {
    auto TmpNameMcin = C.getDriver().GetTemporaryPath(Name, "mcin");
    McinFile = C.addTempFile(C.getArgs().MakeArgString(TmpNameMcin));
    auto TmpNameFb = C.getDriver().GetTemporaryPath(Name, "hipfb");
    BundleFile = C.addTempFile(C.getArgs().MakeArgString(TmpNameFb));
  }
  HIP::constructHIPFatbinCommand(C, JA, BundleFile, Inputs, Args, T);

  // Create a buffer to write the contents of the temp obj generator.
  std::string ObjBuffer;
  llvm::raw_string_ostream ObjStream(ObjBuffer);

  auto HostTriple =
      C.getSingleOffloadToolChain<Action::OFK_Host>()->getTriple();

  // Add MC directives to embed target binaries. We ensure that each
  // section and image is 16-byte aligned. This is not mandatory, but
  // increases the likelihood of data to be aligned with a cache block
  // in several main host machines.
  ObjStream << "#       HIP Object Generator\n";
  ObjStream << "# *** Automatically generated by Clang ***\n";
  if (HostTriple.isWindowsMSVCEnvironment()) {
    ObjStream << "  .section .hip_fatbin, \"dw\"\n";
  } else {
    ObjStream << "  .protected __hip_fatbin\n";
    ObjStream << "  .type __hip_fatbin,@object\n";
    ObjStream << "  .section .hip_fatbin,\"a\",@progbits\n";
  }
  ObjStream << "  .globl __hip_fatbin\n";
  ObjStream << "  .p2align " << llvm::Log2(llvm::Align(HIPCodeObjectAlign))
            << "\n";
  ObjStream << "__hip_fatbin:\n";
  ObjStream << "  .incbin ";
  llvm::sys::printArg(ObjStream, BundleFile, /*Quote=*/true);
  ObjStream << "\n";
  ObjStream.flush();

  // Dump the contents of the temp object file gen if the user requested that.
  // We support this option to enable testing of behavior with -###.
  if (C.getArgs().hasArg(options::OPT_fhip_dump_offload_linker_script))
    llvm::errs() << ObjBuffer;

  // Open script file and write the contents.
  std::error_code EC;
  llvm::raw_fd_ostream Objf(McinFile, EC, llvm::sys::fs::OF_None);

  if (EC) {
    C.getDriver().Diag(clang::diag::err_unable_to_make_temp) << EC.message();
    return;
  }

  Objf << ObjBuffer;

  ArgStringList McArgs{"-triple", Args.MakeArgString(HostTriple.normalize()),
                       "-o",      Output.getFilename(),
                       McinFile,  "--filetype=obj"};
  const char *Mc = Args.MakeArgString(TC.GetProgramPath("llvm-mc"));
  C.addCommand(std::make_unique<Command>(JA, T, ResponseFileSupport::None(), Mc,
                                         McArgs, Inputs, Output));
}
