//===-- Options.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Options.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/InstallAPI/FileList.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"

using namespace clang::driver;
using namespace clang::driver::options;
using namespace llvm::opt;
using namespace llvm::MachO;

namespace clang {
namespace installapi {

bool Options::processDriverOptions(InputArgList &Args) {
  // Handle inputs.
  llvm::append_range(DriverOpts.FileLists, Args.getAllArgValues(OPT_INPUT));

  // Handle output.
  SmallString<PATH_MAX> OutputPath;
  if (auto *Arg = Args.getLastArg(OPT_o)) {
    OutputPath = Arg->getValue();
    if (OutputPath != "-")
      FM->makeAbsolutePath(OutputPath);
    DriverOpts.OutputPath = std::string(OutputPath);
  }

  // Do basic error checking first for mixing -target and -arch options.
  auto *ArgArch = Args.getLastArgNoClaim(OPT_arch);
  auto *ArgTarget = Args.getLastArgNoClaim(OPT_target);
  auto *ArgTargetVariant =
      Args.getLastArgNoClaim(OPT_darwin_target_variant_triple);
  if (ArgArch && (ArgTarget || ArgTargetVariant)) {
    Diags->Report(clang::diag::err_drv_argument_not_allowed_with)
        << ArgArch->getAsString(Args)
        << (ArgTarget ? ArgTarget : ArgTargetVariant)->getAsString(Args);
    return false;
  }

  auto *ArgMinTargetOS = Args.getLastArgNoClaim(OPT_mtargetos_EQ);
  if ((ArgTarget || ArgTargetVariant) && ArgMinTargetOS) {
    Diags->Report(clang::diag::err_drv_cannot_mix_options)
        << ArgTarget->getAsString(Args) << ArgMinTargetOS->getAsString(Args);
    return false;
  }

  // Capture target triples first.
  if (ArgTarget) {
    for (const Arg *A : Args.filtered(OPT_target)) {
      A->claim();
      llvm::Triple TargetTriple(A->getValue());
      Target TAPITarget = Target(TargetTriple);
      if ((TAPITarget.Arch == AK_unknown) ||
          (TAPITarget.Platform == PLATFORM_UNKNOWN)) {
        Diags->Report(clang::diag::err_drv_unsupported_opt_for_target)
            << "installapi" << TargetTriple.str();
        return false;
      }
      DriverOpts.Targets[TAPITarget] = TargetTriple;
    }
  }

  DriverOpts.Verbose = Args.hasArgNoClaim(OPT_v);

  return true;
}

bool Options::processLinkerOptions(InputArgList &Args) {
  // TODO: add error handling.

  // Required arguments.
  if (const Arg *A = Args.getLastArg(options::OPT_install__name))
    LinkerOpts.InstallName = A->getValue();

  // Defaulted or optional arguments.
  if (auto *Arg = Args.getLastArg(OPT_current__version))
    LinkerOpts.CurrentVersion.parse64(Arg->getValue());

  if (auto *Arg = Args.getLastArg(OPT_compatibility__version))
    LinkerOpts.CompatVersion.parse64(Arg->getValue());

  LinkerOpts.IsDylib = Args.hasArg(OPT_dynamiclib);

  LinkerOpts.AppExtensionSafe =
      Args.hasFlag(OPT_fapplication_extension, OPT_fno_application_extension,
                   /*Default=*/LinkerOpts.AppExtensionSafe);

  if (::getenv("LD_NO_ENCRYPT") != nullptr)
    LinkerOpts.AppExtensionSafe = true;

  if (::getenv("LD_APPLICATION_EXTENSION_SAFE") != nullptr)
    LinkerOpts.AppExtensionSafe = true;
  return true;
}

bool Options::processFrontendOptions(InputArgList &Args) {
  // Do not claim any arguments, as they will be passed along for CC1
  // invocations.
  if (auto *A = Args.getLastArgNoClaim(OPT_x)) {
    FEOpts.LangMode = llvm::StringSwitch<clang::Language>(A->getValue())
                          .Case("c", clang::Language::C)
                          .Case("c++", clang::Language::CXX)
                          .Case("objective-c", clang::Language::ObjC)
                          .Case("objective-c++", clang::Language::ObjCXX)
                          .Default(clang::Language::Unknown);

    if (FEOpts.LangMode == clang::Language::Unknown) {
      Diags->Report(clang::diag::err_drv_invalid_value)
          << A->getAsString(Args) << A->getValue();
      return false;
    }
  }
  for (auto *A : Args.filtered(OPT_ObjC, OPT_ObjCXX)) {
    if (A->getOption().matches(OPT_ObjC))
      FEOpts.LangMode = clang::Language::ObjC;
    else
      FEOpts.LangMode = clang::Language::ObjCXX;
  }

  return true;
}

Options::Options(DiagnosticsEngine &Diag, FileManager *FM,
                 InputArgList &ArgList)
    : Diags(&Diag), FM(FM) {
  if (!processDriverOptions(ArgList))
    return;

  if (!processLinkerOptions(ArgList))
    return;

  if (!processFrontendOptions(ArgList))
    return;

  /// Any unclaimed arguments should be handled by invoking the clang frontend.
  for (const Arg *A : ArgList) {
    if (A->isClaimed())
      continue;

    FrontendArgs.emplace_back(A->getSpelling());
    llvm::copy(A->getValues(), std::back_inserter(FrontendArgs));
  }
  FrontendArgs.push_back("-fsyntax-only");
}

InstallAPIContext Options::createContext() {
  InstallAPIContext Ctx;
  Ctx.FM = FM;
  Ctx.Diags = Diags;

  // InstallAPI requires two level namespacing.
  Ctx.BA.TwoLevelNamespace = true;

  Ctx.BA.InstallName = LinkerOpts.InstallName;
  Ctx.BA.CurrentVersion = LinkerOpts.CurrentVersion;
  Ctx.BA.CompatVersion = LinkerOpts.CompatVersion;
  Ctx.BA.AppExtensionSafe = LinkerOpts.AppExtensionSafe;
  Ctx.FT = DriverOpts.OutFT;
  Ctx.OutputLoc = DriverOpts.OutputPath;
  Ctx.LangMode = FEOpts.LangMode;

  // Process inputs.
  for (const std::string &ListPath : DriverOpts.FileLists) {
    auto Buffer = FM->getBufferForFile(ListPath);
    if (auto Err = Buffer.getError()) {
      Diags->Report(diag::err_cannot_open_file) << ListPath;
      return Ctx;
    }
    if (auto Err = FileListReader::loadHeaders(std::move(Buffer.get()),
                                               Ctx.InputHeaders)) {
      Diags->Report(diag::err_cannot_open_file) << ListPath;
      return Ctx;
    }
  }

  return Ctx;
}

} // namespace installapi
} // namespace clang
