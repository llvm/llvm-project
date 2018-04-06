//===--- XRayArgs.cpp - Arguments for XRay --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "clang/Driver/XRayArgs.h"
#include "ToolChains/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SpecialCaseList.h"

using namespace clang;
using namespace clang::driver;
using namespace llvm::opt;

namespace {
constexpr char XRayInstrumentOption[] = "-fxray-instrument";
constexpr char XRayInstructionThresholdOption[] =
    "-fxray-instruction-threshold=";
} // namespace

XRayArgs::XRayArgs(const ToolChain &TC, const ArgList &Args) {
  const Driver &D = TC.getDriver();
  const llvm::Triple &Triple = TC.getTriple();
  if (Args.hasFlag(options::OPT_fxray_instrument,
                   options::OPT_fnoxray_instrument, false)) {
    if (Triple.getOS() == llvm::Triple::Linux) {
      switch (Triple.getArch()) {
      case llvm::Triple::x86_64:
      case llvm::Triple::arm:
      case llvm::Triple::aarch64:
      case llvm::Triple::ppc64le:
      case llvm::Triple::mips:
      case llvm::Triple::mipsel:
      case llvm::Triple::mips64:
      case llvm::Triple::mips64el:
        break;
      default:
        D.Diag(diag::err_drv_clang_unsupported)
            << (std::string(XRayInstrumentOption) + " on " + Triple.str());
      }
    } else if (Triple.getOS() == llvm::Triple::FreeBSD ||
               Triple.getOS() == llvm::Triple::OpenBSD) {
        if (Triple.getArch() != llvm::Triple::x86_64) {
          D.Diag(diag::err_drv_clang_unsupported)
              << (std::string(XRayInstrumentOption) + " on " + Triple.str());
        }
    } else {
      D.Diag(diag::err_drv_clang_unsupported)
          << (std::string(XRayInstrumentOption) + " on non-supported target OS");
    }
    XRayInstrument = true;
    if (const Arg *A =
            Args.getLastArg(options::OPT_fxray_instruction_threshold_,
                            options::OPT_fxray_instruction_threshold_EQ)) {
      StringRef S = A->getValue();
      if (S.getAsInteger(0, InstructionThreshold) || InstructionThreshold < 0)
        D.Diag(clang::diag::err_drv_invalid_value) << A->getAsString(Args) << S;
    }

    // By default, the back-end will not emit the lowering for XRay customevent
    // calls if the function is not instrumented. In the future we will change
    // this default to be the reverse, but in the meantime we're going to
    // introduce the new functionality behind a flag.
    if (Args.hasFlag(options::OPT_fxray_always_emit_customevents,
                     options::OPT_fnoxray_always_emit_customevents, false))
      XRayAlwaysEmitCustomEvents = true;

    if (!Args.hasFlag(options::OPT_fxray_link_deps,
                      options::OPT_fnoxray_link_deps, true))
      XRayRT = false;

    // Validate the always/never attribute files. We also make sure that they
    // are treated as actual dependencies.
    for (const auto &Filename :
         Args.getAllArgValues(options::OPT_fxray_always_instrument)) {
      if (llvm::sys::fs::exists(Filename)) {
        AlwaysInstrumentFiles.push_back(Filename);
        ExtraDeps.push_back(Filename);
      } else
        D.Diag(clang::diag::err_drv_no_such_file) << Filename;
    }

    for (const auto &Filename :
         Args.getAllArgValues(options::OPT_fxray_never_instrument)) {
      if (llvm::sys::fs::exists(Filename)) {
        NeverInstrumentFiles.push_back(Filename);
        ExtraDeps.push_back(Filename);
      } else
        D.Diag(clang::diag::err_drv_no_such_file) << Filename;
    }
  }
}

void XRayArgs::addArgs(const ToolChain &TC, const ArgList &Args,
                       ArgStringList &CmdArgs, types::ID InputType) const {
  if (!XRayInstrument)
    return;

  CmdArgs.push_back(XRayInstrumentOption);

  if (XRayAlwaysEmitCustomEvents)
    CmdArgs.push_back("-fxray-always-emit-customevents");

  CmdArgs.push_back(Args.MakeArgString(Twine(XRayInstructionThresholdOption) +
                                       Twine(InstructionThreshold)));

  for (const auto &Always : AlwaysInstrumentFiles) {
    SmallString<64> AlwaysInstrumentOpt("-fxray-always-instrument=");
    AlwaysInstrumentOpt += Always;
    CmdArgs.push_back(Args.MakeArgString(AlwaysInstrumentOpt));
  }

  for (const auto &Never : NeverInstrumentFiles) {
    SmallString<64> NeverInstrumentOpt("-fxray-never-instrument=");
    NeverInstrumentOpt += Never;
    CmdArgs.push_back(Args.MakeArgString(NeverInstrumentOpt));
  }

  for (const auto &Dep : ExtraDeps) {
    SmallString<64> ExtraDepOpt("-fdepfile-entry=");
    ExtraDepOpt += Dep;
    CmdArgs.push_back(Args.MakeArgString(ExtraDepOpt));
  }
}
