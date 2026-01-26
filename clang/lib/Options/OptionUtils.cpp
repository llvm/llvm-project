//===--- OptionUtils.cpp - Utilities for command line arguments -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Options/OptionUtils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Options/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace llvm::opt;

namespace {
template <typename IntTy>
IntTy getLastArgIntValueImpl(const ArgList &Args, OptSpecifier Id,
                             IntTy Default, DiagnosticsEngine *Diags,
                             unsigned Base) {
  IntTy Res = Default;
  if (Arg *A = Args.getLastArg(Id)) {
    if (StringRef(A->getValue()).getAsInteger(Base, Res)) {
      if (Diags)
        Diags->Report(diag::err_drv_invalid_int_value)
            << A->getAsString(Args) << A->getValue();
    }
  }
  return Res;
}
} // namespace

int clang::getLastArgIntValue(const ArgList &Args, OptSpecifier Id, int Default,
                              DiagnosticsEngine *Diags, unsigned Base) {
  return getLastArgIntValueImpl<int>(Args, Id, Default, Diags, Base);
}

uint64_t clang::getLastArgUInt64Value(const ArgList &Args, OptSpecifier Id,
                                      uint64_t Default,
                                      DiagnosticsEngine *Diags, unsigned Base) {
  return getLastArgIntValueImpl<uint64_t>(Args, Id, Default, Diags, Base);
}

StringRef clang::parseMPreferVectorWidthOption(clang::DiagnosticsEngine &Diags,
                                               const llvm::opt::ArgList &Args) {
  const Arg *A = Args.getLastArg(options::OPT_mprefer_vector_width_EQ);
  if (!A)
    return "";

  StringRef Value = A->getValue();
  unsigned Width LLVM_ATTRIBUTE_UNINITIALIZED;

  // Only "none" and Integer values are accepted by
  // -mprefer-vector-width=<value>.
  if (Value != "none" && Value.getAsInteger(10, Width)) {
    Diags.Report(clang::diag::err_drv_invalid_value)
        << A->getOption().getName() << Value;
    return "";
  }

  return Value;
}

// This is a helper function for validating the optional refinement step
// parameter in reciprocal argument strings. Return false if there is an error
// parsing the refinement step. Otherwise, return true and set the Position
// of the refinement step in the input string.
static bool getRefinementStep(StringRef In, clang::DiagnosticsEngine &Diags,
                              const Arg &A, size_t &Position) {
  const char RefinementStepToken = ':';
  Position = In.find(RefinementStepToken);
  if (Position != StringRef::npos) {
    StringRef Option = A.getOption().getName();
    StringRef RefStep = In.substr(Position + 1);
    // Allow exactly one numeric character for the additional refinement
    // step parameter. This is reasonable for all currently-supported
    // operations and architectures because we would expect that a larger value
    // of refinement steps would cause the estimate "optimization" to
    // under-perform the native operation. Also, if the estimate does not
    // converge quickly, it probably will not ever converge, so further
    // refinement steps will not produce a better answer.
    if (RefStep.size() != 1) {
      Diags.Report(diag::err_drv_invalid_value) << Option << RefStep;
      return false;
    }
    char RefStepChar = RefStep[0];
    if (RefStepChar < '0' || RefStepChar > '9') {
      Diags.Report(diag::err_drv_invalid_value) << Option << RefStep;
      return false;
    }
  }
  return true;
}

StringRef clang::parseMRecipOption(clang::DiagnosticsEngine &Diags,
                                   const ArgList &Args) {
  StringRef DisabledPrefixIn = "!";
  StringRef DisabledPrefixOut = "!";
  StringRef EnabledPrefixOut = "";
  StringRef Out = "";

  const Arg *A = Args.getLastArg(options::OPT_mrecip, options::OPT_mrecip_EQ);
  if (!A)
    return "";

  const unsigned NumOptions = A->getNumValues();
  if (NumOptions == 0) {
    // No option is the same as "all".
    return "all";
  }

  // Pass through "all", "none", or "default" with an optional refinement step.
  if (NumOptions == 1) {
    StringRef Val = A->getValue(0);
    size_t RefStepLoc;
    if (!getRefinementStep(Val, Diags, *A, RefStepLoc))
      return "";
    StringRef ValBase = Val.slice(0, RefStepLoc);
    if (ValBase == "all" || ValBase == "none" || ValBase == "default") {
      return Val;
    }
  }

  // Each reciprocal type may be enabled or disabled individually.
  // Check each input value for validity, concatenate them all back together,
  // and pass through.

  llvm::StringMap<bool> OptionStrings;
  OptionStrings.insert(std::make_pair("divd", false));
  OptionStrings.insert(std::make_pair("divf", false));
  OptionStrings.insert(std::make_pair("divh", false));
  OptionStrings.insert(std::make_pair("vec-divd", false));
  OptionStrings.insert(std::make_pair("vec-divf", false));
  OptionStrings.insert(std::make_pair("vec-divh", false));
  OptionStrings.insert(std::make_pair("sqrtd", false));
  OptionStrings.insert(std::make_pair("sqrtf", false));
  OptionStrings.insert(std::make_pair("sqrth", false));
  OptionStrings.insert(std::make_pair("vec-sqrtd", false));
  OptionStrings.insert(std::make_pair("vec-sqrtf", false));
  OptionStrings.insert(std::make_pair("vec-sqrth", false));

  for (unsigned i = 0; i != NumOptions; ++i) {
    StringRef Val = A->getValue(i);

    bool IsDisabled = Val.starts_with(DisabledPrefixIn);
    // Ignore the disablement token for string matching.
    if (IsDisabled)
      Val = Val.substr(1);

    size_t RefStep;
    if (!getRefinementStep(Val, Diags, *A, RefStep))
      return "";

    StringRef ValBase = Val.slice(0, RefStep);
    llvm::StringMap<bool>::iterator OptionIter = OptionStrings.find(ValBase);
    if (OptionIter == OptionStrings.end()) {
      // Try again specifying float suffix.
      OptionIter = OptionStrings.find(ValBase.str() + 'f');
      if (OptionIter == OptionStrings.end()) {
        // The input name did not match any known option string.
        Diags.Report(diag::err_drv_unknown_argument) << Val;
        return "";
      }
      // The option was specified without a half or float or double suffix.
      // Make sure that the double or half entry was not already specified.
      // The float entry will be checked below.
      if (OptionStrings[ValBase.str() + 'd'] ||
          OptionStrings[ValBase.str() + 'h']) {
        Diags.Report(diag::err_drv_invalid_value)
            << A->getOption().getName() << Val;
        return "";
      }
    }

    if (OptionIter->second == true) {
      // Duplicate option specified.
      Diags.Report(diag::err_drv_invalid_value)
          << A->getOption().getName() << Val;
      return "";
    }

    // Mark the matched option as found. Do not allow duplicate specifiers.
    OptionIter->second = true;

    // If the precision was not specified, also mark the double and half entry
    // as found.
    if (ValBase.back() != 'f' && ValBase.back() != 'd' &&
        ValBase.back() != 'h') {
      OptionStrings[ValBase.str() + 'd'] = true;
      OptionStrings[ValBase.str() + 'h'] = true;
    }

    // Build the output string.
    StringRef Prefix = IsDisabled ? DisabledPrefixOut : EnabledPrefixOut;
    Out = Args.MakeArgString(Out + Prefix + Val);
    if (i != NumOptions - 1)
      Out = Args.MakeArgString(Out + ",");
  }

  return Out;
}

std::string clang::GetResourcesPath(StringRef BinaryPath) {
  // Since the resource directory is embedded in the module hash, it's important
  // that all places that need it call this function, so that they get the
  // exact same string ("a/../b/" and "b/" get different hashes, for example).

  // Dir is bin/ or lib/, depending on where BinaryPath is.
  StringRef Dir = llvm::sys::path::parent_path(BinaryPath);
  SmallString<128> P(Dir);

  StringRef ConfiguredResourceDir(CLANG_RESOURCE_DIR);
  if (!ConfiguredResourceDir.empty()) {
    // FIXME: We should fix the behavior of llvm::sys::path::append so we don't
    // need to check for absolute paths here.
    if (llvm::sys::path::is_absolute(ConfiguredResourceDir))
      P = ConfiguredResourceDir;
    else
      llvm::sys::path::append(P, ConfiguredResourceDir);
  } else {
    // On Windows, libclang.dll is in bin/.
    // On non-Windows, libclang.so/.dylib is in lib/.
    // With a static-library build of libclang, LibClangPath will contain the
    // path of the embedding binary, which for LLVM binaries will be in bin/.
    // ../lib gets us to lib/ in both cases.
    P = llvm::sys::path::parent_path(Dir);
    // This search path is also created in the COFF driver of lld, so any
    // changes here also needs to happen in lld/COFF/Driver.cpp
    llvm::sys::path::append(P, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
                            CLANG_VERSION_MAJOR_STRING);
  }

  return std::string(P);
}

std::string clang::GetResourcesPath(const char *Argv0, void *MainAddr) {
  const std::string ClangExecutable =
      llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
  return GetResourcesPath(ClangExecutable);
}
