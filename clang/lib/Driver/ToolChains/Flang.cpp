//===-- Flang.cpp - Flang+LLVM ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "Flang.h"
#include "CommonArgs.h"

#include "clang/Driver/Options.h"

#include <cassert>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

/// Add -x lang to \p CmdArgs for \p Input.
static void addDashXForInput(const ArgList &Args, const InputInfo &Input,
                             ArgStringList &CmdArgs) {
  CmdArgs.push_back("-x");
  // Map the driver type to the frontend type.
  CmdArgs.push_back(types::getTypeName(Input.getType()));
}

void Flang::addFortranDialectOptions(const ArgList &Args,
                                     ArgStringList &CmdArgs) const {
  Args.AddAllArgs(
      CmdArgs, {options::OPT_ffixed_form, options::OPT_ffree_form,
                options::OPT_ffixed_line_length_EQ, options::OPT_fopenmp,
                options::OPT_fopenacc, options::OPT_finput_charset_EQ,
                options::OPT_fimplicit_none, options::OPT_fno_implicit_none,
                options::OPT_fbackslash, options::OPT_fno_backslash,
                options::OPT_flogical_abbreviations,
                options::OPT_fno_logical_abbreviations,
                options::OPT_fxor_operator, options::OPT_fno_xor_operator,
                options::OPT_falternative_parameter_statement,
                options::OPT_fdefault_real_8, options::OPT_fdefault_integer_8,
                options::OPT_fdefault_double_8, options::OPT_flarge_sizes,
                options::OPT_fno_automatic});
}

void Flang::addPreprocessingOptions(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_P, options::OPT_D, options::OPT_U,
                   options::OPT_I, options::OPT_cpp, options::OPT_nocpp});
}

void Flang::addOtherOptions(const ArgList &Args, ArgStringList &CmdArgs) const {
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_module_dir, options::OPT_fdebug_module_writer,
                   options::OPT_fintrinsic_modules_path, options::OPT_pedantic,
                   options::OPT_std_EQ, options::OPT_W_Joined,
                   options::OPT_fconvert_EQ, options::OPT_fpass_plugin_EQ,
                   options::OPT_funderscoring, options::OPT_fno_underscoring});

  Arg *stackArrays =
      Args.getLastArg(options::OPT_Ofast, options::OPT_fstack_arrays,
                      options::OPT_fno_stack_arrays);
  if (stackArrays &&
      !stackArrays->getOption().matches(options::OPT_fno_stack_arrays))
    CmdArgs.push_back("-fstack-arrays");
}

void Flang::addPicOptions(const ArgList &Args, ArgStringList &CmdArgs) const {
  // ParsePICArgs parses -fPIC/-fPIE and their variants and returns a tuple of
  // (RelocationModel, PICLevel, IsPIE).
  llvm::Reloc::Model RelocationModel;
  unsigned PICLevel;
  bool IsPIE;
  std::tie(RelocationModel, PICLevel, IsPIE) =
      ParsePICArgs(getToolChain(), Args);

  if (auto *RMName = RelocationModelName(RelocationModel)) {
    CmdArgs.push_back("-mrelocation-model");
    CmdArgs.push_back(RMName);
  }
  if (PICLevel > 0) {
    CmdArgs.push_back("-pic-level");
    CmdArgs.push_back(PICLevel == 1 ? "1" : "2");
    if (IsPIE)
      CmdArgs.push_back("-pic-is-pie");
  }
}

void Flang::addTargetOptions(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  const ToolChain &TC = getToolChain();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  const Driver &D = TC.getDriver();

  std::string CPU = getCPUName(D, Args, Triple);
  if (!CPU.empty()) {
    CmdArgs.push_back("-target-cpu");
    CmdArgs.push_back(Args.MakeArgString(CPU));
  }

  // Add the target features.
  switch (TC.getArch()) {
  default:
    break;
  case llvm::Triple::aarch64:
    [[fallthrough]];
  case llvm::Triple::x86_64:
    getTargetFeatures(D, Triple, Args, CmdArgs, /*ForAs*/ false);
    break;
  }

  // TODO: Add target specific flags, ABI, mtune option etc.
}

static void addFloatingPointOptions(const Driver &D, const ArgList &Args,
                                    ArgStringList &CmdArgs) {
  StringRef FPContract;
  bool HonorINFs = true;
  bool HonorNaNs = true;
  bool ApproxFunc = false;
  bool SignedZeros = true;
  bool AssociativeMath = false;
  bool ReciprocalMath = false;

  if (const Arg *A = Args.getLastArg(options::OPT_ffp_contract)) {
    const StringRef Val = A->getValue();
    if (Val == "fast" || Val == "off") {
      FPContract = Val;
    } else if (Val == "on") {
      // Warn instead of error because users might have makefiles written for
      // gfortran (which accepts -ffp-contract=on)
      D.Diag(diag::warn_drv_unsupported_option_for_flang)
          << Val << A->getOption().getName() << "off";
      FPContract = "off";
    } else
      // Clang's "fast-honor-pragmas" option is not supported because it is
      // non-standard
      D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getSpelling() << Val;
  }

  for (const Arg *A : Args) {
    auto optId = A->getOption().getID();
    switch (optId) {
    // if this isn't an FP option, skip the claim below
    default:
      continue;

    case options::OPT_fhonor_infinities:
      HonorINFs = true;
      break;
    case options::OPT_fno_honor_infinities:
      HonorINFs = false;
      break;
    case options::OPT_fhonor_nans:
      HonorNaNs = true;
      break;
    case options::OPT_fno_honor_nans:
      HonorNaNs = false;
      break;
    case options::OPT_fapprox_func:
      ApproxFunc = true;
      break;
    case options::OPT_fno_approx_func:
      ApproxFunc = false;
      break;
    case options::OPT_fsigned_zeros:
      SignedZeros = true;
      break;
    case options::OPT_fno_signed_zeros:
      SignedZeros = false;
      break;
    case options::OPT_fassociative_math:
      AssociativeMath = true;
      break;
    case options::OPT_fno_associative_math:
      AssociativeMath = false;
      break;
    case options::OPT_freciprocal_math:
      ReciprocalMath = true;
      break;
    case options::OPT_fno_reciprocal_math:
      ReciprocalMath = false;
      break;
    case options::OPT_Ofast:
      [[fallthrough]];
    case options::OPT_ffast_math:
      HonorINFs = false;
      HonorNaNs = false;
      AssociativeMath = true;
      ReciprocalMath = true;
      ApproxFunc = true;
      SignedZeros = false;
      FPContract = "fast";
      break;
    case options::OPT_fno_fast_math:
      HonorINFs = true;
      HonorNaNs = true;
      AssociativeMath = false;
      ReciprocalMath = false;
      ApproxFunc = false;
      SignedZeros = true;
      // -fno-fast-math should undo -ffast-math so I return FPContract to the
      // default. It is important to check it is "fast" (the default) so that
      // --ffp-contract=off -fno-fast-math --> -ffp-contract=off
      if (FPContract == "fast")
        FPContract = "";
      break;
    }

    // If we handled this option claim it
    A->claim();
  }

  if (!HonorINFs && !HonorNaNs && AssociativeMath && ReciprocalMath &&
      ApproxFunc && !SignedZeros &&
      (FPContract == "fast" || FPContract == "")) {
    CmdArgs.push_back("-ffast-math");
    return;
  }

  if (!FPContract.empty())
    CmdArgs.push_back(Args.MakeArgString("-ffp-contract=" + FPContract));

  if (!HonorINFs)
    CmdArgs.push_back("-menable-no-infs");

  if (!HonorNaNs)
    CmdArgs.push_back("-menable-no-nans");

  if (ApproxFunc)
    CmdArgs.push_back("-fapprox-func");

  if (!SignedZeros)
    CmdArgs.push_back("-fno-signed-zeros");

  if (AssociativeMath && !SignedZeros)
    CmdArgs.push_back("-mreassociate");

  if (ReciprocalMath)
    CmdArgs.push_back("-freciprocal-math");
}

void Flang::ConstructJob(Compilation &C, const JobAction &JA,
                         const InputInfo &Output, const InputInfoList &Inputs,
                         const ArgList &Args, const char *LinkingOutput) const {
  const auto &TC = getToolChain();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  const std::string &TripleStr = Triple.getTriple();

  const Driver &D = TC.getDriver();
  ArgStringList CmdArgs;
  DiagnosticsEngine &Diags = D.getDiags();

  // Invoke ourselves in -fc1 mode.
  CmdArgs.push_back("-fc1");

  // Add the "effective" target triple.
  CmdArgs.push_back("-triple");
  CmdArgs.push_back(Args.MakeArgString(TripleStr));

  if (isa<PreprocessJobAction>(JA)) {
      CmdArgs.push_back("-E");
  } else if (isa<CompileJobAction>(JA) || isa<BackendJobAction>(JA)) {
    if (JA.getType() == types::TY_Nothing) {
      CmdArgs.push_back("-fsyntax-only");
    } else if (JA.getType() == types::TY_AST) {
      CmdArgs.push_back("-emit-ast");
    } else if (JA.getType() == types::TY_LLVM_IR ||
               JA.getType() == types::TY_LTO_IR) {
      CmdArgs.push_back("-emit-llvm");
    } else if (JA.getType() == types::TY_LLVM_BC ||
               JA.getType() == types::TY_LTO_BC) {
      CmdArgs.push_back("-emit-llvm-bc");
    } else if (JA.getType() == types::TY_PP_Asm) {
      CmdArgs.push_back("-S");
    } else {
      assert(false && "Unexpected output type!");
    }
  } else if (isa<AssembleJobAction>(JA)) {
    CmdArgs.push_back("-emit-obj");
  } else {
    assert(false && "Unexpected action class for Flang tool.");
  }

  const InputInfo &Input = Inputs[0];
  types::ID InputType = Input.getType();

  // Add preprocessing options like -I, -D, etc. if we are using the
  // preprocessor (i.e. skip when dealing with e.g. binary files).
  if (types::getPreprocessedType(InputType) != types::TY_INVALID)
    addPreprocessingOptions(Args, CmdArgs);

  addFortranDialectOptions(Args, CmdArgs);

  // Color diagnostics are parsed by the driver directly from argv and later
  // re-parsed to construct this job; claim any possible color diagnostic here
  // to avoid warn_drv_unused_argument.
  Args.getLastArg(options::OPT_fcolor_diagnostics,
                  options::OPT_fno_color_diagnostics);
  if (Diags.getDiagnosticOptions().ShowColors)
    CmdArgs.push_back("-fcolor-diagnostics");

  // LTO mode is parsed by the Clang driver library.
  LTOKind LTOMode = D.getLTOMode(/* IsOffload */ false);
  assert(LTOMode != LTOK_Unknown && "Unknown LTO mode.");
  if (LTOMode == LTOK_Full)
    CmdArgs.push_back("-flto=full");
  else if (LTOMode == LTOK_Thin) {
    Diags.Report(
        Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                              "the option '-flto=thin' is a work in progress"));
    CmdArgs.push_back("-flto=thin");
  }

  // -fPIC and related options.
  addPicOptions(Args, CmdArgs);

  // Floating point related options
  addFloatingPointOptions(D, Args, CmdArgs);

  // Add target args, features, etc.
  addTargetOptions(Args, CmdArgs);

  // Add other compile options
  addOtherOptions(Args, CmdArgs);

  // Forward -Xflang arguments to -fc1
  Args.AddAllArgValues(CmdArgs, options::OPT_Xflang);

  // Forward -mllvm options to the LLVM option parser. In practice, this means
  // forwarding to `-fc1` as that's where the LLVM parser is run.
  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    A->claim();
    A->render(Args, CmdArgs);
  }

  for (const Arg *A : Args.filtered(options::OPT_mmlir)) {
    A->claim();
    A->render(Args, CmdArgs);
  }

  // Remove any unsupported gfortran diagnostic options
  for (const Arg *A : Args.filtered(options::OPT_flang_ignored_w_Group)) {
    A->claim();
    D.Diag(diag::warn_drv_unsupported_diag_option_for_flang)
        << A->getOption().getName();
  }

  // Optimization level for CodeGen.
  if (const Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O4)) {
      CmdArgs.push_back("-O3");
      D.Diag(diag::warn_O4_is_O3);
    } else if (A->getOption().matches(options::OPT_Ofast)) {
      CmdArgs.push_back("-O3");
    } else {
      A->render(Args, CmdArgs);
    }
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  assert(Input.isFilename() && "Invalid input.");

  addDashXForInput(Args, Input, CmdArgs);

  CmdArgs.push_back(Input.getFilename());

  // TODO: Replace flang-new with flang once the new driver replaces the
  // throwaway driver
  const char *Exec = Args.MakeArgString(D.GetProgramPath("flang-new", TC));
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileUTF8(),
                                         Exec, CmdArgs, Inputs, Output));
}

Flang::Flang(const ToolChain &TC) : Tool("flang-new", "flang frontend", TC) {}

Flang::~Flang() {}
