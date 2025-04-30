//===-- ClassicFlang.cpp - Flang+LLVM ToolChain Implementations -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClassicFlang.h"
#include "CommonArgs.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/ObjCRuntime.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/XRayArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/TargetParser/TargetParser.h"

#ifdef LLVM_ON_UNIX
#include <unistd.h> // For getuid().
#endif

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

class ClassicFlangMacroBuilder : public MacroBuilder {
  ArgStringList &CmdArgs;
  const ArgList &DriverArgs;

public:
  ClassicFlangMacroBuilder(ArgStringList &UpperCmdArgs,
                           const ArgList &DriverArgs,
                           llvm::raw_string_ostream &Output)
      : MacroBuilder(Output), CmdArgs(UpperCmdArgs), DriverArgs(DriverArgs) {}
  virtual void defineMacro(const Twine &Name,
                           const Twine &Value = "1") override {
    CmdArgs.push_back("-def");
    CmdArgs.push_back(DriverArgs.MakeArgString(Name + Twine('=') + Value));
  }
};

void ClassicFlang::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();
  ArgStringList CommonCmdArgs;
  ArgStringList UpperCmdArgs;
  ArgStringList LowerCmdArgs;
  SmallString<256> Stem;
  std::string OutFile;
  bool NeedIEEE = true;
  bool NeedFastMath = false;
  bool NeedRelaxedMath = false;
  bool AssociativeMath = false;
  bool SignedZeros = true;

  // Check number of inputs for sanity. We need at least one input.
  assert(Inputs.size() >= 1 && "Must have at least one input.");

  /***** Process file arguments to both parts *****/
  const InputInfo &Input = Inputs[0];
  types::ID InputType = Input.getType();
  // Check file type sanity
  assert(types::isAcceptedByFlang(InputType) && "Can only accept Fortran");

  if (Args.hasArg(options::OPT_fsyntax_only, options::OPT_E)) {
    // For -fsyntax-only and -E produce temp files only
    Stem = C.getDriver().GetTemporaryPath("", "");
  } else {
    OutFile = Output.getFilename();
    Stem = llvm::sys::path::filename(OutFile);
    llvm::sys::path::replace_extension(Stem, "");
  }

#ifdef ENABLE_CLASSIC_FLANG
  if (Args.hasArg(options::OPT_emit_flang_llvm)) {
    // -emit-flang-llvm only supports asm output so claim -S to prevent warning
    Args.ClaimAllArgs(options::OPT_S);
  }
#endif

  // Add input file name to the compilation line
  UpperCmdArgs.push_back(Input.getBaseInput());

  // Add temporary output for ILM
  const char *ILMFile = Args.MakeArgString(Stem + ".ilm");
  LowerCmdArgs.push_back(ILMFile);
  C.addTempFile(ILMFile);

  // Generate -cmdline
  std::string CmdLine("'+flang");
  // ignore the first argument which reads "--driver-mode=fortran"
  for (unsigned i = 1; i < Args.getNumInputArgStrings(); ++i) {
    CmdLine.append(" ");
    CmdLine.append(Args.getArgString(i));
  }
  CmdLine.append("'");

  CommonCmdArgs.push_back("-cmdline");
  CommonCmdArgs.push_back(Args.MakeArgString(CmdLine));

  /***** Process common args *****/

  // Add "inform level" flag
  if (Args.hasArg(options::OPT_Minform_EQ)) {
    // Parse arguments to set its value
    for (Arg *A : Args.filtered(options::OPT_Minform_EQ)) {
      A->claim();
      CommonCmdArgs.push_back("-inform");
      CommonCmdArgs.push_back(A->getValue(0));
    }
  } else {
    // Default value
    CommonCmdArgs.push_back("-inform");
    CommonCmdArgs.push_back("warn");
  }

  for (auto Arg : Args.filtered(options::OPT_Msave_on)) {
    Arg->claim();
    CommonCmdArgs.push_back("-save");
  }

  for (auto Arg : Args.filtered(options::OPT_Msave_off)) {
    Arg->claim();
    CommonCmdArgs.push_back("-nosave");
  }

  // Treat denormalized numbers as zero: On
  for (auto Arg : Args.filtered(options::OPT_Mdaz_on)) {
    Arg->claim();
    CommonCmdArgs.push_back("-x");
    CommonCmdArgs.push_back("129");
    CommonCmdArgs.push_back("4");
    CommonCmdArgs.push_back("-y");
    CommonCmdArgs.push_back("129");
    CommonCmdArgs.push_back("0x400");
  }

  // Treat denormalized numbers as zero: Off
  for (auto Arg : Args.filtered(options::OPT_Mdaz_off)) {
    Arg->claim();
    CommonCmdArgs.push_back("-y");
    CommonCmdArgs.push_back("129");
    CommonCmdArgs.push_back("4");
    CommonCmdArgs.push_back("-x");
    CommonCmdArgs.push_back("129");
    CommonCmdArgs.push_back("0x400");
  }

  // Store constants in writable data segment
  for (auto Arg : Args.filtered(options::OPT_Mwritable_constants)) {
    Arg->claim();
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("183");
    LowerCmdArgs.push_back("0x20000000");
  }

  // Bounds checking: On
  for (auto Arg : Args.filtered(options::OPT_Mbounds_on)) {
    Arg->claim();
    CommonCmdArgs.push_back("-x");
    CommonCmdArgs.push_back("70");
    CommonCmdArgs.push_back("2");
  }

  // Bounds checking: Off
  for (auto Arg : Args.filtered(options::OPT_Mbounds_off)) {
    Arg->claim();
    CommonCmdArgs.push_back("-y");
    CommonCmdArgs.push_back("70");
    CommonCmdArgs.push_back("2");
  }

  // Generate code allowing recursive subprograms
  for (auto Arg : Args.filtered(options::OPT_Mrecursive_on)) {
    Arg->claim();
    CommonCmdArgs.push_back("-recursive");
  }

  // Disable recursive subprograms
  for (auto Arg : Args.filtered(options::OPT_Mrecursive_off)) {
    Arg->claim();
    CommonCmdArgs.push_back("-norecursive");
  }

  // Enable generating reentrant code (disable optimizations that inhibit it)
  for (auto Arg : Args.filtered(options::OPT_Mreentrant_on)) {
    Arg->claim();
    CommonCmdArgs.push_back("-reentrant");
  }

  // Allow optimizations inhibiting reentrancy
  for (auto Arg : Args.filtered(options::OPT_Mreentrant_off)) {
    Arg->claim();
    CommonCmdArgs.push_back("-noreentrant");
  }

  // Swap byte order for unformatted IO
  for (auto Arg :
       Args.filtered(options::OPT_Mbyteswapio, options::OPT_byteswapio)) {
    Arg->claim();
    CommonCmdArgs.push_back("-x");
    CommonCmdArgs.push_back("125");
    CommonCmdArgs.push_back("2");
  }

  // Contiguous pointer checks
  if (Arg *A = Args.getLastArg(options::OPT_fsanitize_EQ)) {
    for (StringRef val : A->getValues()) {
      if (val == "discontiguous" || val == "undefined") {
        // -x 54 0x40 -x 54 0x80 -x 54 0x200
        UpperCmdArgs.push_back("-x");
        UpperCmdArgs.push_back("54");
        UpperCmdArgs.push_back("0x2c0");

        // -fsanitze=discontiguous has no meaning in LLVM, only flang driver
        // needs to recognize it. However -fsanitize=undefined needs to be
        // passed on for further processing by the non-flang part of the driver.
        if (val == "discontiguous")
          A->claim();
        break;
      }
    }
  }

  // Treat backslashes as regular characters
  for (auto Arg :
       Args.filtered(options::OPT_fno_backslash, options::OPT_Mbackslash)) {
    Arg->claim();
    CommonCmdArgs.push_back("-x");
    CommonCmdArgs.push_back("124");
    CommonCmdArgs.push_back("0x40");
  }

  // Treat backslashes as C-style escape characters
  for (auto Arg :
       Args.filtered(options::OPT_fbackslash, options::OPT_Mnobackslash)) {
    Arg->claim();
    CommonCmdArgs.push_back("-y");
    CommonCmdArgs.push_back("124");
    CommonCmdArgs.push_back("0x40");
  }

  // handle OpemMP options
  if (auto *A =
          Args.getLastArg(options::OPT_mp, options::OPT_nomp,
                          options::OPT_fopenmp, options::OPT_fno_openmp)) {
    for (auto Arg : Args.filtered(options::OPT_mp, options::OPT_nomp)) {
      Arg->claim();
    }
    for (auto Arg :
         Args.filtered(options::OPT_fopenmp, options::OPT_fno_openmp)) {
      Arg->claim();
    }

    if (A->getOption().matches(options::OPT_mp) ||
        A->getOption().matches(options::OPT_fopenmp)) {

      CommonCmdArgs.push_back("-mp");

      // Allocate threadprivate data local to the thread
      CommonCmdArgs.push_back("-x");
      CommonCmdArgs.push_back("69");
      CommonCmdArgs.push_back("0x200");

      // Use the 'fair' schedule as the default static schedule
      // for parallel do loops
      CommonCmdArgs.push_back("-x");
      CommonCmdArgs.push_back("69");
      CommonCmdArgs.push_back("0x400");

      // Disable use of native atomic instructions
      // for OpenMP atomics pending either a named
      // option or a libatomic bundled with flang.
      UpperCmdArgs.push_back("-x");
      UpperCmdArgs.push_back("69");
      UpperCmdArgs.push_back("0x1000");
    }
  }

  // Align large objects on cache lines
  for (auto Arg : Args.filtered(options::OPT_Mcache_align_on)) {
    Arg->claim();
    CommonCmdArgs.push_back("-x");
    CommonCmdArgs.push_back("119");
    CommonCmdArgs.push_back("0x10000000");
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("129");
    LowerCmdArgs.push_back("0x40000000");
  }

  // Disable special alignment of large objects
  for (auto Arg : Args.filtered(options::OPT_Mcache_align_off)) {
    Arg->claim();
    CommonCmdArgs.push_back("-y");
    CommonCmdArgs.push_back("119");
    CommonCmdArgs.push_back("0x10000000");
    LowerCmdArgs.push_back("-y");
    LowerCmdArgs.push_back("129");
    LowerCmdArgs.push_back("0x40000000");
  }

  // -Mstack_arrays
  for (auto Arg : Args.filtered(options::OPT_Mstackarrays)) {
    Arg->claim();
    CommonCmdArgs.push_back("-x");
    CommonCmdArgs.push_back("54");
    CommonCmdArgs.push_back("8");
  }

  // -Memit-dwarf-common-blocks-name, only add xbit to flang2.
  for (auto Arg : Args.filtered(options::OPT_Memit_dwarf_common_blocks_name)) {
    Arg->claim();
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("183");
    LowerCmdArgs.push_back("0x40000000");
  }

  // -Munixlogical, only add xbit to flang2.
  for (auto Arg : Args.filtered(options::OPT_Munixlogical)) {
    Arg->claim();
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("125");
    LowerCmdArgs.push_back("0x8");
  }

  // Last argument of -g/-gdwarfX should be taken.
  Arg *GArg = Args.getLastArg(options::OPT_g_Flag);
  Arg *GDwarfArg =
      Args.getLastArg(options::OPT_gdwarf_2, options::OPT_gdwarf_3,
                      options::OPT_gdwarf_4, options::OPT_gdwarf_5);

  if (GArg || GDwarfArg) {

    for (auto Arg : Args.filtered(options::OPT_g_Flag, options::OPT_gdwarf_2,
                                  options::OPT_gdwarf_3, options::OPT_gdwarf_4,
                                  options::OPT_gdwarf_5)) {
      Arg->claim();
    }

    CommonCmdArgs.push_back("-x");
    CommonCmdArgs.push_back("120");

    if (!GDwarfArg) // -g without -gdwarf-X produces default (DWARFv4)
      CommonCmdArgs.push_back("0x1000000");
    else if (GDwarfArg->getOption().matches(options::OPT_gdwarf_2)) // -gdwarf-2
      CommonCmdArgs.push_back("0x200");
    else if (GDwarfArg->getOption().matches(options::OPT_gdwarf_3)) // -gdwarf-3
      CommonCmdArgs.push_back("0x4000");
    else if (GDwarfArg->getOption().matches(options::OPT_gdwarf_4)) // -gdwarf-4
      CommonCmdArgs.push_back("0x1000000");
    else if (GDwarfArg->getOption().matches(options::OPT_gdwarf_5)) // -gdwarf-5
      CommonCmdArgs.push_back("0x2000000");
    // Handle `-gpubnames` option separately.
    for (auto Arg : Args.filtered(options::OPT_gpubnames)) {
      Arg->claim();
      CommonCmdArgs.push_back("-x");
      CommonCmdArgs.push_back("120");
      CommonCmdArgs.push_back("0x40000000"); // -gpubnames
    }
  }

  // -Mipa has no effect
  if (Arg *A = Args.getLastArg(options::OPT_Mipa)) {
    D.Diag(diag::warn_drv_clang_unsupported) << A->getAsString(Args);
  }

  // -Minline has no effect
  if (Arg *A = Args.getLastArg(options::OPT_Minline_on)) {
    D.Diag(diag::warn_drv_clang_unsupported) << A->getAsString(Args);
  }

  // Handle -fdefault-real-8 (and its alias, -r8) and -fno-default-real-8
  if (Arg *A = Args.getLastArg(options::OPT_r8, options::OPT_fdefault_real_8,
                               options::OPT_fno_default_real_8)) {
    const char *fl;
    // For -f version add -x flag, for -fno add -y
    if (A->getOption().matches(options::OPT_fno_default_real_8)) {
      fl = "-y";
    } else {
      fl = "-x";
    }

    for (Arg *A : Args.filtered(options::OPT_r8, options::OPT_fdefault_real_8,
                                options::OPT_fno_default_real_8)) {
      A->claim();
    }

    UpperCmdArgs.push_back(fl);
    UpperCmdArgs.push_back("124");
    UpperCmdArgs.push_back("0x8");
    UpperCmdArgs.push_back(fl);
    UpperCmdArgs.push_back("124");
    UpperCmdArgs.push_back("0x80000");
  }

  // Process and claim -i8/-fdefault-integer-8/-fno-default-integer-8 argument
  if (Arg *A = Args.getLastArg(options::OPT_i8, options::OPT_fdefault_integer_8,
                               options::OPT_fno_default_integer_8)) {
    const char *fl;

    if (A->getOption().matches(options::OPT_fno_default_integer_8)) {
      fl = "-y";
    } else {
      fl = "-x";
    }

    for (Arg *A :
         Args.filtered(options::OPT_i8, options::OPT_fdefault_integer_8,
                       options::OPT_fno_default_integer_8)) {
      A->claim();
    }

    UpperCmdArgs.push_back(fl);
    UpperCmdArgs.push_back("124");
    UpperCmdArgs.push_back("0x10");
  }

  // Pass an arbitrary flag for first part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Wh_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    SmallVector<StringRef, 8> PassArgs;
    Value.split(PassArgs, StringRef(","));
    for (StringRef PassArg : PassArgs) {
      UpperCmdArgs.push_back(Args.MakeArgString(PassArg));
    }
  }

  // Flush to zero mode
  // Disabled by default, but can be enabled by a switch
  if (Args.hasArg(options::OPT_Mflushz_on)) {
    // For -Mflushz set -x 129 2 for second part of Fortran frontend
    for (Arg *A : Args.filtered(options::OPT_Mflushz_on)) {
      A->claim();
      LowerCmdArgs.push_back("-x");
      LowerCmdArgs.push_back("129");
      LowerCmdArgs.push_back("2");
    }
  } else {
    LowerCmdArgs.push_back("-y");
    LowerCmdArgs.push_back("129");
    LowerCmdArgs.push_back("2");
    for (Arg *A : Args.filtered(options::OPT_Mflushz_off)) {
      A->claim();
    }
  }

  // For -fPIC set -x 62 8 for second part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_fPIC)) {
    A->claim();
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("62");
    LowerCmdArgs.push_back("8");
  }

  StringRef OptOStr("0");
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O4)) {
      OptOStr = "4"; // FIXME what should this be?
    } else if (A->getOption().matches(options::OPT_Ofast)) {
      OptOStr = "2"; // FIXME what should this be?
    } else if (A->getOption().matches(options::OPT_O0)) {
      // intentionally do nothing
    } else {
      assert(A->getOption().matches(options::OPT_O) && "Must have a -O flag");
      StringRef S(A->getValue());
      if ((S == "s") || (S == "z")) {
        // -Os = size; -Oz = more size
        OptOStr = "2"; // FIXME -Os|-Oz => -opt ?
      } else if ((S == "1") || (S == "2") || (S == "3")) {
        OptOStr = S;
      } else {
        OptOStr = "4";
      }
    }
  }
  unsigned OptLevel = std::stoi(OptOStr.str());

  if (Args.hasArg(options::OPT_g_Group)) {
    // pass -g to lower and upper
    CommonCmdArgs.push_back("-debug");
  }

  /* Pick the last among conflicting flags, if a positive and negative flag
     exists for ex. "-ffast-math -fno-fast-math" they get nullified. Also any
     previously overwritten flag remains that way.
     For ex. "-Kieee -ffast-math -fno-fast-math". -Kieee gets overwritten by
     -ffast-math which then gets negated by -fno-fast-math, finally behaving as
     if none of those flags were passed.
  */
  for (Arg *A : Args.filtered(
           options::OPT_ffast_math, options::OPT_fno_fast_math,
           options::OPT_Ofast, options::OPT_Kieee_off, options::OPT_Kieee_on,
           options::OPT_frelaxed_math, options::OPT_fassociative_math,
           options::OPT_fno_associative_math, options::OPT_fsigned_zeros,
           options::OPT_fno_signed_zeros)) {
    if (A->getOption().matches(options::OPT_ffast_math) ||
        A->getOption().matches(options::OPT_Ofast)) {
      NeedIEEE = NeedRelaxedMath = false;
      NeedFastMath = true;
    } else if (A->getOption().matches(options::OPT_Kieee_on)) {
      NeedFastMath = NeedRelaxedMath = AssociativeMath = false;
      NeedIEEE = SignedZeros = true;
    } else if (A->getOption().matches(options::OPT_frelaxed_math)) {
      NeedFastMath = NeedIEEE = false;
      NeedRelaxedMath = true;
    } else if (A->getOption().matches(options::OPT_fno_fast_math)) {
      NeedFastMath = false;
    } else if (A->getOption().matches(options::OPT_Kieee_off)) {
      NeedIEEE = false;
    } else if (A->getOption().matches(options::OPT_fassociative_math)) {
      AssociativeMath = true;
      NeedIEEE = SignedZeros = false;
    } else if (A->getOption().matches(options::OPT_fno_associative_math)) {
      AssociativeMath = false;
    } else if (A->getOption().matches(options::OPT_fsigned_zeros)) {
      SignedZeros = true;
      AssociativeMath = false;
    } else if (A->getOption().matches(options::OPT_fno_signed_zeros)) {
      SignedZeros = NeedIEEE = false;
    }
    A->claim();
  }

  // fp-contract=fast is the default
  bool EnableFPContraction = true;
  if (Arg *A = Args.getLastArg(options::OPT_ffp_contract, options::OPT_Mfma_on,
                               options::OPT_fma, options::OPT_Mfma_off,
                               options::OPT_nofma)) {
    auto Opt = A->getOption();
    if (Opt.matches(options::OPT_ffp_contract)) {
      StringRef Val = A->getValue();
      if ((Val == "fast") || (Val == "on")) {
        EnableFPContraction = true;
      } else if (Val == "off") {
        EnableFPContraction = false;
      } else {
        D.Diag(diag::err_drv_unsupported_option_argument)
            << A->getOption().getName() << Val;
      }
    } else if (Opt.matches(options::OPT_Mfma_on) ||
               Opt.matches(options::OPT_fma)) {
      EnableFPContraction = true;
    } else {
      EnableFPContraction = false;
    }
  }

  if (OptLevel == 0)
    EnableFPContraction = false;

  // Emit contract math instructions.
  // Step 1 : Generate fma instructions in flang (can override with fma flag)
  // Step 2 : Propagate fma contract information to LLVM to further
  //          exploit contraction opportunities
  if (EnableFPContraction) {
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("172");
    LowerCmdArgs.push_back("0x40000000");
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("179");
    LowerCmdArgs.push_back("1");
    // Step 2
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("216");
    LowerCmdArgs.push_back("0x1000");
  } else {
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("171");
    LowerCmdArgs.push_back("0x40000000");
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("178");
    LowerCmdArgs.push_back("1");
  }

  if (NeedFastMath) {
    // Lower: -x 216 1
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("216");
    LowerCmdArgs.push_back("1");
    // Common: -ieee 0
    CommonCmdArgs.push_back("-ieee");
    CommonCmdArgs.push_back("0");
  } else if (NeedIEEE) {
    // Common: -y 129 2
    CommonCmdArgs.push_back("-y");
    CommonCmdArgs.push_back("129");
    CommonCmdArgs.push_back("2");
    // Lower: -x 6 0x100
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("6");
    LowerCmdArgs.push_back("0x100");
    // Lower: -x 42 0x400000
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("42");
    LowerCmdArgs.push_back("0x400000");
    // Lower: -y 129 4
    LowerCmdArgs.push_back("-y");
    LowerCmdArgs.push_back("129");
    LowerCmdArgs.push_back("4");
    // Lower: -x 129 0x400
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("129");
    LowerCmdArgs.push_back("0x400");
    // Lower: -y 216 1 (OPT_fno_fast_math)
    LowerCmdArgs.push_back("-y");
    LowerCmdArgs.push_back("216");
    LowerCmdArgs.push_back("1");
    // Common: -ieee 1
    CommonCmdArgs.push_back("-ieee");
    CommonCmdArgs.push_back("1");
  } else if (NeedRelaxedMath) {
    // Lower: -x 15 0x400
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("15");
    LowerCmdArgs.push_back("0x400");
    // Lower: -y 216 1 (OPT_fno_fast_math)
    LowerCmdArgs.push_back("-y");
    LowerCmdArgs.push_back("216");
    LowerCmdArgs.push_back("1");
    // Common: -ieee 0
    CommonCmdArgs.push_back("-ieee");
    CommonCmdArgs.push_back("0");
  } else {
    // Common: -ieee 0
    CommonCmdArgs.push_back("-ieee");
    CommonCmdArgs.push_back("0");
  }

  /***** Upper part of the Fortran frontend *****/

  // TODO do we need to invoke this under GDB sometimes?
  const char *UpperExec =
      Args.MakeArgString(getToolChain().GetProgramPath("flang1"));

  UpperCmdArgs.push_back("-opt");
  UpperCmdArgs.push_back(Args.MakeArgString(OptOStr));
  UpperCmdArgs.push_back("-terse");
  UpperCmdArgs.push_back("1");
  UpperCmdArgs.push_back("-inform");
  UpperCmdArgs.push_back("warn");
  UpperCmdArgs.push_back("-nohpf");
  UpperCmdArgs.push_back("-nostatic");
  UpperCmdArgs.append(CommonCmdArgs.begin(),
                      CommonCmdArgs.end()); // Append common arguments
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("19");
  UpperCmdArgs.push_back("0x400000");
  UpperCmdArgs.push_back("-quad");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("68");
  UpperCmdArgs.push_back("0x1");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("59");
  UpperCmdArgs.push_back("4");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("15");
  UpperCmdArgs.push_back("2");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("49");
  UpperCmdArgs.push_back("0x400004");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("51");
  UpperCmdArgs.push_back("0x20");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("57");
  UpperCmdArgs.push_back("0x4c");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("58");
  UpperCmdArgs.push_back("0x10000");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("124");
  UpperCmdArgs.push_back("0x1000");
  UpperCmdArgs.push_back("-tp");
  UpperCmdArgs.push_back("px");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("57");
  UpperCmdArgs.push_back("0xfb0000");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("58");
  UpperCmdArgs.push_back("0x78031040");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("47");
  UpperCmdArgs.push_back("0x08");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("48");
  UpperCmdArgs.push_back("4608");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("49");
  UpperCmdArgs.push_back("0x100");
  if (OptLevel >= 2) {
    UpperCmdArgs.push_back("-x");
    UpperCmdArgs.push_back("70");
    UpperCmdArgs.push_back("0x6c00");
    UpperCmdArgs.push_back("-x");
    UpperCmdArgs.push_back("119");
    UpperCmdArgs.push_back("0x10000000");
    UpperCmdArgs.push_back("-x");
    UpperCmdArgs.push_back("129");
    UpperCmdArgs.push_back("2");
    UpperCmdArgs.push_back("-x");
    UpperCmdArgs.push_back("47");
    UpperCmdArgs.push_back("0x400000");
    UpperCmdArgs.push_back("-x");
    UpperCmdArgs.push_back("52");
    UpperCmdArgs.push_back("2");
  }

  // Add system include arguments.
  getToolChain().AddFlangSystemIncludeArgs(Args, UpperCmdArgs);

  // Use clang's predefined macros
  DiagnosticsEngine DE(new DiagnosticIDs(), new DiagnosticOptions,
                       new IgnoringDiagConsumer());
  std::shared_ptr<clang::TargetOptions> TO =
      std::make_shared<clang::TargetOptions>();
  TO->Triple = Triple.getTriple();
  std::shared_ptr<TargetInfo> TI(clang::TargetInfo::CreateTargetInfo(DE, TO));
  std::string PredefineBuffer;
  llvm::raw_string_ostream Predefines(PredefineBuffer);
  ClassicFlangMacroBuilder Builder(UpperCmdArgs, Args, Predefines);

  LangOptions LO;
  VersionTuple VT = getToolChain().computeMSVCVersion(&D, Args);
  if (!VT.empty()) {
    // Set the MSCompatibility version. Subminor version has 5 decimal digits.
    // Minor and major versions have 2 decimal digits each.
    LO.MSCompatibilityVersion = VT.getMajor() * 10000000 +
                                VT.getMinor().value_or(0) * 100000 +
                                VT.getSubminor().value_or(0);
  }

  // Define Target specific macros like __linux__
  TI->getTargetDefines(LO, Builder);

  Builder.defineMacro("__SIZE_TYPE__",
                      TargetInfo::getTypeName(TI->getSizeType()));
  Builder.defineMacro(
      "__PTRDIFF_TYPE__",
      TargetInfo::getTypeName(TI->getPtrDiffType(LangAS::Default)));

  if (TI->getPointerWidth(LangAS::Default) == 64 && TI->getLongWidth() == 64 &&
      TI->getIntWidth() == 32) {
    Builder.defineMacro("_LP64");
    Builder.defineMacro("__LP64__");
  }

  if (TI->getPointerWidth(LangAS::Default) == 32 && TI->getLongWidth() == 32 &&
      TI->getIntWidth() == 32) {
    Builder.defineMacro("_ILP32");
    Builder.defineMacro("__ILP32__");
  }

  DefineTypeSize("__LONG_MAX__", TargetInfo::SignedLong, *TI, Builder);

  // Add additional predefined macros
  switch (Triple.getArch()) {
  case llvm::Triple::aarch64:
    UpperCmdArgs.push_back("-def");
    UpperCmdArgs.push_back("__ARM_ARCH__=8");
    break;
  case llvm::Triple::x86_64:
    UpperCmdArgs.push_back("-def");
    UpperCmdArgs.push_back("__amd_64__amd64__");
    UpperCmdArgs.push_back("-def");
    UpperCmdArgs.push_back("__k8");
    UpperCmdArgs.push_back("-def");
    UpperCmdArgs.push_back("__k8__");
    break;
  default: /* generic 64-bit */
      ;
  }
  UpperCmdArgs.push_back("-def");
  UpperCmdArgs.push_back("__THROW=");
  UpperCmdArgs.push_back("-def");
  UpperCmdArgs.push_back("__extension__=");
  UpperCmdArgs.push_back("-def");
  UpperCmdArgs.push_back("__PGLLVM__");

  // Enable preprocessor
  if (Args.hasArg(options::OPT_Mpreprocess) || Args.hasArg(options::OPT_cpp) ||
      Args.hasArg(options::OPT_E) ||
      types::getPreprocessedType(InputType) != types::TY_INVALID) {
    UpperCmdArgs.push_back("-preprocess");
    for (auto Arg : Args.filtered(options::OPT_Mpreprocess, options::OPT_cpp,
                                  options::OPT_E)) {
      Arg->claim();
    }

    // When -E option is provided, run only the fortran preprocessor.
    // Only in -E mode, consume -P if it exists
    if (Args.hasArg(options::OPT_E)) {
      UpperCmdArgs.push_back("-es");
      // Line marker mode is disabled
      if (Args.hasArg(options::OPT_P)) {
        Args.ClaimAllArgs(options::OPT_P);
      } else {
        // -pp enables line marker mode in fortran preprocessor
        UpperCmdArgs.push_back("-pp");
      }
    }
  }

  // Enable standards checking
  if (Args.hasArg(options::OPT_Mstandard)) {
    UpperCmdArgs.push_back("-standard");
    for (auto Arg : Args.filtered(options::OPT_Mstandard)) {
      Arg->claim();
    }
  }

  // Free or fixed form file
  if (Args.hasArg(options::OPT_fortran_format_Group)) {
    // Override file name suffix, scan arguments for that
    for (Arg *A : Args.filtered(options::OPT_fortran_format_Group)) {
      A->claim();
      switch (A->getOption().getID()) {
      default:
        llvm_unreachable("missed a case");
      case options::OPT_ffixed_form:
      case options::OPT_fno_free_form:
      case options::OPT_Mfixed:
      case options::OPT_Mfree_off:
      case options::OPT_Mfreeform_off:
        UpperCmdArgs.push_back("-nofreeform");
        break;
      case options::OPT_ffree_form:
      case options::OPT_fno_fixed_form:
      case options::OPT_Mfree_on:
      case options::OPT_Mfreeform_on:
        UpperCmdArgs.push_back("-freeform");
        break;
      }
    }
  } else {
    // Deduce format from file name suffix
    if (types::isFreeFormFortran(InputType)) {
      UpperCmdArgs.push_back("-freeform");
    } else {
      UpperCmdArgs.push_back("-nofreeform");
    }
  }

  // Extend lines to 132 characters
  for (auto Arg : Args.filtered(options::OPT_Mextend)) {
    Arg->claim();
    UpperCmdArgs.push_back("-extend");
  }

  for (auto Arg : Args.filtered(options::OPT_ffixed_line_length_VALUE)) {
    StringRef Value = Arg->getValue();
    if (Value == "72") {
      Arg->claim();
    } else if (Value == "132") {
      Arg->claim();
      UpperCmdArgs.push_back("-extend");
    } else {
      D.Diag(diag::err_drv_unsupported_fixed_line_length)
          << Arg->getAsString(Args);
    }
  }

  // Add user-defined include directories
  for (auto Arg : Args.filtered(options::OPT_I)) {
    Arg->claim();
    UpperCmdArgs.push_back("-idir");
    UpperCmdArgs.push_back(Arg->getValue(0));
  }

  // Add user-defined module directories
  for (auto Arg : Args.filtered(options::OPT_ModuleDir, options::OPT_J)) {
    Arg->claim();
    UpperCmdArgs.push_back("-moddir");
    UpperCmdArgs.push_back(Arg->getValue(0));
  }

  // "Define" preprocessor flags
  for (auto Arg : Args.filtered(options::OPT_D)) {
    Arg->claim();
    UpperCmdArgs.push_back("-def");
    UpperCmdArgs.push_back(Arg->getValue(0));
  }

  // "Define" preprocessor flags
  for (auto Arg : Args.filtered(options::OPT_U)) {
    Arg->claim();
    UpperCmdArgs.push_back("-undef");
    UpperCmdArgs.push_back(Arg->getValue(0));
  }

  UpperCmdArgs.push_back("-vect");
  UpperCmdArgs.push_back("48");

  // Semantics for assignments to allocatables
  if (Arg *A = Args.getLastArg(options::OPT_Mallocatable_EQ)) {
    // Argument is passed explicitly
    StringRef Value = A->getValue();
    if (Value == "03") {            // Enable Fortran 2003 semantics
      UpperCmdArgs.push_back("-x"); // Set XBIT
    } else if (Value == "95") {     // Enable Fortran 2003 semantics
      UpperCmdArgs.push_back("-y"); // Unset XBIT
    } else {
      D.Diag(diag::err_drv_invalid_allocatable_mode) << A->getAsString(Args);
    }
  } else {                        // No argument passed
    UpperCmdArgs.push_back("-x"); // Default is 03
  }
  UpperCmdArgs.push_back("54");
  UpperCmdArgs.push_back("1"); // XBIT value

  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("70");
  UpperCmdArgs.push_back("0x40000000");
  UpperCmdArgs.push_back("-y");
  UpperCmdArgs.push_back("163");
  UpperCmdArgs.push_back("0xc0000000");
  UpperCmdArgs.push_back("-x");
  UpperCmdArgs.push_back("189");
  UpperCmdArgs.push_back("0x10");

  // Enable NULL pointer checking
  if (Args.hasArg(options::OPT_Mchkptr)) {
    UpperCmdArgs.push_back("-x");
    UpperCmdArgs.push_back("70");
    UpperCmdArgs.push_back("4");
    for (auto Arg : Args.filtered(options::OPT_Mchkptr)) {
      Arg->claim();
    }
  }

  // Set a -x flag for first part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Hx_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    auto XFlag = Value.split(",");
    UpperCmdArgs.push_back("-x");
    UpperCmdArgs.push_back(Args.MakeArgString(XFlag.first));
    UpperCmdArgs.push_back(Args.MakeArgString(XFlag.second));
  }

  // Set a -y flag for first part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Hy_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    auto XFlag = Value.split(",");
    UpperCmdArgs.push_back("-y");
    UpperCmdArgs.push_back(Args.MakeArgString(XFlag.first));
    UpperCmdArgs.push_back(Args.MakeArgString(XFlag.second));
  }

  // Set a -q (debug) flag for first part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Hq_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    auto XFlag = Value.split(",");
    UpperCmdArgs.push_back("-q");
    UpperCmdArgs.push_back(Args.MakeArgString(XFlag.first));
    UpperCmdArgs.push_back(Args.MakeArgString(XFlag.second));
  }

  // Set a -qq (debug) flag for first part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Hqq_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    auto XFlag = Value.split(",");
    UpperCmdArgs.push_back("-qq");
    UpperCmdArgs.push_back(Args.MakeArgString(XFlag.first));
    UpperCmdArgs.push_back(Args.MakeArgString(XFlag.second));
  }

  const char *STBFile = Args.MakeArgString(Stem + ".stb");
  C.addTempFile(STBFile);
  UpperCmdArgs.push_back("-stbfile");
  UpperCmdArgs.push_back(STBFile);

  const char *ModuleExportFile = Args.MakeArgString(Stem + ".cmod");
  C.addTempFile(ModuleExportFile);
  UpperCmdArgs.push_back("-modexport");
  UpperCmdArgs.push_back(ModuleExportFile);

  const char *ModuleIndexFile = Args.MakeArgString(Stem + ".cmdx");
  C.addTempFile(ModuleIndexFile);
  UpperCmdArgs.push_back("-modindex");
  UpperCmdArgs.push_back(ModuleIndexFile);

  UpperCmdArgs.push_back("-output");
  UpperCmdArgs.push_back(ILMFile);

  SmallString<256> Path;
  if (Args.getAllArgValues(options::OPT_fopenmp_targets_EQ).size() > 0) {
    SmallString<128> TargetInfo;
    Path = llvm::sys::path::parent_path(Output.getFilename());
    Arg *Tgts = Args.getLastArg(options::OPT_fopenmp_targets_EQ);
    assert(Tgts && Tgts->getNumValues() &&
           "OpenMP offloading has to have targets specified.");
    for (unsigned i = 0; i < Tgts->getNumValues(); ++i) {
      if (i)
        TargetInfo += ',';
      llvm::Triple T(Tgts->getValue(i));
      TargetInfo += T.getTriple();
    }
    UpperCmdArgs.push_back("-fopenmp-targets");
    UpperCmdArgs.push_back(Args.MakeArgString(TargetInfo.str()));
  }

  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileUTF8(),
                                         UpperExec, UpperCmdArgs, Inputs));

  // For -fsyntax-only or -E that is it
  if (Args.hasArg(options::OPT_fsyntax_only) || Args.hasArg(options::OPT_E))
    return;

  /***** Lower part of Fortran frontend *****/

  const char *LowerExec =
      Args.MakeArgString(getToolChain().GetProgramPath("flang2"));

  // TODO FLANG arg handling
  LowerCmdArgs.push_back("-fn");
  LowerCmdArgs.push_back(Input.getBaseInput());
  LowerCmdArgs.push_back("-opt");
  LowerCmdArgs.push_back(Args.MakeArgString(OptOStr));
  LowerCmdArgs.push_back("-terse");
  LowerCmdArgs.push_back("1");
  LowerCmdArgs.push_back("-inform");
  LowerCmdArgs.push_back("warn");
  LowerCmdArgs.append(CommonCmdArgs.begin(),
                      CommonCmdArgs.end()); // Append common arguments
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("68");
  LowerCmdArgs.push_back("0x1");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("51");
  LowerCmdArgs.push_back("0x20");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("119");
  LowerCmdArgs.push_back("0xa10000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("122");
  LowerCmdArgs.push_back("0x40");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("123");
  LowerCmdArgs.push_back("0x1000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("127");
  LowerCmdArgs.push_back("4");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("127");
  LowerCmdArgs.push_back("17");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("19");
  LowerCmdArgs.push_back("0x400000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("28");
  LowerCmdArgs.push_back("0x40000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("120");
  LowerCmdArgs.push_back("0x10000000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("70");
  LowerCmdArgs.push_back("0x8000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("122");
  LowerCmdArgs.push_back("1");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("125");
  LowerCmdArgs.push_back("0x20000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("164");
  LowerCmdArgs.push_back("0x800000");
  LowerCmdArgs.push_back("-quad");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("59");
  LowerCmdArgs.push_back("4");
  LowerCmdArgs.push_back("-tp");
  LowerCmdArgs.push_back("px");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("120");
  LowerCmdArgs.push_back("0x1000"); // debug lite
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("124");
  LowerCmdArgs.push_back("0x1400");
  LowerCmdArgs.push_back("-y");
  LowerCmdArgs.push_back("15");
  LowerCmdArgs.push_back("2");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("57");
  LowerCmdArgs.push_back("0x3b0000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("58");
  LowerCmdArgs.push_back("0x48000000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("49");
  LowerCmdArgs.push_back("0x100");
  LowerCmdArgs.push_back("-astype");
  LowerCmdArgs.push_back("0");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("183");
  LowerCmdArgs.push_back("4");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("121");
  LowerCmdArgs.push_back("0x800");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("54");
  LowerCmdArgs.push_back("0x10");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("70");
  LowerCmdArgs.push_back("0x40000000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("249");
  LowerCmdArgs.push_back("150"); // LLVM version
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("124");
  LowerCmdArgs.push_back("1");
  LowerCmdArgs.push_back("-y");
  LowerCmdArgs.push_back("163");
  LowerCmdArgs.push_back("0xc0000000");
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("189");
  LowerCmdArgs.push_back("0x10");
  LowerCmdArgs.push_back("-y");
  LowerCmdArgs.push_back("189");
  LowerCmdArgs.push_back("0x4000000");

  if (!SignedZeros) {
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("216");
    LowerCmdArgs.push_back("0x8");
  }
  if (AssociativeMath) {
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back("216");
    LowerCmdArgs.push_back("0x10");
  }

  // Remove "noinline" attriblute
  LowerCmdArgs.push_back("-x");
  LowerCmdArgs.push_back("183");
  LowerCmdArgs.push_back("0x10");

  // Add target features
  std::vector<StringRef> Features;
  std::string FeatureList = "";
  getTargetFeatureList(D, Triple, Args, UpperCmdArgs, false, Features);
  if (!Features.empty()) {
    for (auto Feature : unifyTargetFeatures(Features)) {
      if (!FeatureList.empty())
        FeatureList += ',';
      FeatureList += Feature;
    }

    LowerCmdArgs.push_back("-target_features");
    LowerCmdArgs.push_back(Args.MakeArgString(FeatureList));
  }

  // Add vscale range
  unsigned vscaleMin = 0;
  unsigned vscaleMax = 0;
  bool hasSVE = false;
  if (Arg *A = Args.getLastArg(options::OPT_msve_vector_bits_EQ)) {
    StringRef Val = A->getValue();
    if (Val == "128" || Val == "256" || Val == "512" ||
        Val == "1024" || Val == "2048" || Val == "128+" ||
        Val == "256+" || Val == "512+" || Val == "1024+" ||
        Val == "2048+") {
      unsigned Bits = 0;
      if (Val.ends_with("+"))
        Val = Val.substr(0, Val.size() - 1);
      else {
        bool Invalid = Val.getAsInteger(10, Bits);
        (void)Invalid;
        assert(!Invalid && "Failed to parse value");
        vscaleMax = Bits / 128;
      }

      bool Invalid = Val.getAsInteger(10, Bits);
      (void)Invalid;
      assert(!Invalid && "Failed to parse value");
      vscaleMin = Bits / 128;
    } else if (Val != "scalable")
      getToolChain().getDriver().Diag(diag::warn_drv_clang_unsupported)
          << A->getOption().getName() << Val;
  }
  for (auto Feature : unifyTargetFeatures(Features)) {
    if (Feature.starts_with("+sve")) {
      hasSVE = true;
      break;
    }
  }
  if (vscaleMin || vscaleMax) {
    LowerCmdArgs.push_back("-vscale_range_min");
    LowerCmdArgs.push_back(
        Args.MakeArgString(std::to_string(vscaleMin ? vscaleMin : 1)));
    LowerCmdArgs.push_back("-vscale_range_max");
    LowerCmdArgs.push_back(Args.MakeArgString(std::to_string(vscaleMax)));
  } else {
    if (hasSVE) {
      LowerCmdArgs.push_back("-vscale_range_min");
      LowerCmdArgs.push_back(Args.MakeArgString(std::to_string(1)));
      LowerCmdArgs.push_back("-vscale_range_max");
      LowerCmdArgs.push_back(Args.MakeArgString(std::to_string(16)));
    }
  }

  // Set a -x flag for second part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Mx_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    auto XFlag = Value.split(",");
    LowerCmdArgs.push_back("-x");
    LowerCmdArgs.push_back(Args.MakeArgString(XFlag.first));
    LowerCmdArgs.push_back(Args.MakeArgString(XFlag.second));
  }

  // Set a -y flag for second part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_My_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    auto XFlag = Value.split(",");
    LowerCmdArgs.push_back("-y");
    LowerCmdArgs.push_back(Args.MakeArgString(XFlag.first));
    LowerCmdArgs.push_back(Args.MakeArgString(XFlag.second));
  }

  // Set a -q (debug) flag for second part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Mq_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    auto XFlag = Value.split(",");
    LowerCmdArgs.push_back("-q");
    LowerCmdArgs.push_back(Args.MakeArgString(XFlag.first));
    LowerCmdArgs.push_back(Args.MakeArgString(XFlag.second));
  }

  // Set a -qq (debug) flag for second part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Mqq_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    auto XFlag = Value.split(",");
    LowerCmdArgs.push_back("-qq");
    LowerCmdArgs.push_back(Args.MakeArgString(XFlag.first));
    LowerCmdArgs.push_back(Args.MakeArgString(XFlag.second));
  }

  // Pass an arbitrary flag for second part of Fortran frontend
  for (Arg *A : Args.filtered(options::OPT_Wm_EQ)) {
    A->claim();
    StringRef Value = A->getValue();
    SmallVector<StringRef, 8> PassArgs;
    Value.split(PassArgs, StringRef(","));
    for (StringRef PassArg : PassArgs) {
      LowerCmdArgs.push_back(Args.MakeArgString(PassArg));
    }
  }

  LowerCmdArgs.push_back("-stbfile");
  LowerCmdArgs.push_back(STBFile);

  Path = llvm::sys::path::parent_path(Output.getFilename());
  bool IsOpenMPDevice = JA.isDeviceOffloading(Action::OFK_OpenMP);

  /* OpenMP GPU Offload */
  if (Args.getAllArgValues(options::OPT_fopenmp_targets_EQ).size() > 0) {
    SmallString<128> TargetInfo;    //("-fopenmp-targets ");
    SmallString<256> TargetInfoAsm; //("-fopenmp-targets-asm ");

    Arg *Tgts = Args.getLastArg(options::OPT_fopenmp_targets_EQ);
    assert(Tgts && Tgts->getNumValues() &&
           "OpenMP offloading has to have targets specified.");
    for (unsigned i = 0; i < Tgts->getNumValues(); ++i) {
      if (i)
        TargetInfo += ',';
      // We need to get the string from the triple because it may be not exactly
      // the same as the one we get directly from the arguments.
      llvm::Triple T(Tgts->getValue(i));
      TargetInfo += T.getTriple();
      // We also need to give a output file
      TargetInfoAsm += Path;
      TargetInfoAsm += "/";
      TargetInfoAsm += Stem;
      TargetInfoAsm += "-";
      TargetInfoAsm += T.getTriple();
      TargetInfoAsm += ".ll";
    }
    // The driver is aware that flang2 can generate multiple files at the same
    // time. We mimic it here by exchanging the output files. The driver always
    // uses the output file of -asm.
    LowerCmdArgs.push_back("-fopenmp-targets");
    LowerCmdArgs.push_back(Args.MakeArgString(TargetInfo.str()));
    if (IsOpenMPDevice) {
      LowerCmdArgs.push_back("-fopenmp-targets-asm");
      LowerCmdArgs.push_back(Args.MakeArgString(OutFile));
      LowerCmdArgs.push_back("-asm");
      LowerCmdArgs.push_back(Args.MakeArgString(TargetInfoAsm.str()));
    } else {
      LowerCmdArgs.push_back("-fopenmp-targets-asm");
      LowerCmdArgs.push_back(Args.MakeArgString(TargetInfoAsm.str()));
      LowerCmdArgs.push_back("-asm");
      LowerCmdArgs.push_back(Args.MakeArgString(OutFile));
    }
  } else {
    LowerCmdArgs.push_back("-asm");
    LowerCmdArgs.push_back(Args.MakeArgString(OutFile));
  }

  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileUTF8(),
                                         LowerExec, LowerCmdArgs, Inputs));
}
