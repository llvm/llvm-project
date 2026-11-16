//===--- ClangCLArgs.cpp - clang-cl arguments -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangCLArgs.h"
#include "clang/Options/Options.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/TargetParser/Triple.h"
#include <cstddef>
#include <string>

using namespace clang;
using namespace clang::driver;
using namespace llvm::opt;

ClangCLArgs::ClangCLArgs(const ArgList &Args, const llvm::Triple &HostTriple)
    : SupportsForcingFramePointer(HostTriple.getArch() !=
                                  llvm::Triple::x86_64) {
  // Expand only the last /O[12xd], preserving overrides such as /O2 /Oy-.
  for (Arg *A : Args.filtered(options::OPT__SLASH_O)) {
    llvm::StringRef OptStr = A->getValue();
    for (size_t I = 0, E = OptStr.size(); I != E; ++I) {
      char OptChar = OptStr[I];
      char PrevChar = I > 0 ? OptStr[I - 1] : '0';
      if (PrevChar == 'b') {
        // OptChar does not expand; it's an argument to the previous char.
        continue;
      }
      if (OptChar == '1' || OptChar == '2' || OptChar == 'x' || OptChar == 'd')
        ExpandChar = OptStr.data() + I;
    }
  }
}

bool ClangCLArgs::translateArg(Arg *A, DerivedArgList &DAL,
                               const DerivedArgList *Owner) const {
  const DerivedArgList &SynthesizedArgs = Owner ? *Owner : DAL;
  const Arg *BaseArg = &A->getBaseArg();
  const OptTable &Opts = getDriverOptTable();
  switch (A->getOption().getID()) {
  case options::OPT__SLASH_permissive:
    DAL.append(SynthesizedArgs.MakeFlagArg(
        BaseArg, Opts.getOption(options::OPT_fdelayed_template_parsing)));
    DAL.append(SynthesizedArgs.MakeFlagArg(
        BaseArg, Opts.getOption(options::OPT_fno_operator_names)));
    return true;
  case options::OPT__SLASH_permissive_:
    DAL.append(SynthesizedArgs.MakeFlagArg(
        BaseArg, Opts.getOption(options::OPT_fno_delayed_template_parsing)));
    DAL.append(SynthesizedArgs.MakeFlagArg(
        BaseArg, Opts.getOption(options::OPT_foperator_names)));
    return true;
  default:
    break;
  }

  if (!A->getOption().matches(options::OPT__SLASH_O))
    return false;

  // Keep the original argument for unused-option diagnostics.
  DAL.append(A);

  llvm::StringRef OptStr = A->getValue();
  for (size_t I = 0, E = OptStr.size(); I != E; ++I) {
    const char &OptChar = *(OptStr.data() + I);
    switch (OptChar) {
    default:
      break;
    case '1':
    case '2':
    case 'x':
    case 'd':
      // Ignore /O[12xd] flags that aren't the last one on the command line.
      // Only the last one gets expanded.
      if (&OptChar != ExpandChar) {
        A->claim();
        break;
      }
      if (OptChar == 'd') {
        DAL.append(SynthesizedArgs.MakeFlagArg(
            BaseArg, Opts.getOption(options::OPT_O0)));
      } else {
        if (OptChar == '1') {
          DAL.append(SynthesizedArgs.MakeJoinedArg(
              BaseArg, Opts.getOption(options::OPT_O), "s"));
        } else if (OptChar == '2' || OptChar == 'x') {
          DAL.append(SynthesizedArgs.MakeFlagArg(
              BaseArg, Opts.getOption(options::OPT_fbuiltin)));
          DAL.append(SynthesizedArgs.MakeJoinedArg(
              BaseArg, Opts.getOption(options::OPT_O), "3"));
        }
        if (SupportsForcingFramePointer &&
            !DAL.hasArgNoClaim(options::OPT_fno_omit_frame_pointer))
          DAL.append(SynthesizedArgs.MakeFlagArg(
              BaseArg, Opts.getOption(options::OPT_fomit_frame_pointer)));
        if (OptChar == '1' || OptChar == '2')
          DAL.append(SynthesizedArgs.MakeFlagArg(
              BaseArg, Opts.getOption(options::OPT_ffunction_sections)));
      }
      break;
    case 'b':
      if (I + 1 != E && llvm::isDigit(OptStr[I + 1])) {
        switch (OptStr[I + 1]) {
        case '0':
          DAL.append(SynthesizedArgs.MakeFlagArg(
              BaseArg, Opts.getOption(options::OPT_fno_inline)));
          break;
        case '1':
          DAL.append(SynthesizedArgs.MakeFlagArg(
              BaseArg, Opts.getOption(options::OPT_finline_hint_functions)));
          break;
        case '2':
        case '3':
          DAL.append(SynthesizedArgs.MakeFlagArg(
              BaseArg, Opts.getOption(options::OPT_finline_functions)));
          break;
        }
        ++I;
      }
      break;
    case 'g':
      A->claim();
      break;
    case 'i':
      if (I + 1 != E && OptStr[I + 1] == '-') {
        ++I;
        DAL.append(SynthesizedArgs.MakeFlagArg(
            BaseArg, Opts.getOption(options::OPT_fno_builtin)));
      } else {
        DAL.append(SynthesizedArgs.MakeFlagArg(
            BaseArg, Opts.getOption(options::OPT_fbuiltin)));
      }
      break;
    case 's':
      DAL.append(SynthesizedArgs.MakeJoinedArg(
          BaseArg, Opts.getOption(options::OPT_O), "s"));
      break;
    case 't':
      DAL.append(SynthesizedArgs.MakeJoinedArg(
          BaseArg, Opts.getOption(options::OPT_O), "3"));
      break;
    case 'y': {
      bool OmitFramePointer = true;
      if (I + 1 != E && OptStr[I + 1] == '-') {
        OmitFramePointer = false;
        ++I;
      }
      if (SupportsForcingFramePointer) {
        if (OmitFramePointer)
          DAL.append(SynthesizedArgs.MakeFlagArg(
              BaseArg, Opts.getOption(options::OPT_fomit_frame_pointer)));
        else
          DAL.append(SynthesizedArgs.MakeFlagArg(
              BaseArg, Opts.getOption(options::OPT_fno_omit_frame_pointer)));
      } else {
        // Silently accept /Oy- on x86-64 for portable clang-cl build flags.
        A->claim();
      }
      break;
    }
    }
  }
  return true;
}

DerivedArgList *ClangCLArgs::translateArgs(const DerivedArgList &Args,
                                           const llvm::Triple &HostTriple,
                                           const DerivedArgList &Owner) {
  ClangCLArgs Translator(Args, HostTriple);
  auto *DAL = new DerivedArgList(Args.getBaseArgs());
  for (Arg *A : Args) {
    if (!A->getOption().matches(options::OPT__SLASH_O) &&
        A->getBaseArg().getOption().matches(options::OPT__SLASH_O))
      continue;
    if (!Translator.translateArg(A, *DAL, &Owner))
      DAL->append(A);
  }
  return DAL;
}

const char *ClangCLArgs::translateMacroDefinition(const char *Value,
                                                  const ArgList &Args) {
  llvm::StringRef Val = Value;
  size_t Hash = Val.find('#');
  if (Hash == llvm::StringRef::npos || Hash > Val.find('='))
    return Value;

  std::string NewVal = std::string(Val);
  NewVal[Hash] = '=';
  return Args.MakeArgString(NewVal);
}
