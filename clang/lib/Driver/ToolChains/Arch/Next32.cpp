//===--- Next32.cpp - Next32 Helpers for Tools ------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Next32.h"
#include "ToolChains/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

void next32::getNext32TargetFeatures(const Driver &D,
                                     const llvm::Triple &Triple,
                                     const ArgList &Args,
                                     std::vector<StringRef> &Features) {
  // Add any that the user explicitly requested on the command line,
  // which may override the defaults.
  handleTargetFeaturesGroup(D, Triple, Args, Features,
                            options::OPT_m_next32_Features_Group);
}
