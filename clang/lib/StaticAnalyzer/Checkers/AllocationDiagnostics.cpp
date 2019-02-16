//=- AllocationDiagnostics.cpp - Config options for allocation diags *- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Declares the configuration functions for leaks/allocation diagnostics.
//
//===--------------------------

#include "AllocationDiagnostics.h"

namespace clang {
namespace ento {

bool shouldIncludeAllocationSiteInLeakDiagnostics(AnalyzerOptions &AOpts) {
  return AOpts.getBooleanOption("leak-diagnostics-reference-allocation",
                                false);
}

}}
