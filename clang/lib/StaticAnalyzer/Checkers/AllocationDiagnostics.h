//=--- AllocationDiagnostics.h - Config options for allocation diags *- C++ -*-//
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
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_ALLOCATIONDIAGNOSTICS_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_ALLOCATIONDIAGNOSTICS_H

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"

namespace clang { namespace ento {

/// \brief Returns true if leak diagnostics should directly reference
/// the allocatin site (where possible).
///
/// The default is false.
///
bool shouldIncludeAllocationSiteInLeakDiagnostics(AnalyzerOptions &AOpts);

}}

#endif

