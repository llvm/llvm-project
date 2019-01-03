//===--- JSONAggregation.h - Index data aggregation in JSON format --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CINDEXTEST_JSONAGGREGATION_H
#define LLVM_CLANG_TOOLS_CINDEXTEST_JSONAGGREGATION_H

#include "clang/Basic/LLVM.h"

namespace clang {
namespace index {

/// Returns true if an error occurred, false otherwise.
bool aggregateDataAsJSON(StringRef StorePath, raw_ostream &OS);

} // end namespace index
} // end namespace clang

#endif
