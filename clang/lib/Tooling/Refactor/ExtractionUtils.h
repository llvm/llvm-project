//===--- ExtractionUtils.h - Extraction helper functions ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_TOOLING_REFACTOR_EXTRACTION_UTILS_H
#define LLVM_CLANG_LIB_TOOLING_REFACTOR_EXTRACTION_UTILS_H

#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"

namespace clang {

class Expr;

namespace tooling {
namespace extract {

/// Returns a good name for an extracted variable based on the declaration
/// that's used in the given expression \p E.
Optional<StringRef> nameForExtractedVariable(const Expr *E);

} // end namespace extract
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_LIB_TOOLING_REFACTOR_EXTRACTION_UTILS_H
