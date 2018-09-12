//===--- AllDiagnostics.h - Aggregate Diagnostic headers --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Includes all the separate Diagnostic headers & some related helpers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ALLDIAGNOSTICS_H
#define LLVM_CLANG_BASIC_ALLDIAGNOSTICS_H

#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/CommentDiagnostic.h"
#include "clang/Analysis/AnalysisDiagnostic.h"
#include "clang/CrossTU/CrossTUDiagnostic.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Serialization/SerializationDiagnostic.h"
#include "clang/Tooling/Refactoring/RefactoringDiagnostic.h"

namespace clang {
template <size_t SizeOfStr, typename FieldType>
class StringSizerHelper {
  static_assert(SizeOfStr <= FieldType(~0U), "Field too small!");
public:
  enum { Size = SizeOfStr };
};
} // end namespace clang

#define STR_SIZE(str, fieldTy) clang::StringSizerHelper<sizeof(str)-1, \
                                                        fieldTy>::Size

#endif
