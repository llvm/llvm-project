//===----- SemaRISCV.h ---- RISC-V target-specific routines ---*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to RISC-V.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMARISCV_H
#define LLVM_CLANG_SEMA_SEMARISCV_H

#include "clang/AST/ASTFwd.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {
namespace sema {
class RISCVIntrinsicManager;
} // namespace sema

class ParsedAttr;
class TargetInfo;

class SemaRISCV : public SemaBase {
public:
  SemaRISCV(Sema &S);

  bool CheckLMUL(CallExpr *TheCall, unsigned ArgNum);
  bool CheckBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                CallExpr *TheCall);
  void checkRVVTypeSupport(QualType Ty, SourceLocation Loc, Decl *D,
                           const llvm::StringMap<bool> &FeatureMap);

  bool isValidRVVBitcast(QualType srcType, QualType destType);

  void handleInterruptAttr(Decl *D, const ParsedAttr &AL);
  bool isAliasValid(unsigned BuiltinID, llvm::StringRef AliasName);
  bool isValidFMVExtension(StringRef Ext);

  /// Indicate RISC-V vector builtin functions enabled or not.
  bool DeclareRVVBuiltins = false;

  /// Indicate RISC-V SiFive vector builtin functions enabled or not.
  bool DeclareSiFiveVectorBuiltins = false;

  /// Indicate RISC-V Andes vector builtin functions enabled or not.
  bool DeclareAndesVectorBuiltins = false;

  std::unique_ptr<sema::RISCVIntrinsicManager> IntrinsicManager;

  bool checkTargetVersionAttr(const StringRef Param, const SourceLocation Loc);
  bool checkTargetClonesAttr(SmallVectorImpl<StringRef> &Params,
                             SmallVectorImpl<SourceLocation> &Locs,
                             SmallVectorImpl<SmallString<64>> &NewParams);
};

std::unique_ptr<sema::RISCVIntrinsicManager>
CreateRISCVIntrinsicManager(Sema &S);
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMARISCV_H
