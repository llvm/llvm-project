//===-- ExpressionTypeSystemHelper.h ---------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ExpressionTypeSystemHelper_h
#define ExpressionTypeSystemHelper_h

#include "llvm/Support/Casting.h"

namespace lldb_private {

/// \class ExpressionTypeSystemHelper ExpressionTypeSystemHelper.h
/// "lldb/Expression/ExpressionTypeSystemHelper.h"
/// A helper object that the Expression can pass to its ExpressionParser
/// to provide generic information that
/// any type of expression will need to supply.  It's only job is to support
/// dyn_cast so that the expression parser can cast it back to the requisite
/// specific type.
///

class ExpressionTypeSystemHelper {
public:
  enum LLVMCastKind {
    eKindClangHelper,
    eKindSwiftHelper,
    eKindGoHelper,
    kNumKinds
  };

  LLVMCastKind getKind() const { return m_kind; }

  ExpressionTypeSystemHelper(LLVMCastKind kind) : m_kind(kind) {}

  ~ExpressionTypeSystemHelper() {}

protected:
  LLVMCastKind m_kind;
};

} // namespace lldb_private

#endif /* ExpressionTypeSystemHelper_h */
