//===-- RecordFieldReorganizer.h - Interface for manipulating field order --*-
// C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file contains the base class that defines an interface for
// manipulating a RecordDecl's field layouts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_AST_RECORDFIELDREORGANIZER_H
#define LLVM_CLANG_LIB_AST_RECORDFIELDREORGANIZER_H

#include "Decl.h"

namespace clang {

// FIXME: Find a better alternative to SmallVector with hardcoded size!

class RecordFieldReorganizer {
public:
  virtual ~RecordFieldReorganizer() = default;
  void reorganizeFields(const ASTContext &C, const RecordDecl *D) const;

protected:
  virtual void reorganize(const ASTContext &C, const RecordDecl *D,
                          SmallVector<Decl *, 64> &NewOrder) const = 0;

private:
  void commit(const RecordDecl *D,
              SmallVectorImpl<Decl *> &NewFieldOrder) const;
};

class Randstruct : public RecordFieldReorganizer {
protected:
  virtual void reorganize(const ASTContext &C, const RecordDecl *D,
                          SmallVector<Decl *, 64> &NewOrder) const override;
};

} // namespace clang

#endif
