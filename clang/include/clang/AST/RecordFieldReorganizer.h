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
#include <random>

namespace clang {

// FIXME: Find a better alternative to SmallVector with hardcoded size!

class RecordFieldReorganizer {
public:
  virtual ~RecordFieldReorganizer() = default;
  void reorganizeFields(const ASTContext &C, const RecordDecl *D);

protected:
  virtual void reorganize(const ASTContext &C, const RecordDecl *D,
                          SmallVector<Decl *, 64> &NewOrder) = 0;

private:
  void commit(const RecordDecl *D,
              SmallVectorImpl<Decl *> &NewFieldOrder) const;
};

class Randstruct : public RecordFieldReorganizer {
public:
  Randstruct(std::string seed) : Seq(seed.begin(), seed.end()), rng(Seq) {}

  /// Determines if the Record can be safely and easily randomized based on certain criteria (see implementation).
  static bool isTriviallyRandomizable(const RecordDecl *D);
protected:
  SmallVector<Decl *, 64> randomize(SmallVector<Decl *, 64> fields);
  SmallVector<Decl *, 64> perfrandomize(const ASTContext &ctx,
                                      SmallVector<Decl *, 64> fields);
  virtual void reorganize(const ASTContext &C, const RecordDecl *D,
                          SmallVector<Decl *, 64> &NewOrder) override;
private:
  std::seed_seq Seq;
  std::default_random_engine rng;
};

} // namespace clang

#endif
