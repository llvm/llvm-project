//===- DebugTypes.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_DEBUGTYPES_H
#define LLD_COFF_DEBUGTYPES_H

#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {
class PrecompRecord;
class TypeServer2Record;
} // namespace codeview
} // namespace llvm

namespace lld {
namespace coff {

class ObjFile;

class TpiSource {
public:
  enum TpiKind { Regular, PCH, UsingPCH, PDB, UsingPDB };

  TpiSource(TpiKind K, ObjFile *F);
  virtual ~TpiSource() {}

  const TpiKind Kind;
  ObjFile *File;
};

TpiSource *makeTpiSource(ObjFile *F);
TpiSource *makeTypeServerSource(ObjFile *F);
TpiSource *makeUseTypeServerSource(ObjFile *F,
                                   llvm::codeview::TypeServer2Record *TS);
TpiSource *makePrecompSource(ObjFile *F);
TpiSource *makeUsePrecompSource(ObjFile *F,
                                llvm::codeview::PrecompRecord *Precomp);

// Temporary interface to get the dependency
template <typename T> const T &retrieveDependencyInfo(TpiSource *Source);

} // namespace coff
} // namespace lld

#endif