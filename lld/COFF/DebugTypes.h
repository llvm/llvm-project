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
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace codeview {
class PrecompRecord;
class TypeServer2Record;
} // namespace codeview
namespace pdb {
class NativeSession;
}
} // namespace llvm

namespace lld {
namespace coff {

class ObjFile;

class TpiSource {
public:
  enum TpiKind { Regular, PCH, UsingPCH, PDB, UsingPDB };

  TpiSource(TpiKind K, const ObjFile *F);
  virtual ~TpiSource() {}

  const TpiKind Kind;
  const ObjFile *File;
};

TpiSource *makeTpiSource(const ObjFile *F);
TpiSource *makeUseTypeServerSource(const ObjFile *F,
                                   const llvm::codeview::TypeServer2Record *TS);
TpiSource *makePrecompSource(const ObjFile *F);
TpiSource *makeUsePrecompSource(const ObjFile *F,
                                const llvm::codeview::PrecompRecord *Precomp);

void loadTypeServerSource(llvm::MemoryBufferRef M);

// Temporary interface to get the dependency
template <typename T> const T &retrieveDependencyInfo(const TpiSource *Source);

// Temporary interface until we move PDBLinker::maybeMergeTypeServerPDB here
llvm::Expected<llvm::pdb::NativeSession *>
findTypeServerSource(const ObjFile *F);

} // namespace coff
} // namespace lld

#endif