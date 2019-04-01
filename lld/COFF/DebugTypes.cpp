//===- DebugTypes.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DebugTypes.h"
#include "InputFiles.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"

using namespace lld;
using namespace lld::coff;
using namespace llvm;
using namespace llvm::codeview;

namespace {
class TypeServerSource : public TpiSource {
public:
  TypeServerSource(ObjFile *F) : TpiSource(PDB, F) {}
};

class UseTypeServerSource : public TpiSource {
public:
  UseTypeServerSource(ObjFile *F, TypeServer2Record *TS)
      : TpiSource(UsingPDB, F), TypeServerDependency(*TS) {}

  // Information about the PDB type server dependency, that needs to be loaded
  // in before merging this OBJ.
  TypeServer2Record TypeServerDependency;
};

class PrecompSource : public TpiSource {
public:
  PrecompSource(ObjFile *F) : TpiSource(PCH, F) {}
};

class UsePrecompSource : public TpiSource {
public:
  UsePrecompSource(ObjFile *F, PrecompRecord *Precomp)
      : TpiSource(UsingPCH, F), PrecompDependency(*Precomp) {}

  // Information about the Precomp OBJ dependency, that needs to be loaded in
  // before merging this OBJ.
  PrecompRecord PrecompDependency;
};
} // namespace

static std::vector<std::unique_ptr<TpiSource>> GC;

TpiSource::TpiSource(TpiKind K, ObjFile *F) : Kind(K), File(F) {
  GC.push_back(std::unique_ptr<TpiSource>(this));
}

TpiSource *coff::makeTpiSource(ObjFile *F) {
  return new TpiSource(TpiSource::Regular, F);
}

TpiSource *coff::makeTypeServerSource(ObjFile *F) {
  return new TypeServerSource(F);
}

TpiSource *coff::makeUseTypeServerSource(ObjFile *F, TypeServer2Record *TS) {
  return new UseTypeServerSource(F, TS);
}

TpiSource *coff::makePrecompSource(ObjFile *F) { return new PrecompSource(F); }

TpiSource *coff::makeUsePrecompSource(ObjFile *F, PrecompRecord *Precomp) {
  return new UsePrecompSource(F, Precomp);
}

template <>
const PrecompRecord &lld::coff::retrieveDependencyInfo(TpiSource *Source) {
  assert(Source->Kind == TpiSource::UsingPCH);
  return ((UsePrecompSource *)Source)->PrecompDependency;
}

template <>
const TypeServer2Record &lld::coff::retrieveDependencyInfo(TpiSource *Source) {
  assert(Source->Kind == TpiSource::UsingPDB);
  return ((UseTypeServerSource *)Source)->TypeServerDependency;
}