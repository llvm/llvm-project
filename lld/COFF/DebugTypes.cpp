//===- DebugTypes.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DebugTypes.h"
#include "Driver.h"
#include "InputFiles.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/Support/Path.h"

using namespace lld;
using namespace lld::coff;
using namespace llvm;
using namespace llvm::codeview;

namespace {
// The TypeServerSource class represents a PDB type server, a file referenced by
// OBJ files compiled with MSVC /Zi. A single PDB can be shared by several OBJ
// files, therefore there must be only once instance per OBJ lot. The file path
// is discovered from the dependent OBJ's debug type stream. The
// TypeServerSource object is then queued and loaded by the COFF Driver. The
// debug type stream for such PDB files will be merged first in the final PDB,
// before any dependent OBJ.
class TypeServerSource : public TpiSource {
public:
  explicit TypeServerSource(MemoryBufferRef M, llvm::pdb::NativeSession *S)
      : TpiSource(PDB, nullptr), Session(S), MB(M) {}

  // Queue a PDB type server for loading in the COFF Driver
  static void enqueue(const ObjFile *DependentFile,
                      const TypeServer2Record &TS);

  // Create an instance
  static Expected<TypeServerSource *> getInstance(MemoryBufferRef M);

  // Fetch the PDB instance loaded for a corresponding dependent OBJ.
  static Expected<TypeServerSource *>
  findFromFile(const ObjFile *DependentFile);

  static std::map<std::string, std::pair<std::string, TypeServerSource *>>
      Instances;

  // The interface to the PDB (if it was opened successfully)
  std::unique_ptr<llvm::pdb::NativeSession> Session;

private:
  MemoryBufferRef MB;
};

// This class represents the debug type stream of an OBJ file that depends on a
// PDB type server (see TypeServerSource).
class UseTypeServerSource : public TpiSource {
public:
  UseTypeServerSource(const ObjFile *F, const TypeServer2Record *TS)
      : TpiSource(UsingPDB, F), TypeServerDependency(*TS) {}

  // Information about the PDB type server dependency, that needs to be loaded
  // in before merging this OBJ.
  TypeServer2Record TypeServerDependency;
};

// This class represents the debug type stream of a Microsoft precompiled
// headers OBJ (PCH OBJ). This OBJ kind needs to be merged first in the output
// PDB, before any other OBJs that depend on this. Note that only MSVC generate
// such files, clang does not.
class PrecompSource : public TpiSource {
public:
  PrecompSource(const ObjFile *F) : TpiSource(PCH, F) {}
};

// This class represents the debug type stream of an OBJ file that depends on a
// Microsoft precompiled headers OBJ (see PrecompSource).
class UsePrecompSource : public TpiSource {
public:
  UsePrecompSource(const ObjFile *F, const PrecompRecord *Precomp)
      : TpiSource(UsingPCH, F), PrecompDependency(*Precomp) {}

  // Information about the Precomp OBJ dependency, that needs to be loaded in
  // before merging this OBJ.
  PrecompRecord PrecompDependency;
};
} // namespace

static std::vector<std::unique_ptr<TpiSource>> GC;

TpiSource::TpiSource(TpiKind K, const ObjFile *F) : Kind(K), File(F) {
  GC.push_back(std::unique_ptr<TpiSource>(this));
}

TpiSource *lld::coff::makeTpiSource(const ObjFile *F) {
  return new TpiSource(TpiSource::Regular, F);
}

TpiSource *lld::coff::makeUseTypeServerSource(const ObjFile *F,
                                              const TypeServer2Record *TS) {
  TypeServerSource::enqueue(F, *TS);
  return new UseTypeServerSource(F, TS);
}

TpiSource *lld::coff::makePrecompSource(const ObjFile *F) {
  return new PrecompSource(F);
}

TpiSource *lld::coff::makeUsePrecompSource(const ObjFile *F,
                                           const PrecompRecord *Precomp) {
  return new UsePrecompSource(F, Precomp);
}

namespace lld {
namespace coff {
template <>
const PrecompRecord &retrieveDependencyInfo(const TpiSource *Source) {
  assert(Source->Kind == TpiSource::UsingPCH);
  return ((const UsePrecompSource *)Source)->PrecompDependency;
}

template <>
const TypeServer2Record &retrieveDependencyInfo(const TpiSource *Source) {
  assert(Source->Kind == TpiSource::UsingPDB);
  return ((const UseTypeServerSource *)Source)->TypeServerDependency;
}
} // namespace coff
} // namespace lld

std::map<std::string, std::pair<std::string, TypeServerSource *>>
    TypeServerSource::Instances;

// Make a PDB path assuming the PDB is in the same folder as the OBJ
static std::string getPdbBaseName(const ObjFile *File, StringRef TSPath) {
  StringRef LocalPath =
      !File->ParentName.empty() ? File->ParentName : File->getName();
  SmallString<128> Path = sys::path::parent_path(LocalPath);

  // Currently, type server PDBs are only created by MSVC cl, which only runs
  // on Windows, so we can assume type server paths are Windows style.
  sys::path::append(Path, sys::path::filename(TSPath, sys::path::Style::windows));
  return Path.str();
}

// The casing of the PDB path stamped in the OBJ can differ from the actual path
// on disk. With this, we ensure to always use lowercase as a key for the
// PDBInputFile::Instances map, at least on Windows.
static std::string normalizePdbPath(StringRef path) {
#if defined(_WIN32)
  return path.lower();
#else // LINUX
  return path;
#endif
}

// If existing, return the actual PDB path on disk.
static Optional<std::string> findPdbPath(StringRef PDBPath,
                                         const ObjFile *DependentFile) {
  // Ensure the file exists before anything else. In some cases, if the path
  // points to a removable device, Driver::enqueuePath() would fail with an
  // error (EAGAIN, "resource unavailable try again") which we want to skip
  // silently.
  if (llvm::sys::fs::exists(PDBPath))
    return normalizePdbPath(PDBPath);
  std::string Ret = getPdbBaseName(DependentFile, PDBPath);
  if (llvm::sys::fs::exists(Ret))
    return normalizePdbPath(Ret);
  return None;
}

// Fetch the PDB instance that was already loaded by the COFF Driver.
Expected<TypeServerSource *>
TypeServerSource::findFromFile(const ObjFile *DependentFile) {
  const TypeServer2Record &TS =
      retrieveDependencyInfo<TypeServer2Record>(DependentFile->DebugTypesObj);

  Optional<std::string> P = findPdbPath(TS.Name, DependentFile);
  if (!P)
    return createFileError(TS.Name, errorCodeToError(std::error_code(
                                        ENOENT, std::generic_category())));

  auto It = TypeServerSource::Instances.find(*P);
  // The PDB file exists on disk, at this point we expect it to have been
  // inserted in the map by TypeServerSource::loadPDB()
  assert(It != TypeServerSource::Instances.end());

  std::pair<std::string, TypeServerSource *> &PDB = It->second;

  if (!PDB.second)
    return createFileError(
        *P, createStringError(inconvertibleErrorCode(), PDB.first.c_str()));

  pdb::PDBFile &PDBFile = (PDB.second)->Session->getPDBFile();
  pdb::InfoStream &Info = cantFail(PDBFile.getPDBInfoStream());

  // Just because a file with a matching name was found doesn't mean it can be
  // used. The GUID must match between the PDB header and the OBJ
  // TypeServer2 record. The 'Age' is used by MSVC incremental compilation.
  if (Info.getGuid() != TS.getGuid())
    return createFileError(
        TS.Name,
        make_error<pdb::PDBError>(pdb::pdb_error_code::signature_out_of_date));

  return PDB.second;
}

// FIXME: Temporary interface until PDBLinker::maybeMergeTypeServerPDB() is
// moved here.
Expected<llvm::pdb::NativeSession *>
lld::coff::findTypeServerSource(const ObjFile *F) {
  Expected<TypeServerSource *> TS = TypeServerSource::findFromFile(F);
  if (!TS)
    return TS.takeError();
  return TS.get()->Session.get();
}

// Queue a PDB type server for loading in the COFF Driver
void TypeServerSource::enqueue(const ObjFile *DependentFile,
                               const TypeServer2Record &TS) {
  // Start by finding where the PDB is located (either the record path or next
  // to the OBJ file)
  Optional<std::string> P = findPdbPath(TS.Name, DependentFile);
  if (!P)
    return;
  auto It = TypeServerSource::Instances.emplace(
      *P, std::pair<std::string, TypeServerSource *>{});
  if (!It.second)
    return; // another OBJ already scheduled this PDB for load

  Driver->enqueuePath(*P, false);
}

// Create an instance of TypeServerSource or an error string if the PDB couldn't
// be loaded. The error message will be displayed later, when the referring OBJ
// will be merged in. NOTE - a PDB load failure is not a link error: some
// debug info will simply be missing from the final PDB - that is the default
// accepted behavior.
void lld::coff::loadTypeServerSource(llvm::MemoryBufferRef M) {
  std::string Path = normalizePdbPath(M.getBufferIdentifier());

  Expected<TypeServerSource *> TS = TypeServerSource::getInstance(M);
  if (!TS)
    TypeServerSource::Instances[Path] = {toString(TS.takeError()), nullptr};
  else
    TypeServerSource::Instances[Path] = {{}, *TS};
}

Expected<TypeServerSource *> TypeServerSource::getInstance(MemoryBufferRef M) {
  std::unique_ptr<llvm::pdb::IPDBSession> ISession;
  Error Err = pdb::NativeSession::createFromPdb(
      MemoryBuffer::getMemBuffer(M, false), ISession);
  if (Err)
    return std::move(Err);

  std::unique_ptr<llvm::pdb::NativeSession> Session(
      static_cast<pdb::NativeSession *>(ISession.release()));

  pdb::PDBFile &PDBFile = Session->getPDBFile();
  Expected<pdb::InfoStream &> Info = PDBFile.getPDBInfoStream();
  // All PDB Files should have an Info stream.
  if (!Info)
    return Info.takeError();
  return new TypeServerSource(M, Session.release());
}
