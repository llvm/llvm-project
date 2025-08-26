//===- DependencyScanningCASFilesystem.cpp - clang-scan-deps fs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningCASFilesystem.h"
#include "clang/Basic/Version.h"
#include "clang/Lex/DependencyDirectivesScanner.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

template <typename T> static T reportAsFatalIfError(Expected<T> ValOrErr) {
  if (!ValOrErr)
    llvm::report_fatal_error(ValOrErr.takeError());
  return std::move(*ValOrErr);
}

static void reportAsFatalIfError(llvm::Error E) {
  if (E)
    llvm::report_fatal_error(std::move(E));
}

using llvm::Error;

DependencyScanningCASFilesystem::DependencyScanningCASFilesystem(
    IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> WorkerFS,
    llvm::cas::ActionCache &Cache)
    : FS(WorkerFS), Entries(EntryAlloc), CAS(WorkerFS->getCAS()), Cache(Cache) {
}

const char DependencyScanningCASFilesystem::ID = 0;
DependencyScanningCASFilesystem::~DependencyScanningCASFilesystem() = default;

static Expected<cas::ObjectRef>
storeDepDirectives(cas::ObjectStore &CAS,
                   ArrayRef<dependency_directives_scan::Directive> Directives) {
  llvm::SmallString<1024> Buffer;
  llvm::raw_svector_ostream OS(Buffer);
  llvm::support::endian::Writer W(OS, llvm::endianness::little);
  size_t NumTokens = 0;
  for (const auto &Directive : Directives)
    NumTokens += Directive.Tokens.size();
  W.write(NumTokens);
  for (const auto &Directive : Directives) {
    for (const auto &Token : Directive.Tokens) {
      W.write(Token.Offset);
      W.write(Token.Length);
      W.write(static_cast<std::underlying_type<decltype(Token.Kind)>::type>(
          Token.Kind));
      W.write(Token.Flags);
    }
  }

  size_t TokenIdx = 0;
  W.write(Directives.size());
  for (const auto &Directive : Directives) {
    W.write(static_cast<std::underlying_type<decltype(Directive.Kind)>::type>(
        Directive.Kind));
    W.write(TokenIdx);
    W.write(Directive.Tokens.size());
    TokenIdx += Directive.Tokens.size();
  }

  return CAS.storeFromString({}, Buffer);
}

template <typename T> static void readle(StringRef &Slice, T &Out) {
  using namespace llvm::support::endian;
  if (Slice.size() < sizeof(T))
    llvm::report_fatal_error("buffer too small");
  Out = read<T, llvm::endianness::little>(Slice.begin());
  Slice = Slice.drop_front(sizeof(T));
}

static Error loadDepDirectives(
    cas::ObjectStore &CAS, cas::ObjectRef Ref,
    llvm::SmallVectorImpl<dependency_directives_scan::Token> &DepTokens,
    llvm::SmallVectorImpl<dependency_directives_scan::Directive>
        &DepDirectives) {
  using namespace dependency_directives_scan;
  auto Blob = CAS.getProxy(Ref);
  if (!Blob)
    return Blob.takeError();

  StringRef Data = Blob->getData();
  StringRef Cursor = Data;

  size_t NumTokens = 0;
  readle(Cursor, NumTokens);
  DepTokens.reserve(NumTokens);
  for (size_t I = 0; I < NumTokens; ++I) {
    DepTokens.emplace_back(0, 0, (tok::TokenKind)0, 0);
    auto &Token = DepTokens.back();
    readle(Cursor, Token.Offset);
    readle(Cursor, Token.Length);
    std::underlying_type<decltype(Token.Kind)>::type Kind;
    readle(Cursor, Kind);
    Token.Kind = static_cast<tok::TokenKind>(Kind);
    readle(Cursor, Token.Flags);
  }

  size_t NumDirectives = 0;
  readle(Cursor, NumDirectives);
  DepDirectives.reserve(NumDirectives);
  for (size_t I = 0; I < NumDirectives; ++I) {
    std::underlying_type<DirectiveKind>::type Kind;
    readle(Cursor, Kind);
    size_t TokenStart, NumTokens;
    readle(Cursor, TokenStart);
    readle(Cursor, NumTokens);
    assert(NumTokens <= DepTokens.size() &&
           TokenStart <= DepTokens.size() - NumTokens);
    DepDirectives.emplace_back(
        static_cast<DirectiveKind>(Kind),
        ArrayRef<Token>(DepTokens.begin() + TokenStart,
                        DepTokens.begin() + TokenStart + NumTokens));
  }
  assert(Cursor.empty());
  return Error::success();
}

void DependencyScanningCASFilesystem::scanForDirectives(
    llvm::cas::ObjectRef InputDataID, StringRef Identifier,
    SmallVectorImpl<dependency_directives_scan::Token> &Tokens,
    SmallVectorImpl<dependency_directives_scan::Directive> &Directives) {
  using namespace llvm;
  using namespace llvm::cas;

  // Get a blob for the clang version string.
  if (!ClangFullVersionID)
    ClangFullVersionID =
        reportAsFatalIfError(CAS.storeFromString({}, getClangFullVersion()));

  // Get a blob for the dependency directives scan command.
  if (!DepDirectivesID)
    DepDirectivesID =
        reportAsFatalIfError(CAS.storeFromString({}, "directives"));

  // Get an empty blob.
  if (!EmptyBlobID)
    EmptyBlobID = reportAsFatalIfError(CAS.storeFromString({}, ""));

  // Construct a tree for the input.
  std::optional<CASID> InputID;
  {
    HierarchicalTreeBuilder Builder;
    Builder.push(*ClangFullVersionID, TreeEntry::Regular, "version");
    Builder.push(*DepDirectivesID, TreeEntry::Regular, "command");
    Builder.push(InputDataID, TreeEntry::Regular, "data");
    InputID = reportAsFatalIfError(Builder.create(CAS)).getID();
  }

  // Check the result cache.
  if (std::optional<CASID> OutputID =
          reportAsFatalIfError(Cache.get(*InputID))) {
    if (std::optional<ObjectRef> OutputRef = CAS.getReference(*OutputID)) {
      if (OutputRef == EmptyBlobID)
        return; // Cached directive scanning failure.
      reportAsFatalIfError(
          loadDepDirectives(CAS, *OutputRef, Tokens, Directives));
      return;
    }
  }

  StringRef InputData =
      reportAsFatalIfError(CAS.getProxy(InputDataID)).getData();

  if (scanSourceForDependencyDirectives(InputData, Tokens, Directives)) {
    // FIXME: Propagate the diagnostic if desired by the client.
    // Failure. Cache empty directives.
    Tokens.clear();
    Directives.clear();
    reportAsFatalIfError(Cache.put(*InputID, CAS.getID(*EmptyBlobID)));
    return;
  }

  // Success. Add to the CAS and get back persistent output data.
  cas::ObjectRef DirectivesID =
      reportAsFatalIfError(storeDepDirectives(CAS, Directives));
  // Cache the computation.
  reportAsFatalIfError(Cache.put(*InputID, CAS.getID(DirectivesID)));
}

Expected<StringRef>
DependencyScanningCASFilesystem::getOriginal(cas::CASID InputDataID) {
  Expected<cas::ObjectProxy> Blob = CAS.getProxy(InputDataID);
  if (Blob)
    return Blob->getData();
  return Blob.takeError();
}

static bool shouldCacheStatFailures(StringRef Filename) {
  StringRef Ext = llvm::sys::path::extension(Filename);
  if (Ext.empty())
    return false; // This may be the module cache directory.
  return true;
}

llvm::cas::CachingOnDiskFileSystem &
DependencyScanningCASFilesystem::getCachingFS() {
  return static_cast<llvm::cas::CachingOnDiskFileSystem &>(*FS);
}

DependencyScanningCASFilesystem::LookupPathResult
DependencyScanningCASFilesystem::lookupPath(const Twine &Path) {
  SmallString<256> PathStorage;
  StringRef PathRef = Path.toStringRef(PathStorage);

  {
    auto I = Entries.find(PathRef);
    if (I != Entries.end()) {
      // FIXME: Gross hack to ensure this file gets tracked as part of the
      // compilation. Instead, we should add an explicit hook somehow /
      // somewhere.
      (void)getCachingFS().status(PathRef);
      return LookupPathResult{&I->second, std::error_code()};
    }
  }

  std::optional<cas::CASID> FileID;
  llvm::ErrorOr<llvm::vfs::Status> MaybeStatus =
      getCachingFS().statusAndFileID(PathRef, FileID);
  if (!MaybeStatus) {
    if (shouldCacheStatFailures(PathRef))
      Entries[PathRef].EC = MaybeStatus.getError();
    return LookupPathResult{nullptr, MaybeStatus.getError()};
  }

  // Underlying file system caches directories. No need to duplicate.
  if (MaybeStatus->isDirectory())
    return LookupPathResult{nullptr, std::move(MaybeStatus)};

  auto &Entry = Entries[PathRef];
  Entry.CASContents = CAS.getReference(*FileID);
  llvm::ErrorOr<StringRef> Buffer = expectedToErrorOr(getOriginal(*FileID));
  if (!Buffer) {
    // Cache CAS failures. Not going to recover later.
    Entry.EC = Buffer.getError();
    return LookupPathResult{&Entry, std::error_code()};
  }

  Entry.Buffer = std::move(*Buffer);
  Entry.Status = llvm::vfs::Status(
      PathRef, MaybeStatus->getUniqueID(),
      MaybeStatus->getLastModificationTime(), MaybeStatus->getUser(),
      MaybeStatus->getGroup(), Entry.Buffer->size(), MaybeStatus->getType(),
      MaybeStatus->getPermissions());
  return LookupPathResult{&Entry, std::error_code()};
}

llvm::ErrorOr<llvm::vfs::Status>
DependencyScanningCASFilesystem::status(const Twine &Path) {
  LookupPathResult Result = lookupPath(Path);
  if (!Result.Entry)
    return std::move(Result.Status);
  if (Result.Entry->EC)
    return Result.Entry->EC;
  return Result.Entry->Status;
}

bool DependencyScanningCASFilesystem::exists(const Twine &Path) {
  // Existence check does not require caching the result at the dependency
  // scanning level. The CachingOnDiskFileSystem tracks the exists call, which
  // ensures it is included in any resulting CASFileSystem.
  return getCachingFS().exists(Path);
}

IntrusiveRefCntPtr<llvm::cas::CASBackedFileSystem>
DependencyScanningCASFilesystem::createThreadSafeProxyFS() {
  llvm::report_fatal_error("not implemented");
}

namespace {

class DepScanFile final : public llvm::cas::CASBackedFile {
public:
  DepScanFile(StringRef Buffer, cas::ObjectRef CASContents,
              llvm::vfs::Status Stat)
      : Buffer(Buffer), CASContents(CASContents), Stat(std::move(Stat)) {}

  llvm::ErrorOr<llvm::vfs::Status> status() override { return Stat; }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t FileSize, bool RequiresNullTerminator,
            bool IsVolatile) override {
    SmallString<256> Storage;
    return llvm::MemoryBuffer::getMemBuffer(Buffer, Name.toStringRef(Storage));
  }

  cas::ObjectRef getObjectRefForContent() override { return CASContents; }

  std::error_code close() override { return {}; }

private:
  StringRef Buffer;
  cas::ObjectRef CASContents;
  llvm::vfs::Status Stat;
};

} // end anonymous namespace

Expected<std::unique_ptr<llvm::cas::CASBackedFile>>
DependencyScanningCASFilesystem::openCASBackedFileForRead(const Twine &Path) {
  LookupPathResult Result = lookupPath(Path);
  if (!Result.Entry) {
    if (std::error_code EC = Result.Status.getError())
      return llvm::errorCodeToError(EC);
    assert(Result.Status->getType() ==
           llvm::sys::fs::file_type::directory_file);
    return llvm::createFileError(
        Path, std::make_error_code(std::errc::is_a_directory));
  }
  if (Result.Entry->EC)
    return llvm::errorCodeToError(Result.Entry->EC);

  assert(Result.Entry->CASContents);
  return std::make_unique<DepScanFile>(
      *Result.Entry->Buffer, *Result.Entry->CASContents, Result.Entry->Status);
}

std::optional<ArrayRef<dependency_directives_scan::Directive>>
DependencyScanningCASFilesystem::getDirectiveTokens(const Twine &Path) {
  LookupPathResult Result = lookupPath(Path);

  if (Result.Entry) {
    if (Result.Entry->DepDirectives.empty()) {
      SmallString<256> PathStorage;
      StringRef PathRef = Path.toStringRef(PathStorage);
      FileEntry &Entry = const_cast<FileEntry &>(*Result.Entry);
      scanForDirectives(*Entry.CASContents, PathRef, Entry.DepTokens,
                        Entry.DepDirectives);
    }

    if (!Result.Entry->DepDirectives.empty())
      return ArrayRef(Result.Entry->DepDirectives);
  }
  return std::nullopt;
}
