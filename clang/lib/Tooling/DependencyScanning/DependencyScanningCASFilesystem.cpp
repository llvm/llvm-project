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
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
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
    IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> WorkerFS)
    : FS(WorkerFS), Entries(EntryAlloc), CAS(WorkerFS->getCAS()) {}

DependencyScanningCASFilesystem::~DependencyScanningCASFilesystem() = default;

static Expected<cas::ObjectRef>
storeDepDirectives(cas::CASDB &CAS,
                   ArrayRef<dependency_directives_scan::Directive> Directives) {
  llvm::SmallString<1024> Buffer;
  llvm::raw_svector_ostream OS(Buffer);
  llvm::support::endian::Writer W(OS, llvm::support::endianness::little);
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

  auto Blob = CAS.createProxy(None, Buffer);
  if (!Blob)
    return Blob.takeError();
  return Blob->getRef();
}

template <typename T> static void readle(StringRef &Slice, T &Out) {
  using namespace llvm::support::endian;
  if (Slice.size() < sizeof(T))
    llvm::report_fatal_error("buffer too small");
  Out = read<T, llvm::support::little>(Slice.begin());
  Slice = Slice.drop_front(sizeof(T));
}

static Error loadDepDirectives(
    cas::CASDB &CAS, cas::CASID ID,
    llvm::SmallVectorImpl<dependency_directives_scan::Token> &DepTokens,
    llvm::SmallVectorImpl<dependency_directives_scan::Directive>
        &DepDirectives) {
  using namespace dependency_directives_scan;
  auto Blob = CAS.getProxy(ID);
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
        reportAsFatalIfError(CAS.createProxy(None, getClangFullVersion()))
            .getRef();

  // Get a blob for the dependency directives scan command.
  if (!DepDirectivesID)
    DepDirectivesID =
        reportAsFatalIfError(CAS.createProxy(None, "directives")).getRef();

  // Get an empty blob.
  if (!EmptyBlobID)
    EmptyBlobID = reportAsFatalIfError(CAS.createProxy(None, "")).getRef();

  // Construct a tree for the input.
  Optional<CASID> InputID;
  {
    HierarchicalTreeBuilder Builder;
    Builder.push(*ClangFullVersionID, TreeEntry::Regular, "version");
    Builder.push(*DepDirectivesID, TreeEntry::Regular, "command");
    Builder.push(InputDataID, TreeEntry::Regular, "data");
    InputID =
        CAS.getID(CAS.getReference(reportAsFatalIfError(Builder.create(CAS))));
  }

  // Check the result cache.
  if (Optional<CASID> OutputID =
          expectedToOptional(CAS.getCachedResult(*InputID))) {
    reportAsFatalIfError(loadDepDirectives(CAS, *OutputID, Tokens, Directives));
    return;
  }

  StringRef InputData =
      reportAsFatalIfError(CAS.getProxy(InputDataID)).getData();

  if (scanSourceForDependencyDirectives(InputData, Tokens, Directives)) {
    // FIXME: Propagate the diagnostic if desired by the client.
    // Failure. Cache empty directives.
    Tokens.clear();
    Directives.clear();
    reportAsFatalIfError(
        CAS.putCachedResult(*InputID, CAS.getID(*EmptyBlobID)));
    return;
  }

  // Success. Add to the CAS and get back persistent output data.
  cas::CASID DirectivesID =
      CAS.getID(reportAsFatalIfError(storeDepDirectives(CAS, Directives)));
  // Cache the computation.
  reportAsFatalIfError(CAS.putCachedResult(*InputID, DirectivesID));
}

Expected<StringRef>
DependencyScanningCASFilesystem::getOriginal(cas::CASID InputDataID) {
  Expected<cas::ObjectProxy> Blob = CAS.getProxy(InputDataID);
  if (Blob)
    return Blob->getData();
  return Blob.takeError();
}

/// Whitelist file extensions that should be minimized, treating no extension as
/// a source file that should be minimized.
///
/// This is kinda hacky, it would be better if we knew what kind of file Clang
/// was expecting instead.
static bool shouldScanForDirectivesBasedOnExtension(StringRef Filename) {
  StringRef Ext = llvm::sys::path::extension(Filename);
  if (Ext.empty())
    return true; // C++ standard library
  return llvm::StringSwitch<bool>(Ext)
      .CasesLower(".c", ".cc", ".cpp", ".c++", ".cxx", true)
      .CasesLower(".h", ".hh", ".hpp", ".h++", ".hxx", true)
      .CasesLower(".m", ".mm", true)
      .CasesLower(".i", ".ii", ".mi", ".mmi", true)
      .CasesLower(".def", ".inc", true)
      .Default(false);
}

static bool shouldCacheStatFailures(StringRef Filename) {
  StringRef Ext = llvm::sys::path::extension(Filename);
  if (Ext.empty())
    return false; // This may be the module cache directory.
  return shouldScanForDirectivesBasedOnExtension(
      Filename); // Only cache stat failures on source files.
}

bool DependencyScanningCASFilesystem::shouldScanForDirectives(
    StringRef RawFilename) {
  return shouldScanForDirectivesBasedOnExtension(RawFilename);
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

  Optional<cas::CASID> FileID;
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
  Entry.ID = *FileID;
  llvm::ErrorOr<StringRef> Buffer = expectedToErrorOr(getOriginal(*FileID));
  if (!Buffer) {
    // Cache CAS failures. Not going to recover later.
    Entry.EC = Buffer.getError();
    return LookupPathResult{&Entry, std::error_code()};
  }

  if (shouldScanForDirectives(PathRef))
    scanForDirectives(*CAS.getReference(*FileID), PathRef, Entry.DepTokens,
                      Entry.DepDirectives);

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

Optional<llvm::cas::CASID>
DependencyScanningCASFilesystem::getFileCASID(const Twine &Path) {
  LookupPathResult Result = lookupPath(Path);
  if (!Result.Entry)
    return None;
  if (Result.Entry->EC)
    return None;
  assert(Result.Entry->ID);
  return Result.Entry->ID;
}

IntrusiveRefCntPtr<llvm::cas::ThreadSafeFileSystem>
DependencyScanningCASFilesystem::createThreadSafeProxyFS() {
  llvm::report_fatal_error("not implemented");
}

namespace {

class DepScanFile final : public llvm::vfs::File {
public:
  DepScanFile(StringRef Buffer, llvm::vfs::Status Stat)
      : Buffer(Buffer), Stat(std::move(Stat)) {}

  llvm::ErrorOr<llvm::vfs::Status> status() override { return Stat; }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t FileSize, bool RequiresNullTerminator,
            bool IsVolatile) override {
    SmallString<256> Storage;
    return llvm::MemoryBuffer::getMemBuffer(Buffer, Name.toStringRef(Storage));
  }

  std::error_code close() override { return {}; }

private:
  StringRef Buffer;
  llvm::vfs::Status Stat;
};

} // end anonymous namespace

llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
DependencyScanningCASFilesystem::openFileForRead(const Twine &Path) {
  LookupPathResult Result = lookupPath(Path);
  if (!Result.Entry) {
    if (std::error_code EC = Result.Status.getError())
      return EC;
    assert(Result.Status->getType() ==
           llvm::sys::fs::file_type::directory_file);
    return std::make_error_code(std::errc::is_a_directory);
  }
  if (Result.Entry->EC)
    return Result.Entry->EC;

  return std::make_unique<DepScanFile>(*Result.Entry->Buffer,
                                       Result.Entry->Status);
}

Optional<ArrayRef<dependency_directives_scan::Directive>>
DependencyScanningCASFilesystem::getDirectiveTokens(const Twine &Path) const {
  SmallString<256> PathStorage;
  StringRef PathRef = Path.toStringRef(PathStorage);
  auto I = Entries.find(PathRef);
  if (I == Entries.end())
    return None;

  const FileEntry &Entry = I->second;
  if (Entry.DepDirectives.empty())
    return None;
  return llvm::makeArrayRef(Entry.DepDirectives);
}
