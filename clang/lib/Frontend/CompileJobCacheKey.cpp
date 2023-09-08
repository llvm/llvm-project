//===- CompileJobCacheKey.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompileJobCacheKey.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Basic/Version.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/CAS/Utils.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::cas;
using namespace llvm;
using namespace llvm::cas;

static llvm::cas::CASID
createCompileJobCacheKeyForArgs(ObjectStore &CAS,
                                ArrayRef<const char *> CC1Args,
                                ObjectRef RootRef, bool IsIncludeTree) {
  assert(!CC1Args.empty() && StringRef(CC1Args[0]) == "-cc1");
  SmallString<256> CommandLine;
  for (StringRef Arg : CC1Args) {
    CommandLine.append(Arg);
    CommandLine.push_back(0);
  }

  llvm::cas::HierarchicalTreeBuilder Builder;
  if (IsIncludeTree) {
    Builder.push(RootRef, llvm::cas::TreeEntry::Regular, "include-tree");
  } else {
    Builder.push(RootRef, llvm::cas::TreeEntry::Tree, "filesystem");
  }
  Builder.push(
      llvm::cantFail(CAS.storeFromString(std::nullopt, CommandLine)),
      llvm::cas::TreeEntry::Regular, "command-line");
  Builder.push(
      llvm::cantFail(CAS.storeFromString(std::nullopt, "-cc1")),
      llvm::cas::TreeEntry::Regular, "computation");

  // FIXME: The version is maybe insufficient...
  Builder.push(
      llvm::cantFail(CAS.storeFromString(std::nullopt, getClangFullVersion())),
      llvm::cas::TreeEntry::Regular, "version");

  return llvm::cantFail(Builder.create(CAS)).getID();
}

static std::optional<llvm::cas::CASID>
createCompileJobCacheKeyImpl(ObjectStore &CAS, DiagnosticsEngine &Diags,
                             CompilerInvocation CI) {
  FrontendOptions &FrontendOpts = CI.getFrontendOpts();
  DependencyOutputOptions &DepOpts = CI.getDependencyOutputOpts();

  // Keep the key independent of the paths of these outputs.
  if (!FrontendOpts.OutputFile.empty())
    FrontendOpts.OutputFile = "-";
  if (!DepOpts.OutputFile.empty())
    DepOpts.OutputFile = "-";
  // Dependency options that do not affect the list of files are canonicalized.
  if (!DepOpts.Targets.empty())
    DepOpts.Targets = {"-"};
  DepOpts.UsePhonyTargets = false;
  // These are added in when the dependency file is generated, but they don't
  // affect the actual compilation.
  DepOpts.ExtraDeps.clear();

  // Canonicalize diagnostic options.

  DiagnosticOptions &DiagOpts = CI.getDiagnosticOpts();
  // These options affect diagnostic rendering but not the cached diagnostics.
  DiagOpts.ShowLine = false;
  DiagOpts.ShowColumn = false;
  DiagOpts.ShowLocation = false;
  DiagOpts.ShowLevel = false;
  DiagOpts.AbsolutePath = false;
  DiagOpts.ShowCarets = false;
  DiagOpts.ShowFixits = false;
  DiagOpts.ShowSourceRanges = false;
  DiagOpts.ShowParseableFixits = false;
  DiagOpts.ShowPresumedLoc = false;
  DiagOpts.ShowOptionNames = false;
  DiagOpts.ShowNoteIncludeStack = false;
  DiagOpts.ShowCategories = false;
  DiagOpts.ShowColors = false;
  DiagOpts.UseANSIEscapeCodes = false;
  DiagOpts.VerifyDiagnostics = false;
  DiagOpts.setVerifyIgnoreUnexpected(DiagnosticLevelMask::None);
  // Note: ErrorLimit affects the set of diagnostics emitted, but since we do
  // not cache compilation failures, it's safe to clear here.
  DiagOpts.ErrorLimit = 0;
  DiagOpts.MacroBacktraceLimit = 0;
  DiagOpts.SnippetLineLimit = 0;
  DiagOpts.TabStop = 0;
  DiagOpts.MessageLength = 0;
  DiagOpts.DiagnosticLogFile.clear();
  DiagOpts.DiagnosticSerializationFile.clear();
  DiagOpts.VerifyPrefixes.clear();

  llvm::erase_if(DiagOpts.Remarks, [](StringRef Remark) -> bool {
    // These are intended for caching introspection, they are not cached.
    return Remark.startswith("compile-job-cache");
  });

  // Generate a new command-line in case Invocation has been canonicalized.
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  llvm::SmallVector<const char *> Argv;
  Argv.push_back("-cc1");
  CI.generateCC1CommandLine(
      Argv, [&Saver](const llvm::Twine &T) { return Saver.save(T).data(); });

  bool IsIncludeTree = !CI.getFrontendOpts().CASIncludeTreeID.empty();

  // FIXME: currently correct since the main executable is always in the root
  // from scanning, but we should probably make it explicit here...
  StringRef RootIDString = IsIncludeTree
                               ? CI.getFrontendOpts().CASIncludeTreeID
                               : CI.getFileSystemOpts().CASFileSystemRootID;
  Expected<llvm::cas::CASID> RootID = CAS.parseID(RootIDString);
  if (!RootID) {
    llvm::consumeError(RootID.takeError());
    Diags.Report(diag::err_cas_cannot_parse_root_id) << RootIDString;
    return std::nullopt;
  }
  std::optional<llvm::cas::ObjectRef> RootRef = CAS.getReference(*RootID);
  if (!RootRef) {
    Diags.Report(diag::err_cas_missing_root_id) << RootIDString;
    return std::nullopt;
  }

  return createCompileJobCacheKeyForArgs(CAS, Argv, *RootRef, IsIncludeTree);
}

static CompileJobCachingOptions
canonicalizeForCaching(llvm::cas::ObjectStore &CAS, DiagnosticsEngine &Diags,
                       CompilerInvocation &Invocation) {
  CompileJobCachingOptions Opts;
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();

  // Canonicalize settings for caching, extracting settings that affect the
  // compilation even if will clear them during the main compilation.
  FrontendOpts.CacheCompileJob = false;
  Opts.CompilationCachingServicePath =
      std::move(FrontendOpts.CompilationCachingServicePath);
  FrontendOpts.CompilationCachingServicePath.clear();
  Opts.WriteOutputAsCASID = FrontendOpts.WriteOutputAsCASID;
  FrontendOpts.WriteOutputAsCASID = false;
  Opts.DisableCachedCompileJobReplay =
      FrontendOpts.DisableCachedCompileJobReplay;
  FrontendOpts.DisableCachedCompileJobReplay = false;
  FrontendOpts.IncludeTimestamps = false;
  std::swap(Opts.PathPrefixMappings, FrontendOpts.PathPrefixMappings);

  // Hide the CAS configuration, canonicalizing it to keep the path to the
  // CAS from leaking to the compile job, where it might affecting its
  // output (e.g., in a diagnostic).
  //
  // TODO: Extract CASOptions.Path first if we need it later since it'll
  // disappear here.
  Invocation.getCASOpts().freezeConfig(Diags);

  // TODO: Canonicalize DiagnosticOptions here to be "serialized" only. Pass in
  // a hook to mirror diagnostics to stderr (when writing there), and handle
  // other outputs during replay.

  // TODO: migrate the output path changes from createCompileJobCacheKey here.
  return Opts;
}

std::optional<llvm::cas::CASID> clang::canonicalizeAndCreateCacheKey(
    llvm::cas::ObjectStore &CAS, DiagnosticsEngine &Diags,
    CompilerInvocation &Invocation, CompileJobCachingOptions &Opts) {
  Opts = canonicalizeForCaching(CAS, Diags, Invocation);
  return createCompileJobCacheKeyImpl(CAS, Diags, Invocation);
}

std::optional<llvm::cas::CASID>
clang::createCompileJobCacheKey(ObjectStore &CAS, DiagnosticsEngine &Diags,
                                const CompilerInvocation &OriginalCI) {
  CompilerInvocation CI(OriginalCI);
  (void)canonicalizeForCaching(CAS, Diags, CI);
  return createCompileJobCacheKeyImpl(CAS, Diags, std::move(CI));
}

std::optional<llvm::cas::CASID>
clang::createCompileJobCacheKey(ObjectStore &CAS, DiagnosticsEngine &Diags,
                                const CowCompilerInvocation &OriginalCI) {
  CompilerInvocation CI(OriginalCI);
  (void)canonicalizeForCaching(CAS, Diags, CI);
  return createCompileJobCacheKeyImpl(CAS, Diags, std::move(CI));
}

static Error printFileSystem(ObjectStore &CAS, ObjectRef Root,
                             raw_ostream &OS) {
  TreeSchema Schema(CAS);
  return Schema.walkFileTreeRecursively(
      CAS, Root, [&](const NamedTreeEntry &Entry, std::optional<TreeProxy> Tree) {
        if (Entry.getKind() != TreeEntry::Tree || Tree->empty()) {
          OS << "\n  ";
          Entry.print(OS, CAS);
        }
        return Error::success();
      });
}

static Error printCompileJobCacheKey(ObjectStore &CAS, ObjectProxy Node,
                                     raw_ostream &OS) {
  auto strError = [](const Twine &Err) {
    return createStringError(inconvertibleErrorCode(), Err);
  };

  TreeSchema Schema(CAS);
  Expected<TreeProxy> Tree = Schema.load(Node);
  if (!Tree)
    return Tree.takeError();

  // Not exhaustive, but quick check that this looks like a cache key.
  if (!Tree->lookup("computation"))
    return strError("cas object is not a valid cache key");

  return Tree->forEachEntry([&](const NamedTreeEntry &E) -> Error {
    OS << E.getName() << ": " << CAS.getID(E.getRef());
    if (E.getKind() == TreeEntry::Tree)
      return printFileSystem(CAS, E.getRef(), OS);

    if (E.getKind() != TreeEntry::Regular)
      return strError("expected blob for entry " + E.getName());

    if (E.getName() == "include-tree") {
      auto IncludeTree = IncludeTreeRoot::get(CAS, E.getRef());
      if (!IncludeTree)
        return IncludeTree.takeError();
      OS << '\n';
      return IncludeTree->print(OS, 2);
    }

    auto Blob = CAS.getProxy(E.getRef());
    if (!Blob)
      return Blob.takeError();

    auto Data = Blob->getData();
    if (E.getName() == "command-line") {
      StringRef Arg;
      StringRef Trailing;
      do {
        std::tie(Arg, Data) = Data.split('\0');
        if (Arg.startswith("-")) {
          OS << Trailing << "\n  " << Arg;
        } else {
          OS << " " << Arg;
        }
        Trailing = " \\";
      } while (!Data.empty());
    } else {
      OS << "\n  " << Data;
    }
    OS << "\n";
    return Error::success();
  });
}

Error clang::printCompileJobCacheKey(ObjectStore &CAS, const CASID &Key,
                                     raw_ostream &OS) {
  auto H = CAS.getProxy(Key);
  if (!H)
    return H.takeError();
  TreeSchema Schema(CAS);
  if (!Schema.isNode(*H)) {
    std::string ErrStr;
    llvm::raw_string_ostream Err(ErrStr);
    Err << "expected cache key to be a CAS tree; got ";
    H->getID().print(Err);
    return createStringError(inconvertibleErrorCode(), Err.str());
  }
  return ::printCompileJobCacheKey(CAS, *H, OS);
}
