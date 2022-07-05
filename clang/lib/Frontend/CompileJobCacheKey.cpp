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
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/CAS/Utils.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;
using namespace llvm::cas;

llvm::cas::CASID
clang::createCompileJobCacheKey(CASDB &CAS, ArrayRef<const char *> CC1Args,
                                llvm::cas::CASID FileSystemRootID) {
  Optional<llvm::cas::ObjectRef> RootRef = CAS.getReference(FileSystemRootID);
  if (!RootRef)
    report_fatal_error(
        createStringError(inconvertibleErrorCode(),
                          "cannot handle unknown compile-job cache key: " +
                              FileSystemRootID.toString()));
  assert(!CC1Args.empty() && StringRef(CC1Args[0]) == "-cc1");
  SmallString<256> CommandLine;
  for (StringRef Arg : CC1Args) {
    CommandLine.append(Arg);
    CommandLine.push_back(0);
  }

  llvm::cas::HierarchicalTreeBuilder Builder;
  Builder.push(*RootRef, llvm::cas::TreeEntry::Tree, "filesystem");
  Builder.push(
      CAS.getReference(llvm::cantFail(CAS.storeFromString(None, CommandLine))),
      llvm::cas::TreeEntry::Regular, "command-line");
  Builder.push(
      CAS.getReference(llvm::cantFail(CAS.storeFromString(None, "-cc1"))),
      llvm::cas::TreeEntry::Regular, "computation");

  // FIXME: The version is maybe insufficient...
  Builder.push(CAS.getReference(llvm::cantFail(
                   CAS.storeFromString(None, getClangFullVersion()))),
               llvm::cas::TreeEntry::Regular, "version");

  return CAS.getID(llvm::cantFail(Builder.create(CAS)));
}

Optional<llvm::cas::CASID>
clang::createCompileJobCacheKey(CASDB &CAS, DiagnosticsEngine &Diags,
                                const CompilerInvocation &Invocation) {
  // Generate a new command-line in case Invocation has been canonicalized.
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  llvm::SmallVector<const char *> Argv;
  Argv.push_back("-cc1");
  Invocation.generateCC1CommandLine(
      Argv, [&Saver](const llvm::Twine &T) { return Saver.save(T).data(); });

  // FIXME: currently correct since the main executable is always in the root
  // from scanning, but we should probably make it explicit here...
  StringRef RootIDString = Invocation.getFileSystemOpts().CASFileSystemRootID;
  Expected<llvm::cas::CASID> RootID = CAS.parseID(RootIDString);
  if (!RootID) {
    llvm::consumeError(RootID.takeError());
    Diags.Report(diag::err_cas_cannot_parse_root_id) << RootIDString;
    return None;
  }

  return createCompileJobCacheKey(CAS, Argv, *RootID);
}

static Error printFileSystem(CASDB &CAS, ObjectRef Ref, raw_ostream &OS) {
  Expected<ObjectHandle> Root = CAS.load(Ref);
  if (!Root)
    return Root.takeError();

  TreeSchema Schema(CAS);
  return Schema.walkFileTreeRecursively(
      CAS, *Root,
      [&](const NamedTreeEntry &Entry, Optional<TreeProxy> Tree) {
        if (Entry.getKind() != TreeEntry::Tree || Tree->empty()) {
          OS << "\n  ";
          Entry.print(OS, CAS);
        }
        return Error::success();
      });
}

static Error printCompileJobCacheKey(CASDB &CAS, ObjectHandle Node,
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

Error clang::printCompileJobCacheKey(CASDB &CAS, CASID Key, raw_ostream &OS) {
  auto H = CAS.load(Key);
  if (!H)
    return H.takeError();
  if (!*H)
    return createStringError(inconvertibleErrorCode(),
                             "cache key not present in CAS");
  TreeSchema Schema(CAS);
  if (!Schema.isNode(**H)) {
    std::string ErrStr;
    llvm::raw_string_ostream Err(ErrStr);
    Err << "expected cache key to be a CAS tree; got ";
    (*H)->print(Err);
    return createStringError(inconvertibleErrorCode(), Err.str());
  }
  return ::printCompileJobCacheKey(CAS, **H, OS);
}
