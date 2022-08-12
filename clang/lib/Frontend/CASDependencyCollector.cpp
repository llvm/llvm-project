//===--- CASDependencyCollector.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CASDependencyCollector.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CASOutputBackend.h"
#include "llvm/Support/VirtualOutputBackends.h"

using namespace clang;
using namespace clang::cas;

CASDependencyCollector::CASDependencyCollector(
    const DependencyOutputOptions &Opts, cas::CASDB &CAS,
    std::function<void(Optional<cas::ObjectRef>)> Callback)
    : DependencyFileGenerator(Opts, llvm::vfs::makeNullOutputBackend()),
      CAS(CAS), Callback(std::move(Callback)) {}

llvm::Error CASDependencyCollector::replay(const DependencyOutputOptions &Opts,
                                           CASDB &CAS, ObjectRef DepsRef,
                                           llvm::raw_ostream &OS) {
  auto Refs = CAS.load(DepsRef);
  if (!Refs)
    return Refs.takeError();

  CASDependencyCollector DC(Opts, CAS, nullptr);
  auto Err = CAS.forEachRef(*Refs, [&](ObjectRef Ref) -> llvm::Error {
    auto PathHandle = CAS.load(Ref);
    if (!PathHandle)
      return PathHandle.takeError();
    StringRef Path = CAS.getDataString(*PathHandle);
    // This assumes the replay has the same filtering options as when it was
    // originally computed. That avoids needing to store many unnecessary paths.
    // FIXME: if a prefix map is enabled, we should remap the paths to the
    // invocation's environment.
    DC.addDependency(Path);
    return llvm::Error::success();
  });
  if (Err)
    return Err;

  DC.outputDependencyFile(OS);
  return llvm::Error::success();
}

void CASDependencyCollector::finishedMainFile(DiagnosticsEngine &Diags) {
  ArrayRef<std::string> Files = getDependencies();
  std::vector<ObjectRef> Refs;
  Refs.reserve(Files.size());
  for (StringRef File : Files) {
    auto Handle = CAS.storeFromString({}, File);
    if (!Handle) {
      Diags.Report({}, diag::err_cas_store) << toString(Handle.takeError());
      return;
    }
    Refs.push_back(CAS.getReference(*Handle));
  }

  auto Handle = CAS.store(Refs, {});
  if (Handle)
    Callback(CAS.getReference(*Handle));
  else
    Diags.Report({}, diag::err_cas_store) << toString(Handle.takeError());
}
