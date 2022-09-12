//===--- CASDependencyCollector.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CASDependencyCollector.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "llvm/CAS/CASOutputBackend.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/VirtualOutputBackends.h"

using namespace clang;
using namespace clang::cas;

/// \returns \p DependencyOutputOptions but with \p ExtraDeps cleared out.
///
/// This is useful to avoid recording these filenames in the CAS.
static DependencyOutputOptions dropExtraDeps(DependencyOutputOptions Opts) {
  Opts.ExtraDeps.clear();
  return Opts;
}

CASDependencyCollector::CASDependencyCollector(
    DependencyOutputOptions Opts, cas::ObjectStore &CAS,
    std::function<void(Optional<cas::ObjectRef>)> Callback)
    : DependencyFileGenerator(dropExtraDeps(std::move(Opts)),
                              llvm::vfs::makeNullOutputBackend()),
      CAS(CAS), Callback(std::move(Callback)) {}

llvm::Error CASDependencyCollector::replay(const DependencyOutputOptions &Opts,
                                           ObjectStore &CAS, ObjectRef DepsRef,
                                           llvm::raw_ostream &OS) {
  auto Refs = CAS.getProxy(DepsRef);
  if (!Refs)
    return Refs.takeError();

  CASDependencyCollector DC(Opts, CAS, nullptr);

  // Add the filenames from DependencyOutputOptions::ExtraDeps. These are kept
  // out of the compilation cache key since they can be passed-in and added at
  // the point where the dependency file is generated, without needing to affect
  // the cached compilation.
  for (const std::pair<std::string, ExtraDepKind> &Dep : Opts.ExtraDeps) {
    DC.addDependency(Dep.first);
  }

  auto Err = Refs->forEachReference([&](ObjectRef Ref) -> llvm::Error {
    auto PathHandle = CAS.getProxy(Ref);
    if (!PathHandle)
      return PathHandle.takeError();
    StringRef Path = PathHandle->getData();
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
    Refs.push_back(*Handle);
  }

  auto Handle = CAS.store(Refs, {});
  if (Handle)
    Callback(*Handle);
  else
    Diags.Report({}, diag::err_cas_store) << toString(Handle.takeError());
}
