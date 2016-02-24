//===--- ModuleDependencyCollector.cpp - Collect module dependencies ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect the dependencies of a set of modules.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
/// Private implementation for ModuleDependencyCollector
class ModuleDependencyListener : public ASTReaderListener {
  ModuleDependencyCollector &Collector;

  std::error_code copyToRoot(StringRef Src);
public:
  ModuleDependencyListener(ModuleDependencyCollector &Collector)
      : Collector(Collector) {}
  bool needsInputFileVisitation() override { return true; }
  bool needsSystemInputFileVisitation() override { return true; }
  bool visitInputFile(StringRef Filename, bool IsSystem, bool IsOverridden,
                      bool IsExplicitModule) override;
};
}

void ModuleDependencyCollector::attachToASTReader(ASTReader &R) {
  R.addListener(llvm::make_unique<ModuleDependencyListener>(*this));
}

void ModuleDependencyCollector::writeFileMap() {
  if (Seen.empty())
    return;

  SmallString<256> Dest = getDest();
  llvm::sys::path::append(Dest, "vfs.yaml");

  std::error_code EC;
  llvm::raw_fd_ostream OS(Dest, EC, llvm::sys::fs::F_Text);
  if (EC) {
    setHasErrors();
    return;
  }
  VFSWriter.write(OS);
}

std::error_code ModuleDependencyListener::copyToRoot(StringRef Src) {
  using namespace llvm::sys;

  // We need an absolute path to append to the root.
  SmallString<256> AbsoluteSrc = Src;
  fs::make_absolute(AbsoluteSrc);
  // Canonicalize to a native path to avoid mixed separator styles.
  path::native(AbsoluteSrc);
  // TODO: We probably need to handle .. as well as . in order to have valid
  // input to the YAMLVFSWriter.
  path::remove_dots(AbsoluteSrc);

  // Build the destination path.
  SmallString<256> Dest = Collector.getDest();
  path::append(Dest, path::relative_path(AbsoluteSrc));

  // Copy the file into place.
  if (std::error_code EC = fs::create_directories(path::parent_path(Dest),
                                                   /*IgnoreExisting=*/true))
    return EC;
  if (std::error_code EC = fs::copy_file(AbsoluteSrc, Dest))
    return EC;
  // Use the absolute path under the root for the file mapping.
  Collector.addFileMapping(AbsoluteSrc, Dest);
  return std::error_code();
}

bool ModuleDependencyListener::visitInputFile(StringRef Filename, bool IsSystem,
                                              bool IsOverridden,
                                              bool IsExplicitModule) {
  if (Collector.insertSeen(Filename))
    if (copyToRoot(Filename))
      Collector.setHasErrors();
  return true;
}
