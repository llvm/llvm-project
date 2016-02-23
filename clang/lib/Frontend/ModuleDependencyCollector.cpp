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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Config/config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
/// Private implementation for ModuleDependencyCollector
class ModuleDependencyListener : public ASTReaderListener {
  ModuleDependencyCollector &Collector;
  llvm::StringMap<std::string> SymLinkMap;

  bool getRealPath(StringRef SrcPath, SmallVectorImpl<char> &Result);
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

// TODO: move this to Support/Path.h?
static bool real_path(StringRef SrcPath, SmallVectorImpl<char> &RealPath) {
#ifdef HAVE_REALPATH
  char CanonicalPath[PATH_MAX];

  // TODO: emit a warning in case this fails...?
  if (!realpath(SrcPath.str().c_str(), CanonicalPath))
    return false;

  SmallString<256> RPath(CanonicalPath);
  RealPath.swap(RPath);
  return true;
#else
  // FIXME: Add support for systems without realpath.
  return false;
#endif
}

bool ModuleDependencyListener::getRealPath(StringRef SrcPath,
                                           SmallVectorImpl<char> &Result) {
  using namespace llvm::sys;
  SmallString<256> RealPath;
  StringRef FileName = path::filename(SrcPath);
  std::string Dir = path::parent_path(SrcPath).str();
  auto DirWithSymLink = SymLinkMap.find(Dir);

  // Use real_path to fix any symbolic link component present in a path.
  // Computing the real path is expensive, cache the search through the
  // parent path directory.
  if (DirWithSymLink == SymLinkMap.end()) {
    if (!real_path(Dir, RealPath))
      return false;
    SymLinkMap[Dir] = RealPath.str();
  } else {
    RealPath = DirWithSymLink->second;
  }

  path::append(RealPath, FileName);
  Result.swap(RealPath);
  return true;
}

std::error_code ModuleDependencyListener::copyToRoot(StringRef Src) {
  using namespace llvm::sys;

  // We need an absolute path to append to the root.
  SmallString<256> AbsoluteSrc = Src;
  fs::make_absolute(AbsoluteSrc);
  // Canonicalize to a native path to avoid mixed separator styles.
  path::native(AbsoluteSrc);
  // Remove redundant leading "./" pieces and consecutive separators.
  AbsoluteSrc = path::remove_leading_dotslash(AbsoluteSrc);

  // Canonicalize path by removing "..", "." components.
  SmallString<256> CanonicalPath = AbsoluteSrc;
  path::remove_dots(CanonicalPath, /*remove_dot_dot=*/true);

  // If a ".." component is present after a symlink component, remove_dots may
  // lead to the wrong real destination path. Let the source be canonicalized
  // like that but make sure the destination uses the real path.
  bool HasDotDotInPath =
      std::count(path::begin(AbsoluteSrc), path::end(AbsoluteSrc), "..") > 0;
  SmallString<256> RealPath;
  bool HasRemovedSymlinkComponent = HasDotDotInPath &&
                             getRealPath(AbsoluteSrc, RealPath) &&
                             !StringRef(CanonicalPath).equals(RealPath);

  // Build the destination path.
  SmallString<256> Dest = Collector.getDest();
  path::append(Dest, path::relative_path(HasRemovedSymlinkComponent ? RealPath
                                                             : CanonicalPath));

  // Copy the file into place.
  if (std::error_code EC = fs::create_directories(path::parent_path(Dest),
                                                   /*IgnoreExisting=*/true))
    return EC;
  if (std::error_code EC = fs::copy_file(
          HasRemovedSymlinkComponent ? RealPath : CanonicalPath, Dest))
    return EC;

  // Use the canonical path under the root for the file mapping. Also create
  // an additional entry for the real path.
  Collector.addFileMapping(CanonicalPath, Dest);
  if (HasRemovedSymlinkComponent)
    Collector.addFileMapping(RealPath, Dest);

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
