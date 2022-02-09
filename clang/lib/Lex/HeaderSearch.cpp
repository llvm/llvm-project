//===- HeaderSearch.cpp - Resolve Header File Locations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the DirectoryLookup and HeaderSearch interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/HeaderMap.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Capacity.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <system_error>
#include <utility>

using namespace clang;

#define DEBUG_TYPE "file-search"

ALWAYS_ENABLED_STATISTIC(NumIncluded, "Number of attempted #includes.");
ALWAYS_ENABLED_STATISTIC(
    NumMultiIncludeFileOptzn,
    "Number of #includes skipped due to the multi-include optimization.");
ALWAYS_ENABLED_STATISTIC(NumFrameworkLookups, "Number of framework lookups.");
ALWAYS_ENABLED_STATISTIC(NumSubFrameworkLookups,
                         "Number of subframework lookups.");

const IdentifierInfo *
HeaderFileInfo::getControllingMacro(ExternalPreprocessorSource *External) {
  if (ControllingMacro) {
    if (ControllingMacro->isOutOfDate()) {
      assert(External && "We must have an external source if we have a "
                         "controlling macro that is out of date.");
      External->updateOutOfDateIdentifier(
          *const_cast<IdentifierInfo *>(ControllingMacro));
    }
    return ControllingMacro;
  }

  if (!ControllingMacroID || !External)
    return nullptr;

  ControllingMacro = External->GetIdentifier(ControllingMacroID);
  return ControllingMacro;
}

ExternalHeaderFileInfoSource::~ExternalHeaderFileInfoSource() = default;

HeaderSearch::HeaderSearch(std::shared_ptr<HeaderSearchOptions> HSOpts,
                           SourceManager &SourceMgr, DiagnosticsEngine &Diags,
                           const LangOptions &LangOpts,
                           const TargetInfo *Target)
    : HSOpts(std::move(HSOpts)), Diags(Diags),
      FileMgr(SourceMgr.getFileManager()), FrameworkMap(64),
      ModMap(SourceMgr, Diags, LangOpts, Target, *this) {}

void HeaderSearch::PrintStats() {
  llvm::errs() << "\n*** HeaderSearch Stats:\n"
               << FileInfo.size() << " files tracked.\n";
  unsigned NumOnceOnlyFiles = 0;
  for (unsigned i = 0, e = FileInfo.size(); i != e; ++i)
    NumOnceOnlyFiles += (FileInfo[i].isPragmaOnce || FileInfo[i].isImport);
  llvm::errs() << "  " << NumOnceOnlyFiles << " #import/#pragma once files.\n";

  llvm::errs() << "  " << NumIncluded << " #include/#include_next/#import.\n"
               << "    " << NumMultiIncludeFileOptzn
               << " #includes skipped due to the multi-include optimization.\n";

  llvm::errs() << NumFrameworkLookups << " framework lookups.\n"
               << NumSubFrameworkLookups << " subframework lookups.\n";
}

void HeaderSearch::SetSearchPaths(
    std::vector<DirectoryLookup> dirs, unsigned int angledDirIdx,
    unsigned int systemDirIdx, bool noCurDirSearch,
    llvm::DenseMap<unsigned int, unsigned int> searchDirToHSEntry) {
  assert(angledDirIdx <= systemDirIdx && systemDirIdx <= dirs.size() &&
         "Directory indices are unordered");
  SearchDirs = std::move(dirs);
  SearchDirsUsage.assign(SearchDirs.size(), false);
  AngledDirIdx = angledDirIdx;
  SystemDirIdx = systemDirIdx;
  NoCurDirSearch = noCurDirSearch;
  SearchDirToHSEntry = std::move(searchDirToHSEntry);
  //LookupFileCache.clear();
}

void HeaderSearch::AddSearchPath(const DirectoryLookup &dir, bool isAngled) {
  unsigned idx = isAngled ? SystemDirIdx : AngledDirIdx;
  SearchDirs.insert(SearchDirs.begin() + idx, dir);
  SearchDirsUsage.insert(SearchDirsUsage.begin() + idx, false);
  if (!isAngled)
    AngledDirIdx++;
  SystemDirIdx++;
}

std::vector<bool> HeaderSearch::computeUserEntryUsage() const {
  std::vector<bool> UserEntryUsage(HSOpts->UserEntries.size());
  for (unsigned I = 0, E = SearchDirsUsage.size(); I < E; ++I) {
    // Check whether this DirectoryLookup has been successfully used.
    if (SearchDirsUsage[I]) {
      auto UserEntryIdxIt = SearchDirToHSEntry.find(I);
      // Check whether this DirectoryLookup maps to a HeaderSearch::UserEntry.
      if (UserEntryIdxIt != SearchDirToHSEntry.end())
        UserEntryUsage[UserEntryIdxIt->second] = true;
    }
  }
  return UserEntryUsage;
}

/// CreateHeaderMap - This method returns a HeaderMap for the specified
/// FileEntry, uniquing them through the 'HeaderMaps' datastructure.
const HeaderMap *HeaderSearch::CreateHeaderMap(const FileEntry *FE) {
  // We expect the number of headermaps to be small, and almost always empty.
  // If it ever grows, use of a linear search should be re-evaluated.
  if (!HeaderMaps.empty()) {
    for (unsigned i = 0, e = HeaderMaps.size(); i != e; ++i)
      // Pointer equality comparison of FileEntries works because they are
      // already uniqued by inode.
      if (HeaderMaps[i].first == FE)
        return HeaderMaps[i].second.get();
  }

  if (std::unique_ptr<HeaderMap> HM = HeaderMap::Create(FE, FileMgr)) {
    HeaderMaps.emplace_back(FE, std::move(HM));
    return HeaderMaps.back().second.get();
  }

  return nullptr;
}

/// Get filenames for all registered header maps.
void HeaderSearch::getHeaderMapFileNames(
    SmallVectorImpl<std::string> &Names) const {
  for (auto &HM : HeaderMaps)
    Names.push_back(std::string(HM.first->getName()));
}

std::string HeaderSearch::getCachedModuleFileName(Module *Module) {
  const FileEntry *ModuleMap =
      getModuleMap().getModuleMapFileForUniquing(Module);
  return getCachedModuleFileName(Module->Name, ModuleMap->getName());
}

std::string HeaderSearch::getPrebuiltModuleFileName(StringRef ModuleName,
                                                    bool FileMapOnly) {
  // First check the module name to pcm file map.
  auto i(HSOpts->PrebuiltModuleFiles.find(ModuleName));
  if (i != HSOpts->PrebuiltModuleFiles.end())
    return i->second;

  if (FileMapOnly || HSOpts->PrebuiltModulePaths.empty())
    return {};

  // Then go through each prebuilt module directory and try to find the pcm
  // file.
  for (const std::string &Dir : HSOpts->PrebuiltModulePaths) {
    SmallString<256> Result(Dir);
    llvm::sys::fs::make_absolute(Result);
    llvm::sys::path::append(Result, ModuleName + ".pcm");
    if (getFileMgr().getFile(Result.str()))
      return std::string(Result);
  }
  return {};
}

std::string HeaderSearch::getPrebuiltImplicitModuleFileName(Module *Module) {
  const FileEntry *ModuleMap =
      getModuleMap().getModuleMapFileForUniquing(Module);
  StringRef ModuleName = Module->Name;
  StringRef ModuleMapPath = ModuleMap->getName();
  StringRef ModuleCacheHash = HSOpts->DisableModuleHash ? "" : getModuleHash();
  for (const std::string &Dir : HSOpts->PrebuiltModulePaths) {
    SmallString<256> CachePath(Dir);
    llvm::sys::fs::make_absolute(CachePath);
    llvm::sys::path::append(CachePath, ModuleCacheHash);
    std::string FileName =
        getCachedModuleFileNameImpl(ModuleName, ModuleMapPath, CachePath);
    if (!FileName.empty() && getFileMgr().getFile(FileName))
      return FileName;
  }
  return {};
}

std::string HeaderSearch::getCachedModuleFileName(StringRef ModuleName,
                                                  StringRef ModuleMapPath) {
  return getCachedModuleFileNameImpl(ModuleName, ModuleMapPath,
                                     getModuleCachePath());
}

std::string HeaderSearch::getCachedModuleFileNameImpl(StringRef ModuleName,
                                                      StringRef ModuleMapPath,
                                                      StringRef CachePath) {
  // If we don't have a module cache path or aren't supposed to use one, we
  // can't do anything.
  if (CachePath.empty())
    return {};

  SmallString<256> Result(CachePath);
  llvm::sys::fs::make_absolute(Result);

  if (HSOpts->DisableModuleHash) {
    llvm::sys::path::append(Result, ModuleName + ".pcm");
  } else {
    // Construct the name <ModuleName>-<hash of ModuleMapPath>.pcm which should
    // ideally be globally unique to this particular module. Name collisions
    // in the hash are safe (because any translation unit can only import one
    // module with each name), but result in a loss of caching.
    //
    // To avoid false-negatives, we form as canonical a path as we can, and map
    // to lower-case in case we're on a case-insensitive file system.
    std::string Parent =
        std::string(llvm::sys::path::parent_path(ModuleMapPath));
    if (Parent.empty())
      Parent = ".";
    auto Dir = FileMgr.getDirectory(Parent);
    if (!Dir)
      return {};
    auto DirName = FileMgr.getCanonicalName(*Dir);
    auto FileName = llvm::sys::path::filename(ModuleMapPath);

    llvm::hash_code Hash =
      llvm::hash_combine(DirName.lower(), FileName.lower());

    SmallString<128> HashStr;
    llvm::APInt(64, size_t(Hash)).toStringUnsigned(HashStr, /*Radix*/36);
    llvm::sys::path::append(Result, ModuleName + "-" + HashStr + ".pcm");
  }
  return Result.str().str();
}

Module *HeaderSearch::lookupModule(StringRef ModuleName,
                                   SourceLocation ImportLoc, bool AllowSearch,
                                   bool AllowExtraModuleMapSearch) {
  // Look in the module map to determine if there is a module by this name.
  Module *Module = ModMap.findModule(ModuleName);
  if (Module || !AllowSearch || !HSOpts->ImplicitModuleMaps)
    return Module;

  StringRef SearchName = ModuleName;
  Module = lookupModule(ModuleName, SearchName, ImportLoc,
                        AllowExtraModuleMapSearch);

  // The facility for "private modules" -- adjacent, optional module maps named
  // module.private.modulemap that are supposed to define private submodules --
  // may have different flavors of names: FooPrivate, Foo_Private and Foo.Private.
  //
  // Foo.Private is now deprecated in favor of Foo_Private. Users of FooPrivate
  // should also rename to Foo_Private. Representing private as submodules
  // could force building unwanted dependencies into the parent module and cause
  // dependency cycles.
  if (!Module && SearchName.consume_back("_Private"))
    Module = lookupModule(ModuleName, SearchName, ImportLoc,
                          AllowExtraModuleMapSearch);
  if (!Module && SearchName.consume_back("Private"))
    Module = lookupModule(ModuleName, SearchName, ImportLoc,
                          AllowExtraModuleMapSearch);
  return Module;
}

Module *HeaderSearch::lookupModule(StringRef ModuleName, StringRef SearchName,
                                   SourceLocation ImportLoc,
                                   bool AllowExtraModuleMapSearch) {
  Module *Module = nullptr;
  unsigned Idx;

  // Look through the various header search paths to load any available module
  // maps, searching for a module map that describes this module.
  for (Idx = 0; Idx != SearchDirs.size(); ++Idx) {
    if (SearchDirs[Idx].isFramework()) {
      // Search for or infer a module map for a framework. Here we use
      // SearchName rather than ModuleName, to permit finding private modules
      // named FooPrivate in buggy frameworks named Foo.
      SmallString<128> FrameworkDirName;
      FrameworkDirName += SearchDirs[Idx].getFrameworkDir()->getName();
      llvm::sys::path::append(FrameworkDirName, SearchName + ".framework");
      if (auto FrameworkDir = FileMgr.getDirectory(FrameworkDirName)) {
        bool IsSystem
          = SearchDirs[Idx].getDirCharacteristic() != SrcMgr::C_User;
        Module = loadFrameworkModule(ModuleName, *FrameworkDir, IsSystem);
        if (Module)
          break;
      }
    }

    // FIXME: Figure out how header maps and module maps will work together.

    // Only deal with normal search directories.
    if (!SearchDirs[Idx].isNormalDir())
      continue;

    bool IsSystem = SearchDirs[Idx].isSystemHeaderDirectory();
    // Search for a module map file in this directory.
    if (loadModuleMapFile(SearchDirs[Idx].getDir(), IsSystem,
                          /*IsFramework*/false) == LMM_NewlyLoaded) {
      // We just loaded a module map file; check whether the module is
      // available now.
      Module = ModMap.findModule(ModuleName);
      if (Module)
        break;
    }

    // Search for a module map in a subdirectory with the same name as the
    // module.
    SmallString<128> NestedModuleMapDirName;
    NestedModuleMapDirName = SearchDirs[Idx].getDir()->getName();
    llvm::sys::path::append(NestedModuleMapDirName, ModuleName);
    if (loadModuleMapFile(NestedModuleMapDirName, IsSystem,
                          /*IsFramework*/false) == LMM_NewlyLoaded){
      // If we just loaded a module map file, look for the module again.
      Module = ModMap.findModule(ModuleName);
      if (Module)
        break;
    }

    // If we've already performed the exhaustive search for module maps in this
    // search directory, don't do it again.
    if (SearchDirs[Idx].haveSearchedAllModuleMaps())
      continue;

    // Load all module maps in the immediate subdirectories of this search
    // directory if ModuleName was from @import.
    if (AllowExtraModuleMapSearch)
      loadSubdirectoryModuleMaps(SearchDirs[Idx]);

    // Look again for the module.
    Module = ModMap.findModule(ModuleName);
    if (Module)
      break;
  }

  if (Module)
    noteLookupUsage(Idx, ImportLoc);

  return Module;
}

//===----------------------------------------------------------------------===//
// File lookup within a DirectoryLookup scope
//===----------------------------------------------------------------------===//

/// getName - Return the directory or filename corresponding to this lookup
/// object.
StringRef DirectoryLookup::getName() const {
  // FIXME: Use the name from \c DirectoryEntryRef.
  if (isNormalDir())
    return getDir()->getName();
  if (isFramework())
    return getFrameworkDir()->getName();
  assert(isHeaderMap() && "Unknown DirectoryLookup");
  return getHeaderMap()->getFileName();
}

Optional<FileEntryRef> HeaderSearch::getFileAndSuggestModule(
    StringRef FileName, SourceLocation IncludeLoc, const DirectoryEntry *Dir,
    bool IsSystemHeaderDir, Module *RequestingModule,
    ModuleMap::KnownHeader *SuggestedModule) {
  // If we have a module map that might map this header, load it and
  // check whether we'll have a suggestion for a module.
  auto File = getFileMgr().getFileRef(FileName, /*OpenFile=*/true);
  if (!File) {
    // For rare, surprising errors (e.g. "out of file handles"), diag the EC
    // message.
    std::error_code EC = llvm::errorToErrorCode(File.takeError());
    if (EC != llvm::errc::no_such_file_or_directory &&
        EC != llvm::errc::invalid_argument &&
        EC != llvm::errc::is_a_directory && EC != llvm::errc::not_a_directory) {
      Diags.Report(IncludeLoc, diag::err_cannot_open_file)
          << FileName << EC.message();
    }
    return None;
  }

  // If there is a module that corresponds to this header, suggest it.
  if (!findUsableModuleForHeader(
          &File->getFileEntry(), Dir ? Dir : File->getFileEntry().getDir(),
          RequestingModule, SuggestedModule, IsSystemHeaderDir))
    return None;

  return *File;
}

/// LookupFile - Lookup the specified file in this search path, returning it
/// if it exists or returning null if not.
Optional<FileEntryRef> DirectoryLookup::LookupFile(
    StringRef &Filename, HeaderSearch &HS, SourceLocation IncludeLoc,
    SmallVectorImpl<char> *SearchPath, SmallVectorImpl<char> *RelativePath,
    Module *RequestingModule, ModuleMap::KnownHeader *SuggestedModule,
    bool &InUserSpecifiedSystemFramework, bool &IsFrameworkFound,
    bool &IsInHeaderMap, SmallVectorImpl<char> &MappedName) const {
  InUserSpecifiedSystemFramework = false;
  IsInHeaderMap = false;
  MappedName.clear();

  SmallString<1024> TmpDir;
  if (isNormalDir()) {
    // Concatenate the requested file onto the directory.
    TmpDir = getDir()->getName();
    llvm::sys::path::append(TmpDir, Filename);
    if (SearchPath) {
      StringRef SearchPathRef(getDir()->getName());
      SearchPath->clear();
      SearchPath->append(SearchPathRef.begin(), SearchPathRef.end());
    }
    if (RelativePath) {
      RelativePath->clear();
      RelativePath->append(Filename.begin(), Filename.end());
    }

    return HS.getFileAndSuggestModule(TmpDir, IncludeLoc, getDir(),
                                      isSystemHeaderDirectory(),
                                      RequestingModule, SuggestedModule);
  }

  if (isFramework())
    return DoFrameworkLookup(Filename, HS, SearchPath, RelativePath,
                             RequestingModule, SuggestedModule,
                             InUserSpecifiedSystemFramework, IsFrameworkFound);

  assert(isHeaderMap() && "Unknown directory lookup");
  const HeaderMap *HM = getHeaderMap();
  SmallString<1024> Path;
  StringRef Dest = HM->lookupFilename(Filename, Path);
  if (Dest.empty())
    return None;

  IsInHeaderMap = true;

  auto FixupSearchPath = [&]() {
    if (SearchPath) {
      StringRef SearchPathRef(getName());
      SearchPath->clear();
      SearchPath->append(SearchPathRef.begin(), SearchPathRef.end());
    }
    if (RelativePath) {
      RelativePath->clear();
      RelativePath->append(Filename.begin(), Filename.end());
    }
  };

  // Check if the headermap maps the filename to a framework include
  // ("Foo.h" -> "Foo/Foo.h"), in which case continue header lookup using the
  // framework include.
  if (llvm::sys::path::is_relative(Dest)) {
    MappedName.append(Dest.begin(), Dest.end());
    Filename = StringRef(MappedName.begin(), MappedName.size());
    Dest = HM->lookupFilename(Filename, Path);
  }

  if (auto Res = HS.getFileMgr().getOptionalFileRef(Dest)) {
    FixupSearchPath();
    return *Res;
  }

  // Header maps need to be marked as used whenever the filename matches.
  // The case where the target file **exists** is handled by callee of this
  // function as part of the regular logic that applies to include search paths.
  // The case where the target file **does not exist** is handled here:
  HS.noteLookupUsage(*HS.searchDirIdx(*this), IncludeLoc);
  return None;
}

/// Given a framework directory, find the top-most framework directory.
///
/// \param FileMgr The file manager to use for directory lookups.
/// \param DirName The name of the framework directory.
/// \param SubmodulePath Will be populated with the submodule path from the
/// returned top-level module to the originally named framework.
static const DirectoryEntry *
getTopFrameworkDir(FileManager &FileMgr, StringRef DirName,
                   SmallVectorImpl<std::string> &SubmodulePath) {
  assert(llvm::sys::path::extension(DirName) == ".framework" &&
         "Not a framework directory");

  // Note: as an egregious but useful hack we use the real path here, because
  // frameworks moving between top-level frameworks to embedded frameworks tend
  // to be symlinked, and we base the logical structure of modules on the
  // physical layout. In particular, we need to deal with crazy includes like
  //
  //   #include <Foo/Frameworks/Bar.framework/Headers/Wibble.h>
  //
  // where 'Bar' used to be embedded in 'Foo', is now a top-level framework
  // which one should access with, e.g.,
  //
  //   #include <Bar/Wibble.h>
  //
  // Similar issues occur when a top-level framework has moved into an
  // embedded framework.
  const DirectoryEntry *TopFrameworkDir = nullptr;
  if (auto TopFrameworkDirOrErr = FileMgr.getDirectory(DirName))
    TopFrameworkDir = *TopFrameworkDirOrErr;

  if (TopFrameworkDir)
    DirName = FileMgr.getCanonicalName(TopFrameworkDir);
  do {
    // Get the parent directory name.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      break;

    // Determine whether this directory exists.
    auto Dir = FileMgr.getDirectory(DirName);
    if (!Dir)
      break;

    // If this is a framework directory, then we're a subframework of this
    // framework.
    if (llvm::sys::path::extension(DirName) == ".framework") {
      SubmodulePath.push_back(std::string(llvm::sys::path::stem(DirName)));
      TopFrameworkDir = *Dir;
    }
  } while (true);

  return TopFrameworkDir;
}

static bool needModuleLookup(Module *RequestingModule,
                             bool HasSuggestedModule) {
  return HasSuggestedModule ||
         (RequestingModule && RequestingModule->NoUndeclaredIncludes);
}

/// DoFrameworkLookup - Do a lookup of the specified file in the current
/// DirectoryLookup, which is a framework directory.
Optional<FileEntryRef> DirectoryLookup::DoFrameworkLookup(
    StringRef Filename, HeaderSearch &HS, SmallVectorImpl<char> *SearchPath,
    SmallVectorImpl<char> *RelativePath, Module *RequestingModule,
    ModuleMap::KnownHeader *SuggestedModule,
    bool &InUserSpecifiedSystemFramework, bool &IsFrameworkFound) const {
  FileManager &FileMgr = HS.getFileMgr();

  // Framework names must have a '/' in the filename.
  size_t SlashPos = Filename.find('/');
  if (SlashPos == StringRef::npos)
    return None;

  // Find out if this is the home for the specified framework, by checking
  // HeaderSearch.  Possible answers are yes/no and unknown.
  FrameworkCacheEntry &CacheEntry =
    HS.LookupFrameworkCache(Filename.substr(0, SlashPos));

  // If it is known and in some other directory, fail.
  if (CacheEntry.Directory && CacheEntry.Directory != getFrameworkDir())
    return None;

  // Otherwise, construct the path to this framework dir.

  // FrameworkName = "/System/Library/Frameworks/"
  SmallString<1024> FrameworkName;
  FrameworkName += getFrameworkDirRef()->getName();
  if (FrameworkName.empty() || FrameworkName.back() != '/')
    FrameworkName.push_back('/');

  // FrameworkName = "/System/Library/Frameworks/Cocoa"
  StringRef ModuleName(Filename.begin(), SlashPos);
  FrameworkName += ModuleName;

  // FrameworkName = "/System/Library/Frameworks/Cocoa.framework/"
  FrameworkName += ".framework/";

  // If the cache entry was unresolved, populate it now.
  if (!CacheEntry.Directory) {
    ++NumFrameworkLookups;

    // If the framework dir doesn't exist, we fail.
    auto Dir = FileMgr.getDirectory(FrameworkName);
    if (!Dir)
      return None;

    // Otherwise, if it does, remember that this is the right direntry for this
    // framework.
    CacheEntry.Directory = getFrameworkDir();

    // If this is a user search directory, check if the framework has been
    // user-specified as a system framework.
    if (getDirCharacteristic() == SrcMgr::C_User) {
      SmallString<1024> SystemFrameworkMarker(FrameworkName);
      SystemFrameworkMarker += ".system_framework";
      if (llvm::sys::fs::exists(SystemFrameworkMarker)) {
        CacheEntry.IsUserSpecifiedSystemFramework = true;
      }
    }
  }

  // Set out flags.
  InUserSpecifiedSystemFramework = CacheEntry.IsUserSpecifiedSystemFramework;
  IsFrameworkFound = CacheEntry.Directory;

  if (RelativePath) {
    RelativePath->clear();
    RelativePath->append(Filename.begin()+SlashPos+1, Filename.end());
  }

  // Check "/System/Library/Frameworks/Cocoa.framework/Headers/file.h"
  unsigned OrigSize = FrameworkName.size();

  FrameworkName += "Headers/";

  if (SearchPath) {
    SearchPath->clear();
    // Without trailing '/'.
    SearchPath->append(FrameworkName.begin(), FrameworkName.end()-1);
  }

  FrameworkName.append(Filename.begin()+SlashPos+1, Filename.end());

  auto File =
      FileMgr.getOptionalFileRef(FrameworkName, /*OpenFile=*/!SuggestedModule);
  if (!File) {
    // Check "/System/Library/Frameworks/Cocoa.framework/PrivateHeaders/file.h"
    const char *Private = "Private";
    FrameworkName.insert(FrameworkName.begin()+OrigSize, Private,
                         Private+strlen(Private));
    if (SearchPath)
      SearchPath->insert(SearchPath->begin()+OrigSize, Private,
                         Private+strlen(Private));

    File = FileMgr.getOptionalFileRef(FrameworkName,
                                      /*OpenFile=*/!SuggestedModule);
  }

  // If we found the header and are allowed to suggest a module, do so now.
  if (File && needModuleLookup(RequestingModule, SuggestedModule)) {
    // Find the framework in which this header occurs.
    StringRef FrameworkPath = File->getFileEntry().getDir()->getName();
    bool FoundFramework = false;
    do {
      // Determine whether this directory exists.
      auto Dir = FileMgr.getDirectory(FrameworkPath);
      if (!Dir)
        break;

      // If this is a framework directory, then we're a subframework of this
      // framework.
      if (llvm::sys::path::extension(FrameworkPath) == ".framework") {
        FoundFramework = true;
        break;
      }

      // Get the parent directory name.
      FrameworkPath = llvm::sys::path::parent_path(FrameworkPath);
      if (FrameworkPath.empty())
        break;
    } while (true);

    bool IsSystem = getDirCharacteristic() != SrcMgr::C_User;
    if (FoundFramework) {
      if (!HS.findUsableModuleForFrameworkHeader(
              &File->getFileEntry(), FrameworkPath, RequestingModule,
              SuggestedModule, IsSystem))
        return None;
    } else {
      if (!HS.findUsableModuleForHeader(&File->getFileEntry(), getDir(),
                                        RequestingModule, SuggestedModule,
                                        IsSystem))
        return None;
    }
  }
  if (File)
    return *File;
  return None;
}

void HeaderSearch::cacheLookupSuccess(LookupFileCacheInfo &CacheLookup,
                                      unsigned HitIdx, SourceLocation Loc) {
  CacheLookup.HitIdx = HitIdx;
  noteLookupUsage(HitIdx, Loc);
}

void HeaderSearch::noteLookupUsage(unsigned HitIdx, SourceLocation Loc) {
  SearchDirsUsage[HitIdx] = true;

  auto UserEntryIdxIt = SearchDirToHSEntry.find(HitIdx);
  if (UserEntryIdxIt != SearchDirToHSEntry.end())
    Diags.Report(Loc, diag::remark_pp_search_path_usage)
        << HSOpts->UserEntries[UserEntryIdxIt->second].Path;
}

void HeaderSearch::setTarget(const TargetInfo &Target) {
  ModMap.setTarget(Target);
}

//===----------------------------------------------------------------------===//
// Header File Location.
//===----------------------------------------------------------------------===//

/// Return true with a diagnostic if the file that MSVC would have found
/// fails to match the one that Clang would have found with MSVC header search
/// disabled.
static bool checkMSVCHeaderSearch(DiagnosticsEngine &Diags,
                                  const FileEntry *MSFE, const FileEntry *FE,
                                  SourceLocation IncludeLoc) {
  if (MSFE && FE != MSFE) {
    Diags.Report(IncludeLoc, diag::ext_pp_include_search_ms) << MSFE->getName();
    return true;
  }
  return false;
}

static const char *copyString(StringRef Str, llvm::BumpPtrAllocator &Alloc) {
  assert(!Str.empty());
  char *CopyStr = Alloc.Allocate<char>(Str.size()+1);
  std::copy(Str.begin(), Str.end(), CopyStr);
  CopyStr[Str.size()] = '\0';
  return CopyStr;
}

static bool isFrameworkStylePath(StringRef Path, bool &IsPrivateHeader,
                                 SmallVectorImpl<char> &FrameworkName,
                                 SmallVectorImpl<char> &IncludeSpelling) {
  using namespace llvm::sys;
  path::const_iterator I = path::begin(Path);
  path::const_iterator E = path::end(Path);
  IsPrivateHeader = false;

  // Detect different types of framework style paths:
  //
  //   ...Foo.framework/{Headers,PrivateHeaders}
  //   ...Foo.framework/Versions/{A,Current}/{Headers,PrivateHeaders}
  //   ...Foo.framework/Frameworks/Nested.framework/{Headers,PrivateHeaders}
  //   ...<other variations with 'Versions' like in the above path>
  //
  // and some other variations among these lines.
  int FoundComp = 0;
  while (I != E) {
    if (*I == "Headers") {
      ++FoundComp;
    } else if (*I == "PrivateHeaders") {
      ++FoundComp;
      IsPrivateHeader = true;
    } else if (I->endswith(".framework")) {
      StringRef Name = I->drop_back(10); // Drop .framework
      // Need to reset the strings and counter to support nested frameworks.
      FrameworkName.clear();
      FrameworkName.append(Name.begin(), Name.end());
      IncludeSpelling.clear();
      IncludeSpelling.append(Name.begin(), Name.end());
      FoundComp = 1;
    } else if (FoundComp >= 2) {
      IncludeSpelling.push_back('/');
      IncludeSpelling.append(I->begin(), I->end());
    }
    ++I;
  }

  return !FrameworkName.empty() && FoundComp >= 2;
}

static void
diagnoseFrameworkInclude(DiagnosticsEngine &Diags, SourceLocation IncludeLoc,
                         StringRef Includer, StringRef IncludeFilename,
                         const FileEntry *IncludeFE, bool isAngled = false,
                         bool FoundByHeaderMap = false) {
  bool IsIncluderPrivateHeader = false;
  SmallString<128> FromFramework, ToFramework;
  SmallString<128> FromIncludeSpelling, ToIncludeSpelling;
  if (!isFrameworkStylePath(Includer, IsIncluderPrivateHeader, FromFramework,
                            FromIncludeSpelling))
    return;
  bool IsIncludeePrivateHeader = false;
  bool IsIncludeeInFramework =
      isFrameworkStylePath(IncludeFE->getName(), IsIncludeePrivateHeader,
                           ToFramework, ToIncludeSpelling);

  if (!isAngled && !FoundByHeaderMap) {
    SmallString<128> NewInclude("<");
    if (IsIncludeeInFramework) {
      NewInclude += ToIncludeSpelling;
      NewInclude += ">";
    } else {
      NewInclude += IncludeFilename;
      NewInclude += ">";
    }
    Diags.Report(IncludeLoc, diag::warn_quoted_include_in_framework_header)
        << IncludeFilename
        << FixItHint::CreateReplacement(IncludeLoc, NewInclude);
  }

  // Headers in Foo.framework/Headers should not include headers
  // from Foo.framework/PrivateHeaders, since this violates public/private
  // API boundaries and can cause modular dependency cycles.
  if (!IsIncluderPrivateHeader && IsIncludeeInFramework &&
      IsIncludeePrivateHeader && FromFramework == ToFramework)
    Diags.Report(IncludeLoc, diag::warn_framework_include_private_from_public)
        << IncludeFilename;
}

/// LookupFile - Given a "foo" or \<foo> reference, look up the indicated file,
/// return null on failure.  isAngled indicates whether the file reference is
/// for system \#include's or not (i.e. using <> instead of ""). Includers, if
/// non-empty, indicates where the \#including file(s) are, in case a relative
/// search is needed. Microsoft mode will pass all \#including files.
Optional<FileEntryRef> HeaderSearch::LookupFile(
    StringRef Filename, SourceLocation IncludeLoc, bool isAngled,
    const DirectoryLookup *FromDir, const DirectoryLookup **CurDirArg,
    ArrayRef<std::pair<const FileEntry *, const DirectoryEntry *>> Includers,
    SmallVectorImpl<char> *SearchPath, SmallVectorImpl<char> *RelativePath,
    Module *RequestingModule, ModuleMap::KnownHeader *SuggestedModule,
    bool *IsMapped, bool *IsFrameworkFound, bool SkipCache,
    bool BuildSystemModule) {
  const DirectoryLookup *CurDirLocal = nullptr;
  const DirectoryLookup *&CurDir = CurDirArg ? *CurDirArg : CurDirLocal;

  if (IsMapped)
    *IsMapped = false;

  if (IsFrameworkFound)
    *IsFrameworkFound = false;

  if (SuggestedModule)
    *SuggestedModule = ModuleMap::KnownHeader();

  // If 'Filename' is absolute, check to see if it exists and no searching.
  if (llvm::sys::path::is_absolute(Filename)) {
    CurDir = nullptr;

    // If this was an #include_next "/absolute/file", fail.
    if (FromDir)
      return None;

    if (SearchPath)
      SearchPath->clear();
    if (RelativePath) {
      RelativePath->clear();
      RelativePath->append(Filename.begin(), Filename.end());
    }
    // Otherwise, just return the file.
    return getFileAndSuggestModule(Filename, IncludeLoc, nullptr,
                                   /*IsSystemHeaderDir*/false,
                                   RequestingModule, SuggestedModule);
  }

  // This is the header that MSVC's header search would have found.
  ModuleMap::KnownHeader MSSuggestedModule;
  Optional<FileEntryRef> MSFE;

  // Unless disabled, check to see if the file is in the #includer's
  // directory.  This cannot be based on CurDir, because each includer could be
  // a #include of a subdirectory (#include "foo/bar.h") and a subsequent
  // include of "baz.h" should resolve to "whatever/foo/baz.h".
  // This search is not done for <> headers.
  if (!Includers.empty() && !isAngled && !NoCurDirSearch) {
    SmallString<1024> TmpDir;
    bool First = true;
    for (const auto &IncluderAndDir : Includers) {
      const FileEntry *Includer = IncluderAndDir.first;

      // Concatenate the requested file onto the directory.
      // FIXME: Portability.  Filename concatenation should be in sys::Path.
      TmpDir = IncluderAndDir.second->getName();
      TmpDir.push_back('/');
      TmpDir.append(Filename.begin(), Filename.end());

      // FIXME: We don't cache the result of getFileInfo across the call to
      // getFileAndSuggestModule, because it's a reference to an element of
      // a container that could be reallocated across this call.
      //
      // If we have no includer, that means we're processing a #include
      // from a module build. We should treat this as a system header if we're
      // building a [system] module.
      bool IncluderIsSystemHeader =
          Includer ? getFileInfo(Includer).DirInfo != SrcMgr::C_User :
          BuildSystemModule;
      if (Optional<FileEntryRef> FE = getFileAndSuggestModule(
              TmpDir, IncludeLoc, IncluderAndDir.second, IncluderIsSystemHeader,
              RequestingModule, SuggestedModule)) {
        if (!Includer) {
          assert(First && "only first includer can have no file");
          return FE;
        }

        // Leave CurDir unset.
        // This file is a system header or C++ unfriendly if the old file is.
        //
        // Note that we only use one of FromHFI/ToHFI at once, due to potential
        // reallocation of the underlying vector potentially making the first
        // reference binding dangling.
        HeaderFileInfo &FromHFI = getFileInfo(Includer);
        unsigned DirInfo = FromHFI.DirInfo;
        bool IndexHeaderMapHeader = FromHFI.IndexHeaderMapHeader;
        StringRef Framework = FromHFI.Framework;

        HeaderFileInfo &ToHFI = getFileInfo(&FE->getFileEntry());
        ToHFI.DirInfo = DirInfo;
        ToHFI.IndexHeaderMapHeader = IndexHeaderMapHeader;
        ToHFI.Framework = Framework;

        if (SearchPath) {
          StringRef SearchPathRef(IncluderAndDir.second->getName());
          SearchPath->clear();
          SearchPath->append(SearchPathRef.begin(), SearchPathRef.end());
        }
        if (RelativePath) {
          RelativePath->clear();
          RelativePath->append(Filename.begin(), Filename.end());
        }
        if (First) {
          diagnoseFrameworkInclude(Diags, IncludeLoc,
                                   IncluderAndDir.second->getName(), Filename,
                                   &FE->getFileEntry());
          return FE;
        }

        // Otherwise, we found the path via MSVC header search rules.  If
        // -Wmsvc-include is enabled, we have to keep searching to see if we
        // would've found this header in -I or -isystem directories.
        if (Diags.isIgnored(diag::ext_pp_include_search_ms, IncludeLoc)) {
          return FE;
        } else {
          MSFE = FE;
          if (SuggestedModule) {
            MSSuggestedModule = *SuggestedModule;
            *SuggestedModule = ModuleMap::KnownHeader();
          }
          break;
        }
      }
      First = false;
    }
  }

  CurDir = nullptr;

  // If this is a system #include, ignore the user #include locs.
  unsigned i = isAngled ? AngledDirIdx : 0;

  // If this is a #include_next request, start searching after the directory the
  // file was found in.
  if (FromDir)
    i = FromDir-&SearchDirs[0];

  // Cache all of the lookups performed by this method.  Many headers are
  // multiply included, and the "pragma once" optimization prevents them from
  // being relex/pp'd, but they would still have to search through a
  // (potentially huge) series of SearchDirs to find it.
  LookupFileCacheInfo &CacheLookup = LookupFileCache[Filename];

  // If the entry has been previously looked up, the first value will be
  // non-zero.  If the value is equal to i (the start point of our search), then
  // this is a matching hit.
  if (!SkipCache && CacheLookup.StartIdx == i+1) {
    // Skip querying potentially lots of directories for this lookup.
    i = CacheLookup.HitIdx;
    if (CacheLookup.MappedName) {
      Filename = CacheLookup.MappedName;
      if (IsMapped)
        *IsMapped = true;
    }
  } else {
    // Otherwise, this is the first query, or the previous query didn't match
    // our search start.  We will fill in our found location below, so prime the
    // start point value.
    CacheLookup.reset(/*StartIdx=*/i+1);
  }

  SmallString<64> MappedName;

  // Check each directory in sequence to see if it contains this file.
  for (; i != SearchDirs.size(); ++i) {
    bool InUserSpecifiedSystemFramework = false;
    bool IsInHeaderMap = false;
    bool IsFrameworkFoundInDir = false;
    Optional<FileEntryRef> File = SearchDirs[i].LookupFile(
        Filename, *this, IncludeLoc, SearchPath, RelativePath, RequestingModule,
        SuggestedModule, InUserSpecifiedSystemFramework, IsFrameworkFoundInDir,
        IsInHeaderMap, MappedName);
    if (!MappedName.empty()) {
      assert(IsInHeaderMap && "MappedName should come from a header map");
      CacheLookup.MappedName =
          copyString(MappedName, LookupFileCache.getAllocator());
    }
    if (IsMapped)
      // A filename is mapped when a header map remapped it to a relative path
      // used in subsequent header search or to an absolute path pointing to an
      // existing file.
      *IsMapped |= (!MappedName.empty() || (IsInHeaderMap && File));
    if (IsFrameworkFound)
      // Because we keep a filename remapped for subsequent search directory
      // lookups, ignore IsFrameworkFoundInDir after the first remapping and not
      // just for remapping in a current search directory.
      *IsFrameworkFound |= (IsFrameworkFoundInDir && !CacheLookup.MappedName);
    if (!File)
      continue;

    CurDir = &SearchDirs[i];

    // This file is a system header or C++ unfriendly if the dir is.
    HeaderFileInfo &HFI = getFileInfo(&File->getFileEntry());
    HFI.DirInfo = CurDir->getDirCharacteristic();

    // If the directory characteristic is User but this framework was
    // user-specified to be treated as a system framework, promote the
    // characteristic.
    if (HFI.DirInfo == SrcMgr::C_User && InUserSpecifiedSystemFramework)
      HFI.DirInfo = SrcMgr::C_System;

    // If the filename matches a known system header prefix, override
    // whether the file is a system header.
    for (unsigned j = SystemHeaderPrefixes.size(); j; --j) {
      if (Filename.startswith(SystemHeaderPrefixes[j-1].first)) {
        HFI.DirInfo = SystemHeaderPrefixes[j-1].second ? SrcMgr::C_System
                                                       : SrcMgr::C_User;
        break;
      }
    }

    // Set the `Framework` info if this file is in a header map with framework
    // style include spelling or found in a framework dir. The header map case
    // is possible when building frameworks which use header maps.
    if (CurDir->isHeaderMap() && isAngled) {
      size_t SlashPos = Filename.find('/');
      if (SlashPos != StringRef::npos)
        HFI.Framework =
            getUniqueFrameworkName(StringRef(Filename.begin(), SlashPos));
      if (CurDir->isIndexHeaderMap())
        HFI.IndexHeaderMapHeader = 1;
    } else if (CurDir->isFramework()) {
      size_t SlashPos = Filename.find('/');
      if (SlashPos != StringRef::npos)
        HFI.Framework =
            getUniqueFrameworkName(StringRef(Filename.begin(), SlashPos));
    }

    if (checkMSVCHeaderSearch(Diags, MSFE ? &MSFE->getFileEntry() : nullptr,
                              &File->getFileEntry(), IncludeLoc)) {
      if (SuggestedModule)
        *SuggestedModule = MSSuggestedModule;
      return MSFE;
    }

    bool FoundByHeaderMap = !IsMapped ? false : *IsMapped;
    if (!Includers.empty())
      diagnoseFrameworkInclude(
          Diags, IncludeLoc, Includers.front().second->getName(), Filename,
          &File->getFileEntry(), isAngled, FoundByHeaderMap);

    // Remember this location for the next lookup we do.
    cacheLookupSuccess(CacheLookup, i, IncludeLoc);
    return File;
  }

  // If we are including a file with a quoted include "foo.h" from inside
  // a header in a framework that is currently being built, and we couldn't
  // resolve "foo.h" any other way, change the include to <Foo/foo.h>, where
  // "Foo" is the name of the framework in which the including header was found.
  if (!Includers.empty() && Includers.front().first && !isAngled &&
      !Filename.contains('/')) {
    HeaderFileInfo &IncludingHFI = getFileInfo(Includers.front().first);
    if (IncludingHFI.IndexHeaderMapHeader) {
      SmallString<128> ScratchFilename;
      ScratchFilename += IncludingHFI.Framework;
      ScratchFilename += '/';
      ScratchFilename += Filename;

      Optional<FileEntryRef> File = LookupFile(
          ScratchFilename, IncludeLoc, /*isAngled=*/true, FromDir, &CurDir,
          Includers.front(), SearchPath, RelativePath, RequestingModule,
          SuggestedModule, IsMapped, /*IsFrameworkFound=*/nullptr);

      if (checkMSVCHeaderSearch(Diags, MSFE ? &MSFE->getFileEntry() : nullptr,
                                File ? &File->getFileEntry() : nullptr,
                                IncludeLoc)) {
        if (SuggestedModule)
          *SuggestedModule = MSSuggestedModule;
        return MSFE;
      }

      cacheLookupSuccess(LookupFileCache[Filename],
                         LookupFileCache[ScratchFilename].HitIdx, IncludeLoc);
      // FIXME: SuggestedModule.
      return File;
    }
  }

  if (checkMSVCHeaderSearch(Diags, MSFE ? &MSFE->getFileEntry() : nullptr,
                            nullptr, IncludeLoc)) {
    if (SuggestedModule)
      *SuggestedModule = MSSuggestedModule;
    return MSFE;
  }

  // Otherwise, didn't find it. Remember we didn't find this.
  CacheLookup.HitIdx = SearchDirs.size();
  return None;
}

/// LookupSubframeworkHeader - Look up a subframework for the specified
/// \#include file.  For example, if \#include'ing <HIToolbox/HIToolbox.h> from
/// within ".../Carbon.framework/Headers/Carbon.h", check to see if HIToolbox
/// is a subframework within Carbon.framework.  If so, return the FileEntry
/// for the designated file, otherwise return null.
Optional<FileEntryRef> HeaderSearch::LookupSubframeworkHeader(
    StringRef Filename, const FileEntry *ContextFileEnt,
    SmallVectorImpl<char> *SearchPath, SmallVectorImpl<char> *RelativePath,
    Module *RequestingModule, ModuleMap::KnownHeader *SuggestedModule) {
  assert(ContextFileEnt && "No context file?");

  // Framework names must have a '/' in the filename.  Find it.
  // FIXME: Should we permit '\' on Windows?
  size_t SlashPos = Filename.find('/');
  if (SlashPos == StringRef::npos)
    return None;

  // Look up the base framework name of the ContextFileEnt.
  StringRef ContextName = ContextFileEnt->getName();

  // If the context info wasn't a framework, couldn't be a subframework.
  const unsigned DotFrameworkLen = 10;
  auto FrameworkPos = ContextName.find(".framework");
  if (FrameworkPos == StringRef::npos ||
      (ContextName[FrameworkPos + DotFrameworkLen] != '/' &&
       ContextName[FrameworkPos + DotFrameworkLen] != '\\'))
    return None;

  SmallString<1024> FrameworkName(ContextName.data(), ContextName.data() +
                                                          FrameworkPos +
                                                          DotFrameworkLen + 1);

  // Append Frameworks/HIToolbox.framework/
  FrameworkName += "Frameworks/";
  FrameworkName.append(Filename.begin(), Filename.begin()+SlashPos);
  FrameworkName += ".framework/";

  auto &CacheLookup =
      *FrameworkMap.insert(std::make_pair(Filename.substr(0, SlashPos),
                                          FrameworkCacheEntry())).first;

  // Some other location?
  if (CacheLookup.second.Directory &&
      CacheLookup.first().size() == FrameworkName.size() &&
      memcmp(CacheLookup.first().data(), &FrameworkName[0],
             CacheLookup.first().size()) != 0)
    return None;

  // Cache subframework.
  if (!CacheLookup.second.Directory) {
    ++NumSubFrameworkLookups;

    // If the framework dir doesn't exist, we fail.
    auto Dir = FileMgr.getDirectory(FrameworkName);
    if (!Dir)
      return None;

    // Otherwise, if it does, remember that this is the right direntry for this
    // framework.
    CacheLookup.second.Directory = *Dir;
  }


  if (RelativePath) {
    RelativePath->clear();
    RelativePath->append(Filename.begin()+SlashPos+1, Filename.end());
  }

  // Check ".../Frameworks/HIToolbox.framework/Headers/HIToolbox.h"
  SmallString<1024> HeadersFilename(FrameworkName);
  HeadersFilename += "Headers/";
  if (SearchPath) {
    SearchPath->clear();
    // Without trailing '/'.
    SearchPath->append(HeadersFilename.begin(), HeadersFilename.end()-1);
  }

  HeadersFilename.append(Filename.begin()+SlashPos+1, Filename.end());
  auto File = FileMgr.getOptionalFileRef(HeadersFilename, /*OpenFile=*/true);
  if (!File) {
    // Check ".../Frameworks/HIToolbox.framework/PrivateHeaders/HIToolbox.h"
    HeadersFilename = FrameworkName;
    HeadersFilename += "PrivateHeaders/";
    if (SearchPath) {
      SearchPath->clear();
      // Without trailing '/'.
      SearchPath->append(HeadersFilename.begin(), HeadersFilename.end()-1);
    }

    HeadersFilename.append(Filename.begin()+SlashPos+1, Filename.end());
    File = FileMgr.getOptionalFileRef(HeadersFilename, /*OpenFile=*/true);

    if (!File)
      return None;
  }

  // This file is a system header or C++ unfriendly if the old file is.
  //
  // Note that the temporary 'DirInfo' is required here, as either call to
  // getFileInfo could resize the vector and we don't want to rely on order
  // of evaluation.
  unsigned DirInfo = getFileInfo(ContextFileEnt).DirInfo;
  getFileInfo(&File->getFileEntry()).DirInfo = DirInfo;

  FrameworkName.pop_back(); // remove the trailing '/'
  if (!findUsableModuleForFrameworkHeader(&File->getFileEntry(), FrameworkName,
                                          RequestingModule, SuggestedModule,
                                          /*IsSystem*/ false))
    return None;

  return *File;
}

//===----------------------------------------------------------------------===//
// File Info Management.
//===----------------------------------------------------------------------===//

/// Merge the header file info provided by \p OtherHFI into the current
/// header file info (\p HFI)
static void mergeHeaderFileInfo(HeaderFileInfo &HFI,
                                const HeaderFileInfo &OtherHFI) {
  assert(OtherHFI.External && "expected to merge external HFI");

  HFI.isImport |= OtherHFI.isImport;
  HFI.isPragmaOnce |= OtherHFI.isPragmaOnce;
  HFI.isModuleHeader |= OtherHFI.isModuleHeader;

  if (!HFI.ControllingMacro && !HFI.ControllingMacroID) {
    HFI.ControllingMacro = OtherHFI.ControllingMacro;
    HFI.ControllingMacroID = OtherHFI.ControllingMacroID;
  }

  HFI.DirInfo = OtherHFI.DirInfo;
  HFI.External = (!HFI.IsValid || HFI.External);
  HFI.IsValid = true;
  HFI.IndexHeaderMapHeader = OtherHFI.IndexHeaderMapHeader;

  if (HFI.Framework.empty())
    HFI.Framework = OtherHFI.Framework;
}

/// getFileInfo - Return the HeaderFileInfo structure for the specified
/// FileEntry.
HeaderFileInfo &HeaderSearch::getFileInfo(const FileEntry *FE) {
  if (FE->getUID() >= FileInfo.size())
    FileInfo.resize(FE->getUID() + 1);

  HeaderFileInfo *HFI = &FileInfo[FE->getUID()];
  // FIXME: Use a generation count to check whether this is really up to date.
  if (ExternalSource && !HFI->Resolved) {
    auto ExternalHFI = ExternalSource->GetHeaderFileInfo(FE);
    if (ExternalHFI.IsValid) {
      HFI->Resolved = true;
      if (ExternalHFI.External)
        mergeHeaderFileInfo(*HFI, ExternalHFI);
    }
  }

  HFI->IsValid = true;
  // We have local information about this header file, so it's no longer
  // strictly external.
  HFI->External = false;
  return *HFI;
}

const HeaderFileInfo *
HeaderSearch::getExistingFileInfo(const FileEntry *FE,
                                  bool WantExternal) const {
  // If we have an external source, ensure we have the latest information.
  // FIXME: Use a generation count to check whether this is really up to date.
  HeaderFileInfo *HFI;
  if (ExternalSource) {
    if (FE->getUID() >= FileInfo.size()) {
      if (!WantExternal)
        return nullptr;
      FileInfo.resize(FE->getUID() + 1);
    }

    HFI = &FileInfo[FE->getUID()];
    if (!WantExternal && (!HFI->IsValid || HFI->External))
      return nullptr;
    if (!HFI->Resolved) {
      auto ExternalHFI = ExternalSource->GetHeaderFileInfo(FE);
      if (ExternalHFI.IsValid) {
        HFI->Resolved = true;
        if (ExternalHFI.External)
          mergeHeaderFileInfo(*HFI, ExternalHFI);
      }
    }
  } else if (FE->getUID() >= FileInfo.size()) {
    return nullptr;
  } else {
    HFI = &FileInfo[FE->getUID()];
  }

  if (!HFI->IsValid || (HFI->External && !WantExternal))
    return nullptr;

  return HFI;
}

bool HeaderSearch::isFileMultipleIncludeGuarded(const FileEntry *File) {
  // Check if we've entered this file and found an include guard or #pragma
  // once. Note that we dor't check for #import, because that's not a property
  // of the file itself.
  if (auto *HFI = getExistingFileInfo(File))
    return HFI->isPragmaOnce || HFI->ControllingMacro ||
           HFI->ControllingMacroID;
  return false;
}

void HeaderSearch::MarkFileModuleHeader(const FileEntry *FE,
                                        ModuleMap::ModuleHeaderRole Role,
                                        bool isCompilingModuleHeader) {
  bool isModularHeader = !(Role & ModuleMap::TextualHeader);

  // Don't mark the file info as non-external if there's nothing to change.
  if (!isCompilingModuleHeader) {
    if (!isModularHeader)
      return;
    auto *HFI = getExistingFileInfo(FE);
    if (HFI && HFI->isModuleHeader)
      return;
  }

  auto &HFI = getFileInfo(FE);
  HFI.isModuleHeader |= isModularHeader;
  HFI.isCompilingModuleHeader |= isCompilingModuleHeader;
}

bool HeaderSearch::ShouldEnterIncludeFile(Preprocessor &PP,
                                          const FileEntry *File, bool isImport,
                                          bool ModulesEnabled, Module *M,
                                          bool &IsFirstIncludeOfFile) {
  ++NumIncluded; // Count # of attempted #includes.

  IsFirstIncludeOfFile = false;

  // Get information about this file.
  HeaderFileInfo &FileInfo = getFileInfo(File);

  // FIXME: this is a workaround for the lack of proper modules-aware support
  // for #import / #pragma once
  auto TryEnterImported = [&]() -> bool {
    if (!ModulesEnabled)
      return false;
    // Ensure FileInfo bits are up to date.
    ModMap.resolveHeaderDirectives(File);
    // Modules with builtins are special; multiple modules use builtins as
    // modular headers, example:
    //
    //    module stddef { header "stddef.h" export * }
    //
    // After module map parsing, this expands to:
    //
    //    module stddef {
    //      header "/path_to_builtin_dirs/stddef.h"
    //      textual "stddef.h"
    //    }
    //
    // It's common that libc++ and system modules will both define such
    // submodules. Make sure cached results for a builtin header won't
    // prevent other builtin modules from potentially entering the builtin
    // header. Note that builtins are header guarded and the decision to
    // actually enter them is postponed to the controlling macros logic below.
    bool TryEnterHdr = false;
    if (FileInfo.isCompilingModuleHeader && FileInfo.isModuleHeader)
      TryEnterHdr = ModMap.isBuiltinHeader(File);

    // Textual headers can be #imported from different modules. Since ObjC
    // headers find in the wild might rely only on #import and do not contain
    // controlling macros, be conservative and only try to enter textual headers
    // if such macro is present.
    if (!FileInfo.isModuleHeader &&
        FileInfo.getControllingMacro(ExternalLookup))
      TryEnterHdr = true;
    return TryEnterHdr;
  };

  // If this is a #import directive, check that we have not already imported
  // this header.
  if (isImport) {
    // If this has already been imported, don't import it again.
    FileInfo.isImport = true;

    // Has this already been #import'ed or #include'd?
    if (PP.alreadyIncluded(File) && !TryEnterImported())
      return false;
  } else {
    // Otherwise, if this is a #include of a file that was previously #import'd
    // or if this is the second #include of a #pragma once file, ignore it.
    if ((FileInfo.isPragmaOnce || FileInfo.isImport) && !TryEnterImported())
      return false;
  }

  // Next, check to see if the file is wrapped with #ifndef guards.  If so, and
  // if the macro that guards it is defined, we know the #include has no effect.
  if (const IdentifierInfo *ControllingMacro
      = FileInfo.getControllingMacro(ExternalLookup)) {
    // If the header corresponds to a module, check whether the macro is already
    // defined in that module rather than checking in the current set of visible
    // modules.
    if (M ? PP.isMacroDefinedInLocalModule(ControllingMacro, M)
          : PP.isMacroDefined(ControllingMacro)) {
      ++NumMultiIncludeFileOptzn;
      return false;
    }
  }

  IsFirstIncludeOfFile = PP.markIncluded(File);

  return true;
}

size_t HeaderSearch::getTotalMemory() const {
  return SearchDirs.capacity()
    + llvm::capacity_in_bytes(FileInfo)
    + llvm::capacity_in_bytes(HeaderMaps)
    + LookupFileCache.getAllocator().getTotalMemory()
    + FrameworkMap.getAllocator().getTotalMemory();
}

Optional<unsigned> HeaderSearch::searchDirIdx(const DirectoryLookup &DL) const {
  for (unsigned I = 0; I < SearchDirs.size(); ++I)
    if (&SearchDirs[I] == &DL)
      return I;
  return None;
}

StringRef HeaderSearch::getUniqueFrameworkName(StringRef Framework) {
  return FrameworkNames.insert(Framework).first->first();
}

bool HeaderSearch::hasModuleMap(StringRef FileName,
                                const DirectoryEntry *Root,
                                bool IsSystem) {
  if (!HSOpts->ImplicitModuleMaps)
    return false;

  SmallVector<const DirectoryEntry *, 2> FixUpDirectories;

  StringRef DirName = FileName;
  do {
    // Get the parent directory name.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      return false;

    // Determine whether this directory exists.
    auto Dir = FileMgr.getDirectory(DirName);
    if (!Dir)
      return false;

    // Try to load the module map file in this directory.
    switch (loadModuleMapFile(*Dir, IsSystem,
                              llvm::sys::path::extension((*Dir)->getName()) ==
                                  ".framework")) {
    case LMM_NewlyLoaded:
    case LMM_AlreadyLoaded:
      // Success. All of the directories we stepped through inherit this module
      // map file.
      for (unsigned I = 0, N = FixUpDirectories.size(); I != N; ++I)
        DirectoryHasModuleMap[FixUpDirectories[I]] = true;
      return true;

    case LMM_NoDirectory:
    case LMM_InvalidModuleMap:
      break;
    }

    // If we hit the top of our search, we're done.
    if (*Dir == Root)
      return false;

    // Keep track of all of the directories we checked, so we can mark them as
    // having module maps if we eventually do find a module map.
    FixUpDirectories.push_back(*Dir);
  } while (true);
}

ModuleMap::KnownHeader
HeaderSearch::findModuleForHeader(const FileEntry *File,
                                  bool AllowTextual) const {
  if (ExternalSource) {
    // Make sure the external source has handled header info about this file,
    // which includes whether the file is part of a module.
    (void)getExistingFileInfo(File);
  }
  return ModMap.findModuleForHeader(File, AllowTextual);
}

ArrayRef<ModuleMap::KnownHeader>
HeaderSearch::findAllModulesForHeader(const FileEntry *File) const {
  if (ExternalSource) {
    // Make sure the external source has handled header info about this file,
    // which includes whether the file is part of a module.
    (void)getExistingFileInfo(File);
  }
  return ModMap.findAllModulesForHeader(File);
}

static bool suggestModule(HeaderSearch &HS, const FileEntry *File,
                          Module *RequestingModule,
                          ModuleMap::KnownHeader *SuggestedModule) {
  ModuleMap::KnownHeader Module =
      HS.findModuleForHeader(File, /*AllowTextual*/true);

  // If this module specifies [no_undeclared_includes], we cannot find any
  // file that's in a non-dependency module.
  if (RequestingModule && Module && RequestingModule->NoUndeclaredIncludes) {
    HS.getModuleMap().resolveUses(RequestingModule, /*Complain*/ false);
    if (!RequestingModule->directlyUses(Module.getModule())) {
      // Builtin headers are a special case. Multiple modules can use the same
      // builtin as a modular header (see also comment in
      // ShouldEnterIncludeFile()), so the builtin header may have been
      // "claimed" by an unrelated module. This shouldn't prevent us from
      // including the builtin header textually in this module.
      if (HS.getModuleMap().isBuiltinHeader(File)) {
        if (SuggestedModule)
          *SuggestedModule = ModuleMap::KnownHeader();
        return true;
      }
      return false;
    }
  }

  if (SuggestedModule)
    *SuggestedModule = (Module.getRole() & ModuleMap::TextualHeader)
                           ? ModuleMap::KnownHeader()
                           : Module;

  return true;
}

bool HeaderSearch::findUsableModuleForHeader(
    const FileEntry *File, const DirectoryEntry *Root, Module *RequestingModule,
    ModuleMap::KnownHeader *SuggestedModule, bool IsSystemHeaderDir) {
  if (File && needModuleLookup(RequestingModule, SuggestedModule)) {
    // If there is a module that corresponds to this header, suggest it.
    hasModuleMap(File->getName(), Root, IsSystemHeaderDir);
    return suggestModule(*this, File, RequestingModule, SuggestedModule);
  }
  return true;
}

bool HeaderSearch::findUsableModuleForFrameworkHeader(
    const FileEntry *File, StringRef FrameworkName, Module *RequestingModule,
    ModuleMap::KnownHeader *SuggestedModule, bool IsSystemFramework) {
  // If we're supposed to suggest a module, look for one now.
  if (needModuleLookup(RequestingModule, SuggestedModule)) {
    // Find the top-level framework based on this framework.
    SmallVector<std::string, 4> SubmodulePath;
    const DirectoryEntry *TopFrameworkDir
      = ::getTopFrameworkDir(FileMgr, FrameworkName, SubmodulePath);

    // Determine the name of the top-level framework.
    StringRef ModuleName = llvm::sys::path::stem(TopFrameworkDir->getName());

    // Load this framework module. If that succeeds, find the suggested module
    // for this header, if any.
    loadFrameworkModule(ModuleName, TopFrameworkDir, IsSystemFramework);

    // FIXME: This can find a module not part of ModuleName, which is
    // important so that we're consistent about whether this header
    // corresponds to a module. Possibly we should lock down framework modules
    // so that this is not possible.
    return suggestModule(*this, File, RequestingModule, SuggestedModule);
  }
  return true;
}

static const FileEntry *getPrivateModuleMap(const FileEntry *File,
                                            FileManager &FileMgr) {
  StringRef Filename = llvm::sys::path::filename(File->getName());
  SmallString<128>  PrivateFilename(File->getDir()->getName());
  if (Filename == "module.map")
    llvm::sys::path::append(PrivateFilename, "module_private.map");
  else if (Filename == "module.modulemap")
    llvm::sys::path::append(PrivateFilename, "module.private.modulemap");
  else
    return nullptr;
  if (auto File = FileMgr.getFile(PrivateFilename))
    return *File;
  return nullptr;
}

bool HeaderSearch::loadModuleMapFile(const FileEntry *File, bool IsSystem,
                                     FileID ID, unsigned *Offset,
                                     StringRef OriginalModuleMapFile) {
  // Find the directory for the module. For frameworks, that may require going
  // up from the 'Modules' directory.
  const DirectoryEntry *Dir = nullptr;
  if (getHeaderSearchOpts().ModuleMapFileHomeIsCwd) {
    if (auto DirOrErr = FileMgr.getDirectory("."))
      Dir = *DirOrErr;
  } else {
    if (!OriginalModuleMapFile.empty()) {
      // We're building a preprocessed module map. Find or invent the directory
      // that it originally occupied.
      auto DirOrErr = FileMgr.getDirectory(
          llvm::sys::path::parent_path(OriginalModuleMapFile));
      if (DirOrErr) {
        Dir = *DirOrErr;
      } else {
        auto *FakeFile = FileMgr.getVirtualFile(OriginalModuleMapFile, 0, 0);
        Dir = FakeFile->getDir();
      }
    } else {
      Dir = File->getDir();
    }

    StringRef DirName(Dir->getName());
    if (llvm::sys::path::filename(DirName) == "Modules") {
      DirName = llvm::sys::path::parent_path(DirName);
      if (DirName.endswith(".framework"))
        if (auto DirOrErr = FileMgr.getDirectory(DirName))
          Dir = *DirOrErr;
      // FIXME: This assert can fail if there's a race between the above check
      // and the removal of the directory.
      assert(Dir && "parent must exist");
    }
  }

  switch (loadModuleMapFileImpl(File, IsSystem, Dir, ID, Offset)) {
  case LMM_AlreadyLoaded:
  case LMM_NewlyLoaded:
    return false;
  case LMM_NoDirectory:
  case LMM_InvalidModuleMap:
    return true;
  }
  llvm_unreachable("Unknown load module map result");
}

HeaderSearch::LoadModuleMapResult
HeaderSearch::loadModuleMapFileImpl(const FileEntry *File, bool IsSystem,
                                    const DirectoryEntry *Dir, FileID ID,
                                    unsigned *Offset) {
  assert(File && "expected FileEntry");

  // Check whether we've already loaded this module map, and mark it as being
  // loaded in case we recursively try to load it from itself.
  auto AddResult = LoadedModuleMaps.insert(std::make_pair(File, true));
  if (!AddResult.second)
    return AddResult.first->second ? LMM_AlreadyLoaded : LMM_InvalidModuleMap;

  if (ModMap.parseModuleMapFile(File, IsSystem, Dir, ID, Offset)) {
    LoadedModuleMaps[File] = false;
    return LMM_InvalidModuleMap;
  }

  // Try to load a corresponding private module map.
  if (const FileEntry *PMMFile = getPrivateModuleMap(File, FileMgr)) {
    if (ModMap.parseModuleMapFile(PMMFile, IsSystem, Dir)) {
      LoadedModuleMaps[File] = false;
      return LMM_InvalidModuleMap;
    }
  }

  // This directory has a module map.
  return LMM_NewlyLoaded;
}

const FileEntry *
HeaderSearch::lookupModuleMapFile(const DirectoryEntry *Dir, bool IsFramework) {
  if (!HSOpts->ImplicitModuleMaps)
    return nullptr;
  // For frameworks, the preferred spelling is Modules/module.modulemap, but
  // module.map at the framework root is also accepted.
  SmallString<128> ModuleMapFileName(Dir->getName());
  if (IsFramework)
    llvm::sys::path::append(ModuleMapFileName, "Modules");
  llvm::sys::path::append(ModuleMapFileName, "module.modulemap");
  if (auto F = FileMgr.getFile(ModuleMapFileName))
    return *F;

  // Continue to allow module.map
  ModuleMapFileName = Dir->getName();
  llvm::sys::path::append(ModuleMapFileName, "module.map");
  if (auto F = FileMgr.getFile(ModuleMapFileName))
    return *F;

  // For frameworks, allow to have a private module map with a preferred
  // spelling when a public module map is absent.
  if (IsFramework) {
    ModuleMapFileName = Dir->getName();
    llvm::sys::path::append(ModuleMapFileName, "Modules",
                            "module.private.modulemap");
    if (auto F = FileMgr.getFile(ModuleMapFileName))
      return *F;
  }
  return nullptr;
}

Module *HeaderSearch::loadFrameworkModule(StringRef Name,
                                          const DirectoryEntry *Dir,
                                          bool IsSystem) {
  if (Module *Module = ModMap.findModule(Name))
    return Module;

  // Try to load a module map file.
  switch (loadModuleMapFile(Dir, IsSystem, /*IsFramework*/true)) {
  case LMM_InvalidModuleMap:
    // Try to infer a module map from the framework directory.
    if (HSOpts->ImplicitModuleMaps)
      ModMap.inferFrameworkModule(Dir, IsSystem, /*Parent=*/nullptr);
    break;

  case LMM_AlreadyLoaded:
  case LMM_NoDirectory:
    return nullptr;

  case LMM_NewlyLoaded:
    break;
  }

  return ModMap.findModule(Name);
}

HeaderSearch::LoadModuleMapResult
HeaderSearch::loadModuleMapFile(StringRef DirName, bool IsSystem,
                                bool IsFramework) {
  if (auto Dir = FileMgr.getDirectory(DirName))
    return loadModuleMapFile(*Dir, IsSystem, IsFramework);

  return LMM_NoDirectory;
}

HeaderSearch::LoadModuleMapResult
HeaderSearch::loadModuleMapFile(const DirectoryEntry *Dir, bool IsSystem,
                                bool IsFramework) {
  auto KnownDir = DirectoryHasModuleMap.find(Dir);
  if (KnownDir != DirectoryHasModuleMap.end())
    return KnownDir->second ? LMM_AlreadyLoaded : LMM_InvalidModuleMap;

  if (const FileEntry *ModuleMapFile = lookupModuleMapFile(Dir, IsFramework)) {
    LoadModuleMapResult Result =
        loadModuleMapFileImpl(ModuleMapFile, IsSystem, Dir);
    // Add Dir explicitly in case ModuleMapFile is in a subdirectory.
    // E.g. Foo.framework/Modules/module.modulemap
    //      ^Dir                  ^ModuleMapFile
    if (Result == LMM_NewlyLoaded)
      DirectoryHasModuleMap[Dir] = true;
    else if (Result == LMM_InvalidModuleMap)
      DirectoryHasModuleMap[Dir] = false;
    return Result;
  }
  return LMM_InvalidModuleMap;
}

void HeaderSearch::collectAllModules(SmallVectorImpl<Module *> &Modules) {
  Modules.clear();

  if (HSOpts->ImplicitModuleMaps) {
    // Load module maps for each of the header search directories.
    for (unsigned Idx = 0, N = SearchDirs.size(); Idx != N; ++Idx) {
      bool IsSystem = SearchDirs[Idx].isSystemHeaderDirectory();
      if (SearchDirs[Idx].isFramework()) {
        std::error_code EC;
        SmallString<128> DirNative;
        llvm::sys::path::native(SearchDirs[Idx].getFrameworkDir()->getName(),
                                DirNative);

        // Search each of the ".framework" directories to load them as modules.
        llvm::vfs::FileSystem &FS = FileMgr.getVirtualFileSystem();
        for (llvm::vfs::directory_iterator Dir = FS.dir_begin(DirNative, EC),
                                           DirEnd;
             Dir != DirEnd && !EC; Dir.increment(EC)) {
          if (llvm::sys::path::extension(Dir->path()) != ".framework")
            continue;

          auto FrameworkDir =
              FileMgr.getDirectory(Dir->path());
          if (!FrameworkDir)
            continue;

          // Load this framework module.
          loadFrameworkModule(llvm::sys::path::stem(Dir->path()), *FrameworkDir,
                              IsSystem);
        }
        continue;
      }

      // FIXME: Deal with header maps.
      if (SearchDirs[Idx].isHeaderMap())
        continue;

      // Try to load a module map file for the search directory.
      loadModuleMapFile(SearchDirs[Idx].getDir(), IsSystem,
                        /*IsFramework*/ false);

      // Try to load module map files for immediate subdirectories of this
      // search directory.
      loadSubdirectoryModuleMaps(SearchDirs[Idx]);
    }
  }

  // Populate the list of modules.
  llvm::transform(ModMap.modules(), std::back_inserter(Modules),
                  [](const auto &NameAndMod) { return NameAndMod.second; });
}

void HeaderSearch::loadTopLevelSystemModules() {
  if (!HSOpts->ImplicitModuleMaps)
    return;

  // Load module maps for each of the header search directories.
  for (unsigned Idx = 0, N = SearchDirs.size(); Idx != N; ++Idx) {
    // We only care about normal header directories.
    if (!SearchDirs[Idx].isNormalDir()) {
      continue;
    }

    // Try to load a module map file for the search directory.
    loadModuleMapFile(SearchDirs[Idx].getDir(),
                      SearchDirs[Idx].isSystemHeaderDirectory(),
                      SearchDirs[Idx].isFramework());
  }
}

void HeaderSearch::loadSubdirectoryModuleMaps(DirectoryLookup &SearchDir) {
  assert(HSOpts->ImplicitModuleMaps &&
         "Should not be loading subdirectory module maps");

  if (SearchDir.haveSearchedAllModuleMaps())
    return;

  std::error_code EC;
  SmallString<128> Dir = SearchDir.getDir()->getName();
  FileMgr.makeAbsolutePath(Dir);
  SmallString<128> DirNative;
  llvm::sys::path::native(Dir, DirNative);
  llvm::vfs::FileSystem &FS = FileMgr.getVirtualFileSystem();
  for (llvm::vfs::directory_iterator Dir = FS.dir_begin(DirNative, EC), DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    bool IsFramework = llvm::sys::path::extension(Dir->path()) == ".framework";
    if (IsFramework == SearchDir.isFramework())
      loadModuleMapFile(Dir->path(), SearchDir.isSystemHeaderDirectory(),
                        SearchDir.isFramework());
  }

  SearchDir.setSearchedAllModuleMaps(true);
}

std::string HeaderSearch::suggestPathToFileForDiagnostics(
    const FileEntry *File, llvm::StringRef MainFile, bool *IsSystem) {
  // FIXME: We assume that the path name currently cached in the FileEntry is
  // the most appropriate one for this analysis (and that it's spelled the
  // same way as the corresponding header search path).
  return suggestPathToFileForDiagnostics(File->getName(), /*WorkingDir=*/"",
                                         MainFile, IsSystem);
}

std::string HeaderSearch::suggestPathToFileForDiagnostics(
    llvm::StringRef File, llvm::StringRef WorkingDir, llvm::StringRef MainFile,
    bool *IsSystem) {
  using namespace llvm::sys;

  unsigned BestPrefixLength = 0;
  // Checks whether `Dir` is a strict path prefix of `File`. If so and that's
  // the longest prefix we've seen so for it, returns true and updates the
  // `BestPrefixLength` accordingly.
  auto CheckDir = [&](llvm::StringRef Dir) -> bool {
    llvm::SmallString<32> DirPath(Dir.begin(), Dir.end());
    if (!WorkingDir.empty() && !path::is_absolute(Dir))
      fs::make_absolute(WorkingDir, DirPath);
    path::remove_dots(DirPath, /*remove_dot_dot=*/true);
    Dir = DirPath;
    for (auto NI = path::begin(File), NE = path::end(File),
              DI = path::begin(Dir), DE = path::end(Dir);
         /*termination condition in loop*/; ++NI, ++DI) {
      // '.' components in File are ignored.
      while (NI != NE && *NI == ".")
        ++NI;
      if (NI == NE)
        break;

      // '.' components in Dir are ignored.
      while (DI != DE && *DI == ".")
        ++DI;
      if (DI == DE) {
        // Dir is a prefix of File, up to '.' components and choice of path
        // separators.
        unsigned PrefixLength = NI - path::begin(File);
        if (PrefixLength > BestPrefixLength) {
          BestPrefixLength = PrefixLength;
          return true;
        }
        break;
      }

      // Consider all path separators equal.
      if (NI->size() == 1 && DI->size() == 1 &&
          path::is_separator(NI->front()) && path::is_separator(DI->front()))
        continue;

      // Special case Apple .sdk folders since the search path is typically a
      // symlink like `iPhoneSimulator14.5.sdk` while the file is instead
      // located in `iPhoneSimulator.sdk` (the real folder).
      if (NI->endswith(".sdk") && DI->endswith(".sdk")) {
        StringRef NBasename = path::stem(*NI);
        StringRef DBasename = path::stem(*DI);
        if (DBasename.startswith(NBasename))
          continue;
      }

      if (*NI != *DI)
        break;
    }
    return false;
  };

  bool BestPrefixIsFramework = false;
  for (unsigned I = 0; I != SearchDirs.size(); ++I) {
    if (SearchDirs[I].isNormalDir()) {
      StringRef Dir = SearchDirs[I].getDir()->getName();
      if (CheckDir(Dir)) {
        if (IsSystem)
          *IsSystem = BestPrefixLength ? I >= SystemDirIdx : false;
        BestPrefixIsFramework = false;
      }
    } else if (SearchDirs[I].isFramework()) {
      StringRef Dir = SearchDirs[I].getFrameworkDir()->getName();
      if (CheckDir(Dir)) {
        if (IsSystem)
          *IsSystem = BestPrefixLength ? I >= SystemDirIdx : false;
        BestPrefixIsFramework = true;
      }
    }
  }

  // Try to shorten include path using TUs directory, if we couldn't find any
  // suitable prefix in include search paths.
  if (!BestPrefixLength && CheckDir(path::parent_path(MainFile))) {
    if (IsSystem)
      *IsSystem = false;
    BestPrefixIsFramework = false;
  }

  // Try resolving resulting filename via reverse search in header maps,
  // key from header name is user prefered name for the include file.
  StringRef Filename = File.drop_front(BestPrefixLength);
  for (unsigned I = 0; I != SearchDirs.size(); ++I) {
    if (!SearchDirs[I].isHeaderMap())
      continue;

    StringRef SpelledFilename =
        SearchDirs[I].getHeaderMap()->reverseLookupFilename(Filename);
    if (!SpelledFilename.empty()) {
      Filename = SpelledFilename;
      BestPrefixIsFramework = false;
      break;
    }
  }

  // If the best prefix is a framework path, we need to compute the proper
  // include spelling for the framework header.
  bool IsPrivateHeader;
  SmallString<128> FrameworkName, IncludeSpelling;
  if (BestPrefixIsFramework &&
      isFrameworkStylePath(Filename, IsPrivateHeader, FrameworkName,
                           IncludeSpelling)) {
    Filename = IncludeSpelling;
  }
  return path::convert_to_slash(Filename);
}
