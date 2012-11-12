//===--- HeaderSearch.cpp - Resolve Header File Locations ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the DirectoryLookup and HeaderSearch interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/HeaderMap.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Capacity.h"
#include <cstdio>
using namespace clang;

const IdentifierInfo *
HeaderFileInfo::getControllingMacro(ExternalIdentifierLookup *External) {
  if (ControllingMacro)
    return ControllingMacro;

  if (!ControllingMacroID || !External)
    return 0;

  ControllingMacro = External->GetIdentifier(ControllingMacroID);
  return ControllingMacro;
}

ExternalHeaderFileInfoSource::~ExternalHeaderFileInfoSource() {}

HeaderSearch::HeaderSearch(llvm::IntrusiveRefCntPtr<HeaderSearchOptions> HSOpts,
                           FileManager &FM, DiagnosticsEngine &Diags,
                           const LangOptions &LangOpts, 
                           const TargetInfo *Target)
  : HSOpts(HSOpts), FileMgr(FM), FrameworkMap(64),
    ModMap(FileMgr, *Diags.getClient(), LangOpts, Target)
{
  AngledDirIdx = 0;
  SystemDirIdx = 0;
  NoCurDirSearch = false;

  ExternalLookup = 0;
  ExternalSource = 0;
  NumIncluded = 0;
  NumMultiIncludeFileOptzn = 0;
  NumFrameworkLookups = NumSubFrameworkLookups = 0;
}

HeaderSearch::~HeaderSearch() {
  // Delete headermaps.
  for (unsigned i = 0, e = HeaderMaps.size(); i != e; ++i)
    delete HeaderMaps[i].second;
}

void HeaderSearch::PrintStats() {
  fprintf(stderr, "\n*** HeaderSearch Stats:\n");
  fprintf(stderr, "%d files tracked.\n", (int)FileInfo.size());
  unsigned NumOnceOnlyFiles = 0, MaxNumIncludes = 0, NumSingleIncludedFiles = 0;
  for (unsigned i = 0, e = FileInfo.size(); i != e; ++i) {
    NumOnceOnlyFiles += FileInfo[i].isImport;
    if (MaxNumIncludes < FileInfo[i].NumIncludes)
      MaxNumIncludes = FileInfo[i].NumIncludes;
    NumSingleIncludedFiles += FileInfo[i].NumIncludes == 1;
  }
  fprintf(stderr, "  %d #import/#pragma once files.\n", NumOnceOnlyFiles);
  fprintf(stderr, "  %d included exactly once.\n", NumSingleIncludedFiles);
  fprintf(stderr, "  %d max times a file is included.\n", MaxNumIncludes);

  fprintf(stderr, "  %d #include/#include_next/#import.\n", NumIncluded);
  fprintf(stderr, "    %d #includes skipped due to"
          " the multi-include optimization.\n", NumMultiIncludeFileOptzn);

  fprintf(stderr, "%d framework lookups.\n", NumFrameworkLookups);
  fprintf(stderr, "%d subframework lookups.\n", NumSubFrameworkLookups);
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
        return HeaderMaps[i].second;
  }

  if (const HeaderMap *HM = HeaderMap::Create(FE, FileMgr)) {
    HeaderMaps.push_back(std::make_pair(FE, HM));
    return HM;
  }

  return 0;
}

std::string HeaderSearch::getModuleFileName(Module *Module) {
  // If we don't have a module cache path, we can't do anything.
  if (ModuleCachePath.empty()) 
    return std::string();


  SmallString<256> Result(ModuleCachePath);
  llvm::sys::path::append(Result, Module->getTopLevelModule()->Name + ".pcm");
  return Result.str().str();
}

std::string HeaderSearch::getModuleFileName(StringRef ModuleName) {
  // If we don't have a module cache path, we can't do anything.
  if (ModuleCachePath.empty()) 
    return std::string();
  
  
  SmallString<256> Result(ModuleCachePath);
  llvm::sys::path::append(Result, ModuleName + ".pcm");
  return Result.str().str();
}

Module *HeaderSearch::lookupModule(StringRef ModuleName, bool AllowSearch) {
  // Look in the module map to determine if there is a module by this name.
  Module *Module = ModMap.findModule(ModuleName);
  if (Module || !AllowSearch)
    return Module;
  
  // Look through the various header search paths to load any avai;able module 
  // maps, searching for a module map that describes this module.
  for (unsigned Idx = 0, N = SearchDirs.size(); Idx != N; ++Idx) {
    if (SearchDirs[Idx].isFramework()) {
      // Search for or infer a module map for a framework.
      SmallString<128> FrameworkDirName;
      FrameworkDirName += SearchDirs[Idx].getFrameworkDir()->getName();
      llvm::sys::path::append(FrameworkDirName, ModuleName + ".framework");
      if (const DirectoryEntry *FrameworkDir 
            = FileMgr.getDirectory(FrameworkDirName)) {
        bool IsSystem
          = SearchDirs[Idx].getDirCharacteristic() != SrcMgr::C_User;
        Module = loadFrameworkModule(ModuleName, FrameworkDir, IsSystem);
        if (Module)
          break;
      }
    }
    
    // FIXME: Figure out how header maps and module maps will work together.
    
    // Only deal with normal search directories.
    if (!SearchDirs[Idx].isNormalDir())
      continue;
    
    // Search for a module map file in this directory.
    if (loadModuleMapFile(SearchDirs[Idx].getDir()) == LMM_NewlyLoaded) {
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
    if (loadModuleMapFile(NestedModuleMapDirName) == LMM_NewlyLoaded) {
      // If we just loaded a module map file, look for the module again.
      Module = ModMap.findModule(ModuleName);
      if (Module)
        break;
    }
  }
  
  return Module;
}

//===----------------------------------------------------------------------===//
// File lookup within a DirectoryLookup scope
//===----------------------------------------------------------------------===//

/// getName - Return the directory or filename corresponding to this lookup
/// object.
const char *DirectoryLookup::getName() const {
  if (isNormalDir())
    return getDir()->getName();
  if (isFramework())
    return getFrameworkDir()->getName();
  assert(isHeaderMap() && "Unknown DirectoryLookup");
  return getHeaderMap()->getFileName();
}


/// LookupFile - Lookup the specified file in this search path, returning it
/// if it exists or returning null if not.
const FileEntry *DirectoryLookup::LookupFile(
    StringRef Filename,
    HeaderSearch &HS,
    SmallVectorImpl<char> *SearchPath,
    SmallVectorImpl<char> *RelativePath,
    Module **SuggestedModule,
    bool &InUserSpecifiedSystemFramework) const {
  InUserSpecifiedSystemFramework = false;

  SmallString<1024> TmpDir;
  if (isNormalDir()) {
    // Concatenate the requested file onto the directory.
    TmpDir = getDir()->getName();
    llvm::sys::path::append(TmpDir, Filename);
    if (SearchPath != NULL) {
      StringRef SearchPathRef(getDir()->getName());
      SearchPath->clear();
      SearchPath->append(SearchPathRef.begin(), SearchPathRef.end());
    }
    if (RelativePath != NULL) {
      RelativePath->clear();
      RelativePath->append(Filename.begin(), Filename.end());
    }
    
    // If we have a module map that might map this header, load it and
    // check whether we'll have a suggestion for a module.
    if (SuggestedModule && HS.hasModuleMap(TmpDir, getDir())) {
      const FileEntry *File = HS.getFileMgr().getFile(TmpDir.str(), 
                                                      /*openFile=*/false);
      if (!File)
        return File;
      
      // If there is a module that corresponds to this header, 
      // suggest it.
      *SuggestedModule = HS.findModuleForHeader(File);
      return File;
    }
    
    return HS.getFileMgr().getFile(TmpDir.str(), /*openFile=*/true);
  }

  if (isFramework())
    return DoFrameworkLookup(Filename, HS, SearchPath, RelativePath,
                             SuggestedModule, InUserSpecifiedSystemFramework);

  assert(isHeaderMap() && "Unknown directory lookup");
  const FileEntry * const Result = getHeaderMap()->LookupFile(
      Filename, HS.getFileMgr());
  if (Result) {
    if (SearchPath != NULL) {
      StringRef SearchPathRef(getName());
      SearchPath->clear();
      SearchPath->append(SearchPathRef.begin(), SearchPathRef.end());
    }
    if (RelativePath != NULL) {
      RelativePath->clear();
      RelativePath->append(Filename.begin(), Filename.end());
    }
  }
  return Result;
}


/// DoFrameworkLookup - Do a lookup of the specified file in the current
/// DirectoryLookup, which is a framework directory.
const FileEntry *DirectoryLookup::DoFrameworkLookup(
    StringRef Filename,
    HeaderSearch &HS,
    SmallVectorImpl<char> *SearchPath,
    SmallVectorImpl<char> *RelativePath,
    Module **SuggestedModule,
    bool &InUserSpecifiedSystemFramework) const
{
  FileManager &FileMgr = HS.getFileMgr();

  // Framework names must have a '/' in the filename.
  size_t SlashPos = Filename.find('/');
  if (SlashPos == StringRef::npos) return 0;

  // Find out if this is the home for the specified framework, by checking
  // HeaderSearch.  Possible answers are yes/no and unknown.
  HeaderSearch::FrameworkCacheEntry &CacheEntry =
    HS.LookupFrameworkCache(Filename.substr(0, SlashPos));

  // If it is known and in some other directory, fail.
  if (CacheEntry.Directory && CacheEntry.Directory != getFrameworkDir())
    return 0;

  // Otherwise, construct the path to this framework dir.

  // FrameworkName = "/System/Library/Frameworks/"
  SmallString<1024> FrameworkName;
  FrameworkName += getFrameworkDir()->getName();
  if (FrameworkName.empty() || FrameworkName.back() != '/')
    FrameworkName.push_back('/');

  // FrameworkName = "/System/Library/Frameworks/Cocoa"
  StringRef ModuleName(Filename.begin(), SlashPos);
  FrameworkName += ModuleName;

  // FrameworkName = "/System/Library/Frameworks/Cocoa.framework/"
  FrameworkName += ".framework/";

  // If the cache entry was unresolved, populate it now.
  if (CacheEntry.Directory == 0) {
    HS.IncrementFrameworkLookupCount();

    // If the framework dir doesn't exist, we fail.
    const DirectoryEntry *Dir = FileMgr.getDirectory(FrameworkName.str());
    if (Dir == 0) return 0;

    // Otherwise, if it does, remember that this is the right direntry for this
    // framework.
    CacheEntry.Directory = getFrameworkDir();

    // If this is a user search directory, check if the framework has been
    // user-specified as a system framework.
    if (getDirCharacteristic() == SrcMgr::C_User) {
      SmallString<1024> SystemFrameworkMarker(FrameworkName);
      SystemFrameworkMarker += ".system_framework";
      if (llvm::sys::fs::exists(SystemFrameworkMarker.str())) {
        CacheEntry.IsUserSpecifiedSystemFramework = true;
      }
    }
  }

  // Set the 'user-specified system framework' flag.
  InUserSpecifiedSystemFramework = CacheEntry.IsUserSpecifiedSystemFramework;

  if (RelativePath != NULL) {
    RelativePath->clear();
    RelativePath->append(Filename.begin()+SlashPos+1, Filename.end());
  }

  // If we're allowed to look for modules, try to load or create the module
  // corresponding to this framework.
  Module *Module = 0;
  if (SuggestedModule) {
    if (const DirectoryEntry *FrameworkDir
                                        = FileMgr.getDirectory(FrameworkName)) {
      bool IsSystem = getDirCharacteristic() != SrcMgr::C_User;
      Module = HS.loadFrameworkModule(ModuleName, FrameworkDir, IsSystem);
    }
  }
  
  // Check "/System/Library/Frameworks/Cocoa.framework/Headers/file.h"
  unsigned OrigSize = FrameworkName.size();

  FrameworkName += "Headers/";

  if (SearchPath != NULL) {
    SearchPath->clear();
    // Without trailing '/'.
    SearchPath->append(FrameworkName.begin(), FrameworkName.end()-1);
  }

  // Determine whether this is the module we're building or not.
  bool AutomaticImport = Module;  
  FrameworkName.append(Filename.begin()+SlashPos+1, Filename.end());
  if (const FileEntry *FE = FileMgr.getFile(FrameworkName.str(),
                                            /*openFile=*/!AutomaticImport)) {
    if (AutomaticImport)
      *SuggestedModule = HS.findModuleForHeader(FE);
    return FE;
  }

  // Check "/System/Library/Frameworks/Cocoa.framework/PrivateHeaders/file.h"
  const char *Private = "Private";
  FrameworkName.insert(FrameworkName.begin()+OrigSize, Private,
                       Private+strlen(Private));
  if (SearchPath != NULL)
    SearchPath->insert(SearchPath->begin()+OrigSize, Private,
                       Private+strlen(Private));

  const FileEntry *FE = FileMgr.getFile(FrameworkName.str(), 
                                        /*openFile=*/!AutomaticImport);
  if (FE && AutomaticImport)
    *SuggestedModule = HS.findModuleForHeader(FE);
  return FE;
}

void HeaderSearch::setTarget(const TargetInfo &Target) {
  ModMap.setTarget(Target);
}


//===----------------------------------------------------------------------===//
// Header File Location.
//===----------------------------------------------------------------------===//


/// LookupFile - Given a "foo" or \<foo> reference, look up the indicated file,
/// return null on failure.  isAngled indicates whether the file reference is
/// for system \#include's or not (i.e. using <> instead of "").  CurFileEnt, if
/// non-null, indicates where the \#including file is, in case a relative search
/// is needed.
const FileEntry *HeaderSearch::LookupFile(
    StringRef Filename,
    bool isAngled,
    const DirectoryLookup *FromDir,
    const DirectoryLookup *&CurDir,
    const FileEntry *CurFileEnt,
    SmallVectorImpl<char> *SearchPath,
    SmallVectorImpl<char> *RelativePath,
    Module **SuggestedModule,
    bool SkipCache)
{
  if (SuggestedModule)
    *SuggestedModule = 0;
    
  // If 'Filename' is absolute, check to see if it exists and no searching.
  if (llvm::sys::path::is_absolute(Filename)) {
    CurDir = 0;

    // If this was an #include_next "/absolute/file", fail.
    if (FromDir) return 0;

    if (SearchPath != NULL)
      SearchPath->clear();
    if (RelativePath != NULL) {
      RelativePath->clear();
      RelativePath->append(Filename.begin(), Filename.end());
    }
    // Otherwise, just return the file.
    return FileMgr.getFile(Filename, /*openFile=*/true);
  }

  // Unless disabled, check to see if the file is in the #includer's
  // directory.  This has to be based on CurFileEnt, not CurDir, because
  // CurFileEnt could be a #include of a subdirectory (#include "foo/bar.h") and
  // a subsequent include of "baz.h" should resolve to "whatever/foo/baz.h".
  // This search is not done for <> headers.
  if (CurFileEnt && !isAngled && !NoCurDirSearch) {
    SmallString<1024> TmpDir;
    // Concatenate the requested file onto the directory.
    // FIXME: Portability.  Filename concatenation should be in sys::Path.
    TmpDir += CurFileEnt->getDir()->getName();
    TmpDir.push_back('/');
    TmpDir.append(Filename.begin(), Filename.end());
    if (const FileEntry *FE = FileMgr.getFile(TmpDir.str(),/*openFile=*/true)) {
      // Leave CurDir unset.
      // This file is a system header or C++ unfriendly if the old file is.
      //
      // Note that we only use one of FromHFI/ToHFI at once, due to potential
      // reallocation of the underlying vector potentially making the first
      // reference binding dangling.
      HeaderFileInfo &FromHFI = getFileInfo(CurFileEnt);
      unsigned DirInfo = FromHFI.DirInfo;
      bool IndexHeaderMapHeader = FromHFI.IndexHeaderMapHeader;
      StringRef Framework = FromHFI.Framework;

      HeaderFileInfo &ToHFI = getFileInfo(FE);
      ToHFI.DirInfo = DirInfo;
      ToHFI.IndexHeaderMapHeader = IndexHeaderMapHeader;
      ToHFI.Framework = Framework;

      if (SearchPath != NULL) {
        StringRef SearchPathRef(CurFileEnt->getDir()->getName());
        SearchPath->clear();
        SearchPath->append(SearchPathRef.begin(), SearchPathRef.end());
      }
      if (RelativePath != NULL) {
        RelativePath->clear();
        RelativePath->append(Filename.begin(), Filename.end());
      }
      return FE;
    }
  }

  CurDir = 0;

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
  std::pair<unsigned, unsigned> &CacheLookup =
    LookupFileCache.GetOrCreateValue(Filename).getValue();

  // If the entry has been previously looked up, the first value will be
  // non-zero.  If the value is equal to i (the start point of our search), then
  // this is a matching hit.
  if (!SkipCache && CacheLookup.first == i+1) {
    // Skip querying potentially lots of directories for this lookup.
    i = CacheLookup.second;
  } else {
    // Otherwise, this is the first query, or the previous query didn't match
    // our search start.  We will fill in our found location below, so prime the
    // start point value.
    CacheLookup.first = i+1;
  }

  // Check each directory in sequence to see if it contains this file.
  for (; i != SearchDirs.size(); ++i) {
    bool InUserSpecifiedSystemFramework = false;
    const FileEntry *FE =
      SearchDirs[i].LookupFile(Filename, *this, SearchPath, RelativePath,
                               SuggestedModule, InUserSpecifiedSystemFramework);
    if (!FE) continue;

    CurDir = &SearchDirs[i];

    // This file is a system header or C++ unfriendly if the dir is.
    HeaderFileInfo &HFI = getFileInfo(FE);
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

    // If this file is found in a header map and uses the framework style of
    // includes, then this header is part of a framework we're building.
    if (CurDir->isIndexHeaderMap()) {
      size_t SlashPos = Filename.find('/');
      if (SlashPos != StringRef::npos) {
        HFI.IndexHeaderMapHeader = 1;
        HFI.Framework = getUniqueFrameworkName(StringRef(Filename.begin(), 
                                                         SlashPos));
      }
    }
    
    // Remember this location for the next lookup we do.
    CacheLookup.second = i;
    return FE;
  }

  // If we are including a file with a quoted include "foo.h" from inside
  // a header in a framework that is currently being built, and we couldn't
  // resolve "foo.h" any other way, change the include to <Foo/foo.h>, where
  // "Foo" is the name of the framework in which the including header was found.
  if (CurFileEnt && !isAngled && Filename.find('/') == StringRef::npos) {
    HeaderFileInfo &IncludingHFI = getFileInfo(CurFileEnt);
    if (IncludingHFI.IndexHeaderMapHeader) {
      SmallString<128> ScratchFilename;
      ScratchFilename += IncludingHFI.Framework;
      ScratchFilename += '/';
      ScratchFilename += Filename;
      
      const FileEntry *Result = LookupFile(ScratchFilename, /*isAngled=*/true,
                                           FromDir, CurDir, CurFileEnt, 
                                           SearchPath, RelativePath,
                                           SuggestedModule);
      std::pair<unsigned, unsigned> &CacheLookup 
        = LookupFileCache.GetOrCreateValue(Filename).getValue();
      CacheLookup.second
        = LookupFileCache.GetOrCreateValue(ScratchFilename).getValue().second;
      return Result;
    }
  }

  // Otherwise, didn't find it. Remember we didn't find this.
  CacheLookup.second = SearchDirs.size();
  return 0;
}

/// LookupSubframeworkHeader - Look up a subframework for the specified
/// \#include file.  For example, if \#include'ing <HIToolbox/HIToolbox.h> from
/// within ".../Carbon.framework/Headers/Carbon.h", check to see if HIToolbox
/// is a subframework within Carbon.framework.  If so, return the FileEntry
/// for the designated file, otherwise return null.
const FileEntry *HeaderSearch::
LookupSubframeworkHeader(StringRef Filename,
                         const FileEntry *ContextFileEnt,
                         SmallVectorImpl<char> *SearchPath,
                         SmallVectorImpl<char> *RelativePath) {
  assert(ContextFileEnt && "No context file?");

  // Framework names must have a '/' in the filename.  Find it.
  // FIXME: Should we permit '\' on Windows?
  size_t SlashPos = Filename.find('/');
  if (SlashPos == StringRef::npos) return 0;

  // Look up the base framework name of the ContextFileEnt.
  const char *ContextName = ContextFileEnt->getName();

  // If the context info wasn't a framework, couldn't be a subframework.
  const unsigned DotFrameworkLen = 10;
  const char *FrameworkPos = strstr(ContextName, ".framework");
  if (FrameworkPos == 0 || 
      (FrameworkPos[DotFrameworkLen] != '/' && 
       FrameworkPos[DotFrameworkLen] != '\\'))
    return 0;

  SmallString<1024> FrameworkName(ContextName, FrameworkPos+DotFrameworkLen+1);

  // Append Frameworks/HIToolbox.framework/
  FrameworkName += "Frameworks/";
  FrameworkName.append(Filename.begin(), Filename.begin()+SlashPos);
  FrameworkName += ".framework/";

  llvm::StringMapEntry<FrameworkCacheEntry> &CacheLookup =
    FrameworkMap.GetOrCreateValue(Filename.substr(0, SlashPos));

  // Some other location?
  if (CacheLookup.getValue().Directory &&
      CacheLookup.getKeyLength() == FrameworkName.size() &&
      memcmp(CacheLookup.getKeyData(), &FrameworkName[0],
             CacheLookup.getKeyLength()) != 0)
    return 0;

  // Cache subframework.
  if (CacheLookup.getValue().Directory == 0) {
    ++NumSubFrameworkLookups;

    // If the framework dir doesn't exist, we fail.
    const DirectoryEntry *Dir = FileMgr.getDirectory(FrameworkName.str());
    if (Dir == 0) return 0;

    // Otherwise, if it does, remember that this is the right direntry for this
    // framework.
    CacheLookup.getValue().Directory = Dir;
  }

  const FileEntry *FE = 0;

  if (RelativePath != NULL) {
    RelativePath->clear();
    RelativePath->append(Filename.begin()+SlashPos+1, Filename.end());
  }

  // Check ".../Frameworks/HIToolbox.framework/Headers/HIToolbox.h"
  SmallString<1024> HeadersFilename(FrameworkName);
  HeadersFilename += "Headers/";
  if (SearchPath != NULL) {
    SearchPath->clear();
    // Without trailing '/'.
    SearchPath->append(HeadersFilename.begin(), HeadersFilename.end()-1);
  }

  HeadersFilename.append(Filename.begin()+SlashPos+1, Filename.end());
  if (!(FE = FileMgr.getFile(HeadersFilename.str(), /*openFile=*/true))) {

    // Check ".../Frameworks/HIToolbox.framework/PrivateHeaders/HIToolbox.h"
    HeadersFilename = FrameworkName;
    HeadersFilename += "PrivateHeaders/";
    if (SearchPath != NULL) {
      SearchPath->clear();
      // Without trailing '/'.
      SearchPath->append(HeadersFilename.begin(), HeadersFilename.end()-1);
    }

    HeadersFilename.append(Filename.begin()+SlashPos+1, Filename.end());
    if (!(FE = FileMgr.getFile(HeadersFilename.str(), /*openFile=*/true)))
      return 0;
  }

  // This file is a system header or C++ unfriendly if the old file is.
  //
  // Note that the temporary 'DirInfo' is required here, as either call to
  // getFileInfo could resize the vector and we don't want to rely on order
  // of evaluation.
  unsigned DirInfo = getFileInfo(ContextFileEnt).DirInfo;
  getFileInfo(FE).DirInfo = DirInfo;
  return FE;
}

/// \brief Helper static function to normalize a path for injection into
/// a synthetic header.
/*static*/ std::string
HeaderSearch::NormalizeDashIncludePath(StringRef File, FileManager &FileMgr) {
  // Implicit include paths should be resolved relative to the current
  // working directory first, and then use the regular header search
  // mechanism. The proper way to handle this is to have the
  // predefines buffer located at the current working directory, but
  // it has no file entry. For now, workaround this by using an
  // absolute path if we find the file here, and otherwise letting
  // header search handle it.
  SmallString<128> Path(File);
  llvm::sys::fs::make_absolute(Path);
  bool exists;
  if (llvm::sys::fs::exists(Path.str(), exists) || !exists)
    Path = File;
  else if (exists)
    FileMgr.getFile(File);

  return Lexer::Stringify(Path.str());
}

//===----------------------------------------------------------------------===//
// File Info Management.
//===----------------------------------------------------------------------===//

/// \brief Merge the header file info provided by \p OtherHFI into the current
/// header file info (\p HFI)
static void mergeHeaderFileInfo(HeaderFileInfo &HFI, 
                                const HeaderFileInfo &OtherHFI) {
  HFI.isImport |= OtherHFI.isImport;
  HFI.isPragmaOnce |= OtherHFI.isPragmaOnce;
  HFI.NumIncludes += OtherHFI.NumIncludes;
  
  if (!HFI.ControllingMacro && !HFI.ControllingMacroID) {
    HFI.ControllingMacro = OtherHFI.ControllingMacro;
    HFI.ControllingMacroID = OtherHFI.ControllingMacroID;
  }
  
  if (OtherHFI.External) {
    HFI.DirInfo = OtherHFI.DirInfo;
    HFI.External = OtherHFI.External;
    HFI.IndexHeaderMapHeader = OtherHFI.IndexHeaderMapHeader;
  }

  if (HFI.Framework.empty())
    HFI.Framework = OtherHFI.Framework;
  
  HFI.Resolved = true;
}
                                
/// getFileInfo - Return the HeaderFileInfo structure for the specified
/// FileEntry.
HeaderFileInfo &HeaderSearch::getFileInfo(const FileEntry *FE) {
  if (FE->getUID() >= FileInfo.size())
    FileInfo.resize(FE->getUID()+1);
  
  HeaderFileInfo &HFI = FileInfo[FE->getUID()];
  if (ExternalSource && !HFI.Resolved)
    mergeHeaderFileInfo(HFI, ExternalSource->GetHeaderFileInfo(FE));
  return HFI;
}

bool HeaderSearch::isFileMultipleIncludeGuarded(const FileEntry *File) {
  // Check if we've ever seen this file as a header.
  if (File->getUID() >= FileInfo.size())
    return false;

  // Resolve header file info from the external source, if needed.
  HeaderFileInfo &HFI = FileInfo[File->getUID()];
  if (ExternalSource && !HFI.Resolved)
    mergeHeaderFileInfo(HFI, ExternalSource->GetHeaderFileInfo(File));

  return HFI.isPragmaOnce || HFI.ControllingMacro || HFI.ControllingMacroID;
}

void HeaderSearch::setHeaderFileInfoForUID(HeaderFileInfo HFI, unsigned UID) {
  if (UID >= FileInfo.size())
    FileInfo.resize(UID+1);
  HFI.Resolved = true;
  FileInfo[UID] = HFI;
}

bool HeaderSearch::ShouldEnterIncludeFile(const FileEntry *File, bool isImport){
  ++NumIncluded; // Count # of attempted #includes.

  // Get information about this file.
  HeaderFileInfo &FileInfo = getFileInfo(File);

  // If this is a #import directive, check that we have not already imported
  // this header.
  if (isImport) {
    // If this has already been imported, don't import it again.
    FileInfo.isImport = true;

    // Has this already been #import'ed or #include'd?
    if (FileInfo.NumIncludes) return false;
  } else {
    // Otherwise, if this is a #include of a file that was previously #import'd
    // or if this is the second #include of a #pragma once file, ignore it.
    if (FileInfo.isImport)
      return false;
  }

  // Next, check to see if the file is wrapped with #ifndef guards.  If so, and
  // if the macro that guards it is defined, we know the #include has no effect.
  if (const IdentifierInfo *ControllingMacro
      = FileInfo.getControllingMacro(ExternalLookup))
    if (ControllingMacro->hasMacroDefinition()) {
      ++NumMultiIncludeFileOptzn;
      return false;
    }

  // Increment the number of times this file has been included.
  ++FileInfo.NumIncludes;

  return true;
}

size_t HeaderSearch::getTotalMemory() const {
  return SearchDirs.capacity()
    + llvm::capacity_in_bytes(FileInfo)
    + llvm::capacity_in_bytes(HeaderMaps)
    + LookupFileCache.getAllocator().getTotalMemory()
    + FrameworkMap.getAllocator().getTotalMemory();
}

StringRef HeaderSearch::getUniqueFrameworkName(StringRef Framework) {
  return FrameworkNames.GetOrCreateValue(Framework).getKey();
}

bool HeaderSearch::hasModuleMap(StringRef FileName, 
                                const DirectoryEntry *Root) {
  llvm::SmallVector<const DirectoryEntry *, 2> FixUpDirectories;
  
  StringRef DirName = FileName;
  do {
    // Get the parent directory name.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      return false;
    
    // Determine whether this directory exists.
    const DirectoryEntry *Dir = FileMgr.getDirectory(DirName);
    if (!Dir)
      return false;
    
    // Try to load the module map file in this directory.
    switch (loadModuleMapFile(Dir)) {
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
    if (Dir == Root)
      return false;
        
    // Keep track of all of the directories we checked, so we can mark them as
    // having module maps if we eventually do find a module map.
    FixUpDirectories.push_back(Dir);
  } while (true);
}

Module *HeaderSearch::findModuleForHeader(const FileEntry *File) {
  if (Module *Mod = ModMap.findModuleForHeader(File))
    return Mod;
  
  return 0;
}

bool HeaderSearch::loadModuleMapFile(const FileEntry *File) {
  const DirectoryEntry *Dir = File->getDir();
  
  llvm::DenseMap<const DirectoryEntry *, bool>::iterator KnownDir
    = DirectoryHasModuleMap.find(Dir);
  if (KnownDir != DirectoryHasModuleMap.end())
    return !KnownDir->second;
  
  bool Result = ModMap.parseModuleMapFile(File);
  if (!Result && llvm::sys::path::filename(File->getName()) == "module.map") {
    // If the file we loaded was a module.map, look for the corresponding
    // module_private.map.
    SmallString<128> PrivateFilename(Dir->getName());
    llvm::sys::path::append(PrivateFilename, "module_private.map");
    if (const FileEntry *PrivateFile = FileMgr.getFile(PrivateFilename))
      Result = ModMap.parseModuleMapFile(PrivateFile);
  }
  
  DirectoryHasModuleMap[Dir] = !Result;  
  return Result;
}

Module *HeaderSearch::loadFrameworkModule(StringRef Name, 
                                          const DirectoryEntry *Dir,
                                          bool IsSystem) {
  if (Module *Module = ModMap.findModule(Name))
    return Module;
  
  // Try to load a module map file.
  switch (loadModuleMapFile(Dir)) {
  case LMM_InvalidModuleMap:
    break;
    
  case LMM_AlreadyLoaded:
  case LMM_NoDirectory:
    return 0;
    
  case LMM_NewlyLoaded:
    return ModMap.findModule(Name);
  }

  // The top-level framework directory, from which we'll infer a framework
  // module.
  const DirectoryEntry *TopFrameworkDir = Dir;
  
  // The path from the module we're actually looking for back to the top-level
  // framework name.
  llvm::SmallVector<StringRef, 2> SubmodulePath;
  SubmodulePath.push_back(Name);
  
  // Walk the directory structure to find any enclosing frameworks.
#ifdef LLVM_ON_UNIX
  // Note: as an egregious but useful hack we use the real path here, because
  // frameworks moving from top-level frameworks to embedded frameworks tend
  // to be symlinked from the top-level location to the embedded location,
  // and we need to resolve lookups as if we had found the embedded location.
  char RealDirName[PATH_MAX];
  StringRef DirName;
  if (realpath(Dir->getName(), RealDirName))
    DirName = RealDirName;
  else
    DirName = Dir->getName();
#else
  StringRef DirName = Dir->getName();
#endif
  do {
    // Get the parent directory name.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      break;
    
    // Determine whether this directory exists.
    Dir = FileMgr.getDirectory(DirName);
    if (!Dir)
      break;
    
    // If this is a framework directory, then we're a subframework of this
    // framework.
    if (llvm::sys::path::extension(DirName) == ".framework") {
      SubmodulePath.push_back(llvm::sys::path::stem(DirName));
      TopFrameworkDir = Dir;
    }
  } while (true);

  // Determine whether we're allowed to infer a module map.
  bool canInfer = false;
  if (llvm::sys::path::has_parent_path(TopFrameworkDir->getName())) {
    // Figure out the parent path.
    StringRef Parent = llvm::sys::path::parent_path(TopFrameworkDir->getName());
    if (const DirectoryEntry *ParentDir = FileMgr.getDirectory(Parent)) {
      // If there's a module map file in the parent directory, it can
      // explicitly allow us to infer framework modules.
      switch (loadModuleMapFile(ParentDir)) {
        case LMM_AlreadyLoaded:
        case LMM_NewlyLoaded: {
          StringRef Name = llvm::sys::path::stem(TopFrameworkDir->getName());
          canInfer = ModMap.canInferFrameworkModule(ParentDir, Name, IsSystem);
          break;
        }
        case LMM_InvalidModuleMap:
        case LMM_NoDirectory:
          break;
      }
    }
  }

  // If we're not allowed to infer a module map, we're done.
  if (!canInfer)
    return 0;

  // Try to infer a module map from the top-level framework directory.
  Module *Result = ModMap.inferFrameworkModule(SubmodulePath.back(), 
                                               TopFrameworkDir,
                                               IsSystem,
                                               /*Parent=*/0);
  
  // Follow the submodule path to find the requested (sub)framework module
  // within the top-level framework module.
  SubmodulePath.pop_back();
  while (!SubmodulePath.empty() && Result) {
    Result = ModMap.lookupModuleQualified(SubmodulePath.back(), Result);
    SubmodulePath.pop_back();
  }
  return Result;
}


HeaderSearch::LoadModuleMapResult 
HeaderSearch::loadModuleMapFile(StringRef DirName) {
  if (const DirectoryEntry *Dir = FileMgr.getDirectory(DirName))
    return loadModuleMapFile(Dir);
  
  return LMM_NoDirectory;
}

HeaderSearch::LoadModuleMapResult 
HeaderSearch::loadModuleMapFile(const DirectoryEntry *Dir) {
  llvm::DenseMap<const DirectoryEntry *, bool>::iterator KnownDir
    = DirectoryHasModuleMap.find(Dir);
  if (KnownDir != DirectoryHasModuleMap.end())
    return KnownDir->second? LMM_AlreadyLoaded : LMM_InvalidModuleMap;
  
  SmallString<128> ModuleMapFileName;
  ModuleMapFileName += Dir->getName();
  unsigned ModuleMapDirNameLen = ModuleMapFileName.size();
  llvm::sys::path::append(ModuleMapFileName, "module.map");
  if (const FileEntry *ModuleMapFile = FileMgr.getFile(ModuleMapFileName)) {
    // We have found a module map file. Try to parse it.
    if (ModMap.parseModuleMapFile(ModuleMapFile)) {
      // No suitable module map.
      DirectoryHasModuleMap[Dir] = false;
      return LMM_InvalidModuleMap;
    }

    // This directory has a module map.
    DirectoryHasModuleMap[Dir] = true;
    
    // Check whether there is a private module map that we need to load as well.
    ModuleMapFileName.erase(ModuleMapFileName.begin() + ModuleMapDirNameLen,
                            ModuleMapFileName.end());
    llvm::sys::path::append(ModuleMapFileName, "module_private.map");
    if (const FileEntry *PrivateModuleMapFile
                                        = FileMgr.getFile(ModuleMapFileName)) {
      if (ModMap.parseModuleMapFile(PrivateModuleMapFile)) {
        // No suitable module map.
        DirectoryHasModuleMap[Dir] = false;
        return LMM_InvalidModuleMap;
      }      
    }
    
    return LMM_NewlyLoaded;
  }
  
  // No suitable module map.
  DirectoryHasModuleMap[Dir] = false;
  return LMM_InvalidModuleMap;
}

void HeaderSearch::collectAllModules(llvm::SmallVectorImpl<Module *> &Modules) {
  Modules.clear();
  
  // Load module maps for each of the header search directories.
  for (unsigned Idx = 0, N = SearchDirs.size(); Idx != N; ++Idx) {
    if (SearchDirs[Idx].isFramework()) {
      llvm::error_code EC;
      SmallString<128> DirNative;
      llvm::sys::path::native(SearchDirs[Idx].getFrameworkDir()->getName(),
                              DirNative);
      
      // Search each of the ".framework" directories to load them as modules.
      bool IsSystem = SearchDirs[Idx].getDirCharacteristic() != SrcMgr::C_User;
      for (llvm::sys::fs::directory_iterator Dir(DirNative.str(), EC), DirEnd;
           Dir != DirEnd && !EC; Dir.increment(EC)) {
        if (llvm::sys::path::extension(Dir->path()) != ".framework")
          continue;
        
        const DirectoryEntry *FrameworkDir = FileMgr.getDirectory(Dir->path());
        if (!FrameworkDir)
          continue;
        
        // Load this framework module.
        loadFrameworkModule(llvm::sys::path::stem(Dir->path()), FrameworkDir,
                            IsSystem);
      }
      continue;
    }
    
    // FIXME: Deal with header maps.
    if (SearchDirs[Idx].isHeaderMap())
      continue;
    
    // Try to load a module map file for the search directory.
    loadModuleMapFile(SearchDirs[Idx].getDir());
    
    // Try to load module map files for immediate subdirectories of this search
    // directory.
    llvm::error_code EC;
    SmallString<128> DirNative;
    llvm::sys::path::native(SearchDirs[Idx].getDir()->getName(), DirNative);
    for (llvm::sys::fs::directory_iterator Dir(DirNative.str(), EC), DirEnd;
         Dir != DirEnd && !EC; Dir.increment(EC)) {
      loadModuleMapFile(Dir->path());
    }
  }
  
  // Populate the list of modules.
  for (ModuleMap::module_iterator M = ModMap.module_begin(), 
                               MEnd = ModMap.module_end();
       M != MEnd; ++M) {
    Modules.push_back(M->getValue());
  }
}
