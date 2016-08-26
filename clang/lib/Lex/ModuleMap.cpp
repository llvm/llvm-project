//===--- ModuleMap.cpp - Describe the layout of modules ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ModuleMap implementation, which describes the layout
// of a module as it relates to headers.
//
//===----------------------------------------------------------------------===//
#include "clang/Lex/ModuleMap.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/LiteralSupport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <stdlib.h>
#if defined(LLVM_ON_UNIX)
#include <limits.h>
#endif
using namespace clang;

Module::ExportDecl 
ModuleMap::resolveExport(Module *Mod, 
                         const Module::UnresolvedExportDecl &Unresolved,
                         bool Complain) const {
  // We may have just a wildcard.
  if (Unresolved.Id.empty()) {
    assert(Unresolved.Wildcard && "Invalid unresolved export");
    return Module::ExportDecl(nullptr, true);
  }
  
  // Resolve the module-id.
  Module *Context = resolveModuleId(Unresolved.Id, Mod, Complain);
  if (!Context)
    return Module::ExportDecl();

  return Module::ExportDecl(Context, Unresolved.Wildcard);
}

Module *ModuleMap::resolveModuleId(const ModuleId &Id, Module *Mod,
                                   bool Complain) const {
  // Find the starting module.
  Module *Context = lookupModuleUnqualified(Id[0].first, Mod);
  if (!Context) {
    if (Complain)
      Diags.Report(Id[0].second, diag::err_mmap_missing_module_unqualified)
      << Id[0].first << Mod->getFullModuleName();

    return nullptr;
  }

  // Dig into the module path.
  for (unsigned I = 1, N = Id.size(); I != N; ++I) {
    Module *Sub = lookupModuleQualified(Id[I].first, Context);
    if (!Sub) {
      if (Complain)
        Diags.Report(Id[I].second, diag::err_mmap_missing_module_qualified)
        << Id[I].first << Context->getFullModuleName()
        << SourceRange(Id[0].second, Id[I-1].second);

      return nullptr;
    }

    Context = Sub;
  }

  return Context;
}

ModuleMap::ModuleMap(SourceManager &SourceMgr, DiagnosticsEngine &Diags,
                     const LangOptions &LangOpts, const TargetInfo *Target,
                     HeaderSearch &HeaderInfo)
    : SourceMgr(SourceMgr), Diags(Diags), LangOpts(LangOpts), Target(Target),
      HeaderInfo(HeaderInfo), BuiltinIncludeDir(nullptr),
      SourceModule(nullptr), NumCreatedModules(0) {
  MMapLangOpts.LineComment = true;
}

ModuleMap::~ModuleMap() {
  for (auto &M : Modules)
    delete M.getValue();
}

void ModuleMap::setTarget(const TargetInfo &Target) {
  assert((!this->Target || this->Target == &Target) && 
         "Improper target override");
  this->Target = &Target;
}

/// \brief "Sanitize" a filename so that it can be used as an identifier.
static StringRef sanitizeFilenameAsIdentifier(StringRef Name,
                                              SmallVectorImpl<char> &Buffer) {
  if (Name.empty())
    return Name;

  if (!isValidIdentifier(Name)) {
    // If we don't already have something with the form of an identifier,
    // create a buffer with the sanitized name.
    Buffer.clear();
    if (isDigit(Name[0]))
      Buffer.push_back('_');
    Buffer.reserve(Buffer.size() + Name.size());
    for (unsigned I = 0, N = Name.size(); I != N; ++I) {
      if (isIdentifierBody(Name[I]))
        Buffer.push_back(Name[I]);
      else
        Buffer.push_back('_');
    }

    Name = StringRef(Buffer.data(), Buffer.size());
  }

  while (llvm::StringSwitch<bool>(Name)
#define KEYWORD(Keyword,Conditions) .Case(#Keyword, true)
#define ALIAS(Keyword, AliasOf, Conditions) .Case(Keyword, true)
#include "clang/Basic/TokenKinds.def"
           .Default(false)) {
    if (Name.data() != Buffer.data())
      Buffer.append(Name.begin(), Name.end());
    Buffer.push_back('_');
    Name = StringRef(Buffer.data(), Buffer.size());
  }

  return Name;
}

/// \brief Determine whether the given file name is the name of a builtin
/// header, supplied by Clang to replace, override, or augment existing system
/// headers.
static bool isBuiltinHeader(StringRef FileName) {
  return llvm::StringSwitch<bool>(FileName)
           .Case("float.h", true)
           .Case("iso646.h", true)
           .Case("limits.h", true)
           .Case("stdalign.h", true)
           .Case("stdarg.h", true)
           .Case("stdatomic.h", true)
           .Case("stdbool.h", true)
           .Case("stddef.h", true)
           .Case("stdint.h", true)
           .Case("tgmath.h", true)
           .Case("unwind.h", true)
           .Default(false);
}

ModuleMap::HeadersMap::iterator
ModuleMap::findKnownHeader(const FileEntry *File) {
  HeadersMap::iterator Known = Headers.find(File);
  if (HeaderInfo.getHeaderSearchOpts().ImplicitModuleMaps &&
      Known == Headers.end() && File->getDir() == BuiltinIncludeDir &&
      isBuiltinHeader(llvm::sys::path::filename(File->getName()))) {
    HeaderInfo.loadTopLevelSystemModules();
    return Headers.find(File);
  }
  return Known;
}

ModuleMap::KnownHeader
ModuleMap::findHeaderInUmbrellaDirs(const FileEntry *File,
                    SmallVectorImpl<const DirectoryEntry *> &IntermediateDirs) {
  if (UmbrellaDirs.empty())
    return KnownHeader();

  const DirectoryEntry *Dir = File->getDir();
  assert(Dir && "file in no directory");

  // Note: as an egregious but useful hack we use the real path here, because
  // frameworks moving from top-level frameworks to embedded frameworks tend
  // to be symlinked from the top-level location to the embedded location,
  // and we need to resolve lookups as if we had found the embedded location.
  StringRef DirName = SourceMgr.getFileManager().getCanonicalName(Dir);

  // Keep walking up the directory hierarchy, looking for a directory with
  // an umbrella header.
  do {
    auto KnownDir = UmbrellaDirs.find(Dir);
    if (KnownDir != UmbrellaDirs.end())
      return KnownHeader(KnownDir->second, NormalHeader);

    IntermediateDirs.push_back(Dir);

    // Retrieve our parent path.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      break;

    // Resolve the parent path to a directory entry.
    Dir = SourceMgr.getFileManager().getDirectory(DirName);
  } while (Dir);
  return KnownHeader();
}

static bool violatesPrivateInclude(Module *RequestingModule,
                                   const FileEntry *IncFileEnt,
                                   ModuleMap::KnownHeader Header) {
#ifndef NDEBUG
  if (Header.getRole() & ModuleMap::PrivateHeader) {
    // Check for consistency between the module header role
    // as obtained from the lookup and as obtained from the module.
    // This check is not cheap, so enable it only for debugging.
    bool IsPrivate = false;
    SmallVectorImpl<Module::Header> *HeaderList[] = {
        &Header.getModule()->Headers[Module::HK_Private],
        &Header.getModule()->Headers[Module::HK_PrivateTextual]};
    for (auto *Hs : HeaderList)
      IsPrivate |=
          std::find_if(Hs->begin(), Hs->end(), [&](const Module::Header &H) {
            return H.Entry == IncFileEnt;
          }) != Hs->end();
    assert(IsPrivate && "inconsistent headers and roles");
  }
#endif
  return !Header.isAccessibleFrom(RequestingModule);
}

static Module *getTopLevelOrNull(Module *M) {
  return M ? M->getTopLevelModule() : nullptr;
}

void ModuleMap::diagnoseHeaderInclusion(Module *RequestingModule,
                                        bool RequestingModuleIsModuleInterface,
                                        SourceLocation FilenameLoc,
                                        StringRef Filename,
                                        const FileEntry *File) {
  // No errors for indirect modules. This may be a bit of a problem for modules
  // with no source files.
  if (getTopLevelOrNull(RequestingModule) != getTopLevelOrNull(SourceModule))
    return;

  if (RequestingModule)
    resolveUses(RequestingModule, /*Complain=*/false);

  bool Excluded = false;
  Module *Private = nullptr;
  Module *NotUsed = nullptr;

  HeadersMap::iterator Known = findKnownHeader(File);
  if (Known != Headers.end()) {
    for (const KnownHeader &Header : Known->second) {
      // Remember private headers for later printing of a diagnostic.
      if (violatesPrivateInclude(RequestingModule, File, Header)) {
        Private = Header.getModule();
        continue;
      }

      // If uses need to be specified explicitly, we are only allowed to return
      // modules that are explicitly used by the requesting module.
      if (RequestingModule && LangOpts.ModulesDeclUse &&
          !RequestingModule->directlyUses(Header.getModule())) {
        NotUsed = Header.getModule();
        continue;
      }

      // We have found a module that we can happily use.
      return;
    }

    Excluded = true;
  }

  // We have found a header, but it is private.
  if (Private) {
    Diags.Report(FilenameLoc, diag::warn_use_of_private_header_outside_module)
        << Filename;
    return;
  }

  // We have found a module, but we don't use it.
  if (NotUsed) {
    Diags.Report(FilenameLoc, diag::err_undeclared_use_of_module)
        << RequestingModule->getFullModuleName() << Filename;
    return;
  }

  if (Excluded || isHeaderInUmbrellaDirs(File))
    return;

  // At this point, only non-modular includes remain.

  if (LangOpts.ModulesStrictDeclUse) {
    Diags.Report(FilenameLoc, diag::err_undeclared_use_of_module)
        << RequestingModule->getFullModuleName() << Filename;
  } else if (RequestingModule && RequestingModuleIsModuleInterface) {
    diag::kind DiagID = RequestingModule->getTopLevelModule()->IsFramework ?
        diag::warn_non_modular_include_in_framework_module :
        diag::warn_non_modular_include_in_module;
    Diags.Report(FilenameLoc, DiagID) << RequestingModule->getFullModuleName();
  }
}

static bool isBetterKnownHeader(const ModuleMap::KnownHeader &New,
                                const ModuleMap::KnownHeader &Old) {
  // Prefer available modules.
  if (New.getModule()->isAvailable() && !Old.getModule()->isAvailable())
    return true;

  // Prefer a public header over a private header.
  if ((New.getRole() & ModuleMap::PrivateHeader) !=
      (Old.getRole() & ModuleMap::PrivateHeader))
    return !(New.getRole() & ModuleMap::PrivateHeader);

  // Prefer a non-textual header over a textual header.
  if ((New.getRole() & ModuleMap::TextualHeader) !=
      (Old.getRole() & ModuleMap::TextualHeader))
    return !(New.getRole() & ModuleMap::TextualHeader);

  // Don't have a reason to choose between these. Just keep the first one.
  return false;
}

ModuleMap::KnownHeader ModuleMap::findModuleForHeader(const FileEntry *File) {
  auto MakeResult = [&](ModuleMap::KnownHeader R) -> ModuleMap::KnownHeader {
    if (R.getRole() & ModuleMap::TextualHeader)
      return ModuleMap::KnownHeader();
    return R;
  };

  HeadersMap::iterator Known = findKnownHeader(File);
  if (Known != Headers.end()) {
    ModuleMap::KnownHeader Result;
    // Iterate over all modules that 'File' is part of to find the best fit.
    for (KnownHeader &H : Known->second) {
      // Prefer a header from the source module over all others.
      if (H.getModule()->getTopLevelModule() == SourceModule)
        return MakeResult(H);
      if (!Result || isBetterKnownHeader(H, Result))
        Result = H;
    }
    return MakeResult(Result);
  }

  return MakeResult(findOrCreateModuleForHeaderInUmbrellaDir(File));
}

ModuleMap::KnownHeader
ModuleMap::findOrCreateModuleForHeaderInUmbrellaDir(const FileEntry *File) {
  assert(!Headers.count(File) && "already have a module for this header");

  SmallVector<const DirectoryEntry *, 2> SkippedDirs;
  KnownHeader H = findHeaderInUmbrellaDirs(File, SkippedDirs);
  if (H) {
    Module *Result = H.getModule();

    // Search up the module stack until we find a module with an umbrella
    // directory.
    Module *UmbrellaModule = Result;
    while (!UmbrellaModule->getUmbrellaDir() && UmbrellaModule->Parent)
      UmbrellaModule = UmbrellaModule->Parent;

    if (UmbrellaModule->InferSubmodules) {
      const FileEntry *UmbrellaModuleMap =
          getModuleMapFileForUniquing(UmbrellaModule);

      // Infer submodules for each of the directories we found between
      // the directory of the umbrella header and the directory where
      // the actual header is located.
      bool Explicit = UmbrellaModule->InferExplicitSubmodules;

      for (unsigned I = SkippedDirs.size(); I != 0; --I) {
        // Find or create the module that corresponds to this directory name.
        SmallString<32> NameBuf;
        StringRef Name = sanitizeFilenameAsIdentifier(
            llvm::sys::path::stem(SkippedDirs[I-1]->getName()), NameBuf);
        Result = findOrCreateModule(Name, Result, /*IsFramework=*/false,
                                    Explicit).first;
        InferredModuleAllowedBy[Result] = UmbrellaModuleMap;
        Result->IsInferred = true;

        // Associate the module and the directory.
        UmbrellaDirs[SkippedDirs[I-1]] = Result;

        // If inferred submodules export everything they import, add a
        // wildcard to the set of exports.
        if (UmbrellaModule->InferExportWildcard && Result->Exports.empty())
          Result->Exports.push_back(Module::ExportDecl(nullptr, true));
      }

      // Infer a submodule with the same name as this header file.
      SmallString<32> NameBuf;
      StringRef Name = sanitizeFilenameAsIdentifier(
                         llvm::sys::path::stem(File->getName()), NameBuf);
      Result = findOrCreateModule(Name, Result, /*IsFramework=*/false,
                                  Explicit).first;
      InferredModuleAllowedBy[Result] = UmbrellaModuleMap;
      Result->IsInferred = true;
      Result->addTopHeader(File);

      // If inferred submodules export everything they import, add a
      // wildcard to the set of exports.
      if (UmbrellaModule->InferExportWildcard && Result->Exports.empty())
        Result->Exports.push_back(Module::ExportDecl(nullptr, true));
    } else {
      // Record each of the directories we stepped through as being part of
      // the module we found, since the umbrella header covers them all.
      for (unsigned I = 0, N = SkippedDirs.size(); I != N; ++I)
        UmbrellaDirs[SkippedDirs[I]] = Result;
    }

    KnownHeader Header(Result, NormalHeader);
    Headers[File].push_back(Header);
    return Header;
  }

  return KnownHeader();
}

ArrayRef<ModuleMap::KnownHeader>
ModuleMap::findAllModulesForHeader(const FileEntry *File) const {
  auto It = Headers.find(File);
  if (It == Headers.end())
    return None;
  return It->second;
}

bool ModuleMap::isHeaderInUnavailableModule(const FileEntry *Header) const {
  return isHeaderUnavailableInModule(Header, nullptr);
}

bool
ModuleMap::isHeaderUnavailableInModule(const FileEntry *Header,
                                       const Module *RequestingModule) const {
  HeadersMap::const_iterator Known = Headers.find(Header);
  if (Known != Headers.end()) {
    for (SmallVectorImpl<KnownHeader>::const_iterator
             I = Known->second.begin(),
             E = Known->second.end();
         I != E; ++I) {
      if (I->isAvailable() && (!RequestingModule ||
                               I->getModule()->isSubModuleOf(RequestingModule)))
        return false;
    }
    return true;
  }

  const DirectoryEntry *Dir = Header->getDir();
  SmallVector<const DirectoryEntry *, 2> SkippedDirs;
  StringRef DirName = Dir->getName();

  auto IsUnavailable = [&](const Module *M) {
    return !M->isAvailable() && (!RequestingModule ||
                                 M->isSubModuleOf(RequestingModule));
  };

  // Keep walking up the directory hierarchy, looking for a directory with
  // an umbrella header.
  do {
    llvm::DenseMap<const DirectoryEntry *, Module *>::const_iterator KnownDir
      = UmbrellaDirs.find(Dir);
    if (KnownDir != UmbrellaDirs.end()) {
      Module *Found = KnownDir->second;
      if (IsUnavailable(Found))
        return true;

      // Search up the module stack until we find a module with an umbrella
      // directory.
      Module *UmbrellaModule = Found;
      while (!UmbrellaModule->getUmbrellaDir() && UmbrellaModule->Parent)
        UmbrellaModule = UmbrellaModule->Parent;

      if (UmbrellaModule->InferSubmodules) {
        for (unsigned I = SkippedDirs.size(); I != 0; --I) {
          // Find or create the module that corresponds to this directory name.
          SmallString<32> NameBuf;
          StringRef Name = sanitizeFilenameAsIdentifier(
                             llvm::sys::path::stem(SkippedDirs[I-1]->getName()),
                             NameBuf);
          Found = lookupModuleQualified(Name, Found);
          if (!Found)
            return false;
          if (IsUnavailable(Found))
            return true;
        }
        
        // Infer a submodule with the same name as this header file.
        SmallString<32> NameBuf;
        StringRef Name = sanitizeFilenameAsIdentifier(
                           llvm::sys::path::stem(Header->getName()),
                           NameBuf);
        Found = lookupModuleQualified(Name, Found);
        if (!Found)
          return false;
      }

      return IsUnavailable(Found);
    }
    
    SkippedDirs.push_back(Dir);
    
    // Retrieve our parent path.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      break;
    
    // Resolve the parent path to a directory entry.
    Dir = SourceMgr.getFileManager().getDirectory(DirName);
  } while (Dir);
  
  return false;
}

Module *ModuleMap::findModule(StringRef Name) const {
  llvm::StringMap<Module *>::const_iterator Known = Modules.find(Name);
  if (Known != Modules.end())
    return Known->getValue();

  return nullptr;
}

Module *ModuleMap::lookupModuleUnqualified(StringRef Name,
                                           Module *Context) const {
  for(; Context; Context = Context->Parent) {
    if (Module *Sub = lookupModuleQualified(Name, Context))
      return Sub;
  }
  
  return findModule(Name);
}

Module *ModuleMap::lookupModuleQualified(StringRef Name, Module *Context) const{
  if (!Context)
    return findModule(Name);
  
  return Context->findSubmodule(Name);
}

std::pair<Module *, bool> 
ModuleMap::findOrCreateModule(StringRef Name, Module *Parent, bool IsFramework,
                              bool IsExplicit) {
  // Try to find an existing module with this name.
  if (Module *Sub = lookupModuleQualified(Name, Parent))
    return std::make_pair(Sub, false);
  
  // Create a new module with this name.
  Module *Result = new Module(Name, SourceLocation(), Parent,
                              IsFramework, IsExplicit, NumCreatedModules++);
  if (!Parent) {
    if (LangOpts.CurrentModule == Name)
      SourceModule = Result;
    Modules[Name] = Result;
  }
  return std::make_pair(Result, true);
}

Module *ModuleMap::createModuleForInterfaceUnit(SourceLocation Loc,
                                                StringRef Name) {
  assert(LangOpts.CurrentModule == Name && "module name mismatch");
  assert(!Modules[Name] && "redefining existing module");

  auto *Result =
      new Module(Name, Loc, nullptr, /*IsFramework*/ false,
                 /*IsExplicit*/ false, NumCreatedModules++);
  Modules[Name] = SourceModule = Result;

  // Mark the main source file as being within the newly-created module so that
  // declarations and macros are properly visibility-restricted to it.
  auto *MainFile = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());
  assert(MainFile && "no input file for module interface");
  Headers[MainFile].push_back(KnownHeader(Result, PrivateHeader));

  return Result;
}

/// \brief For a framework module, infer the framework against which we
/// should link.
static void inferFrameworkLink(Module *Mod, const DirectoryEntry *FrameworkDir,
                               FileManager &FileMgr) {
  assert(Mod->IsFramework && "Can only infer linking for framework modules");
  assert(!Mod->isSubFramework() &&
         "Can only infer linking for top-level frameworks");

  SmallString<128> LibName;
  LibName += FrameworkDir->getName();
  llvm::sys::path::append(LibName, Mod->Name);

  // The library name of a framework has more than one possible extension since
  // the introduction of the text-based dynamic library format. We need to check
  // for both before we give up.
  static const char *frameworkExtensions[] = {"", ".tbd"};
  for (const auto *extension : frameworkExtensions) {
    llvm::sys::path::replace_extension(LibName, extension);
    if (FileMgr.getFile(LibName)) {
      Mod->LinkLibraries.push_back(Module::LinkLibrary(Mod->Name,
                                                       /*IsFramework=*/true));
      return;
    }
  }
}

Module *ModuleMap::inferFrameworkModule(const DirectoryEntry *FrameworkDir,
                                        bool IsSystem, Module *Parent) {
  Attributes Attrs;
  Attrs.IsSystem = IsSystem;
  return inferFrameworkModule(FrameworkDir, Attrs, Parent);
}

Module *ModuleMap::inferFrameworkModule(const DirectoryEntry *FrameworkDir,
                                        Attributes Attrs, Module *Parent) {
  // Note: as an egregious but useful hack we use the real path here, because
  // we might be looking at an embedded framework that symlinks out to a
  // top-level framework, and we need to infer as if we were naming the
  // top-level framework.
  StringRef FrameworkDirName =
      SourceMgr.getFileManager().getCanonicalName(FrameworkDir);

  // In case this is a case-insensitive filesystem, use the canonical
  // directory name as the ModuleName, since modules are case-sensitive.
  // FIXME: we should be able to give a fix-it hint for the correct spelling.
  SmallString<32> ModuleNameStorage;
  StringRef ModuleName = sanitizeFilenameAsIdentifier(
      llvm::sys::path::stem(FrameworkDirName), ModuleNameStorage);

  // Check whether we've already found this module.
  if (Module *Mod = lookupModuleQualified(ModuleName, Parent))
    return Mod;
  
  FileManager &FileMgr = SourceMgr.getFileManager();

  // If the framework has a parent path from which we're allowed to infer
  // a framework module, do so.
  const FileEntry *ModuleMapFile = nullptr;
  if (!Parent) {
    // Determine whether we're allowed to infer a module map.
    bool canInfer = false;
    if (llvm::sys::path::has_parent_path(FrameworkDirName)) {
      // Figure out the parent path.
      StringRef Parent = llvm::sys::path::parent_path(FrameworkDirName);
      if (const DirectoryEntry *ParentDir = FileMgr.getDirectory(Parent)) {
        // Check whether we have already looked into the parent directory
        // for a module map.
        llvm::DenseMap<const DirectoryEntry *, InferredDirectory>::const_iterator
          inferred = InferredDirectories.find(ParentDir);
        if (inferred == InferredDirectories.end()) {
          // We haven't looked here before. Load a module map, if there is
          // one.
          bool IsFrameworkDir = Parent.endswith(".framework");
          if (const FileEntry *ModMapFile =
                HeaderInfo.lookupModuleMapFile(ParentDir, IsFrameworkDir)) {
            parseModuleMapFile(ModMapFile, Attrs.IsSystem, ParentDir);
            inferred = InferredDirectories.find(ParentDir);
          }

          if (inferred == InferredDirectories.end())
            inferred = InferredDirectories.insert(
                         std::make_pair(ParentDir, InferredDirectory())).first;
        }

        if (inferred->second.InferModules) {
          // We're allowed to infer for this directory, but make sure it's okay
          // to infer this particular module.
          StringRef Name = llvm::sys::path::stem(FrameworkDirName);
          canInfer = std::find(inferred->second.ExcludedModules.begin(),
                               inferred->second.ExcludedModules.end(),
                               Name) == inferred->second.ExcludedModules.end();

          Attrs.IsSystem |= inferred->second.Attrs.IsSystem;
          Attrs.IsExternC |= inferred->second.Attrs.IsExternC;
          Attrs.IsExhaustive |= inferred->second.Attrs.IsExhaustive;
          ModuleMapFile = inferred->second.ModuleMapFile;
        }
      }
    }

    // If we're not allowed to infer a framework module, don't.
    if (!canInfer)
      return nullptr;
  } else
    ModuleMapFile = getModuleMapFileForUniquing(Parent);


  // Look for an umbrella header.
  SmallString<128> UmbrellaName = StringRef(FrameworkDir->getName());
  llvm::sys::path::append(UmbrellaName, "Headers", ModuleName + ".h");
  const FileEntry *UmbrellaHeader = FileMgr.getFile(UmbrellaName);
  
  // FIXME: If there's no umbrella header, we could probably scan the
  // framework to load *everything*. But, it's not clear that this is a good
  // idea.
  if (!UmbrellaHeader)
    return nullptr;

  Module *Result = new Module(ModuleName, SourceLocation(), Parent,
                              /*IsFramework=*/true, /*IsExplicit=*/false,
                              NumCreatedModules++);
  InferredModuleAllowedBy[Result] = ModuleMapFile;
  Result->IsInferred = true;
  if (!Parent) {
    if (LangOpts.CurrentModule == ModuleName)
      SourceModule = Result;
    Modules[ModuleName] = Result;
  }

  Result->IsSystem |= Attrs.IsSystem;
  Result->IsExternC |= Attrs.IsExternC;
  Result->ConfigMacrosExhaustive |= Attrs.IsExhaustive;
  Result->Directory = FrameworkDir;

  // umbrella header "umbrella-header-name"
  //
  // The "Headers/" component of the name is implied because this is
  // a framework module.
  setUmbrellaHeader(Result, UmbrellaHeader, ModuleName + ".h");
  
  // export *
  Result->Exports.push_back(Module::ExportDecl(nullptr, true));

  // module * { export * }
  Result->InferSubmodules = true;
  Result->InferExportWildcard = true;
  
  // Look for subframeworks.
  std::error_code EC;
  SmallString<128> SubframeworksDirName
    = StringRef(FrameworkDir->getName());
  llvm::sys::path::append(SubframeworksDirName, "Frameworks");
  llvm::sys::path::native(SubframeworksDirName);
  vfs::FileSystem &FS = *FileMgr.getVirtualFileSystem();
  for (vfs::directory_iterator Dir = FS.dir_begin(SubframeworksDirName, EC),
                               DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    if (!StringRef(Dir->getName()).endswith(".framework"))
      continue;

    if (const DirectoryEntry *SubframeworkDir =
            FileMgr.getDirectory(Dir->getName())) {
      // Note: as an egregious but useful hack, we use the real path here and
      // check whether it is actually a subdirectory of the parent directory.
      // This will not be the case if the 'subframework' is actually a symlink
      // out to a top-level framework.
      StringRef SubframeworkDirName = FileMgr.getCanonicalName(SubframeworkDir);
      bool FoundParent = false;
      do {
        // Get the parent directory name.
        SubframeworkDirName
          = llvm::sys::path::parent_path(SubframeworkDirName);
        if (SubframeworkDirName.empty())
          break;

        if (FileMgr.getDirectory(SubframeworkDirName) == FrameworkDir) {
          FoundParent = true;
          break;
        }
      } while (true);

      if (!FoundParent)
        continue;

      // FIXME: Do we want to warn about subframeworks without umbrella headers?
      inferFrameworkModule(SubframeworkDir, Attrs, Result);
    }
  }

  // If the module is a top-level framework, automatically link against the
  // framework.
  if (!Result->isSubFramework()) {
    inferFrameworkLink(Result, FrameworkDir, FileMgr);
  }

  return Result;
}

void ModuleMap::setUmbrellaHeader(Module *Mod, const FileEntry *UmbrellaHeader,
                                  Twine NameAsWritten) {
  Headers[UmbrellaHeader].push_back(KnownHeader(Mod, NormalHeader));
  Mod->Umbrella = UmbrellaHeader;
  Mod->UmbrellaAsWritten = NameAsWritten.str();
  UmbrellaDirs[UmbrellaHeader->getDir()] = Mod;

  // Notify callbacks that we just added a new header.
  for (const auto &Cb : Callbacks)
    Cb->moduleMapAddUmbrellaHeader(&SourceMgr.getFileManager(), UmbrellaHeader);
}

void ModuleMap::setUmbrellaDir(Module *Mod, const DirectoryEntry *UmbrellaDir,
                               Twine NameAsWritten) {
  Mod->Umbrella = UmbrellaDir;
  Mod->UmbrellaAsWritten = NameAsWritten.str();
  UmbrellaDirs[UmbrellaDir] = Mod;
}

static Module::HeaderKind headerRoleToKind(ModuleMap::ModuleHeaderRole Role) {
  switch ((int)Role) {
  default: llvm_unreachable("unknown header role");
  case ModuleMap::NormalHeader:
    return Module::HK_Normal;
  case ModuleMap::PrivateHeader:
    return Module::HK_Private;
  case ModuleMap::TextualHeader:
    return Module::HK_Textual;
  case ModuleMap::PrivateHeader | ModuleMap::TextualHeader:
    return Module::HK_PrivateTextual;
  }
}

void ModuleMap::addHeader(Module *Mod, Module::Header Header,
                          ModuleHeaderRole Role, bool Imported) {
  KnownHeader KH(Mod, Role);

  // Only add each header to the headers list once.
  // FIXME: Should we diagnose if a header is listed twice in the
  // same module definition?
  auto &HeaderList = Headers[Header.Entry];
  for (auto H : HeaderList)
    if (H == KH)
      return;

  HeaderList.push_back(KH);
  Mod->Headers[headerRoleToKind(Role)].push_back(std::move(Header));

  bool isCompilingModuleHeader =
      LangOpts.isCompilingModule() && Mod->getTopLevelModule() == SourceModule;
  if (!Imported || isCompilingModuleHeader) {
    // When we import HeaderFileInfo, the external source is expected to
    // set the isModuleHeader flag itself.
    HeaderInfo.MarkFileModuleHeader(Header.Entry, Role,
                                    isCompilingModuleHeader);
  }

  // Notify callbacks that we just added a new header.
  for (const auto &Cb : Callbacks)
    Cb->moduleMapAddHeader(Header.Entry->getName());
}

void ModuleMap::excludeHeader(Module *Mod, Module::Header Header) {
  // Add this as a known header so we won't implicitly add it to any
  // umbrella directory module.
  // FIXME: Should we only exclude it from umbrella modules within the
  // specified module?
  (void) Headers[Header.Entry];

  Mod->Headers[Module::HK_Excluded].push_back(std::move(Header));
}

const FileEntry *
ModuleMap::getContainingModuleMapFile(const Module *Module) const {
  if (Module->DefinitionLoc.isInvalid())
    return nullptr;

  return SourceMgr.getFileEntryForID(
           SourceMgr.getFileID(Module->DefinitionLoc));
}

const FileEntry *ModuleMap::getModuleMapFileForUniquing(const Module *M) const {
  if (M->IsInferred) {
    assert(InferredModuleAllowedBy.count(M) && "missing inferred module map");
    return InferredModuleAllowedBy.find(M)->second;
  }
  return getContainingModuleMapFile(M);
}

void ModuleMap::setInferredModuleAllowedBy(Module *M, const FileEntry *ModMap) {
  assert(M->IsInferred && "module not inferred");
  InferredModuleAllowedBy[M] = ModMap;
}

LLVM_DUMP_METHOD void ModuleMap::dump() {
  llvm::errs() << "Modules:";
  for (llvm::StringMap<Module *>::iterator M = Modules.begin(), 
                                        MEnd = Modules.end(); 
       M != MEnd; ++M)
    M->getValue()->print(llvm::errs(), 2);
  
  llvm::errs() << "Headers:";
  for (HeadersMap::iterator H = Headers.begin(), HEnd = Headers.end();
       H != HEnd; ++H) {
    llvm::errs() << "  \"" << H->first->getName() << "\" -> ";
    for (SmallVectorImpl<KnownHeader>::const_iterator I = H->second.begin(),
                                                      E = H->second.end();
         I != E; ++I) {
      if (I != H->second.begin())
        llvm::errs() << ",";
      llvm::errs() << I->getModule()->getFullModuleName();
    }
    llvm::errs() << "\n";
  }
}

bool ModuleMap::resolveExports(Module *Mod, bool Complain) {
  auto Unresolved = std::move(Mod->UnresolvedExports);
  Mod->UnresolvedExports.clear();
  for (auto &UE : Unresolved) {
    Module::ExportDecl Export = resolveExport(Mod, UE, Complain);
    if (Export.getPointer() || Export.getInt())
      Mod->Exports.push_back(Export);
    else
      Mod->UnresolvedExports.push_back(UE);
  }
  return !Mod->UnresolvedExports.empty();
}

bool ModuleMap::resolveUses(Module *Mod, bool Complain) {
  auto Unresolved = std::move(Mod->UnresolvedDirectUses);
  Mod->UnresolvedDirectUses.clear();
  for (auto &UDU : Unresolved) {
    Module *DirectUse = resolveModuleId(UDU, Mod, Complain);
    if (DirectUse)
      Mod->DirectUses.push_back(DirectUse);
    else
      Mod->UnresolvedDirectUses.push_back(UDU);
  }
  return !Mod->UnresolvedDirectUses.empty();
}

bool ModuleMap::resolveConflicts(Module *Mod, bool Complain) {
  auto Unresolved = std::move(Mod->UnresolvedConflicts);
  Mod->UnresolvedConflicts.clear();
  for (auto &UC : Unresolved) {
    if (Module *OtherMod = resolveModuleId(UC.Id, Mod, Complain)) {
      Module::Conflict Conflict;
      Conflict.Other = OtherMod;
      Conflict.Message = UC.Message;
      Mod->Conflicts.push_back(Conflict);
    } else
      Mod->UnresolvedConflicts.push_back(UC);
  }
  return !Mod->UnresolvedConflicts.empty();
}

Module *ModuleMap::inferModuleFromLocation(FullSourceLoc Loc) {
  if (Loc.isInvalid())
    return nullptr;

  if (UmbrellaDirs.empty() && Headers.empty())
    return nullptr;

  // Use the expansion location to determine which module we're in.
  FullSourceLoc ExpansionLoc = Loc.getExpansionLoc();
  if (!ExpansionLoc.isFileID())
    return nullptr;

  const SourceManager &SrcMgr = Loc.getManager();
  FileID ExpansionFileID = ExpansionLoc.getFileID();
  
  while (const FileEntry *ExpansionFile
           = SrcMgr.getFileEntryForID(ExpansionFileID)) {
    // Find the module that owns this header (if any).
    if (Module *Mod = findModuleForHeader(ExpansionFile).getModule())
      return Mod;
    
    // No module owns this header, so look up the inclusion chain to see if
    // any included header has an associated module.
    SourceLocation IncludeLoc = SrcMgr.getIncludeLoc(ExpansionFileID);
    if (IncludeLoc.isInvalid())
      return nullptr;

    ExpansionFileID = SrcMgr.getFileID(IncludeLoc);
  }

  return nullptr;
}

//----------------------------------------------------------------------------//
// Module map file parser
//----------------------------------------------------------------------------//

namespace clang {
  /// \brief A token in a module map file.
  struct MMToken {
    enum TokenKind {
      Comma,
      ConfigMacros,
      Conflict,
      EndOfFile,
      HeaderKeyword,
      Identifier,
      Exclaim,
      ExcludeKeyword,
      ExplicitKeyword,
      ExportKeyword,
      ExternKeyword,
      FrameworkKeyword,
      LinkKeyword,
      ModuleKeyword,
      Period,
      PrivateKeyword,
      UmbrellaKeyword,
      UseKeyword,
      RequiresKeyword,
      Star,
      StringLiteral,
      TextualKeyword,
      LBrace,
      RBrace,
      LSquare,
      RSquare
    } Kind;
    
    unsigned Location;
    unsigned StringLength;
    const char *StringData;
    
    void clear() {
      Kind = EndOfFile;
      Location = 0;
      StringLength = 0;
      StringData = nullptr;
    }
    
    bool is(TokenKind K) const { return Kind == K; }
    
    SourceLocation getLocation() const {
      return SourceLocation::getFromRawEncoding(Location);
    }
    
    StringRef getString() const {
      return StringRef(StringData, StringLength);
    }
  };

  class ModuleMapParser {
    Lexer &L;
    SourceManager &SourceMgr;

    /// \brief Default target information, used only for string literal
    /// parsing.
    const TargetInfo *Target;

    DiagnosticsEngine &Diags;
    ModuleMap &Map;

    /// \brief The current module map file.
    const FileEntry *ModuleMapFile;
    
    /// \brief The directory that file names in this module map file should
    /// be resolved relative to.
    const DirectoryEntry *Directory;

    /// \brief The directory containing Clang-supplied headers.
    const DirectoryEntry *BuiltinIncludeDir;

    /// \brief Whether this module map is in a system header directory.
    bool IsSystem;
    
    /// \brief Whether an error occurred.
    bool HadError;
        
    /// \brief Stores string data for the various string literals referenced
    /// during parsing.
    llvm::BumpPtrAllocator StringData;
    
    /// \brief The current token.
    MMToken Tok;
    
    /// \brief The active module.
    Module *ActiveModule;

    /// \brief Whether a module uses the 'requires excluded' hack to mark its
    /// contents as 'textual'.
    ///
    /// On older Darwin SDK versions, 'requires excluded' is used to mark the
    /// contents of the Darwin.C.excluded (assert.h) and Tcl.Private modules as
    /// non-modular headers.  For backwards compatibility, we continue to
    /// support this idiom for just these modules, and map the headers to
    /// 'textual' to match the original intent.
    llvm::SmallPtrSet<Module *, 2> UsesRequiresExcludedHack;

    /// \brief Consume the current token and return its location.
    SourceLocation consumeToken();
    
    /// \brief Skip tokens until we reach the a token with the given kind
    /// (or the end of the file).
    void skipUntil(MMToken::TokenKind K);

    typedef SmallVector<std::pair<std::string, SourceLocation>, 2> ModuleId;
    bool parseModuleId(ModuleId &Id);
    void parseModuleDecl();
    void parseExternModuleDecl();
    void parseRequiresDecl();
    void parseHeaderDecl(clang::MMToken::TokenKind,
                         SourceLocation LeadingLoc);
    void parseUmbrellaDirDecl(SourceLocation UmbrellaLoc);
    void parseExportDecl();
    void parseUseDecl();
    void parseLinkDecl();
    void parseConfigMacros();
    void parseConflict();
    void parseInferredModuleDecl(bool Framework, bool Explicit);

    typedef ModuleMap::Attributes Attributes;
    bool parseOptionalAttributes(Attributes &Attrs);
    
  public:
    explicit ModuleMapParser(Lexer &L, SourceManager &SourceMgr, 
                             const TargetInfo *Target,
                             DiagnosticsEngine &Diags,
                             ModuleMap &Map,
                             const FileEntry *ModuleMapFile,
                             const DirectoryEntry *Directory,
                             const DirectoryEntry *BuiltinIncludeDir,
                             bool IsSystem)
      : L(L), SourceMgr(SourceMgr), Target(Target), Diags(Diags), Map(Map), 
        ModuleMapFile(ModuleMapFile), Directory(Directory),
        BuiltinIncludeDir(BuiltinIncludeDir), IsSystem(IsSystem),
        HadError(false), ActiveModule(nullptr)
    {
      Tok.clear();
      consumeToken();
    }
    
    bool parseModuleMapFile();
  };
}

SourceLocation ModuleMapParser::consumeToken() {
retry:
  SourceLocation Result = Tok.getLocation();
  Tok.clear();
  
  Token LToken;
  L.LexFromRawLexer(LToken);
  Tok.Location = LToken.getLocation().getRawEncoding();
  switch (LToken.getKind()) {
  case tok::raw_identifier: {
    StringRef RI = LToken.getRawIdentifier();
    Tok.StringData = RI.data();
    Tok.StringLength = RI.size();
    Tok.Kind = llvm::StringSwitch<MMToken::TokenKind>(RI)
                 .Case("config_macros", MMToken::ConfigMacros)
                 .Case("conflict", MMToken::Conflict)
                 .Case("exclude", MMToken::ExcludeKeyword)
                 .Case("explicit", MMToken::ExplicitKeyword)
                 .Case("export", MMToken::ExportKeyword)
                 .Case("extern", MMToken::ExternKeyword)
                 .Case("framework", MMToken::FrameworkKeyword)
                 .Case("header", MMToken::HeaderKeyword)
                 .Case("link", MMToken::LinkKeyword)
                 .Case("module", MMToken::ModuleKeyword)
                 .Case("private", MMToken::PrivateKeyword)
                 .Case("requires", MMToken::RequiresKeyword)
                 .Case("textual", MMToken::TextualKeyword)
                 .Case("umbrella", MMToken::UmbrellaKeyword)
                 .Case("use", MMToken::UseKeyword)
                 .Default(MMToken::Identifier);
    break;
  }

  case tok::comma:
    Tok.Kind = MMToken::Comma;
    break;

  case tok::eof:
    Tok.Kind = MMToken::EndOfFile;
    break;
      
  case tok::l_brace:
    Tok.Kind = MMToken::LBrace;
    break;

  case tok::l_square:
    Tok.Kind = MMToken::LSquare;
    break;
      
  case tok::period:
    Tok.Kind = MMToken::Period;
    break;
      
  case tok::r_brace:
    Tok.Kind = MMToken::RBrace;
    break;
      
  case tok::r_square:
    Tok.Kind = MMToken::RSquare;
    break;
      
  case tok::star:
    Tok.Kind = MMToken::Star;
    break;
      
  case tok::exclaim:
    Tok.Kind = MMToken::Exclaim;
    break;
      
  case tok::string_literal: {
    if (LToken.hasUDSuffix()) {
      Diags.Report(LToken.getLocation(), diag::err_invalid_string_udl);
      HadError = true;
      goto retry;
    }

    // Parse the string literal.
    LangOptions LangOpts;
    StringLiteralParser StringLiteral(LToken, SourceMgr, LangOpts, *Target);
    if (StringLiteral.hadError)
      goto retry;
    
    // Copy the string literal into our string data allocator.
    unsigned Length = StringLiteral.GetStringLength();
    char *Saved = StringData.Allocate<char>(Length + 1);
    memcpy(Saved, StringLiteral.GetString().data(), Length);
    Saved[Length] = 0;
    
    // Form the token.
    Tok.Kind = MMToken::StringLiteral;
    Tok.StringData = Saved;
    Tok.StringLength = Length;
    break;
  }
      
  case tok::comment:
    goto retry;
      
  default:
    Diags.Report(LToken.getLocation(), diag::err_mmap_unknown_token);
    HadError = true;
    goto retry;
  }
  
  return Result;
}

void ModuleMapParser::skipUntil(MMToken::TokenKind K) {
  unsigned braceDepth = 0;
  unsigned squareDepth = 0;
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
      return;

    case MMToken::LBrace:
      if (Tok.is(K) && braceDepth == 0 && squareDepth == 0)
        return;
        
      ++braceDepth;
      break;

    case MMToken::LSquare:
      if (Tok.is(K) && braceDepth == 0 && squareDepth == 0)
        return;
      
      ++squareDepth;
      break;

    case MMToken::RBrace:
      if (braceDepth > 0)
        --braceDepth;
      else if (Tok.is(K))
        return;
      break;

    case MMToken::RSquare:
      if (squareDepth > 0)
        --squareDepth;
      else if (Tok.is(K))
        return;
      break;

    default:
      if (braceDepth == 0 && squareDepth == 0 && Tok.is(K))
        return;
      break;
    }
    
   consumeToken();
  } while (true);
}

/// \brief Parse a module-id.
///
///   module-id:
///     identifier
///     identifier '.' module-id
///
/// \returns true if an error occurred, false otherwise.
bool ModuleMapParser::parseModuleId(ModuleId &Id) {
  Id.clear();
  do {
    if (Tok.is(MMToken::Identifier) || Tok.is(MMToken::StringLiteral)) {
      Id.push_back(std::make_pair(Tok.getString(), Tok.getLocation()));
      consumeToken();
    } else {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module_name);
      return true;
    }
    
    if (!Tok.is(MMToken::Period))
      break;
    
    consumeToken();
  } while (true);
  
  return false;
}

namespace {
  /// \brief Enumerates the known attributes.
  enum AttributeKind {
    /// \brief An unknown attribute.
    AT_unknown,
    /// \brief The 'system' attribute.
    AT_system,
    /// \brief The 'extern_c' attribute.
    AT_extern_c,
    /// \brief The 'exhaustive' attribute.
    AT_exhaustive
  };
}

/// \brief Parse a module declaration.
///
///   module-declaration:
///     'extern' 'module' module-id string-literal
///     'explicit'[opt] 'framework'[opt] 'module' module-id attributes[opt] 
///       { module-member* }
///
///   module-member:
///     requires-declaration
///     header-declaration
///     submodule-declaration
///     export-declaration
///     link-declaration
///
///   submodule-declaration:
///     module-declaration
///     inferred-submodule-declaration
void ModuleMapParser::parseModuleDecl() {
  assert(Tok.is(MMToken::ExplicitKeyword) || Tok.is(MMToken::ModuleKeyword) ||
         Tok.is(MMToken::FrameworkKeyword) || Tok.is(MMToken::ExternKeyword));
  if (Tok.is(MMToken::ExternKeyword)) {
    parseExternModuleDecl();
    return;
  }

  // Parse 'explicit' or 'framework' keyword, if present.
  SourceLocation ExplicitLoc;
  bool Explicit = false;
  bool Framework = false;

  // Parse 'explicit' keyword, if present.
  if (Tok.is(MMToken::ExplicitKeyword)) {
    ExplicitLoc = consumeToken();
    Explicit = true;
  }

  // Parse 'framework' keyword, if present.
  if (Tok.is(MMToken::FrameworkKeyword)) {
    consumeToken();
    Framework = true;
  } 
  
  // Parse 'module' keyword.
  if (!Tok.is(MMToken::ModuleKeyword)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module);
    consumeToken();
    HadError = true;
    return;
  }
  consumeToken(); // 'module' keyword

  // If we have a wildcard for the module name, this is an inferred submodule.
  // Parse it. 
  if (Tok.is(MMToken::Star))
    return parseInferredModuleDecl(Framework, Explicit);
  
  // Parse the module name.
  ModuleId Id;
  if (parseModuleId(Id)) {
    HadError = true;
    return;
  }

  if (ActiveModule) {
    if (Id.size() > 1) {
      Diags.Report(Id.front().second, diag::err_mmap_nested_submodule_id)
        << SourceRange(Id.front().second, Id.back().second);
      
      HadError = true;
      return;
    }
  } else if (Id.size() == 1 && Explicit) {
    // Top-level modules can't be explicit.
    Diags.Report(ExplicitLoc, diag::err_mmap_explicit_top_level);
    Explicit = false;
    ExplicitLoc = SourceLocation();
    HadError = true;
  }
  
  Module *PreviousActiveModule = ActiveModule;  
  if (Id.size() > 1) {
    // This module map defines a submodule. Go find the module of which it
    // is a submodule.
    ActiveModule = nullptr;
    const Module *TopLevelModule = nullptr;
    for (unsigned I = 0, N = Id.size() - 1; I != N; ++I) {
      if (Module *Next = Map.lookupModuleQualified(Id[I].first, ActiveModule)) {
        if (I == 0)
          TopLevelModule = Next;
        ActiveModule = Next;
        continue;
      }
      
      if (ActiveModule) {
        Diags.Report(Id[I].second, diag::err_mmap_missing_module_qualified)
          << Id[I].first
          << ActiveModule->getTopLevelModule()->getFullModuleName();
      } else {
        Diags.Report(Id[I].second, diag::err_mmap_expected_module_name);
      }
      HadError = true;
      return;
    }

    if (ModuleMapFile != Map.getContainingModuleMapFile(TopLevelModule)) {
      assert(ModuleMapFile != Map.getModuleMapFileForUniquing(TopLevelModule) &&
             "submodule defined in same file as 'module *' that allowed its "
             "top-level module");
      Map.addAdditionalModuleMapFile(TopLevelModule, ModuleMapFile);
    }
  }
  
  StringRef ModuleName = Id.back().first;
  SourceLocation ModuleNameLoc = Id.back().second;
  
  // Parse the optional attribute list.
  Attributes Attrs;
  if (parseOptionalAttributes(Attrs))
    return;

  
  // Parse the opening brace.
  if (!Tok.is(MMToken::LBrace)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_lbrace)
      << ModuleName;
    HadError = true;
    return;
  }  
  SourceLocation LBraceLoc = consumeToken();
  
  // Determine whether this (sub)module has already been defined.
  if (Module *Existing = Map.lookupModuleQualified(ModuleName, ActiveModule)) {
    if (Existing->DefinitionLoc.isInvalid() && !ActiveModule) {
      // Skip the module definition.
      skipUntil(MMToken::RBrace);
      if (Tok.is(MMToken::RBrace))
        consumeToken();
      else {
        Diags.Report(Tok.getLocation(), diag::err_mmap_expected_rbrace);
        Diags.Report(LBraceLoc, diag::note_mmap_lbrace_match);
        HadError = true;        
      }
      return;
    }
    
    Diags.Report(ModuleNameLoc, diag::err_mmap_module_redefinition)
      << ModuleName;
    Diags.Report(Existing->DefinitionLoc, diag::note_mmap_prev_definition);
    
    // Skip the module definition.
    skipUntil(MMToken::RBrace);
    if (Tok.is(MMToken::RBrace))
      consumeToken();
    
    HadError = true;
    return;
  }

  // Start defining this module.
  ActiveModule = Map.findOrCreateModule(ModuleName, ActiveModule, Framework,
                                        Explicit).first;
  ActiveModule->DefinitionLoc = ModuleNameLoc;
  if (Attrs.IsSystem || IsSystem)
    ActiveModule->IsSystem = true;
  if (Attrs.IsExternC)
    ActiveModule->IsExternC = true;
  ActiveModule->Directory = Directory;

  bool Done = false;
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
    case MMToken::RBrace:
      Done = true;
      break;

    case MMToken::ConfigMacros:
      parseConfigMacros();
      break;

    case MMToken::Conflict:
      parseConflict();
      break;

    case MMToken::ExplicitKeyword:
    case MMToken::ExternKeyword:
    case MMToken::FrameworkKeyword:
    case MMToken::ModuleKeyword:
      parseModuleDecl();
      break;

    case MMToken::ExportKeyword:
      parseExportDecl();
      break;

    case MMToken::UseKeyword:
      parseUseDecl();
      break;
        
    case MMToken::RequiresKeyword:
      parseRequiresDecl();
      break;

    case MMToken::TextualKeyword:
      parseHeaderDecl(MMToken::TextualKeyword, consumeToken());
      break;

    case MMToken::UmbrellaKeyword: {
      SourceLocation UmbrellaLoc = consumeToken();
      if (Tok.is(MMToken::HeaderKeyword))
        parseHeaderDecl(MMToken::UmbrellaKeyword, UmbrellaLoc);
      else
        parseUmbrellaDirDecl(UmbrellaLoc);
      break;
    }

    case MMToken::ExcludeKeyword:
      parseHeaderDecl(MMToken::ExcludeKeyword, consumeToken());
      break;

    case MMToken::PrivateKeyword:
      parseHeaderDecl(MMToken::PrivateKeyword, consumeToken());
      break;

    case MMToken::HeaderKeyword:
      parseHeaderDecl(MMToken::HeaderKeyword, consumeToken());
      break;

    case MMToken::LinkKeyword:
      parseLinkDecl();
      break;

    default:
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_member);
      consumeToken();
      break;        
    }
  } while (!Done);

  if (Tok.is(MMToken::RBrace))
    consumeToken();
  else {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_rbrace);
    Diags.Report(LBraceLoc, diag::note_mmap_lbrace_match);
    HadError = true;
  }

  // If the active module is a top-level framework, and there are no link
  // libraries, automatically link against the framework.
  if (ActiveModule->IsFramework && !ActiveModule->isSubFramework() &&
      ActiveModule->LinkLibraries.empty()) {
    inferFrameworkLink(ActiveModule, Directory, SourceMgr.getFileManager());
  }

  // If the module meets all requirements but is still unavailable, mark the
  // whole tree as unavailable to prevent it from building.
  if (!ActiveModule->IsAvailable && !ActiveModule->IsMissingRequirement &&
      ActiveModule->Parent) {
    ActiveModule->getTopLevelModule()->markUnavailable();
    ActiveModule->getTopLevelModule()->MissingHeaders.append(
      ActiveModule->MissingHeaders.begin(), ActiveModule->MissingHeaders.end());
  }

  // We're done parsing this module. Pop back to the previous module.
  ActiveModule = PreviousActiveModule;
}

/// \brief Parse an extern module declaration.
///
///   extern module-declaration:
///     'extern' 'module' module-id string-literal
void ModuleMapParser::parseExternModuleDecl() {
  assert(Tok.is(MMToken::ExternKeyword));
  SourceLocation ExternLoc = consumeToken(); // 'extern' keyword

  // Parse 'module' keyword.
  if (!Tok.is(MMToken::ModuleKeyword)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module);
    consumeToken();
    HadError = true;
    return;
  }
  consumeToken(); // 'module' keyword

  // Parse the module name.
  ModuleId Id;
  if (parseModuleId(Id)) {
    HadError = true;
    return;
  }

  // Parse the referenced module map file name.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_mmap_file);
    HadError = true;
    return;
  }
  std::string FileName = Tok.getString();
  consumeToken(); // filename

  StringRef FileNameRef = FileName;
  SmallString<128> ModuleMapFileName;
  if (llvm::sys::path::is_relative(FileNameRef)) {
    ModuleMapFileName += Directory->getName();
    llvm::sys::path::append(ModuleMapFileName, FileName);
    FileNameRef = ModuleMapFileName;
  }
  if (const FileEntry *File = SourceMgr.getFileManager().getFile(FileNameRef))
    Map.parseModuleMapFile(
        File, /*IsSystem=*/false,
        Map.HeaderInfo.getHeaderSearchOpts().ModuleMapFileHomeIsCwd
            ? Directory
            : File->getDir(), ExternLoc);
}

/// Whether to add the requirement \p Feature to the module \p M.
///
/// This preserves backwards compatibility for two hacks in the Darwin system
/// module map files:
///
/// 1. The use of 'requires excluded' to make headers non-modular, which
///    should really be mapped to 'textual' now that we have this feature.  We
///    drop the 'excluded' requirement, and set \p IsRequiresExcludedHack to
///    true.  Later, this bit will be used to map all the headers inside this
///    module to 'textual'.
///
///    This affects Darwin.C.excluded (for assert.h) and Tcl.Private.
///
/// 2. Removes a bogus cplusplus requirement from IOKit.avc.  This requirement
///    was never correct and causes issues now that we check it, so drop it.
static bool shouldAddRequirement(Module *M, StringRef Feature,
                                 bool &IsRequiresExcludedHack) {
  static const StringRef DarwinCExcluded[] = {"Darwin", "C", "excluded"};
  static const StringRef TclPrivate[] = {"Tcl", "Private"};
  static const StringRef IOKitAVC[] = {"IOKit", "avc"};

  if (Feature == "excluded" && (M->fullModuleNameIs(DarwinCExcluded) ||
                                M->fullModuleNameIs(TclPrivate))) {
    IsRequiresExcludedHack = true;
    return false;
  } else if (Feature == "cplusplus" && M->fullModuleNameIs(IOKitAVC)) {
    return false;
  }

  return true;
}

/// \brief Parse a requires declaration.
///
///   requires-declaration:
///     'requires' feature-list
///
///   feature-list:
///     feature ',' feature-list
///     feature
///
///   feature:
///     '!'[opt] identifier
void ModuleMapParser::parseRequiresDecl() {
  assert(Tok.is(MMToken::RequiresKeyword));

  // Parse 'requires' keyword.
  consumeToken();

  // Parse the feature-list.
  do {
    bool RequiredState = true;
    if (Tok.is(MMToken::Exclaim)) {
      RequiredState = false;
      consumeToken();
    }

    if (!Tok.is(MMToken::Identifier)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_feature);
      HadError = true;
      return;
    }

    // Consume the feature name.
    std::string Feature = Tok.getString();
    consumeToken();

    bool IsRequiresExcludedHack = false;
    bool ShouldAddRequirement =
        shouldAddRequirement(ActiveModule, Feature, IsRequiresExcludedHack);

    if (IsRequiresExcludedHack)
      UsesRequiresExcludedHack.insert(ActiveModule);

    if (ShouldAddRequirement) {
      // Add this feature.
      ActiveModule->addRequirement(Feature, RequiredState, Map.LangOpts,
                                   *Map.Target);
    }

    if (!Tok.is(MMToken::Comma))
      break;

    // Consume the comma.
    consumeToken();
  } while (true);
}

/// \brief Append to \p Paths the set of paths needed to get to the 
/// subframework in which the given module lives.
static void appendSubframeworkPaths(Module *Mod,
                                    SmallVectorImpl<char> &Path) {
  // Collect the framework names from the given module to the top-level module.
  SmallVector<StringRef, 2> Paths;
  for (; Mod; Mod = Mod->Parent) {
    if (Mod->IsFramework)
      Paths.push_back(Mod->Name);
  }
  
  if (Paths.empty())
    return;
  
  // Add Frameworks/Name.framework for each subframework.
  for (unsigned I = Paths.size() - 1; I != 0; --I)
    llvm::sys::path::append(Path, "Frameworks", Paths[I-1] + ".framework");
}

/// \brief Parse a header declaration.
///
///   header-declaration:
///     'textual'[opt] 'header' string-literal
///     'private' 'textual'[opt] 'header' string-literal
///     'exclude' 'header' string-literal
///     'umbrella' 'header' string-literal
///
/// FIXME: Support 'private textual header'.
void ModuleMapParser::parseHeaderDecl(MMToken::TokenKind LeadingToken,
                                      SourceLocation LeadingLoc) {
  // We've already consumed the first token.
  ModuleMap::ModuleHeaderRole Role = ModuleMap::NormalHeader;
  if (LeadingToken == MMToken::PrivateKeyword) {
    Role = ModuleMap::PrivateHeader;
    // 'private' may optionally be followed by 'textual'.
    if (Tok.is(MMToken::TextualKeyword)) {
      LeadingToken = Tok.Kind;
      consumeToken();
    }
  }

  if (LeadingToken == MMToken::TextualKeyword)
    Role = ModuleMap::ModuleHeaderRole(Role | ModuleMap::TextualHeader);

  if (UsesRequiresExcludedHack.count(ActiveModule)) {
    // Mark this header 'textual' (see doc comment for
    // Module::UsesRequiresExcludedHack).
    Role = ModuleMap::ModuleHeaderRole(Role | ModuleMap::TextualHeader);
  }

  if (LeadingToken != MMToken::HeaderKeyword) {
    if (!Tok.is(MMToken::HeaderKeyword)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_header)
          << (LeadingToken == MMToken::PrivateKeyword ? "private" :
              LeadingToken == MMToken::ExcludeKeyword ? "exclude" :
              LeadingToken == MMToken::TextualKeyword ? "textual" : "umbrella");
      return;
    }
    consumeToken();
  }

  // Parse the header name.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_header) 
      << "header";
    HadError = true;
    return;
  }
  Module::UnresolvedHeaderDirective Header;
  Header.FileName = Tok.getString();
  Header.FileNameLoc = consumeToken();
  
  // Check whether we already have an umbrella.
  if (LeadingToken == MMToken::UmbrellaKeyword && ActiveModule->Umbrella) {
    Diags.Report(Header.FileNameLoc, diag::err_mmap_umbrella_clash)
      << ActiveModule->getFullModuleName();
    HadError = true;
    return;
  }

  // Look for this file.
  const FileEntry *File = nullptr;
  const FileEntry *BuiltinFile = nullptr;
  SmallString<128> RelativePathName;
  if (llvm::sys::path::is_absolute(Header.FileName)) {
    RelativePathName = Header.FileName;
    File = SourceMgr.getFileManager().getFile(RelativePathName);
  } else {
    // Search for the header file within the search directory.
    SmallString<128> FullPathName(Directory->getName());
    unsigned FullPathLength = FullPathName.size();
    
    if (ActiveModule->isPartOfFramework()) {
      appendSubframeworkPaths(ActiveModule, RelativePathName);
      
      // Check whether this file is in the public headers.
      llvm::sys::path::append(RelativePathName, "Headers", Header.FileName);
      llvm::sys::path::append(FullPathName, RelativePathName);
      File = SourceMgr.getFileManager().getFile(FullPathName);
      
      if (!File) {
        // Check whether this file is in the private headers.
        // FIXME: Should we retain the subframework paths here?
        RelativePathName.clear();
        FullPathName.resize(FullPathLength);
        llvm::sys::path::append(RelativePathName, "PrivateHeaders",
                                Header.FileName);
        llvm::sys::path::append(FullPathName, RelativePathName);
        File = SourceMgr.getFileManager().getFile(FullPathName);
      }
    } else {
      // Lookup for normal headers.
      llvm::sys::path::append(RelativePathName, Header.FileName);
      llvm::sys::path::append(FullPathName, RelativePathName);
      File = SourceMgr.getFileManager().getFile(FullPathName);

      // If this is a system module with a top-level header, this header
      // may have a counterpart (or replacement) in the set of headers
      // supplied by Clang. Find that builtin header.
      if (ActiveModule->IsSystem && LeadingToken != MMToken::UmbrellaKeyword &&
          BuiltinIncludeDir && BuiltinIncludeDir != Directory &&
          isBuiltinHeader(Header.FileName)) {
        SmallString<128> BuiltinPathName(BuiltinIncludeDir->getName());
        llvm::sys::path::append(BuiltinPathName, Header.FileName);
        BuiltinFile = SourceMgr.getFileManager().getFile(BuiltinPathName);

        // If Clang supplies this header but the underlying system does not,
        // just silently swap in our builtin version. Otherwise, we'll end
        // up adding both (later).
        //
        // For local visibility, entirely replace the system file with our
        // one and textually include the system one. We need to pass macros
        // from our header to the system one if we #include_next it.
        //
        // FIXME: Can we do this in all cases?
        if (BuiltinFile && (!File || Map.LangOpts.ModulesLocalVisibility)) {
          File = BuiltinFile;
          RelativePathName = BuiltinPathName;
          BuiltinFile = nullptr;
        }
      }
    }
  }

  // FIXME: We shouldn't be eagerly stat'ing every file named in a module map.
  // Come up with a lazy way to do this.
  if (File) {
    if (LeadingToken == MMToken::UmbrellaKeyword) {
      const DirectoryEntry *UmbrellaDir = File->getDir();
      if (Module *UmbrellaModule = Map.UmbrellaDirs[UmbrellaDir]) {
        Diags.Report(LeadingLoc, diag::err_mmap_umbrella_clash)
          << UmbrellaModule->getFullModuleName();
        HadError = true;
      } else {
        // Record this umbrella header.
        Map.setUmbrellaHeader(ActiveModule, File, RelativePathName.str());
      }
    } else if (LeadingToken == MMToken::ExcludeKeyword) {
      Module::Header H = {RelativePathName.str(), File};
      Map.excludeHeader(ActiveModule, H);
    } else {
      // If there is a builtin counterpart to this file, add it now, before
      // the "real" header, so we build the built-in one first when building
      // the module.
      if (BuiltinFile) {
        // FIXME: Taking the name from the FileEntry is unstable and can give
        // different results depending on how we've previously named that file
        // in this build.
        Module::Header H = { BuiltinFile->getName(), BuiltinFile };
        Map.addHeader(ActiveModule, H, Role);
      }

      // Record this header.
      Module::Header H = { RelativePathName.str(), File };
      Map.addHeader(ActiveModule, H, Role);
    }
  } else if (LeadingToken != MMToken::ExcludeKeyword) {
    // Ignore excluded header files. They're optional anyway.

    // If we find a module that has a missing header, we mark this module as
    // unavailable and store the header directive for displaying diagnostics.
    Header.IsUmbrella = LeadingToken == MMToken::UmbrellaKeyword;
    ActiveModule->markUnavailable();
    ActiveModule->MissingHeaders.push_back(Header);
  }
}

static int compareModuleHeaders(const Module::Header *A,
                                const Module::Header *B) {
  return A->NameAsWritten.compare(B->NameAsWritten);
}

/// \brief Parse an umbrella directory declaration.
///
///   umbrella-dir-declaration:
///     umbrella string-literal
void ModuleMapParser::parseUmbrellaDirDecl(SourceLocation UmbrellaLoc) {
  // Parse the directory name.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_header) 
      << "umbrella";
    HadError = true;
    return;
  }

  std::string DirName = Tok.getString();
  SourceLocation DirNameLoc = consumeToken();
  
  // Check whether we already have an umbrella.
  if (ActiveModule->Umbrella) {
    Diags.Report(DirNameLoc, diag::err_mmap_umbrella_clash)
      << ActiveModule->getFullModuleName();
    HadError = true;
    return;
  }

  // Look for this file.
  const DirectoryEntry *Dir = nullptr;
  if (llvm::sys::path::is_absolute(DirName))
    Dir = SourceMgr.getFileManager().getDirectory(DirName);
  else {
    SmallString<128> PathName;
    PathName = Directory->getName();
    llvm::sys::path::append(PathName, DirName);
    Dir = SourceMgr.getFileManager().getDirectory(PathName);
  }
  
  if (!Dir) {
    Diags.Report(DirNameLoc, diag::err_mmap_umbrella_dir_not_found)
      << DirName;
    HadError = true;
    return;
  }

  if (UsesRequiresExcludedHack.count(ActiveModule)) {
    // Mark this header 'textual' (see doc comment for
    // ModuleMapParser::UsesRequiresExcludedHack). Although iterating over the
    // directory is relatively expensive, in practice this only applies to the
    // uncommonly used Tcl module on Darwin platforms.
    std::error_code EC;
    SmallVector<Module::Header, 6> Headers;
    vfs::FileSystem &FS = *SourceMgr.getFileManager().getVirtualFileSystem();
    for (vfs::recursive_directory_iterator I(FS, Dir->getName(), EC), E;
         I != E && !EC; I.increment(EC)) {
      if (const FileEntry *FE =
              SourceMgr.getFileManager().getFile(I->getName())) {

        Module::Header Header = {I->getName(), FE};
        Headers.push_back(std::move(Header));
      }
    }

    // Sort header paths so that the pcm doesn't depend on iteration order.
    llvm::array_pod_sort(Headers.begin(), Headers.end(), compareModuleHeaders);

    for (auto &Header : Headers)
      Map.addHeader(ActiveModule, std::move(Header), ModuleMap::TextualHeader);
    return;
  }

  if (Module *OwningModule = Map.UmbrellaDirs[Dir]) {
    Diags.Report(UmbrellaLoc, diag::err_mmap_umbrella_clash)
      << OwningModule->getFullModuleName();
    HadError = true;
    return;
  }

  // Record this umbrella directory.
  Map.setUmbrellaDir(ActiveModule, Dir, DirName);
}

/// \brief Parse a module export declaration.
///
///   export-declaration:
///     'export' wildcard-module-id
///
///   wildcard-module-id:
///     identifier
///     '*'
///     identifier '.' wildcard-module-id
void ModuleMapParser::parseExportDecl() {
  assert(Tok.is(MMToken::ExportKeyword));
  SourceLocation ExportLoc = consumeToken();
  
  // Parse the module-id with an optional wildcard at the end.
  ModuleId ParsedModuleId;
  bool Wildcard = false;
  do {
    // FIXME: Support string-literal module names here.
    if (Tok.is(MMToken::Identifier)) {
      ParsedModuleId.push_back(std::make_pair(Tok.getString(), 
                                              Tok.getLocation()));
      consumeToken();
      
      if (Tok.is(MMToken::Period)) {
        consumeToken();
        continue;
      } 
      
      break;
    }
    
    if(Tok.is(MMToken::Star)) {
      Wildcard = true;
      consumeToken();
      break;
    }
    
    Diags.Report(Tok.getLocation(), diag::err_mmap_module_id);
    HadError = true;
    return;
  } while (true);
  
  Module::UnresolvedExportDecl Unresolved = { 
    ExportLoc, ParsedModuleId, Wildcard 
  };
  ActiveModule->UnresolvedExports.push_back(Unresolved);
}

/// \brief Parse a module use declaration.
///
///   use-declaration:
///     'use' wildcard-module-id
void ModuleMapParser::parseUseDecl() {
  assert(Tok.is(MMToken::UseKeyword));
  auto KWLoc = consumeToken();
  // Parse the module-id.
  ModuleId ParsedModuleId;
  parseModuleId(ParsedModuleId);

  if (ActiveModule->Parent)
    Diags.Report(KWLoc, diag::err_mmap_use_decl_submodule);
  else
    ActiveModule->UnresolvedDirectUses.push_back(ParsedModuleId);
}

/// \brief Parse a link declaration.
///
///   module-declaration:
///     'link' 'framework'[opt] string-literal
void ModuleMapParser::parseLinkDecl() {
  assert(Tok.is(MMToken::LinkKeyword));
  SourceLocation LinkLoc = consumeToken();

  // Parse the optional 'framework' keyword.
  bool IsFramework = false;
  if (Tok.is(MMToken::FrameworkKeyword)) {
    consumeToken();
    IsFramework = true;
  }

  // Parse the library name
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_library_name)
      << IsFramework << SourceRange(LinkLoc);
    HadError = true;
    return;
  }

  std::string LibraryName = Tok.getString();
  consumeToken();
  ActiveModule->LinkLibraries.push_back(Module::LinkLibrary(LibraryName,
                                                            IsFramework));
}

/// \brief Parse a configuration macro declaration.
///
///   module-declaration:
///     'config_macros' attributes[opt] config-macro-list?
///
///   config-macro-list:
///     identifier (',' identifier)?
void ModuleMapParser::parseConfigMacros() {
  assert(Tok.is(MMToken::ConfigMacros));
  SourceLocation ConfigMacrosLoc = consumeToken();

  // Only top-level modules can have configuration macros.
  if (ActiveModule->Parent) {
    Diags.Report(ConfigMacrosLoc, diag::err_mmap_config_macro_submodule);
  }

  // Parse the optional attributes.
  Attributes Attrs;
  if (parseOptionalAttributes(Attrs))
    return;

  if (Attrs.IsExhaustive && !ActiveModule->Parent) {
    ActiveModule->ConfigMacrosExhaustive = true;
  }

  // If we don't have an identifier, we're done.
  // FIXME: Support macros with the same name as a keyword here.
  if (!Tok.is(MMToken::Identifier))
    return;

  // Consume the first identifier.
  if (!ActiveModule->Parent) {
    ActiveModule->ConfigMacros.push_back(Tok.getString().str());
  }
  consumeToken();

  do {
    // If there's a comma, consume it.
    if (!Tok.is(MMToken::Comma))
      break;
    consumeToken();

    // We expect to see a macro name here.
    // FIXME: Support macros with the same name as a keyword here.
    if (!Tok.is(MMToken::Identifier)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_config_macro);
      break;
    }

    // Consume the macro name.
    if (!ActiveModule->Parent) {
      ActiveModule->ConfigMacros.push_back(Tok.getString().str());
    }
    consumeToken();
  } while (true);
}

/// \brief Format a module-id into a string.
static std::string formatModuleId(const ModuleId &Id) {
  std::string result;
  {
    llvm::raw_string_ostream OS(result);

    for (unsigned I = 0, N = Id.size(); I != N; ++I) {
      if (I)
        OS << ".";
      OS << Id[I].first;
    }
  }

  return result;
}

/// \brief Parse a conflict declaration.
///
///   module-declaration:
///     'conflict' module-id ',' string-literal
void ModuleMapParser::parseConflict() {
  assert(Tok.is(MMToken::Conflict));
  SourceLocation ConflictLoc = consumeToken();
  Module::UnresolvedConflict Conflict;

  // Parse the module-id.
  if (parseModuleId(Conflict.Id))
    return;

  // Parse the ','.
  if (!Tok.is(MMToken::Comma)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_conflicts_comma)
      << SourceRange(ConflictLoc);
    return;
  }
  consumeToken();

  // Parse the message.
  if (!Tok.is(MMToken::StringLiteral)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_conflicts_message)
      << formatModuleId(Conflict.Id);
    return;
  }
  Conflict.Message = Tok.getString().str();
  consumeToken();

  // Add this unresolved conflict.
  ActiveModule->UnresolvedConflicts.push_back(Conflict);
}

/// \brief Parse an inferred module declaration (wildcard modules).
///
///   module-declaration:
///     'explicit'[opt] 'framework'[opt] 'module' * attributes[opt]
///       { inferred-module-member* }
///
///   inferred-module-member:
///     'export' '*'
///     'exclude' identifier
void ModuleMapParser::parseInferredModuleDecl(bool Framework, bool Explicit) {
  assert(Tok.is(MMToken::Star));
  SourceLocation StarLoc = consumeToken();
  bool Failed = false;

  // Inferred modules must be submodules.
  if (!ActiveModule && !Framework) {
    Diags.Report(StarLoc, diag::err_mmap_top_level_inferred_submodule);
    Failed = true;
  }

  if (ActiveModule) {
    // Inferred modules must have umbrella directories.
    if (!Failed && ActiveModule->IsAvailable &&
        !ActiveModule->getUmbrellaDir()) {
      Diags.Report(StarLoc, diag::err_mmap_inferred_no_umbrella);
      Failed = true;
    }
    
    // Check for redefinition of an inferred module.
    if (!Failed && ActiveModule->InferSubmodules) {
      Diags.Report(StarLoc, diag::err_mmap_inferred_redef);
      if (ActiveModule->InferredSubmoduleLoc.isValid())
        Diags.Report(ActiveModule->InferredSubmoduleLoc,
                     diag::note_mmap_prev_definition);
      Failed = true;
    }

    // Check for the 'framework' keyword, which is not permitted here.
    if (Framework) {
      Diags.Report(StarLoc, diag::err_mmap_inferred_framework_submodule);
      Framework = false;
    }
  } else if (Explicit) {
    Diags.Report(StarLoc, diag::err_mmap_explicit_inferred_framework);
    Explicit = false;
  }

  // If there were any problems with this inferred submodule, skip its body.
  if (Failed) {
    if (Tok.is(MMToken::LBrace)) {
      consumeToken();
      skipUntil(MMToken::RBrace);
      if (Tok.is(MMToken::RBrace))
        consumeToken();
    }
    HadError = true;
    return;
  }

  // Parse optional attributes.
  Attributes Attrs;
  if (parseOptionalAttributes(Attrs))
    return;

  if (ActiveModule) {
    // Note that we have an inferred submodule.
    ActiveModule->InferSubmodules = true;
    ActiveModule->InferredSubmoduleLoc = StarLoc;
    ActiveModule->InferExplicitSubmodules = Explicit;
  } else {
    // We'll be inferring framework modules for this directory.
    Map.InferredDirectories[Directory].InferModules = true;
    Map.InferredDirectories[Directory].Attrs = Attrs;
    Map.InferredDirectories[Directory].ModuleMapFile = ModuleMapFile;
    // FIXME: Handle the 'framework' keyword.
  }

  // Parse the opening brace.
  if (!Tok.is(MMToken::LBrace)) {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_lbrace_wildcard);
    HadError = true;
    return;
  }  
  SourceLocation LBraceLoc = consumeToken();

  // Parse the body of the inferred submodule.
  bool Done = false;
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
    case MMToken::RBrace:
      Done = true;
      break;

    case MMToken::ExcludeKeyword: {
      if (ActiveModule) {
        Diags.Report(Tok.getLocation(), diag::err_mmap_expected_inferred_member)
          << (ActiveModule != nullptr);
        consumeToken();
        break;
      }

      consumeToken();
      // FIXME: Support string-literal module names here.
      if (!Tok.is(MMToken::Identifier)) {
        Diags.Report(Tok.getLocation(), diag::err_mmap_missing_exclude_name);
        break;
      }

      Map.InferredDirectories[Directory].ExcludedModules
        .push_back(Tok.getString());
      consumeToken();
      break;
    }

    case MMToken::ExportKeyword:
      if (!ActiveModule) {
        Diags.Report(Tok.getLocation(), diag::err_mmap_expected_inferred_member)
          << (ActiveModule != nullptr);
        consumeToken();
        break;
      }

      consumeToken();
      if (Tok.is(MMToken::Star)) 
        ActiveModule->InferExportWildcard = true;
      else
        Diags.Report(Tok.getLocation(), 
                     diag::err_mmap_expected_export_wildcard);
      consumeToken();
      break;

    case MMToken::ExplicitKeyword:
    case MMToken::ModuleKeyword:
    case MMToken::HeaderKeyword:
    case MMToken::PrivateKeyword:
    case MMToken::UmbrellaKeyword:
    default:
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_inferred_member)
          << (ActiveModule != nullptr);
      consumeToken();
      break;        
    }
  } while (!Done);
  
  if (Tok.is(MMToken::RBrace))
    consumeToken();
  else {
    Diags.Report(Tok.getLocation(), diag::err_mmap_expected_rbrace);
    Diags.Report(LBraceLoc, diag::note_mmap_lbrace_match);
    HadError = true;
  }
}

/// \brief Parse optional attributes.
///
///   attributes:
///     attribute attributes
///     attribute
///
///   attribute:
///     [ identifier ]
///
/// \param Attrs Will be filled in with the parsed attributes.
///
/// \returns true if an error occurred, false otherwise.
bool ModuleMapParser::parseOptionalAttributes(Attributes &Attrs) {
  bool HadError = false;
  
  while (Tok.is(MMToken::LSquare)) {
    // Consume the '['.
    SourceLocation LSquareLoc = consumeToken();

    // Check whether we have an attribute name here.
    if (!Tok.is(MMToken::Identifier)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_attribute);
      skipUntil(MMToken::RSquare);
      if (Tok.is(MMToken::RSquare))
        consumeToken();
      HadError = true;
    }

    // Decode the attribute name.
    AttributeKind Attribute
      = llvm::StringSwitch<AttributeKind>(Tok.getString())
          .Case("exhaustive", AT_exhaustive)
          .Case("extern_c", AT_extern_c)
          .Case("system", AT_system)
          .Default(AT_unknown);
    switch (Attribute) {
    case AT_unknown:
      Diags.Report(Tok.getLocation(), diag::warn_mmap_unknown_attribute)
        << Tok.getString();
      break;

    case AT_system:
      Attrs.IsSystem = true;
      break;

    case AT_extern_c:
      Attrs.IsExternC = true;
      break;

    case AT_exhaustive:
      Attrs.IsExhaustive = true;
      break;
    }
    consumeToken();

    // Consume the ']'.
    if (!Tok.is(MMToken::RSquare)) {
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_rsquare);
      Diags.Report(LSquareLoc, diag::note_mmap_lsquare_match);
      skipUntil(MMToken::RSquare);
      HadError = true;
    }

    if (Tok.is(MMToken::RSquare))
      consumeToken();
  }

  return HadError;
}

/// \brief Parse a module map file.
///
///   module-map-file:
///     module-declaration*
bool ModuleMapParser::parseModuleMapFile() {
  do {
    switch (Tok.Kind) {
    case MMToken::EndOfFile:
      return HadError;
      
    case MMToken::ExplicitKeyword:
    case MMToken::ExternKeyword:
    case MMToken::ModuleKeyword:
    case MMToken::FrameworkKeyword:
      parseModuleDecl();
      break;

    case MMToken::Comma:
    case MMToken::ConfigMacros:
    case MMToken::Conflict:
    case MMToken::Exclaim:
    case MMToken::ExcludeKeyword:
    case MMToken::ExportKeyword:
    case MMToken::HeaderKeyword:
    case MMToken::Identifier:
    case MMToken::LBrace:
    case MMToken::LinkKeyword:
    case MMToken::LSquare:
    case MMToken::Period:
    case MMToken::PrivateKeyword:
    case MMToken::RBrace:
    case MMToken::RSquare:
    case MMToken::RequiresKeyword:
    case MMToken::Star:
    case MMToken::StringLiteral:
    case MMToken::TextualKeyword:
    case MMToken::UmbrellaKeyword:
    case MMToken::UseKeyword:
      Diags.Report(Tok.getLocation(), diag::err_mmap_expected_module);
      HadError = true;
      consumeToken();
      break;
    }
  } while (true);
}

bool ModuleMap::parseModuleMapFile(const FileEntry *File, bool IsSystem,
                                   const DirectoryEntry *Dir,
                                   SourceLocation ExternModuleLoc) {
  llvm::DenseMap<const FileEntry *, bool>::iterator Known
    = ParsedModuleMap.find(File);
  if (Known != ParsedModuleMap.end())
    return Known->second;

  assert(Target && "Missing target information");
  auto FileCharacter = IsSystem ? SrcMgr::C_System : SrcMgr::C_User;
  FileID ID = SourceMgr.createFileID(File, ExternModuleLoc, FileCharacter);
  const llvm::MemoryBuffer *Buffer = SourceMgr.getBuffer(ID);
  if (!Buffer)
    return ParsedModuleMap[File] = true;

  // Parse this module map file.
  Lexer L(ID, SourceMgr.getBuffer(ID), SourceMgr, MMapLangOpts);
  SourceLocation Start = L.getSourceLocation();
  ModuleMapParser Parser(L, SourceMgr, Target, Diags, *this, File, Dir,
                         BuiltinIncludeDir, IsSystem);
  bool Result = Parser.parseModuleMapFile();
  ParsedModuleMap[File] = Result;

  // Notify callbacks that we parsed it.
  for (const auto &Cb : Callbacks)
    Cb->moduleMapFileRead(Start, *File, IsSystem);
  return Result;
}
