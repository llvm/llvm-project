//===- ModuleMap.cpp - Describe the layout of modules ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/ModuleMapFile.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace clang;

void ModuleMapCallbacks::anchor() {}

void ModuleMap::resolveLinkAsDependencies(Module *Mod) {
  auto PendingLinkAs = PendingLinkAsModule.find(Mod->Name);
  if (PendingLinkAs != PendingLinkAsModule.end()) {
    for (auto &Name : PendingLinkAs->second) {
      auto *M = findModule(Name.getKey());
      if (M)
        M->UseExportAsModuleLinkName = true;
    }
  }
}

void ModuleMap::addLinkAsDependency(Module *Mod) {
  if (findModule(Mod->ExportAsModule))
    Mod->UseExportAsModuleLinkName = true;
  else
    PendingLinkAsModule[Mod->ExportAsModule].insert(Mod->Name);
}

Module::HeaderKind ModuleMap::headerRoleToKind(ModuleHeaderRole Role) {
  switch ((int)Role) {
  case NormalHeader:
    return Module::HK_Normal;
  case PrivateHeader:
    return Module::HK_Private;
  case TextualHeader:
    return Module::HK_Textual;
  case PrivateHeader | TextualHeader:
    return Module::HK_PrivateTextual;
  case ExcludedHeader:
    return Module::HK_Excluded;
  }
  llvm_unreachable("unknown header role");
}

ModuleMap::ModuleHeaderRole
ModuleMap::headerKindToRole(Module::HeaderKind Kind) {
  switch ((int)Kind) {
  case Module::HK_Normal:
    return NormalHeader;
  case Module::HK_Private:
    return PrivateHeader;
  case Module::HK_Textual:
    return TextualHeader;
  case Module::HK_PrivateTextual:
    return ModuleHeaderRole(PrivateHeader | TextualHeader);
  case Module::HK_Excluded:
    return ExcludedHeader;
  }
  llvm_unreachable("unknown header kind");
}

bool ModuleMap::isModular(ModuleHeaderRole Role) {
  return !(Role & (ModuleMap::TextualHeader | ModuleMap::ExcludedHeader));
}

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
    return {};

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

/// Append to \p Paths the set of paths needed to get to the
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
  for (StringRef Framework : llvm::drop_begin(llvm::reverse(Paths)))
    llvm::sys::path::append(Path, "Frameworks", Framework + ".framework");
}

OptionalFileEntryRef ModuleMap::findHeader(
    Module *M, const Module::UnresolvedHeaderDirective &Header,
    SmallVectorImpl<char> &RelativePathName, bool &NeedsFramework) {
  // Search for the header file within the module's home directory.
  auto Directory = M->Directory;
  SmallString<128> FullPathName(Directory->getName());

  auto GetFile = [&](StringRef Filename) -> OptionalFileEntryRef {
    auto File =
        expectedToOptional(SourceMgr.getFileManager().getFileRef(Filename));
    if (!File || (Header.Size && File->getSize() != *Header.Size) ||
        (Header.ModTime && File->getModificationTime() != *Header.ModTime))
      return std::nullopt;
    return *File;
  };

  auto GetFrameworkFile = [&]() -> OptionalFileEntryRef {
    unsigned FullPathLength = FullPathName.size();
    appendSubframeworkPaths(M, RelativePathName);
    unsigned RelativePathLength = RelativePathName.size();

    // Check whether this file is in the public headers.
    llvm::sys::path::append(RelativePathName, "Headers", Header.FileName);
    llvm::sys::path::append(FullPathName, RelativePathName);
    if (auto File = GetFile(FullPathName))
      return File;

    // Check whether this file is in the private headers.
    // Ideally, private modules in the form 'FrameworkName.Private' should
    // be defined as 'module FrameworkName.Private', and not as
    // 'framework module FrameworkName.Private', since a 'Private.Framework'
    // does not usually exist. However, since both are currently widely used
    // for private modules, make sure we find the right path in both cases.
    if (M->IsFramework && M->Name == "Private")
      RelativePathName.clear();
    else
      RelativePathName.resize(RelativePathLength);
    FullPathName.resize(FullPathLength);
    llvm::sys::path::append(RelativePathName, "PrivateHeaders",
                            Header.FileName);
    llvm::sys::path::append(FullPathName, RelativePathName);
    return GetFile(FullPathName);
  };

  if (llvm::sys::path::is_absolute(Header.FileName)) {
    RelativePathName.clear();
    RelativePathName.append(Header.FileName.begin(), Header.FileName.end());
    return GetFile(Header.FileName);
  }

  if (M->isPartOfFramework())
    return GetFrameworkFile();

  // Lookup for normal headers.
  llvm::sys::path::append(RelativePathName, Header.FileName);
  llvm::sys::path::append(FullPathName, RelativePathName);
  auto NormalHdrFile = GetFile(FullPathName);

  if (!NormalHdrFile && Directory->getName().ends_with(".framework")) {
    // The lack of 'framework' keyword in a module declaration it's a simple
    // mistake we can diagnose when the header exists within the proper
    // framework style path.
    FullPathName.assign(Directory->getName());
    RelativePathName.clear();
    if (GetFrameworkFile()) {
      Diags.Report(Header.FileNameLoc,
                   diag::warn_mmap_incomplete_framework_module_declaration)
          << Header.FileName << M->getFullModuleName();
      NeedsFramework = true;
    }
    return std::nullopt;
  }

  return NormalHdrFile;
}

/// Determine whether the given file name is the name of a builtin
/// header, supplied by Clang to replace, override, or augment existing system
/// headers.
static bool isBuiltinHeaderName(StringRef FileName) {
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

/// Determine whether the given module name is the name of a builtin
/// module that is cyclic with a system module  on some platforms.
static bool isBuiltInModuleName(StringRef ModuleName) {
  return llvm::StringSwitch<bool>(ModuleName)
           .Case("_Builtin_float", true)
           .Case("_Builtin_inttypes", true)
           .Case("_Builtin_iso646", true)
           .Case("_Builtin_limits", true)
           .Case("_Builtin_stdalign", true)
           .Case("_Builtin_stdarg", true)
           .Case("_Builtin_stdatomic", true)
           .Case("_Builtin_stdbool", true)
           .Case("_Builtin_stddef", true)
           .Case("_Builtin_stdint", true)
           .Case("_Builtin_stdnoreturn", true)
           .Case("_Builtin_tgmath", true)
           .Case("_Builtin_unwind", true)
           .Default(false);
}

void ModuleMap::resolveHeader(Module *Mod,
                              const Module::UnresolvedHeaderDirective &Header,
                              bool &NeedsFramework) {
  SmallString<128> RelativePathName;
  if (OptionalFileEntryRef File =
          findHeader(Mod, Header, RelativePathName, NeedsFramework)) {
    if (Header.IsUmbrella) {
      const DirectoryEntry *UmbrellaDir = &File->getDir().getDirEntry();
      if (Module *UmbrellaMod = UmbrellaDirs[UmbrellaDir])
        Diags.Report(Header.FileNameLoc, diag::err_mmap_umbrella_clash)
          << UmbrellaMod->getFullModuleName();
      else
        // Record this umbrella header.
        setUmbrellaHeaderAsWritten(Mod, *File, Header.FileName,
                                   RelativePathName.str());
    } else {
      Module::Header H = {Header.FileName, std::string(RelativePathName),
                          *File};
      addHeader(Mod, H, headerKindToRole(Header.Kind));
    }
  } else if (Header.HasBuiltinHeader && !Header.Size && !Header.ModTime) {
    // There's a builtin header but no corresponding on-disk header. Assume
    // this was supposed to modularize the builtin header alone.
  } else if (Header.Kind == Module::HK_Excluded) {
    // Ignore missing excluded header files. They're optional anyway.
  } else {
    // If we find a module that has a missing header, we mark this module as
    // unavailable and store the header directive for displaying diagnostics.
    Mod->MissingHeaders.push_back(Header);
    // A missing header with stat information doesn't make the module
    // unavailable; this keeps our behavior consistent as headers are lazily
    // resolved. (Such a module still can't be built though, except from
    // preprocessed source.)
    if (!Header.Size && !Header.ModTime)
      Mod->markUnavailable(/*Unimportable=*/false);
  }
}

bool ModuleMap::resolveAsBuiltinHeader(
    Module *Mod, const Module::UnresolvedHeaderDirective &Header) {
  if (Header.Kind == Module::HK_Excluded ||
      llvm::sys::path::is_absolute(Header.FileName) ||
      Mod->isPartOfFramework() || !Mod->IsSystem || Header.IsUmbrella ||
      !BuiltinIncludeDir || BuiltinIncludeDir == Mod->Directory ||
      !LangOpts.BuiltinHeadersInSystemModules || !isBuiltinHeaderName(Header.FileName))
    return false;

  // This is a system module with a top-level header. This header
  // may have a counterpart (or replacement) in the set of headers
  // supplied by Clang. Find that builtin header.
  SmallString<128> Path;
  llvm::sys::path::append(Path, BuiltinIncludeDir->getName(), Header.FileName);
  auto File = SourceMgr.getFileManager().getOptionalFileRef(Path);
  if (!File)
    return false;

  Module::Header H = {Header.FileName, Header.FileName, *File};
  auto Role = headerKindToRole(Header.Kind);
  addHeader(Mod, H, Role);
  return true;
}

ModuleMap::ModuleMap(SourceManager &SourceMgr, DiagnosticsEngine &Diags,
                     const LangOptions &LangOpts, const TargetInfo *Target,
                     HeaderSearch &HeaderInfo)
    : SourceMgr(SourceMgr), Diags(Diags), LangOpts(LangOpts), Target(Target),
      HeaderInfo(HeaderInfo) {
}

ModuleMap::~ModuleMap() = default;

void ModuleMap::setTarget(const TargetInfo &Target) {
  assert((!this->Target || this->Target == &Target) &&
         "Improper target override");
  this->Target = &Target;
}

/// "Sanitize" a filename so that it can be used as an identifier.
static StringRef sanitizeFilenameAsIdentifier(StringRef Name,
                                              SmallVectorImpl<char> &Buffer) {
  if (Name.empty())
    return Name;

  if (!isValidAsciiIdentifier(Name)) {
    // If we don't already have something with the form of an identifier,
    // create a buffer with the sanitized name.
    Buffer.clear();
    if (isDigit(Name[0]))
      Buffer.push_back('_');
    Buffer.reserve(Buffer.size() + Name.size());
    for (unsigned I = 0, N = Name.size(); I != N; ++I) {
      if (isAsciiIdentifierContinue(Name[I]))
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

bool ModuleMap::isBuiltinHeader(FileEntryRef File) {
  return File.getDir() == BuiltinIncludeDir && LangOpts.BuiltinHeadersInSystemModules &&
         isBuiltinHeaderName(llvm::sys::path::filename(File.getName()));
}

bool ModuleMap::shouldImportRelativeToBuiltinIncludeDir(StringRef FileName,
                                                        Module *Module) const {
  return LangOpts.BuiltinHeadersInSystemModules && BuiltinIncludeDir &&
         Module->IsSystem && !Module->isPartOfFramework() &&
         isBuiltinHeaderName(FileName);
}

ModuleMap::HeadersMap::iterator ModuleMap::findKnownHeader(FileEntryRef File) {
  resolveHeaderDirectives(File);
  HeadersMap::iterator Known = Headers.find(File);
  if (HeaderInfo.getHeaderSearchOpts().ImplicitModuleMaps &&
      Known == Headers.end() && ModuleMap::isBuiltinHeader(File)) {
    HeaderInfo.loadTopLevelSystemModules();
    return Headers.find(File);
  }
  return Known;
}

ModuleMap::KnownHeader ModuleMap::findHeaderInUmbrellaDirs(
    FileEntryRef File, SmallVectorImpl<DirectoryEntryRef> &IntermediateDirs) {
  if (UmbrellaDirs.empty())
    return {};

  OptionalDirectoryEntryRef Dir = File.getDir();

  // Note: as an egregious but useful hack we use the real path here, because
  // frameworks moving from top-level frameworks to embedded frameworks tend
  // to be symlinked from the top-level location to the embedded location,
  // and we need to resolve lookups as if we had found the embedded location.
  StringRef DirName = SourceMgr.getFileManager().getCanonicalName(*Dir);

  // Keep walking up the directory hierarchy, looking for a directory with
  // an umbrella header.
  do {
    auto KnownDir = UmbrellaDirs.find(*Dir);
    if (KnownDir != UmbrellaDirs.end())
      return KnownHeader(KnownDir->second, NormalHeader);

    IntermediateDirs.push_back(*Dir);

    // Retrieve our parent path.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      break;

    // Resolve the parent path to a directory entry.
    Dir = SourceMgr.getFileManager().getOptionalDirectoryRef(DirName);
  } while (Dir);
  return {};
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
    ArrayRef<Module::Header> HeaderList[] = {
        Header.getModule()->getHeaders(Module::HK_Private),
        Header.getModule()->getHeaders(Module::HK_PrivateTextual)};
    for (auto Hs : HeaderList)
      IsPrivate |= llvm::any_of(
          Hs, [&](const Module::Header &H) { return H.Entry == IncFileEnt; });
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
                                        StringRef Filename, FileEntryRef File) {
  // No errors for indirect modules. This may be a bit of a problem for modules
  // with no source files.
  if (getTopLevelOrNull(RequestingModule) != getTopLevelOrNull(SourceModule))
    return;

  if (RequestingModule) {
    resolveUses(RequestingModule, /*Complain=*/false);
    resolveHeaderDirectives(RequestingModule, /*File=*/std::nullopt);
  }

  bool Excluded = false;
  Module *Private = nullptr;
  Module *NotUsed = nullptr;

  HeadersMap::iterator Known = findKnownHeader(File);
  if (Known != Headers.end()) {
    for (const KnownHeader &Header : Known->second) {
      // Excluded headers don't really belong to a module.
      if (Header.getRole() == ModuleMap::ExcludedHeader) {
        Excluded = true;
        continue;
      }

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
    Diags.Report(FilenameLoc, diag::err_undeclared_use_of_module_indirect)
        << RequestingModule->getTopLevelModule()->Name << Filename
        << NotUsed->Name;
    return;
  }

  if (Excluded || isHeaderInUmbrellaDirs(File))
    return;

  // At this point, only non-modular includes remain.

  if (RequestingModule && LangOpts.ModulesStrictDeclUse) {
    Diags.Report(FilenameLoc, diag::err_undeclared_use_of_module)
        << RequestingModule->getTopLevelModule()->Name << Filename;
  } else if (RequestingModule && RequestingModuleIsModuleInterface &&
             LangOpts.isCompilingModule()) {
    // Do not diagnose when we are not compiling a module.
    diag::kind DiagID = RequestingModule->getTopLevelModule()->IsFramework ?
        diag::warn_non_modular_include_in_framework_module :
        diag::warn_non_modular_include_in_module;
    Diags.Report(FilenameLoc, DiagID) << RequestingModule->getFullModuleName()
        << File.getName();
  }
}

static bool isBetterKnownHeader(const ModuleMap::KnownHeader &New,
                                const ModuleMap::KnownHeader &Old) {
  // Prefer available modules.
  // FIXME: Considering whether the module is available rather than merely
  // importable is non-hermetic and can result in surprising behavior for
  // prebuilt modules. Consider only checking for importability here.
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

  // Prefer a non-excluded header over an excluded header.
  if ((New.getRole() == ModuleMap::ExcludedHeader) !=
      (Old.getRole() == ModuleMap::ExcludedHeader))
    return New.getRole() != ModuleMap::ExcludedHeader;

  // Don't have a reason to choose between these. Just keep the first one.
  return false;
}

ModuleMap::KnownHeader ModuleMap::findModuleForHeader(FileEntryRef File,
                                                      bool AllowTextual,
                                                      bool AllowExcluded) {
  auto MakeResult = [&](ModuleMap::KnownHeader R) -> ModuleMap::KnownHeader {
    if (!AllowTextual && R.getRole() & ModuleMap::TextualHeader)
      return {};
    return R;
  };

  HeadersMap::iterator Known = findKnownHeader(File);
  if (Known != Headers.end()) {
    ModuleMap::KnownHeader Result;
    // Iterate over all modules that 'File' is part of to find the best fit.
    for (KnownHeader &H : Known->second) {
      // Cannot use a module if the header is excluded in it.
      if (!AllowExcluded && H.getRole() == ModuleMap::ExcludedHeader)
        continue;
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
ModuleMap::findOrCreateModuleForHeaderInUmbrellaDir(FileEntryRef File) {
  assert(!Headers.count(File) && "already have a module for this header");

  SmallVector<DirectoryEntryRef, 2> SkippedDirs;
  KnownHeader H = findHeaderInUmbrellaDirs(File, SkippedDirs);
  if (H) {
    Module *Result = H.getModule();

    // Search up the module stack until we find a module with an umbrella
    // directory.
    Module *UmbrellaModule = Result;
    while (!UmbrellaModule->getEffectiveUmbrellaDir() && UmbrellaModule->Parent)
      UmbrellaModule = UmbrellaModule->Parent;

    if (UmbrellaModule->InferSubmodules) {
      FileID UmbrellaModuleMap = getModuleMapFileIDForUniquing(UmbrellaModule);

      // Infer submodules for each of the directories we found between
      // the directory of the umbrella header and the directory where
      // the actual header is located.
      bool Explicit = UmbrellaModule->InferExplicitSubmodules;

      for (DirectoryEntryRef SkippedDir : llvm::reverse(SkippedDirs)) {
        // Find or create the module that corresponds to this directory name.
        SmallString<32> NameBuf;
        StringRef Name = sanitizeFilenameAsIdentifier(
            llvm::sys::path::stem(SkippedDir.getName()), NameBuf);
        Result = findOrCreateModuleFirst(Name, Result, /*IsFramework=*/false,
                                         Explicit);
        setInferredModuleAllowedBy(Result, UmbrellaModuleMap);

        // Associate the module and the directory.
        UmbrellaDirs[SkippedDir] = Result;

        // If inferred submodules export everything they import, add a
        // wildcard to the set of exports.
        if (UmbrellaModule->InferExportWildcard && Result->Exports.empty())
          Result->Exports.push_back(Module::ExportDecl(nullptr, true));
      }

      // Infer a submodule with the same name as this header file.
      SmallString<32> NameBuf;
      StringRef Name = sanitizeFilenameAsIdentifier(
                         llvm::sys::path::stem(File.getName()), NameBuf);
      Result = findOrCreateModuleFirst(Name, Result, /*IsFramework=*/false,
                                       Explicit);
      setInferredModuleAllowedBy(Result, UmbrellaModuleMap);
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

  return {};
}

ArrayRef<ModuleMap::KnownHeader>
ModuleMap::findAllModulesForHeader(FileEntryRef File) {
  HeadersMap::iterator Known = findKnownHeader(File);
  if (Known != Headers.end())
    return Known->second;

  if (findOrCreateModuleForHeaderInUmbrellaDir(File))
    return Headers.find(File)->second;

  return {};
}

ArrayRef<ModuleMap::KnownHeader>
ModuleMap::findResolvedModulesForHeader(FileEntryRef File) const {
  // FIXME: Is this necessary?
  resolveHeaderDirectives(File);
  auto It = Headers.find(File);
  if (It == Headers.end())
    return {};
  return It->second;
}

bool ModuleMap::isHeaderInUnavailableModule(FileEntryRef Header) const {
  return isHeaderUnavailableInModule(Header, nullptr);
}

bool ModuleMap::isHeaderUnavailableInModule(
    FileEntryRef Header, const Module *RequestingModule) const {
  resolveHeaderDirectives(Header);
  HeadersMap::const_iterator Known = Headers.find(Header);
  if (Known != Headers.end()) {
    for (SmallVectorImpl<KnownHeader>::const_iterator
             I = Known->second.begin(),
             E = Known->second.end();
         I != E; ++I) {

      if (I->getRole() == ModuleMap::ExcludedHeader)
        continue;

      if (I->isAvailable() &&
          (!RequestingModule ||
           I->getModule()->isSubModuleOf(RequestingModule))) {
        // When no requesting module is available, the caller is looking if a
        // header is part a module by only looking into the module map. This is
        // done by warn_uncovered_module_header checks; don't consider textual
        // headers part of it in this mode, otherwise we get misleading warnings
        // that a umbrella header is not including a textual header.
        if (!RequestingModule && I->getRole() == ModuleMap::TextualHeader)
          continue;
        return false;
      }
    }
    return true;
  }

  OptionalDirectoryEntryRef Dir = Header.getDir();
  SmallVector<DirectoryEntryRef, 2> SkippedDirs;
  StringRef DirName = Dir->getName();

  auto IsUnavailable = [&](const Module *M) {
    return !M->isAvailable() && (!RequestingModule ||
                                 M->isSubModuleOf(RequestingModule));
  };

  // Keep walking up the directory hierarchy, looking for a directory with
  // an umbrella header.
  do {
    auto KnownDir = UmbrellaDirs.find(*Dir);
    if (KnownDir != UmbrellaDirs.end()) {
      Module *Found = KnownDir->second;
      if (IsUnavailable(Found))
        return true;

      // Search up the module stack until we find a module with an umbrella
      // directory.
      Module *UmbrellaModule = Found;
      while (!UmbrellaModule->getEffectiveUmbrellaDir() &&
             UmbrellaModule->Parent)
        UmbrellaModule = UmbrellaModule->Parent;

      if (UmbrellaModule->InferSubmodules) {
        for (DirectoryEntryRef SkippedDir : llvm::reverse(SkippedDirs)) {
          // Find or create the module that corresponds to this directory name.
          SmallString<32> NameBuf;
          StringRef Name = sanitizeFilenameAsIdentifier(
              llvm::sys::path::stem(SkippedDir.getName()), NameBuf);
          Found = lookupModuleQualified(Name, Found);
          if (!Found)
            return false;
          if (IsUnavailable(Found))
            return true;
        }

        // Infer a submodule with the same name as this header file.
        SmallString<32> NameBuf;
        StringRef Name = sanitizeFilenameAsIdentifier(
                           llvm::sys::path::stem(Header.getName()),
                           NameBuf);
        Found = lookupModuleQualified(Name, Found);
        if (!Found)
          return false;
      }

      return IsUnavailable(Found);
    }

    SkippedDirs.push_back(*Dir);

    // Retrieve our parent path.
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.empty())
      break;

    // Resolve the parent path to a directory entry.
    Dir = SourceMgr.getFileManager().getOptionalDirectoryRef(DirName);
  } while (Dir);

  return false;
}

Module *ModuleMap::findModule(StringRef Name) const {
  llvm::StringMap<Module *>::const_iterator Known = Modules.find(Name);
  if (Known != Modules.end())
    return Known->getValue();

  return nullptr;
}

Module *ModuleMap::findOrInferSubmodule(Module *Parent, StringRef Name) {
  if (Module *SubM = Parent->findSubmodule(Name))
    return SubM;
  if (!Parent->InferSubmodules)
    return nullptr;
  Module *Result = new (ModulesAlloc.Allocate())
      Module(ModuleConstructorTag{}, Name, SourceLocation(), Parent, false,
             Parent->InferExplicitSubmodules, 0);
  Result->InferExplicitSubmodules = Parent->InferExplicitSubmodules;
  Result->InferSubmodules = Parent->InferSubmodules;
  Result->InferExportWildcard = Parent->InferExportWildcard;
  if (Result->InferExportWildcard)
    Result->Exports.push_back(Module::ExportDecl(nullptr, true));
  return Result;
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

std::pair<Module *, bool> ModuleMap::findOrCreateModule(StringRef Name,
                                                        Module *Parent,
                                                        bool IsFramework,
                                                        bool IsExplicit) {
  // Try to find an existing module with this name.
  if (Module *Sub = lookupModuleQualified(Name, Parent))
    return std::make_pair(Sub, false);

  // Create a new module with this name.
  Module *M = createModule(Name, Parent, IsFramework, IsExplicit);
  return std::make_pair(M, true);
}

Module *ModuleMap::createModule(StringRef Name, Module *Parent,
                                bool IsFramework, bool IsExplicit) {
  assert(lookupModuleQualified(Name, Parent) == nullptr &&
         "Creating duplicate submodule");

  Module *Result = new (ModulesAlloc.Allocate())
      Module(ModuleConstructorTag{}, Name, SourceLocation(), Parent,
             IsFramework, IsExplicit, NumCreatedModules++);
  if (!Parent) {
    if (LangOpts.CurrentModule == Name)
      SourceModule = Result;
    Modules[Name] = Result;
    ModuleScopeIDs[Result] = CurrentModuleScopeID;
  }
  return Result;
}

Module *ModuleMap::createGlobalModuleFragmentForModuleUnit(SourceLocation Loc,
                                                           Module *Parent) {
  auto *Result = new (ModulesAlloc.Allocate()) Module(
      ModuleConstructorTag{}, "<global>", Loc, Parent, /*IsFramework=*/false,
      /*IsExplicit=*/true, NumCreatedModules++);
  Result->Kind = Module::ExplicitGlobalModuleFragment;
  // If the created module isn't owned by a parent, send it to PendingSubmodules
  // to wait for its parent.
  if (!Result->Parent)
    PendingSubmodules.emplace_back(Result);
  return Result;
}

Module *
ModuleMap::createImplicitGlobalModuleFragmentForModuleUnit(SourceLocation Loc,
                                                           Module *Parent) {
  assert(Parent && "We should only create an implicit global module fragment "
                   "in a module purview");
  // Note: Here the `IsExplicit` parameter refers to the semantics in clang
  // modules. All the non-explicit submodules in clang modules will be exported
  // too. Here we simplify the implementation by using the concept.
  auto *Result = new (ModulesAlloc.Allocate())
      Module(ModuleConstructorTag{}, "<implicit global>", Loc, Parent,
             /*IsFramework=*/false, /*IsExplicit=*/false, NumCreatedModules++);
  Result->Kind = Module::ImplicitGlobalModuleFragment;
  return Result;
}

Module *
ModuleMap::createPrivateModuleFragmentForInterfaceUnit(Module *Parent,
                                                       SourceLocation Loc) {
  auto *Result = new (ModulesAlloc.Allocate()) Module(
      ModuleConstructorTag{}, "<private>", Loc, Parent, /*IsFramework=*/false,
      /*IsExplicit=*/true, NumCreatedModules++);
  Result->Kind = Module::PrivateModuleFragment;
  return Result;
}

Module *ModuleMap::createModuleUnitWithKind(SourceLocation Loc, StringRef Name,
                                            Module::ModuleKind Kind) {
  auto *Result = new (ModulesAlloc.Allocate())
      Module(ModuleConstructorTag{}, Name, Loc, nullptr, /*IsFramework=*/false,
             /*IsExplicit=*/false, NumCreatedModules++);
  Result->Kind = Kind;

  // Reparent any current global module fragment as a submodule of this module.
  for (auto &Submodule : PendingSubmodules)
    Submodule->setParent(Result);
  PendingSubmodules.clear();
  return Result;
}

Module *ModuleMap::createModuleForInterfaceUnit(SourceLocation Loc,
                                                StringRef Name) {
  assert(LangOpts.CurrentModule == Name && "module name mismatch");
  assert(!Modules[Name] && "redefining existing module");

  auto *Result =
      createModuleUnitWithKind(Loc, Name, Module::ModuleInterfaceUnit);
  Modules[Name] = SourceModule = Result;

  // Mark the main source file as being within the newly-created module so that
  // declarations and macros are properly visibility-restricted to it.
  auto MainFile = SourceMgr.getFileEntryRefForID(SourceMgr.getMainFileID());
  assert(MainFile && "no input file for module interface");
  Headers[*MainFile].push_back(KnownHeader(Result, PrivateHeader));

  return Result;
}

Module *ModuleMap::createModuleForImplementationUnit(SourceLocation Loc,
                                                     StringRef Name) {
  assert(LangOpts.CurrentModule == Name && "module name mismatch");
  // The interface for this implementation must exist and be loaded.
  assert(Modules[Name] && Modules[Name]->Kind == Module::ModuleInterfaceUnit &&
         "creating implementation module without an interface");

  // Create an entry in the modules map to own the implementation unit module.
  // User module names must not start with a period (so that this cannot clash
  // with any legal user-defined module name).
  StringRef IName = ".ImplementationUnit";
  assert(!Modules[IName] && "multiple implementation units?");

  auto *Result =
      createModuleUnitWithKind(Loc, Name, Module::ModuleImplementationUnit);
  Modules[IName] = SourceModule = Result;

  // Check that the main file is present.
  assert(SourceMgr.getFileEntryForID(SourceMgr.getMainFileID()) &&
         "no input file for module implementation");

  return Result;
}

Module *ModuleMap::createHeaderUnit(SourceLocation Loc, StringRef Name,
                                    Module::Header H) {
  assert(LangOpts.CurrentModule == Name && "module name mismatch");
  assert(!Modules[Name] && "redefining existing module");

  auto *Result = new (ModulesAlloc.Allocate())
      Module(ModuleConstructorTag{}, Name, Loc, nullptr, /*IsFramework=*/false,
             /*IsExplicit=*/false, NumCreatedModules++);
  Result->Kind = Module::ModuleHeaderUnit;
  Modules[Name] = SourceModule = Result;
  addHeader(Result, H, NormalHeader);
  return Result;
}

/// For a framework module, infer the framework against which we
/// should link.
static void inferFrameworkLink(Module *Mod) {
  assert(Mod->IsFramework && "Can only infer linking for framework modules");
  assert(!Mod->isSubFramework() &&
         "Can only infer linking for top-level frameworks");

  StringRef FrameworkName(Mod->Name);
  FrameworkName.consume_back("_Private");
  Mod->LinkLibraries.push_back(Module::LinkLibrary(FrameworkName.str(),
                                                   /*IsFramework=*/true));
}

Module *ModuleMap::inferFrameworkModule(DirectoryEntryRef FrameworkDir,
                                        bool IsSystem, Module *Parent) {
  Attributes Attrs;
  Attrs.IsSystem = IsSystem;
  return inferFrameworkModule(FrameworkDir, Attrs, Parent);
}

Module *ModuleMap::inferFrameworkModule(DirectoryEntryRef FrameworkDir,
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
  FileID ModuleMapFID;
  if (!Parent) {
    // Determine whether we're allowed to infer a module map.
    bool canInfer = false;
    if (llvm::sys::path::has_parent_path(FrameworkDirName)) {
      // Figure out the parent path.
      StringRef Parent = llvm::sys::path::parent_path(FrameworkDirName);
      if (auto ParentDir = FileMgr.getOptionalDirectoryRef(Parent)) {
        // Check whether we have already looked into the parent directory
        // for a module map.
        llvm::DenseMap<const DirectoryEntry *, InferredDirectory>::const_iterator
          inferred = InferredDirectories.find(*ParentDir);
        if (inferred == InferredDirectories.end()) {
          // We haven't looked here before. Load a module map, if there is
          // one.
          bool IsFrameworkDir = Parent.ends_with(".framework");
          if (OptionalFileEntryRef ModMapFile =
                  HeaderInfo.lookupModuleMapFile(*ParentDir, IsFrameworkDir)) {
            // TODO: Parsing a module map should populate `InferredDirectories`
            //       so we don't need to do a full load here.
            parseAndLoadModuleMapFile(*ModMapFile, Attrs.IsSystem, *ParentDir);
            inferred = InferredDirectories.find(*ParentDir);
          }

          if (inferred == InferredDirectories.end())
            inferred = InferredDirectories.insert(
                         std::make_pair(*ParentDir, InferredDirectory())).first;
        }

        if (inferred->second.InferModules) {
          // We're allowed to infer for this directory, but make sure it's okay
          // to infer this particular module.
          StringRef Name = llvm::sys::path::stem(FrameworkDirName);
          canInfer =
              !llvm::is_contained(inferred->second.ExcludedModules, Name);

          Attrs.IsSystem |= inferred->second.Attrs.IsSystem;
          Attrs.IsExternC |= inferred->second.Attrs.IsExternC;
          Attrs.IsExhaustive |= inferred->second.Attrs.IsExhaustive;
          Attrs.NoUndeclaredIncludes |=
              inferred->second.Attrs.NoUndeclaredIncludes;
          ModuleMapFID = inferred->second.ModuleMapFID;
        }
      }
    }

    // If we're not allowed to infer a framework module, don't.
    if (!canInfer)
      return nullptr;
  } else {
    ModuleMapFID = getModuleMapFileIDForUniquing(Parent);
  }

  // Look for an umbrella header.
  SmallString<128> UmbrellaName = FrameworkDir.getName();
  llvm::sys::path::append(UmbrellaName, "Headers", ModuleName + ".h");
  auto UmbrellaHeader = FileMgr.getOptionalFileRef(UmbrellaName);

  // FIXME: If there's no umbrella header, we could probably scan the
  // framework to load *everything*. But, it's not clear that this is a good
  // idea.
  if (!UmbrellaHeader)
    return nullptr;

  Module *Result = new (ModulesAlloc.Allocate())
      Module(ModuleConstructorTag{}, ModuleName, SourceLocation(), Parent,
             /*IsFramework=*/true, /*IsExplicit=*/false, NumCreatedModules++);
  setInferredModuleAllowedBy(Result, ModuleMapFID);
  if (!Parent) {
    if (LangOpts.CurrentModule == ModuleName)
      SourceModule = Result;
    Modules[ModuleName] = Result;
    ModuleScopeIDs[Result] = CurrentModuleScopeID;
  }

  Result->IsSystem |= Attrs.IsSystem;
  Result->IsExternC |= Attrs.IsExternC;
  Result->ConfigMacrosExhaustive |= Attrs.IsExhaustive;
  Result->NoUndeclaredIncludes |= Attrs.NoUndeclaredIncludes;
  Result->Directory = FrameworkDir;

  // Chop off the first framework bit, as that is implied.
  StringRef RelativePath = UmbrellaName.str().substr(
      Result->getTopLevelModule()->Directory->getName().size());
  RelativePath = llvm::sys::path::relative_path(RelativePath);

  // umbrella header "umbrella-header-name"
  setUmbrellaHeaderAsWritten(Result, *UmbrellaHeader, ModuleName + ".h",
                             RelativePath);

  // export *
  Result->Exports.push_back(Module::ExportDecl(nullptr, true));

  // module * { export * }
  Result->InferSubmodules = true;
  Result->InferExportWildcard = true;

  // Look for subframeworks.
  std::error_code EC;
  SmallString<128> SubframeworksDirName = FrameworkDir.getName();
  llvm::sys::path::append(SubframeworksDirName, "Frameworks");
  llvm::sys::path::native(SubframeworksDirName);
  llvm::vfs::FileSystem &FS = FileMgr.getVirtualFileSystem();
  for (llvm::vfs::directory_iterator
           Dir = FS.dir_begin(SubframeworksDirName, EC),
           DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    if (!StringRef(Dir->path()).ends_with(".framework"))
      continue;

    if (auto SubframeworkDir = FileMgr.getOptionalDirectoryRef(Dir->path())) {
      // Note: as an egregious but useful hack, we use the real path here and
      // check whether it is actually a subdirectory of the parent directory.
      // This will not be the case if the 'subframework' is actually a symlink
      // out to a top-level framework.
      StringRef SubframeworkDirName =
          FileMgr.getCanonicalName(*SubframeworkDir);
      bool FoundParent = false;
      do {
        // Get the parent directory name.
        SubframeworkDirName
          = llvm::sys::path::parent_path(SubframeworkDirName);
        if (SubframeworkDirName.empty())
          break;

        if (auto SubDir =
                FileMgr.getOptionalDirectoryRef(SubframeworkDirName)) {
          if (*SubDir == FrameworkDir) {
            FoundParent = true;
            break;
          }
        }
      } while (true);

      if (!FoundParent)
        continue;

      // FIXME: Do we want to warn about subframeworks without umbrella headers?
      inferFrameworkModule(*SubframeworkDir, Attrs, Result);
    }
  }

  // If the module is a top-level framework, automatically link against the
  // framework.
  if (!Result->isSubFramework())
    inferFrameworkLink(Result);

  return Result;
}

Module *ModuleMap::createShadowedModule(StringRef Name, bool IsFramework,
                                        Module *ShadowingModule) {

  // Create a new module with this name.
  Module *Result = new (ModulesAlloc.Allocate())
      Module(ModuleConstructorTag{}, Name, SourceLocation(), /*Parent=*/nullptr,
             IsFramework, /*IsExplicit=*/false, NumCreatedModules++);
  Result->ShadowingModule = ShadowingModule;
  Result->markUnavailable(/*Unimportable*/true);
  ModuleScopeIDs[Result] = CurrentModuleScopeID;
  ShadowModules.push_back(Result);

  return Result;
}

void ModuleMap::setUmbrellaHeaderAsWritten(
    Module *Mod, FileEntryRef UmbrellaHeader, const Twine &NameAsWritten,
    const Twine &PathRelativeToRootModuleDirectory) {
  Headers[UmbrellaHeader].push_back(KnownHeader(Mod, NormalHeader));
  Mod->Umbrella = UmbrellaHeader;
  Mod->UmbrellaAsWritten = NameAsWritten.str();
  Mod->UmbrellaRelativeToRootModuleDirectory =
      PathRelativeToRootModuleDirectory.str();
  UmbrellaDirs[UmbrellaHeader.getDir()] = Mod;

  // Notify callbacks that we just added a new header.
  for (const auto &Cb : Callbacks)
    Cb->moduleMapAddUmbrellaHeader(UmbrellaHeader);
}

void ModuleMap::setUmbrellaDirAsWritten(
    Module *Mod, DirectoryEntryRef UmbrellaDir, const Twine &NameAsWritten,
    const Twine &PathRelativeToRootModuleDirectory) {
  Mod->Umbrella = UmbrellaDir;
  Mod->UmbrellaAsWritten = NameAsWritten.str();
  Mod->UmbrellaRelativeToRootModuleDirectory =
      PathRelativeToRootModuleDirectory.str();
  UmbrellaDirs[UmbrellaDir] = Mod;
}

void ModuleMap::addUnresolvedHeader(Module *Mod,
                                    Module::UnresolvedHeaderDirective Header,
                                    bool &NeedsFramework) {
  // If there is a builtin counterpart to this file, add it now so it can
  // wrap the system header.
  if (resolveAsBuiltinHeader(Mod, Header)) {
    // If we have both a builtin and system version of the file, the
    // builtin version may want to inject macros into the system header, so
    // force the system header to be treated as a textual header in this
    // case.
    Header.Kind = headerRoleToKind(ModuleMap::ModuleHeaderRole(
        headerKindToRole(Header.Kind) | ModuleMap::TextualHeader));
    Header.HasBuiltinHeader = true;
  }

  // If possible, don't stat the header until we need to. This requires the
  // user to have provided us with some stat information about the file.
  // FIXME: Add support for lazily stat'ing umbrella headers and excluded
  // headers.
  if ((Header.Size || Header.ModTime) && !Header.IsUmbrella &&
      Header.Kind != Module::HK_Excluded) {
    // We expect more variation in mtime than size, so if we're given both,
    // use the mtime as the key.
    if (Header.ModTime)
      LazyHeadersByModTime[*Header.ModTime].push_back(Mod);
    else
      LazyHeadersBySize[*Header.Size].push_back(Mod);
    Mod->UnresolvedHeaders.push_back(Header);
    return;
  }

  // We don't have stat information or can't defer looking this file up.
  // Perform the lookup now.
  resolveHeader(Mod, Header, NeedsFramework);
}

void ModuleMap::resolveHeaderDirectives(const FileEntry *File) const {
  auto BySize = LazyHeadersBySize.find(File->getSize());
  if (BySize != LazyHeadersBySize.end()) {
    for (auto *M : BySize->second)
      resolveHeaderDirectives(M, File);
    LazyHeadersBySize.erase(BySize);
  }

  auto ByModTime = LazyHeadersByModTime.find(File->getModificationTime());
  if (ByModTime != LazyHeadersByModTime.end()) {
    for (auto *M : ByModTime->second)
      resolveHeaderDirectives(M, File);
    LazyHeadersByModTime.erase(ByModTime);
  }
}

void ModuleMap::resolveHeaderDirectives(
    Module *Mod, std::optional<const FileEntry *> File) const {
  bool NeedsFramework = false;
  SmallVector<Module::UnresolvedHeaderDirective, 1> NewHeaders;
  const auto Size = File ? (*File)->getSize() : 0;
  const auto ModTime = File ? (*File)->getModificationTime() : 0;

  for (auto &Header : Mod->UnresolvedHeaders) {
    if (File && ((Header.ModTime && Header.ModTime != ModTime) ||
                 (Header.Size && Header.Size != Size)))
      NewHeaders.push_back(Header);
    else
      // This operation is logically const; we're just changing how we represent
      // the header information for this file.
      const_cast<ModuleMap *>(this)->resolveHeader(Mod, Header, NeedsFramework);
  }
  Mod->UnresolvedHeaders.swap(NewHeaders);
}

void ModuleMap::addHeader(Module *Mod, Module::Header Header,
                          ModuleHeaderRole Role, bool Imported) {
  KnownHeader KH(Mod, Role);

  FileEntryRef HeaderEntry = Header.Entry;

  // Only add each header to the headers list once.
  // FIXME: Should we diagnose if a header is listed twice in the
  // same module definition?
  auto &HeaderList = Headers[HeaderEntry];
  if (llvm::is_contained(HeaderList, KH))
    return;

  HeaderList.push_back(KH);
  Mod->addHeader(headerRoleToKind(Role), std::move(Header));

  bool isCompilingModuleHeader = Mod->isForBuilding(LangOpts);
  if (!Imported || isCompilingModuleHeader) {
    // When we import HeaderFileInfo, the external source is expected to
    // set the isModuleHeader flag itself.
    HeaderInfo.MarkFileModuleHeader(HeaderEntry, Role, isCompilingModuleHeader);
  }

  // Notify callbacks that we just added a new header.
  for (const auto &Cb : Callbacks)
    Cb->moduleMapAddHeader(HeaderEntry.getName());
}

bool ModuleMap::parseModuleMapFile(FileEntryRef File, bool IsSystem,
                                   DirectoryEntryRef Dir, FileID ID,
                                   SourceLocation ExternModuleLoc) {
  llvm::DenseMap<const FileEntry *, const modulemap::ModuleMapFile *>::iterator
      Known = ParsedModuleMap.find(File);
  if (Known != ParsedModuleMap.end())
    return Known->second == nullptr;

  // If the module map file wasn't already entered, do so now.
  if (ID.isInvalid()) {
    ID = SourceMgr.translateFile(File);
    if (ID.isInvalid() || SourceMgr.isLoadedFileID(ID)) {
      auto FileCharacter =
          IsSystem ? SrcMgr::C_System_ModuleMap : SrcMgr::C_User_ModuleMap;
      ID = SourceMgr.createFileID(File, ExternModuleLoc, FileCharacter);
    }
  }

  std::optional<llvm::MemoryBufferRef> Buffer = SourceMgr.getBufferOrNone(ID);
  if (!Buffer) {
    ParsedModuleMap[File] = nullptr;
    return true;
  }

  Diags.Report(diag::remark_mmap_parse) << File.getName();
  std::optional<modulemap::ModuleMapFile> MaybeMMF =
      modulemap::parseModuleMap(ID, Dir, SourceMgr, Diags, IsSystem, nullptr);

  if (!MaybeMMF) {
    ParsedModuleMap[File] = nullptr;
    return true;
  }

  ParsedModuleMaps.push_back(
      std::make_unique<modulemap::ModuleMapFile>(std::move(*MaybeMMF)));
  const modulemap::ModuleMapFile &MMF = *ParsedModuleMaps.back();
  std::vector<const modulemap::ExternModuleDecl *> PendingExternalModuleMaps;
  for (const auto &Decl : MMF.Decls) {
    std::visit(llvm::makeVisitor(
                   [&](const modulemap::ModuleDecl &MD) {
                     // Only use the first part of the name even for submodules.
                     // This will correctly load the submodule declarations when
                     // the module is loaded.
                     auto &ModuleDecls =
                         ParsedModules[StringRef(MD.Id.front().first)];
                     ModuleDecls.push_back(std::pair(&MMF, &MD));
                   },
                   [&](const modulemap::ExternModuleDecl &EMD) {
                     PendingExternalModuleMaps.push_back(&EMD);
                   }),
               Decl);
  }

  for (const modulemap::ExternModuleDecl *EMD : PendingExternalModuleMaps) {
    StringRef FileNameRef = EMD->Path;
    SmallString<128> ModuleMapFileName;
    if (llvm::sys::path::is_relative(FileNameRef)) {
      ModuleMapFileName += Dir.getName();
      llvm::sys::path::append(ModuleMapFileName, EMD->Path);
      FileNameRef = ModuleMapFileName;
    }

    if (auto EFile =
            SourceMgr.getFileManager().getOptionalFileRef(FileNameRef)) {
      parseModuleMapFile(*EFile, IsSystem, EFile->getDir(), FileID(),
                         ExternModuleLoc);
    }
  }

  ParsedModuleMap[File] = &MMF;

  for (const auto &Cb : Callbacks)
    Cb->moduleMapFileRead(SourceLocation(), File, IsSystem);

  return false;
}

FileID ModuleMap::getContainingModuleMapFileID(const Module *Module) const {
  if (Module->DefinitionLoc.isInvalid())
    return {};

  return SourceMgr.getFileID(Module->DefinitionLoc);
}

OptionalFileEntryRef
ModuleMap::getContainingModuleMapFile(const Module *Module) const {
  return SourceMgr.getFileEntryRefForID(getContainingModuleMapFileID(Module));
}

FileID ModuleMap::getModuleMapFileIDForUniquing(const Module *M) const {
  if (M->IsInferred) {
    assert(InferredModuleAllowedBy.count(M) && "missing inferred module map");
    return InferredModuleAllowedBy.find(M)->second;
  }
  return getContainingModuleMapFileID(M);
}

OptionalFileEntryRef
ModuleMap::getModuleMapFileForUniquing(const Module *M) const {
  return SourceMgr.getFileEntryRefForID(getModuleMapFileIDForUniquing(M));
}

void ModuleMap::setInferredModuleAllowedBy(Module *M, FileID ModMapFID) {
  M->IsInferred = true;
  InferredModuleAllowedBy[M] = ModMapFID;
}

std::error_code
ModuleMap::canonicalizeModuleMapPath(SmallVectorImpl<char> &Path) {
  StringRef Dir = llvm::sys::path::parent_path({Path.data(), Path.size()});

  // Do not canonicalize within the framework; the module map loader expects
  // Modules/ not Versions/A/Modules.
  if (llvm::sys::path::filename(Dir) == "Modules") {
    StringRef Parent = llvm::sys::path::parent_path(Dir);
    if (Parent.ends_with(".framework"))
      Dir = Parent;
  }

  FileManager &FM = SourceMgr.getFileManager();
  auto DirEntry = FM.getDirectoryRef(Dir.empty() ? "." : Dir);
  if (!DirEntry)
    return llvm::errorToErrorCode(DirEntry.takeError());

  // Canonicalize the directory.
  StringRef CanonicalDir = FM.getCanonicalName(*DirEntry);
  if (CanonicalDir != Dir)
    llvm::sys::path::replace_path_prefix(Path, Dir, CanonicalDir);

  // In theory, the filename component should also be canonicalized if it
  // on a case-insensitive filesystem. However, the extra canonicalization is
  // expensive and if clang looked up the filename it will always be lowercase.

  // Remove ., remove redundant separators, and switch to native separators.
  // This is needed for separators between CanonicalDir and the filename.
  llvm::sys::path::remove_dots(Path);

  return std::error_code();
}

void ModuleMap::addAdditionalModuleMapFile(const Module *M,
                                           FileEntryRef ModuleMap) {
  AdditionalModMaps[M].insert(ModuleMap);
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
    llvm::errs() << "  \"" << H->first.getName() << "\" -> ";
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
  auto *Top = Mod->getTopLevelModule();
  auto Unresolved = std::move(Top->UnresolvedDirectUses);
  Top->UnresolvedDirectUses.clear();
  for (auto &UDU : Unresolved) {
    Module *DirectUse = resolveModuleId(UDU, Top, Complain);
    if (DirectUse)
      Top->DirectUses.push_back(DirectUse);
    else
      Top->UnresolvedDirectUses.push_back(UDU);
  }
  return !Top->UnresolvedDirectUses.empty();
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

//----------------------------------------------------------------------------//
// Module map file loader
//----------------------------------------------------------------------------//

namespace clang {
class ModuleMapLoader {
  SourceManager &SourceMgr;

  DiagnosticsEngine &Diags;
  ModuleMap &Map;

  /// The current module map file.
  FileID ModuleMapFID;

  /// Source location of most recent loaded module declaration
  SourceLocation CurrModuleDeclLoc;

  /// The directory that file names in this module map file should
  /// be resolved relative to.
  DirectoryEntryRef Directory;

  /// Whether this module map is in a system header directory.
  bool IsSystem;

  /// Whether an error occurred.
  bool HadError = false;

  /// The active module.
  Module *ActiveModule = nullptr;

  /// Whether a module uses the 'requires excluded' hack to mark its
  /// contents as 'textual'.
  ///
  /// On older Darwin SDK versions, 'requires excluded' is used to mark the
  /// contents of the Darwin.C.excluded (assert.h) and Tcl.Private modules as
  /// non-modular headers.  For backwards compatibility, we continue to
  /// support this idiom for just these modules, and map the headers to
  /// 'textual' to match the original intent.
  llvm::SmallPtrSet<Module *, 2> UsesRequiresExcludedHack;

  void handleModuleDecl(const modulemap::ModuleDecl &MD);
  void handleExternModuleDecl(const modulemap::ExternModuleDecl &EMD);
  void handleRequiresDecl(const modulemap::RequiresDecl &RD);
  void handleHeaderDecl(const modulemap::HeaderDecl &HD);
  void handleUmbrellaDirDecl(const modulemap::UmbrellaDirDecl &UDD);
  void handleExportDecl(const modulemap::ExportDecl &ED);
  void handleExportAsDecl(const modulemap::ExportAsDecl &EAD);
  void handleUseDecl(const modulemap::UseDecl &UD);
  void handleLinkDecl(const modulemap::LinkDecl &LD);
  void handleConfigMacros(const modulemap::ConfigMacrosDecl &CMD);
  void handleConflict(const modulemap::ConflictDecl &CD);
  void handleInferredModuleDecl(const modulemap::ModuleDecl &MD);

  /// Private modules are canonicalized as Foo_Private. Clang provides extra
  /// module map search logic to find the appropriate private module when PCH
  /// is used with implicit module maps. Warn when private modules are written
  /// in other ways (FooPrivate and Foo.Private), providing notes and fixits.
  void diagnosePrivateModules(SourceLocation StartLoc);

  using Attributes = ModuleMap::Attributes;

public:
  ModuleMapLoader(SourceManager &SourceMgr, DiagnosticsEngine &Diags,
                  ModuleMap &Map, FileID ModuleMapFID,
                  DirectoryEntryRef Directory, bool IsSystem)
      : SourceMgr(SourceMgr), Diags(Diags), Map(Map),
        ModuleMapFID(ModuleMapFID), Directory(Directory), IsSystem(IsSystem) {}

  bool loadModuleDecl(const modulemap::ModuleDecl &MD);
  bool loadExternModuleDecl(const modulemap::ExternModuleDecl &EMD);
  bool parseAndLoadModuleMapFile(const modulemap::ModuleMapFile &MMF);
};

} // namespace clang

/// Private modules are canonicalized as Foo_Private. Clang provides extra
/// module map search logic to find the appropriate private module when PCH
/// is used with implicit module maps. Warn when private modules are written
/// in other ways (FooPrivate and Foo.Private), providing notes and fixits.
void ModuleMapLoader::diagnosePrivateModules(SourceLocation StartLoc) {
  auto GenNoteAndFixIt = [&](StringRef BadName, StringRef Canonical,
                             const Module *M, SourceRange ReplLoc) {
    auto D = Diags.Report(ActiveModule->DefinitionLoc,
                          diag::note_mmap_rename_top_level_private_module);
    D << BadName << M->Name;
    D << FixItHint::CreateReplacement(ReplLoc, Canonical);
  };

  for (auto E = Map.module_begin(); E != Map.module_end(); ++E) {
    auto const *M = E->getValue();
    if (M->Directory != ActiveModule->Directory)
      continue;

    SmallString<128> FullName(ActiveModule->getFullModuleName());
    if (!FullName.starts_with(M->Name) && !FullName.ends_with("Private"))
      continue;
    SmallString<128> FixedPrivModDecl;
    SmallString<128> Canonical(M->Name);
    Canonical.append("_Private");

    // Foo.Private -> Foo_Private
    if (ActiveModule->Parent && ActiveModule->Name == "Private" && !M->Parent &&
        M->Name == ActiveModule->Parent->Name) {
      Diags.Report(ActiveModule->DefinitionLoc,
                   diag::warn_mmap_mismatched_private_submodule)
          << FullName;

      SourceLocation FixItInitBegin = CurrModuleDeclLoc;
      if (StartLoc.isValid())
        FixItInitBegin = StartLoc;

      if (ActiveModule->Parent->IsFramework)
        FixedPrivModDecl.append("framework ");
      FixedPrivModDecl.append("module ");
      FixedPrivModDecl.append(Canonical);

      GenNoteAndFixIt(FullName, FixedPrivModDecl, M,
                      SourceRange(FixItInitBegin, ActiveModule->DefinitionLoc));
      continue;
    }

    // FooPrivate and whatnots -> Foo_Private
    if (!ActiveModule->Parent && !M->Parent && M->Name != ActiveModule->Name &&
        ActiveModule->Name != Canonical) {
      Diags.Report(ActiveModule->DefinitionLoc,
                   diag::warn_mmap_mismatched_private_module_name)
          << ActiveModule->Name;
      GenNoteAndFixIt(ActiveModule->Name, Canonical, M,
                      SourceRange(ActiveModule->DefinitionLoc));
    }
  }
}

void ModuleMapLoader::handleModuleDecl(const modulemap::ModuleDecl &MD) {
  if (MD.Id.front().first == "*")
    return handleInferredModuleDecl(MD);

  CurrModuleDeclLoc = MD.Location;

  Module *PreviousActiveModule = ActiveModule;
  if (MD.Id.size() > 1) {
    // This module map defines a submodule. Go find the module of which it
    // is a submodule.
    ActiveModule = nullptr;
    const Module *TopLevelModule = nullptr;
    for (unsigned I = 0, N = MD.Id.size() - 1; I != N; ++I) {
      if (Module *Next =
              Map.lookupModuleQualified(MD.Id[I].first, ActiveModule)) {
        if (I == 0)
          TopLevelModule = Next;
        ActiveModule = Next;
        continue;
      }

      Diags.Report(MD.Id[I].second, diag::err_mmap_missing_parent_module)
          << MD.Id[I].first << (ActiveModule != nullptr)
          << (ActiveModule
                  ? ActiveModule->getTopLevelModule()->getFullModuleName()
                  : "");
      HadError = true;
    }

    if (TopLevelModule &&
        ModuleMapFID != Map.getContainingModuleMapFileID(TopLevelModule)) {
      assert(ModuleMapFID !=
                 Map.getModuleMapFileIDForUniquing(TopLevelModule) &&
             "submodule defined in same file as 'module *' that allowed its "
             "top-level module");
      Map.addAdditionalModuleMapFile(
          TopLevelModule, *SourceMgr.getFileEntryRefForID(ModuleMapFID));
    }
  }

  StringRef ModuleName = MD.Id.back().first;
  SourceLocation ModuleNameLoc = MD.Id.back().second;

  // Determine whether this (sub)module has already been defined.
  Module *ShadowingModule = nullptr;
  if (Module *Existing = Map.lookupModuleQualified(ModuleName, ActiveModule)) {
    // We might see a (re)definition of a module that we already have a
    // definition for in four cases:
    //  - If we loaded one definition from an AST file and we've just found a
    //    corresponding definition in a module map file, or
    bool LoadedFromASTFile = Existing->IsFromModuleFile;
    //  - If we previously inferred this module from different module map file.
    bool Inferred = Existing->IsInferred;
    //  - If we're building a framework that vends a module map, we might've
    //    previously seen the one in intermediate products and now the system
    //    one.
    // FIXME: If we're parsing module map file that looks like this:
    //          framework module FW { ... }
    //          module FW.Sub { ... }
    //        We can't check the framework qualifier, since it's not attached to
    //        the definition of Sub. Checking that qualifier on \c Existing is
    //        not correct either, since we might've previously seen:
    //          module FW { ... }
    //          module FW.Sub { ... }
    //        We should enforce consistency of redefinitions so that we can rely
    //        that \c Existing is part of a framework iff the redefinition of FW
    //        we have just skipped had it too. Once we do that, stop checking
    //        the local framework qualifier and only rely on \c Existing.
    bool PartOfFramework = MD.Framework || Existing->isPartOfFramework();
    //  - If we're building a (preprocessed) module and we've just loaded the
    //    module map file from which it was created.
    bool ParsedAsMainInput =
        Map.LangOpts.getCompilingModule() == LangOptions::CMK_ModuleMap &&
        Map.LangOpts.CurrentModule == ModuleName &&
        SourceMgr.getDecomposedLoc(ModuleNameLoc).first !=
            SourceMgr.getDecomposedLoc(Existing->DefinitionLoc).first;
    // TODO: Remove this check when we can avoid loading module maps multiple
    //       times.
    bool SameModuleDecl = ModuleNameLoc == Existing->DefinitionLoc;
    if (LoadedFromASTFile || Inferred || PartOfFramework || ParsedAsMainInput ||
        SameModuleDecl) {
      ActiveModule = PreviousActiveModule;
      // Skip the module definition.
      return;
    }

    if (!Existing->Parent && Map.mayShadowNewModule(Existing)) {
      ShadowingModule = Existing;
    } else {
      // This is not a shawdowed module decl, it is an illegal redefinition.
      Diags.Report(ModuleNameLoc, diag::err_mmap_module_redefinition)
          << ModuleName;
      Diags.Report(Existing->DefinitionLoc, diag::note_mmap_prev_definition);
      HadError = true;
      return;
    }
  }

  // Start defining this module.
  if (ShadowingModule) {
    ActiveModule =
        Map.createShadowedModule(ModuleName, MD.Framework, ShadowingModule);
  } else {
    ActiveModule = Map.findOrCreateModuleFirst(ModuleName, ActiveModule,
                                               MD.Framework, MD.Explicit);
  }

  ActiveModule->DefinitionLoc = ModuleNameLoc;
  if (MD.Attrs.IsSystem || IsSystem)
    ActiveModule->IsSystem = true;
  if (MD.Attrs.IsExternC)
    ActiveModule->IsExternC = true;
  if (MD.Attrs.NoUndeclaredIncludes)
    ActiveModule->NoUndeclaredIncludes = true;
  ActiveModule->Directory = Directory;

  StringRef MapFileName(
      SourceMgr.getFileEntryRefForID(ModuleMapFID)->getName());
  if (MapFileName.ends_with("module.private.modulemap") ||
      MapFileName.ends_with("module_private.map")) {
    ActiveModule->ModuleMapIsPrivate = true;
  }

  // Private modules named as FooPrivate, Foo.Private or similar are likely a
  // user error; provide warnings, notes and fixits to direct users to use
  // Foo_Private instead.
  SourceLocation StartLoc =
      SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID());
  if (Map.HeaderInfo.getHeaderSearchOpts().ImplicitModuleMaps &&
      !Diags.isIgnored(diag::warn_mmap_mismatched_private_submodule,
                       StartLoc) &&
      !Diags.isIgnored(diag::warn_mmap_mismatched_private_module_name,
                       StartLoc) &&
      ActiveModule->ModuleMapIsPrivate)
    diagnosePrivateModules(MD.Location);

  for (const modulemap::Decl &Decl : MD.Decls) {
    std::visit(
        llvm::makeVisitor(
            [&](const modulemap::RequiresDecl &RD) { handleRequiresDecl(RD); },
            [&](const modulemap::HeaderDecl &HD) { handleHeaderDecl(HD); },
            [&](const modulemap::UmbrellaDirDecl &UDD) {
              handleUmbrellaDirDecl(UDD);
            },
            [&](const modulemap::ModuleDecl &MD) { handleModuleDecl(MD); },
            [&](const modulemap::ExportDecl &ED) { handleExportDecl(ED); },
            [&](const modulemap::ExportAsDecl &EAD) {
              handleExportAsDecl(EAD);
            },
            [&](const modulemap::ExternModuleDecl &EMD) {
              handleExternModuleDecl(EMD);
            },
            [&](const modulemap::UseDecl &UD) { handleUseDecl(UD); },
            [&](const modulemap::LinkDecl &LD) { handleLinkDecl(LD); },
            [&](const modulemap::ConfigMacrosDecl &CMD) {
              handleConfigMacros(CMD);
            },
            [&](const modulemap::ConflictDecl &CD) { handleConflict(CD); },
            [&](const modulemap::ExcludeDecl &ED) {
              Diags.Report(ED.Location, diag::err_mmap_expected_member);
            }),
        Decl);
  }

  // If the active module is a top-level framework, and there are no link
  // libraries, automatically link against the framework.
  if (ActiveModule->IsFramework && !ActiveModule->isSubFramework() &&
      ActiveModule->LinkLibraries.empty())
    inferFrameworkLink(ActiveModule);

  // If the module meets all requirements but is still unavailable, mark the
  // whole tree as unavailable to prevent it from building.
  if (!ActiveModule->IsAvailable && !ActiveModule->IsUnimportable &&
      ActiveModule->Parent) {
    ActiveModule->getTopLevelModule()->markUnavailable(/*Unimportable=*/false);
    ActiveModule->getTopLevelModule()->MissingHeaders.append(
      ActiveModule->MissingHeaders.begin(), ActiveModule->MissingHeaders.end());
  }

  // We're done parsing this module. Pop back to the previous module.
  ActiveModule = PreviousActiveModule;
}

void ModuleMapLoader::handleExternModuleDecl(
    const modulemap::ExternModuleDecl &EMD) {
  StringRef FileNameRef = EMD.Path;
  SmallString<128> ModuleMapFileName;
  if (llvm::sys::path::is_relative(FileNameRef)) {
    ModuleMapFileName += Directory.getName();
    llvm::sys::path::append(ModuleMapFileName, EMD.Path);
    FileNameRef = ModuleMapFileName;
  }
  if (auto File = SourceMgr.getFileManager().getOptionalFileRef(FileNameRef))
    Map.parseAndLoadModuleMapFile(
        *File, IsSystem,
        Map.HeaderInfo.getHeaderSearchOpts().ModuleMapFileHomeIsCwd
            ? Directory
            : File->getDir(),
        FileID(), nullptr, EMD.Location);
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
  if (Feature == "excluded" &&
      (M->fullModuleNameIs({"Darwin", "C", "excluded"}) ||
       M->fullModuleNameIs({"Tcl", "Private"}))) {
    IsRequiresExcludedHack = true;
    return false;
  } else if (Feature == "cplusplus" && M->fullModuleNameIs({"IOKit", "avc"})) {
    return false;
  }

  return true;
}

void ModuleMapLoader::handleRequiresDecl(const modulemap::RequiresDecl &RD) {

  for (const modulemap::RequiresFeature &RF : RD.Features) {
    bool IsRequiresExcludedHack = false;
    bool ShouldAddRequirement =
        shouldAddRequirement(ActiveModule, RF.Feature, IsRequiresExcludedHack);

    if (IsRequiresExcludedHack)
      UsesRequiresExcludedHack.insert(ActiveModule);

    if (ShouldAddRequirement) {
      // Add this feature.
      ActiveModule->addRequirement(RF.Feature, RF.RequiredState, Map.LangOpts,
                                   *Map.Target);
    }
  }
}

void ModuleMapLoader::handleHeaderDecl(const modulemap::HeaderDecl &HD) {
  // We've already consumed the first token.
  ModuleMap::ModuleHeaderRole Role = ModuleMap::NormalHeader;

  if (HD.Private) {
    Role = ModuleMap::PrivateHeader;
  } else if (HD.Excluded) {
    Role = ModuleMap::ExcludedHeader;
  }

  if (HD.Textual)
    Role = ModuleMap::ModuleHeaderRole(Role | ModuleMap::TextualHeader);

  if (UsesRequiresExcludedHack.count(ActiveModule)) {
    // Mark this header 'textual' (see doc comment for
    // Module::UsesRequiresExcludedHack).
    Role = ModuleMap::ModuleHeaderRole(Role | ModuleMap::TextualHeader);
  }

  Module::UnresolvedHeaderDirective Header;
  Header.FileName = HD.Path;
  Header.FileNameLoc = HD.PathLoc;
  Header.IsUmbrella = HD.Umbrella;
  Header.Kind = Map.headerRoleToKind(Role);

  // Check whether we already have an umbrella.
  if (Header.IsUmbrella &&
      !std::holds_alternative<std::monostate>(ActiveModule->Umbrella)) {
    Diags.Report(Header.FileNameLoc, diag::err_mmap_umbrella_clash)
      << ActiveModule->getFullModuleName();
    HadError = true;
    return;
  }

  if (HD.Size)
    Header.Size = HD.Size;
  if (HD.MTime)
    Header.ModTime = HD.MTime;

  bool NeedsFramework = false;
  // Don't add headers to the builtin modules if the builtin headers belong to
  // the system modules, with the exception of __stddef_max_align_t.h which
  // always had its own module.
  if (!Map.LangOpts.BuiltinHeadersInSystemModules ||
      !isBuiltInModuleName(ActiveModule->getTopLevelModuleName()) ||
      ActiveModule->fullModuleNameIs({"_Builtin_stddef", "max_align_t"}))
    Map.addUnresolvedHeader(ActiveModule, std::move(Header), NeedsFramework);

  if (NeedsFramework)
    Diags.Report(CurrModuleDeclLoc, diag::note_mmap_add_framework_keyword)
      << ActiveModule->getFullModuleName()
      << FixItHint::CreateReplacement(CurrModuleDeclLoc, "framework module");
}

static bool compareModuleHeaders(const Module::Header &A,
                                 const Module::Header &B) {
  return A.NameAsWritten < B.NameAsWritten;
}

void ModuleMapLoader::handleUmbrellaDirDecl(
    const modulemap::UmbrellaDirDecl &UDD) {
  std::string DirName = std::string(UDD.Path);
  std::string DirNameAsWritten = DirName;

  // Check whether we already have an umbrella.
  if (!std::holds_alternative<std::monostate>(ActiveModule->Umbrella)) {
    Diags.Report(UDD.Location, diag::err_mmap_umbrella_clash)
        << ActiveModule->getFullModuleName();
    HadError = true;
    return;
  }

  // Look for this file.
  OptionalDirectoryEntryRef Dir;
  if (llvm::sys::path::is_absolute(DirName)) {
    Dir = SourceMgr.getFileManager().getOptionalDirectoryRef(DirName);
  } else {
    SmallString<128> PathName;
    PathName = Directory.getName();
    llvm::sys::path::append(PathName, DirName);
    Dir = SourceMgr.getFileManager().getOptionalDirectoryRef(PathName);
  }

  if (!Dir) {
    Diags.Report(UDD.Location, diag::warn_mmap_umbrella_dir_not_found)
        << DirName;
    return;
  }

  if (UsesRequiresExcludedHack.count(ActiveModule)) {
    // Mark this header 'textual' (see doc comment for
    // ModuleMapLoader::UsesRequiresExcludedHack). Although iterating over the
    // directory is relatively expensive, in practice this only applies to the
    // uncommonly used Tcl module on Darwin platforms.
    std::error_code EC;
    SmallVector<Module::Header, 6> Headers;
    llvm::vfs::FileSystem &FS =
        SourceMgr.getFileManager().getVirtualFileSystem();
    for (llvm::vfs::recursive_directory_iterator I(FS, Dir->getName(), EC), E;
         I != E && !EC; I.increment(EC)) {
      if (auto FE = SourceMgr.getFileManager().getOptionalFileRef(I->path())) {
        Module::Header Header = {"", std::string(I->path()), *FE};
        Headers.push_back(std::move(Header));
      }
    }

    // Sort header paths so that the pcm doesn't depend on iteration order.
    llvm::stable_sort(Headers, compareModuleHeaders);

    for (auto &Header : Headers)
      Map.addHeader(ActiveModule, std::move(Header), ModuleMap::TextualHeader);
    return;
  }

  if (Module *OwningModule = Map.UmbrellaDirs[*Dir]) {
    Diags.Report(UDD.Location, diag::err_mmap_umbrella_clash)
        << OwningModule->getFullModuleName();
    HadError = true;
    return;
  }

  // Record this umbrella directory.
  Map.setUmbrellaDirAsWritten(ActiveModule, *Dir, DirNameAsWritten, DirName);
}

void ModuleMapLoader::handleExportDecl(const modulemap::ExportDecl &ED) {
  Module::UnresolvedExportDecl Unresolved = {ED.Location, ED.Id, ED.Wildcard};
  ActiveModule->UnresolvedExports.push_back(Unresolved);
}

void ModuleMapLoader::handleExportAsDecl(const modulemap::ExportAsDecl &EAD) {
  const auto &ModName = EAD.Id.front();

  if (!ActiveModule->ExportAsModule.empty()) {
    if (ActiveModule->ExportAsModule == ModName.first) {
      Diags.Report(ModName.second, diag::warn_mmap_redundant_export_as)
          << ActiveModule->Name << ModName.first;
    } else {
      Diags.Report(ModName.second, diag::err_mmap_conflicting_export_as)
          << ActiveModule->Name << ActiveModule->ExportAsModule
          << ModName.first;
    }
  }

  ActiveModule->ExportAsModule = ModName.first;
  Map.addLinkAsDependency(ActiveModule);
}

void ModuleMapLoader::handleUseDecl(const modulemap::UseDecl &UD) {
  if (ActiveModule->Parent)
    Diags.Report(UD.Location, diag::err_mmap_use_decl_submodule);
  else
    ActiveModule->UnresolvedDirectUses.push_back(UD.Id);
}

void ModuleMapLoader::handleLinkDecl(const modulemap::LinkDecl &LD) {
  ActiveModule->LinkLibraries.push_back(
      Module::LinkLibrary(std::string{LD.Library}, LD.Framework));
}

void ModuleMapLoader::handleConfigMacros(
    const modulemap::ConfigMacrosDecl &CMD) {
  if (ActiveModule->Parent) {
    Diags.Report(CMD.Location, diag::err_mmap_config_macro_submodule);
    return;
  }

  // TODO: Is this really the behavior we want for multiple config_macros
  //       declarations? If any of them are exhaustive then all of them are.
  if (CMD.Exhaustive) {
    ActiveModule->ConfigMacrosExhaustive = true;
  }
  ActiveModule->ConfigMacros.insert(ActiveModule->ConfigMacros.end(),
                                    CMD.Macros.begin(), CMD.Macros.end());
}

void ModuleMapLoader::handleConflict(const modulemap::ConflictDecl &CD) {
  Module::UnresolvedConflict Conflict;

  Conflict.Id = CD.Id;
  Conflict.Message = CD.Message;

  // FIXME: when we move to C++20 we should consider using emplace_back
  ActiveModule->UnresolvedConflicts.push_back(std::move(Conflict));
}

void ModuleMapLoader::handleInferredModuleDecl(
    const modulemap::ModuleDecl &MD) {
  SourceLocation StarLoc = MD.Id.front().second;

  // Inferred modules must be submodules.
  if (!ActiveModule && !MD.Framework) {
    Diags.Report(StarLoc, diag::err_mmap_top_level_inferred_submodule);
    return;
  }

  if (ActiveModule) {
    // Inferred modules must have umbrella directories.
    if (ActiveModule->IsAvailable && !ActiveModule->getEffectiveUmbrellaDir()) {
      Diags.Report(StarLoc, diag::err_mmap_inferred_no_umbrella);
      return;
    }

    // Check for redefinition of an inferred module.
    if (ActiveModule->InferSubmodules) {
      Diags.Report(StarLoc, diag::err_mmap_inferred_redef);
      if (ActiveModule->InferredSubmoduleLoc.isValid())
        Diags.Report(ActiveModule->InferredSubmoduleLoc,
                     diag::note_mmap_prev_definition);
      return;
    }

    // Check for the 'framework' keyword, which is not permitted here.
    if (MD.Framework) {
      Diags.Report(StarLoc, diag::err_mmap_inferred_framework_submodule);
      return;
    }
  } else if (MD.Explicit) {
    Diags.Report(StarLoc, diag::err_mmap_explicit_inferred_framework);
    return;
  }

  if (ActiveModule) {
    // Note that we have an inferred submodule.
    ActiveModule->InferSubmodules = true;
    ActiveModule->InferredSubmoduleLoc = StarLoc;
    ActiveModule->InferExplicitSubmodules = MD.Explicit;
  } else {
    // We'll be inferring framework modules for this directory.
    auto &InfDir = Map.InferredDirectories[Directory];
    InfDir.InferModules = true;
    InfDir.Attrs = MD.Attrs;
    InfDir.ModuleMapFID = ModuleMapFID;
    // FIXME: Handle the 'framework' keyword.
  }

  for (const modulemap::Decl &Decl : MD.Decls) {
    std::visit(
        llvm::makeVisitor(
            [&](const auto &Other) {
              Diags.Report(Other.Location,
                           diag::err_mmap_expected_inferred_member)
                  << (ActiveModule != nullptr);
            },
            [&](const modulemap::ExcludeDecl &ED) {
              // Only inferred frameworks can have exclude decls
              if (ActiveModule) {
                Diags.Report(ED.Location,
                             diag::err_mmap_expected_inferred_member)
                    << (ActiveModule != nullptr);
                HadError = true;
                return;
              }
              Map.InferredDirectories[Directory].ExcludedModules.emplace_back(
                  ED.Module);
            },
            [&](const modulemap::ExportDecl &ED) {
              // Only inferred submodules can have export decls
              if (!ActiveModule) {
                Diags.Report(ED.Location,
                             diag::err_mmap_expected_inferred_member)
                    << (ActiveModule != nullptr);
                HadError = true;
                return;
              }

              if (ED.Wildcard && ED.Id.size() == 0)
                ActiveModule->InferExportWildcard = true;
              else
                Diags.Report(ED.Id.front().second,
                             diag::err_mmap_expected_export_wildcard);
            }),
        Decl);
  }
}

bool ModuleMapLoader::loadModuleDecl(const modulemap::ModuleDecl &MD) {
  handleModuleDecl(MD);
  return HadError;
}

bool ModuleMapLoader::loadExternModuleDecl(
    const modulemap::ExternModuleDecl &EMD) {
  handleExternModuleDecl(EMD);
  return HadError;
}

bool ModuleMapLoader::parseAndLoadModuleMapFile(
    const modulemap::ModuleMapFile &MMF) {
  for (const auto &Decl : MMF.Decls) {
    std::visit(
        llvm::makeVisitor(
            [&](const modulemap::ModuleDecl &MD) { handleModuleDecl(MD); },
            [&](const modulemap::ExternModuleDecl &EMD) {
              handleExternModuleDecl(EMD);
            }),
        Decl);
  }
  return HadError;
}

Module *ModuleMap::findOrLoadModule(StringRef Name) {
  llvm::StringMap<Module *>::const_iterator Known = Modules.find(Name);
  if (Known != Modules.end())
    return Known->getValue();

  auto ParsedMod = ParsedModules.find(Name);
  if (ParsedMod == ParsedModules.end())
    return nullptr;

  Diags.Report(diag::remark_mmap_load_module) << Name;

  for (const auto &ModuleDecl : ParsedMod->second) {
    const modulemap::ModuleMapFile &MMF = *ModuleDecl.first;
    ModuleMapLoader Loader(SourceMgr, Diags, const_cast<ModuleMap &>(*this),
                           MMF.ID, *MMF.Dir, MMF.IsSystem);
    if (Loader.loadModuleDecl(*ModuleDecl.second))
      return nullptr;
  }

  return findModule(Name);
}

bool ModuleMap::parseAndLoadModuleMapFile(FileEntryRef File, bool IsSystem,
                                          DirectoryEntryRef Dir, FileID ID,
                                          unsigned *Offset,
                                          SourceLocation ExternModuleLoc) {
  assert(Target && "Missing target information");
  llvm::DenseMap<const FileEntry *, bool>::iterator Known =
      LoadedModuleMap.find(File);
  if (Known != LoadedModuleMap.end())
    return Known->second;

  // If the module map file wasn't already entered, do so now.
  if (ID.isInvalid()) {
    ID = SourceMgr.translateFile(File);
    // TODO: The way we compute affecting module maps requires this to be a
    //       local FileID. This should be changed to reuse loaded FileIDs when
    //       available, and change the way that affecting module maps are
    //       computed to not require this.
    if (ID.isInvalid() || SourceMgr.isLoadedFileID(ID)) {
      auto FileCharacter =
          IsSystem ? SrcMgr::C_System_ModuleMap : SrcMgr::C_User_ModuleMap;
      ID = SourceMgr.createFileID(File, ExternModuleLoc, FileCharacter);
    }
  }

  assert(Target && "Missing target information");
  std::optional<llvm::MemoryBufferRef> Buffer = SourceMgr.getBufferOrNone(ID);
  if (!Buffer)
    return LoadedModuleMap[File] = true;
  assert((!Offset || *Offset <= Buffer->getBufferSize()) &&
         "invalid buffer offset");

  std::optional<modulemap::ModuleMapFile> MMF =
      modulemap::parseModuleMap(ID, Dir, SourceMgr, Diags, IsSystem, Offset);
  bool Result = false;
  if (MMF) {
    Diags.Report(diag::remark_mmap_load) << File.getName();
    ModuleMapLoader Loader(SourceMgr, Diags, *this, ID, Dir, IsSystem);
    Result = Loader.parseAndLoadModuleMapFile(*MMF);
  }
  LoadedModuleMap[File] = Result;

  // Notify callbacks that we observed it.
  // FIXME: We should only report module maps that were actually used.
  for (const auto &Cb : Callbacks)
    Cb->moduleMapFileRead(MMF ? MMF->Start : SourceLocation(), File, IsSystem);

  return Result;
}
