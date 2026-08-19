//===--- DependencyFile.cpp - Generate dependency file --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This code generates dependency files.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/MakeSupport.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace clang;

namespace {
struct DepCollectorPPCallbacks : public PPCallbacks {
  DependencyCollector &DepCollector;
  Preprocessor &PP;
  DepCollectorPPCallbacks(DependencyCollector &L, Preprocessor &PP)
      : DepCollector(L), PP(PP) {}

  void LexedFileChanged(FileID FID, LexedFileChangeReason Reason,
                        SrcMgr::CharacteristicKind FileType, FileID PrevFID,
                        SourceLocation Loc) override {
    if (Reason != PPCallbacks::LexedFileChangeReason::EnterFile)
      return;

    // Dependency generation really does want to go all the way to the
    // file entry for a source location to find out what is depended on.
    // We do not want #line markers to affect dependency generation!
    if (std::optional<StringRef> Filename =
            PP.getSourceManager().getNonBuiltinFilenameForID(FID))
      DepCollector.maybeAddDependency(
          llvm::sys::path::remove_leading_dotslash(*Filename),
          /*FromModule*/ false, isSystem(FileType), /*IsModuleFile*/ false,
          /*IsDirectModuleImport*/ false, /*IsMissing*/ false);
  }

  void FileSkipped(const FileEntryRef &SkippedFile, const Token &FilenameTok,
                   SrcMgr::CharacteristicKind FileType) override {
    StringRef Filename =
        llvm::sys::path::remove_leading_dotslash(SkippedFile.getName());
    DepCollector.maybeAddDependency(Filename, /*FromModule=*/false,
                                    /*IsSystem=*/isSystem(FileType),
                                    /*IsModuleFile=*/false,
                                    /*IsDirectModuleImport=*/false,
                                    /*IsMissing=*/false);
  }

  void EmbedDirective(SourceLocation, StringRef, bool,
                      OptionalFileEntryRef File,
                      const LexEmbedParametersResult &) override {
    assert(File && "expected to only be called when the file is found");
    StringRef FileName =
        llvm::sys::path::remove_leading_dotslash(File->getName());
    DepCollector.maybeAddDependency(FileName,
                                    /*FromModule*/ false,
                                    /*IsSystem*/ false,
                                    /*IsModuleFile*/ false,
                                    /*IsDirectModuleImport*/ false,
                                    /*IsMissing*/ false);
  }

  bool EmbedFileNotFound(StringRef FileName) override {
    DepCollector.maybeAddDependency(
        llvm::sys::path::remove_leading_dotslash(FileName),
        /*FromModule=*/false,
        /*IsSystem=*/false,
        /*IsModuleFile=*/false,
        /*IsDirectModuleImport=*/false,
        /*IsMissing=*/true);
    // Return true to silence the file not found diagnostic.
    return true;
  }

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override {
    if (!File)
      DepCollector.maybeAddDependency(FileName, /*FromModule*/ false,
                                      /*IsSystem*/ false,
                                      /*IsModuleFile*/ false,
                                      /*IsDirectModuleImport*/ false,
                                      /*IsMissing*/ true);
    // Files that actually exist are handled by FileChanged.
  }

  void HasEmbed(SourceLocation, StringRef, bool,
                OptionalFileEntryRef File) override {
    if (!File)
      return;
    StringRef Filename =
        llvm::sys::path::remove_leading_dotslash(File->getName());
    DepCollector.maybeAddDependency(Filename,
                                    /*FromModule=*/false, false,
                                    /*IsModuleFile=*/false,
                                    /*IsDirectModuleImport=*/false,
                                    /*IsMissing=*/false);
  }

  void HasInclude(SourceLocation Loc, StringRef SpelledFilename, bool IsAngled,
                  OptionalFileEntryRef File,
                  SrcMgr::CharacteristicKind FileType) override {
    if (!File)
      return;
    StringRef Filename =
        llvm::sys::path::remove_leading_dotslash(File->getName());
    DepCollector.maybeAddDependency(Filename, /*FromModule=*/false,
                                    /*IsSystem=*/isSystem(FileType),
                                    /*IsModuleFile=*/false,
                                    /*IsDirectModuleImport=*/false,
                                    /*IsMissing=*/false);
  }

  void EndOfMainFile() override {
    DepCollector.finishedMainFile(PP.getDiagnostics());
  }
};

struct DepCollectorMMCallbacks : public ModuleMapCallbacks {
  DependencyCollector &DepCollector;
  DepCollectorMMCallbacks(DependencyCollector &DC) : DepCollector(DC) {}

  void moduleMapFileRead(SourceLocation Loc, FileEntryRef Entry,
                         bool IsSystem) override {
    StringRef Filename = Entry.getName();
    DepCollector.maybeAddDependency(Filename, /*FromModule*/ false,
                                    /*IsSystem*/ IsSystem,
                                    /*IsModuleFile*/ false,
                                    /*IsDirectModuleImport*/ false,
                                    /*IsMissing*/ false);
  }
};

struct DepCollectorASTListener : public ASTReaderListener {
  DependencyCollector &DepCollector;
  FileManager &FileMgr;
  DepCollectorASTListener(DependencyCollector &L, FileManager &FileMgr)
      : DepCollector(L), FileMgr(FileMgr) {}
  bool needsInputFileVisitation() override { return true; }
  bool needsSystemInputFileVisitation() override {
    return DepCollector.needSystemDependencies();
  }
  void visitModuleFile(ModuleFileName Filename, serialization::ModuleKind Kind,
                       bool DirectlyImported) override {
    DepCollector.maybeAddDependency(Filename, /*FromModule*/ true,
                                    /*IsSystem*/ false, /*IsModuleFile*/ true,
                                    /*IsDirectModuleImport*/ DirectlyImported,
                                    /*IsMissing*/ false);
  }
  bool visitInputFile(StringRef Filename, bool IsSystem,
                      bool IsOverridden, bool IsExplicitModule) override {
    if (IsOverridden || IsExplicitModule)
      return true;

    // Run this through the FileManager in order to respect 'use-external-name'
    // in case we have a VFS overlay.
    if (auto FE = FileMgr.getOptionalFileRef(Filename))
      Filename = FE->getName();

    DepCollector.maybeAddDependency(Filename, /*FromModule*/ true, IsSystem,
                                    /*IsModuleFile*/ false,
                                    /*IsDirectModuleImport*/ false,
                                    /*IsMissing*/ false);
    return true;
  }
};
} // end anonymous namespace

void DependencyCollector::maybeAddDependency(StringRef Filename,
                                             bool FromModule, bool IsSystem,
                                             bool IsModuleFile,
                                             bool IsDirectModuleImport,
                                             bool IsMissing) {
  if (sawDependency(Filename, FromModule, IsSystem, IsModuleFile,
                    IsDirectModuleImport, IsMissing))
    addDependency(Filename);
}

bool DependencyCollector::addDependency(StringRef Filename) {
  StringRef SearchPath;
#ifdef _WIN32
  // Make the search insensitive to case and separators.
  llvm::SmallString<256> TmpPath = Filename;
  llvm::sys::path::native(TmpPath);
  std::transform(TmpPath.begin(), TmpPath.end(), TmpPath.begin(), ::tolower);
  SearchPath = TmpPath.str();
#else
  SearchPath = Filename;
#endif

  if (Seen.insert(SearchPath).second) {
    Dependencies.push_back(std::string(Filename));
    return true;
  }
  return false;
}

static bool isSpecialFilename(StringRef Filename) {
  return Filename == "<built-in>";
}

bool DependencyCollector::sawDependency(StringRef Filename, bool FromModule,
                                        bool IsSystem, bool IsModuleFile,
                                        bool IsDirectModuleImport,
                                        bool IsMissing) {
  return !isSpecialFilename(Filename) &&
         (needSystemDependencies() || !IsSystem);
}

DependencyCollector::~DependencyCollector() { }
void DependencyCollector::attachToPreprocessor(Preprocessor &PP) {
  PP.addPPCallbacks(std::make_unique<DepCollectorPPCallbacks>(*this, PP));
  PP.getHeaderSearchInfo().getModuleMap().addModuleMapCallbacks(
      std::make_unique<DepCollectorMMCallbacks>(*this));
}
void DependencyCollector::attachToASTReader(ASTReader &R) {
  R.addListener(
      std::make_unique<DepCollectorASTListener>(*this, R.getFileManager()));
}

DependencyFileGenerator::DependencyFileGenerator(
    const DependencyOutputOptions &Opts)
    : OutputFile(Opts.OutputFile), Targets(Opts.Targets),
      IncludeSystemHeaders(Opts.IncludeSystemHeaders),
      PhonyTarget(Opts.UsePhonyTargets),
      AddMissingHeaderDeps(Opts.AddMissingHeaderDeps), SeenMissingHeader(false),
      IncludeModuleFiles(
          static_cast<ModuleFileDepsKind>(Opts.IncludeModuleFiles)),
      OutputFormat(Opts.OutputFormat), InputFileIndex(0) {
  for (const auto &ExtraDep : Opts.ExtraDeps) {
    if (addDependency(ExtraDep.first))
      ++InputFileIndex;
  }
}

void DependencyFileGenerator::attachToPreprocessor(Preprocessor &PP) {
  // Disable the "file not found" diagnostic if the -MG option was given.
  if (AddMissingHeaderDeps)
    PP.SetSuppressIncludeNotFoundError(true);

  DependencyCollector::attachToPreprocessor(PP);
}

bool DependencyFileGenerator::sawDependency(StringRef Filename, bool FromModule,
                                            bool IsSystem, bool IsModuleFile,
                                            bool IsDirectModuleImport,
                                            bool IsMissing) {
  if (IsMissing) {
    // Handle the case of missing file from an inclusion directive.
    if (AddMissingHeaderDeps)
      return true;
    SeenMissingHeader = true;
    return false;
  }
  if (IsModuleFile) {
    if (IncludeModuleFiles == MFDK_None)
      return false;
    if (IncludeModuleFiles == MFDK_Direct && !IsDirectModuleImport)
      return false;
  }

  if (isSpecialFilename(Filename))
    return false;

  if (IncludeSystemHeaders)
    return true;

  return !IsSystem;
}

void DependencyFileGenerator::finishedMainFile(DiagnosticsEngine &Diags) {
  outputDependencyFile(Diags);
}

void DependencyFileGenerator::outputDependencyFile(DiagnosticsEngine &Diags) {
  if (SeenMissingHeader) {
    llvm::sys::fs::remove(OutputFile);
    return;
  }

  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputFile, EC, llvm::sys::fs::OF_TextWithCRLF);
  if (EC) {
    Diags.Report(diag::err_fe_error_opening) << OutputFile << EC.message();
    return;
  }

  outputDependencyFile(OS);
}

void DependencyFileGenerator::outputDependencyFile(llvm::raw_ostream &OS) {
  clang::printMakeDependencyFile(OS, Targets, getDependencies(), OutputFormat,
                                 PhonyTarget, InputFileIndex);
}
