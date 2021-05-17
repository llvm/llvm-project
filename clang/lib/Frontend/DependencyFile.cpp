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

#include "clang/Frontend/Utils.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
struct DepCollectorPPCallbacks : public PPCallbacks {
  DependencyCollector &DepCollector;
  SourceManager &SM;
  DiagnosticsEngine &Diags;
  DepCollectorPPCallbacks(DependencyCollector &L, SourceManager &SM,
                          DiagnosticsEngine &Diags)
      : DepCollector(L), SM(SM), Diags(Diags) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    if (Reason != PPCallbacks::EnterFile)
      return;

    // Dependency generation really does want to go all the way to the
    // file entry for a source location to find out what is depended on.
    // We do not want #line markers to affect dependency generation!
    if (Optional<StringRef> Filename = SM.getNonBuiltinFilenameForID(
            SM.getFileID(SM.getExpansionLoc(Loc))))
      DepCollector.maybeAddDependency(
          llvm::sys::path::remove_leading_dotslash(*Filename),
          /*FromModule*/ false, isSystem(FileType), /*IsModuleFile*/ false,
          /*IsMissing*/ false);
  }

  void FileSkipped(const FileEntryRef &SkippedFile, const Token &FilenameTok,
                   SrcMgr::CharacteristicKind FileType) override {
    StringRef Filename =
        llvm::sys::path::remove_leading_dotslash(SkippedFile.getName());
    DepCollector.maybeAddDependency(Filename, /*FromModule=*/false,
                                    /*IsSystem=*/isSystem(FileType),
                                    /*IsModuleFile=*/false,
                                    /*IsMissing=*/false);
  }

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override {
    if (!File)
      DepCollector.maybeAddDependency(FileName, /*FromModule*/false,
                                     /*IsSystem*/false, /*IsModuleFile*/false,
                                     /*IsMissing*/true);
    // Files that actually exist are handled by FileChanged.
  }

  void HasInclude(SourceLocation Loc, StringRef SpelledFilename, bool IsAngled,
                  Optional<FileEntryRef> File,
                  SrcMgr::CharacteristicKind FileType) override {
    if (!File)
      return;
    StringRef Filename =
        llvm::sys::path::remove_leading_dotslash(File->getName());
    DepCollector.maybeAddDependency(Filename, /*FromModule=*/false,
                                    /*IsSystem=*/isSystem(FileType),
                                    /*IsModuleFile=*/false,
                                    /*IsMissing=*/false);
  }

  void EndOfMainFile() override { DepCollector.finishedMainFile(Diags); }
};

struct DepCollectorMMCallbacks : public ModuleMapCallbacks {
  DependencyCollector &DepCollector;
  DepCollectorMMCallbacks(DependencyCollector &DC) : DepCollector(DC) {}

  void moduleMapFileRead(SourceLocation Loc, const FileEntry &Entry,
                         bool IsSystem) override {
    StringRef Filename = Entry.getName();
    DepCollector.maybeAddDependency(Filename, /*FromModule*/false,
                                    /*IsSystem*/IsSystem,
                                    /*IsModuleFile*/false,
                                    /*IsMissing*/false);
  }
};

struct DepCollectorASTListener : public ASTReaderListener {
  DependencyCollector &DepCollector;
  DepCollectorASTListener(DependencyCollector &L) : DepCollector(L) { }
  bool needsInputFileVisitation() override { return true; }
  bool needsSystemInputFileVisitation() override {
    return DepCollector.needSystemDependencies();
  }
  void visitModuleFile(StringRef Filename,
                       serialization::ModuleKind Kind) override {
    DepCollector.maybeAddDependency(Filename, /*FromModule*/true,
                                   /*IsSystem*/false, /*IsModuleFile*/true,
                                   /*IsMissing*/false);
  }
  bool visitInputFile(StringRef Filename, bool IsSystem,
                      bool IsOverridden, bool IsExplicitModule) override {
    if (IsOverridden || IsExplicitModule)
      return true;

    DepCollector.maybeAddDependency(Filename, /*FromModule*/true, IsSystem,
                                   /*IsModuleFile*/false, /*IsMissing*/false);
    return true;
  }
};
} // end anonymous namespace

void DependencyCollector::maybeAddDependency(StringRef Filename,
                                             bool FromModule, bool IsSystem,
                                             bool IsModuleFile,
                                             bool IsMissing) {
  if (sawDependency(Filename, FromModule, IsSystem, IsModuleFile, IsMissing))
    addDependency(Filename);
}

bool DependencyCollector::addDependency(StringRef Filename) {
  if (Seen.insert(Filename).second) {
    Dependencies.push_back(std::string(Filename));
    return true;
  }
  return false;
}

static bool isSpecialFilename(StringRef Filename) {
  return llvm::StringSwitch<bool>(Filename)
      .Case("<built-in>", true)
      .Case("<stdin>", true)
      .Default(false);
}

bool DependencyCollector::sawDependency(StringRef Filename, bool FromModule,
                                        bool IsSystem, bool IsModuleFile,
                                        bool IsMissing) {
  return !isSpecialFilename(Filename) &&
         (needSystemDependencies() || !IsSystem);
}

DependencyCollector::~DependencyCollector() { }
void DependencyCollector::attachToPreprocessor(Preprocessor &PP) {
  PP.addPPCallbacks(std::make_unique<DepCollectorPPCallbacks>(
      *this, PP.getSourceManager(), PP.getDiagnostics()));
  PP.getHeaderSearchInfo().getModuleMap().addModuleMapCallbacks(
      std::make_unique<DepCollectorMMCallbacks>(*this));
}
void DependencyCollector::attachToASTReader(ASTReader &R) {
  R.addListener(std::make_unique<DepCollectorASTListener>(*this));
}

DependencyFileGenerator::DependencyFileGenerator(
    const DependencyOutputOptions &Opts)
    : OutputFile(Opts.OutputFile), Targets(Opts.Targets),
      IncludeSystemHeaders(Opts.IncludeSystemHeaders),
      PhonyTarget(Opts.UsePhonyTargets),
      AddMissingHeaderDeps(Opts.AddMissingHeaderDeps), SeenMissingHeader(false),
      IncludeModuleFiles(Opts.IncludeModuleFiles),
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
                                            bool IsMissing) {
  if (IsMissing) {
    // Handle the case of missing file from an inclusion directive.
    if (AddMissingHeaderDeps)
      return true;
    SeenMissingHeader = true;
    return false;
  }
  if (IsModuleFile && !IncludeModuleFiles)
    return false;

  if (isSpecialFilename(Filename))
    return false;

  if (IncludeSystemHeaders)
    return true;

  return !IsSystem;
}

void DependencyFileGenerator::finishedMainFile(DiagnosticsEngine &Diags) {
  outputDependencyFile(Diags);
}

/// Print the filename, with escaping or quoting that accommodates the three
/// most likely tools that use dependency files: GNU Make, BSD Make, and
/// NMake/Jom.
///
/// BSD Make is the simplest case: It does no escaping at all.  This means
/// characters that are normally delimiters, i.e. space and # (the comment
/// character) simply aren't supported in filenames.
///
/// GNU Make does allow space and # in filenames, but to avoid being treated
/// as a delimiter or comment, these must be escaped with a backslash. Because
/// backslash is itself the escape character, if a backslash appears in a
/// filename, it should be escaped as well.  (As a special case, $ is escaped
/// as $$, which is the normal Make way to handle the $ character.)
/// For compatibility with BSD Make and historical practice, if GNU Make
/// un-escapes characters in a filename but doesn't find a match, it will
/// retry with the unmodified original string.
///
/// GCC tries to accommodate both Make formats by escaping any space or #
/// characters in the original filename, but not escaping backslashes.  The
/// apparent intent is so that filenames with backslashes will be handled
/// correctly by BSD Make, and by GNU Make in its fallback mode of using the
/// unmodified original string; filenames with # or space characters aren't
/// supported by BSD Make at all, but will be handled correctly by GNU Make
/// due to the escaping.
///
/// A corner case that GCC gets only partly right is when the original filename
/// has a backslash immediately followed by space or #.  GNU Make would expect
/// this backslash to be escaped; however GCC escapes the original backslash
/// only when followed by space, not #.  It will therefore take a dependency
/// from a directive such as
///     #include "a\ b\#c.h"
/// and emit it as
///     a\\\ b\\#c.h
/// which GNU Make will interpret as
///     a\ b\
/// followed by a comment. Failing to find this file, it will fall back to the
/// original string, which probably doesn't exist either; in any case it won't
/// find
///     a\ b\#c.h
/// which is the actual filename specified by the include directive.
///
/// Clang does what GCC does, rather than what GNU Make expects.
///
/// NMake/Jom has a different set of scary characters, but wraps filespecs in
/// double-quotes to avoid misinterpreting them; see
/// https://msdn.microsoft.com/en-us/library/dd9y37ha.aspx for NMake info,
/// https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx
/// for Windows file-naming info.
static void PrintFilename(raw_ostream &OS, StringRef Filename,
                          DependencyOutputFormat OutputFormat) {
  // Convert filename to platform native path
  llvm::SmallString<256> NativePath;
  llvm::sys::path::native(Filename.str(), NativePath);

  if (OutputFormat == DependencyOutputFormat::NMake) {
    // Add quotes if needed. These are the characters listed as "special" to
    // NMake, that are legal in a Windows filespec, and that could cause
    // misinterpretation of the dependency string.
    if (NativePath.find_first_of(" #${}^!") != StringRef::npos)
      OS << '\"' << NativePath << '\"';
    else
      OS << NativePath;
    return;
  }
  assert(OutputFormat == DependencyOutputFormat::Make);
  for (unsigned i = 0, e = NativePath.size(); i != e; ++i) {
    if (NativePath[i] == '#') // Handle '#' the broken gcc way.
      OS << '\\';
    else if (NativePath[i] == ' ') { // Handle space correctly.
      OS << '\\';
      unsigned j = i;
      while (j > 0 && NativePath[--j] == '\\')
        OS << '\\';
    } else if (NativePath[i] == '$') // $ is escaped by $$.
      OS << '$';
    OS << NativePath[i];
  }
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
  // Write out the dependency targets, trying to avoid overly long
  // lines when possible. We try our best to emit exactly the same
  // dependency file as GCC (4.2), assuming the included files are the
  // same.
  const unsigned MaxColumns = 75;
  unsigned Columns = 0;

  for (StringRef Target : Targets) {
    unsigned N = Target.size();
    if (Columns == 0) {
      Columns += N;
    } else if (Columns + N + 2 > MaxColumns) {
      Columns = N + 2;
      OS << " \\\n  ";
    } else {
      Columns += N + 1;
      OS << ' ';
    }
    // Targets already quoted as needed.
    OS << Target;
  }

  OS << ':';
  Columns += 1;

  // Now add each dependency in the order it was seen, but avoiding
  // duplicates.
  ArrayRef<std::string> Files = getDependencies();
  for (StringRef File : Files) {
    // Start a new line if this would exceed the column limit. Make
    // sure to leave space for a trailing " \" in case we need to
    // break the line on the next iteration.
    unsigned N = File.size();
    if (Columns + (N + 1) + 2 > MaxColumns) {
      OS << " \\\n ";
      Columns = 2;
    }
    OS << ' ';
    PrintFilename(OS, File, OutputFormat);
    Columns += N + 1;
  }
  OS << '\n';

  // Create phony targets if requested.
  if (PhonyTarget && !Files.empty()) {
    unsigned Index = 0;
    for (auto I = Files.begin(), E = Files.end(); I != E; ++I) {
      if (Index++ == InputFileIndex)
        continue;
      OS << '\n';
      PrintFilename(OS, *I, OutputFormat);
      OS << ":\n";
    }
  }
}
