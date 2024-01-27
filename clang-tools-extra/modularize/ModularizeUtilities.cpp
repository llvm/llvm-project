//===--- extra/modularize/ModularizeUtilities.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a class for loading and validating a module map or
// header list by checking that all headers in the corresponding directories
// are accounted for.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "CoverageChecker.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "ModularizeUtilities.h"

using namespace clang;
using namespace llvm;
using namespace Modularize;

namespace {
// Subclass TargetOptions so we can construct it inline with
// the minimal option, the triple.
class ModuleMapTargetOptions : public clang::TargetOptions {
public:
  ModuleMapTargetOptions() { Triple = llvm::sys::getDefaultTargetTriple(); }
};
} // namespace

// ModularizeUtilities class implementation.

// Constructor.
ModularizeUtilities::ModularizeUtilities(std::vector<std::string> &InputPaths,
                                         llvm::StringRef Prefix,
                                         llvm::StringRef ProblemFilesListPath)
    : InputFilePaths(InputPaths), HeaderPrefix(Prefix),
      ProblemFilesPath(ProblemFilesListPath), HasModuleMap(false),
      MissingHeaderCount(0),
      // Init clang stuff needed for loading the module map and preprocessing.
      LangOpts(new LangOptions()), DiagIDs(new DiagnosticIDs()),
      DiagnosticOpts(new DiagnosticOptions()),
      DC(llvm::errs(), DiagnosticOpts.get()),
      Diagnostics(
          new DiagnosticsEngine(DiagIDs, DiagnosticOpts.get(), &DC, false)),
      TargetOpts(new ModuleMapTargetOptions()),
      Target(TargetInfo::CreateTargetInfo(*Diagnostics, TargetOpts)),
      FileMgr(new FileManager(FileSystemOpts)),
      SourceMgr(new SourceManager(*Diagnostics, *FileMgr, false)),
      HeaderInfo(new HeaderSearch(std::make_shared<HeaderSearchOptions>(),
                                  *SourceMgr, *Diagnostics, *LangOpts,
                                  Target.get())) {}

// Create instance of ModularizeUtilities, to simplify setting up
// subordinate objects.
ModularizeUtilities *ModularizeUtilities::createModularizeUtilities(
    std::vector<std::string> &InputPaths, llvm::StringRef Prefix,
    llvm::StringRef ProblemFilesListPath) {

  return new ModularizeUtilities(InputPaths, Prefix, ProblemFilesListPath);
}

// Load all header lists and dependencies.
std::error_code ModularizeUtilities::loadAllHeaderListsAndDependencies() {
  // For each input file.
  for (auto I = InputFilePaths.begin(), E = InputFilePaths.end(); I != E; ++I) {
    llvm::StringRef InputPath = *I;
    // If it's a module map.
    if (InputPath.ends_with(".modulemap")) {
      // Load the module map.
      if (std::error_code EC = loadModuleMap(InputPath))
        return EC;
    } else {
      // Else we assume it's a header list and load it.
      if (std::error_code EC = loadSingleHeaderListsAndDependencies(InputPath)) {
        errs() << "modularize: error: Unable to get header list '" << InputPath
          << "': " << EC.message() << '\n';
        return EC;
      }
    }
  }
  // If we have a problem files list.
  if (ProblemFilesPath.size() != 0) {
    // Load problem files list.
    if (std::error_code EC = loadProblemHeaderList(ProblemFilesPath)) {
      errs() << "modularize: error: Unable to get problem header list '" << ProblemFilesPath
        << "': " << EC.message() << '\n';
      return EC;
    }
  }
  return std::error_code();
}

// Do coverage checks.
// For each loaded module map, do header coverage check.
// Starting from the directory of the module.modulemap file,
// Find all header files, optionally looking only at files
// covered by the include path options, and compare against
// the headers referenced by the module.modulemap file.
// Display warnings for unaccounted-for header files.
// Returns 0 if there were no errors or warnings, 1 if there
// were warnings, 2 if any other problem, such as a bad
// module map path argument was specified.
std::error_code ModularizeUtilities::doCoverageCheck(
    std::vector<std::string> &IncludePaths,
    llvm::ArrayRef<std::string> CommandLine) {
  int ModuleMapCount = ModuleMaps.size();
  int ModuleMapIndex;
  std::error_code EC;
  for (ModuleMapIndex = 0; ModuleMapIndex < ModuleMapCount; ++ModuleMapIndex) {
    std::unique_ptr<clang::ModuleMap> &ModMap = ModuleMaps[ModuleMapIndex];
    auto Checker = CoverageChecker::createCoverageChecker(
        InputFilePaths[ModuleMapIndex], IncludePaths, CommandLine,
        ModMap.get());
    std::error_code LocalEC = Checker->doChecks();
    if (LocalEC.value() > 0)
      EC = LocalEC;
  }
  return EC;
}

// Load single header list and dependencies.
std::error_code ModularizeUtilities::loadSingleHeaderListsAndDependencies(
    llvm::StringRef InputPath) {

  // By default, use the path component of the list file name.
  SmallString<256> HeaderDirectory(InputPath);
  llvm::sys::path::remove_filename(HeaderDirectory);
  SmallString<256> CurrentDirectory;
  llvm::sys::fs::current_path(CurrentDirectory);

  // Get the prefix if we have one.
  if (HeaderPrefix.size() != 0)
    HeaderDirectory = HeaderPrefix;

  // Read the header list file into a buffer.
  ErrorOr<std::unique_ptr<MemoryBuffer>> listBuffer =
    MemoryBuffer::getFile(InputPath);
  if (std::error_code EC = listBuffer.getError())
    return EC;

  // Parse the header list into strings.
  SmallVector<StringRef, 32> Strings;
  listBuffer.get()->getBuffer().split(Strings, "\n", -1, false);

  // Collect the header file names from the string list.
  for (SmallVectorImpl<StringRef>::iterator I = Strings.begin(),
    E = Strings.end();
    I != E; ++I) {
    StringRef Line = I->trim();
    // Ignore comments and empty lines.
    if (Line.empty() || (Line[0] == '#'))
      continue;
    std::pair<StringRef, StringRef> TargetAndDependents = Line.split(':');
    SmallString<256> HeaderFileName;
    // Prepend header file name prefix if it's not absolute.
    if (llvm::sys::path::is_absolute(TargetAndDependents.first))
      llvm::sys::path::native(TargetAndDependents.first, HeaderFileName);
    else {
      if (HeaderDirectory.size() != 0)
        HeaderFileName = HeaderDirectory;
      else
        HeaderFileName = CurrentDirectory;
      llvm::sys::path::append(HeaderFileName, TargetAndDependents.first);
      llvm::sys::path::native(HeaderFileName);
    }
    // Handle optional dependencies.
    DependentsVector Dependents;
    SmallVector<StringRef, 4> DependentsList;
    TargetAndDependents.second.split(DependentsList, " ", -1, false);
    int Count = DependentsList.size();
    for (int Index = 0; Index < Count; ++Index) {
      SmallString<256> Dependent;
      if (llvm::sys::path::is_absolute(DependentsList[Index]))
        Dependent = DependentsList[Index];
      else {
        if (HeaderDirectory.size() != 0)
          Dependent = HeaderDirectory;
        else
          Dependent = CurrentDirectory;
        llvm::sys::path::append(Dependent, DependentsList[Index]);
      }
      llvm::sys::path::native(Dependent);
      Dependents.push_back(getCanonicalPath(Dependent.str()));
    }
    // Get canonical form.
    HeaderFileName = getCanonicalPath(HeaderFileName);
    // Save the resulting header file path and dependencies.
    HeaderFileNames.push_back(std::string(HeaderFileName));
    Dependencies[HeaderFileName.str()] = Dependents;
  }
  return std::error_code();
}

// Load problem header list.
std::error_code ModularizeUtilities::loadProblemHeaderList(
  llvm::StringRef InputPath) {

  // By default, use the path component of the list file name.
  SmallString<256> HeaderDirectory(InputPath);
  llvm::sys::path::remove_filename(HeaderDirectory);
  SmallString<256> CurrentDirectory;
  llvm::sys::fs::current_path(CurrentDirectory);

  // Get the prefix if we have one.
  if (HeaderPrefix.size() != 0)
    HeaderDirectory = HeaderPrefix;

  // Read the header list file into a buffer.
  ErrorOr<std::unique_ptr<MemoryBuffer>> listBuffer =
    MemoryBuffer::getFile(InputPath);
  if (std::error_code EC = listBuffer.getError())
    return EC;

  // Parse the header list into strings.
  SmallVector<StringRef, 32> Strings;
  listBuffer.get()->getBuffer().split(Strings, "\n", -1, false);

  // Collect the header file names from the string list.
  for (SmallVectorImpl<StringRef>::iterator I = Strings.begin(),
    E = Strings.end();
    I != E; ++I) {
    StringRef Line = I->trim();
    // Ignore comments and empty lines.
    if (Line.empty() || (Line[0] == '#'))
      continue;
    SmallString<256> HeaderFileName;
    // Prepend header file name prefix if it's not absolute.
    if (llvm::sys::path::is_absolute(Line))
      llvm::sys::path::native(Line, HeaderFileName);
    else {
      if (HeaderDirectory.size() != 0)
        HeaderFileName = HeaderDirectory;
      else
        HeaderFileName = CurrentDirectory;
      llvm::sys::path::append(HeaderFileName, Line);
      llvm::sys::path::native(HeaderFileName);
    }
    // Get canonical form.
    HeaderFileName = getCanonicalPath(HeaderFileName);
    // Save the resulting header file path.
    ProblemFileNames.push_back(std::string(HeaderFileName));
  }
  return std::error_code();
}

// Load single module map and extract header file list.
std::error_code ModularizeUtilities::loadModuleMap(
    llvm::StringRef InputPath) {
  // Get file entry for module.modulemap file.
  auto ModuleMapEntryOrErr = SourceMgr->getFileManager().getFileRef(InputPath);

  // return error if not found.
  if (!ModuleMapEntryOrErr) {
    llvm::errs() << "error: File \"" << InputPath << "\" not found.\n";
    return errorToErrorCode(ModuleMapEntryOrErr.takeError());
  }
  FileEntryRef ModuleMapEntry = *ModuleMapEntryOrErr;

  // Because the module map parser uses a ForwardingDiagnosticConsumer,
  // which doesn't forward the BeginSourceFile call, we do it explicitly here.
  DC.BeginSourceFile(*LangOpts, nullptr);

  // Figure out the home directory for the module map file.
  DirectoryEntryRef Dir = ModuleMapEntry.getDir();
  StringRef DirName(Dir.getName());
  if (llvm::sys::path::filename(DirName) == "Modules") {
    DirName = llvm::sys::path::parent_path(DirName);
    if (DirName.ends_with(".framework")) {
      auto FrameworkDirOrErr = FileMgr->getDirectoryRef(DirName);
      if (!FrameworkDirOrErr) {
        // This can happen if there's a race between the above check and the
        // removal of the directory.
        return errorToErrorCode(FrameworkDirOrErr.takeError());
      }
      Dir = *FrameworkDirOrErr;
    }
  }

  std::unique_ptr<ModuleMap> ModMap;
  ModMap.reset(new ModuleMap(*SourceMgr, *Diagnostics, *LangOpts,
    Target.get(), *HeaderInfo));

  // Parse module.modulemap file into module map.
  if (ModMap->parseModuleMapFile(ModuleMapEntry, false, Dir)) {
    return std::error_code(1, std::generic_category());
  }

  // Do matching end call.
  DC.EndSourceFile();

  // Reset missing header count.
  MissingHeaderCount = 0;

  if (!collectModuleMapHeaders(ModMap.get()))
    return std::error_code(1, std::generic_category());

  // Save module map.
  ModuleMaps.push_back(std::move(ModMap));

  // Indicate we are using module maps.
  HasModuleMap = true;

  // Return code of 1 for missing headers.
  if (MissingHeaderCount)
    return std::error_code(1, std::generic_category());

  return std::error_code();
}

// Collect module map headers.
// Walks the modules and collects referenced headers into
// HeaderFileNames.
bool ModularizeUtilities::collectModuleMapHeaders(clang::ModuleMap *ModMap) {
  SmallVector<std::pair<StringRef, const clang::Module *>, 0> Vec;
  for (auto &M : ModMap->modules())
    Vec.emplace_back(M.first(), M.second);
  llvm::sort(Vec, llvm::less_first());
  for (auto &I : Vec)
    if (!collectModuleHeaders(*I.second))
      return false;
  return true;
}

// Collect referenced headers from one module.
// Collects the headers referenced in the given module into
// HeaderFileNames.
bool ModularizeUtilities::collectModuleHeaders(const clang::Module &Mod) {

  // Ignore explicit modules because they often have dependencies
  // we can't know.
  if (Mod.IsExplicit)
    return true;

  // Treat headers in umbrella directory as dependencies.
  DependentsVector UmbrellaDependents;

  // Recursively do submodules.
  for (auto *Submodule : Mod.submodules())
    collectModuleHeaders(*Submodule);

  if (std::optional<clang::Module::Header> UmbrellaHeader =
          Mod.getUmbrellaHeaderAsWritten()) {
    std::string HeaderPath = getCanonicalPath(UmbrellaHeader->Entry.getName());
    // Collect umbrella header.
    HeaderFileNames.push_back(HeaderPath);

    // FUTURE: When needed, umbrella header header collection goes here.
  } else if (std::optional<clang::Module::DirectoryName> UmbrellaDir =
                 Mod.getUmbrellaDirAsWritten()) {
    // If there normal headers, assume these are umbrellas and skip collection.
    if (Mod.Headers->size() == 0) {
      // Collect headers in umbrella directory.
      if (!collectUmbrellaHeaders(UmbrellaDir->Entry.getName(),
                                  UmbrellaDependents))
        return false;
    }
  }

  // We ignore HK_Private, HK_Textual, HK_PrivateTextual, and HK_Excluded,
  // assuming they are marked as such either because of unsuitability for
  // modules or because they are meant to be included by another header,
  // and thus should be ignored by modularize.

  int NormalHeaderCount = Mod.Headers[clang::Module::HK_Normal].size();

  for (int Index = 0; Index < NormalHeaderCount; ++Index) {
    DependentsVector NormalDependents;
    // Collect normal header.
    const clang::Module::Header &Header(
      Mod.Headers[clang::Module::HK_Normal][Index]);
    std::string HeaderPath = getCanonicalPath(Header.Entry.getName());
    HeaderFileNames.push_back(HeaderPath);
  }

  int MissingCountThisModule = Mod.MissingHeaders.size();

  for (int Index = 0; Index < MissingCountThisModule; ++Index) {
    std::string MissingFile = Mod.MissingHeaders[Index].FileName;
    SourceLocation Loc = Mod.MissingHeaders[Index].FileNameLoc;
    errs() << Loc.printToString(*SourceMgr)
      << ": error : Header not found: " << MissingFile << "\n";
  }

  MissingHeaderCount += MissingCountThisModule;

  return true;
}

// Collect headers from an umbrella directory.
bool ModularizeUtilities::collectUmbrellaHeaders(StringRef UmbrellaDirName,
  DependentsVector &Dependents) {
  // Initialize directory name.
  SmallString<256> Directory(UmbrellaDirName);
  // Walk the directory.
  std::error_code EC;
  for (llvm::sys::fs::directory_iterator I(Directory.str(), EC), E; I != E;
    I.increment(EC)) {
    if (EC)
      return false;
    std::string File(I->path());
    llvm::ErrorOr<llvm::sys::fs::basic_file_status> Status = I->status();
    if (!Status)
      return false;
    llvm::sys::fs::file_type Type = Status->type();
    // If the file is a directory, ignore the name and recurse.
    if (Type == llvm::sys::fs::file_type::directory_file) {
      if (!collectUmbrellaHeaders(File, Dependents))
        return false;
      continue;
    }
    // If the file does not have a common header extension, ignore it.
    if (!isHeader(File))
      continue;
    // Save header name.
    std::string HeaderPath = getCanonicalPath(File);
    Dependents.push_back(HeaderPath);
  }
  return true;
}

// Replace .. embedded in path for purposes of having
// a canonical path.
static std::string replaceDotDot(StringRef Path) {
  SmallString<128> Buffer;
  llvm::sys::path::const_iterator B = llvm::sys::path::begin(Path),
    E = llvm::sys::path::end(Path);
  while (B != E) {
    if (B->compare(".") == 0) {
    }
    else if (B->compare("..") == 0)
      llvm::sys::path::remove_filename(Buffer);
    else
      llvm::sys::path::append(Buffer, *B);
    ++B;
  }
  if (Path.ends_with("/") || Path.ends_with("\\"))
    Buffer.append(1, Path.back());
  return Buffer.c_str();
}

// Convert header path to canonical form.
// The canonical form is basically just use forward slashes, and remove "./".
// \param FilePath The file path, relative to the module map directory.
// \returns The file path in canonical form.
std::string ModularizeUtilities::getCanonicalPath(StringRef FilePath) {
  std::string Tmp(replaceDotDot(FilePath));
  std::replace(Tmp.begin(), Tmp.end(), '\\', '/');
  StringRef Tmp2(Tmp);
  if (Tmp2.starts_with("./"))
    Tmp = std::string(Tmp2.substr(2));
  return Tmp;
}

// Check for header file extension.
// If the file extension is .h, .inc, or missing, it's
// assumed to be a header.
// \param FileName The file name.  Must not be a directory.
// \returns true if it has a header extension or no extension.
bool ModularizeUtilities::isHeader(StringRef FileName) {
  StringRef Extension = llvm::sys::path::extension(FileName);
  if (Extension.size() == 0)
    return true;
  if (Extension.equals_insensitive(".h"))
    return true;
  if (Extension.equals_insensitive(".inc"))
    return true;
  return false;
}

// Get directory path component from file path.
// \returns the component of the given path, which will be
// relative if the given path is relative, absolute if the
// given path is absolute, or "." if the path has no leading
// path component.
std::string ModularizeUtilities::getDirectoryFromPath(StringRef Path) {
  SmallString<256> Directory(Path);
  sys::path::remove_filename(Directory);
  if (Directory.size() == 0)
    return ".";
  return std::string(Directory.str());
}

// Add unique problem file.
// Also standardizes the path.
void ModularizeUtilities::addUniqueProblemFile(std::string FilePath) {
  FilePath = getCanonicalPath(FilePath);
  // Don't add if already present.
  for(auto &TestFilePath : ProblemFileNames) {
    if (TestFilePath == FilePath)
      return;
  }
  ProblemFileNames.push_back(FilePath);
}

// Add file with no compile errors.
// Also standardizes the path.
void ModularizeUtilities::addNoCompileErrorsFile(std::string FilePath) {
  FilePath = getCanonicalPath(FilePath);
  GoodFileNames.push_back(FilePath);
}

// List problem files.
void ModularizeUtilities::displayProblemFiles() {
  errs() << "\nThese are the files with possible errors:\n\n";
  for (auto &ProblemFile : ProblemFileNames) {
    errs() << ProblemFile << "\n";
  }
}

// List files with no problems.
void ModularizeUtilities::displayGoodFiles() {
  errs() << "\nThese are the files with no detected errors:\n\n";
  for (auto &GoodFile : HeaderFileNames) {
    bool Good = true;
    for (auto &ProblemFile : ProblemFileNames) {
      if (ProblemFile == GoodFile) {
        Good = false;
        break;
      }
    }
    if (Good)
      errs() << GoodFile << "\n";
  }
}

// List files with problem files commented out.
void ModularizeUtilities::displayCombinedFiles() {
  errs() <<
    "\nThese are the combined files, with problem files preceded by #:\n\n";
  for (auto &File : HeaderFileNames) {
    bool Good = true;
    for (auto &ProblemFile : ProblemFileNames) {
      if (ProblemFile == File) {
        Good = false;
        break;
      }
    }
    errs() << (Good ? "" : "#") << File << "\n";
  }
}
