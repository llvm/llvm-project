//===--- ModuleAssistant.cpp - Module map generation manager --*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the module generation entry point function,
// createModuleMap, a Module class for representing a module,
// and various implementation functions for doing the underlying
// work, described below.
//
// The "Module" class represents a module, with members for storing the module
// name, associated header file names, and sub-modules, and an "output"
// function that recursively writes the module definitions.
//
// The "createModuleMap" function implements the top-level logic of the
// assistant mode.  It calls a loadModuleDescriptions function to walk
// the header list passed to it and creates a tree of Module objects
// representing the module hierarchy, represented by a "Module" object,
// the "RootModule".  This root module may or may not represent an actual
// module in the module map, depending on the "--root-module" option passed
// to modularize.  It then calls a writeModuleMap function to set up the
// module map file output and walk the module tree, outputting the module
// map file using a stream obtained and managed by an
// llvm::ToolOutputFile object.
//
//===----------------------------------------------------------------------===//

#include "Modularize.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <vector>

// Local definitions:

namespace {

// Internal class definitions:

// Represents a module.
class Module {
public:
  Module(llvm::StringRef Name, bool Problem);
  ~Module();
  bool output(llvm::raw_fd_ostream &OS, int Indent);
  Module *findSubModule(llvm::StringRef SubName);

public:
  std::string Name;
  std::vector<std::string> HeaderFileNames;
  std::vector<Module *> SubModules;
  bool IsProblem;
};

} // end anonymous namespace.

// Module functions:

// Constructors.
Module::Module(llvm::StringRef Name, bool Problem)
  : Name(Name), IsProblem(Problem) {}

// Destructor.
Module::~Module() {
  // Free submodules.
  while (!SubModules.empty()) {
    Module *last = SubModules.back();
    SubModules.pop_back();
    delete last;
  }
}

// Write a module hierarchy to the given output stream.
bool Module::output(llvm::raw_fd_ostream &OS, int Indent) {
  // If this is not the nameless root module, start a module definition.
  if (Name.size() != 0) {
    OS.indent(Indent);
    OS << "module " << Name << " {\n";
    Indent += 2;
  }

  // Output submodules.
  for (auto I = SubModules.begin(), E = SubModules.end(); I != E; ++I) {
    if (!(*I)->output(OS, Indent))
      return false;
  }

  // Output header files.
  for (auto I = HeaderFileNames.begin(), E = HeaderFileNames.end(); I != E;
       ++I) {
    OS.indent(Indent);
    if (IsProblem || strstr((*I).c_str(), ".inl"))
      OS << "exclude header \"" << *I << "\"\n";
    else
      OS << "header \"" << *I << "\"\n";
  }

  // If this module has header files, output export directive.
  if (HeaderFileNames.size() != 0) {
    OS.indent(Indent);
    OS << "export *\n";
  }

  // If this is not the nameless root module, close the module definition.
  if (Name.size() != 0) {
    Indent -= 2;
    OS.indent(Indent);
    OS << "}\n";
  }

  return true;
}

// Lookup a sub-module.
Module *Module::findSubModule(llvm::StringRef SubName) {
  for (auto I = SubModules.begin(), E = SubModules.end(); I != E; ++I) {
    if ((*I)->Name == SubName)
      return *I;
  }
  return nullptr;
}

// Implementation functions:

// Reserved keywords in module.modulemap syntax.
// Keep in sync with keywords in module map parser in Lex/ModuleMap.cpp,
// such as in ModuleMapParser::consumeToken().
static const char *const ReservedNames[] = {
  "config_macros", "export",   "module", "conflict", "framework",
  "requires",      "exclude",  "header", "private",  "explicit",
  "link",          "umbrella", "extern", "use",      nullptr // Flag end.
};

// Convert module name to a non-keyword.
// Prepends a '_' to the name if and only if the name is a keyword.
static std::string
ensureNoCollisionWithReservedName(llvm::StringRef MightBeReservedName) {
  std::string SafeName = MightBeReservedName;
  for (int Index = 0; ReservedNames[Index] != nullptr; ++Index) {
    if (MightBeReservedName == ReservedNames[Index]) {
      SafeName.insert(0, "_");
      break;
    }
  }
  return SafeName;
}

// Convert module name to a non-keyword.
// Prepends a '_' to the name if and only if the name is a keyword.
static std::string
ensureVaidModuleName(llvm::StringRef MightBeInvalidName) {
  std::string SafeName = MightBeInvalidName;
  std::replace(SafeName.begin(), SafeName.end(), '-', '_');
  std::replace(SafeName.begin(), SafeName.end(), '.', '_');
  if (isdigit(SafeName[0]))
    SafeName = "_" + SafeName;
  return SafeName;
}

// Add one module, given a header file path.
static bool addModuleDescription(Module *RootModule,
                                 llvm::StringRef HeaderFilePath,
                                 llvm::StringRef HeaderPrefix,
                                 DependencyMap &Dependencies,
                                 bool IsProblemFile) {
  Module *CurrentModule = RootModule;
  DependentsVector &FileDependents = Dependencies[HeaderFilePath];
  std::string FilePath;
  // Strip prefix.
  // HeaderFilePath should be compared to natively-canonicalized Prefix.
  llvm::SmallString<256> NativePath, NativePrefix;
  llvm::sys::path::native(HeaderFilePath, NativePath);
  llvm::sys::path::native(HeaderPrefix, NativePrefix);
  if (NativePath.startswith(NativePrefix))
    FilePath = NativePath.substr(NativePrefix.size() + 1);
  else
    FilePath = HeaderFilePath;
  int Count = FileDependents.size();
  // Headers that go into modules must not depend on other files being
  // included first.  If there are any dependents, warn user and omit.
  if (Count != 0) {
    llvm::errs() << "warning: " << FilePath
                 << " depends on other headers being included first,"
                    " meaning the module.modulemap won't compile."
                    "  This header will be omitted from the module map.\n";
    return true;
  }
  // Make canonical.
  std::replace(FilePath.begin(), FilePath.end(), '\\', '/');
  // Insert module into tree, using subdirectories as submodules.
  for (llvm::sys::path::const_iterator I = llvm::sys::path::begin(FilePath),
                                       E = llvm::sys::path::end(FilePath);
       I != E; ++I) {
    if ((*I)[0] == '.')
      continue;
    std::string Stem = llvm::sys::path::stem(*I);
    Stem = ensureNoCollisionWithReservedName(Stem);
    Stem = ensureVaidModuleName(Stem);
    Module *SubModule = CurrentModule->findSubModule(Stem);
    if (!SubModule) {
      SubModule = new Module(Stem, IsProblemFile);
      CurrentModule->SubModules.push_back(SubModule);
    }
    CurrentModule = SubModule;
  }
  // Add header file name to headers.
  CurrentModule->HeaderFileNames.push_back(FilePath);
  return true;
}

// Create the internal module tree representation.
static Module *loadModuleDescriptions(
    llvm::StringRef RootModuleName, llvm::ArrayRef<std::string> HeaderFileNames,
    llvm::ArrayRef<std::string> ProblemFileNames,
    DependencyMap &Dependencies, llvm::StringRef HeaderPrefix) {

  // Create root module.
  auto *RootModule = new Module(RootModuleName, false);

  llvm::SmallString<256> CurrentDirectory;
  llvm::sys::fs::current_path(CurrentDirectory);

  // If no header prefix, use current directory.
  if (HeaderPrefix.size() == 0)
    HeaderPrefix = CurrentDirectory;

  // Walk the header file names and output the module map.
  for (llvm::ArrayRef<std::string>::iterator I = HeaderFileNames.begin(),
                                             E = HeaderFileNames.end();
       I != E; ++I) {
    std::string Header(*I);
    bool IsProblemFile = false;
    for (auto &ProblemFile : ProblemFileNames) {
      if (ProblemFile == Header) {
        IsProblemFile = true;
        break;
      }
    }
    // Add as a module.
    if (!addModuleDescription(RootModule, Header, HeaderPrefix, Dependencies, IsProblemFile))
      return nullptr;
  }

  return RootModule;
}

// Kick off the writing of the module map.
static bool writeModuleMap(llvm::StringRef ModuleMapPath,
                           llvm::StringRef HeaderPrefix, Module *RootModule) {
  llvm::SmallString<256> HeaderDirectory(ModuleMapPath);
  llvm::sys::path::remove_filename(HeaderDirectory);
  llvm::SmallString<256> FilePath;

  // Get the module map file path to be used.
  if ((HeaderDirectory.size() == 0) && (HeaderPrefix.size() != 0)) {
    FilePath = HeaderPrefix;
    // Prepend header file name prefix if it's not absolute.
    llvm::sys::path::append(FilePath, ModuleMapPath);
    llvm::sys::path::native(FilePath);
  } else {
    FilePath = ModuleMapPath;
    llvm::sys::path::native(FilePath);
  }

  // Set up module map output file.
  std::error_code EC;
  llvm::ToolOutputFile Out(FilePath, EC, llvm::sys::fs::F_Text);
  if (EC) {
    llvm::errs() << Argv0 << ": error opening " << FilePath << ":"
                 << EC.message() << "\n";
    return false;
  }

  // Get output stream from tool output buffer/manager.
  llvm::raw_fd_ostream &OS = Out.os();

  // Output file comment.
  OS << "// " << ModuleMapPath << "\n";
  OS << "// Generated by: " << CommandLine << "\n\n";

  // Write module hierarchy from internal representation.
  if (!RootModule->output(OS, 0))
    return false;

  // Tell ToolOutputFile that we want to keep the file.
  Out.keep();

  return true;
}

// Global functions:

// Module map generation entry point.
bool createModuleMap(llvm::StringRef ModuleMapPath,
                     llvm::ArrayRef<std::string> HeaderFileNames,
                     llvm::ArrayRef<std::string> ProblemFileNames,
                     DependencyMap &Dependencies, llvm::StringRef HeaderPrefix,
                     llvm::StringRef RootModuleName) {
  // Load internal representation of modules.
  std::unique_ptr<Module> RootModule(
    loadModuleDescriptions(
      RootModuleName, HeaderFileNames, ProblemFileNames, Dependencies,
      HeaderPrefix));
  if (!RootModule.get())
    return false;

  // Write module map file.
  return writeModuleMap(ModuleMapPath, HeaderPrefix, RootModule.get());
}
