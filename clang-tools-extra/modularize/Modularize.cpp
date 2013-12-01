//===- extra/modularize/Modularize.cpp - Check modularized headers --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool that checks whether a set of headers provides
// the consistent definitions required to use modules. For example, it detects
// whether the same entity (say, a NULL macro or size_t typedef) is defined in
// multiple headers or whether a header produces different definitions under
// different circumstances. These conditions cause modules built from the
// headers to behave poorly, and should be fixed before introducing a module
// map.
//
// Modularize takes as argument a file name for a file containing the
// newline-separated list of headers to check with respect to each other.
// Lines beginning with '#' and empty lines are ignored.
// Header file names followed by a colon and other space-separated
// file names will include those extra files as dependencies.
// The file names can be relative or full paths, but must be on the
// same line.
//
// Modularize also accepts regular front-end arguments.
//
// Usage:   modularize [-prefix (optional header path prefix)]
//   (include-files_list) [(front-end-options) ...]
//
// Note that unless a "-prefix (header path)" option is specified,
// non-absolute file paths in the header list file will be relative
// to the header list file directory.  Use -prefix to specify a different
// directory.
//
// Note that by default, the underlying Clang front end assumes .h files
// contain C source.  If your .h files in the file list contain C++ source,
// you should append the following to your command lines: -x c++
//
// Modularize will do normal parsing, reporting normal errors and warnings,
// but will also report special error messages like the following:
//
//   error: '(symbol)' defined at multiple locations:
//       (file):(row):(column)
//       (file):(row):(column)
//
//   error: header '(file)' has different contents depending on how it was
//     included
//
// The latter might be followed by messages like the following:
//
//   note: '(symbol)' in (file) at (row):(column) not always provided
//
// Checks will also be performed for macro expansions, defined(macro)
// expressions, and preprocessor conditional directives that evaluate
// inconsistently, and can produce error messages like the following:
//
//   (...)/SubHeader.h:11:5:
//   #if SYMBOL == 1
//       ^
//   error: Macro instance 'SYMBOL' has different values in this header,
//          depending on how it was included.
//     'SYMBOL' expanded to: '1' with respect to these inclusion paths:
//       (...)/Header1.h
//         (...)/SubHeader.h
//   (...)/SubHeader.h:3:9:
//   #define SYMBOL 1
//             ^
//   Macro defined here.
//     'SYMBOL' expanded to: '2' with respect to these inclusion paths:
//       (...)/Header2.h
//           (...)/SubHeader.h
//   (...)/SubHeader.h:7:9:
//   #define SYMBOL 2
//             ^
//   Macro defined here.
//
// Checks will also be performed for '#include' directives that are
// nested inside 'extern "C/C++" {}' or 'namespace (name) {}' blocks,
// and can produce error message like the following:
//
// IncludeInExtern.h:2:3
//   #include "Empty.h"
//   ^
// error: Include directive within extern "C" {}.
// IncludeInExtern.h:1:1
// extern "C" {
// ^
// The "extern "C" {}" block is here.
//
// See PreprocessorTracker.cpp for additional details.
//
// Modularize also has an option ("-module-map-path=module.map") that will
// skip the checks, and instead act as a module.map generation assistant,
// generating a module map file based on the header list.  An optional
// "-root-module=(rootName)" argument can specify a root module to be
// created in the generated module.map file.  Note that you will likely
// need to edit this file to suit the needs of your headers.
//
// An example command line for generating a module.map file:
//
//   modularize -module-map-path=module.map -root-module=myroot headerlist.txt
//
// Note that if the headers in the header list have partial paths, sub-modules
// will be created for the subdirectires involved, assuming that the
// subdirectories contain headers to be grouped into a module, but still with
// individual modules for the headers in the subdirectory.
//
// See the ModuleAssistant.cpp file comments for additional details about the
// implementation of the assistant mode.
//
// Future directions:
//
// Basically, we want to add new checks for whatever we can check with respect
// to checking headers for module'ability.
//
// Some ideas:
//
// 1. Omit duplicate "not always provided" messages
//
// 2. Add options to disable any of the checks, in case
// there is some problem with them, or the messages get too verbose.
//
// 3. Try to figure out the preprocessor conditional directives that
// contribute to problems and tie them to the inconsistent definitions.
//
// 4. There are some legitimate uses of preprocessor macros that
// modularize will flag as errors, such as repeatedly #include'ing
// a file and using interleaving defined/undefined macros
// to change declarations in the included file.  Is there a way
// to address this?  Maybe have modularize accept a list of macros
// to ignore.  Otherwise you can just exclude the file, after checking
// for legitimate errors.
//
// 5. What else?
//
// General clean-up and refactoring:
//
// 1. The Location class seems to be something that we might
// want to design to be applicable to a wider range of tools, and stick it
// somewhere into Tooling/ in mainline
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Config/config.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "Modularize.h"
#include "PreprocessorTracker.h"

using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace clang::tooling;
using namespace llvm;
using namespace llvm::opt;
using namespace Modularize;

// Option to specify a file name for a list of header files to check.
cl::opt<std::string>
ListFileName(cl::Positional,
             cl::desc("<name of file containing list of headers to check>"));

// Collect all other arguments, which will be passed to the front end.
cl::list<std::string>
CC1Arguments(cl::ConsumeAfter,
             cl::desc("<arguments to be passed to front end>..."));

// Option to specify a prefix to be prepended to the header names.
cl::opt<std::string> HeaderPrefix(
    "prefix", cl::init(""),
    cl::desc(
        "Prepend header file paths with this prefix."
        " If not specified,"
        " the files are considered to be relative to the header list file."));

// Option for assistant mode, telling modularize to output a module map
// based on the headers list, and where to put it.
cl::opt<std::string> ModuleMapPath(
    "module-map-path", cl::init(""),
    cl::desc("Turn on module map output and specify output path or file name."
             " If no path is specified and if prefix option is specified,"
             " use prefix for file path."));

// Option for assistant mode, telling modularize to output a module map
// based on the headers list, and where to put it.
cl::opt<std::string>
RootModule("root-module", cl::init(""),
           cl::desc("Specify the name of the root module."));

// Save the program name for error messages.
const char *Argv0;
// Save the command line for comments.
std::string CommandLine;

// Read the header list file and collect the header file names and
// optional dependencies.
error_code getHeaderFileNames(SmallVectorImpl<std::string> &HeaderFileNames,
                              DependencyMap &Dependencies,
                              StringRef ListFileName, StringRef HeaderPrefix) {
  // By default, use the path component of the list file name.
  SmallString<256> HeaderDirectory(ListFileName);
  sys::path::remove_filename(HeaderDirectory);
  SmallString<256> CurrentDirectory;
  sys::fs::current_path(CurrentDirectory);

  // Get the prefix if we have one.
  if (HeaderPrefix.size() != 0)
    HeaderDirectory = HeaderPrefix;

  // Read the header list file into a buffer.
  OwningPtr<MemoryBuffer> listBuffer;
  if (error_code ec = MemoryBuffer::getFile(ListFileName, listBuffer)) {
    return ec;
  }

  // Parse the header list into strings.
  SmallVector<StringRef, 32> Strings;
  listBuffer->getBuffer().split(Strings, "\n", -1, false);

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
    if (sys::path::is_absolute(TargetAndDependents.first))
      llvm::sys::path::native(TargetAndDependents.first, HeaderFileName);
    else {
      if (HeaderDirectory.size() != 0)
        HeaderFileName = HeaderDirectory;
      else
        HeaderFileName = CurrentDirectory;
      sys::path::append(HeaderFileName, TargetAndDependents.first);
      sys::path::native(HeaderFileName);
    }
    // Handle optional dependencies.
    DependentsVector Dependents;
    SmallVector<StringRef, 4> DependentsList;
    TargetAndDependents.second.split(DependentsList, " ", -1, false);
    int Count = DependentsList.size();
    for (int Index = 0; Index < Count; ++Index) {
      SmallString<256> Dependent;
      if (sys::path::is_absolute(DependentsList[Index]))
        Dependent = DependentsList[Index];
      else {
        if (HeaderDirectory.size() != 0)
          Dependent = HeaderDirectory;
        else
          Dependent = CurrentDirectory;
        sys::path::append(Dependent, DependentsList[Index]);
      }
      sys::path::native(Dependent);
      Dependents.push_back(Dependent.str());
    }
    // Save the resulting header file path and dependencies.
    HeaderFileNames.push_back(HeaderFileName.str());
    Dependencies[HeaderFileName.str()] = Dependents;
  }

  return error_code::success();
}

// Helper function for finding the input file in an arguments list.
std::string findInputFile(const CommandLineArguments &CLArgs) {
  OwningPtr<OptTable> Opts(createDriverOptTable());
  const unsigned IncludedFlagsBitmask = options::CC1Option;
  unsigned MissingArgIndex, MissingArgCount;
  SmallVector<const char *, 256> Argv;
  for (CommandLineArguments::const_iterator I = CLArgs.begin(),
                                            E = CLArgs.end();
       I != E; ++I)
    Argv.push_back(I->c_str());
  OwningPtr<InputArgList> Args(
      Opts->ParseArgs(Argv.data(), Argv.data() + Argv.size(), MissingArgIndex,
                      MissingArgCount, IncludedFlagsBitmask));
  std::vector<std::string> Inputs = Args->getAllArgValues(OPT_INPUT);
  return Inputs.back();
}

// We provide this derivation to add in "-include (file)" arguments for header
// dependencies.
class AddDependenciesAdjuster : public ArgumentsAdjuster {
public:
  AddDependenciesAdjuster(DependencyMap &Dependencies)
      : Dependencies(Dependencies) {}

private:
  // Callback for adjusting commandline arguments.
  CommandLineArguments Adjust(const CommandLineArguments &Args) {
    std::string InputFile = findInputFile(Args);
    DependentsVector &FileDependents = Dependencies[InputFile];
    int Count = FileDependents.size();
    if (Count == 0)
      return Args;
    CommandLineArguments NewArgs(Args);
    for (int Index = 0; Index < Count; ++Index) {
      NewArgs.push_back("-include");
      std::string File(std::string("\"") + FileDependents[Index] +
                       std::string("\""));
      NewArgs.push_back(FileDependents[Index]);
    }
    return NewArgs;
  }
  DependencyMap &Dependencies;
};

// FIXME: The Location class seems to be something that we might
// want to design to be applicable to a wider range of tools, and stick it
// somewhere into Tooling/ in mainline
struct Location {
  const FileEntry *File;
  unsigned Line, Column;

  Location() : File(), Line(), Column() {}

  Location(SourceManager &SM, SourceLocation Loc) : File(), Line(), Column() {
    Loc = SM.getExpansionLoc(Loc);
    if (Loc.isInvalid())
      return;

    std::pair<FileID, unsigned> Decomposed = SM.getDecomposedLoc(Loc);
    File = SM.getFileEntryForID(Decomposed.first);
    if (!File)
      return;

    Line = SM.getLineNumber(Decomposed.first, Decomposed.second);
    Column = SM.getColumnNumber(Decomposed.first, Decomposed.second);
  }

  operator bool() const { return File != 0; }

  friend bool operator==(const Location &X, const Location &Y) {
    return X.File == Y.File && X.Line == Y.Line && X.Column == Y.Column;
  }

  friend bool operator!=(const Location &X, const Location &Y) {
    return !(X == Y);
  }

  friend bool operator<(const Location &X, const Location &Y) {
    if (X.File != Y.File)
      return X.File < Y.File;
    if (X.Line != Y.Line)
      return X.Line < Y.Line;
    return X.Column < Y.Column;
  }
  friend bool operator>(const Location &X, const Location &Y) { return Y < X; }
  friend bool operator<=(const Location &X, const Location &Y) {
    return !(Y < X);
  }
  friend bool operator>=(const Location &X, const Location &Y) {
    return !(X < Y);
  }
};

struct Entry {
  enum EntryKind {
    EK_Tag,
    EK_Value,
    EK_Macro,

    EK_NumberOfKinds
  } Kind;

  Location Loc;

  StringRef getKindName() { return getKindName(Kind); }
  static StringRef getKindName(EntryKind kind);
};

// Return a string representing the given kind.
StringRef Entry::getKindName(Entry::EntryKind kind) {
  switch (kind) {
  case EK_Tag:
    return "tag";
  case EK_Value:
    return "value";
  case EK_Macro:
    return "macro";
  case EK_NumberOfKinds:
    break;
  }
  llvm_unreachable("invalid Entry kind");
}

struct HeaderEntry {
  std::string Name;
  Location Loc;

  friend bool operator==(const HeaderEntry &X, const HeaderEntry &Y) {
    return X.Loc == Y.Loc && X.Name == Y.Name;
  }
  friend bool operator!=(const HeaderEntry &X, const HeaderEntry &Y) {
    return !(X == Y);
  }
  friend bool operator<(const HeaderEntry &X, const HeaderEntry &Y) {
    return X.Loc < Y.Loc || (X.Loc == Y.Loc && X.Name < Y.Name);
  }
  friend bool operator>(const HeaderEntry &X, const HeaderEntry &Y) {
    return Y < X;
  }
  friend bool operator<=(const HeaderEntry &X, const HeaderEntry &Y) {
    return !(Y < X);
  }
  friend bool operator>=(const HeaderEntry &X, const HeaderEntry &Y) {
    return !(X < Y);
  }
};

typedef std::vector<HeaderEntry> HeaderContents;

class EntityMap : public StringMap<SmallVector<Entry, 2> > {
public:
  DenseMap<const FileEntry *, HeaderContents> HeaderContentMismatches;

  void add(const std::string &Name, enum Entry::EntryKind Kind, Location Loc) {
    // Record this entity in its header.
    HeaderEntry HE = { Name, Loc };
    CurHeaderContents[Loc.File].push_back(HE);

    // Check whether we've seen this entry before.
    SmallVector<Entry, 2> &Entries = (*this)[Name];
    for (unsigned I = 0, N = Entries.size(); I != N; ++I) {
      if (Entries[I].Kind == Kind && Entries[I].Loc == Loc)
        return;
    }

    // We have not seen this entry before; record it.
    Entry E = { Kind, Loc };
    Entries.push_back(E);
  }

  void mergeCurHeaderContents() {
    for (DenseMap<const FileEntry *, HeaderContents>::iterator
             H = CurHeaderContents.begin(),
             HEnd = CurHeaderContents.end();
         H != HEnd; ++H) {
      // Sort contents.
      std::sort(H->second.begin(), H->second.end());

      // Check whether we've seen this header before.
      DenseMap<const FileEntry *, HeaderContents>::iterator KnownH =
          AllHeaderContents.find(H->first);
      if (KnownH == AllHeaderContents.end()) {
        // We haven't seen this header before; record its contents.
        AllHeaderContents.insert(*H);
        continue;
      }

      // If the header contents are the same, we're done.
      if (H->second == KnownH->second)
        continue;

      // Determine what changed.
      std::set_symmetric_difference(
          H->second.begin(), H->second.end(), KnownH->second.begin(),
          KnownH->second.end(),
          std::back_inserter(HeaderContentMismatches[H->first]));
    }

    CurHeaderContents.clear();
  }

private:
  DenseMap<const FileEntry *, HeaderContents> CurHeaderContents;
  DenseMap<const FileEntry *, HeaderContents> AllHeaderContents;
};

class CollectEntitiesVisitor
    : public RecursiveASTVisitor<CollectEntitiesVisitor> {
public:
  CollectEntitiesVisitor(SourceManager &SM, EntityMap &Entities,
                         Preprocessor &PP, PreprocessorTracker &PPTracker,
                         int &HadErrors)
      : SM(SM), Entities(Entities), PP(PP), PPTracker(PPTracker),
        HadErrors(HadErrors) {}

  bool TraverseStmt(Stmt *S) { return true; }
  bool TraverseType(QualType T) { return true; }
  bool TraverseTypeLoc(TypeLoc TL) { return true; }
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *NNS) { return true; }
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) {
    return true;
  }
  bool TraverseDeclarationNameInfo(DeclarationNameInfo NameInfo) {
    return true;
  }
  bool TraverseTemplateName(TemplateName Template) { return true; }
  bool TraverseTemplateArgument(const TemplateArgument &Arg) { return true; }
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc) {
    return true;
  }
  bool TraverseTemplateArguments(const TemplateArgument *Args,
                                 unsigned NumArgs) {
    return true;
  }
  bool TraverseConstructorInitializer(CXXCtorInitializer *Init) { return true; }
  bool TraverseLambdaCapture(LambdaExpr::Capture C) { return true; }

  // Check 'extern "*" {}' block for #include directives.
  bool VisitLinkageSpecDecl(LinkageSpecDecl *D) {
    // Bail if not a block.
    if (!D->hasBraces())
      return true;
    SourceRange BlockRange = D->getSourceRange();
    const char *LinkageLabel;
    switch (D->getLanguage()) {
    case LinkageSpecDecl::lang_c:
      LinkageLabel = "extern \"C\" {}";
      break;
    case LinkageSpecDecl::lang_cxx:
      LinkageLabel = "extern \"C++\" {}";
      break;
    }
    if (!PPTracker.checkForIncludesInBlock(PP, BlockRange, LinkageLabel,
                                           errs()))
      HadErrors = 1;
    return true;
  }

  // Check 'namespace (name) {}' block for #include directives.
  bool VisitNamespaceDecl(const NamespaceDecl *D) {
    SourceRange BlockRange = D->getSourceRange();
    std::string Label("namespace ");
    Label += D->getName();
    Label += " {}";
    if (!PPTracker.checkForIncludesInBlock(PP, BlockRange, Label.c_str(),
                                           errs()))
      HadErrors = 1;
    return true;
  }

  // Collect definition entities.
  bool VisitNamedDecl(NamedDecl *ND) {
    // We only care about file-context variables.
    if (!ND->getDeclContext()->isFileContext())
      return true;

    // Skip declarations that tend to be properly multiply-declared.
    if (isa<NamespaceDecl>(ND) || isa<UsingDirectiveDecl>(ND) ||
        isa<NamespaceAliasDecl>(ND) ||
        isa<ClassTemplateSpecializationDecl>(ND) || isa<UsingDecl>(ND) ||
        isa<ClassTemplateDecl>(ND) || isa<TemplateTypeParmDecl>(ND) ||
        isa<TypeAliasTemplateDecl>(ND) || isa<UsingShadowDecl>(ND) ||
        isa<FunctionDecl>(ND) || isa<FunctionTemplateDecl>(ND) ||
        (isa<TagDecl>(ND) &&
         !cast<TagDecl>(ND)->isThisDeclarationADefinition()))
      return true;

    // Skip anonymous declarations.
    if (!ND->getDeclName())
      return true;

    // Get the qualified name.
    std::string Name;
    llvm::raw_string_ostream OS(Name);
    ND->printQualifiedName(OS);
    OS.flush();
    if (Name.empty())
      return true;

    Location Loc(SM, ND->getLocation());
    if (!Loc)
      return true;

    Entities.add(Name, isa<TagDecl>(ND) ? Entry::EK_Tag : Entry::EK_Value, Loc);
    return true;
  }

private:
  SourceManager &SM;
  EntityMap &Entities;
  Preprocessor &PP;
  PreprocessorTracker &PPTracker;
  int &HadErrors;
};

class CollectEntitiesConsumer : public ASTConsumer {
public:
  CollectEntitiesConsumer(EntityMap &Entities,
                          PreprocessorTracker &preprocessorTracker,
                          Preprocessor &PP, StringRef InFile, int &HadErrors)
      : Entities(Entities), PPTracker(preprocessorTracker), PP(PP),
        HadErrors(HadErrors) {
    PPTracker.handlePreprocessorEntry(PP, InFile);
  }

  ~CollectEntitiesConsumer() { PPTracker.handlePreprocessorExit(); }

  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    SourceManager &SM = Ctx.getSourceManager();

    // Collect declared entities.
    CollectEntitiesVisitor(SM, Entities, PP, PPTracker, HadErrors)
        .TraverseDecl(Ctx.getTranslationUnitDecl());

    // Collect macro definitions.
    for (Preprocessor::macro_iterator M = PP.macro_begin(),
                                      MEnd = PP.macro_end();
         M != MEnd; ++M) {
      Location Loc(SM, M->second->getLocation());
      if (!Loc)
        continue;

      Entities.add(M->first->getName().str(), Entry::EK_Macro, Loc);
    }

    // Merge header contents.
    Entities.mergeCurHeaderContents();
  }

private:
  EntityMap &Entities;
  PreprocessorTracker &PPTracker;
  Preprocessor &PP;
  int &HadErrors;
};

class CollectEntitiesAction : public SyntaxOnlyAction {
public:
  CollectEntitiesAction(EntityMap &Entities,
                        PreprocessorTracker &preprocessorTracker,
                        int &HadErrors)
      : Entities(Entities), PPTracker(preprocessorTracker),
        HadErrors(HadErrors) {}

protected:
  virtual clang::ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                                StringRef InFile) {
    return new CollectEntitiesConsumer(Entities, PPTracker,
                                       CI.getPreprocessor(), InFile, HadErrors);
  }

private:
  EntityMap &Entities;
  PreprocessorTracker &PPTracker;
  int &HadErrors;
};

class ModularizeFrontendActionFactory : public FrontendActionFactory {
public:
  ModularizeFrontendActionFactory(EntityMap &Entities,
                                  PreprocessorTracker &preprocessorTracker,
                                  int &HadErrors)
      : Entities(Entities), PPTracker(preprocessorTracker),
        HadErrors(HadErrors) {}

  virtual CollectEntitiesAction *create() {
    return new CollectEntitiesAction(Entities, PPTracker, HadErrors);
  }

private:
  EntityMap &Entities;
  PreprocessorTracker &PPTracker;
  int &HadErrors;
};

int main(int Argc, const char **Argv) {

  // Save program name for error messages.
  Argv0 = Argv[0];

  // Save program arguments for use in module.map comment.
  CommandLine = sys::path::stem(sys::path::filename(Argv0));
  for (int ArgIndex = 1; ArgIndex < Argc; ArgIndex++) {
    CommandLine.append(" ");
    CommandLine.append(Argv[ArgIndex]);
  }

  // This causes options to be parsed.
  cl::ParseCommandLineOptions(Argc, Argv, "modularize.\n");

  // No go if we have no header list file.
  if (ListFileName.size() == 0) {
    cl::PrintHelpMessage();
    return 1;
  }

  // Get header file names and dependencies.
  SmallVector<std::string, 32> Headers;
  DependencyMap Dependencies;
  if (error_code EC = getHeaderFileNames(Headers, Dependencies, ListFileName,
                                         HeaderPrefix)) {
    errs() << Argv[0] << ": error: Unable to get header list '" << ListFileName
           << "': " << EC.message() << '\n';
    return 1;
  }

  // If we are in assistant mode, output the module map and quit.
  if (ModuleMapPath[0]) {
    if (!createModuleMap(ModuleMapPath, Headers, Dependencies, HeaderPrefix,
                         RootModule))
      return 1; // Failed.
    return 0;   // Success - Skip checks in assistant mode.
  }

  // Create the compilation database.
  SmallString<256> PathBuf;
  sys::fs::current_path(PathBuf);
  OwningPtr<CompilationDatabase> Compilations;
  Compilations.reset(
      new FixedCompilationDatabase(Twine(PathBuf), CC1Arguments));

  // Create preprocessor tracker, to watch for macro and conditional problems.
  OwningPtr<PreprocessorTracker> PPTracker(PreprocessorTracker::create());

  // Parse all of the headers, detecting duplicates.
  EntityMap Entities;
  ClangTool Tool(*Compilations, Headers);
  Tool.appendArgumentsAdjuster(new AddDependenciesAdjuster(Dependencies));
  int HadErrors = 0;
  HadErrors |= Tool.run(
      new ModularizeFrontendActionFactory(Entities, *PPTracker, HadErrors));

  // Create a place to save duplicate entity locations, separate bins per kind.
  typedef SmallVector<Location, 8> LocationArray;
  typedef SmallVector<LocationArray, Entry::EK_NumberOfKinds> EntryBinArray;
  EntryBinArray EntryBins;
  int KindIndex;
  for (KindIndex = 0; KindIndex < Entry::EK_NumberOfKinds; ++KindIndex) {
    LocationArray Array;
    EntryBins.push_back(Array);
  }

  // Check for the same entity being defined in multiple places.
  for (EntityMap::iterator E = Entities.begin(), EEnd = Entities.end();
       E != EEnd; ++E) {
    // If only one occurrence, exit early.
    if (E->second.size() == 1)
      continue;
    // Clear entity locations.
    for (EntryBinArray::iterator CI = EntryBins.begin(), CE = EntryBins.end();
         CI != CE; ++CI) {
      CI->clear();
    }
    // Walk the entities of a single name, collecting the locations,
    // separated into separate bins.
    for (unsigned I = 0, N = E->second.size(); I != N; ++I) {
      EntryBins[E->second[I].Kind].push_back(E->second[I].Loc);
    }
    // Report any duplicate entity definition errors.
    int KindIndex = 0;
    for (EntryBinArray::iterator DI = EntryBins.begin(), DE = EntryBins.end();
         DI != DE; ++DI, ++KindIndex) {
      int ECount = DI->size();
      // If only 1 occurrence, skip;
      if (ECount <= 1)
        continue;
      LocationArray::iterator FI = DI->begin();
      StringRef kindName = Entry::getKindName((Entry::EntryKind)KindIndex);
      errs() << "error: " << kindName << " '" << E->first()
             << "' defined at multiple locations:\n";
      for (LocationArray::iterator FE = DI->end(); FI != FE; ++FI) {
        errs() << "    " << FI->File->getName() << ":" << FI->Line << ":"
               << FI->Column << "\n";
      }
      HadErrors = 1;
    }
  }

  // Complain about macro instance in header files that differ based on how
  // they are included.
  if (PPTracker->reportInconsistentMacros(errs()))
    HadErrors = 1;

  // Complain about preprocessor conditional directives in header files that
  // differ based on how they are included.
  if (PPTracker->reportInconsistentConditionals(errs()))
    HadErrors = 1;

  // Complain about any headers that have contents that differ based on how
  // they are included.
  // FIXME: Could we provide information about which preprocessor conditionals
  // are involved?
  for (DenseMap<const FileEntry *, HeaderContents>::iterator
           H = Entities.HeaderContentMismatches.begin(),
           HEnd = Entities.HeaderContentMismatches.end();
       H != HEnd; ++H) {
    if (H->second.empty()) {
      errs() << "internal error: phantom header content mismatch\n";
      continue;
    }

    HadErrors = 1;
    errs() << "error: header '" << H->first->getName()
           << "' has different contents depending on how it was included.\n";
    for (unsigned I = 0, N = H->second.size(); I != N; ++I) {
      errs() << "note: '" << H->second[I].Name << "' in "
             << H->second[I].Loc.File->getName() << " at "
             << H->second[I].Loc.Line << ":" << H->second[I].Loc.Column
             << " not always provided\n";
    }
  }

  return HadErrors;
}
