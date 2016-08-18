//===- extra/modularize/Modularize.cpp - Check modularized headers --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Introduction
//
// This file implements a tool that checks whether a set of headers provides
// the consistent definitions required to use modules.  It can also check an
// existing module map for full coverage of the headers in a directory tree.
//
// For example, in examining headers, it detects whether the same entity
// (say, a NULL macro or size_t typedef) is defined in multiple headers
// or whether a header produces different definitions under
// different circumstances. These conditions cause modules built from the
// headers to behave poorly, and should be fixed before introducing a module
// map.
//
// Modularize takes as input either one or more module maps (by default,
// "module.modulemap") or one or more text files contatining lists of headers
// to check.
//
// In the case of a module map, the module map must be well-formed in
// terms of syntax.  Modularize will extract the header file names
// from the map.  Only normal headers are checked, assuming headers
// marked "private", "textual", or "exclude" are not to be checked
// as a top-level include, assuming they either are included by
// other headers which are checked, or they are not suitable for
// modules.
//
// In the case of a file list, the list is a newline-separated list of headers
// to check with respect to each other.
// Lines beginning with '#' and empty lines are ignored.
// Header file names followed by a colon and other space-separated
// file names will include those extra files as dependencies.
// The file names can be relative or full paths, but must be on the
// same line.
//
// Modularize also accepts regular clang front-end arguments.
//
// Usage:   modularize [(modularize options)]
//   [(include-files_list)|(module map)]+ [(front-end-options) ...]
//
// Options:
//    -prefix=(optional header path prefix)
//          Note that unless a "-prefix (header path)" option is specified,
//          non-absolute file paths in the header list file will be relative
//          to the header list file directory.  Use -prefix to specify a
//          different directory.
//    -module-map-path=(module map)
//          Skip the checks, and instead act as a module.map generation
//          assistant, generating a module map file based on the header list.
//          An optional "-root-module=(rootName)" argument can specify a root
//          module to be created in the generated module.map file.  Note that
//          you will likely need to edit this file to suit the needs of your
//          headers.
//    -problem-files-list=(problem files list file name)
//          For use only with module map assistant.  Input list of files that
//          have problems with respect to modules.  These will still be
//          included in the generated module map, but will be marked as
//          "excluded" headers.
//    -root-module=(root module name)
//          Specifies a root module to be created in the generated module.map
//          file.
//    -block-check-header-list-only
//          Only warn if #include directives are inside extern or namespace
//          blocks if the included header is in the header list.
//    -no-coverage-check
//          Don't do the coverage check.
//    -coverage-check-only
//          Only do the coverage check.
//    -display-file-lists
//          Display lists of good files (no compile errors), problem files,
//          and a combined list with problem files preceded by a '#'.
//          This can be used to quickly determine which files have problems.
//          The latter combined list might be useful in starting to modularize
//          a set of headers.  You can start with a full list of headers,
//          use -display-file-lists option, and then use the combined list as
//          your intermediate list, uncommenting-out headers as you fix them.
//
// Note that by default, the modularize assumes .h files contain C++ source.
// If your .h files in the file list contain another language, you should
// append an appropriate -x option to your command line, i.e.:  -x c
//
// Modularization Issue Checks
//
// In the process of checking headers for modularization issues, modularize
// will do normal parsing, reporting normal errors and warnings,
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
// Module Map Coverage Check
//
// The coverage check uses the Clang ModuleMap class to read and parse the
// module map file.  Starting at the module map file directory, or just the
// include paths, if specified, it will collect the names of all the files it
// considers headers (no extension, .h, or .inc--if you need more, modify the
// isHeader function).  It then compares the headers against those referenced
// in the module map, either explicitly named, or implicitly named via an
// umbrella directory or umbrella file, as parsed by the ModuleMap object.
// If headers are found which are not referenced or covered by an umbrella
// directory or file, warning messages will be produced, and this program
// will return an error code of 1.  Other errors result in an error code of 2.
// If no problems are found, an error code of 0 is returned.
//
// Note that in the case of umbrella headers, this tool invokes the compiler
// to preprocess the file, and uses a callback to collect the header files
// included by the umbrella header or any of its nested includes.  If any
// front end options are needed for these compiler invocations, these
// can be included on the command line after the module map file argument.
//
// Warning message have the form:
//
//  warning: module.modulemap does not account for file: Level3A.h
//
// Note that for the case of the module map referencing a file that does
// not exist, the module map parser in Clang will (at the time of this
// writing) display an error message.
//
// Module Map Assistant - Module Map Generation
//
// Modularize also has an option ("-module-map-path=module.modulemap") that will
// skip the checks, and instead act as a module.modulemap generation assistant,
// generating a module map file based on the header list.  An optional
// "-root-module=(rootName)" argument can specify a root module to be
// created in the generated module.modulemap file.  Note that you will likely
// need to edit this file to suit the needs of your headers.
//
// An example command line for generating a module.modulemap file:
//
//   modularize -module-map-path=module.modulemap -root-module=myroot \
//      headerlist.txt
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

#include "Modularize.h"
#include "ModularizeUtilities.h"
#include "PreprocessorTracker.h"
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

using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace clang::tooling;
using namespace llvm;
using namespace llvm::opt;
using namespace Modularize;

// Option to specify a file name for a list of header files to check.
static cl::list<std::string>
    ListFileNames(cl::Positional, cl::value_desc("list"),
                  cl::desc("<list of one or more header list files>"),
                  cl::CommaSeparated);

// Collect all other arguments, which will be passed to the front end.
static cl::list<std::string>
    CC1Arguments(cl::ConsumeAfter,
                 cl::desc("<arguments to be passed to front end>..."));

// Option to specify a prefix to be prepended to the header names.
static cl::opt<std::string> HeaderPrefix(
    "prefix", cl::init(""),
    cl::desc(
        "Prepend header file paths with this prefix."
        " If not specified,"
        " the files are considered to be relative to the header list file."));

// Option for assistant mode, telling modularize to output a module map
// based on the headers list, and where to put it.
static cl::opt<std::string> ModuleMapPath(
    "module-map-path", cl::init(""),
    cl::desc("Turn on module map output and specify output path or file name."
             " If no path is specified and if prefix option is specified,"
             " use prefix for file path."));

// Option to specify list of problem files for assistant.
// This will cause assistant to exclude these files.
static cl::opt<std::string> ProblemFilesList(
  "problem-files-list", cl::init(""),
  cl::desc(
  "List of files with compilation or modularization problems for"
    " assistant mode.  This will be excluded."));

// Option for assistant mode, telling modularize the name of the root module.
static cl::opt<std::string>
RootModule("root-module", cl::init(""),
           cl::desc("Specify the name of the root module."));

// Option for limiting the #include-inside-extern-or-namespace-block
// check to only those headers explicitly listed in the header list.
// This is a work-around for private includes that purposefully get
// included inside blocks.
static cl::opt<bool>
BlockCheckHeaderListOnly("block-check-header-list-only", cl::init(false),
cl::desc("Only warn if #include directives are inside extern or namespace"
  " blocks if the included header is in the header list."));

// Option for include paths for coverage check.
static cl::list<std::string>
IncludePaths("I", cl::desc("Include path for coverage check."),
cl::ZeroOrMore, cl::value_desc("path"));

// Option for disabling the coverage check.
static cl::opt<bool>
NoCoverageCheck("no-coverage-check", cl::init(false),
cl::desc("Don't do the coverage check."));

// Option for just doing the coverage check.
static cl::opt<bool>
CoverageCheckOnly("coverage-check-only", cl::init(false),
cl::desc("Only do the coverage check."));

// Option for displaying lists of good, bad, and mixed files.
static cl::opt<bool>
DisplayFileLists("display-file-lists", cl::init(false),
cl::desc("Display lists of good files (no compile errors), problem files,"
  " and a combined list with problem files preceded by a '#'."));

// Save the program name for error messages.
const char *Argv0;
// Save the command line for comments.
std::string CommandLine;

// Helper function for finding the input file in an arguments list.
static std::string findInputFile(const CommandLineArguments &CLArgs) {
  std::unique_ptr<OptTable> Opts(createDriverOptTable());
  const unsigned IncludedFlagsBitmask = options::CC1Option;
  unsigned MissingArgIndex, MissingArgCount;
  SmallVector<const char *, 256> Argv;
  for (CommandLineArguments::const_iterator I = CLArgs.begin(),
                                            E = CLArgs.end();
       I != E; ++I)
    Argv.push_back(I->c_str());
  InputArgList Args = Opts->ParseArgs(Argv, MissingArgIndex, MissingArgCount,
                                      IncludedFlagsBitmask);
  std::vector<std::string> Inputs = Args.getAllArgValues(OPT_INPUT);
  return ModularizeUtilities::getCanonicalPath(Inputs.back());
}

// This arguments adjuster inserts "-include (file)" arguments for header
// dependencies.  It also inserts a "-w" option and a "-x c++",
// if no other "-x" option is present.
static ArgumentsAdjuster
getModularizeArgumentsAdjuster(DependencyMap &Dependencies) {
  return [&Dependencies](const CommandLineArguments &Args,
                         StringRef /*unused*/) {
    std::string InputFile = findInputFile(Args);
    DependentsVector &FileDependents = Dependencies[InputFile];
    CommandLineArguments NewArgs(Args);
    if (int Count = FileDependents.size()) {
      for (int Index = 0; Index < Count; ++Index) {
        NewArgs.push_back("-include");
        std::string File(std::string("\"") + FileDependents[Index] +
                         std::string("\""));
        NewArgs.push_back(FileDependents[Index]);
      }
    }
    // Ignore warnings.  (Insert after "clang_tool" at beginning.)
    NewArgs.insert(NewArgs.begin() + 1, "-w");
    // Since we are compiling .h files, assume C++ unless given a -x option.
    if (std::find(NewArgs.begin(), NewArgs.end(), "-x") == NewArgs.end()) {
      NewArgs.insert(NewArgs.begin() + 2, "-x");
      NewArgs.insert(NewArgs.begin() + 3, "c++");
    }
    return NewArgs;
  };
}

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

  operator bool() const { return File != nullptr; }

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
  bool TraverseLambdaCapture(LambdaExpr *LE, const LambdaCapture *C,
                             Expr *Init) {
    return true;
  }

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

  ~CollectEntitiesConsumer() override { PPTracker.handlePreprocessorExit(); }

  void HandleTranslationUnit(ASTContext &Ctx) override {
    SourceManager &SM = Ctx.getSourceManager();

    // Collect declared entities.
    CollectEntitiesVisitor(SM, Entities, PP, PPTracker, HadErrors)
        .TraverseDecl(Ctx.getTranslationUnitDecl());

    // Collect macro definitions.
    for (Preprocessor::macro_iterator M = PP.macro_begin(),
                                      MEnd = PP.macro_end();
         M != MEnd; ++M) {
      Location Loc(SM, M->second.getLatest()->getLocation());
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
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return llvm::make_unique<CollectEntitiesConsumer>(
        Entities, PPTracker, CI.getPreprocessor(), InFile, HadErrors);
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

  CollectEntitiesAction *create() override {
    return new CollectEntitiesAction(Entities, PPTracker, HadErrors);
  }

private:
  EntityMap &Entities;
  PreprocessorTracker &PPTracker;
  int &HadErrors;
};

class CompileCheckVisitor
  : public RecursiveASTVisitor<CompileCheckVisitor> {
public:
  CompileCheckVisitor() {}

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
  bool TraverseLambdaCapture(LambdaExpr *LE, const LambdaCapture *C,
                             Expr *Init) {
    return true;
  }

  // Check 'extern "*" {}' block for #include directives.
  bool VisitLinkageSpecDecl(LinkageSpecDecl *D) {
    return true;
  }

  // Check 'namespace (name) {}' block for #include directives.
  bool VisitNamespaceDecl(const NamespaceDecl *D) {
    return true;
  }

  // Collect definition entities.
  bool VisitNamedDecl(NamedDecl *ND) {
    return true;
  }
};

class CompileCheckConsumer : public ASTConsumer {
public:
  CompileCheckConsumer() {}

  void HandleTranslationUnit(ASTContext &Ctx) override {
    CompileCheckVisitor().TraverseDecl(Ctx.getTranslationUnitDecl());
  }
};

class CompileCheckAction : public SyntaxOnlyAction {
public:
  CompileCheckAction() {}

protected:
  std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return llvm::make_unique<CompileCheckConsumer>();
  }
};

class CompileCheckFrontendActionFactory : public FrontendActionFactory {
public:
  CompileCheckFrontendActionFactory() {}

  CompileCheckAction *create() override {
    return new CompileCheckAction();
  }
};

int main(int Argc, const char **Argv) {

  // Save program name for error messages.
  Argv0 = Argv[0];

  // Save program arguments for use in module.modulemap comment.
  CommandLine = sys::path::stem(sys::path::filename(Argv0));
  for (int ArgIndex = 1; ArgIndex < Argc; ArgIndex++) {
    CommandLine.append(" ");
    CommandLine.append(Argv[ArgIndex]);
  }

  // This causes options to be parsed.
  cl::ParseCommandLineOptions(Argc, Argv, "modularize.\n");

  // No go if we have no header list file.
  if (ListFileNames.size() == 0) {
    cl::PrintHelpMessage();
    return 1;
  }

  std::unique_ptr<ModularizeUtilities> ModUtil;
  int HadErrors = 0;

  ModUtil.reset(
    ModularizeUtilities::createModularizeUtilities(
      ListFileNames, HeaderPrefix, ProblemFilesList));

  // Get header file names and dependencies.
  if (ModUtil->loadAllHeaderListsAndDependencies())
    HadErrors = 1;

  // If we are in assistant mode, output the module map and quit.
  if (ModuleMapPath.length() != 0) {
    if (!createModuleMap(ModuleMapPath, ModUtil->HeaderFileNames,
                         ModUtil->ProblemFileNames,
                         ModUtil->Dependencies, HeaderPrefix, RootModule))
      return 1; // Failed.
    return 0;   // Success - Skip checks in assistant mode.
  }

  // If we're doing module maps.
  if (!NoCoverageCheck && ModUtil->HasModuleMap) {
    // Do coverage check.
    if (ModUtil->doCoverageCheck(IncludePaths, CommandLine))
      HadErrors = 1;
  }

  // Bail early if only doing the coverage check.
  if (CoverageCheckOnly)
    return HadErrors;

  // Create the compilation database.
  SmallString<256> PathBuf;
  sys::fs::current_path(PathBuf);
  std::unique_ptr<CompilationDatabase> Compilations;
  Compilations.reset(
      new FixedCompilationDatabase(Twine(PathBuf), CC1Arguments));

  // Create preprocessor tracker, to watch for macro and conditional problems.
  std::unique_ptr<PreprocessorTracker> PPTracker(
    PreprocessorTracker::create(ModUtil->HeaderFileNames,
                                BlockCheckHeaderListOnly));

  // Coolect entities here.
  EntityMap Entities;

  // Because we can't easily determine which files failed
  // during the tool run, if we're collecting the file lists
  // for display, we do a first compile pass on individual
  // files to find which ones don't compile stand-alone.
  if (DisplayFileLists) {
    // First, make a pass to just get compile errors.
    for (auto &CompileCheckFile : ModUtil->HeaderFileNames) {
      llvm::SmallVector<std::string, 32> CompileCheckFileArray;
      CompileCheckFileArray.push_back(CompileCheckFile);
      ClangTool CompileCheckTool(*Compilations, CompileCheckFileArray);
      CompileCheckTool.appendArgumentsAdjuster(
        getModularizeArgumentsAdjuster(ModUtil->Dependencies));
      int CompileCheckFileErrors = 0;
      CompileCheckFrontendActionFactory CompileCheckFactory;
      CompileCheckFileErrors |= CompileCheckTool.run(&CompileCheckFactory);
      if (CompileCheckFileErrors != 0) {
        ModUtil->addUniqueProblemFile(CompileCheckFile);   // Save problem file.
        HadErrors |= 1;
      }
      else
        ModUtil->addNoCompileErrorsFile(CompileCheckFile); // Save good file.
    }
  }

  // Then we make another pass on the good files to do the rest of the work.
  ClangTool Tool(*Compilations,
    (DisplayFileLists ? ModUtil->GoodFileNames : ModUtil->HeaderFileNames));
  Tool.appendArgumentsAdjuster(
    getModularizeArgumentsAdjuster(ModUtil->Dependencies));
  ModularizeFrontendActionFactory Factory(Entities, *PPTracker, HadErrors);
  HadErrors |= Tool.run(&Factory);

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
      // If only 1 occurrence of this entity, skip it, we only report duplicates.
      if (ECount <= 1)
        continue;
      LocationArray::iterator FI = DI->begin();
      StringRef kindName = Entry::getKindName((Entry::EntryKind)KindIndex);
      errs() << "error: " << kindName << " '" << E->first()
             << "' defined at multiple locations:\n";
      for (LocationArray::iterator FE = DI->end(); FI != FE; ++FI) {
        errs() << "    " << FI->File->getName() << ":" << FI->Line << ":"
               << FI->Column << "\n";
        ModUtil->addUniqueProblemFile(FI->File->getName());
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
    ModUtil->addUniqueProblemFile(H->first->getName());
    errs() << "error: header '" << H->first->getName()
           << "' has different contents depending on how it was included.\n";
    for (unsigned I = 0, N = H->second.size(); I != N; ++I) {
      errs() << "note: '" << H->second[I].Name << "' in "
             << H->second[I].Loc.File->getName() << " at "
             << H->second[I].Loc.Line << ":" << H->second[I].Loc.Column
             << " not always provided\n";
    }
  }

  if (DisplayFileLists) {
    ModUtil->displayProblemFiles();
    ModUtil->displayGoodFiles();
    ModUtil->displayCombinedFiles();
  }

  return HadErrors;
}
