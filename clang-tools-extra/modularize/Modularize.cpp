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
// error: '(symbol)' defined at multiple locations:
//     (file):(row):(column)
//     (file):(row):(column)
//
// error: header '(file)' has different contents dependening on how it was
//   included
//
// The latter might be followed by messages like the following:
//
// note: '(symbol)' in (file) at (row):(column) not always provided
//
// Future directions:
//
// Basically, we want to add new checks for whatever we can check with respect
// to checking headers for module'ability.
//
// Some ideas:
//
// 1. Try to figure out the preprocessor conditional directives that
// contribute to problems.
//
// 2. Check for correct and consistent usage of extern "C" {} and other
// directives. Warn about #include inside extern "C" {}.
//
// 3. What else?
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
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace clang::tooling;
using namespace clang;
using namespace llvm;

// Option to specify a file name for a list of header files to check.
cl::opt<std::string>
ListFileName(cl::Positional,
             cl::desc("<name of file containing list of headers to check>"));

// Collect all other arguments, which will be passed to the front end.
cl::list<std::string> CC1Arguments(
    cl::ConsumeAfter, cl::desc("<arguments to be passed to front end>..."));

// Option to specify a prefix to be prepended to the header names.
cl::opt<std::string> HeaderPrefix(
    "prefix", cl::init(""),
    cl::desc(
        "Prepend header file paths with this prefix."
        " If not specified,"
        " the files are considered to be relative to the header list file."));

// Read the header list file and collect the header file names.
error_code GetHeaderFileNames(SmallVectorImpl<std::string> &headerFileNames,
                              StringRef listFileName, StringRef headerPrefix) {

  // By default, use the path component of the list file name.
  SmallString<256> headerDirectory(listFileName);
  sys::path::remove_filename(headerDirectory);

  // Get the prefix if we have one.
  if (headerPrefix.size() != 0)
    headerDirectory = headerPrefix;

  // Read the header list file into a buffer.
  OwningPtr<MemoryBuffer> listBuffer;
  if (error_code ec = MemoryBuffer::getFile(ListFileName, listBuffer)) {
    return ec;
  }

  // Parse the header list into strings.
  SmallVector<StringRef, 32> strings;
  listBuffer->getBuffer().split(strings, "\n", -1, false);

  // Collect the header file names from the string list.
  for (SmallVectorImpl<StringRef>::iterator I = strings.begin(),
                                            E = strings.end();
       I != E; ++I) {
    StringRef line = (*I).trim();
    // Ignore comments and empty lines.
    if (line.empty() || (line[0] == '#'))
      continue;
    SmallString<256> headerFileName;
    // Prepend header file name prefix if it's not absolute.
    if (sys::path::is_absolute(line))
      headerFileName = line;
    else {
      headerFileName = headerDirectory;
      sys::path::append(headerFileName, line);
    }
    // Save the resulting header file path.
    headerFileNames.push_back(headerFileName.str());
  }

  return error_code::success();
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

class CollectEntitiesVisitor :
    public RecursiveASTVisitor<CollectEntitiesVisitor> {
public:
  CollectEntitiesVisitor(SourceManager &SM, EntityMap &Entities)
      : SM(SM), Entities(Entities) {}

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

  bool VisitNamedDecl(NamedDecl *ND) {
    // We only care about file-context variables.
    if (!ND->getDeclContext()->isFileContext())
      return true;

    // Skip declarations that tend to be properly multiply-declared.
    if (isa<NamespaceDecl>(ND) || isa<UsingDirectiveDecl>(ND) ||
        isa<NamespaceAliasDecl>(ND) ||
        isa<ClassTemplateSpecializationDecl>(ND) || isa<UsingDecl>(ND) ||
        isa<UsingShadowDecl>(ND) || isa<FunctionDecl>(ND) ||
        isa<FunctionTemplateDecl>(ND) ||
        (isa<TagDecl>(ND) &&
         !cast<TagDecl>(ND)->isThisDeclarationADefinition()))
      return true;

    std::string Name = ND->getNameAsString();
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
};

class CollectEntitiesConsumer : public ASTConsumer {
public:
  CollectEntitiesConsumer(EntityMap &Entities, Preprocessor &PP)
      : Entities(Entities), PP(PP) {}

  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    SourceManager &SM = Ctx.getSourceManager();

    // Collect declared entities.
    CollectEntitiesVisitor(SM, Entities)
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
  Preprocessor &PP;
};

class CollectEntitiesAction : public SyntaxOnlyAction {
public:
  CollectEntitiesAction(EntityMap &Entities) : Entities(Entities) {}
protected:
  virtual clang::ASTConsumer *
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
    return new CollectEntitiesConsumer(Entities, CI.getPreprocessor());
  }
private:
  EntityMap &Entities;
};

class ModularizeFrontendActionFactory : public FrontendActionFactory {
public:
  ModularizeFrontendActionFactory(EntityMap &Entities) : Entities(Entities) {}

  virtual CollectEntitiesAction *create() {
    return new CollectEntitiesAction(Entities);
  }
private:
  EntityMap &Entities;
};

int main(int argc, const char **argv) {

  // This causes options to be parsed.
  cl::ParseCommandLineOptions(argc, argv, "modularize.\n");

  // No go if we have no header list file.
  if (ListFileName.size() == 0) {
    cl::PrintHelpMessage();
    return 1;
  }

  // Get header file names.
  SmallVector<std::string, 32> Headers;
  if (error_code ec = GetHeaderFileNames(Headers, ListFileName, HeaderPrefix)) {
    errs() << argv[0] << ": error: Unable to get header list '" << ListFileName
           << "': " << ec.message() << '\n';
    return 1;
  }

  // Create the compilation database.
  SmallString<256> PathBuf;
  sys::fs::current_path(PathBuf);
  OwningPtr<CompilationDatabase> Compilations;
  Compilations.reset(
      new FixedCompilationDatabase(Twine(PathBuf), CC1Arguments));

  // Parse all of the headers, detecting duplicates.
  EntityMap Entities;
  ClangTool Tool(*Compilations, Headers);
  int HadErrors = Tool.run(new ModularizeFrontendActionFactory(Entities));

  // Create a place to save duplicate entity locations, separate bins per kind.
  typedef SmallVector<Location, 8> LocationArray;
  typedef SmallVector<LocationArray, Entry::EK_NumberOfKinds> EntryBinArray;
  EntryBinArray EntryBins;
  int kindIndex;
  for (kindIndex = 0; kindIndex < Entry::EK_NumberOfKinds; ++kindIndex) {
    LocationArray array;
    EntryBins.push_back(array);
  }

  // Check for the same entity being defined in multiple places.
  for (EntityMap::iterator E = Entities.begin(), EEnd = Entities.end();
       E != EEnd; ++E) {
    // If only one occurance, exit early.
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
    int kindIndex = 0;
    for (EntryBinArray::iterator DI = EntryBins.begin(), DE = EntryBins.end();
         DI != DE; ++DI, ++kindIndex) {
      int eCount = DI->size();
      // If only 1 occurance, skip;
      if (eCount <= 1)
        continue;
      LocationArray::iterator FI = DI->begin();
      StringRef kindName = Entry::getKindName((Entry::EntryKind) kindIndex);
      errs() << "error: " << kindName << " '" << E->first()
             << "' defined at multiple locations:\n";
      for (LocationArray::iterator FE = DI->end(); FI != FE; ++FI) {
        errs() << "    " << FI->File->getName() << ":" << FI->Line << ":"
               << FI->Column << "\n";
      }
      HadErrors = 1;
    }
  }

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
           << "' has different contents dependening on how it was included\n";
    for (unsigned I = 0, N = H->second.size(); I != N; ++I) {
      errs() << "note: '" << H->second[I].Name << "' in " << H->second[I]
          .Loc.File->getName() << " at " << H->second[I].Loc.Line << ":"
             << H->second[I].Loc.Column << " not always provided\n";
    }
  }

  return HadErrors;
}
