//===-- core_main.cpp - Core Index Tool testbed ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "JSONAggregation.h"
#include "indexstore/IndexStoreCXX.h"
#include "clang/DirectoryWatcher/DirectoryWatcher.h"
#include "clang/Basic/LangOptions.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexDataStoreSymbolUtils.h"
#include "clang/Index/IndexRecordReader.h"
#include "clang/Index/IndexUnitReader.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Index/CodegenNameGenerator.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/PrettyStackTrace.h"
#include <thread>

using namespace clang;
using namespace clang::index;
using namespace llvm;

extern "C" int indextest_core_main(int argc, const char **argv);

namespace {

enum class ActionType {
  None,
  PrintSourceSymbols,
  PrintRecord,
  PrintUnit,
  PrintStoreFormatVersion,
  AggregateAsJSON,
  WatchDir,
};

namespace options {

static cl::OptionCategory IndexTestCoreCategory("index-test-core options");

static cl::opt<ActionType>
Action(cl::desc("Action:"), cl::init(ActionType::None),
       cl::values(
          clEnumValN(ActionType::PrintSourceSymbols,
                     "print-source-symbols", "Print symbols from source"),
          clEnumValN(ActionType::PrintRecord,
                     "print-record", "Print record info"),
          clEnumValN(ActionType::PrintUnit,
                     "print-unit", "Print unit info"),
          clEnumValN(ActionType::PrintStoreFormatVersion,
                     "print-store-format-version", "Print store format version"),
          clEnumValN(ActionType::AggregateAsJSON,
                     "aggregate-json", "Aggregate index data in JSON format"),
          clEnumValN(ActionType::WatchDir,
                     "watch-dir", "Watch directory for file events")),
       cl::cat(IndexTestCoreCategory));

static cl::opt<std::string>
OutputFile("o", cl::desc("output file"),
           cl::cat(IndexTestCoreCategory));

static cl::list<std::string>
InputFiles(cl::Positional, cl::desc("<filename>..."));

static cl::extrahelp MoreHelp(
  "\nAdd \"-- <compiler arguments>\" at the end to setup the compiler "
  "invocation\n"
);

static cl::opt<bool>
DumpModuleImports("dump-imported-module-files",
               cl::desc("Print symbols and input files from imported modules"));

static cl::opt<bool>
IncludeLocals("include-locals", cl::desc("Print local symbols"));

static cl::opt<std::string>
ModuleFilePath("module-file",
               cl::desc("Path to module file to print symbols from"));
static cl::opt<std::string>
  ModuleFormat("fmodule-format", cl::init("raw"),
        cl::desc("Container format for clang modules and PCH, 'raw' or 'obj'"));

static cl::opt<std::string>
FilePathAndRange("filepath",
               cl::desc("File path that can optionally include a line range"));

}
} // anonymous namespace

static void printSymbolInfo(SymbolInfo SymInfo, raw_ostream &OS);
static void printSymbolNameAndUSR(const Decl *D, ASTContext &Ctx,
                                  raw_ostream &OS);
static void printSymbolNameAndUSR(const clang::Module *Mod, raw_ostream &OS);

namespace {

class PrintIndexDataConsumer : public IndexDataConsumer {
  raw_ostream &OS;
  std::unique_ptr<CodegenNameGenerator> CGNameGen;
  std::shared_ptr<Preprocessor> PP;

public:
  PrintIndexDataConsumer(raw_ostream &OS) : OS(OS) {
  }

  void initialize(ASTContext &Ctx) override {
    CGNameGen.reset(new CodegenNameGenerator(Ctx));
  }

  void setPreprocessor(std::shared_ptr<Preprocessor> PP) override {
    this->PP = std::move(PP);
  }

  bool handleDeclOccurence(const Decl *D, SymbolRoleSet Roles,
                           ArrayRef<SymbolRelation> Relations,
                           SourceLocation Loc, ASTNodeInfo ASTNode) override {
    ASTContext &Ctx = D->getASTContext();
    SourceManager &SM = Ctx.getSourceManager();

    Loc = SM.getFileLoc(Loc);
    FileID FID = SM.getFileID(Loc);
    unsigned Line = SM.getLineNumber(FID, SM.getFileOffset(Loc));
    unsigned Col = SM.getColumnNumber(FID, SM.getFileOffset(Loc));
    OS << Line << ':' << Col << " | ";

    printSymbolInfo(getSymbolInfo(D), OS);
    OS << " | ";

    printSymbolNameAndUSR(D, Ctx, OS);
    OS << " | ";

    if (CGNameGen->writeName(D, OS))
      OS << "<no-cgname>";
    OS << " | ";

    printSymbolRoles(Roles, OS);
    OS << " | ";

    OS << "rel: " << Relations.size() << '\n';

    for (auto &SymRel : Relations) {
      OS << '\t';
      printSymbolRoles(SymRel.Roles, OS);
      OS << " | ";
      printSymbolNameAndUSR(SymRel.RelatedSymbol, Ctx, OS);
      OS << '\n';
    }

    return true;
  }

  bool handleModuleOccurence(const ImportDecl *ImportD,
                             const clang::Module *Mod,
                             SymbolRoleSet Roles, SourceLocation Loc) override {
    ASTContext &Ctx = ImportD->getASTContext();
    SourceManager &SM = Ctx.getSourceManager();

    Loc = SM.getFileLoc(Loc);
    FileID FID = SM.getFileID(Loc);
    unsigned Line = SM.getLineNumber(FID, SM.getFileOffset(Loc));
    unsigned Col = SM.getColumnNumber(FID, SM.getFileOffset(Loc));
    OS << Line << ':' << Col << " | ";

    printSymbolInfo(getSymbolInfo(ImportD), OS);
    OS << " | ";

    printSymbolNameAndUSR(Mod, OS);
    OS << " | ";

    printSymbolRoles(Roles, OS);
    OS << " |\n";

    return true;
  }

  bool handleMacroOccurence(const IdentifierInfo *Name, const MacroInfo *MI,
                            SymbolRoleSet Roles, SourceLocation Loc) override {
    assert(PP);
    SourceManager &SM = PP->getSourceManager();

    Loc = SM.getFileLoc(Loc);
    FileID FID = SM.getFileID(Loc);
    unsigned Line = SM.getLineNumber(FID, SM.getFileOffset(Loc));
    unsigned Col = SM.getColumnNumber(FID, SM.getFileOffset(Loc));
    OS << Line << ':' << Col << " | ";

    printSymbolInfo(getSymbolInfoForMacro(*MI), OS);
    OS << " | ";

    OS << Name->getName();
    OS << " | ";

    SmallString<256> USRBuf;
    if (generateUSRForMacro(Name->getName(), MI->getDefinitionLoc(), SM,
                            USRBuf)) {
      OS << "<no-usr>";
    } else {
      OS << USRBuf;
    }
    OS << " | ";

    printSymbolRoles(Roles, OS);
    OS << " |\n";
    return true;
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Print Source Symbols
//===----------------------------------------------------------------------===//

static void dumpModuleFileInputs(serialization::ModuleFile &Mod,
                                 ASTReader &Reader,
                                 raw_ostream &OS) {
  OS << "---- Module Inputs ----\n";
  Reader.visitInputFiles(Mod, /*IncludeSystem=*/true, /*Complain=*/false,
                        [&](const serialization::InputFile &IF, bool isSystem) {
    OS << (isSystem ? "system" : "user") << " | ";
    OS << IF.getFile()->getName() << '\n';
  });
}

static bool printSourceSymbols(const char *Executable,
                               ArrayRef<const char *> Args,
                               bool dumpModuleImports, bool indexLocals) {
  SmallVector<const char *, 4> ArgsWithProgName;
  ArgsWithProgName.push_back(Executable);
  ArgsWithProgName.append(Args.begin(), Args.end());
  IntrusiveRefCntPtr<DiagnosticsEngine>
    Diags(CompilerInstance::createDiagnostics(new DiagnosticOptions));
  auto CInvok = createInvocationFromCommandLine(ArgsWithProgName, Diags);
  if (!CInvok)
    return true;

  raw_ostream &OS = outs();
  auto DataConsumer = std::make_shared<PrintIndexDataConsumer>(OS);
  IndexingOptions IndexOpts;
  IndexOpts.IndexFunctionLocals = indexLocals;
  std::unique_ptr<FrontendAction> IndexAction;
  IndexAction = createIndexingAction(DataConsumer, IndexOpts,
                                     /*WrappedAction=*/nullptr);

  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<ASTUnit> Unit(ASTUnit::LoadFromCompilerInvocationAction(
      std::move(CInvok), PCHContainerOps, Diags, IndexAction.get()));

  if (!Unit)
    return true;

  if (dumpModuleImports) {
    if (auto Reader = Unit->getASTReader()) {
      Reader->getModuleManager().visit([&](serialization::ModuleFile &Mod) -> bool {
        OS << "==== Module " << Mod.ModuleName << " ====\n";
        indexModuleFile(Mod, *Reader, *DataConsumer, IndexOpts);
        dumpModuleFileInputs(Mod, *Reader, OS);
        return true; // skip module dependencies.
      });
    }
  }

  return false;
}

static bool printSourceSymbolsFromModule(StringRef modulePath,
                                         StringRef format) {
  FileSystemOptions FileSystemOpts;
  auto pchContOps = std::make_shared<PCHContainerOperations>();
  // Register the support for object-file-wrapped Clang modules.
  pchContOps->registerReader(llvm::make_unique<ObjectFilePCHContainerReader>());
  auto pchRdr = pchContOps->getReaderOrNull(format);
  if (!pchRdr) {
    errs() << "unknown module format: " << format << '\n';
    return true;
  }

  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions());
  std::unique_ptr<ASTUnit> AU = ASTUnit::LoadFromASTFile(
      modulePath, *pchRdr, ASTUnit::LoadASTOnly, Diags,
      FileSystemOpts, /*UseDebugInfo=*/false,
      /*OnlyLocalDecls=*/true, None,
      /*CaptureDiagnostics=*/false,
      /*AllowPCHWithCompilerErrors=*/true,
      /*UserFilesAreVolatile=*/false);
  if (!AU) {
    errs() << "failed to create TU for: " << modulePath << '\n';
    return true;
  }

  PrintIndexDataConsumer DataConsumer(outs());
  IndexingOptions IndexOpts;
  indexASTUnit(*AU, DataConsumer, IndexOpts);

  return false;
}

//===----------------------------------------------------------------------===//
// Print Record
//===----------------------------------------------------------------------===//

static void printSymbol(const IndexRecordDecl &Rec, raw_ostream &OS);
static void printSymbol(const IndexRecordOccurrence &Rec, raw_ostream &OS);

static int printRecord(StringRef Filename, raw_ostream &OS) {
  std::string Error;
  auto Reader = IndexRecordReader::createWithFilePath(Filename, Error);
  if (!Reader) {
    errs() << Error << '\n';
    return true;
  }

  Reader->foreachDecl(/*noCache=*/true, [&](const IndexRecordDecl *Rec)->bool {
    printSymbol(*Rec, OS);
    return true;
  });
  OS << "------------\n";
  Reader->foreachOccurrence([&](const IndexRecordOccurrence &Rec)->bool {
    printSymbol(Rec, OS);
    return true;
  });

  return false;
};

//===----------------------------------------------------------------------===//
// Print Store Records
//===----------------------------------------------------------------------===//

static void printSymbol(indexstore::IndexRecordSymbol Sym, raw_ostream &OS);
static void printSymbol(indexstore::IndexRecordOccurrence Occur, raw_ostream &OS);

static bool printStoreRecord(indexstore::IndexStore &Store, StringRef RecName,
                             StringRef FilePath, raw_ostream &OS) {
  std::string Error;
  indexstore::IndexRecordReader Reader(Store, RecName, Error);
  if (!Reader) {
    errs() << "error loading record: " << Error << "\n";
    return true;
  }

  StringRef Filename = sys::path::filename(FilePath);
  OS << Filename << '\n';
  OS << "------------\n";
  Reader.foreachSymbol(/*noCache=*/true, [&](indexstore::IndexRecordSymbol Sym) -> bool {
    printSymbol(Sym, OS);
    return true;
  });
  OS << "------------\n";
  Reader.foreachOccurrence([&](indexstore::IndexRecordOccurrence Occur)->bool {
    printSymbol(Occur, OS);
    return true;
  });

  return false;
}

static int printStoreRecords(StringRef StorePath, raw_ostream &OS) {
  std::string Error;
  indexstore::IndexStore Store(StorePath, Error);
  if (!Store) {
    errs() << "error loading store: " << Error << "\n";
    return 1;
  }

  bool Success = Store.foreachUnit(/*sorted=*/true, [&](StringRef UnitName) -> bool {
    indexstore::IndexUnitReader Reader(Store, UnitName, Error);
    if (!Reader) {
      errs() << "error loading unit: " << Error << "\n";
      return false;
    }
    return Reader.foreachDependency([&](indexstore::IndexUnitDependency Dep) -> bool {
      if (Dep.getKind() == indexstore::IndexUnitDependency::DependencyKind::Record) {
        bool Err = printStoreRecord(Store, Dep.getName(), Dep.getFilePath(), OS);
        OS << '\n';
        return !Err;
      }
      return true;
    });
  });

  return !Success;
}

static std::string findRecordNameForFile(indexstore::IndexStore &store,
                                         StringRef filePath) {
  std::string recName;
  store.foreachUnit(/*sorted=*/false, [&](StringRef unitName) -> bool {
    std::string error;
    indexstore::IndexUnitReader Reader(store, unitName, error);
    if (!Reader) {
      errs() << "error loading unit: " << error << "\n";
      return false;
    }
    Reader.foreachDependency([&](indexstore::IndexUnitDependency Dep) -> bool {
      if (Dep.getKind() == indexstore::IndexUnitDependency::DependencyKind::Record) {
        if (Dep.getFilePath() == filePath) {
          recName = Dep.getName();
          return false;
        }
        return true;
      }
      return true;
    });
    return true;
  });
  return recName;
}

static int printStoreFileRecord(StringRef storePath, StringRef filePath,
                                Optional<unsigned> lineStart, unsigned lineCount,
                                raw_ostream &OS) {
  std::string error;
  indexstore::IndexStore store(storePath, error);
  if (!store) {
    errs() << "error loading store: " << error << "\n";
    return 1;
  }

  std::string recName = findRecordNameForFile(store, filePath);
  if (recName.empty()) {
    errs() << "could not find record for '" << filePath << "'\n";
    return 1;
  }

  if (!lineStart.hasValue())
    return printStoreRecord(store, recName, filePath, OS);

  indexstore::IndexRecordReader Reader(store, recName, error);
  if (!Reader) {
    errs() << "error loading record: " << error << "\n";
    return 1;
  }

  Reader.foreachOccurrenceInLineRange(*lineStart, lineCount, [&](indexstore::IndexRecordOccurrence Occur)->bool {
    printSymbol(Occur, OS);
    return true;
  });

  return 0;
}


//===----------------------------------------------------------------------===//
// Print Unit
//===----------------------------------------------------------------------===//

static int printUnit(StringRef Filename, raw_ostream &OS) {
  std::string Error;
  auto Reader = IndexUnitReader::createWithFilePath(Filename, Error);
  if (!Reader) {
    errs() << Error << '\n';
    return true;
  }

  OS << "provider: " << Reader->getProviderIdentifier() << '-' << Reader->getProviderVersion() << '\n';
  OS << "is-system: " << Reader->isSystemUnit() << '\n';
  OS << "is-module: " << Reader->isModuleUnit() << '\n';
  OS << "module-name: " << (Reader->getModuleName().empty() ? "<none>" : Reader->getModuleName()) << '\n';
  OS << "has-main: " << Reader->hasMainFile() << '\n';
  OS << "main-path: " << Reader->getMainFilePath() << '\n';
  OS << "work-dir: " << Reader->getWorkingDirectory() << '\n';
  OS << "out-file: " << Reader->getOutputFile() << '\n';
  OS << "target: " << Reader->getTarget() << '\n';
  OS << "is-debug: " << Reader->isDebugCompilation() << '\n';
  OS << "DEPEND START\n";
  unsigned NumDepends = 0;
  Reader->foreachDependency([&](const IndexUnitReader::DependencyInfo &Dep) -> bool {
    switch (Dep.Kind) {
    case IndexUnitReader::DependencyKind::Unit:
      OS << "Unit | "; break;
    case IndexUnitReader::DependencyKind::Record:
      OS << "Record | "; break;
    case IndexUnitReader::DependencyKind::File:
      OS << "File | "; break;
    }
    OS << (Dep.IsSystem ? "system" : "user");
    OS << " | ";
    if (!Dep.ModuleName.empty())
      OS << Dep.ModuleName << " | ";
    OS << Dep.FilePath;
    if (!Dep.UnitOrRecordName.empty())
      OS << " | " << Dep.UnitOrRecordName;
    OS << '\n';
    ++NumDepends;
    return true;
  });
  OS << "DEPEND END (" << NumDepends << ")\n";
  OS << "INCLUDE START\n";
  unsigned NumIncludes = 0;
  Reader->foreachInclude([&](const IndexUnitReader::IncludeInfo &Inc) -> bool {
    OS << Inc.SourcePath << ":" << Inc.SourceLine << " | ";
    OS << Inc.TargetPath << '\n';
    ++NumIncludes;
    return true;
  });
  OS << "INCLUDE END (" << NumIncludes << ")\n";

  return false;
};

//===----------------------------------------------------------------------===//
// Print Store Units
//===----------------------------------------------------------------------===//

static bool printStoreUnit(indexstore::IndexStore &Store, StringRef UnitName,
                           raw_ostream &OS) {
  std::string Error;
  indexstore::IndexUnitReader Reader(Store, UnitName, Error);
  if (!Reader) {
    errs() << "error loading unit: " << Error << "\n";
    return true;
  }

  OS << "provider: " << Reader.getProviderIdentifier() << '-' << Reader.getProviderVersion() << '\n';
  OS << "is-system: " << Reader.isSystemUnit() << '\n';
  OS << "is-module: " << Reader.isModuleUnit() << '\n';
  OS << "module-name: " << (Reader.getModuleName().empty() ? "<none>" : Reader.getModuleName()) << '\n';
  OS << "has-main: " << Reader.hasMainFile() << '\n';
  OS << "main-path: " << Reader.getMainFilePath() << '\n';
  OS << "work-dir: " << Reader.getWorkingDirectory() << '\n';
  OS << "out-file: " << Reader.getOutputFile() << '\n';
  OS << "target: " << Reader.getTarget() << '\n';
  OS << "is-debug: " << Reader.isDebugCompilation() << '\n';
  OS << "DEPEND START\n";
  unsigned NumDepends = 0;
  Reader.foreachDependency([&](indexstore::IndexUnitDependency Dep) -> bool {
    switch (Dep.getKind()) {
    case indexstore::IndexUnitDependency::DependencyKind::Unit:
      OS << "Unit | "; break;
    case indexstore::IndexUnitDependency::DependencyKind::Record:
      OS << "Record | "; break;
    case indexstore::IndexUnitDependency::DependencyKind::File:
      OS << "File | "; break;
    }
    OS << (Dep.isSystem() ? "system" : "user");
    OS << " | ";
    if (!Dep.getModuleName().empty())
      OS << Dep.getModuleName() << " | ";
    OS << Dep.getFilePath();
    if (!Dep.getName().empty())
      OS << " | " << Dep.getName();
    OS << '\n';
    ++NumDepends;
    return true;
  });
  OS << "DEPEND END (" << NumDepends << ")\n";
  OS << "INCLUDE START\n";
  unsigned NumIncludes = 0;
  Reader.foreachInclude([&](indexstore::IndexUnitInclude Inc) -> bool {
    OS << Inc.getSourcePath() << ":" << Inc.getSourceLine() << " | ";
    OS << Inc.getTargetPath() << '\n';
    ++NumIncludes;
    return true;
  });
  OS << "INCLUDE END (" << NumIncludes << ")\n";

  return false;
}

static int printStoreUnits(StringRef StorePath, raw_ostream &OS) {
  std::string Error;
  indexstore::IndexStore Store(StorePath, Error);
  if (!Store) {
    errs() << "error loading store: " << Error << "\n";
    return 1;
  }

  bool Success = Store.foreachUnit(/*sorted=*/true, [&](StringRef UnitName) -> bool {
    OS << UnitName << '\n';
    OS << "--------\n";
    bool err = printStoreUnit(Store, UnitName, OS);
    OS << '\n';
    return !err;
  });

  return !Success;
}

//===----------------------------------------------------------------------===//
// Helper Utils
//===----------------------------------------------------------------------===//

static void printSymbolInfo(SymbolInfo SymInfo, raw_ostream &OS) {
  OS << getSymbolKindString(SymInfo.Kind);
  if (SymInfo.SubKind != SymbolSubKind::None)
    OS << '/' << getSymbolSubKindString(SymInfo.SubKind);
  if (SymInfo.Properties) {
    OS << '(';
    printSymbolProperties(SymInfo.Properties, OS);
    OS << ')';
  }
  OS << '/' << getSymbolLanguageString(SymInfo.Lang);
}

static void printSymbolNameAndUSR(const Decl *D, ASTContext &Ctx,
                                  raw_ostream &OS) {
  if (printSymbolName(D, Ctx.getLangOpts(), OS)) {
    OS << "<no-name>";
  }
  OS << " | ";

  SmallString<256> USRBuf;
  if (generateUSRForDecl(D, USRBuf)) {
    OS << "<no-usr>";
  } else {
    OS << USRBuf;
  }
}

static void printSymbolNameAndUSR(const clang::Module *Mod, raw_ostream &OS) {
  assert(Mod);
  OS << Mod->getFullModuleName() << " | ";
  generateFullUSRForModule(Mod, OS);
}

static void printSymbol(const IndexRecordDecl &Rec, raw_ostream &OS) {
  printSymbolInfo(Rec.SymInfo, OS);
  OS << " | ";

  if (Rec.Name.empty())
    OS << "<no-name>";
  else
    OS << Rec.Name;
  OS << " | ";

  if (Rec.USR.empty())
    OS << "<no-usr>";
  else
    OS << Rec.USR;
  OS << " | ";

  if (Rec.CodeGenName.empty())
    OS << "<no-cgname>";
  else
    OS << Rec.CodeGenName;
  OS << " | ";

  printSymbolRoles(Rec.Roles, OS);
  OS << " - ";
  printSymbolRoles(Rec.RelatedRoles, OS);
  OS << '\n';
}

static void printSymbol(const IndexRecordOccurrence &Rec, raw_ostream &OS) {
  OS << Rec.Line << ':' << Rec.Column << " | ";
  printSymbolInfo(Rec.Dcl->SymInfo, OS);
  OS << " | ";

  if (Rec.Dcl->USR.empty())
    OS << "<no-usr>";
  else
    OS << Rec.Dcl->USR;
  OS << " | ";

  printSymbolRoles(Rec.Roles, OS);
  OS << " | ";
  OS << "rel: " << Rec.Relations.size() << '\n';
  for (auto &Rel : Rec.Relations) {
    OS << '\t';
    printSymbolRoles(Rel.Roles, OS);
    OS << " | ";
    if (Rel.Dcl->USR.empty())
      OS << "<no-usr>";
    else
      OS << Rel.Dcl->USR;
    OS << '\n';
  }
}

static void printSymbol(indexstore::IndexRecordSymbol Sym, raw_ostream &OS) {
  SymbolInfo SymInfo{getSymbolKind(Sym.getKind()),
                     getSymbolSubKind(Sym.getSubKind()),
                     getSymbolLanguage(Sym.getLanguage()),
                     SymbolPropertySet(Sym.getProperties())};

  printSymbolInfo(SymInfo, OS);
  OS << " | ";

  if (Sym.getName().empty())
    OS << "<no-name>";
  else
    OS << Sym.getName();
  OS << " | ";

  if (Sym.getUSR().empty())
    OS << "<no-usr>";
  else
    OS << Sym.getUSR();
  OS << " | ";

  if (Sym.getCodegenName().empty())
    OS << "<no-cgname>";
  else
    OS << Sym.getCodegenName();
  OS << " | ";

  printSymbolRoles(getSymbolRoles(Sym.getRoles()), OS);
  OS << " - ";
  printSymbolRoles(getSymbolRoles(Sym.getRelatedRoles()), OS);
  OS << '\n';
}

static void printSymbol(indexstore::IndexRecordOccurrence Occur, raw_ostream &OS) {
  OS << Occur.getLineCol().first << ':' << Occur.getLineCol().second << " | ";
  auto Sym = Occur.getSymbol();
  SymbolInfo SymInfo{getSymbolKind(Sym.getKind()),
                     getSymbolSubKind(Sym.getSubKind()),
                     getSymbolLanguage(Sym.getLanguage()),
                     SymbolPropertySet(Sym.getProperties())};

  printSymbolInfo(SymInfo, OS);
  OS << " | ";

  if (Sym.getUSR().empty())
    OS << "<no-usr>";
  else
    OS << Sym.getUSR();
  OS << " | ";

  unsigned NumRelations = 0;
  Occur.foreachRelation([&](indexstore::IndexSymbolRelation) {
    ++NumRelations;
    return true;
  });

  printSymbolRoles(getSymbolRoles(Occur.getRoles()), OS);
  OS << " | ";
  OS << "rel: " << NumRelations << '\n';
  Occur.foreachRelation([&](indexstore::IndexSymbolRelation Rel) {
    OS << '\t';
    printSymbolRoles(getSymbolRoles(Rel.getRoles()), OS);
    OS << " | ";
    auto Sym = Rel.getSymbol();
    if (Sym.getUSR().empty())
      OS << "<no-usr>";
    else
      OS << Sym.getUSR();
    OS << '\n';
    return true;
  });
}

static int watchDirectory(StringRef dirPath) {
  raw_ostream &OS = outs();
  auto receiver = [&](ArrayRef<DirectoryWatcher::Event> Events, bool isInitial) {
    OS << "-- " << Events.size() << " :\n";
    for (auto evt : Events) {
      switch (evt.Kind) {
        case DirectoryWatcher::EventKind::Added:
          OS << "added: "; break;
        case DirectoryWatcher::EventKind::Modified:
          OS << "modified: "; break;
        case DirectoryWatcher::EventKind::Removed:
          OS << "removed: "; break;
        case DirectoryWatcher::EventKind::DirectoryDeleted:
          OS << "dir deleted: "; break;

      }
      OS << evt.Filename << '\n';
    }
  };
  std::string Error;
  auto watcher = DirectoryWatcher::create(dirPath, receiver,
                                          /*waitInitialSync=*/true, Error);
  if (!watcher) {
    errs() << "failed creating directory watcher: " << Error << '\n';
    return 1;
  }

  while(1) {
    std::this_thread::yield();
  }
}

//===----------------------------------------------------------------------===//
// Command line processing.
//===----------------------------------------------------------------------===//

bool deconstructPathAndRange(StringRef input,
                             std::string &filepath,
                             Optional<unsigned> &lineStart,
                             unsigned &lineCount) {
  StringRef path, range;
  std::tie(path, range) = input.split(':');
  StringRef start, end;
  std::tie(start, end) = range.split(':');
  filepath = path;
  lineCount = 0;
  if (start.empty())
    return false;
  unsigned num;
  if (start.getAsInteger(10, num)) {
    errs() << "couldn't convert to integer: " << start << '\n';
    return true;
  }
  lineStart = num;
  if (end.empty())
    return false;
  if (end.getAsInteger(10, num)) {
    errs() << "couldn't convert to integer: " << end << '\n';
    return true;
  }
  lineCount = num-lineStart.getValue();
  return false;
}

int indextest_core_main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  void *MainAddr = (void*) (intptr_t) indextest_core_main;
  std::string Executable = llvm::sys::fs::getMainExecutable(argv[0], MainAddr);

  assert(argv[1] == StringRef("core"));
  ++argv;
  --argc;

  std::vector<const char *> CompArgs;
  const char **DoubleDash = std::find(argv, argv + argc, StringRef("--"));
  if (DoubleDash != argv + argc) {
    CompArgs = std::vector<const char *>(DoubleDash + 1, argv + argc);
    argc = DoubleDash - argv;
  }

  cl::HideUnrelatedOptions(options::IndexTestCoreCategory);
  cl::ParseCommandLineOptions(argc, argv, "index-test-core");

  if (options::Action == ActionType::None) {
    errs() << "error: action required; pass '-help' for options\n";
    return 1;
  }

  if (options::Action == ActionType::PrintSourceSymbols) {
    if (!options::ModuleFilePath.empty()) {
      return printSourceSymbolsFromModule(options::ModuleFilePath,
                                          options::ModuleFormat);
    }
    if (CompArgs.empty()) {
      errs() << "error: missing compiler args; pass '-- <compiler arguments>'\n";
      return 1;
    }
    return printSourceSymbols(Executable.c_str(), CompArgs,
                              options::DumpModuleImports,
                              options::IncludeLocals);
  }

  if (options::Action == ActionType::PrintRecord) {
    if (!options::FilePathAndRange.empty()) {
      std::string filepath;
      Optional<unsigned> lineStart;
      unsigned lineCount;
      if (deconstructPathAndRange(options::FilePathAndRange,
                                  filepath, lineStart, lineCount))
        return 1;

      if (options::InputFiles.empty()) {
        errs() << "error: missing index store path\n";
        return 1;
      }
      return printStoreFileRecord(options::InputFiles[0], filepath, lineStart, lineCount, outs());
    }

    if (options::InputFiles.empty()) {
      errs() << "error: missing input file or directory\n";
      return 1;
    }

    if (sys::fs::is_directory(options::InputFiles[0]))
      return printStoreRecords(options::InputFiles[0], outs());
    else
      return printRecord(options::InputFiles[0], outs());
  }

  if (options::Action == ActionType::PrintUnit) {
    if (options::InputFiles.empty()) {
      errs() << "error: missing input file or directory\n";
      return 1;
    }

    if (sys::fs::is_directory(options::InputFiles[0]))
      return printStoreUnits(options::InputFiles[0], outs());
    else
      return printUnit(options::InputFiles[0], outs());
  }

  if (options::Action == ActionType::PrintStoreFormatVersion) {
    outs() << indexstore::IndexStore::formatVersion() << '\n';
  }

  if (options::Action == ActionType::AggregateAsJSON) {
    if (options::InputFiles.empty()) {
      errs() << "error: missing input data store directory\n";
      return 1;
    }
    StringRef storePath = options::InputFiles[0];
    if (options::OutputFile.empty())
      return aggregateDataAsJSON(storePath, outs());
    std::error_code EC;
    raw_fd_ostream OS(options::OutputFile, EC, llvm::sys::fs::F_None);
    if (EC) {
      errs() << "failed to open output file: " << EC.message() << '\n';
      return 1;
    }
    return aggregateDataAsJSON(storePath, OS);
  }

  if (options::Action == ActionType::WatchDir) {
    if (options::InputFiles.empty()) {
      errs() << "error: missing directory path\n";
      return 1;
    }
    return watchDirectory(options::InputFiles[0]);
  }

  return 0;
}
