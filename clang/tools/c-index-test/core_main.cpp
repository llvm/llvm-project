//===-- core_main.cpp - Core Index Tool testbed ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONAggregation.h"
#include "indexstore/IndexStoreCXX.h"
#include "clang-c/Dependencies.h"
#include "clang/DirectoryWatcher/DirectoryWatcher.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/PathRemapper.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexDataStoreSymbolUtils.h"
#include "clang/Index/IndexRecordReader.h"
#include "clang/Index/IndexUnitReader.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <future>
#include <thread>

using namespace clang;
using namespace clang::index;
using namespace llvm;

extern "C" int indextest_core_main(int argc, const char **argv);
extern "C" int indextest_perform_shell_execution(const char *command_line);

namespace {

enum class ActionType {
  None,
  PrintSourceSymbols,
  PrintRecord,
  PrintUnit,
  PrintStoreFormatVersion,
  AggregateAsJSON,
  ScanDeps,
  ScanDepsByModuleName,
  GenerateDepsReproducer,
  UploadCachedJob,
  MaterializeCachedJob,
  ReplayCachedJob,
  PruneCAS,
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
          clEnumValN(ActionType::ScanDeps, "scan-deps",
                     "Get file dependencies"),
          clEnumValN(ActionType::ScanDepsByModuleName, "scan-deps-by-mod-name",
                     "Get file dependencies by module name alone"),
          clEnumValN(ActionType::GenerateDepsReproducer, "gen-deps-reproducer",
                     "Generate a reproducer for the file"),
          clEnumValN(ActionType::UploadCachedJob, "upload-cached-job",
                     "Upload cached compilation data to upstream CAS"),
          clEnumValN(ActionType::MaterializeCachedJob, "materialize-cached-job",
                     "Materialize cached compilation data from upstream CAS"),
          clEnumValN(ActionType::ReplayCachedJob, "replay-cached-job",
                     "Replay a cached compilation from the CAS"),
          clEnumValN(ActionType::PruneCAS, "prune-cas", "Prune CAS data"),
          clEnumValN(ActionType::WatchDir,
                     "watch-dir", "Watch directory for file events")),
       cl::cat(IndexTestCoreCategory));

static cl::opt<std::string>
OutputFile("o", cl::desc("output file"),
           cl::cat(IndexTestCoreCategory));

static cl::list<std::string>
InputFiles(cl::Positional, cl::desc("<filename>..."));

static cl::list<std::string>
PrefixMap("index-store-prefix-map", cl::desc("<prefix=replacement>..."));

static cl::extrahelp MoreHelp(
  "\nAdd \"-- <compiler arguments>\" at the end to setup the compiler "
  "invocation\n"
);

static cl::opt<bool>
DumpModuleImports("dump-imported-module-files",
               cl::desc("Print symbols and input files from imported modules"));

static cl::opt<bool>
IncludeLocals("include-locals", cl::desc("Print local symbols"));

static cl::opt<bool> IgnoreMacros("ignore-macros",
                                  cl::desc("Skip indexing macros"));

static cl::opt<std::string>
ModuleFilePath("module-file",
               cl::desc("Path to module file to print symbols from"));
static cl::opt<std::string>
  ModuleFormat("fmodule-format", cl::init("raw"),
        cl::desc("Container format for clang modules and PCH, 'raw' or 'obj'"));

static cl::opt<std::string>
FilePathAndRange("filepath",
               cl::desc("File path that can optionally include a line range"));

static cl::opt<std::string>
    ModuleName("module-name", cl::desc("name of the module, of which we are "
                                       "getting file and module dependencies"));

static cl::opt<std::string>
    OutputDir("output-dir", cl::desc("directory for module output files "
                                     "(defaults 'module-outputs')"));
static cl::opt<bool>
    SerializeDiags("serialize-diagnostics",
                   cl::desc("module builds should serialize diagnostics"));
static cl::opt<bool>
    DependencyFile("dependency-file",
                   cl::desc("module builds should write dependency files"));
static cl::list<std::string> DependencyTargets(
    "dependency-target",
    cl::desc("module builds should use the given dependency target(s)"));
static llvm::cl::opt<std::string>
    CASPath("cas-path", llvm::cl::desc("Path for on-disk CAS/cache."));
static llvm::cl::opt<std::string>
    CASPluginPath("fcas-plugin-path", llvm::cl::desc("Path for CAS plugin"));
static cl::list<std::string> CASPluginOpts("fcas-plugin-option",
                                           cl::desc("Plugin CAS Options"));
static llvm::cl::opt<std::string>
    WorkingDir("working-dir", llvm::cl::desc("Path for working directory"));
static cl::opt<bool> TestCASCancellation(
    "test-cas-cancellation",
    cl::desc(
        "perform extra CAS API invocation and cancel it for testing purposes"));
static cl::opt<bool> OptimizeCWD(
    "optimize-cwd",
    cl::desc(
        "instruct the scanner to ignore current working directory if safe."));
} // namespace options
} // anonymous namespace

static void printSymbolInfo(SymbolInfo SymInfo, raw_ostream &OS);
static void printSymbolNameAndUSR(const Decl *D, ASTContext &Ctx,
                                  raw_ostream &OS);
static void printSymbolNameAndUSR(const clang::Module *Mod, raw_ostream &OS);

namespace {

class PrintIndexDataConsumer : public IndexDataConsumer {
  raw_ostream &OS;
  std::unique_ptr<ASTNameGenerator> ASTNameGen;
  std::shared_ptr<Preprocessor> PP;

public:
  PrintIndexDataConsumer(raw_ostream &OS) : OS(OS) {
  }

  void initialize(ASTContext &Ctx) override {
    ASTNameGen.reset(new ASTNameGenerator(Ctx));
  }

  void setPreprocessor(std::shared_ptr<Preprocessor> PP) override {
    this->PP = std::move(PP);
  }

  bool handleDeclOccurrence(const Decl *D, SymbolRoleSet Roles,
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

    if (ASTNameGen->writeName(D, OS))
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

  bool handleModuleOccurrence(const ImportDecl *ImportD,
                              const clang::Module *Mod, SymbolRoleSet Roles,
                              SourceLocation Loc) override {
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

  bool handleMacroOccurrence(const IdentifierInfo *Name, const MacroInfo *MI,
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
                               bool dumpModuleImports, bool indexLocals,
                               bool ignoreMacros) {
  SmallVector<const char *, 4> ArgsWithProgName;
  ArgsWithProgName.push_back(Executable);
  ArgsWithProgName.append(Args.begin(), Args.end());
  auto DiagOpts = std::make_shared<DiagnosticOptions>();
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags(
      CompilerInstance::createDiagnostics(*llvm::vfs::getRealFileSystem(),
                                          *DiagOpts));
  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;
  CIOpts.ProbePrecompiled = true; // FIXME: historical default. Needed?
  auto CInvok = createInvocation(ArgsWithProgName, std::move(CIOpts));
  if (!CInvok)
    return true;

  raw_ostream &OS = outs();
  auto DataConsumer = std::make_shared<PrintIndexDataConsumer>(OS);
  IndexingOptions IndexOpts;
  IndexOpts.IndexFunctionLocals = indexLocals;
  IndexOpts.IndexMacros = !ignoreMacros;
  IndexOpts.IndexMacrosInPreprocessor = !ignoreMacros;
  std::unique_ptr<FrontendAction> IndexAction =
      createIndexingAction(DataConsumer, IndexOpts);

  auto PCHContainerOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<ASTUnit> Unit(ASTUnit::LoadFromCompilerInvocationAction(
      std::move(CInvok), PCHContainerOps, DiagOpts, Diags, IndexAction.get()));

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
  pchContOps->registerReader(std::make_unique<ObjectFilePCHContainerReader>());
  auto pchRdr = pchContOps->getReaderOrNull(format);
  if (!pchRdr) {
    errs() << "unknown module format: " << format << '\n';
    return true;
  }

  HeaderSearchOptions HSOpts;

  auto VFS = llvm::vfs::getRealFileSystem();

  auto DiagOpts = std::make_shared<DiagnosticOptions>();
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*VFS, *DiagOpts);
  std::unique_ptr<ASTUnit> AU = ASTUnit::LoadFromASTFile(
      modulePath, *pchRdr, ASTUnit::LoadASTOnly, VFS, DiagOpts, Diags,
      FileSystemOpts, HSOpts, /*LangOpts=*/nullptr,
      /*OnlyLocalDecls=*/true, CaptureDiagsKind::None,
      /*AllowASTWithCompilerErrors=*/true,
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
}

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

static int printStoreRecords(StringRef StorePath, PathRemapper Remapper,
                             raw_ostream &OS) {
  std::string Error;
  indexstore::IndexStore Store(StorePath, Remapper, Error);
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
          recName = std::string(Dep.getName());
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
                                std::optional<unsigned> lineStart, unsigned lineCount,
                                PathRemapper remapper, raw_ostream &OS) {
  std::string error;
  indexstore::IndexStore store(storePath, remapper, error);
  if (!store) {
    errs() << "error loading store: " << error << "\n";
    return 1;
  }

  std::string recName = findRecordNameForFile(store, filePath);
  if (recName.empty()) {
    errs() << "could not find record for '" << filePath << "'\n";
    return 1;
  }

  if (!lineStart)
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

static int printUnit(StringRef Filename, PathRemapper Remapper,
                     raw_ostream &OS) {
  std::string Error;
  auto Reader = IndexUnitReader::createWithFilePath(Filename, Remapper, Error);
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
}

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

static int printStoreUnits(StringRef StorePath, PathRemapper Remapper,
                           raw_ostream &OS) {
  std::string Error;
  indexstore::IndexStore Store(StorePath, Remapper, Error);
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

static int scanDeps(ArrayRef<const char *> Args, std::string WorkingDirectory,
                    bool SerializeDiags, bool DependencyFile,
                    ArrayRef<std::string> DepTargets, std::string OutputPath,
                    CXCASDatabases DBs,
                    std::optional<std::string> ModuleName = std::nullopt) {
  CXDependencyScannerServiceOptions Opts =
      clang_experimental_DependencyScannerServiceOptions_create();
  auto CleanupOpts = llvm::make_scope_exit([&] {
    clang_experimental_DependencyScannerServiceOptions_dispose(Opts);
  });
  clang_experimental_DependencyScannerServiceOptions_setDependencyMode(
      Opts, CXDependencyMode_Full);

  if (options::OptimizeCWD)
    clang_experimental_DependencyScannerServiceOptions_setCWDOptimization(Opts,
                                                                          1);

  if (DBs)
    clang_experimental_DependencyScannerServiceOptions_setCASDatabases(Opts,
                                                                       DBs);

  CXDependencyScannerService Service =
      clang_experimental_DependencyScannerService_create_v1(Opts);
  CXDependencyScannerWorker Worker =
      clang_experimental_DependencyScannerWorker_create_v0(Service);
  auto DisposeWorkerAndService = llvm::make_scope_exit([&]() {
    clang_experimental_DependencyScannerWorker_dispose_v0(Worker);
    clang_experimental_DependencyScannerService_dispose_v0(Service);
  });

  auto LookupOutput = [&](const char *ModuleName, const char *ContextHash,
                          CXOutputKind Kind, char *Output, size_t MaxLen) {
    std::string Out = OutputPath + "/" + ModuleName + "_" + ContextHash;
    switch (Kind) {
    case CXOutputKind_ModuleFile:
      Out += ".pcm";
      break;
    case CXOutputKind_Dependencies:
      if (!DependencyFile)
        return (size_t)0;
      Out += ".d";
      break;
    case CXOutputKind_DependenciesTarget:
      if (DepTargets.empty())
        return (size_t)0;
      Out = join(DepTargets, StringRef("\0", 1));
      break;
    case CXOutputKind_SerializedDiagnostics:
      if (!SerializeDiags)
        return (size_t)0;
      Out += ".diag";
      break;
    }
    if (0 < Out.size() && Out.size() <= MaxLen)
      memcpy(Output, Out.data(), Out.size());
    return Out.size();
  };

  auto LookupOutputCB = functionObjectToCCallbackRef<size_t(
      const char *ModuleName, const char *ContextHash, CXOutputKind Kind,
      char *Output, size_t MaxLen)>(LookupOutput);

  unsigned CommandIndex = 0;
  auto HandleCommand = [&](const char *ContextHash,
                           const char *FilesystemRootID,
                           const char *IncludeTreeID, CXCStringArray ModuleDeps,
                           CXCStringArray FileDeps, CXCStringArray Args,
                           const char *CacheKey) {
    llvm::outs() << "  command " << CommandIndex++ << ":\n";
    llvm::outs() << "    context-hash: " << ContextHash << "\n";
    if (FilesystemRootID)
      llvm::outs() << "    casfs-root-id: " << FilesystemRootID << "\n";
    if (IncludeTreeID)
      llvm::outs() << "    include-tree-id: " << IncludeTreeID << "\n";
    if (CacheKey)
      llvm::outs() << "    cache-key: " << CacheKey << "\n";
    llvm::outs() << "    module-deps:\n";
    for (const auto &ModuleName :
         ArrayRef(ModuleDeps.Strings, ModuleDeps.Count))
      llvm::outs() << "      " << ModuleName << "\n";
    llvm::outs() << "    file-deps:\n";
    for (const auto &FileName : ArrayRef(FileDeps.Strings, FileDeps.Count))
      llvm::outs() << "      " << FileName << "\n";
    llvm::outs() << "    build-args:";
    for (const auto &Arg : ArrayRef(Args.Strings, Args.Count))
      llvm::outs() << " " << Arg;
    llvm::outs() << "\n";
  };

  CXDependencyScannerWorkerScanSettings ScanSettings =
      clang_experimental_DependencyScannerWorkerScanSettings_create(
          Args.size(), Args.data(), ModuleName ? ModuleName->c_str() : nullptr,
          WorkingDirectory.c_str(), LookupOutputCB.Context,
          LookupOutputCB.Callback);
  auto DisposeScanSettings = llvm::make_scope_exit([&]() {
    clang_experimental_DependencyScannerWorkerScanSettings_dispose(
        ScanSettings);
  });
  CXDepGraph Graph = nullptr;
  auto DisposeDepGraph = llvm::make_scope_exit(
      [&]() { clang_experimental_DepGraph_dispose(Graph); });
  CXErrorCode Err = clang_experimental_DependencyScannerWorker_getDepGraph(
      Worker, ScanSettings, &Graph);

  if (Err == CXError_Success) {
    llvm::outs() << "modules:\n";
    for (size_t I = 0, E = clang_experimental_DepGraph_getNumModules(Graph);
         I < E; ++I) {
      CXDepGraphModule Mod =
          clang_experimental_DepGraph_getModuleTopological(Graph, I);
      const char *Name = clang_experimental_DepGraphModule_getName(Mod);
      const char *ContextHash =
          clang_experimental_DepGraphModule_getContextHash(Mod);
      int CwdIgnored = clang_experimental_DepGraphModule_isCWDIgnored(Mod);
      bool IsInStableDir =
          clang_experimental_DepGraphModule_isInStableDirs(Mod);
      const char *ModuleMapPath =
          clang_experimental_DepGraphModule_getModuleMapPath(Mod);
      const char *ModuleFilesystemRootID =
          clang_experimental_DepGraphModule_getFileSystemRootID(Mod);
      const char *ModuleIncludeTreeID =
          clang_experimental_DepGraphModule_getIncludeTreeID(Mod);
      const char *ModuleCacheKey =
          clang_experimental_DepGraphModule_getCacheKey(Mod);
      CXCStringArray ModuleDeps =
          clang_experimental_DepGraphModule_getModuleDeps(Mod);
      CXCStringArray FileDeps =
          clang_experimental_DepGraphModule_getFileDeps(Mod);
      CXCStringArray BuildArguments =
          clang_experimental_DepGraphModule_getBuildArguments(Mod);
      auto Dispose = llvm::make_scope_exit(
          [&]() { clang_experimental_DepGraphModule_dispose(Mod); });
      llvm::outs() << "  module:\n"
                   << "    name: " << Name << "\n"
                   << "    context-hash: " << ContextHash << "\n"
                   << "    cwd-ignored: " << CwdIgnored << "\n"
                   << "    module-map-path: "
                   << (ModuleMapPath ? ModuleMapPath : "<none>") << "\n";
      if (IsInStableDir)
        llvm::outs() << "    is-in-stable-directories: " << IsInStableDir
                     << "\n";
      if (ModuleFilesystemRootID)
        llvm::outs() << "    casfs-root-id: " << ModuleFilesystemRootID << "\n";
      if (ModuleIncludeTreeID)
        llvm::outs() << "    include-tree-id: " << ModuleIncludeTreeID << "\n";
      if (ModuleCacheKey)
        llvm::outs() << "    cache-key: " << ModuleCacheKey << "\n";
      llvm::outs() << "    module-deps:\n";
      for (const auto &ModuleName :
           ArrayRef(ModuleDeps.Strings, ModuleDeps.Count))
        llvm::outs() << "      " << ModuleName << "\n";
      llvm::outs() << "    file-deps:\n";
      for (const auto &FileName : ArrayRef(FileDeps.Strings, FileDeps.Count))
        // Not reporting SDKSettings.json so that test checks can remain
        // (mostly) platform-agnostic.
        if (!StringRef(FileName).ends_with("SDKSettings.json"))
          llvm::outs() << "      " << FileName << "\n";
      CXDepGraphModuleLinkLibrarySet LinkLibs =
          clang_experimental_DepGraphModule_getLinkLibrarySet(Mod);
      size_t NumLinkLibs =
          clang_experimental_DepGraphModuleLinkLibrarySet_getSize(LinkLibs);
      if (NumLinkLibs) {
        llvm::outs() << "    link libraries:\n";
        for (size_t Idx = 0; Idx < NumLinkLibs; Idx++) {
          CXDepGraphModuleLinkLibrary Lib =
              clang_experimental_DepGraphModuleLinkLibrarySet_getLinkLibrary(
                  LinkLibs, Idx);
          CXString Library =
              clang_experimental_DepGraphModuleLinkLibrary_getLibrary(Lib);
          bool IsFramework =
              clang_experimental_DepGraphModuleLinkLibrary_isFramework(Lib);
          const char *IsFrameworkStr = IsFramework ? "(framework)" : "";
          llvm::outs() << "        " << clang_getCString(Library)
                       << IsFrameworkStr << "\n";
        }
      }
      llvm::outs() << "    build-args:";
      for (const auto &Arg :
           ArrayRef(BuildArguments.Strings, BuildArguments.Count))
        llvm::outs() << " " << Arg;
      llvm::outs() << "\n";
    }

    llvm::outs() << "dependencies:\n";
    const char *TUFileSystemRootID =
        clang_experimental_DepGraph_getTUFileSystemRootID(Graph);
    const char *TUIncludeTreeID =
        clang_experimental_DepGraph_getTUIncludeTreeID(Graph);
    const char *TUContextHash =
        clang_experimental_DepGraph_getTUContextHash(Graph);
    CXCStringArray TUModuleDeps =
        clang_experimental_DepGraph_getTUModuleDeps(Graph);
    CXCStringArray TUFileDeps =
        clang_experimental_DepGraph_getTUFileDeps(Graph);
    for (size_t I = 0, E = clang_experimental_DepGraph_getNumTUCommands(Graph);
         I < E; ++I) {
      CXDepGraphTUCommand Cmd =
          clang_experimental_DepGraph_getTUCommand(Graph, I);
      CXCStringArray Args =
          clang_experimental_DepGraphTUCommand_getBuildArguments(Cmd);
      const char *CacheKey =
          clang_experimental_DepGraphTUCommand_getCacheKey(Cmd);
      auto Dispose = llvm::make_scope_exit(
          [&]() { clang_experimental_DepGraphTUCommand_dispose(Cmd); });
      HandleCommand(TUContextHash, TUFileSystemRootID, TUIncludeTreeID,
                    TUModuleDeps, TUFileDeps, Args, CacheKey);
    }
    return 0;
  }
  llvm::errs() << "error: failed to get dependencies\n";
  CXDiagnosticSet Diags = clang_experimental_DepGraph_getDiagnostics(Graph);
  auto DisposeDiagnosticSet =
      llvm::make_scope_exit([&]() { clang_disposeDiagnosticSet(Diags); });
  for (unsigned I = 0, N = clang_getNumDiagnosticsInSet(Diags); I < N; ++I) {
    CXDiagnostic Diag = clang_getDiagnosticInSet(Diags, I);
    CXString Spelling =
        clang_formatDiagnostic(Diag, clang_defaultDiagnosticDisplayOptions());
    llvm::errs() << clang_getCString(Spelling) << "\n";
    clang_disposeString(Spelling);
    clang_disposeDiagnostic(Diag);
  }

  CXDepScanFSOutOfDateEntrySet OutOfDateEntrySet =
      clang_experimental_DependencyScannerService_getFSCacheOutOfDateEntrySet(
          Service);

  llvm::errs()
      << "note: number of out of date file system cache entries: "
      << clang_experimental_DepScanFSCacheOutOfDateEntrySet_getNumOfEntries(
             OutOfDateEntrySet)
      << "\n";

  clang_experimental_DepScanFSCacheOutOfDateEntrySet_disposeSet(
      OutOfDateEntrySet);

  return 1;
}

static int generateDepsReproducer(ArrayRef<const char *> Args,
                                  std::string WorkingDirectory,
                                  std::string ReproLocation) {
  CXDependencyScannerReproducerOptions Opts =
      clang_experimental_DependencyScannerReproducerOptions_create(
          Args.size(), Args.data(), /*ModuleName=*/nullptr,
          WorkingDirectory.c_str(),
          ReproLocation.empty() ? nullptr : ReproLocation.c_str(),
          /*UseUniqueReproducerName=*/ReproLocation.empty());
  auto DisposeOpts = llvm::make_scope_exit([&] {
    clang_experimental_DependencyScannerReproducerOptions_dispose(Opts);
  });
  CXString MessageString;
  auto DisposeMessageString = llvm::make_scope_exit([&]() {
    clang_disposeString(MessageString);
  });
  CXErrorCode ExitCode =
      clang_experimental_DependencyScanner_generateReproducer(Opts,
                                                              &MessageString);
  if (ExitCode == CXError_Success) {
    llvm::outs() << clang_getCString(MessageString) << "\n";
  } else {
    llvm::errs() << "error: " << clang_getCString(MessageString) << "\n";
  }
  return (ExitCode == CXError_Success) ? 0 : 1;
}

static int uploadCachedJob(std::string CacheKey, CXCASDatabases DBs) {
  CXError Err = nullptr;
  CXCASCachedCompilation CComp = clang_experimental_cas_getCachedCompilation(
      DBs, CacheKey.c_str(), /*Globally*/ false, &Err);
  auto CleanupCachedComp = llvm::make_scope_exit(
      [&] { clang_experimental_cas_CachedCompilation_dispose(CComp); });
  if (!CComp) {
    if (Err) {
      llvm::errs() << clang_Error_getDescription(Err) << "\n";
      clang_Error_dispose(Err);
    } else {
      llvm::errs() << "cache key was not found\n";
    }
    return 1;
  }

  /// \returns true of an error occurred.
  auto invokeMakeGlobal = [&](bool Cancel) -> bool {
    CXCASCancellationToken CancelToken = nullptr;
    auto CleanupCancelTok = llvm::make_scope_exit([&] {
      if (Cancel)
        clang_experimental_cas_CancellationToken_dispose(CancelToken);
    });

    std::promise<CXError> CallPromise;
    clang_experimental_cas_CachedCompilation_makeGlobal(
        CComp, &CallPromise,
        [](void *Ctx, CXError Err) {
          static_cast<std::promise<CXError> *>(Ctx)->set_value(Err);
        },
        Cancel ? &CancelToken : nullptr);
    if (Cancel) {
      clang_experimental_cas_CancellationToken_cancel(CancelToken);
    }
    CXError CallRes = CallPromise.get_future().get();
    if (CallRes) {
      llvm::errs() << clang_Error_getDescription(CallRes) << "\n";
      return true;
    }
    return false;
  };

  if (options::TestCASCancellation) {
    // Cancel an invocation for testing purposes.
    if (invokeMakeGlobal(/*Cancel=*/true))
      return 1;
  }
  if (invokeMakeGlobal(/*Cancel=*/false))
    return 1;

  return 0;
}

static int materializeCachedJob(std::string CacheKey, CXCASDatabases DBs) {
  /// \returns true of an error occurred.
  auto invokeGetCachedCompilation =
      [&](bool Cancel, CXCASCachedCompilation &OutComp) -> bool {
    OutComp = nullptr;
    CXCASCancellationToken CancelToken = nullptr;
    auto CleanupCancelTok = llvm::make_scope_exit([&] {
      if (Cancel)
        clang_experimental_cas_CancellationToken_dispose(CancelToken);
    });

    struct CompResult {
      CXCASCachedCompilation Comp = nullptr;
      CXError Err = nullptr;
    };
    std::promise<CompResult> CompPromise;
    auto CompFuture = CompPromise.get_future();
    struct CompCall {
      std::promise<CompResult> Promise;
    };
    CompCall *CallCtx = new CompCall{std::move(CompPromise)};
    clang_experimental_cas_getCachedCompilation_async(
        DBs, CacheKey.c_str(), /*Globally*/ true, CallCtx,
        [](void *Ctx, CXCASCachedCompilation Comp, CXError Err) {
          std::unique_ptr<CompCall> CallCtx(static_cast<CompCall *>(Ctx));
          CallCtx->Promise.set_value(CompResult{Comp, Err});
        },
        Cancel ? &CancelToken : nullptr);
    if (Cancel) {
      clang_experimental_cas_CancellationToken_cancel(CancelToken);
    }
    CompResult Res = CompFuture.get();
    OutComp = Res.Comp;
    if (!OutComp && !Cancel) {
      if (Res.Err) {
        llvm::errs() << clang_Error_getDescription(Res.Err) << "\n";
        clang_Error_dispose(Res.Err);
      } else {
        llvm::errs() << "cache key was not found\n";
      }
      return true;
    }
    return false;
  };

  CXCASCachedCompilation CComp = nullptr;
  auto CleanupCachedComp = llvm::make_scope_exit([&] {
    if (CComp)
      clang_experimental_cas_CachedCompilation_dispose(CComp);
  });

  if (options::TestCASCancellation) {
    // Cancel an invocation for testing purposes.
    if (invokeGetCachedCompilation(/*Cancel=*/true, CComp))
      return 1;
  }
  if (invokeGetCachedCompilation(/*Cancel=*/false, CComp))
    return 1;

  for (unsigned
           I = 0,
           E = clang_experimental_cas_CachedCompilation_getNumOutputs(CComp);
       I != E; ++I) {
    if (clang_experimental_cas_CachedCompilation_isOutputMaterialized(CComp, I))
      continue;
    CXString OutputID =
        clang_experimental_cas_CachedCompilation_getOutputCASIDString(CComp, I);
    auto CleanupOutputID =
        llvm::make_scope_exit([&] { clang_disposeString(OutputID); });

    /// \returns true of an error occurred.
    auto invokeLoadObject = [&](bool Cancel, CXCASObject &OutObj) -> bool {
      OutObj = nullptr;
      CXCASCancellationToken CancelToken = nullptr;
      auto CleanupCancelTok = llvm::make_scope_exit([&] {
        if (Cancel)
          clang_experimental_cas_CancellationToken_dispose(CancelToken);
      });

      struct LoadResult {
        CXCASObject Obj = nullptr;
        CXError Err = nullptr;
      };
      std::promise<LoadResult> LoadPromise;
      auto LoadFuture = LoadPromise.get_future();
      struct LoadCall {
        std::promise<LoadResult> Promise;
      };
      LoadCall *CallCtx = new LoadCall{std::move(LoadPromise)};
      clang_experimental_cas_loadObjectByString_async(
          DBs, clang_getCString(OutputID), CallCtx,
          [](void *Ctx, CXCASObject Obj, CXError Err) {
            std::unique_ptr<LoadCall> CallCtx(static_cast<LoadCall *>(Ctx));
            CallCtx->Promise.set_value(LoadResult{Obj, Err});
          },
          Cancel ? &CancelToken : nullptr);
      if (Cancel) {
        clang_experimental_cas_CancellationToken_cancel(CancelToken);
      }
      LoadResult Res = LoadFuture.get();
      OutObj = Res.Obj;
      if (!OutObj && !Cancel) {
        if (Res.Err) {
          llvm::errs() << clang_Error_getDescription(Res.Err) << "\n";
          clang_Error_dispose(Res.Err);
        } else {
          llvm::errs() << "cache key was not found\n";
        }
        return true;
      }
      return false;
    };

    CXCASObject CASObj = nullptr;
    auto CleanupLoadObj = llvm::make_scope_exit([&] {
      if (CASObj)
        clang_experimental_cas_CASObject_dispose(CASObj);
    });

    if (options::TestCASCancellation) {
      // Cancel an invocation for testing purposes.
      if (invokeLoadObject(/*Cancel=*/true, CASObj))
        return 1;
    }
    if (invokeLoadObject(/*Cancel=*/false, CASObj))
      return 1;

    if (!clang_experimental_cas_CachedCompilation_isOutputMaterialized(CComp,
                                                                       I))
      report_fatal_error("output was not materialized?");
  }

  return 0;
}

static int replayCachedJob(ArrayRef<const char *> Args,
                           std::string WorkingDirectory, std::string CacheKey,
                           CXCASDatabases DBs) {
  CXError Err = nullptr;
  CXCASCachedCompilation CComp = clang_experimental_cas_getCachedCompilation(
      DBs, CacheKey.c_str(), /*Globally*/ false, &Err);
  auto CleanupCachedComp = llvm::make_scope_exit(
      [&] { clang_experimental_cas_CachedCompilation_dispose(CComp); });
  if (!CComp) {
    if (Err) {
      llvm::errs() << clang_Error_getDescription(Err) << "\n";
      clang_Error_dispose(Err);
    } else {
      llvm::errs() << "cache key was not found\n";
    }
    return 1;
  }

  CXCASReplayResult ReplayRes = clang_experimental_cas_replayCompilation(
      CComp, Args.size(), Args.data(),
      WorkingDirectory.empty() ? nullptr : WorkingDirectory.c_str(),
      /*reserved*/ nullptr, &Err);
  auto CleanupReplayRes = llvm::make_scope_exit(
      [&] { clang_experimental_cas_ReplayResult_dispose(ReplayRes); });
  if (!ReplayRes) {
    llvm::errs() << clang_Error_getDescription(Err) << "\n";
    clang_Error_dispose(Err);
    return 1;
  }

  CXString DiagText = clang_experimental_cas_ReplayResult_getStderr(ReplayRes);
  llvm::errs() << clang_getCString(DiagText);
  clang_disposeString(DiagText);
  return 0;
}

static int pruneCAS(int64_t Limit, CXCASDatabases DBs) {
  CXError Err = nullptr;
  int64_t Size = clang_experimental_cas_Databases_get_storage_size(DBs, &Err);
  if (Size == -2) {
    llvm::errs() << "clang_experimental_cas_Databases_get_storage_size: "
                 << clang_Error_getDescription(Err) << "\n";
    clang_Error_dispose(Err);
    return 1;
  }
  if (Size == -1) {
    llvm::errs()
        << "unsupported clang_experimental_cas_Databases_get_storage_size";
    return 1;
  }
  if (Size == 0) {
    llvm::errs()
        << "clang_experimental_cas_Databases_get_storage_size returned 0";
    return 1;
  }

  if (CXError Err =
          clang_experimental_cas_Databases_set_size_limit(DBs, Limit)) {
    llvm::errs() << "clang_experimental_cas_Databases_set_size_limit: "
                 << clang_Error_getDescription(Err) << "\n";
    clang_Error_dispose(Err);
    return 1;
  }
  if (CXError Err = clang_experimental_cas_Databases_prune_ondisk_data(DBs)) {
    llvm::errs() << "clang_experimental_cas_Databases_prune_ondisk_data: "
                 << clang_Error_getDescription(Err) << "\n";
    clang_Error_dispose(Err);
    return 1;
  }

  return 0;
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
        case DirectoryWatcher::Event::EventKind::Modified:
          OS << "modified: "; break;
        case DirectoryWatcher::Event::EventKind::Removed:
          OS << "removed: "; break;
        case DirectoryWatcher::Event::EventKind::WatchedDirRemoved:
          OS << "dir deleted: "; break;
        case DirectoryWatcher::Event::EventKind::WatcherGotInvalidated:
          OS << "watcher got invalidated: "; break;

      }
      OS << evt.Filename << '\n';
    }
  };
  auto watcher = DirectoryWatcher::create(dirPath, receiver,
                                          /*waitInitialSync=*/true);
  if (!watcher) {
    errs() << "failed creating directory watcher" << '\n';
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
                             std::optional<unsigned> &lineStart,
                             unsigned &lineCount) {
  StringRef path, start, end;
  std::tie(path, end) = input.rsplit(':');
  std::tie(path, start) = path.rsplit(':');
  filepath = std::string(path);
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
  lineCount = num - *lineStart;
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

  PathRemapper PathRemapper;
  for (const auto &Mapping : options::PrefixMap) {
    llvm::StringRef MappingRef(Mapping);
    if (!MappingRef.contains('=')) {
      errs() << "error: prefix map argument should be of form prefix=value,"
             << " but got: " << MappingRef << "\n";
      return 1;
    }
    auto Split = MappingRef.split('=');
    PathRemapper.addMapping(Split.first, Split.second);
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
                              options::IncludeLocals, options::IgnoreMacros);
  }

  if (options::Action == ActionType::PrintRecord) {
    if (!options::FilePathAndRange.empty()) {
      std::string filepath;
      std::optional<unsigned> lineStart;
      unsigned lineCount;
      if (deconstructPathAndRange(options::FilePathAndRange,
                                  filepath, lineStart, lineCount))
        return 1;

      if (options::InputFiles.empty()) {
        errs() << "error: missing index store path\n";
        return 1;
      }
      return printStoreFileRecord(options::InputFiles[0], filepath, lineStart,
                                  lineCount, PathRemapper, outs());
    }

    if (options::InputFiles.empty()) {
      errs() << "error: missing input file or directory\n";
      return 1;
    }

    if (sys::fs::is_directory(options::InputFiles[0]))
      return printStoreRecords(options::InputFiles[0], PathRemapper, outs());
    else
      return printRecord(options::InputFiles[0], outs());
  }

  if (options::Action == ActionType::PrintUnit) {
    if (options::InputFiles.empty()) {
      errs() << "error: missing input file or directory\n";
      return 1;
    }

    if (sys::fs::is_directory(options::InputFiles[0]))
      return printStoreUnits(options::InputFiles[0], PathRemapper, outs());
    else
      return printUnit(options::InputFiles[0], PathRemapper, outs());
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
      return aggregateDataAsJSON(storePath, PathRemapper, outs());
    std::error_code EC;
    raw_fd_ostream OS(options::OutputFile, EC, llvm::sys::fs::OF_None);
    if (EC) {
      errs() << "failed to open output file: " << EC.message() << '\n';
      return 1;
    }
    return aggregateDataAsJSON(storePath, PathRemapper, OS);
  }

  std::optional<std::string> CASPath = options::CASPath.empty()
                                      ? std::nullopt
                                      : std::optional<std::string>(options::CASPath);

  CXCASOptions CASOpts = nullptr;
  CXCASDatabases DBs = nullptr;
  auto CleanupCASOpts = llvm::make_scope_exit([&] {
    if (CASOpts)
      clang_experimental_cas_Options_dispose(CASOpts);
  });
  auto CleanupCaches = llvm::make_scope_exit([&] {
    if (DBs)
      clang_experimental_cas_Databases_dispose(DBs);
  });

  if (CASPath) {
    CASOpts = clang_experimental_cas_Options_create();
    clang_experimental_cas_Options_setOnDiskPath(CASOpts, CASPath->c_str());
    if (!options::CASPluginPath.empty())
      clang_experimental_cas_Options_setPluginPath(
          CASOpts, options::CASPluginPath.c_str());
    for (const auto &PluginOpt : options::CASPluginOpts) {
      auto [Name, Val] = StringRef(PluginOpt).split('=');
      std::string NameStr(Name);
      std::string ValStr(Val);
      clang_experimental_cas_Options_setPluginOption(CASOpts, NameStr.c_str(),
                                                     ValStr.c_str());
    }
    CXString Error;
    DBs = clang_experimental_cas_Databases_create(CASOpts, &Error);
    if (!DBs) {
      llvm::errs() << "error: failed to create cas/cache databases\n";
      llvm::errs() << clang_getCString(Error) << "\n";
      clang_disposeString(Error);
      return 1;
    }
  }

  if (options::Action == ActionType::ScanDeps) {
    if (options::WorkingDir.empty()) {
      errs() << "error: missing -working-dir\n";
      return 1;
    }
    return scanDeps(CompArgs, options::WorkingDir, options::SerializeDiags,
                    options::DependencyFile, options::DependencyTargets,
                    options::OutputDir, DBs);
  }

  if (options::Action == ActionType::ScanDepsByModuleName) {
    // InputFiles should be set to the working directory name.
    if (options::WorkingDir.empty()) {
      errs() << "error: missing -working-dir\n";
      return 1;
    }
    if (options::ModuleName.empty()) {
      errs() << "error: missing module name\n";
      return 1;
    }
    return scanDeps(CompArgs, options::WorkingDir, options::SerializeDiags,
                    options::DependencyFile, options::DependencyTargets,
                    options::OutputDir, DBs, options::ModuleName);
  }

  if (options::Action == ActionType::GenerateDepsReproducer) {
    if (options::WorkingDir.empty()) {
      errs() << "error: missing -working-dir\n";
      return 1;
    }
    return generateDepsReproducer(CompArgs, options::WorkingDir,
                                  options::OutputFile);
  }

  if (options::Action == ActionType::UploadCachedJob) {
    if (options::InputFiles.empty()) {
      errs() << "error: missing cache key\n";
      return 1;
    }
    if (!DBs) {
      errs() << "error: CAS was not configured\n";
      return 1;
    }
    return uploadCachedJob(options::InputFiles[0], DBs);
  }

  if (options::Action == ActionType::MaterializeCachedJob) {
    if (options::InputFiles.empty()) {
      errs() << "error: missing cache key\n";
      return 1;
    }
    if (!DBs) {
      errs() << "error: CAS was not configured\n";
      return 1;
    }
    return materializeCachedJob(options::InputFiles[0], DBs);
  }

  if (options::Action == ActionType::ReplayCachedJob) {
    if (options::InputFiles.empty()) {
      errs() << "error: missing cache key\n";
      return 1;
    }
    if (!DBs) {
      errs() << "error: CAS was not configured\n";
      return 1;
    }
    return replayCachedJob(CompArgs, options::WorkingDir,
                           options::InputFiles[0], DBs);
  }

  if (options::Action == ActionType::PruneCAS) {
    if (options::InputFiles.empty()) {
      errs() << "error: missing size limit\n";
      return 1;
    }
    int64_t Limit;
    if (StringRef(options::InputFiles[0]).getAsInteger(10, Limit)) {
      errs() << "error: size limit not an integer\n";
      return 1;
    }
    if (!DBs) {
      errs() << "error: CAS was not configured\n";
      return 1;
    }
    return pruneCAS(Limit, DBs);
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

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

int indextest_perform_shell_execution(const char *command_line) {
  BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  SmallVector<const char *, 4> Args;
  llvm::cl::TokenizeGNUCommandLine(command_line, Saver, Args);
  auto Program = llvm::sys::findProgramByName(Args[0]);
  if (std::error_code ec = Program.getError()) {
    llvm::errs() << "command not found: " << Args[0] << "\n";
    return ec.value();
  }
  SmallVector<StringRef, 8> execArgs(Args.begin(), Args.end());
  return llvm::sys::ExecuteAndWait(*Program, execArgs);
}
