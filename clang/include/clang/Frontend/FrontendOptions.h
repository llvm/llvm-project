//===--- FrontendOptions.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H
#define LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H

#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/Serialization/ModuleFileExtension.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace llvm {
class MemoryBuffer;
}

namespace clang {

namespace frontend {
  enum ActionKind {
    ASTDeclList,            ///< Parse ASTs and list Decl nodes.
    ASTDump,                ///< Parse ASTs and dump them.
    ASTPrint,               ///< Parse ASTs and print them.
    ASTView,                ///< Parse ASTs and view them in Graphviz.
    DumpRawTokens,          ///< Dump out raw tokens.
    DumpTokens,             ///< Dump out preprocessed tokens.
    EmitAssembly,           ///< Emit a .s file.
    EmitBC,                 ///< Emit a .bc file.
    EmitHTML,               ///< Translate input source into HTML.
    EmitLLVM,               ///< Emit a .ll file.
    EmitLLVMOnly,           ///< Generate LLVM IR, but do not emit anything.
    EmitCodeGenOnly,        ///< Generate machine code, but don't emit anything.
    EmitObj,                ///< Emit a .o file.
    FixIt,                  ///< Parse and apply any fixits to the source.
    GenerateModule,         ///< Generate pre-compiled module from a module map.
    GenerateModuleInterface,///< Generate pre-compiled module from a C++ module
                            ///< interface file.
    GeneratePCH,            ///< Generate pre-compiled header.
    GeneratePTH,            ///< Generate pre-tokenized header.
    InitOnly,               ///< Only execute frontend initialization.
    ModuleFileInfo,         ///< Dump information about a module file.
    VerifyPCH,              ///< Load and verify that a PCH file is usable.
    ParseSyntaxOnly,        ///< Parse and perform semantic analysis.
    PluginAction,           ///< Run a plugin action, \see ActionName.
    PrintDeclContext,       ///< Print DeclContext and their Decls.
    PrintPreamble,          ///< Print the "preamble" of the input file
    PrintPreprocessedInput, ///< -E mode.
    RewriteMacros,          ///< Expand macros but not \#includes.
    RewriteObjC,            ///< ObjC->C Rewriter.
    RewriteTest,            ///< Rewriter playground
    RunAnalysis,            ///< Run one or more source code analyses.
    MigrateSource,          ///< Run migrator.
    RunPreprocessorOnly     ///< Just lex, no output.
  };
}

enum InputKind {
  IK_None,
  IK_Asm,
  IK_C,
  IK_CXX,
  IK_ObjC,
  IK_ObjCXX,
  IK_PreprocessedC,
  IK_PreprocessedCXX,
  IK_PreprocessedObjC,
  IK_PreprocessedObjCXX,
  IK_OpenCL,
  IK_CUDA,
  IK_PreprocessedCuda,
  IK_RenderScript,
  IK_AST,
  IK_LLVM_IR
};

  
/// \brief An input file for the front end.
class FrontendInputFile {
  /// \brief The file name, or "-" to read from standard input.
  std::string File;

  llvm::MemoryBuffer *Buffer;

  /// \brief The kind of input, e.g., C source, AST file, LLVM IR.
  InputKind Kind;

  /// \brief Whether we're dealing with a 'system' input (vs. a 'user' input).
  bool IsSystem;

public:
  FrontendInputFile() : Buffer(nullptr), Kind(IK_None), IsSystem(false) { }
  FrontendInputFile(StringRef File, InputKind Kind, bool IsSystem = false)
    : File(File.str()), Buffer(nullptr), Kind(Kind), IsSystem(IsSystem) { }
  FrontendInputFile(llvm::MemoryBuffer *buffer, InputKind Kind,
                    bool IsSystem = false)
    : Buffer(buffer), Kind(Kind), IsSystem(IsSystem) { }

  InputKind getKind() const { return Kind; }
  bool isSystem() const { return IsSystem; }

  bool isEmpty() const { return File.empty() && Buffer == nullptr; }
  bool isFile() const { return !isBuffer(); }
  bool isBuffer() const { return Buffer != nullptr; }

  StringRef getFile() const {
    assert(isFile());
    return File;
  }
  llvm::MemoryBuffer *getBuffer() const {
    assert(isBuffer());
    return Buffer;
  }
};

/// FrontendOptions - Options for controlling the behavior of the frontend.
class FrontendOptions {
public:
  unsigned DisableFree : 1;                ///< Disable memory freeing on exit.
  unsigned RelocatablePCH : 1;             ///< When generating PCH files,
                                           /// instruct the AST writer to create
                                           /// relocatable PCH files.
  unsigned ShowHelp : 1;                   ///< Show the -help text.
  unsigned ShowStats : 1;                  ///< Show frontend performance
                                           /// metrics and statistics.
  unsigned ShowTimers : 1;                 ///< Show timers for individual
                                           /// actions.
  unsigned ShowVersion : 1;                ///< Show the -version text.
  unsigned FixWhatYouCan : 1;              ///< Apply fixes even if there are
                                           /// unfixable errors.
  unsigned FixOnlyWarnings : 1;            ///< Apply fixes only for warnings.
  unsigned FixAndRecompile : 1;            ///< Apply fixes and recompile.
  unsigned FixToTemporaries : 1;           ///< Apply fixes to temporary files.
  unsigned ARCMTMigrateEmitARCErrors : 1;  /// Emit ARC errors even if the
                                           /// migrator can fix them
  unsigned SkipFunctionBodies : 1;         ///< Skip over function bodies to
                                           /// speed up parsing in cases you do
                                           /// not need them (e.g. with code
                                           /// completion).
  unsigned UseGlobalModuleIndex : 1;       ///< Whether we can use the
                                           ///< global module index if available.
  unsigned GenerateGlobalModuleIndex : 1;  ///< Whether we can generate the
                                           ///< global module index if needed.
  unsigned ASTDumpDecls : 1;               ///< Whether we include declaration
                                           ///< dumps in AST dumps.
  unsigned ASTDumpLookups : 1;             ///< Whether we include lookup table
                                           ///< dumps in AST dumps.
  unsigned BuildingImplicitModule : 1;     ///< Whether we are performing an
                                           ///< implicit module build.
  unsigned ModulesEmbedAllFiles : 1;       ///< Whether we should embed all used
                                           ///< files into the PCM file.
  unsigned IncludeTimestamps : 1;          ///< Whether timestamps should be
                                           ///< written to the produced PCH file.

  CodeCompleteOptions CodeCompleteOpts;

  enum {
    ARCMT_None,
    ARCMT_Check,
    ARCMT_Modify,
    ARCMT_Migrate
  } ARCMTAction;

  enum {
    ObjCMT_None = 0,
    /// \brief Enable migration to modern ObjC literals.
    ObjCMT_Literals = 0x1,
    /// \brief Enable migration to modern ObjC subscripting.
    ObjCMT_Subscripting = 0x2,
    /// \brief Enable migration to modern ObjC readonly property.
    ObjCMT_ReadonlyProperty = 0x4,
    /// \brief Enable migration to modern ObjC readwrite property.
    ObjCMT_ReadwriteProperty = 0x8,
    /// \brief Enable migration to modern ObjC property.
    ObjCMT_Property = (ObjCMT_ReadonlyProperty | ObjCMT_ReadwriteProperty),
    /// \brief Enable annotation of ObjCMethods of all kinds.
    ObjCMT_Annotation = 0x10,
    /// \brief Enable migration of ObjC methods to 'instancetype'.
    ObjCMT_Instancetype = 0x20,
    /// \brief Enable migration to NS_ENUM/NS_OPTIONS macros.
    ObjCMT_NsMacros = 0x40,
    /// \brief Enable migration to add conforming protocols.
    ObjCMT_ProtocolConformance = 0x80,
    /// \brief prefer 'atomic' property over 'nonatomic'.
    ObjCMT_AtomicProperty = 0x100,
    /// \brief annotate property with NS_RETURNS_INNER_POINTER
    ObjCMT_ReturnsInnerPointerProperty = 0x200,
    /// \brief use NS_NONATOMIC_IOSONLY for property 'atomic' attribute
    ObjCMT_NsAtomicIOSOnlyProperty = 0x400,
    /// \brief Enable inferring NS_DESIGNATED_INITIALIZER for ObjC methods.
    ObjCMT_DesignatedInitializer = 0x800,
    /// \brief Enable converting setter/getter expressions to property-dot syntx.
    ObjCMT_PropertyDotSyntax = 0x1000,
    ObjCMT_MigrateDecls = (ObjCMT_ReadonlyProperty | ObjCMT_ReadwriteProperty |
                           ObjCMT_Annotation | ObjCMT_Instancetype |
                           ObjCMT_NsMacros | ObjCMT_ProtocolConformance |
                           ObjCMT_NsAtomicIOSOnlyProperty |
                           ObjCMT_DesignatedInitializer),
    ObjCMT_MigrateAll = (ObjCMT_Literals | ObjCMT_Subscripting |
                         ObjCMT_MigrateDecls | ObjCMT_PropertyDotSyntax)
  };
  unsigned ObjCMTAction;
  std::string ObjCMTWhiteListPath;

  std::string MTMigrateDir;
  std::string ARCMTMigrateReportOut;

  /// The input files and their types.
  std::vector<FrontendInputFile> Inputs;

  /// The output file, if any.
  std::string OutputFile;

  /// If given, the new suffix for fix-it rewritten files.
  std::string FixItSuffix;

  /// If given, filter dumped AST Decl nodes by this substring.
  std::string ASTDumpFilter;

  /// If given, enable code completion at the provided location.
  ParsedSourceLocation CodeCompletionAt;

  /// The frontend action to perform.
  frontend::ActionKind ProgramAction;

  /// The name of the action to run when using a plugin action.
  std::string ActionName;

  /// Args to pass to the plugins
  std::unordered_map<std::string,std::vector<std::string>> PluginArgs;

  /// The list of plugin actions to run in addition to the normal action.
  std::vector<std::string> AddPluginActions;

  /// The list of plugins to load.
  std::vector<std::string> Plugins;

  /// The list of module file extensions.
  std::vector<IntrusiveRefCntPtr<ModuleFileExtension>> ModuleFileExtensions;

  /// \brief The list of module map files to load before processing the input.
  std::vector<std::string> ModuleMapFiles;

  /// \brief The list of additional prebuilt module files to load before
  /// processing the input.
  std::vector<std::string> ModuleFiles;

  /// \brief The list of files to embed into the compiled module file.
  std::vector<std::string> ModulesEmbedFiles;

  /// \brief The list of AST files to merge.
  std::vector<std::string> ASTMergeFiles;

  /// \brief A list of arguments to forward to LLVM's option processing; this
  /// should only be used for debugging and experimental features.
  std::vector<std::string> LLVMArgs;

  /// \brief File name of the file that will provide record layouts
  /// (in the format produced by -fdump-record-layouts).
  std::string OverrideRecordLayoutsFile;

  /// \brief Auxiliary triple for CUDA compilation.
  std::string AuxTriple;

  /// \brief If non-empty, search the pch input file as it was a header
  // included by this file.
  std::string FindPchSource;

  /// Filename to write statistics to.
  std::string StatsFile;

public:
  FrontendOptions() :
    DisableFree(false), RelocatablePCH(false), ShowHelp(false),
    ShowStats(false), ShowTimers(false), ShowVersion(false),
    FixWhatYouCan(false), FixOnlyWarnings(false), FixAndRecompile(false),
    FixToTemporaries(false), ARCMTMigrateEmitARCErrors(false),
    SkipFunctionBodies(false), UseGlobalModuleIndex(true),
    GenerateGlobalModuleIndex(true), ASTDumpDecls(false), ASTDumpLookups(false),
    BuildingImplicitModule(false), ModulesEmbedAllFiles(false),
    IncludeTimestamps(true), ARCMTAction(ARCMT_None),
    ObjCMTAction(ObjCMT_None), ProgramAction(frontend::ParseSyntaxOnly)
  {}

  /// getInputKindForExtension - Return the appropriate input kind for a file
  /// extension. For example, "c" would return IK_C.
  ///
  /// \return The input kind for the extension, or IK_None if the extension is
  /// not recognized.
  static InputKind getInputKindForExtension(StringRef Extension);
};

}  // end namespace clang

#endif
