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
#include "clang/Sema/CodeCompleteOptions.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace llvm {
class MemoryBuffer;
}

namespace clang {

namespace frontend {
  enum ActionKind {
    ASTDeclList,            ///< Parse ASTs and list Decl nodes.
    ASTDump,                ///< Parse ASTs and dump them.
    ASTDumpXML,             ///< Parse ASTs and dump them in XML.
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
    GenerateModule,         ///< Generate pre-compiled module.
    GeneratePCH,            ///< Generate pre-compiled header.
    GeneratePTH,            ///< Generate pre-tokenized header.
    InitOnly,               ///< Only execute frontend initialization.
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
  FrontendInputFile() : Buffer(0), Kind(IK_None) { }
  FrontendInputFile(StringRef File, InputKind Kind, bool IsSystem = false)
    : File(File.str()), Buffer(0), Kind(Kind), IsSystem(IsSystem) { }
  FrontendInputFile(llvm::MemoryBuffer *buffer, InputKind Kind,
                    bool IsSystem = false)
    : Buffer(buffer), Kind(Kind), IsSystem(IsSystem) { }

  InputKind getKind() const { return Kind; }
  bool isSystem() const { return IsSystem; }

  bool isEmpty() const { return File.empty() && Buffer == 0; }
  bool isFile() const { return !isBuffer(); }
  bool isBuffer() const { return Buffer != 0; }

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
    ObjCMT_Subscripting = 0x2
  };
  unsigned ObjCMTAction;

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

  /// Args to pass to the plugin
  std::vector<std::string> PluginArgs;

  /// The list of plugin actions to run in addition to the normal action.
  std::vector<std::string> AddPluginActions;

  /// Args to pass to the additional plugins
  std::vector<std::vector<std::string> > AddPluginArgs;

  /// The list of plugins to load.
  std::vector<std::string> Plugins;

  /// \brief The list of AST files to merge.
  std::vector<std::string> ASTMergeFiles;

  /// \brief A list of arguments to forward to LLVM's option processing; this
  /// should only be used for debugging and experimental features.
  std::vector<std::string> LLVMArgs;

  /// \brief File name of the file that will provide record layouts
  /// (in the format produced by -fdump-record-layouts).
  std::string OverrideRecordLayoutsFile;
  
public:
  FrontendOptions() {
    DisableFree = 0;
    ProgramAction = frontend::ParseSyntaxOnly;
    ActionName = "";
    RelocatablePCH = 0;
    ShowHelp = 0;
    ShowStats = 0;
    ShowTimers = 0;
    ShowVersion = 0;
    ARCMTAction = ARCMT_None;
    ARCMTMigrateEmitARCErrors = 0;
    SkipFunctionBodies = 0;
    ObjCMTAction = ObjCMT_None;
  }

  /// getInputKindForExtension - Return the appropriate input kind for a file
  /// extension. For example, "c" would return IK_C.
  ///
  /// \return The input kind for the extension, or IK_None if the extension is
  /// not recognized.
  static InputKind getInputKindForExtension(StringRef Extension);
};

}  // end namespace clang

#endif
