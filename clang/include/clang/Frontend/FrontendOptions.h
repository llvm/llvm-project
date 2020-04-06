//===- FrontendOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H
#define LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H

#include "clang/AST/ASTDumperUtils.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Serialization/ModuleFileExtension.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {

class MemoryBuffer;

} // namespace llvm

namespace clang {

namespace frontend {

enum ActionKind {
  /// Parse ASTs and list Decl nodes.
  ASTDeclList,

  /// Parse ASTs and dump them.
  ASTDump,

  /// Parse ASTs and print them.
  ASTPrint,

  /// Parse ASTs and view them in Graphviz.
  ASTView,

  /// Dump the compiler configuration.
  DumpCompilerOptions,

  /// Dump out raw tokens.
  DumpRawTokens,

  /// Dump out preprocessed tokens.
  DumpTokens,

  /// Emit a .s file.
  EmitAssembly,

  /// Emit a .bc file.
  EmitBC,

  /// Translate input source into HTML.
  EmitHTML,

  /// Emit a .ll file.
  EmitLLVM,

  /// Generate LLVM IR, but do not emit anything.
  EmitLLVMOnly,

  /// Generate machine code, but don't emit anything.
  EmitCodeGenOnly,

  /// Emit a .o file.
  EmitObj,

  /// Parse and apply any fixits to the source.
  FixIt,

  /// Generate pre-compiled module from a module map.
  GenerateModule,

  /// Generate pre-compiled module from a C++ module interface file.
  GenerateModuleInterface,

  /// Generate pre-compiled module from a set of header files.
  GenerateHeaderModule,

  /// Generate pre-compiled header.
  GeneratePCH,

  /// Generate Interface Stub Files.
  GenerateInterfaceStubs,

  /// Only execute frontend initialization.
  InitOnly,

  /// Dump information about a module file.
  ModuleFileInfo,

  /// Load and verify that a PCH file is usable.
  VerifyPCH,

  /// Parse and perform semantic analysis.
  ParseSyntaxOnly,

  /// Run a plugin action, \see ActionName.
  PluginAction,

  /// Print the "preamble" of the input file
  PrintPreamble,

  /// -E mode.
  PrintPreprocessedInput,

  /// Expand macros but not \#includes.
  RewriteMacros,

  /// ObjC->C Rewriter.
  RewriteObjC,

  /// Rewriter playground
  RewriteTest,

  /// Run one or more source code analyses.
  RunAnalysis,

  /// Dump template instantiations
  TemplightDump,

  /// Run migrator.
  MigrateSource,

  /// Just lex, no output.
  RunPreprocessorOnly,

  /// Print the output of the dependency directives source minimizer.
  PrintDependencyDirectivesSourceMinimizerOutput
};

} // namespace frontend

/// The kind of a file that we've been handed as an input.
class InputKind {
private:
  Language Lang;
  unsigned Fmt : 3;
  unsigned Preprocessed : 1;

public:
  /// The input file format.
  enum Format {
    Source,
    ModuleMap,
    Precompiled
  };

  constexpr InputKind(Language L = Language::Unknown, Format F = Source,
                      bool PP = false)
      : Lang(L), Fmt(F), Preprocessed(PP) {}

  Language getLanguage() const { return static_cast<Language>(Lang); }
  Format getFormat() const { return static_cast<Format>(Fmt); }
  bool isPreprocessed() const { return Preprocessed; }

  /// Is the input kind fully-unknown?
  bool isUnknown() const { return Lang == Language::Unknown && Fmt == Source; }

  /// Is the language of the input some dialect of Objective-C?
  bool isObjectiveC() const {
    return Lang == Language::ObjC || Lang == Language::ObjCXX;
  }

  InputKind getPreprocessed() const {
    return InputKind(getLanguage(), getFormat(), true);
  }

  InputKind withFormat(Format F) const {
    return InputKind(getLanguage(), F, isPreprocessed());
  }
};

/// An input file for the front end.
class FrontendInputFile {
  /// The file name, or "-" to read from standard input.
  std::string File;

  /// The input, if it comes from a buffer rather than a file. This object
  /// does not own the buffer, and the caller is responsible for ensuring
  /// that it outlives any users.
  const llvm::MemoryBuffer *Buffer = nullptr;

  /// The kind of input, e.g., C source, AST file, LLVM IR.
  InputKind Kind;

  /// Whether we're dealing with a 'system' input (vs. a 'user' input).
  bool IsSystem = false;

public:
  FrontendInputFile() = default;
  FrontendInputFile(StringRef File, InputKind Kind, bool IsSystem = false)
      : File(File.str()), Kind(Kind), IsSystem(IsSystem) {}
  FrontendInputFile(const llvm::MemoryBuffer *Buffer, InputKind Kind,
                    bool IsSystem = false)
      : Buffer(Buffer), Kind(Kind), IsSystem(IsSystem) {}

  InputKind getKind() const { return Kind; }
  bool isSystem() const { return IsSystem; }

  bool isEmpty() const { return File.empty() && Buffer == nullptr; }
  bool isFile() const { return !isBuffer(); }
  bool isBuffer() const { return Buffer != nullptr; }
  bool isPreprocessed() const { return Kind.isPreprocessed(); }

  StringRef getFile() const {
    assert(isFile());
    return File;
  }

  const llvm::MemoryBuffer *getBuffer() const {
    assert(isBuffer());
    return Buffer;
  }
};

/// FrontendOptions - Options for controlling the behavior of the frontend.
class FrontendOptions {
public:
  /// Disable memory freeing on exit.
  unsigned DisableFree : 1;

  /// When generating PCH files, instruct the AST writer to create relocatable
  /// PCH files.
  unsigned RelocatablePCH : 1;

  /// Show the -help text.
  unsigned ShowHelp : 1;

  /// Show frontend performance metrics and statistics.
  unsigned ShowStats : 1;

  /// Show timers for individual actions.
  unsigned ShowTimers : 1;

  /// print the supported cpus for the current target
  unsigned PrintSupportedCPUs : 1;

  /// Output time trace profile.
  unsigned TimeTrace : 1;

  /// Show the -version text.
  unsigned ShowVersion : 1;

  /// Apply fixes even if there are unfixable errors.
  unsigned FixWhatYouCan : 1;

  /// Apply fixes only for warnings.
  unsigned FixOnlyWarnings : 1;

  /// Apply fixes and recompile.
  unsigned FixAndRecompile : 1;

  /// Apply fixes to temporary files.
  unsigned FixToTemporaries : 1;

  /// Emit ARC errors even if the migrator can fix them.
  unsigned ARCMTMigrateEmitARCErrors : 1;

  /// Skip over function bodies to speed up parsing in cases you do not need
  /// them (e.g. with code completion).
  unsigned SkipFunctionBodies : 1;

  /// Whether we can use the global module index if available.
  unsigned UseGlobalModuleIndex : 1;

  /// Whether we can generate the global module index if needed.
  unsigned GenerateGlobalModuleIndex : 1;

  /// Whether we include declaration dumps in AST dumps.
  unsigned ASTDumpDecls : 1;

  /// Whether we deserialize all decls when forming AST dumps.
  unsigned ASTDumpAll : 1;

  /// Whether we include lookup table dumps in AST dumps.
  unsigned ASTDumpLookups : 1;

  /// Whether we are performing an implicit module build.
  unsigned BuildingImplicitModule : 1;

  /// Whether we should embed all used files into the PCM file.
  unsigned ModulesEmbedAllFiles : 1;

  /// Whether timestamps should be written to the produced PCH file.
  unsigned IncludeTimestamps : 1;

  /// Should a temporary file be used during compilation.
  unsigned UseTemporary : 1;

  /// When using -emit-module, treat the modulemap as a system module.
  unsigned IsSystemModule : 1;

  CodeCompleteOptions CodeCompleteOpts;

  /// Specifies the output format of the AST.
  ASTDumpOutputFormat ASTDumpFormat = ADOF_Default;

  enum {
    ARCMT_None,
    ARCMT_Check,
    ARCMT_Modify,
    ARCMT_Migrate
  } ARCMTAction = ARCMT_None;

  enum {
    ObjCMT_None = 0,

    /// Enable migration to modern ObjC literals.
    ObjCMT_Literals = 0x1,

    /// Enable migration to modern ObjC subscripting.
    ObjCMT_Subscripting = 0x2,

    /// Enable migration to modern ObjC readonly property.
    ObjCMT_ReadonlyProperty = 0x4,

    /// Enable migration to modern ObjC readwrite property.
    ObjCMT_ReadwriteProperty = 0x8,

    /// Enable migration to modern ObjC property.
    ObjCMT_Property = (ObjCMT_ReadonlyProperty | ObjCMT_ReadwriteProperty),

    /// Enable annotation of ObjCMethods of all kinds.
    ObjCMT_Annotation = 0x10,

    /// Enable migration of ObjC methods to 'instancetype'.
    ObjCMT_Instancetype = 0x20,

    /// Enable migration to NS_ENUM/NS_OPTIONS macros.
    ObjCMT_NsMacros = 0x40,

    /// Enable migration to add conforming protocols.
    ObjCMT_ProtocolConformance = 0x80,

    /// prefer 'atomic' property over 'nonatomic'.
    ObjCMT_AtomicProperty = 0x100,

    /// annotate property with NS_RETURNS_INNER_POINTER
    ObjCMT_ReturnsInnerPointerProperty = 0x200,

    /// use NS_NONATOMIC_IOSONLY for property 'atomic' attribute
    ObjCMT_NsAtomicIOSOnlyProperty = 0x400,

    /// Enable inferring NS_DESIGNATED_INITIALIZER for ObjC methods.
    ObjCMT_DesignatedInitializer = 0x800,

    /// Enable converting setter/getter expressions to property-dot syntx.
    ObjCMT_PropertyDotSyntax = 0x1000,

    ObjCMT_MigrateDecls = (ObjCMT_ReadonlyProperty | ObjCMT_ReadwriteProperty |
                           ObjCMT_Annotation | ObjCMT_Instancetype |
                           ObjCMT_NsMacros | ObjCMT_ProtocolConformance |
                           ObjCMT_NsAtomicIOSOnlyProperty |
                           ObjCMT_DesignatedInitializer),
    ObjCMT_MigrateAll = (ObjCMT_Literals | ObjCMT_Subscripting |
                         ObjCMT_MigrateDecls | ObjCMT_PropertyDotSyntax)
  };
  unsigned ObjCMTAction = ObjCMT_None;
  std::string ObjCMTWhiteListPath;

  std::string MTMigrateDir;
  std::string ARCMTMigrateReportOut;

  /// The input files and their types.
  SmallVector<FrontendInputFile, 0> Inputs;

  /// When the input is a module map, the original module map file from which
  /// that map was inferred, if any (for umbrella modules).
  std::string OriginalModuleMap;

  /// The output file, if any.
  std::string OutputFile;

  /// If given, the new suffix for fix-it rewritten files.
  std::string FixItSuffix;

  /// If given, filter dumped AST Decl nodes by this substring.
  std::string ASTDumpFilter;

  /// If given, enable code completion at the provided location.
  ParsedSourceLocation CodeCompletionAt;

  /// The frontend action to perform.
  frontend::ActionKind ProgramAction = frontend::ParseSyntaxOnly;

  /// The name of the action to run when using a plugin action.
  std::string ActionName;

  /// Args to pass to the plugins
  std::unordered_map<std::string,std::vector<std::string>> PluginArgs;

  /// The list of plugin actions to run in addition to the normal action.
  std::vector<std::string> AddPluginActions;

  /// The list of plugins to load.
  std::vector<std::string> Plugins;

  /// The list of module file extensions.
  std::vector<std::shared_ptr<ModuleFileExtension>> ModuleFileExtensions;

  /// The list of module map files to load before processing the input.
  std::vector<std::string> ModuleMapFiles;

  /// The list of additional prebuilt module files to load before
  /// processing the input.
  std::vector<std::string> ModuleFiles;

  /// The list of files to embed into the compiled module file.
  std::vector<std::string> ModulesEmbedFiles;

  /// The list of AST files to merge.
  std::vector<std::string> ASTMergeFiles;

  /// A list of arguments to forward to LLVM's option processing; this
  /// should only be used for debugging and experimental features.
  std::vector<std::string> LLVMArgs;

  /// File name of the file that will provide record layouts
  /// (in the format produced by -fdump-record-layouts).
  std::string OverrideRecordLayoutsFile;

  /// Auxiliary triple for CUDA/HIP compilation.
  std::string AuxTriple;

  /// Auxiliary target CPU for CUDA/HIP compilation.
  Optional<std::string> AuxTargetCPU;

  /// Auxiliary target features for CUDA/HIP compilation.
  Optional<std::vector<std::string>> AuxTargetFeatures;

  /// Filename to write statistics to.
  std::string StatsFile;

  /// Minimum time granularity (in microseconds) traced by time profiler.
  unsigned TimeTraceGranularity;

public:
  FrontendOptions()
      : DisableFree(false), RelocatablePCH(false), ShowHelp(false),
        ShowStats(false), ShowTimers(false), TimeTrace(false),
        ShowVersion(false), FixWhatYouCan(false), FixOnlyWarnings(false),
        FixAndRecompile(false), FixToTemporaries(false),
        ARCMTMigrateEmitARCErrors(false), SkipFunctionBodies(false),
        UseGlobalModuleIndex(true), GenerateGlobalModuleIndex(true),
        ASTDumpDecls(false), ASTDumpLookups(false),
        BuildingImplicitModule(false), ModulesEmbedAllFiles(false),
        IncludeTimestamps(true), UseTemporary(true), TimeTraceGranularity(500) {}

  /// getInputKindForExtension - Return the appropriate input kind for a file
  /// extension. For example, "c" would return Language::C.
  ///
  /// \return The input kind for the extension, or Language::Unknown if the
  /// extension is not recognized.
  static InputKind getInputKindForExtension(StringRef Extension);
};

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H
