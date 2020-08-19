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
  using PluginArgsTy =
      std::unordered_map<std::string, std::vector<std::string>>;

  using InputsTy = llvm::SmallVector<FrontendInputFile, 0>;

  CodeCompleteOptions CodeCompleteOpts;

  enum { ARCMT_None, ARCMT_Check, ARCMT_Modify, ARCMT_Migrate };

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

    ObjCMT_MigrateDecls =
        (ObjCMT_ReadonlyProperty | ObjCMT_ReadwriteProperty |
         ObjCMT_Annotation | ObjCMT_Instancetype | ObjCMT_NsMacros |
         ObjCMT_ProtocolConformance | ObjCMT_NsAtomicIOSOnlyProperty |
         ObjCMT_DesignatedInitializer),
    ObjCMT_MigrateAll = (ObjCMT_Literals | ObjCMT_Subscripting |
                         ObjCMT_MigrateDecls | ObjCMT_PropertyDotSyntax)
  };

#define FRONTENDOPT(Name, Bits, Description) unsigned Name : Bits;
#define TYPED_FRONTENDOPT(Type, Name, Description) Type Name;
#include "clang/Frontend/FrontendOptions.def"

public:
  FrontendOptions()
      : DisableFree(false), RelocatablePCH(false), ShowHelp(false),
        ShowStats(false), ShowTimers(false), TimeTrace(false),
        ShowVersion(false), FixWhatYouCan(false), FixOnlyWarnings(false),
        FixAndRecompile(false), FixToTemporaries(false),
        ARCMTAction(ARCMT_None), ARCMTMigrateEmitARCErrors(false),
        SkipFunctionBodies(false), UseGlobalModuleIndex(true),
        GenerateGlobalModuleIndex(true), ASTDumpDecls(false),
        ASTDumpLookups(false), BuildingImplicitModule(false),
        ModulesEmbedAllFiles(false), IncludeTimestamps(true),
        UseTemporary(true), ASTDumpFormat(ADOF_Default),
        ObjCMTAction(ObjCMT_None), ProgramAction(frontend::ParseSyntaxOnly),
        TimeTraceGranularity(500), DashX() {}

  /// getInputKindForExtension - Return the appropriate input kind for a file
  /// extension. For example, "c" would return Language::C.
  ///
  /// \return The input kind for the extension, or Language::Unknown if the
  /// extension is not recognized.
  static InputKind getInputKindForExtension(StringRef Extension);
};

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_FRONTENDOPTIONS_H
