//===- CIndexer.h - Clang-C Source Indexing Library -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines CIndexer, a subclass of Indexer that provides extra
// functionality needed by the CIndex library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_LIBCLANG_CINDEXER_H
#define LLVM_CLANG_TOOLS_LIBCLANG_CINDEXER_H

#include "clang-c/Index.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Lex/ModuleLoader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include <utility>
#include <vector>

namespace llvm {
  class CrashRecoveryContext;
}

namespace clang {
class ASTUnit;
class MacroInfo;
class MacroDefinitionRecord;
class SourceLocation;
class Token;
class IdentifierInfo;

class CIndexer {
  bool OnlyLocalDecls;
  bool DisplayDiagnostics;
  unsigned Options; // CXGlobalOptFlags.

  std::string ResourcesPath;
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;

public:
  CIndexer(std::shared_ptr<PCHContainerOperations> PCHContainerOps =
               std::make_shared<PCHContainerOperations>())
      : OnlyLocalDecls(false), DisplayDiagnostics(false),
        Options(CXGlobalOpt_None), PCHContainerOps(std::move(PCHContainerOps)) {
  }

  /// \brief Whether we only want to see "local" declarations (that did not
  /// come from a previous precompiled header). If false, we want to see all
  /// declarations.
  bool getOnlyLocalDecls() const { return OnlyLocalDecls; }
  void setOnlyLocalDecls(bool Local = true) { OnlyLocalDecls = Local; }
  
  bool getDisplayDiagnostics() const { return DisplayDiagnostics; }
  void setDisplayDiagnostics(bool Display = true) {
    DisplayDiagnostics = Display;
  }

  std::shared_ptr<PCHContainerOperations> getPCHContainerOperations() const {
    return PCHContainerOps;
  }

  unsigned getCXGlobalOptFlags() const { return Options; }
  void setCXGlobalOptFlags(unsigned options) { Options = options; }

  bool isOptEnabled(CXGlobalOptFlags opt) const {
    return Options & opt;
  }

  /// \brief Get the path of the clang resource files.
  const std::string &getClangResourcesPath();
};

  /// \brief Return the current size to request for "safety".
  unsigned GetSafetyThreadStackSize();

  /// \brief Set the current size to request for "safety" (or 0, if safety
  /// threads should not be used).
  void SetSafetyThreadStackSize(unsigned Value);

  /// \brief Execution the given code "safely", using crash recovery or safety
  /// threads when possible.
  ///
  /// \return False if a crash was detected.
  bool RunSafely(llvm::CrashRecoveryContext &CRC, llvm::function_ref<void()> Fn,
                 unsigned Size = 0);

  /// \brief Set the thread priority to background.
  /// FIXME: Move to llvm/Support.
  void setThreadBackgroundPriority();

  /// \brief Print libclang's resource usage to standard error.
  void PrintLibclangResourceUsage(CXTranslationUnit TU);

  namespace cxindex {
    void printDiagsToStderr(ASTUnit *Unit);

    /// \brief If \c MacroDefLoc points at a macro definition with \c II as
    /// its name, this retrieves its MacroInfo.
    MacroInfo *getMacroInfo(const IdentifierInfo &II,
                            SourceLocation MacroDefLoc, CXTranslationUnit TU);

    /// \brief Retrieves the corresponding MacroInfo of a MacroDefinitionRecord.
    const MacroInfo *getMacroInfo(const MacroDefinitionRecord *MacroDef,
                                  CXTranslationUnit TU);

    /// \brief If \c Loc resides inside the definition of \c MI and it points at
    /// an identifier that has ever been a macro name, this returns the latest
    /// MacroDefinitionRecord for that name, otherwise it returns NULL.
    MacroDefinitionRecord *checkForMacroInMacroDefinition(const MacroInfo *MI,
                                                          SourceLocation Loc,
                                                          CXTranslationUnit TU);

    /// \brief If \c Tok resides inside the definition of \c MI and it points at
    /// an identifier that has ever been a macro name, this returns the latest
    /// MacroDefinitionRecord for that name, otherwise it returns NULL.
    MacroDefinitionRecord *checkForMacroInMacroDefinition(const MacroInfo *MI,
                                                          const Token &Tok,
                                                          CXTranslationUnit TU);
    }
    }

#endif
