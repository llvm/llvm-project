//===--- Tweak.h -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tweaks are small actions that run over the AST and produce edits, messages
// etc as a result. They are local, i.e. they should take the current editor
// context, e.g. the cursor position and selection into account.
// The actions are executed in two stages:
//   - Stage 1 should check whether the action is available in a current
//     context. It should be cheap and fast to compute as it is executed for all
//     available actions on every client request, which happen quite frequently.
//   - Stage 2 is performed after stage 1 and can be more expensive to compute.
//     It is performed when the user actually chooses the action.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_TWEAK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_REFACTOR_TWEAK_H

#include "ParsedAST.h"
#include "Selection.h"
#include "SourceCode.h"
#include "index/Index.h"
#include "support/Path.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <string>

namespace clang {
namespace clangd {

class FeatureModuleSet;

/// An interface base for small context-sensitive refactoring actions.
/// To implement a new tweak use the following pattern in a .cpp file:
///   class MyTweak : public Tweak {
///   public:
///     const char* id() const override final; // defined by REGISTER_TWEAK.
///     // implement other methods here.
///   };
///   REGISTER_TWEAK(MyTweak);
class Tweak {
public:
  /// Input to prepare and apply tweaks.
  struct Selection {
    Selection(const SymbolIndex *Index, ParsedAST &AST, unsigned RangeBegin,
              unsigned RangeEnd, SelectionTree ASTSelection,
              llvm::vfs::FileSystem *VFS);
    /// The text of the active document.
    llvm::StringRef Code;
    /// The Index for handling codebase related queries.
    const SymbolIndex *Index = nullptr;
    /// The parsed active file. Never null. (Pointer so Selection is movable).
    ParsedAST *AST;
    /// A location of the cursor in the editor.
    // FIXME: Cursor is redundant and should be removed
    SourceLocation Cursor;
    /// The begin offset of the selection
    unsigned SelectionBegin;
    /// The end offset of the selection
    unsigned SelectionEnd;
    /// The AST nodes that were selected.
    SelectionTree ASTSelection;
    /// File system used to access source code (for cross-file tweaks).
    /// This is only populated when applying a tweak, not during prepare.
    llvm::vfs::FileSystem *FS = nullptr;
    // FIXME: provide a way to get sources and ASTs for other files.
  };

  struct Effect {
    /// A message to be displayed to the user.
    std::optional<std::string> ShowMessage;
    FileEdits ApplyEdits;
    /// Whether the edits should be formatted before presenting to the client.
    /// Note that it applies to all files.
    bool FormatEdits = true;

    static Effect showMessage(StringRef S) {
      Effect E;
      E.ShowMessage = std::string(S);
      return E;
    }

    /// Path is the absolute, symlink-resolved path for the file pointed by FID
    /// in SM. Edit is generated from Replacements.
    /// Fails if cannot figure out absolute path for FID.
    static llvm::Expected<std::pair<Path, Edit>>
    fileEdit(const SourceManager &SM, FileID FID,
             tooling::Replacements Replacements);

    /// Creates an effect with an Edit for the main file.
    /// Fails if cannot figure out absolute path for main file.
    static llvm::Expected<Tweak::Effect>
    mainFileEdit(const SourceManager &SM, tooling::Replacements Replacements);
  };

  virtual ~Tweak() = default;
  /// A unique id of the action, it is always equal to the name of the class
  /// defining the Tweak. Definition is provided automatically by
  /// REGISTER_TWEAK.
  virtual const char *id() const = 0;
  /// Run the first stage of the action. Returns true indicating that the
  /// action is available and should be shown to the user. Returns false if the
  /// action is not available.
  /// This function should be fast, if the action requires non-trivial work it
  /// should be moved into 'apply'.
  /// Returns true iff the action is available and apply() can be called on it.
  virtual bool prepare(const Selection &Sel) = 0;
  /// Run the second stage of the action that would produce the actual effect.
  /// EXPECTS: prepare() was called and returned true.
  virtual Expected<Effect> apply(const Selection &Sel) = 0;

  /// A one-line title of the action that should be shown to the users in the
  /// UI.
  /// EXPECTS: prepare() was called and returned true.
  virtual std::string title() const = 0;
  /// Describes what kind of action this is.
  /// EXPECTS: prepare() was called and returned true.
  virtual llvm::StringLiteral kind() const = 0;
  /// Is this a 'hidden' tweak, which are off by default.
  virtual bool hidden() const { return false; }
};

// All tweaks must be registered in the .cpp file next to their definition.
#define REGISTER_TWEAK(Subclass)                                               \
  ::llvm::Registry<::clang::clangd::Tweak>::Add<Subclass>                      \
      TweakRegistrationFor##Subclass(#Subclass, /*Description=*/"");           \
  const char *Subclass::id() const { return #Subclass; }

/// Calls prepare() on all tweaks that satisfy the filter, returning those that
/// can run on the selection.
std::vector<std::unique_ptr<Tweak>>
prepareTweaks(const Tweak::Selection &S,
              llvm::function_ref<bool(const Tweak &)> Filter,
              const FeatureModuleSet *Modules);

// Calls prepare() on the tweak with a given ID.
// If prepare() returns false, returns an error.
// If prepare() returns true, returns the corresponding tweak.
llvm::Expected<std::unique_ptr<Tweak>>
prepareTweak(StringRef ID, const Tweak::Selection &S,
             const FeatureModuleSet *Modules);
} // namespace clangd
} // namespace clang

#endif
