//===- UnsafeBufferUsage.h - Replace pointers with modern C++ ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines an analysis that aids replacing buffer accesses through
//  raw pointers with safer C++ abstractions such as containers and views/spans.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_UNSAFEBUFFERUSAGE_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_UNSAFEBUFFERUSAGE_H

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "llvm/Support/Debug.h"

namespace clang {

using VarGrpTy = std::vector<const VarDecl *>;
using VarGrpRef = ArrayRef<const VarDecl *>;

class VariableGroupsManager {
public:
  VariableGroupsManager() = default;
  virtual ~VariableGroupsManager() = default;
  /// Returns the set of variables (including `Var`) that need to be fixed
  /// together in one step.
  ///
  /// `Var` must be a variable that needs fix (so it must be in a group).
  virtual VarGrpRef getGroupOfVar(const VarDecl *Var) const =0;
};

/// The interface that lets the caller handle unsafe buffer usage analysis
/// results by overriding this class's handle... methods.
class UnsafeBufferUsageHandler {
#ifndef NDEBUG
public:
  // A self-debugging facility that you can use to notify the user when
  // suggestions or fixits are incomplete.
  // Uses std::function to avoid computing the message when it won't
  // actually be displayed.
  using DebugNote = std::pair<SourceLocation, std::string>;
  using DebugNoteList = std::vector<DebugNote>;
  using DebugNoteByVar = std::map<const VarDecl *, DebugNoteList>;
  DebugNoteByVar DebugNotesByVar;
#endif

public:
  UnsafeBufferUsageHandler() = default;
  virtual ~UnsafeBufferUsageHandler() = default;

  /// This analyses produces large fixits that are organized into lists
  /// of primitive fixits (individual insertions/removals/replacements).
  using FixItList = llvm::SmallVectorImpl<FixItHint>;

  /// Invoked when an unsafe operation over raw pointers is found.
  virtual void handleUnsafeOperation(const Stmt *Operation,
                                     bool IsRelatedToDecl) = 0;

  /// Invoked when a fix is suggested against a variable. This function groups
  /// all variables that must be fixed together (i.e their types must be changed to the
  /// same target type to prevent type mismatches) into a single fixit.
  virtual void handleUnsafeVariableGroup(const VarDecl *Variable,
                                         const VariableGroupsManager &VarGrpMgr,
                                         FixItList &&Fixes) = 0;

#ifndef NDEBUG
public:
  bool areDebugNotesRequested() {
    DEBUG_WITH_TYPE("SafeBuffers", return true);
    return false;
  }

  void addDebugNoteForVar(const VarDecl *VD, SourceLocation Loc,
                          std::string Text) {
    if (areDebugNotesRequested())
      DebugNotesByVar[VD].push_back(std::make_pair(Loc, Text));
  }

  void clearDebugNotes() {
    if (areDebugNotesRequested())
      DebugNotesByVar.clear();
  }
#endif

public:
  /// Returns a reference to the `Preprocessor`:
  virtual bool isSafeBufferOptOut(const SourceLocation &Loc) const = 0;

  virtual std::string
  getUnsafeBufferUsageAttributeTextAt(SourceLocation Loc,
                                      StringRef WSSuffix = "") const = 0;
};

// This function invokes the analysis and allows the caller to react to it
// through the handler class.
void checkUnsafeBufferUsage(const Decl *D, UnsafeBufferUsageHandler &Handler,
                            bool EmitSuggestions);

namespace internal {
// Tests if any two `FixItHint`s in `FixIts` conflict.  Two `FixItHint`s
// conflict if they have overlapping source ranges.
bool anyConflict(const llvm::SmallVectorImpl<FixItHint> &FixIts,
                 const SourceManager &SM);
} // namespace internal
} // end namespace clang

#endif /* LLVM_CLANG_ANALYSIS_ANALYSES_UNSAFEBUFFERUSAGE_H */
