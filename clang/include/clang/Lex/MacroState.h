//===- MacroState.h - C Language Family Macro State  *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_MACROSTATE_H
#define LLVM_CLANG_LEX_MACROSTATE_H

#include "clang/Lex/MacroInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"

namespace clang {

class ExternalPreprocessorSource;
class IdentifierInfo;
class LangOptions;
class ModuleMacro;
class SourceManager;
class VisibleModuleSet;

using LeafModuleMacrosMap =
    llvm::DenseMap<const IdentifierInfo *, llvm::TinyPtrVector<ModuleMacro *>>;

struct PPReferences {
  const VisibleModuleSet &VisibleModules;
  const LeafModuleMacrosMap &LeafModuleMacros;
  const SourceManager &SourceMgr;
  const LangOptions &LangOpts;
};

/// Information about a name that has been used to define a module macro.
struct FullModuleMacroInfo {
  /// The most recent macro directive for this identifier.
  MacroDirective *MD;

  /// The active module macros for this identifier.
  llvm::TinyPtrVector<ModuleMacro *> ActiveModuleMacros;

  /// The generation number at which we last updated ActiveModuleMacros.
  /// \see Preprocessor::VisibleModules.
  unsigned ActiveModuleMacrosGeneration = 0;

  /// Whether this macro name is ambiguous.
  bool IsAmbiguous = false;

  /// The module macros that are overridden by this macro.
  llvm::TinyPtrVector<ModuleMacro *> OverriddenMacros;

  FullModuleMacroInfo(MacroDirective *MD) : MD(MD) {}
};

/// The state of a macro for an identifier.
class MacroState {
  mutable llvm::PointerUnion<MacroDirective *, FullModuleMacroInfo *> State;

  FullModuleMacroInfo *
  getFullModuleInfo(ExternalPreprocessorSource &ExternalSource,
                    llvm::BumpPtrAllocator &Allocator, const IdentifierInfo &II,
                    PPReferences PPRefs) const;

public:
  MacroState() : MacroState(nullptr) {}
  MacroState(MacroDirective *MD) : State(MD) {}

  MacroState(MacroState &&O) noexcept : State(O.State) {
    O.State = (MacroDirective *)nullptr;
  }

  MacroState &operator=(MacroState &&O) noexcept {
    auto S = O.State;
    O.State = (MacroDirective *)nullptr;
    State = S;
    return *this;
  }

  ~MacroState() {
    if (auto *Info = llvm::dyn_cast_if_present<FullModuleMacroInfo *>(State))
      Info->~FullModuleMacroInfo();
  }

  MacroDirective *getLatest() const {
    if (auto *Info = llvm::dyn_cast_if_present<FullModuleMacroInfo *>(State))
      return Info->MD;
    return cast<MacroDirective *>(State);
  }

  void setLatest(MacroDirective *MD) {
    if (auto *Info = llvm::dyn_cast_if_present<FullModuleMacroInfo *>(State))
      Info->MD = MD;
    else
      State = MD;
  }

  ModuleMacroInfo getModuleInfo(ExternalPreprocessorSource &ExternalSource,
                                llvm::BumpPtrAllocator &Allocator,
                                const IdentifierInfo &II,
                                PPReferences PPRefs) const {
    if (auto *Info = getFullModuleInfo(ExternalSource, Allocator, II, PPRefs))
      return ModuleMacroInfo{Info->ActiveModuleMacros, Info->IsAmbiguous};
    return {};
  }

  MacroDirective::DefInfo findDirectiveAtLoc(SourceLocation Loc,
                                             SourceManager &SourceMgr) const {
    // FIXME: Incorporate module macros into the result of this.
    if (auto *Latest = getLatest())
      return Latest->findDirectiveAtLoc(Loc, SourceMgr);
    return {};
  }

  void overrideActiveModuleMacros(ExternalPreprocessorSource &ExternalSource,
                                  llvm::BumpPtrAllocator &Allocator,
                                  const IdentifierInfo &II,
                                  PPReferences PPRefs) const {
    if (auto *Info = getFullModuleInfo(ExternalSource, Allocator, II, PPRefs)) {
      Info->OverriddenMacros.insert(Info->OverriddenMacros.end(),
                                    Info->ActiveModuleMacros.begin(),
                                    Info->ActiveModuleMacros.end());
      Info->ActiveModuleMacros.clear();
      Info->IsAmbiguous = false;
    }
  }

  ArrayRef<ModuleMacro *> getOverriddenMacros() const {
    if (auto *Info = llvm::dyn_cast_if_present<FullModuleMacroInfo *>(State))
      return Info->OverriddenMacros;
    return {};
  }

  void setOverriddenMacros(llvm::BumpPtrAllocator &allocator,
                           ArrayRef<ModuleMacro *> Overrides) {
    auto *Info = llvm::dyn_cast_if_present<FullModuleMacroInfo *>(State);
    if (!Info) {
      if (Overrides.empty())
        return;
      Info = new (allocator)
          FullModuleMacroInfo(llvm::cast<MacroDirective *>(State));
      State = Info;
    }
    Info->OverriddenMacros.clear();
    Info->OverriddenMacros.insert(Info->OverriddenMacros.end(),
                                  Overrides.begin(), Overrides.end());
    Info->ActiveModuleMacrosGeneration = 0;
  }
};

} // namespace clang

#endif // LLVM_CLANG_LEX_MACROSTATE_H
