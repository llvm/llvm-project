#include "clang/Lex/MacroState.h"

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/MacroInfo.h"

namespace clang {

/// Update the set of active module macros and ambiguity flag for a module
/// macro name.
static void updateModuleMacroInfo(const IdentifierInfo &II,
                                  FullModuleMacroInfo &Info,
                                  PPReferences PPRefs) {
  assert(Info.ActiveModuleMacrosGeneration !=
             PPRefs.VisibleModules.getGeneration() &&
         "don't need to update this macro name info");
  Info.ActiveModuleMacrosGeneration = PPRefs.VisibleModules.getGeneration();

  auto Leaf = PPRefs.LeafModuleMacros.find(&II);
  if (Leaf == PPRefs.LeafModuleMacros.end()) {
    // No imported macros at all: nothing to do.
    return;
  }

  Info.ActiveModuleMacros.clear();

  // Every macro that's locally overridden is overridden by a visible macro.
  llvm::DenseMap<ModuleMacro *, int> NumHiddenOverrides;
  for (auto *O : Info.OverriddenMacros)
    NumHiddenOverrides[O] = -1;

  // Collect all macros that are not overridden by a visible macro.
  llvm::SmallVector<ModuleMacro *, 16> Worklist;
  for (auto *LeafMM : Leaf->second) {
    assert(LeafMM->getNumOverridingMacros() == 0 && "leaf macro overridden");
    if (NumHiddenOverrides.lookup(LeafMM) == 0)
      Worklist.push_back(LeafMM);
  }
  while (!Worklist.empty()) {
    auto *MM = Worklist.pop_back_val();
    if (PPRefs.VisibleModules.isVisible(MM->getOwningModule())) {
      // We only care about collecting definitions; undefinitions only act
      // to override other definitions.
      if (MM->getMacroInfo())
        Info.ActiveModuleMacros.push_back(MM);
    } else {
      for (auto *O : MM->overrides())
        if ((unsigned)++NumHiddenOverrides[O] == O->getNumOverridingMacros())
          Worklist.push_back(O);
    }
  }
  // Our reverse postorder walk found the macros in reverse order.
  std::reverse(Info.ActiveModuleMacros.begin(), Info.ActiveModuleMacros.end());

  // Determine whether the macro name is ambiguous.
  MacroInfo *MI = nullptr;
  bool IsSystemMacro = true;
  bool IsAmbiguous = false;
  if (auto *MD = Info.MD) {
    while (isa_and_nonnull<VisibilityMacroDirective>(MD))
      MD = MD->getPrevious();
    if (auto *DMD = dyn_cast_or_null<DefMacroDirective>(MD)) {
      MI = DMD->getInfo();
      IsSystemMacro &= PPRefs.SourceMgr.isInSystemHeader(DMD->getLocation());
    }
  }
  for (auto *Active : Info.ActiveModuleMacros) {
    auto *NewMI = Active->getMacroInfo();

    // Before marking the macro as ambiguous, check if this is a case where
    // both macros are in system headers. If so, we trust that the system
    // did not get it wrong. This also handles cases where Clang's own
    // headers have a different spelling of certain system macros:
    //   #define LONG_MAX __LONG_MAX__ (clang's limits.h)
    //   #define LONG_MAX 0x7fffffffffffffffL (system's limits.h)
    //
    // FIXME: Remove the defined-in-system-headers check. clang's limits.h
    // overrides the system limits.h's macros, so there's no conflict here.
    if (MI && NewMI != MI &&
        !MI->isIdenticalTo(*NewMI, PPRefs.SourceMgr, PPRefs.LangOpts,
                           /*Syntactically=*/true))
      IsAmbiguous = true;
    IsSystemMacro &=
        Active->getOwningModule()->IsSystem ||
        PPRefs.SourceMgr.isInSystemHeader(NewMI->getDefinitionLoc());
    MI = NewMI;
  }
  Info.IsAmbiguous = IsAmbiguous && !IsSystemMacro;
}

FullModuleMacroInfo *
MacroState::getFullModuleInfo(ExternalPreprocessorSource &ExternalSource,
                              llvm::BumpPtrAllocator &Allocator,
                              const IdentifierInfo &II,
                              PPReferences PPRefs) const {
  if (II.isOutOfDate())
    ExternalSource.updateOutOfDateIdentifier(II);
  // FIXME: Find a spare bit on IdentifierInfo and store a
  //        HasModuleMacros flag.
  if (!II.hasMacroDefinition() ||
      (!PPRefs.LangOpts.Modules && !PPRefs.LangOpts.ModulesLocalVisibility) ||
      !PPRefs.VisibleModules.getGeneration())
    return nullptr;

  auto *Info = dyn_cast_if_present<FullModuleMacroInfo *>(State);
  if (!Info) {
    Info = new (Allocator) FullModuleMacroInfo(cast<MacroDirective *>(State));
    State = Info;
  }

  if (PPRefs.VisibleModules.getGeneration() !=
      Info->ActiveModuleMacrosGeneration)
    updateModuleMacroInfo(II, *Info, PPRefs);
  return Info;
}

} // namespace clang
