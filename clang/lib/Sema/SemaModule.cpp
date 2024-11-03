//===--- SemaModule.cpp - Semantic Analysis for Modules -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for modules (C++ modules syntax,
//  Objective-C modules syntax, and Clang header modules).
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaInternal.h"
#include <optional>

using namespace clang;
using namespace sema;

static void checkModuleImportContext(Sema &S, Module *M,
                                     SourceLocation ImportLoc, DeclContext *DC,
                                     bool FromInclude = false) {
  SourceLocation ExternCLoc;

  if (auto *LSD = dyn_cast<LinkageSpecDecl>(DC)) {
    switch (LSD->getLanguage()) {
    case LinkageSpecDecl::lang_c:
      if (ExternCLoc.isInvalid())
        ExternCLoc = LSD->getBeginLoc();
      break;
    case LinkageSpecDecl::lang_cxx:
      break;
    }
    DC = LSD->getParent();
  }

  while (isa<LinkageSpecDecl>(DC) || isa<ExportDecl>(DC))
    DC = DC->getParent();

  if (!isa<TranslationUnitDecl>(DC)) {
    S.Diag(ImportLoc, (FromInclude && S.isModuleVisible(M))
                          ? diag::ext_module_import_not_at_top_level_noop
                          : diag::err_module_import_not_at_top_level_fatal)
        << M->getFullModuleName() << DC;
    S.Diag(cast<Decl>(DC)->getBeginLoc(),
           diag::note_module_import_not_at_top_level)
        << DC;
  } else if (!M->IsExternC && ExternCLoc.isValid()) {
    S.Diag(ImportLoc, diag::ext_module_import_in_extern_c)
      << M->getFullModuleName();
    S.Diag(ExternCLoc, diag::note_extern_c_begins_here);
  }
}

// We represent the primary and partition names as 'Paths' which are sections
// of the hierarchical access path for a clang module.  However for C++20
// the periods in a name are just another character, and we will need to
// flatten them into a string.
static std::string stringFromPath(ModuleIdPath Path) {
  std::string Name;
  if (Path.empty())
    return Name;

  for (auto &Piece : Path) {
    if (!Name.empty())
      Name += ".";
    Name += Piece.first->getName();
  }
  return Name;
}

Sema::DeclGroupPtrTy
Sema::ActOnGlobalModuleFragmentDecl(SourceLocation ModuleLoc) {
  if (!ModuleScopes.empty() &&
      ModuleScopes.back().Module->Kind == Module::GlobalModuleFragment) {
    // Under -std=c++2a -fmodules-ts, we can find an explicit 'module;' after
    // already implicitly entering the global module fragment. That's OK.
    assert(getLangOpts().CPlusPlusModules && getLangOpts().ModulesTS &&
           "unexpectedly encountered multiple global module fragment decls");
    ModuleScopes.back().BeginLoc = ModuleLoc;
    return nullptr;
  }

  // We start in the global module; all those declarations are implicitly
  // module-private (though they do not have module linkage).
  Module *GlobalModule =
      PushGlobalModuleFragment(ModuleLoc, /*IsImplicit=*/false);

  // All declarations created from now on are owned by the global module.
  auto *TU = Context.getTranslationUnitDecl();
  // [module.global.frag]p2
  // A global-module-fragment specifies the contents of the global module
  // fragment for a module unit. The global module fragment can be used to
  // provide declarations that are attached to the global module and usable
  // within the module unit.
  //
  // So the declations in the global module shouldn't be visible by default.
  TU->setModuleOwnershipKind(Decl::ModuleOwnershipKind::ReachableWhenImported);
  TU->setLocalOwningModule(GlobalModule);

  // FIXME: Consider creating an explicit representation of this declaration.
  return nullptr;
}

void Sema::HandleStartOfHeaderUnit() {
  assert(getLangOpts().CPlusPlusModules &&
         "Header units are only valid for C++20 modules");
  SourceLocation StartOfTU =
      SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID());

  StringRef HUName = getLangOpts().CurrentModule;
  if (HUName.empty()) {
    HUName = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID())->getName();
    const_cast<LangOptions &>(getLangOpts()).CurrentModule = HUName.str();
  }

  // TODO: Make the C++20 header lookup independent.
  // When the input is pre-processed source, we need a file ref to the original
  // file for the header map.
  auto F = SourceMgr.getFileManager().getOptionalFileRef(HUName);
  // For the sake of error recovery (if someone has moved the original header
  // after creating the pre-processed output) fall back to obtaining the file
  // ref for the input file, which must be present.
  if (!F)
    F = SourceMgr.getFileEntryRefForID(SourceMgr.getMainFileID());
  assert(F && "failed to find the header unit source?");
  Module::Header H{HUName.str(), HUName.str(), *F};
  auto &Map = PP.getHeaderSearchInfo().getModuleMap();
  Module *Mod = Map.createHeaderUnit(StartOfTU, HUName, H);
  assert(Mod && "module creation should not fail");
  ModuleScopes.push_back({}); // No GMF
  ModuleScopes.back().BeginLoc = StartOfTU;
  ModuleScopes.back().Module = Mod;
  ModuleScopes.back().ModuleInterface = true;
  ModuleScopes.back().IsPartition = false;
  VisibleModules.setVisible(Mod, StartOfTU);

  // From now on, we have an owning module for all declarations we see.
  // All of these are implicitly exported.
  auto *TU = Context.getTranslationUnitDecl();
  TU->setModuleOwnershipKind(Decl::ModuleOwnershipKind::Visible);
  TU->setLocalOwningModule(Mod);
}

/// Tests whether the given identifier is reserved as a module name and
/// diagnoses if it is. Returns true if a diagnostic is emitted and false
/// otherwise.
static bool DiagReservedModuleName(Sema &S, const IdentifierInfo *II,
                                   SourceLocation Loc) {
  enum {
    Valid = -1,
    Invalid = 0,
    Reserved = 1,
  } Reason = Valid;

  if (II->isStr("module") || II->isStr("import"))
    Reason = Invalid;
  else if (II->isReserved(S.getLangOpts()) !=
           ReservedIdentifierStatus::NotReserved)
    Reason = Reserved;

  // If the identifier is reserved (not invalid) but is in a system header,
  // we do not diagnose (because we expect system headers to use reserved
  // identifiers).
  if (Reason == Reserved && S.getSourceManager().isInSystemHeader(Loc))
    Reason = Valid;

  if (Reason != Valid) {
    S.Diag(Loc, diag::err_invalid_module_name) << II << (int)Reason;
    return true;
  }
  return false;
}

Sema::DeclGroupPtrTy
Sema::ActOnModuleDecl(SourceLocation StartLoc, SourceLocation ModuleLoc,
                      ModuleDeclKind MDK, ModuleIdPath Path,
                      ModuleIdPath Partition, ModuleImportState &ImportState) {
  assert((getLangOpts().ModulesTS || getLangOpts().CPlusPlusModules) &&
         "should only have module decl in Modules TS or C++20");

  bool IsFirstDecl = ImportState == ModuleImportState::FirstDecl;
  bool SeenGMF = ImportState == ModuleImportState::GlobalFragment;
  // If any of the steps here fail, we count that as invalidating C++20
  // module state;
  ImportState = ModuleImportState::NotACXX20Module;

  bool IsPartition = !Partition.empty();
  if (IsPartition)
    switch (MDK) {
    case ModuleDeclKind::Implementation:
      MDK = ModuleDeclKind::PartitionImplementation;
      break;
    case ModuleDeclKind::Interface:
      MDK = ModuleDeclKind::PartitionInterface;
      break;
    default:
      llvm_unreachable("how did we get a partition type set?");
    }

  // A (non-partition) module implementation unit requires that we are not
  // compiling a module of any kind.  A partition implementation emits an
  // interface (and the AST for the implementation), which will subsequently
  // be consumed to emit a binary.
  // A module interface unit requires that we are not compiling a module map.
  switch (getLangOpts().getCompilingModule()) {
  case LangOptions::CMK_None:
    // It's OK to compile a module interface as a normal translation unit.
    break;

  case LangOptions::CMK_ModuleInterface:
    if (MDK != ModuleDeclKind::Implementation)
      break;

    // We were asked to compile a module interface unit but this is a module
    // implementation unit.
    Diag(ModuleLoc, diag::err_module_interface_implementation_mismatch)
      << FixItHint::CreateInsertion(ModuleLoc, "export ");
    MDK = ModuleDeclKind::Interface;
    break;

  case LangOptions::CMK_ModuleMap:
    Diag(ModuleLoc, diag::err_module_decl_in_module_map_module);
    return nullptr;

  case LangOptions::CMK_HeaderUnit:
    Diag(ModuleLoc, diag::err_module_decl_in_header_unit);
    return nullptr;
  }

  assert(ModuleScopes.size() <= 1 && "expected to be at global module scope");

  // FIXME: Most of this work should be done by the preprocessor rather than
  // here, in order to support macro import.

  // Only one module-declaration is permitted per source file.
  if (isCurrentModulePurview()) {
    Diag(ModuleLoc, diag::err_module_redeclaration);
    Diag(VisibleModules.getImportLoc(ModuleScopes.back().Module),
         diag::note_prev_module_declaration);
    return nullptr;
  }

  assert((!getLangOpts().CPlusPlusModules || getLangOpts().ModulesTS ||
          SeenGMF == (bool)this->GlobalModuleFragment) &&
         "mismatched global module state");

  // In C++20, the module-declaration must be the first declaration if there
  // is no global module fragment.
  if (getLangOpts().CPlusPlusModules && !IsFirstDecl && !SeenGMF) {
    Diag(ModuleLoc, diag::err_module_decl_not_at_start);
    SourceLocation BeginLoc =
        ModuleScopes.empty()
            ? SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID())
            : ModuleScopes.back().BeginLoc;
    if (BeginLoc.isValid()) {
      Diag(BeginLoc, diag::note_global_module_introducer_missing)
          << FixItHint::CreateInsertion(BeginLoc, "module;\n");
    }
  }

  // C++2b [module.unit]p1: ... The identifiers module and import shall not
  // appear as identifiers in a module-name or module-partition. All
  // module-names either beginning with an identifier consisting of std
  // followed by zero or more digits or containing a reserved identifier
  // ([lex.name]) are reserved and shall not be specified in a
  // module-declaration; no diagnostic is required.

  // Test the first part of the path to see if it's std[0-9]+ but allow the
  // name in a system header.
  StringRef FirstComponentName = Path[0].first->getName();
  if (!getSourceManager().isInSystemHeader(Path[0].second) &&
      (FirstComponentName == "std" ||
       (FirstComponentName.startswith("std") &&
        llvm::all_of(FirstComponentName.drop_front(3), &llvm::isDigit)))) {
    Diag(Path[0].second, diag::err_invalid_module_name)
        << Path[0].first << /*reserved*/ 1;
    return nullptr;
  }

  // Then test all of the components in the path to see if any of them are
  // using another kind of reserved or invalid identifier.
  for (auto Part : Path) {
    if (DiagReservedModuleName(*this, Part.first, Part.second))
      return nullptr;
  }

  // Flatten the dots in a module name. Unlike Clang's hierarchical module map
  // modules, the dots here are just another character that can appear in a
  // module name.
  std::string ModuleName = stringFromPath(Path);
  if (IsPartition) {
    ModuleName += ":";
    ModuleName += stringFromPath(Partition);
  }
  // If a module name was explicitly specified on the command line, it must be
  // correct.
  if (!getLangOpts().CurrentModule.empty() &&
      getLangOpts().CurrentModule != ModuleName) {
    Diag(Path.front().second, diag::err_current_module_name_mismatch)
        << SourceRange(Path.front().second, IsPartition
                                                ? Partition.back().second
                                                : Path.back().second)
        << getLangOpts().CurrentModule;
    return nullptr;
  }
  const_cast<LangOptions&>(getLangOpts()).CurrentModule = ModuleName;

  auto &Map = PP.getHeaderSearchInfo().getModuleMap();
  Module *Mod;

  switch (MDK) {
  case ModuleDeclKind::Interface:
  case ModuleDeclKind::PartitionInterface: {
    // We can't have parsed or imported a definition of this module or parsed a
    // module map defining it already.
    if (auto *M = Map.findModule(ModuleName)) {
      Diag(Path[0].second, diag::err_module_redefinition) << ModuleName;
      if (M->DefinitionLoc.isValid())
        Diag(M->DefinitionLoc, diag::note_prev_module_definition);
      else if (OptionalFileEntryRef FE = M->getASTFile())
        Diag(M->DefinitionLoc, diag::note_prev_module_definition_from_ast_file)
            << FE->getName();
      Mod = M;
      break;
    }

    // Create a Module for the module that we're defining.
    Mod = Map.createModuleForInterfaceUnit(ModuleLoc, ModuleName);
    if (MDK == ModuleDeclKind::PartitionInterface)
      Mod->Kind = Module::ModulePartitionInterface;
    assert(Mod && "module creation should not fail");
    break;
  }

  case ModuleDeclKind::Implementation: {
    std::pair<IdentifierInfo *, SourceLocation> ModuleNameLoc(
        PP.getIdentifierInfo(ModuleName), Path[0].second);
    // C++20 A module-declaration that contains neither an export-
    // keyword nor a module-partition implicitly imports the primary
    // module interface unit of the module as if by a module-import-
    // declaration.
    Mod = getModuleLoader().loadModule(ModuleLoc, {ModuleNameLoc},
                                       Module::AllVisible,
                                       /*IsInclusionDirective=*/false);
    if (!Mod) {
      Diag(ModuleLoc, diag::err_module_not_defined) << ModuleName;
      // Create an empty module interface unit for error recovery.
      Mod = Map.createModuleForInterfaceUnit(ModuleLoc, ModuleName);
    }
  } break;

  case ModuleDeclKind::PartitionImplementation:
    // Create an interface, but note that it is an implementation
    // unit.
    Mod = Map.createModuleForInterfaceUnit(ModuleLoc, ModuleName);
    Mod->Kind = Module::ModulePartitionImplementation;
    break;
  }

  if (!this->GlobalModuleFragment) {
    ModuleScopes.push_back({});
    if (getLangOpts().ModulesLocalVisibility)
      ModuleScopes.back().OuterVisibleModules = std::move(VisibleModules);
  } else {
    // We're done with the global module fragment now.
    ActOnEndOfTranslationUnitFragment(TUFragmentKind::Global);
  }

  // Switch from the global module fragment (if any) to the named module.
  ModuleScopes.back().BeginLoc = StartLoc;
  ModuleScopes.back().Module = Mod;
  ModuleScopes.back().ModuleInterface = MDK != ModuleDeclKind::Implementation;
  ModuleScopes.back().IsPartition = IsPartition;
  VisibleModules.setVisible(Mod, ModuleLoc);

  // From now on, we have an owning module for all declarations we see.
  // In C++20 modules, those declaration would be reachable when imported
  // unless explicitily exported.
  // Otherwise, those declarations are module-private unless explicitly
  // exported.
  auto *TU = Context.getTranslationUnitDecl();
  TU->setModuleOwnershipKind(Decl::ModuleOwnershipKind::ReachableWhenImported);
  TU->setLocalOwningModule(Mod);

  // We are in the module purview, but before any other (non import)
  // statements, so imports are allowed.
  ImportState = ModuleImportState::ImportAllowed;

  // For an implementation, We already made an implicit import (its interface).
  // Make and return the import decl to be added to the current TU.
  if (MDK == ModuleDeclKind::Implementation) {
    // Make the import decl for the interface.
    ImportDecl *Import =
        ImportDecl::Create(Context, CurContext, ModuleLoc, Mod, Path[0].second);
    // and return it to be added.
    return ConvertDeclToDeclGroup(Import);
  }

  // FIXME: Create a ModuleDecl.
  return nullptr;
}

Sema::DeclGroupPtrTy
Sema::ActOnPrivateModuleFragmentDecl(SourceLocation ModuleLoc,
                                     SourceLocation PrivateLoc) {
  // C++20 [basic.link]/2:
  //   A private-module-fragment shall appear only in a primary module
  //   interface unit.
  switch (ModuleScopes.empty() ? Module::GlobalModuleFragment
                               : ModuleScopes.back().Module->Kind) {
  case Module::ModuleMapModule:
  case Module::GlobalModuleFragment:
  case Module::ModulePartitionImplementation:
  case Module::ModulePartitionInterface:
  case Module::ModuleHeaderUnit:
    Diag(PrivateLoc, diag::err_private_module_fragment_not_module);
    return nullptr;

  case Module::PrivateModuleFragment:
    Diag(PrivateLoc, diag::err_private_module_fragment_redefined);
    Diag(ModuleScopes.back().BeginLoc, diag::note_previous_definition);
    return nullptr;

  case Module::ModuleInterfaceUnit:
    break;
  }

  if (!ModuleScopes.back().ModuleInterface) {
    Diag(PrivateLoc, diag::err_private_module_fragment_not_module_interface);
    Diag(ModuleScopes.back().BeginLoc,
         diag::note_not_module_interface_add_export)
        << FixItHint::CreateInsertion(ModuleScopes.back().BeginLoc, "export ");
    return nullptr;
  }

  // FIXME: Check this isn't a module interface partition.
  // FIXME: Check that this translation unit does not import any partitions;
  // such imports would violate [basic.link]/2's "shall be the only module unit"
  // restriction.

  // We've finished the public fragment of the translation unit.
  ActOnEndOfTranslationUnitFragment(TUFragmentKind::Normal);

  auto &Map = PP.getHeaderSearchInfo().getModuleMap();
  Module *PrivateModuleFragment =
      Map.createPrivateModuleFragmentForInterfaceUnit(
          ModuleScopes.back().Module, PrivateLoc);
  assert(PrivateModuleFragment && "module creation should not fail");

  // Enter the scope of the private module fragment.
  ModuleScopes.push_back({});
  ModuleScopes.back().BeginLoc = ModuleLoc;
  ModuleScopes.back().Module = PrivateModuleFragment;
  ModuleScopes.back().ModuleInterface = true;
  VisibleModules.setVisible(PrivateModuleFragment, ModuleLoc);

  // All declarations created from now on are scoped to the private module
  // fragment (and are neither visible nor reachable in importers of the module
  // interface).
  auto *TU = Context.getTranslationUnitDecl();
  TU->setModuleOwnershipKind(Decl::ModuleOwnershipKind::ModulePrivate);
  TU->setLocalOwningModule(PrivateModuleFragment);

  // FIXME: Consider creating an explicit representation of this declaration.
  return nullptr;
}

DeclResult Sema::ActOnModuleImport(SourceLocation StartLoc,
                                   SourceLocation ExportLoc,
                                   SourceLocation ImportLoc, ModuleIdPath Path,
                                   bool IsPartition) {

  bool Cxx20Mode = getLangOpts().CPlusPlusModules || getLangOpts().ModulesTS;
  assert((!IsPartition || Cxx20Mode) && "partition seen in non-C++20 code?");

  // For a C++20 module name, flatten into a single identifier with the source
  // location of the first component.
  std::pair<IdentifierInfo *, SourceLocation> ModuleNameLoc;

  std::string ModuleName;
  if (IsPartition) {
    // We already checked that we are in a module purview in the parser.
    assert(!ModuleScopes.empty() && "in a module purview, but no module?");
    Module *NamedMod = ModuleScopes.back().Module;
    // If we are importing into a partition, find the owning named module,
    // otherwise, the name of the importing named module.
    ModuleName = NamedMod->getPrimaryModuleInterfaceName().str();
    ModuleName += ":";
    ModuleName += stringFromPath(Path);
    ModuleNameLoc = {PP.getIdentifierInfo(ModuleName), Path[0].second};
    Path = ModuleIdPath(ModuleNameLoc);
  } else if (Cxx20Mode) {
    ModuleName = stringFromPath(Path);
    ModuleNameLoc = {PP.getIdentifierInfo(ModuleName), Path[0].second};
    Path = ModuleIdPath(ModuleNameLoc);
  }

  // Diagnose self-import before attempting a load.
  // [module.import]/9
  // A module implementation unit of a module M that is not a module partition
  // shall not contain a module-import-declaration nominating M.
  // (for an implementation, the module interface is imported implicitly,
  //  but that's handled in the module decl code).

  if (getLangOpts().CPlusPlusModules && isCurrentModulePurview() &&
      getCurrentModule()->Name == ModuleName) {
    Diag(ImportLoc, diag::err_module_self_import_cxx20)
        << ModuleName << !ModuleScopes.back().ModuleInterface;
    return true;
  }

  Module *Mod = getModuleLoader().loadModule(
      ImportLoc, Path, Module::AllVisible, /*IsInclusionDirective=*/false);
  if (!Mod)
    return true;

  return ActOnModuleImport(StartLoc, ExportLoc, ImportLoc, Mod, Path);
}

/// Determine whether \p D is lexically within an export-declaration.
static const ExportDecl *getEnclosingExportDecl(const Decl *D) {
  for (auto *DC = D->getLexicalDeclContext(); DC; DC = DC->getLexicalParent())
    if (auto *ED = dyn_cast<ExportDecl>(DC))
      return ED;
  return nullptr;
}

DeclResult Sema::ActOnModuleImport(SourceLocation StartLoc,
                                   SourceLocation ExportLoc,
                                   SourceLocation ImportLoc, Module *Mod,
                                   ModuleIdPath Path) {
  VisibleModules.setVisible(Mod, ImportLoc);

  checkModuleImportContext(*this, Mod, ImportLoc, CurContext);

  // FIXME: we should support importing a submodule within a different submodule
  // of the same top-level module. Until we do, make it an error rather than
  // silently ignoring the import.
  // FIXME: Should we warn on a redundant import of the current module?
  if (Mod->isForBuilding(getLangOpts()) &&
      (getLangOpts().isCompilingModule() || !getLangOpts().ModulesTS)) {
    Diag(ImportLoc, getLangOpts().isCompilingModule()
                        ? diag::err_module_self_import
                        : diag::err_module_import_in_implementation)
        << Mod->getFullModuleName() << getLangOpts().CurrentModule;
  }

  SmallVector<SourceLocation, 2> IdentifierLocs;

  if (Path.empty()) {
    // If this was a header import, pad out with dummy locations.
    // FIXME: Pass in and use the location of the header-name token in this
    // case.
    for (Module *ModCheck = Mod; ModCheck; ModCheck = ModCheck->Parent)
      IdentifierLocs.push_back(SourceLocation());
  } else if (getLangOpts().CPlusPlusModules && !Mod->Parent) {
    // A single identifier for the whole name.
    IdentifierLocs.push_back(Path[0].second);
  } else {
    Module *ModCheck = Mod;
    for (unsigned I = 0, N = Path.size(); I != N; ++I) {
      // If we've run out of module parents, just drop the remaining
      // identifiers.  We need the length to be consistent.
      if (!ModCheck)
        break;
      ModCheck = ModCheck->Parent;

      IdentifierLocs.push_back(Path[I].second);
    }
  }

  ImportDecl *Import = ImportDecl::Create(Context, CurContext, StartLoc,
                                          Mod, IdentifierLocs);
  CurContext->addDecl(Import);

  // Sequence initialization of the imported module before that of the current
  // module, if any.
  if (!ModuleScopes.empty())
    Context.addModuleInitializer(ModuleScopes.back().Module, Import);

  // A module (partition) implementation unit shall not be exported.
  if (getLangOpts().CPlusPlusModules && ExportLoc.isValid() &&
      Mod->Kind == Module::ModuleKind::ModulePartitionImplementation) {
    Diag(ExportLoc, diag::err_export_partition_impl)
        << SourceRange(ExportLoc, Path.back().second);
  } else if (!ModuleScopes.empty() &&
             (ModuleScopes.back().ModuleInterface ||
              (getLangOpts().CPlusPlusModules &&
               ModuleScopes.back().Module->isGlobalModule()))) {
    // Re-export the module if the imported module is exported.
    // Note that we don't need to add re-exported module to Imports field
    // since `Exports` implies the module is imported already.
    if (ExportLoc.isValid() || getEnclosingExportDecl(Import))
      getCurrentModule()->Exports.emplace_back(Mod, false);
    else
      getCurrentModule()->Imports.insert(Mod);
  } else if (ExportLoc.isValid()) {
    // [module.interface]p1:
    // An export-declaration shall inhabit a namespace scope and appear in the
    // purview of a module interface unit.
    Diag(ExportLoc, diag::err_export_not_in_module_interface)
        << (!ModuleScopes.empty() &&
            !ModuleScopes.back().ImplicitGlobalModuleFragment);
  }

  // In some cases we need to know if an entity was present in a directly-
  // imported module (as opposed to a transitive import).  This avoids
  // searching both Imports and Exports.
  DirectModuleImports.insert(Mod);

  return Import;
}

void Sema::ActOnModuleInclude(SourceLocation DirectiveLoc, Module *Mod) {
  checkModuleImportContext(*this, Mod, DirectiveLoc, CurContext, true);
  BuildModuleInclude(DirectiveLoc, Mod);
}

void Sema::BuildModuleInclude(SourceLocation DirectiveLoc, Module *Mod) {
  // Determine whether we're in the #include buffer for a module. The #includes
  // in that buffer do not qualify as module imports; they're just an
  // implementation detail of us building the module.
  //
  // FIXME: Should we even get ActOnModuleInclude calls for those?
  bool IsInModuleIncludes =
      TUKind == TU_Module &&
      getSourceManager().isWrittenInMainFile(DirectiveLoc);

  bool ShouldAddImport = !IsInModuleIncludes;

  // If this module import was due to an inclusion directive, create an
  // implicit import declaration to capture it in the AST.
  if (ShouldAddImport) {
    TranslationUnitDecl *TU = getASTContext().getTranslationUnitDecl();
    ImportDecl *ImportD = ImportDecl::CreateImplicit(getASTContext(), TU,
                                                     DirectiveLoc, Mod,
                                                     DirectiveLoc);
    if (!ModuleScopes.empty())
      Context.addModuleInitializer(ModuleScopes.back().Module, ImportD);
    TU->addDecl(ImportD);
    Consumer.HandleImplicitImportDecl(ImportD);
  }

  getModuleLoader().makeModuleVisible(Mod, Module::AllVisible, DirectiveLoc);
  VisibleModules.setVisible(Mod, DirectiveLoc);

  if (getLangOpts().isCompilingModule()) {
    Module *ThisModule = PP.getHeaderSearchInfo().lookupModule(
        getLangOpts().CurrentModule, DirectiveLoc, false, false);
    (void)ThisModule;
    assert(ThisModule && "was expecting a module if building one");
  }
}

void Sema::ActOnModuleBegin(SourceLocation DirectiveLoc, Module *Mod) {
  checkModuleImportContext(*this, Mod, DirectiveLoc, CurContext, true);

  ModuleScopes.push_back({});
  ModuleScopes.back().Module = Mod;
  if (getLangOpts().ModulesLocalVisibility)
    ModuleScopes.back().OuterVisibleModules = std::move(VisibleModules);

  VisibleModules.setVisible(Mod, DirectiveLoc);

  // The enclosing context is now part of this module.
  // FIXME: Consider creating a child DeclContext to hold the entities
  // lexically within the module.
  if (getLangOpts().trackLocalOwningModule()) {
    for (auto *DC = CurContext; DC; DC = DC->getLexicalParent()) {
      cast<Decl>(DC)->setModuleOwnershipKind(
          getLangOpts().ModulesLocalVisibility
              ? Decl::ModuleOwnershipKind::VisibleWhenImported
              : Decl::ModuleOwnershipKind::Visible);
      cast<Decl>(DC)->setLocalOwningModule(Mod);
    }
  }
}

void Sema::ActOnModuleEnd(SourceLocation EomLoc, Module *Mod) {
  if (getLangOpts().ModulesLocalVisibility) {
    VisibleModules = std::move(ModuleScopes.back().OuterVisibleModules);
    // Leaving a module hides namespace names, so our visible namespace cache
    // is now out of date.
    VisibleNamespaceCache.clear();
  }

  assert(!ModuleScopes.empty() && ModuleScopes.back().Module == Mod &&
         "left the wrong module scope");
  ModuleScopes.pop_back();

  // We got to the end of processing a local module. Create an
  // ImportDecl as we would for an imported module.
  FileID File = getSourceManager().getFileID(EomLoc);
  SourceLocation DirectiveLoc;
  if (EomLoc == getSourceManager().getLocForEndOfFile(File)) {
    // We reached the end of a #included module header. Use the #include loc.
    assert(File != getSourceManager().getMainFileID() &&
           "end of submodule in main source file");
    DirectiveLoc = getSourceManager().getIncludeLoc(File);
  } else {
    // We reached an EOM pragma. Use the pragma location.
    DirectiveLoc = EomLoc;
  }
  BuildModuleInclude(DirectiveLoc, Mod);

  // Any further declarations are in whatever module we returned to.
  if (getLangOpts().trackLocalOwningModule()) {
    // The parser guarantees that this is the same context that we entered
    // the module within.
    for (auto *DC = CurContext; DC; DC = DC->getLexicalParent()) {
      cast<Decl>(DC)->setLocalOwningModule(getCurrentModule());
      if (!getCurrentModule())
        cast<Decl>(DC)->setModuleOwnershipKind(
            Decl::ModuleOwnershipKind::Unowned);
    }
  }
}

void Sema::createImplicitModuleImportForErrorRecovery(SourceLocation Loc,
                                                      Module *Mod) {
  // Bail if we're not allowed to implicitly import a module here.
  if (isSFINAEContext() || !getLangOpts().ModulesErrorRecovery ||
      VisibleModules.isVisible(Mod))
    return;

  // Create the implicit import declaration.
  TranslationUnitDecl *TU = getASTContext().getTranslationUnitDecl();
  ImportDecl *ImportD = ImportDecl::CreateImplicit(getASTContext(), TU,
                                                   Loc, Mod, Loc);
  TU->addDecl(ImportD);
  Consumer.HandleImplicitImportDecl(ImportD);

  // Make the module visible.
  getModuleLoader().makeModuleVisible(Mod, Module::AllVisible, Loc);
  VisibleModules.setVisible(Mod, Loc);
}

/// We have parsed the start of an export declaration, including the '{'
/// (if present).
Decl *Sema::ActOnStartExportDecl(Scope *S, SourceLocation ExportLoc,
                                 SourceLocation LBraceLoc) {
  ExportDecl *D = ExportDecl::Create(Context, CurContext, ExportLoc);

  // Set this temporarily so we know the export-declaration was braced.
  D->setRBraceLoc(LBraceLoc);

  CurContext->addDecl(D);
  PushDeclContext(S, D);

  // C++2a [module.interface]p1:
  //   An export-declaration shall appear only [...] in the purview of a module
  //   interface unit. An export-declaration shall not appear directly or
  //   indirectly within [...] a private-module-fragment.
  if (!isCurrentModulePurview()) {
    Diag(ExportLoc, diag::err_export_not_in_module_interface) << 0;
    D->setInvalidDecl();
    return D;
  } else if (!ModuleScopes.back().ModuleInterface) {
    Diag(ExportLoc, diag::err_export_not_in_module_interface) << 1;
    Diag(ModuleScopes.back().BeginLoc,
         diag::note_not_module_interface_add_export)
        << FixItHint::CreateInsertion(ModuleScopes.back().BeginLoc, "export ");
    D->setInvalidDecl();
    return D;
  } else if (ModuleScopes.back().Module->Kind ==
             Module::PrivateModuleFragment) {
    Diag(ExportLoc, diag::err_export_in_private_module_fragment);
    Diag(ModuleScopes.back().BeginLoc, diag::note_private_module_fragment);
    D->setInvalidDecl();
    return D;
  }

  for (const DeclContext *DC = CurContext; DC; DC = DC->getLexicalParent()) {
    if (const auto *ND = dyn_cast<NamespaceDecl>(DC)) {
      //   An export-declaration shall not appear directly or indirectly within
      //   an unnamed namespace [...]
      if (ND->isAnonymousNamespace()) {
        Diag(ExportLoc, diag::err_export_within_anonymous_namespace);
        Diag(ND->getLocation(), diag::note_anonymous_namespace);
        // Don't diagnose internal-linkage declarations in this region.
        D->setInvalidDecl();
        return D;
      }

      //   A declaration is exported if it is [...] a namespace-definition
      //   that contains an exported declaration.
      //
      // Defer exporting the namespace until after we leave it, in order to
      // avoid marking all subsequent declarations in the namespace as exported.
      if (!DeferredExportedNamespaces.insert(ND).second)
        break;
    }
  }

  //   [...] its declaration or declaration-seq shall not contain an
  //   export-declaration.
  if (auto *ED = getEnclosingExportDecl(D)) {
    Diag(ExportLoc, diag::err_export_within_export);
    if (ED->hasBraces())
      Diag(ED->getLocation(), diag::note_export);
    D->setInvalidDecl();
    return D;
  }

  D->setModuleOwnershipKind(Decl::ModuleOwnershipKind::VisibleWhenImported);
  return D;
}

static bool checkExportedDeclContext(Sema &S, DeclContext *DC,
                                     SourceLocation BlockStart);

namespace {
enum class UnnamedDeclKind {
  Empty,
  StaticAssert,
  Asm,
  UsingDirective,
  Namespace,
  Context
};
}

static std::optional<UnnamedDeclKind> getUnnamedDeclKind(Decl *D) {
  if (isa<EmptyDecl>(D))
    return UnnamedDeclKind::Empty;
  if (isa<StaticAssertDecl>(D))
    return UnnamedDeclKind::StaticAssert;
  if (isa<FileScopeAsmDecl>(D))
    return UnnamedDeclKind::Asm;
  if (isa<UsingDirectiveDecl>(D))
    return UnnamedDeclKind::UsingDirective;
  // Everything else either introduces one or more names or is ill-formed.
  return std::nullopt;
}

unsigned getUnnamedDeclDiag(UnnamedDeclKind UDK, bool InBlock) {
  switch (UDK) {
  case UnnamedDeclKind::Empty:
  case UnnamedDeclKind::StaticAssert:
    // Allow empty-declarations and static_asserts in an export block as an
    // extension.
    return InBlock ? diag::ext_export_no_name_block : diag::err_export_no_name;

  case UnnamedDeclKind::UsingDirective:
    // Allow exporting using-directives as an extension.
    return diag::ext_export_using_directive;

  case UnnamedDeclKind::Namespace:
    // Anonymous namespace with no content.
    return diag::introduces_no_names;

  case UnnamedDeclKind::Context:
    // Allow exporting DeclContexts that transitively contain no declarations
    // as an extension.
    return diag::ext_export_no_names;

  case UnnamedDeclKind::Asm:
    return diag::err_export_no_name;
  }
  llvm_unreachable("unknown kind");
}

static void diagExportedUnnamedDecl(Sema &S, UnnamedDeclKind UDK, Decl *D,
                                    SourceLocation BlockStart) {
  S.Diag(D->getLocation(), getUnnamedDeclDiag(UDK, BlockStart.isValid()))
      << (unsigned)UDK;
  if (BlockStart.isValid())
    S.Diag(BlockStart, diag::note_export);
}

/// Check that it's valid to export \p D.
static bool checkExportedDecl(Sema &S, Decl *D, SourceLocation BlockStart) {
  // C++2a [module.interface]p3:
  //   An exported declaration shall declare at least one name
  if (auto UDK = getUnnamedDeclKind(D))
    diagExportedUnnamedDecl(S, *UDK, D, BlockStart);

  //   [...] shall not declare a name with internal linkage.
  bool HasName = false;
  if (auto *ND = dyn_cast<NamedDecl>(D)) {
    // Don't diagnose anonymous union objects; we'll diagnose their members
    // instead.
    HasName = (bool)ND->getDeclName();
    if (HasName && ND->getFormalLinkage() == InternalLinkage) {
      S.Diag(ND->getLocation(), diag::err_export_internal) << ND;
      if (BlockStart.isValid())
        S.Diag(BlockStart, diag::note_export);
    }
  }

  // C++2a [module.interface]p5:
  //   all entities to which all of the using-declarators ultimately refer
  //   shall have been introduced with a name having external linkage
  if (auto *USD = dyn_cast<UsingShadowDecl>(D)) {
    NamedDecl *Target = USD->getUnderlyingDecl();
    Linkage Lk = Target->getFormalLinkage();
    if (Lk == InternalLinkage || Lk == ModuleLinkage) {
      S.Diag(USD->getLocation(), diag::err_export_using_internal)
          << (Lk == InternalLinkage ? 0 : 1) << Target;
      S.Diag(Target->getLocation(), diag::note_using_decl_target);
      if (BlockStart.isValid())
        S.Diag(BlockStart, diag::note_export);
    }
  }

  // Recurse into namespace-scope DeclContexts. (Only namespace-scope
  // declarations are exported.).
  if (auto *DC = dyn_cast<DeclContext>(D)) {
    if (isa<NamespaceDecl>(D) && DC->decls().empty()) {
      if (!HasName)
        // We don't allow an empty anonymous namespace (we don't allow decls
        // in them either, but that's handled in the recursion).
        diagExportedUnnamedDecl(S, UnnamedDeclKind::Namespace, D, BlockStart);
      // We allow an empty named namespace decl.
    } else if (DC->getRedeclContext()->isFileContext() && !isa<EnumDecl>(D))
      return checkExportedDeclContext(S, DC, BlockStart);
  }
  return false;
}

/// Check that it's valid to export all the declarations in \p DC.
static bool checkExportedDeclContext(Sema &S, DeclContext *DC,
                                     SourceLocation BlockStart) {
  bool AllUnnamed = true;
  for (auto *D : DC->decls())
    AllUnnamed &= checkExportedDecl(S, D, BlockStart);
  return AllUnnamed;
}

/// Complete the definition of an export declaration.
Decl *Sema::ActOnFinishExportDecl(Scope *S, Decl *D, SourceLocation RBraceLoc) {
  auto *ED = cast<ExportDecl>(D);
  if (RBraceLoc.isValid())
    ED->setRBraceLoc(RBraceLoc);

  PopDeclContext();

  if (!D->isInvalidDecl()) {
    SourceLocation BlockStart =
        ED->hasBraces() ? ED->getBeginLoc() : SourceLocation();
    for (auto *Child : ED->decls()) {
      if (checkExportedDecl(*this, Child, BlockStart)) {
        // If a top-level child is a linkage-spec declaration, it might contain
        // no declarations (transitively), in which case it's ill-formed.
        diagExportedUnnamedDecl(*this, UnnamedDeclKind::Context, Child,
                                BlockStart);
      }
      if (auto *FD = dyn_cast<FunctionDecl>(Child)) {
        // [dcl.inline]/7
        // If an inline function or variable that is attached to a named module
        // is declared in a definition domain, it shall be defined in that
        // domain.
        // So, if the current declaration does not have a definition, we must
        // check at the end of the TU (or when the PMF starts) to see that we
        // have a definition at that point.
        if (FD->isInlineSpecified() && !FD->isDefined())
          PendingInlineFuncDecls.insert(FD);
      }
    }
  }

  return D;
}

Module *Sema::PushGlobalModuleFragment(SourceLocation BeginLoc,
                                       bool IsImplicit) {
  // We shouldn't create new global module fragment if there is already
  // one.
  if (!GlobalModuleFragment) {
    ModuleMap &Map = PP.getHeaderSearchInfo().getModuleMap();
    GlobalModuleFragment = Map.createGlobalModuleFragmentForModuleUnit(
        BeginLoc, getCurrentModule());
  }

  assert(GlobalModuleFragment && "module creation should not fail");

  // Enter the scope of the global module.
  ModuleScopes.push_back({BeginLoc, GlobalModuleFragment,
                          /*ModuleInterface=*/false,
                          /*IsPartition=*/false,
                          /*ImplicitGlobalModuleFragment=*/IsImplicit,
                          /*OuterVisibleModules=*/{}});
  VisibleModules.setVisible(GlobalModuleFragment, BeginLoc);

  return GlobalModuleFragment;
}

void Sema::PopGlobalModuleFragment() {
  assert(!ModuleScopes.empty() && getCurrentModule()->isGlobalModule() &&
         "left the wrong module scope, which is not global module fragment");
  ModuleScopes.pop_back();
}

bool Sema::isModuleUnitOfCurrentTU(const Module *M) const {
  assert(M);

  Module *CurrentModuleUnit = getCurrentModule();

  // If we are not in a module currently, M must not be the module unit of
  // current TU.
  if (!CurrentModuleUnit)
    return false;

  return M->isSubModuleOf(CurrentModuleUnit->getTopLevelModule());
}
