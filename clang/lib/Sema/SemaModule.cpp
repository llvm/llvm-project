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
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace sema;

static void checkModuleImportContext(Sema &S, Module *M,
                                     SourceLocation ImportLoc, DeclContext *DC,
                                     bool FromInclude = false) {
  SourceLocation ExternCLoc;

  if (auto *LSD = dyn_cast<LinkageSpecDecl>(DC)) {
    switch (LSD->getLanguage()) {
    case LinkageSpecLanguageIDs::C:
      if (ExternCLoc.isInvalid())
        ExternCLoc = LSD->getBeginLoc();
      break;
    case LinkageSpecLanguageIDs::CXX:
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
    Name += Piece.getIdentifierInfo()->getName();
  }
  return Name;
}

/// Helper function for makeTransitiveImportsVisible to decide whether
/// the \param Imported module unit is in the same module with the \param
/// CurrentModule.
/// \param FoundPrimaryModuleInterface is a helper parameter to record the
/// primary module interface unit corresponding to the module \param
/// CurrentModule. Since currently it is expensive to decide whether two module
/// units come from the same module by comparing the module name.
static bool
isImportingModuleUnitFromSameModule(ASTContext &Ctx, Module *Imported,
                                    Module *CurrentModule,
                                    Module *&FoundPrimaryModuleInterface) {
  if (!Imported->isNamedModule())
    return false;

  // The a partition unit we're importing must be in the same module of the
  // current module.
  if (Imported->isModulePartition())
    return true;

  // If we found the primary module interface during the search process, we can
  // return quickly to avoid expensive string comparison.
  if (FoundPrimaryModuleInterface)
    return Imported == FoundPrimaryModuleInterface;

  if (!CurrentModule)
    return false;

  // Then the imported module must be a primary module interface unit.  It
  // is only allowed to import the primary module interface unit from the same
  // module in the implementation unit and the implementation partition unit.

  // Since we'll handle implementation unit above. We can only care
  // about the implementation partition unit here.
  if (!CurrentModule->isModulePartitionImplementation())
    return false;

  if (Ctx.isInSameModule(Imported, CurrentModule)) {
    assert(!FoundPrimaryModuleInterface ||
           FoundPrimaryModuleInterface == Imported);
    FoundPrimaryModuleInterface = Imported;
    return true;
  }

  return false;
}

/// [module.import]p7:
///   Additionally, when a module-import-declaration in a module unit of some
///   module M imports another module unit U of M, it also imports all
///   translation units imported by non-exported module-import-declarations in
///   the module unit purview of U. These rules can in turn lead to the
///   importation of yet more translation units.
static void
makeTransitiveImportsVisible(ASTContext &Ctx, VisibleModuleSet &VisibleModules,
                             Module *Imported, Module *CurrentModule,
                             SourceLocation ImportLoc,
                             bool IsImportingPrimaryModuleInterface = false) {
  assert(Imported->isNamedModule() &&
         "'makeTransitiveImportsVisible()' is intended for standard C++ named "
         "modules only.");

  llvm::SmallVector<Module *, 4> Worklist;
  llvm::SmallSet<Module *, 16> Visited;
  Worklist.push_back(Imported);

  Module *FoundPrimaryModuleInterface =
      IsImportingPrimaryModuleInterface ? Imported : nullptr;

  while (!Worklist.empty()) {
    Module *Importing = Worklist.pop_back_val();

    if (Visited.count(Importing))
      continue;
    Visited.insert(Importing);

    // FIXME: The ImportLoc here is not meaningful. It may be problematic if we
    // use the sourcelocation loaded from the visible modules.
    VisibleModules.setVisible(Importing, ImportLoc);

    if (isImportingModuleUnitFromSameModule(Ctx, Importing, CurrentModule,
                                            FoundPrimaryModuleInterface)) {
      for (Module *TransImported : Importing->Imports)
        Worklist.push_back(TransImported);

      for (auto [Exports, _] : Importing->Exports)
        Worklist.push_back(Exports);
    }
  }
}

Sema::DeclGroupPtrTy
Sema::ActOnGlobalModuleFragmentDecl(SourceLocation ModuleLoc) {
  // We start in the global module;
  Module *GlobalModule =
      PushGlobalModuleFragment(ModuleLoc);

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
    HUName =
        SourceMgr.getFileEntryRefForID(SourceMgr.getMainFileID())->getName();
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

  switch (Reason) {
  case Valid:
    return false;
  case Invalid:
    return S.Diag(Loc, diag::err_invalid_module_name) << II;
  case Reserved:
    S.Diag(Loc, diag::warn_reserved_module_name) << II;
    return false;
  }
  llvm_unreachable("fell off a fully covered switch");
}

Sema::DeclGroupPtrTy
Sema::ActOnModuleDecl(SourceLocation StartLoc, SourceLocation ModuleLoc,
                      ModuleDeclKind MDK, ModuleIdPath Path,
                      ModuleIdPath Partition, ModuleImportState &ImportState,
                      bool SeenNoTrivialPPDirective) {
  assert(getLangOpts().CPlusPlusModules &&
         "should only have module decl in standard C++ modules");

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

  assert((!getLangOpts().CPlusPlusModules ||
          SeenGMF == (bool)this->TheGlobalModuleFragment) &&
         "mismatched global module state");

  // In C++20, A module directive may only appear as the first preprocessing
  // tokens in a file (excluding the global module fragment.).
  if (getLangOpts().CPlusPlusModules &&
      (!IsFirstDecl || SeenNoTrivialPPDirective) && !SeenGMF) {
    Diag(ModuleLoc, diag::err_module_decl_not_at_start);
    SourceLocation BeginLoc = PP.getMainFileFirstPPTokenLoc();
    Diag(BeginLoc, diag::note_global_module_introducer_missing)
        << FixItHint::CreateInsertion(BeginLoc, "module;\n");
  }

  // C++23 [module.unit]p1: ... The identifiers module and import shall not
  // appear as identifiers in a module-name or module-partition. All
  // module-names either beginning with an identifier consisting of std
  // followed by zero or more digits or containing a reserved identifier
  // ([lex.name]) are reserved and shall not be specified in a
  // module-declaration; no diagnostic is required.

  // Test the first part of the path to see if it's std[0-9]+ but allow the
  // name in a system header.
  StringRef FirstComponentName = Path[0].getIdentifierInfo()->getName();
  if (!getSourceManager().isInSystemHeader(Path[0].getLoc()) &&
      (FirstComponentName == "std" ||
       (FirstComponentName.starts_with("std") &&
        llvm::all_of(FirstComponentName.drop_front(3), &llvm::isDigit))))
    Diag(Path[0].getLoc(), diag::warn_reserved_module_name)
        << Path[0].getIdentifierInfo();

  // Then test all of the components in the path to see if any of them are
  // using another kind of reserved or invalid identifier.
  for (auto Part : Path) {
    if (DiagReservedModuleName(*this, Part.getIdentifierInfo(), Part.getLoc()))
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
    Diag(Path.front().getLoc(), diag::err_current_module_name_mismatch)
        << SourceRange(Path.front().getLoc(), IsPartition
                                                  ? Partition.back().getLoc()
                                                  : Path.back().getLoc())
        << getLangOpts().CurrentModule;
    return nullptr;
  }
  const_cast<LangOptions&>(getLangOpts()).CurrentModule = ModuleName;

  auto &Map = PP.getHeaderSearchInfo().getModuleMap();
  Module *Mod;                 // The module we are creating.
  Module *Interface = nullptr; // The interface for an implementation.
  switch (MDK) {
  case ModuleDeclKind::Interface:
  case ModuleDeclKind::PartitionInterface: {
    // We can't have parsed or imported a definition of this module or parsed a
    // module map defining it already.
    if (auto *M = Map.findOrLoadModule(ModuleName)) {
      Diag(Path[0].getLoc(), diag::err_module_redefinition) << ModuleName;
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
    // C++20 A module-declaration that contains neither an export-
    // keyword nor a module-partition implicitly imports the primary
    // module interface unit of the module as if by a module-import-
    // declaration.
    IdentifierLoc ModuleNameLoc(Path[0].getLoc(),
                                PP.getIdentifierInfo(ModuleName));

    // The module loader will assume we're trying to import the module that
    // we're building if `LangOpts.CurrentModule` equals to 'ModuleName'.
    // Change the value for `LangOpts.CurrentModule` temporarily to make the
    // module loader work properly.
    const_cast<LangOptions &>(getLangOpts()).CurrentModule = "";
    Interface = getModuleLoader().loadModule(ModuleLoc, {ModuleNameLoc},
                                             Module::AllVisible,
                                             /*IsInclusionDirective=*/false);
    const_cast<LangOptions&>(getLangOpts()).CurrentModule = ModuleName;

    if (!Interface) {
      Diag(ModuleLoc, diag::err_module_not_defined) << ModuleName;
      // Create an empty module interface unit for error recovery.
      Mod = Map.createModuleForInterfaceUnit(ModuleLoc, ModuleName);
    } else {
      Mod = Map.createModuleForImplementationUnit(ModuleLoc, ModuleName);
    }
  } break;

  case ModuleDeclKind::PartitionImplementation:
    // Create an interface, but note that it is an implementation
    // unit.
    Mod = Map.createModuleForInterfaceUnit(ModuleLoc, ModuleName);
    Mod->Kind = Module::ModulePartitionImplementation;
    break;
  }

  if (!this->TheGlobalModuleFragment) {
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

  getASTContext().setCurrentNamedModule(Mod);

  if (auto *Listener = getASTMutationListener())
    Listener->EnteringModulePurview();

  // We already potentially made an implicit import (in the case of a module
  // implementation unit importing its interface).  Make this module visible
  // and return the import decl to be added to the current TU.
  if (Interface) {
    HadImportedNamedModules = true;

    makeTransitiveImportsVisible(getASTContext(), VisibleModules, Interface,
                                 Mod, ModuleLoc,
                                 /*IsImportingPrimaryModuleInterface=*/true);

    // Make the import decl for the interface in the impl module.
    ImportDecl *Import = ImportDecl::Create(Context, CurContext, ModuleLoc,
                                            Interface, Path[0].getLoc());
    CurContext->addDecl(Import);

    // Sequence initialization of the imported module before that of the current
    // module, if any.
    Context.addModuleInitializer(ModuleScopes.back().Module, Import);
    Mod->Imports.insert(Interface); // As if we imported it.
    // Also save this as a shortcut to checking for decls in the interface
    ThePrimaryInterface = Interface;
    // If we made an implicit import of the module interface, then return the
    // imported module decl.
    return ConvertDeclToDeclGroup(Import);
  }

  return nullptr;
}

Sema::DeclGroupPtrTy
Sema::ActOnPrivateModuleFragmentDecl(SourceLocation ModuleLoc,
                                     SourceLocation PrivateLoc) {
  // C++20 [basic.link]/2:
  //   A private-module-fragment shall appear only in a primary module
  //   interface unit.
  switch (ModuleScopes.empty() ? Module::ExplicitGlobalModuleFragment
                               : ModuleScopes.back().Module->Kind) {
  case Module::ModuleMapModule:
  case Module::ExplicitGlobalModuleFragment:
  case Module::ImplicitGlobalModuleFragment:
  case Module::ModulePartitionImplementation:
  case Module::ModulePartitionInterface:
  case Module::ModuleHeaderUnit:
    Diag(PrivateLoc, diag::err_private_module_fragment_not_module);
    return nullptr;

  case Module::PrivateModuleFragment:
    Diag(PrivateLoc, diag::err_private_module_fragment_redefined);
    Diag(ModuleScopes.back().BeginLoc, diag::note_previous_definition);
    return nullptr;

  case Module::ModuleImplementationUnit:
    Diag(PrivateLoc, diag::err_private_module_fragment_not_module_interface);
    Diag(ModuleScopes.back().BeginLoc,
         diag::note_not_module_interface_add_export)
        << FixItHint::CreateInsertion(ModuleScopes.back().BeginLoc, "export ");
    return nullptr;

  case Module::ModuleInterfaceUnit:
    break;
  }

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
  assert((!IsPartition || getLangOpts().CPlusPlusModules) &&
         "partition seen in non-C++20 code?");

  // For a C++20 module name, flatten into a single identifier with the source
  // location of the first component.
  IdentifierLoc ModuleNameLoc;

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
    ModuleNameLoc =
        IdentifierLoc(Path[0].getLoc(), PP.getIdentifierInfo(ModuleName));
    Path = ModuleIdPath(ModuleNameLoc);
  } else if (getLangOpts().CPlusPlusModules) {
    ModuleName = stringFromPath(Path);
    ModuleNameLoc =
        IdentifierLoc(Path[0].getLoc(), PP.getIdentifierInfo(ModuleName));
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
        << ModuleName << currentModuleIsImplementation();
    return true;
  }

  Module *Mod = getModuleLoader().loadModule(
      ImportLoc, Path, Module::AllVisible, /*IsInclusionDirective=*/false);
  if (!Mod)
    return true;

  if (!Mod->isInterfaceOrPartition() && !ModuleName.empty() &&
      !getLangOpts().ObjC) {
    Diag(ImportLoc, diag::err_module_import_non_interface_nor_parition)
        << ModuleName;
    return true;
  }

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
  if (Mod->isHeaderUnit())
    Diag(ImportLoc, diag::warn_experimental_header_unit);

  if (Mod->isNamedModule())
    makeTransitiveImportsVisible(getASTContext(), VisibleModules, Mod,
                                 getCurrentModule(), ImportLoc);
  else
    VisibleModules.setVisible(Mod, ImportLoc);

  assert((!Mod->isModulePartitionImplementation() || getCurrentModule()) &&
         "We can only import a partition unit in a named module.");
  if (Mod->isModulePartitionImplementation() &&
      getCurrentModule()->isModuleInterfaceUnit())
    Diag(ImportLoc,
         diag::warn_import_implementation_partition_unit_in_interface_unit)
        << Mod->Name;

  checkModuleImportContext(*this, Mod, ImportLoc, CurContext);

  // FIXME: we should support importing a submodule within a different submodule
  // of the same top-level module. Until we do, make it an error rather than
  // silently ignoring the import.
  // FIXME: Should we warn on a redundant import of the current module?
  if (Mod->isForBuilding(getLangOpts())) {
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
    IdentifierLocs.push_back(Path[0].getLoc());
  } else {
    Module *ModCheck = Mod;
    for (unsigned I = 0, N = Path.size(); I != N; ++I) {
      // If we've run out of module parents, just drop the remaining
      // identifiers.  We need the length to be consistent.
      if (!ModCheck)
        break;
      ModCheck = ModCheck->Parent;

      IdentifierLocs.push_back(Path[I].getLoc());
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
        << SourceRange(ExportLoc, Path.back().getLoc());
  } else if (ExportLoc.isValid() &&
             (ModuleScopes.empty() || currentModuleIsImplementation())) {
    // [module.interface]p1:
    // An export-declaration shall inhabit a namespace scope and appear in the
    // purview of a module interface unit.
    Diag(ExportLoc, diag::err_export_not_in_module_interface);
  } else if (!ModuleScopes.empty()) {
    // Re-export the module if the imported module is exported.
    // Note that we don't need to add re-exported module to Imports field
    // since `Exports` implies the module is imported already.
    if (ExportLoc.isValid() || getEnclosingExportDecl(Import))
      getCurrentModule()->Exports.emplace_back(Mod, false);
    else
      getCurrentModule()->Imports.insert(Mod);
  }

  HadImportedNamedModules = true;

  return Import;
}

void Sema::ActOnAnnotModuleInclude(SourceLocation DirectiveLoc, Module *Mod) {
  checkModuleImportContext(*this, Mod, DirectiveLoc, CurContext, true);
  BuildModuleInclude(DirectiveLoc, Mod);
}

void Sema::BuildModuleInclude(SourceLocation DirectiveLoc, Module *Mod) {
  // Determine whether we're in the #include buffer for a module. The #includes
  // in that buffer do not qualify as module imports; they're just an
  // implementation detail of us building the module.
  //
  // FIXME: Should we even get ActOnAnnotModuleInclude calls for those?
  bool IsInModuleIncludes =
      TUKind == TU_ClangModule &&
      getSourceManager().isWrittenInMainFile(DirectiveLoc);

  // If we are really importing a module (not just checking layering) due to an
  // #include in the main file, synthesize an ImportDecl.
  if (getLangOpts().Modules && !IsInModuleIncludes) {
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

void Sema::ActOnAnnotModuleBegin(SourceLocation DirectiveLoc, Module *Mod) {
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

void Sema::ActOnAnnotModuleEnd(SourceLocation EomLoc, Module *Mod) {
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
  if (!getLangOpts().HLSL) {
    if (!isCurrentModulePurview()) {
      Diag(ExportLoc, diag::err_export_not_in_module_interface) << 0;
      D->setInvalidDecl();
      return D;
    } else if (currentModuleIsImplementation()) {
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
      if (!getLangOpts().HLSL && !DeferredExportedNamespaces.insert(ND).second)
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

  if (!getLangOpts().HLSL)
    D->setModuleOwnershipKind(Decl::ModuleOwnershipKind::VisibleWhenImported);

  return D;
}

static bool checkExportedDecl(Sema &, Decl *, SourceLocation);

/// Check that it's valid to export all the declarations in \p DC.
static bool checkExportedDeclContext(Sema &S, DeclContext *DC,
                                     SourceLocation BlockStart) {
  bool AllUnnamed = true;
  for (auto *D : DC->decls())
    AllUnnamed &= checkExportedDecl(S, D, BlockStart);
  return AllUnnamed;
}

/// Check that it's valid to export \p D.
static bool checkExportedDecl(Sema &S, Decl *D, SourceLocation BlockStart) {

  // HLSL: export declaration is valid only on functions
  if (S.getLangOpts().HLSL) {
    // Export-within-export was already diagnosed in ActOnStartExportDecl
    if (!isa<FunctionDecl, ExportDecl>(D)) {
      S.Diag(D->getBeginLoc(), diag::err_hlsl_export_not_on_function);
      D->setInvalidDecl();
      return false;
    }
  }

  //  C++20 [module.interface]p3:
  //   [...] it shall not declare a name with internal linkage.
  bool HasName = false;
  if (auto *ND = dyn_cast<NamedDecl>(D)) {
    // Don't diagnose anonymous union objects; we'll diagnose their members
    // instead.
    HasName = (bool)ND->getDeclName();
    if (HasName && ND->getFormalLinkage() == Linkage::Internal) {
      S.Diag(ND->getLocation(), diag::err_export_internal) << ND;
      if (BlockStart.isValid())
        S.Diag(BlockStart, diag::note_export);
      return false;
    }
  }

  // C++2a [module.interface]p5:
  //   all entities to which all of the using-declarators ultimately refer
  //   shall have been introduced with a name having external linkage
  if (auto *USD = dyn_cast<UsingShadowDecl>(D)) {
    NamedDecl *Target = USD->getUnderlyingDecl();
    Linkage Lk = Target->getFormalLinkage();
    if (Lk == Linkage::Internal || Lk == Linkage::Module) {
      S.Diag(USD->getLocation(), diag::err_export_using_internal)
          << (Lk == Linkage::Internal ? 0 : 1) << Target;
      S.Diag(Target->getLocation(), diag::note_using_decl_target);
      if (BlockStart.isValid())
        S.Diag(BlockStart, diag::note_export);
      return false;
    }
  }

  // Recurse into namespace-scope DeclContexts. (Only namespace-scope
  // declarations are exported).
  if (auto *DC = dyn_cast<DeclContext>(D)) {
    if (!isa<NamespaceDecl>(D))
      return true;

    if (auto *ND = dyn_cast<NamedDecl>(D)) {
      if (!ND->getDeclName()) {
        S.Diag(ND->getLocation(), diag::err_export_anon_ns_internal);
        if (BlockStart.isValid())
          S.Diag(BlockStart, diag::note_export);
        return false;
      } else if (!DC->decls().empty() &&
                 DC->getRedeclContext()->isFileContext()) {
        return checkExportedDeclContext(S, DC, BlockStart);
      }
    }
  }
  return true;
}

Decl *Sema::ActOnFinishExportDecl(Scope *S, Decl *D, SourceLocation RBraceLoc) {
  auto *ED = cast<ExportDecl>(D);
  if (RBraceLoc.isValid())
    ED->setRBraceLoc(RBraceLoc);

  PopDeclContext();

  if (!D->isInvalidDecl()) {
    SourceLocation BlockStart =
        ED->hasBraces() ? ED->getBeginLoc() : SourceLocation();
    for (auto *Child : ED->decls()) {
      checkExportedDecl(*this, Child, BlockStart);
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

  // Anything exported from a module should never be considered unused.
  for (auto *Exported : ED->decls())
    Exported->markUsed(getASTContext());

  return D;
}

Module *Sema::PushGlobalModuleFragment(SourceLocation BeginLoc) {
  // We shouldn't create new global module fragment if there is already
  // one.
  if (!TheGlobalModuleFragment) {
    ModuleMap &Map = PP.getHeaderSearchInfo().getModuleMap();
    TheGlobalModuleFragment = Map.createGlobalModuleFragmentForModuleUnit(
        BeginLoc, getCurrentModule());
  }

  assert(TheGlobalModuleFragment && "module creation should not fail");

  // Enter the scope of the global module.
  ModuleScopes.push_back({BeginLoc, TheGlobalModuleFragment,
                          /*OuterVisibleModules=*/{}});
  VisibleModules.setVisible(TheGlobalModuleFragment, BeginLoc);

  return TheGlobalModuleFragment;
}

void Sema::PopGlobalModuleFragment() {
  assert(!ModuleScopes.empty() &&
         getCurrentModule()->isExplicitGlobalModule() &&
         "left the wrong module scope, which is not global module fragment");
  ModuleScopes.pop_back();
}

Module *Sema::PushImplicitGlobalModuleFragment(SourceLocation BeginLoc) {
  if (!TheImplicitGlobalModuleFragment) {
    ModuleMap &Map = PP.getHeaderSearchInfo().getModuleMap();
    TheImplicitGlobalModuleFragment =
        Map.createImplicitGlobalModuleFragmentForModuleUnit(BeginLoc,
                                                            getCurrentModule());
  }
  assert(TheImplicitGlobalModuleFragment && "module creation should not fail");

  // Enter the scope of the global module.
  ModuleScopes.push_back({BeginLoc, TheImplicitGlobalModuleFragment,
                          /*OuterVisibleModules=*/{}});
  VisibleModules.setVisible(TheImplicitGlobalModuleFragment, BeginLoc);
  return TheImplicitGlobalModuleFragment;
}

void Sema::PopImplicitGlobalModuleFragment() {
  assert(!ModuleScopes.empty() &&
         getCurrentModule()->isImplicitGlobalModule() &&
         "left the wrong module scope, which is not global module fragment");
  ModuleScopes.pop_back();
}

bool Sema::isCurrentModulePurview() const {
  if (!getCurrentModule())
    return false;

  /// Does this Module scope describe part of the purview of a standard named
  /// C++ module?
  switch (getCurrentModule()->Kind) {
  case Module::ModuleInterfaceUnit:
  case Module::ModuleImplementationUnit:
  case Module::ModulePartitionInterface:
  case Module::ModulePartitionImplementation:
  case Module::PrivateModuleFragment:
  case Module::ImplicitGlobalModuleFragment:
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Checking Exposure in modules                                               //
//===----------------------------------------------------------------------===//

namespace {
class ExposureChecker {
public:
  ExposureChecker(Sema &S) : SemaRef(S) {}

  bool checkExposure(const VarDecl *D, bool Diag);
  bool checkExposure(const CXXRecordDecl *D, bool Diag);
  bool checkExposure(const Stmt *S, bool Diag);
  bool checkExposure(const FunctionDecl *D, bool Diag);
  bool checkExposure(const NamedDecl *D, bool Diag);
  void checkExposureInContext(const DeclContext *DC);
  bool isExposureCandidate(const NamedDecl *D);

  bool isTULocal(QualType Ty);
  bool isTULocal(const NamedDecl *ND);
  bool isTULocal(const Expr *E);

  Sema &SemaRef;

private:
  llvm::DenseSet<const NamedDecl *> ExposureSet;
  llvm::DenseSet<const NamedDecl *> KnownNonExposureSet;
};

bool ExposureChecker::isTULocal(QualType Ty) {
  // [basic.link]p15:
  // An entity is TU-local if it is
  // - a type, type alias, namespace, namespace alias, function, variable, or
  // template that
  //   -- has internal linkage, or
  return Ty->getLinkage() == Linkage::Internal;

  // TODO:
  // [basic.link]p15.2:
  //   a type with no name that is defined outside a class-specifier, function
  //   body, or initializer or is introduced by a defining-type-specifier that
  //   is used to declare only TU-local entities,
}

bool ExposureChecker::isTULocal(const NamedDecl *D) {
  if (!D)
    return false;

  // [basic.link]p15:
  // An entity is TU-local if it is
  // - a type, type alias, namespace, namespace alias, function, variable, or
  // template that
  //   -- has internal linkage, or
  if (D->getLinkageInternal() == Linkage::Internal)
    return true;

  if (D->isInAnonymousNamespace())
    return true;

  // [basic.link]p15.1.2:
  // does not have a name with linkage and is declared, or introduced by a
  // lambda-expression, within the definition of a TU-local entity,
  if (D->getLinkageInternal() == Linkage::None)
    if (auto *ND = dyn_cast<NamedDecl>(D->getDeclContext());
        ND && isTULocal(ND))
      return true;

  // [basic.link]p15.3, p15.4:
  // - a specialization of a TU-local template,
  // - a specialization of a template with any TU-local template argument, or
  ArrayRef<TemplateArgument> TemplateArgs;
  NamedDecl *PrimaryTemplate = nullptr;
  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    TemplateArgs = CTSD->getTemplateArgs().asArray();
    PrimaryTemplate = CTSD->getSpecializedTemplate();
    if (isTULocal(PrimaryTemplate))
      return true;
  } else if (auto *VTSD = dyn_cast<VarTemplateSpecializationDecl>(D)) {
    TemplateArgs = VTSD->getTemplateArgs().asArray();
    PrimaryTemplate = VTSD->getSpecializedTemplate();
    if (isTULocal(PrimaryTemplate))
      return true;
  } else if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (auto *TAList = FD->getTemplateSpecializationArgs())
      TemplateArgs = TAList->asArray();

    PrimaryTemplate = FD->getPrimaryTemplate();
    if (isTULocal(PrimaryTemplate))
      return true;
  }

  if (!PrimaryTemplate)
    // Following off, we only check for specializations.
    return false;

  if (KnownNonExposureSet.count(D))
    return false;

  for (auto &TA : TemplateArgs) {
    switch (TA.getKind()) {
    case TemplateArgument::Type:
      if (isTULocal(TA.getAsType()))
        return true;
      break;
    case TemplateArgument::Declaration:
      if (isTULocal(TA.getAsDecl()))
        return true;
      break;
    default:
      break;
    }
  }

  // [basic.link]p15.5
  // - a specialization of a template whose (possibly instantiated) declaration
  // is an exposure.
  if (ExposureSet.count(PrimaryTemplate) ||
      checkExposure(PrimaryTemplate, /*Diag=*/false))
    return true;

  // Avoid calling checkExposure again since it is expensive.
  KnownNonExposureSet.insert(D);
  return false;
}

bool ExposureChecker::isTULocal(const Expr *E) {
  if (!E)
    return false;

  // [basic.link]p16:
  //    A value or object is TU-local if either
  //    - it is of TU-local type,
  if (isTULocal(E->getType()))
    return true;

  E = E->IgnoreParenImpCasts();
  // [basic.link]p16.2:
  //   - it is, or is a pointer to, a TU-local function or the object associated
  //   with a TU-local variable,
  //  - it is an object of class or array type and any of its subobjects or any
  //  of the objects or functions to which its non-static data members of
  //  reference type refer is TU-local and is usable in constant expressions, or
  // FIXME: But how can we know the value of pointers or arrays at compile time?
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (auto *FD = dyn_cast_or_null<FunctionDecl>(DRE->getFoundDecl()))
      return isTULocal(FD);
    else if (auto *VD = dyn_cast_or_null<VarDecl>(DRE->getFoundDecl()))
      return isTULocal(VD);
    else if (auto *RD = dyn_cast_or_null<CXXRecordDecl>(DRE->getFoundDecl()))
      return isTULocal(RD);
  }

  // TODO:
  // [basic.link]p16.4:
  //  it is a reflection value that represents...

  return false;
}

bool ExposureChecker::isExposureCandidate(const NamedDecl *D) {
  if (!D)
    return false;

  // [basic.link]p17:
  //   If a (possibly instantiated) declaration of, or a deduction guide for,
  //   a non-TU-local entity in a module interface unit
  //   (outside the private-module-fragment, if any) or
  //   module partition is an exposure, the program is ill-formed.
  Module *M = D->getOwningModule();
  if (!M || !M->isInterfaceOrPartition())
    return false;

  if (D->isImplicit())
    return false;

  // [basic.link]p14:
  //      A declaration is an exposure if it either names a TU-local entity
  //      (defined below), ignoring:
  //      ...
  //      - friend declarations in a class definition
  if (D->getFriendObjectKind() &&
      isa<CXXRecordDecl>(D->getLexicalDeclContext()))
    return false;

  return true;
}

bool ExposureChecker::checkExposure(const NamedDecl *D, bool Diag) {
  if (!isExposureCandidate(D))
    return false;

  if (auto *FD = dyn_cast<FunctionDecl>(D))
    return checkExposure(FD, Diag);
  if (auto *FTD = dyn_cast<FunctionTemplateDecl>(D))
    return checkExposure(FTD->getTemplatedDecl(), Diag);

  if (auto *VD = dyn_cast<VarDecl>(D))
    return checkExposure(VD, Diag);
  if (auto *VTD = dyn_cast<VarTemplateDecl>(D))
    return checkExposure(VTD->getTemplatedDecl(), Diag);

  if (auto *RD = dyn_cast<CXXRecordDecl>(D))
    return checkExposure(RD, Diag);

  if (auto *CTD = dyn_cast<ClassTemplateDecl>(D))
    return checkExposure(CTD->getTemplatedDecl(), Diag);

  return false;
}

bool ExposureChecker::checkExposure(const FunctionDecl *FD, bool Diag) {
  bool IsExposure = false;
  if (isTULocal(FD->getReturnType())) {
    IsExposure = true;
    if (Diag)
      SemaRef.Diag(FD->getReturnTypeSourceRange().getBegin(),
                   diag::warn_exposure)
          << FD->getReturnType();
  }

  for (ParmVarDecl *Parms : FD->parameters())
    if (isTULocal(Parms->getType())) {
      IsExposure = true;
      if (Diag)
        SemaRef.Diag(Parms->getLocation(), diag::warn_exposure)
            << Parms->getType();
    }

  bool IsImplicitInstantiation =
      FD->getTemplateSpecializationKind() == TSK_ImplicitInstantiation;

  // [basic.link]p14:
  //      A declaration is an exposure if it either names a TU-local entity
  //      (defined below), ignoring:
  //      - the function-body for a non-inline function or function template
  //      (but not the deduced return
  //        type for a (possibly instantiated) definition of a function with a
  //        declared return type that uses a placeholder type
  //        ([dcl.spec.auto])),
  Diag &=
      (FD->isInlined() || IsImplicitInstantiation) && !FD->isDependentContext();

  IsExposure |= checkExposure(FD->getBody(), Diag);
  if (IsExposure)
    ExposureSet.insert(FD);

  return IsExposure;
}

bool ExposureChecker::checkExposure(const VarDecl *VD, bool Diag) {
  bool IsExposure = false;
  // [basic.link]p14:
  //  A declaration is an exposure if it either names a TU-local entity (defined
  //  below), ignoring:
  //  ...
  //  or defines a constexpr variable initialized to a TU-local value (defined
  //  below).
  if (VD->isConstexpr() && isTULocal(VD->getInit())) {
    IsExposure = true;
    if (Diag)
      SemaRef.Diag(VD->getInit()->getExprLoc(), diag::warn_exposure)
          << VD->getInit();
  }

  if (isTULocal(VD->getType())) {
    IsExposure = true;
    if (Diag)
      SemaRef.Diag(VD->getLocation(), diag::warn_exposure) << VD->getType();
  }

  // [basic.link]p14:
  //   ..., ignoring:
  //   - the initializer for a variable or variable template (but not the
  //   variable's type),
  //
  // Note: although the spec says to ignore the initializer for all variable,
  // for the code we generated now for inline variables, it is dangerous if the
  // initializer of an inline variable is TULocal.
  Diag &= !VD->getDeclContext()->isDependentContext() && VD->isInline();
  IsExposure |= checkExposure(VD->getInit(), Diag);
  if (IsExposure)
    ExposureSet.insert(VD);

  return IsExposure;
}

bool ExposureChecker::checkExposure(const CXXRecordDecl *RD, bool Diag) {
  if (!RD->hasDefinition())
    return false;

  bool IsExposure = false;
  for (CXXMethodDecl *Method : RD->methods())
    IsExposure |= checkExposure(Method, Diag);

  for (FieldDecl *FD : RD->fields()) {
    if (isTULocal(FD->getType())) {
      IsExposure = true;
      if (Diag)
        SemaRef.Diag(FD->getLocation(), diag::warn_exposure) << FD->getType();
    }
  }

  for (const CXXBaseSpecifier &Base : RD->bases()) {
    if (isTULocal(Base.getType())) {
      IsExposure = true;
      if (Diag)
        SemaRef.Diag(Base.getBaseTypeLoc(), diag::warn_exposure)
            << Base.getType();
    }
  }

  if (IsExposure)
    ExposureSet.insert(RD);

  return IsExposure;
}

class ReferenceTULocalChecker : public DynamicRecursiveASTVisitor {
public:
  using CallbackTy = std::function<void(DeclRefExpr *, ValueDecl *)>;

  ReferenceTULocalChecker(ExposureChecker &C, CallbackTy &&Callback)
      : Checker(C), Callback(std::move(Callback)) {}

  bool VisitDeclRefExpr(DeclRefExpr *DRE) override {
    ValueDecl *Referenced = DRE->getDecl();
    if (!Referenced)
      return true;

    if (!Checker.isTULocal(Referenced))
      // We don't care if the referenced declaration is not TU-local.
      return true;

    Qualifiers Qual = DRE->getType().getQualifiers();
    // [basic.link]p14:
    //      A declaration is an exposure if it either names a TU-local entity
    //      (defined below), ignoring:
    //      ...
    //      - any reference to a non-volatile const object ...
    if (Qual.hasConst() && !Qual.hasVolatile())
      return true;

    // [basic.link]p14:
    // ..., ignoring:
    //   ...
    //   (p14.4) - ... or reference with internal or no linkage initialized with
    //   a constant expression that is not an odr-use
    ASTContext &Context = Referenced->getASTContext();
    Linkage L = Referenced->getLinkageInternal();
    if (DRE->isNonOdrUse() && (L == Linkage::Internal || L == Linkage::None))
      if (auto *VD = dyn_cast<VarDecl>(Referenced);
          VD && VD->getInit() && !VD->getInit()->isValueDependent() &&
          VD->getInit()->isConstantInitializer(Context, /*IsForRef=*/false))
        return true;

    Callback(DRE, Referenced);
    return true;
  }

  ExposureChecker &Checker;
  CallbackTy Callback;
};

bool ExposureChecker::checkExposure(const Stmt *S, bool Diag) {
  if (!S)
    return false;

  bool HasReferencedTULocals = false;
  ReferenceTULocalChecker Checker(
      *this, [this, &HasReferencedTULocals, Diag](DeclRefExpr *DRE,
                                                  ValueDecl *Referenced) {
        if (Diag) {
          SemaRef.Diag(DRE->getExprLoc(), diag::warn_exposure) << Referenced;
        }
        HasReferencedTULocals = true;
      });
  Checker.TraverseStmt(const_cast<Stmt *>(S));
  return HasReferencedTULocals;
}

void ExposureChecker::checkExposureInContext(const DeclContext *DC) {
  for (auto *TopD : DC->noload_decls()) {
    auto *TopND = dyn_cast<NamedDecl>(TopD);
    if (!TopND)
      continue;

    if (auto *Namespace = dyn_cast<NamespaceDecl>(TopND)) {
      checkExposureInContext(Namespace);
      continue;
    }

    // [basic.link]p17:
    //   If a (possibly instantiated) declaration of, or a deduction guide for,
    //   a non-TU-local entity in a module interface unit
    //   (outside the private-module-fragment, if any) or
    //   module partition is an exposure, the program is ill-formed.
    if (!TopND->isFromASTFile() && isExposureCandidate(TopND) &&
        !isTULocal(TopND))
      checkExposure(TopND, /*Diag=*/true);
  }
}

} // namespace

void Sema::checkExposure(const TranslationUnitDecl *TU) {
  if (!TU)
    return;

  ExposureChecker Checker(*this);

  Module *M = TU->getOwningModule();
  if (M && M->isInterfaceOrPartition())
    Checker.checkExposureInContext(TU);

  // [basic.link]p18:
  //   If a declaration that appears in one translation unit names a TU-local
  //   entity declared in another translation unit that is not a header unit,
  //   the program is ill-formed.
  for (auto FDAndInstantiationLocPair : PendingCheckReferenceForTULocal) {
    FunctionDecl *FD = FDAndInstantiationLocPair.first;
    SourceLocation PointOfInstantiation = FDAndInstantiationLocPair.second;

    if (!FD->hasBody())
      continue;

    ReferenceTULocalChecker(Checker, [&, this](DeclRefExpr *DRE,
                                               ValueDecl *Referenced) {
      // A "defect" in current implementation. Now an implicit instantiation of
      // a template, the instantiation is considered to be in the same module
      // unit as the template instead of the module unit where the instantiation
      // happens.
      //
      // See test/Modules/Exposre-2.cppm for example.
      if (!Referenced->isFromASTFile())
        return;

      if (!Referenced->isInAnotherModuleUnit())
        return;

      // This is not standard conforming. But given there are too many static
      // (inline) functions in headers in existing code, it is more user
      // friendly to ignore them temporarily now. maybe we can have another flag
      // for this.
      if (Referenced->getOwningModule()->isExplicitGlobalModule() &&
          isa<FunctionDecl>(Referenced))
        return;

      Diag(PointOfInstantiation,
           diag::warn_reference_tu_local_entity_in_other_tu)
          << FD << Referenced
          << Referenced->getOwningModule()->getTopLevelModuleName();
    }).TraverseStmt(FD->getBody());
  }
}

void Sema::checkReferenceToTULocalFromOtherTU(
    FunctionDecl *FD, SourceLocation PointOfInstantiation) {
  // Checking if a declaration have any reference to TU-local entities in other
  // TU is expensive. Try to avoid it as much as possible.
  if (!FD || !HadImportedNamedModules)
    return;

  PendingCheckReferenceForTULocal.push_back(
      std::make_pair(FD, PointOfInstantiation));
}
