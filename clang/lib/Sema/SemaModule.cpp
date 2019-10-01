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
  auto &Map = PP.getHeaderSearchInfo().getModuleMap();
  auto *GlobalModule = Map.createGlobalModuleFragmentForModuleUnit(ModuleLoc);
  assert(GlobalModule && "module creation should not fail");

  // Enter the scope of the global module.
  ModuleScopes.push_back({});
  ModuleScopes.back().BeginLoc = ModuleLoc;
  ModuleScopes.back().Module = GlobalModule;
  VisibleModules.setVisible(GlobalModule, ModuleLoc);

  // All declarations created from now on are owned by the global module.
  auto *TU = Context.getTranslationUnitDecl();
  TU->setModuleOwnershipKind(Decl::ModuleOwnershipKind::Visible);
  TU->setLocalOwningModule(GlobalModule);

  // FIXME: Consider creating an explicit representation of this declaration.
  return nullptr;
}

Sema::DeclGroupPtrTy
Sema::ActOnModuleDecl(SourceLocation StartLoc, SourceLocation ModuleLoc,
                      ModuleDeclKind MDK, ModuleIdPath Path, bool IsFirstDecl) {
  assert((getLangOpts().ModulesTS || getLangOpts().CPlusPlusModules) &&
         "should only have module decl in Modules TS or C++20");

  // A module implementation unit requires that we are not compiling a module
  // of any kind. A module interface unit requires that we are not compiling a
  // module map.
  switch (getLangOpts().getCompilingModule()) {
  case LangOptions::CMK_None:
    // It's OK to compile a module interface as a normal translation unit.
    break;

  case LangOptions::CMK_ModuleInterface:
    if (MDK != ModuleDeclKind::Implementation)
      break;

    // We were asked to compile a module interface unit but this is a module
    // implementation unit. That indicates the 'export' is missing.
    Diag(ModuleLoc, diag::err_module_interface_implementation_mismatch)
      << FixItHint::CreateInsertion(ModuleLoc, "export ");
    MDK = ModuleDeclKind::Interface;
    break;

  case LangOptions::CMK_ModuleMap:
    Diag(ModuleLoc, diag::err_module_decl_in_module_map_module);
    return nullptr;

  case LangOptions::CMK_HeaderModule:
    Diag(ModuleLoc, diag::err_module_decl_in_header_module);
    return nullptr;
  }

  assert(ModuleScopes.size() <= 1 && "expected to be at global module scope");

  // FIXME: Most of this work should be done by the preprocessor rather than
  // here, in order to support macro import.

  // Only one module-declaration is permitted per source file.
  if (!ModuleScopes.empty() &&
      ModuleScopes.back().Module->isModulePurview()) {
    Diag(ModuleLoc, diag::err_module_redeclaration);
    Diag(VisibleModules.getImportLoc(ModuleScopes.back().Module),
         diag::note_prev_module_declaration);
    return nullptr;
  }

  // Find the global module fragment we're adopting into this module, if any.
  Module *GlobalModuleFragment = nullptr;
  if (!ModuleScopes.empty() &&
      ModuleScopes.back().Module->Kind == Module::GlobalModuleFragment)
    GlobalModuleFragment = ModuleScopes.back().Module;

  // In C++20, the module-declaration must be the first declaration if there
  // is no global module fragment.
  if (getLangOpts().CPlusPlusModules && !IsFirstDecl && !GlobalModuleFragment) {
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

  // Flatten the dots in a module name. Unlike Clang's hierarchical module map
  // modules, the dots here are just another character that can appear in a
  // module name.
  std::string ModuleName;
  for (auto &Piece : Path) {
    if (!ModuleName.empty())
      ModuleName += ".";
    ModuleName += Piece.first->getName();
  }

  // If a module name was explicitly specified on the command line, it must be
  // correct.
  if (!getLangOpts().CurrentModule.empty() &&
      getLangOpts().CurrentModule != ModuleName) {
    Diag(Path.front().second, diag::err_current_module_name_mismatch)
        << SourceRange(Path.front().second, Path.back().second)
        << getLangOpts().CurrentModule;
    return nullptr;
  }
  const_cast<LangOptions&>(getLangOpts()).CurrentModule = ModuleName;

  auto &Map = PP.getHeaderSearchInfo().getModuleMap();
  Module *Mod;

  switch (MDK) {
  case ModuleDeclKind::Interface: {
    // We can't have parsed or imported a definition of this module or parsed a
    // module map defining it already.
    if (auto *M = Map.findModule(ModuleName)) {
      Diag(Path[0].second, diag::err_module_redefinition) << ModuleName;
      if (M->DefinitionLoc.isValid())
        Diag(M->DefinitionLoc, diag::note_prev_module_definition);
      else if (const auto *FE = M->getASTFile())
        Diag(M->DefinitionLoc, diag::note_prev_module_definition_from_ast_file)
            << FE->getName();
      Mod = M;
      break;
    }

    // Create a Module for the module that we're defining.
    Mod = Map.createModuleForInterfaceUnit(ModuleLoc, ModuleName,
                                           GlobalModuleFragment);
    assert(Mod && "module creation should not fail");
    break;
  }

  case ModuleDeclKind::Implementation:
    std::pair<IdentifierInfo *, SourceLocation> ModuleNameLoc(
        PP.getIdentifierInfo(ModuleName), Path[0].second);
    Mod = getModuleLoader().loadModule(ModuleLoc, {ModuleNameLoc},
                                       Module::AllVisible,
                                       /*IsIncludeDirective=*/false);
    if (!Mod) {
      Diag(ModuleLoc, diag::err_module_not_defined) << ModuleName;
      // Create an empty module interface unit for error recovery.
      Mod = Map.createModuleForInterfaceUnit(ModuleLoc, ModuleName,
                                             GlobalModuleFragment);
    }
    break;
  }

  if (!GlobalModuleFragment) {
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
  VisibleModules.setVisible(Mod, ModuleLoc);

  // From now on, we have an owning module for all declarations we see.
  // However, those declarations are module-private unless explicitly
  // exported.
  auto *TU = Context.getTranslationUnitDecl();
  TU->setModuleOwnershipKind(Decl::ModuleOwnershipKind::ModulePrivate);
  TU->setLocalOwningModule(Mod);

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
                                   SourceLocation ImportLoc,
                                   ModuleIdPath Path) {
  // Flatten the module path for a Modules TS module name.
  std::pair<IdentifierInfo *, SourceLocation> ModuleNameLoc;
  if (getLangOpts().ModulesTS) {
    std::string ModuleName;
    for (auto &Piece : Path) {
      if (!ModuleName.empty())
        ModuleName += ".";
      ModuleName += Piece.first->getName();
    }
    ModuleNameLoc = {PP.getIdentifierInfo(ModuleName), Path[0].second};
    Path = ModuleIdPath(ModuleNameLoc);
  }

  Module *Mod =
      getModuleLoader().loadModule(ImportLoc, Path, Module::AllVisible,
                                   /*IsIncludeDirective=*/false);
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
                                   SourceLocation ImportLoc,
                                   Module *Mod, ModuleIdPath Path) {
  VisibleModules.setVisible(Mod, ImportLoc);

  checkModuleImportContext(*this, Mod, ImportLoc, CurContext);

  // FIXME: we should support importing a submodule within a different submodule
  // of the same top-level module. Until we do, make it an error rather than
  // silently ignoring the import.
  // Import-from-implementation is valid in the Modules TS. FIXME: Should we
  // warn on a redundant import of the current module?
  // FIXME: Import of a module from an implementation partition of the same
  // module is permitted.
  if (Mod->getTopLevelModuleName() == getLangOpts().CurrentModule &&
      (getLangOpts().isCompilingModule() || !getLangOpts().ModulesTS)) {
    Diag(ImportLoc, getLangOpts().isCompilingModule()
                        ? diag::err_module_self_import
                        : diag::err_module_import_in_implementation)
        << Mod->getFullModuleName() << getLangOpts().CurrentModule;
  }

  SmallVector<SourceLocation, 2> IdentifierLocs;
  Module *ModCheck = Mod;
  for (unsigned I = 0, N = Path.size(); I != N; ++I) {
    // If we've run out of module parents, just drop the remaining identifiers.
    // We need the length to be consistent.
    if (!ModCheck)
      break;
    ModCheck = ModCheck->Parent;

    IdentifierLocs.push_back(Path[I].second);
  }

  // If this was a header import, pad out with dummy locations.
  // FIXME: Pass in and use the location of the header-name token in this case.
  if (Path.empty()) {
    for (; ModCheck; ModCheck = ModCheck->Parent) {
      IdentifierLocs.push_back(SourceLocation());
    }
  }

  ImportDecl *Import = ImportDecl::Create(Context, CurContext, StartLoc,
                                          Mod, IdentifierLocs);
  CurContext->addDecl(Import);

  // Sequence initialization of the imported module before that of the current
  // module, if any.
  if (!ModuleScopes.empty())
    Context.addModuleInitializer(ModuleScopes.back().Module, Import);

  // Re-export the module if needed.
  if (!ModuleScopes.empty() && ModuleScopes.back().ModuleInterface) {
    if (ExportLoc.isValid() || getEnclosingExportDecl(Import))
      getCurrentModule()->Exports.emplace_back(Mod, false);
  } else if (ExportLoc.isValid()) {
    Diag(ExportLoc, diag::err_export_not_in_module_interface);
  }

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

  // C++2a [module.interface]p1:
  //   An export-declaration shall appear only [...] in the purview of a module
  //   interface unit. An export-declaration shall not appear directly or
  //   indirectly within [...] a private-module-fragment.
  if (ModuleScopes.empty() || !ModuleScopes.back().Module->isModulePurview()) {
    Diag(ExportLoc, diag::err_export_not_in_module_interface) << 0;
  } else if (!ModuleScopes.back().ModuleInterface) {
    Diag(ExportLoc, diag::err_export_not_in_module_interface) << 1;
    Diag(ModuleScopes.back().BeginLoc,
         diag::note_not_module_interface_add_export)
        << FixItHint::CreateInsertion(ModuleScopes.back().BeginLoc, "export ");
  } else if (ModuleScopes.back().Module->Kind ==
             Module::PrivateModuleFragment) {
    Diag(ExportLoc, diag::err_export_in_private_module_fragment);
    Diag(ModuleScopes.back().BeginLoc, diag::note_private_module_fragment);
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
        break;
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
  }

  CurContext->addDecl(D);
  PushDeclContext(S, D);
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
  Context
};
}

static llvm::Optional<UnnamedDeclKind> getUnnamedDeclKind(Decl *D) {
  if (isa<EmptyDecl>(D))
    return UnnamedDeclKind::Empty;
  if (isa<StaticAssertDecl>(D))
    return UnnamedDeclKind::StaticAssert;
  if (isa<FileScopeAsmDecl>(D))
    return UnnamedDeclKind::Asm;
  if (isa<UsingDirectiveDecl>(D))
    return UnnamedDeclKind::UsingDirective;
  // Everything else either introduces one or more names or is ill-formed.
  return llvm::None;
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
  if (auto *ND = dyn_cast<NamedDecl>(D)) {
    // Don't diagnose anonymous union objects; we'll diagnose their members
    // instead.
    if (ND->getDeclName() && ND->getFormalLinkage() == InternalLinkage) {
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
    if (Target->getFormalLinkage() == InternalLinkage) {
      S.Diag(USD->getLocation(), diag::err_export_using_internal) << Target;
      S.Diag(Target->getLocation(), diag::note_using_decl_target);
      if (BlockStart.isValid())
        S.Diag(BlockStart, diag::note_export);
    }
  }

  // Recurse into namespace-scope DeclContexts. (Only namespace-scope
  // declarations are exported.)
  if (auto *DC = dyn_cast<DeclContext>(D))
    if (DC->getRedeclContext()->isFileContext() && !isa<EnumDecl>(D))
      return checkExportedDeclContext(S, DC, BlockStart);
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
    }
  }

  return D;
}
