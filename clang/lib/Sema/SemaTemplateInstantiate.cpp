//===------- SemaTemplateInstantiate.cpp - C++ Template Instantiation ------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation.
//
//===----------------------------------------------------------------------===/

#include "clang/Sema/SemaInternal.h"
#include "TreeTransform.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;
using namespace sema;

//===----------------------------------------------------------------------===/
// Template Instantiation Support
//===----------------------------------------------------------------------===/

/// \brief Retrieve the template argument list(s) that should be used to
/// instantiate the definition of the given declaration.
///
/// \param D the declaration for which we are computing template instantiation
/// arguments.
///
/// \param Innermost if non-NULL, the innermost template argument list.
///
/// \param RelativeToPrimary true if we should get the template
/// arguments relative to the primary template, even when we're
/// dealing with a specialization. This is only relevant for function
/// template specializations.
///
/// \param Pattern If non-NULL, indicates the pattern from which we will be
/// instantiating the definition of the given declaration, \p D. This is
/// used to determine the proper set of template instantiation arguments for
/// friend function template specializations.
MultiLevelTemplateArgumentList
Sema::getTemplateInstantiationArgs(NamedDecl *D, 
                                   const TemplateArgumentList *Innermost,
                                   bool RelativeToPrimary,
                                   const FunctionDecl *Pattern) {
  // Accumulate the set of template argument lists in this structure.
  MultiLevelTemplateArgumentList Result;

  if (Innermost)
    Result.addOuterTemplateArguments(Innermost);
  
  DeclContext *Ctx = dyn_cast<DeclContext>(D);
  if (!Ctx) {
    Ctx = D->getDeclContext();
    
    // If we have a template template parameter with translation unit context,
    // then we're performing substitution into a default template argument of
    // this template template parameter before we've constructed the template
    // that will own this template template parameter. In this case, we
    // use empty template parameter lists for all of the outer templates
    // to avoid performing any substitutions.
    if (Ctx->isTranslationUnit()) {
      if (TemplateTemplateParmDecl *TTP 
                                      = dyn_cast<TemplateTemplateParmDecl>(D)) {
        for (unsigned I = 0, N = TTP->getDepth() + 1; I != N; ++I)
          Result.addOuterTemplateArguments(0, 0);
        return Result;
      }
    }
  }
  
  while (!Ctx->isFileContext()) {
    // Add template arguments from a class template instantiation.
    if (ClassTemplateSpecializationDecl *Spec
          = dyn_cast<ClassTemplateSpecializationDecl>(Ctx)) {
      // We're done when we hit an explicit specialization.
      if (Spec->getSpecializationKind() == TSK_ExplicitSpecialization &&
          !isa<ClassTemplatePartialSpecializationDecl>(Spec))
        break;

      Result.addOuterTemplateArguments(&Spec->getTemplateInstantiationArgs());
      
      // If this class template specialization was instantiated from a 
      // specialized member that is a class template, we're done.
      assert(Spec->getSpecializedTemplate() && "No class template?");
      if (Spec->getSpecializedTemplate()->isMemberSpecialization())
        break;
    }
    // Add template arguments from a function template specialization.
    else if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Ctx)) {
      if (!RelativeToPrimary &&
          (Function->getTemplateSpecializationKind() == 
                                                  TSK_ExplicitSpecialization &&
           !Function->getClassScopeSpecializationPattern()))
        break;
          
      if (const TemplateArgumentList *TemplateArgs
            = Function->getTemplateSpecializationArgs()) {
        // Add the template arguments for this specialization.
        Result.addOuterTemplateArguments(TemplateArgs);

        // If this function was instantiated from a specialized member that is
        // a function template, we're done.
        assert(Function->getPrimaryTemplate() && "No function template?");
        if (Function->getPrimaryTemplate()->isMemberSpecialization())
          break;
      } else if (FunctionTemplateDecl *FunTmpl
                                   = Function->getDescribedFunctionTemplate()) {
        // Add the "injected" template arguments.
        std::pair<const TemplateArgument *, unsigned>
          Injected = FunTmpl->getInjectedTemplateArgs();
        Result.addOuterTemplateArguments(Injected.first, Injected.second);
      }
      
      // If this is a friend declaration and it declares an entity at
      // namespace scope, take arguments from its lexical parent
      // instead of its semantic parent, unless of course the pattern we're
      // instantiating actually comes from the file's context!
      if (Function->getFriendObjectKind() &&
          Function->getDeclContext()->isFileContext() &&
          (!Pattern || !Pattern->getLexicalDeclContext()->isFileContext())) {
        Ctx = Function->getLexicalDeclContext();
        RelativeToPrimary = false;
        continue;
      }
    } else if (CXXRecordDecl *Rec = dyn_cast<CXXRecordDecl>(Ctx)) {
      if (ClassTemplateDecl *ClassTemplate = Rec->getDescribedClassTemplate()) {
        QualType T = ClassTemplate->getInjectedClassNameSpecialization();
        const TemplateSpecializationType *TST
          = cast<TemplateSpecializationType>(Context.getCanonicalType(T));
        Result.addOuterTemplateArguments(TST->getArgs(), TST->getNumArgs());
        if (ClassTemplate->isMemberSpecialization())
          break;
      }
    }

    Ctx = Ctx->getParent();
    RelativeToPrimary = false;
  }

  return Result;
}

bool Sema::ActiveTemplateInstantiation::isInstantiationRecord() const {
  switch (Kind) {
  case TemplateInstantiation:
  case ExceptionSpecInstantiation:
  case DefaultTemplateArgumentInstantiation:
  case DefaultFunctionArgumentInstantiation:
  case ExplicitTemplateArgumentSubstitution:
  case DeducedTemplateArgumentSubstitution:
  case PriorTemplateArgumentSubstitution:
    return true;

  case DefaultTemplateArgumentChecking:
    return false;
  }

  llvm_unreachable("Invalid InstantiationKind!");
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      Decl *Entity,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                        SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = ActiveTemplateInstantiation::TemplateInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Entity);
    Inst.TemplateArgs = 0;
    Inst.NumTemplateArgs = 0;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.InNonInstantiationSFINAEContext = false;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      FunctionDecl *Entity, ExceptionSpecification,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                        SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = ActiveTemplateInstantiation::ExceptionSpecInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Entity);
    Inst.TemplateArgs = 0;
    Inst.NumTemplateArgs = 0;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.InNonInstantiationSFINAEContext = false;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      TemplateDecl *Template,
                      ArrayRef<TemplateArgument> TemplateArgs,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                     SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind
      = ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Template);
    Inst.TemplateArgs = TemplateArgs.data();
    Inst.NumTemplateArgs = TemplateArgs.size();
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.InNonInstantiationSFINAEContext = false;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      FunctionTemplateDecl *FunctionTemplate,
                      ArrayRef<TemplateArgument> TemplateArgs,
                      ActiveTemplateInstantiation::InstantiationKind Kind,
                      sema::TemplateDeductionInfo &DeductionInfo,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                     SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = CheckInstantiationDepth(PointOfInstantiation, InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = Kind;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(FunctionTemplate);
    Inst.TemplateArgs = TemplateArgs.data();
    Inst.NumTemplateArgs = TemplateArgs.size();
    Inst.DeductionInfo = &DeductionInfo;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.InNonInstantiationSFINAEContext = false;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
    
    if (!Inst.isInstantiationRecord())
      ++SemaRef.NonInstantiationEntries;
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      ClassTemplatePartialSpecializationDecl *PartialSpec,
                      ArrayRef<TemplateArgument> TemplateArgs,
                      sema::TemplateDeductionInfo &DeductionInfo,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                     SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = CheckInstantiationDepth(PointOfInstantiation, InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = ActiveTemplateInstantiation::DeducedTemplateArgumentSubstitution;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(PartialSpec);
    Inst.TemplateArgs = TemplateArgs.data();
    Inst.NumTemplateArgs = TemplateArgs.size();
    Inst.DeductionInfo = &DeductionInfo;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.InNonInstantiationSFINAEContext = false;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      ParmVarDecl *Param,
                      ArrayRef<TemplateArgument> TemplateArgs,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                     SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = CheckInstantiationDepth(PointOfInstantiation, InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind
      = ActiveTemplateInstantiation::DefaultFunctionArgumentInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Param);
    Inst.TemplateArgs = TemplateArgs.data();
    Inst.NumTemplateArgs = TemplateArgs.size();
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.InNonInstantiationSFINAEContext = false;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      NamedDecl *Template, NonTypeTemplateParmDecl *Param,
                      ArrayRef<TemplateArgument> TemplateArgs,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                     SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = CheckInstantiationDepth(PointOfInstantiation, InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = ActiveTemplateInstantiation::PriorTemplateArgumentSubstitution;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Template = Template;
    Inst.Entity = reinterpret_cast<uintptr_t>(Param);
    Inst.TemplateArgs = TemplateArgs.data();
    Inst.NumTemplateArgs = TemplateArgs.size();
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.InNonInstantiationSFINAEContext = false;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      NamedDecl *Template, TemplateTemplateParmDecl *Param,
                      ArrayRef<TemplateArgument> TemplateArgs,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                     SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = CheckInstantiationDepth(PointOfInstantiation, InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = ActiveTemplateInstantiation::PriorTemplateArgumentSubstitution;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Template = Template;
    Inst.Entity = reinterpret_cast<uintptr_t>(Param);
    Inst.TemplateArgs = TemplateArgs.data();
    Inst.NumTemplateArgs = TemplateArgs.size();
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.InNonInstantiationSFINAEContext = false;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      TemplateDecl *Template, NamedDecl *Param,
                      ArrayRef<TemplateArgument> TemplateArgs,
                      SourceRange InstantiationRange)
  : SemaRef(SemaRef),
    SavedInNonInstantiationSFINAEContext(
                                     SemaRef.InNonInstantiationSFINAEContext)
{
  Invalid = false;
  
  ActiveTemplateInstantiation Inst;
  Inst.Kind = ActiveTemplateInstantiation::DefaultTemplateArgumentChecking;
  Inst.PointOfInstantiation = PointOfInstantiation;
  Inst.Template = Template;
  Inst.Entity = reinterpret_cast<uintptr_t>(Param);
  Inst.TemplateArgs = TemplateArgs.data();
  Inst.NumTemplateArgs = TemplateArgs.size();
  Inst.InstantiationRange = InstantiationRange;
  SemaRef.InNonInstantiationSFINAEContext = false;
  SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  
  assert(!Inst.isInstantiationRecord());
  ++SemaRef.NonInstantiationEntries;
}

void Sema::InstantiatingTemplate::Clear() {
  if (!Invalid) {
    if (!SemaRef.ActiveTemplateInstantiations.back().isInstantiationRecord()) {
      assert(SemaRef.NonInstantiationEntries > 0);
      --SemaRef.NonInstantiationEntries;
    }
    SemaRef.InNonInstantiationSFINAEContext
      = SavedInNonInstantiationSFINAEContext;
    SemaRef.ActiveTemplateInstantiations.pop_back();
    Invalid = true;
  }
}

bool Sema::InstantiatingTemplate::CheckInstantiationDepth(
                                        SourceLocation PointOfInstantiation,
                                           SourceRange InstantiationRange) {
  assert(SemaRef.NonInstantiationEntries <=
                                   SemaRef.ActiveTemplateInstantiations.size());
  if ((SemaRef.ActiveTemplateInstantiations.size() - 
          SemaRef.NonInstantiationEntries)
        <= SemaRef.getLangOpts().InstantiationDepth)
    return false;

  SemaRef.Diag(PointOfInstantiation,
               diag::err_template_recursion_depth_exceeded)
    << SemaRef.getLangOpts().InstantiationDepth
    << InstantiationRange;
  SemaRef.Diag(PointOfInstantiation, diag::note_template_recursion_depth)
    << SemaRef.getLangOpts().InstantiationDepth;
  return true;
}

/// \brief Prints the current instantiation stack through a series of
/// notes.
void Sema::PrintInstantiationStack() {
  // Determine which template instantiations to skip, if any.
  unsigned SkipStart = ActiveTemplateInstantiations.size(), SkipEnd = SkipStart;
  unsigned Limit = Diags.getTemplateBacktraceLimit();
  if (Limit && Limit < ActiveTemplateInstantiations.size()) {
    SkipStart = Limit / 2 + Limit % 2;
    SkipEnd = ActiveTemplateInstantiations.size() - Limit / 2;
  }

  // FIXME: In all of these cases, we need to show the template arguments
  unsigned InstantiationIdx = 0;
  for (SmallVector<ActiveTemplateInstantiation, 16>::reverse_iterator
         Active = ActiveTemplateInstantiations.rbegin(),
         ActiveEnd = ActiveTemplateInstantiations.rend();
       Active != ActiveEnd;
       ++Active, ++InstantiationIdx) {
    // Skip this instantiation?
    if (InstantiationIdx >= SkipStart && InstantiationIdx < SkipEnd) {
      if (InstantiationIdx == SkipStart) {
        // Note that we're skipping instantiations.
        Diags.Report(Active->PointOfInstantiation,
                     diag::note_instantiation_contexts_suppressed)
          << unsigned(ActiveTemplateInstantiations.size() - Limit);
      }
      continue;
    }

    switch (Active->Kind) {
    case ActiveTemplateInstantiation::TemplateInstantiation: {
      Decl *D = reinterpret_cast<Decl *>(Active->Entity);
      if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(D)) {
        unsigned DiagID = diag::note_template_member_class_here;
        if (isa<ClassTemplateSpecializationDecl>(Record))
          DiagID = diag::note_template_class_instantiation_here;
        Diags.Report(Active->PointOfInstantiation, DiagID)
          << Context.getTypeDeclType(Record)
          << Active->InstantiationRange;
      } else if (FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
        unsigned DiagID;
        if (Function->getPrimaryTemplate())
          DiagID = diag::note_function_template_spec_here;
        else
          DiagID = diag::note_template_member_function_here;
        Diags.Report(Active->PointOfInstantiation, DiagID)
          << Function
          << Active->InstantiationRange;
      } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
        Diags.Report(Active->PointOfInstantiation,
                     diag::note_template_static_data_member_def_here)
          << VD
          << Active->InstantiationRange;
      } else if (EnumDecl *ED = dyn_cast<EnumDecl>(D)) {
        Diags.Report(Active->PointOfInstantiation,
                     diag::note_template_enum_def_here)
          << ED
          << Active->InstantiationRange;
      } else {
        Diags.Report(Active->PointOfInstantiation,
                     diag::note_template_type_alias_instantiation_here)
          << cast<TypeAliasTemplateDecl>(D)
          << Active->InstantiationRange;
      }
      break;
    }

    case ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation: {
      TemplateDecl *Template = cast<TemplateDecl>((Decl *)Active->Entity);
      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                                         Active->TemplateArgs,
                                                      Active->NumTemplateArgs,
                                                      getPrintingPolicy());
      Diags.Report(Active->PointOfInstantiation,
                   diag::note_default_arg_instantiation_here)
        << (Template->getNameAsString() + TemplateArgsStr)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::ExplicitTemplateArgumentSubstitution: {
      FunctionTemplateDecl *FnTmpl
        = cast<FunctionTemplateDecl>((Decl *)Active->Entity);
      Diags.Report(Active->PointOfInstantiation,
                   diag::note_explicit_template_arg_substitution_here)
        << FnTmpl 
        << getTemplateArgumentBindingsText(FnTmpl->getTemplateParameters(), 
                                           Active->TemplateArgs, 
                                           Active->NumTemplateArgs)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::DeducedTemplateArgumentSubstitution:
      if (ClassTemplatePartialSpecializationDecl *PartialSpec
            = dyn_cast<ClassTemplatePartialSpecializationDecl>(
                                                    (Decl *)Active->Entity)) {
        Diags.Report(Active->PointOfInstantiation,
                     diag::note_partial_spec_deduct_instantiation_here)
          << Context.getTypeDeclType(PartialSpec)
          << getTemplateArgumentBindingsText(
                                         PartialSpec->getTemplateParameters(), 
                                             Active->TemplateArgs, 
                                             Active->NumTemplateArgs)
          << Active->InstantiationRange;
      } else {
        FunctionTemplateDecl *FnTmpl
          = cast<FunctionTemplateDecl>((Decl *)Active->Entity);
        Diags.Report(Active->PointOfInstantiation,
                     diag::note_function_template_deduction_instantiation_here)
          << FnTmpl
          << getTemplateArgumentBindingsText(FnTmpl->getTemplateParameters(), 
                                             Active->TemplateArgs, 
                                             Active->NumTemplateArgs)
          << Active->InstantiationRange;
      }
      break;

    case ActiveTemplateInstantiation::DefaultFunctionArgumentInstantiation: {
      ParmVarDecl *Param = cast<ParmVarDecl>((Decl *)Active->Entity);
      FunctionDecl *FD = cast<FunctionDecl>(Param->getDeclContext());

      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                                         Active->TemplateArgs,
                                                      Active->NumTemplateArgs,
                                                      getPrintingPolicy());
      Diags.Report(Active->PointOfInstantiation,
                   diag::note_default_function_arg_instantiation_here)
        << (FD->getNameAsString() + TemplateArgsStr)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::PriorTemplateArgumentSubstitution: {
      NamedDecl *Parm = cast<NamedDecl>((Decl *)Active->Entity);
      std::string Name;
      if (!Parm->getName().empty())
        Name = std::string(" '") + Parm->getName().str() + "'";
                    
      TemplateParameterList *TemplateParams = 0;
      if (TemplateDecl *Template = dyn_cast<TemplateDecl>(Active->Template))
        TemplateParams = Template->getTemplateParameters();
      else
        TemplateParams =
          cast<ClassTemplatePartialSpecializationDecl>(Active->Template)
                                                      ->getTemplateParameters();
      Diags.Report(Active->PointOfInstantiation,
                   diag::note_prior_template_arg_substitution)
        << isa<TemplateTemplateParmDecl>(Parm)
        << Name
        << getTemplateArgumentBindingsText(TemplateParams, 
                                           Active->TemplateArgs, 
                                           Active->NumTemplateArgs)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::DefaultTemplateArgumentChecking: {
      TemplateParameterList *TemplateParams = 0;
      if (TemplateDecl *Template = dyn_cast<TemplateDecl>(Active->Template))
        TemplateParams = Template->getTemplateParameters();
      else
        TemplateParams =
          cast<ClassTemplatePartialSpecializationDecl>(Active->Template)
                                                      ->getTemplateParameters();

      Diags.Report(Active->PointOfInstantiation,
                   diag::note_template_default_arg_checking)
        << getTemplateArgumentBindingsText(TemplateParams, 
                                           Active->TemplateArgs, 
                                           Active->NumTemplateArgs)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::ExceptionSpecInstantiation:
      Diags.Report(Active->PointOfInstantiation,
                   diag::note_template_exception_spec_instantiation_here)
        << cast<FunctionDecl>((Decl *)Active->Entity)
        << Active->InstantiationRange;
      break;
    }
  }
}

llvm::Optional<TemplateDeductionInfo *> Sema::isSFINAEContext() const {
  if (InNonInstantiationSFINAEContext)
    return llvm::Optional<TemplateDeductionInfo *>(0);

  for (SmallVector<ActiveTemplateInstantiation, 16>::const_reverse_iterator
         Active = ActiveTemplateInstantiations.rbegin(),
         ActiveEnd = ActiveTemplateInstantiations.rend();
       Active != ActiveEnd;
       ++Active) 
  {
    switch(Active->Kind) {
    case ActiveTemplateInstantiation::TemplateInstantiation:
      // An instantiation of an alias template may or may not be a SFINAE
      // context, depending on what else is on the stack.
      if (isa<TypeAliasTemplateDecl>(reinterpret_cast<Decl *>(Active->Entity)))
        break;
      // Fall through.
    case ActiveTemplateInstantiation::DefaultFunctionArgumentInstantiation:
    case ActiveTemplateInstantiation::ExceptionSpecInstantiation:
      // This is a template instantiation, so there is no SFINAE.
      return llvm::Optional<TemplateDeductionInfo *>();

    case ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation:
    case ActiveTemplateInstantiation::PriorTemplateArgumentSubstitution:
    case ActiveTemplateInstantiation::DefaultTemplateArgumentChecking:
      // A default template argument instantiation and substitution into
      // template parameters with arguments for prior parameters may or may 
      // not be a SFINAE context; look further up the stack.
      break;

    case ActiveTemplateInstantiation::ExplicitTemplateArgumentSubstitution:
    case ActiveTemplateInstantiation::DeducedTemplateArgumentSubstitution:
      // We're either substitution explicitly-specified template arguments
      // or deduced template arguments, so SFINAE applies.
      assert(Active->DeductionInfo && "Missing deduction info pointer");
      return Active->DeductionInfo;
    }
  }

  return llvm::Optional<TemplateDeductionInfo *>();
}

/// \brief Retrieve the depth and index of a parameter pack.
static std::pair<unsigned, unsigned> 
getDepthAndIndex(NamedDecl *ND) {
  if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(ND))
    return std::make_pair(TTP->getDepth(), TTP->getIndex());
  
  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(ND))
    return std::make_pair(NTTP->getDepth(), NTTP->getIndex());
  
  TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(ND);
  return std::make_pair(TTP->getDepth(), TTP->getIndex());
}

//===----------------------------------------------------------------------===/
// Template Instantiation for Types
//===----------------------------------------------------------------------===/
namespace {
  class TemplateInstantiator : public TreeTransform<TemplateInstantiator> {
    const MultiLevelTemplateArgumentList &TemplateArgs;
    SourceLocation Loc;
    DeclarationName Entity;

  public:
    typedef TreeTransform<TemplateInstantiator> inherited;

    TemplateInstantiator(Sema &SemaRef,
                         const MultiLevelTemplateArgumentList &TemplateArgs,
                         SourceLocation Loc,
                         DeclarationName Entity)
      : inherited(SemaRef), TemplateArgs(TemplateArgs), Loc(Loc),
        Entity(Entity) { }

    /// \brief Determine whether the given type \p T has already been
    /// transformed.
    ///
    /// For the purposes of template instantiation, a type has already been
    /// transformed if it is NULL or if it is not dependent.
    bool AlreadyTransformed(QualType T);

    /// \brief Returns the location of the entity being instantiated, if known.
    SourceLocation getBaseLocation() { return Loc; }

    /// \brief Returns the name of the entity being instantiated, if any.
    DeclarationName getBaseEntity() { return Entity; }

    /// \brief Sets the "base" location and entity when that
    /// information is known based on another transformation.
    void setBase(SourceLocation Loc, DeclarationName Entity) {
      this->Loc = Loc;
      this->Entity = Entity;
    }

    bool TryExpandParameterPacks(SourceLocation EllipsisLoc,
                                 SourceRange PatternRange,
                             llvm::ArrayRef<UnexpandedParameterPack> Unexpanded,
                                 bool &ShouldExpand,
                                 bool &RetainExpansion,
                                 llvm::Optional<unsigned> &NumExpansions) {
      return getSema().CheckParameterPacksForExpansion(EllipsisLoc, 
                                                       PatternRange, Unexpanded,
                                                       TemplateArgs, 
                                                       ShouldExpand,
                                                       RetainExpansion,
                                                       NumExpansions);
    }

    void ExpandingFunctionParameterPack(ParmVarDecl *Pack) { 
      SemaRef.CurrentInstantiationScope->MakeInstantiatedLocalArgPack(Pack);
    }
    
    TemplateArgument ForgetPartiallySubstitutedPack() {
      TemplateArgument Result;
      if (NamedDecl *PartialPack
            = SemaRef.CurrentInstantiationScope->getPartiallySubstitutedPack()){
        MultiLevelTemplateArgumentList &TemplateArgs
          = const_cast<MultiLevelTemplateArgumentList &>(this->TemplateArgs);
        unsigned Depth, Index;
        llvm::tie(Depth, Index) = getDepthAndIndex(PartialPack);
        if (TemplateArgs.hasTemplateArgument(Depth, Index)) {
          Result = TemplateArgs(Depth, Index);
          TemplateArgs.setArgument(Depth, Index, TemplateArgument());
        }
      }
      
      return Result;
    }
    
    void RememberPartiallySubstitutedPack(TemplateArgument Arg) {
      if (Arg.isNull())
        return;
      
      if (NamedDecl *PartialPack
            = SemaRef.CurrentInstantiationScope->getPartiallySubstitutedPack()){
        MultiLevelTemplateArgumentList &TemplateArgs
        = const_cast<MultiLevelTemplateArgumentList &>(this->TemplateArgs);
        unsigned Depth, Index;
        llvm::tie(Depth, Index) = getDepthAndIndex(PartialPack);
        TemplateArgs.setArgument(Depth, Index, Arg);
      }
    }

    /// \brief Transform the given declaration by instantiating a reference to
    /// this declaration.
    Decl *TransformDecl(SourceLocation Loc, Decl *D);

    void transformAttrs(Decl *Old, Decl *New) { 
      SemaRef.InstantiateAttrs(TemplateArgs, Old, New);
    }

    void transformedLocalDecl(Decl *Old, Decl *New) {
      SemaRef.CurrentInstantiationScope->InstantiatedLocal(Old, New);
    }
    
    /// \brief Transform the definition of the given declaration by
    /// instantiating it.
    Decl *TransformDefinition(SourceLocation Loc, Decl *D);

    /// \brief Transform the first qualifier within a scope by instantiating the
    /// declaration.
    NamedDecl *TransformFirstQualifierInScope(NamedDecl *D, SourceLocation Loc);
      
    /// \brief Rebuild the exception declaration and register the declaration
    /// as an instantiated local.
    VarDecl *RebuildExceptionDecl(VarDecl *ExceptionDecl, 
                                  TypeSourceInfo *Declarator,
                                  SourceLocation StartLoc,
                                  SourceLocation NameLoc,
                                  IdentifierInfo *Name);

    /// \brief Rebuild the Objective-C exception declaration and register the 
    /// declaration as an instantiated local.
    VarDecl *RebuildObjCExceptionDecl(VarDecl *ExceptionDecl, 
                                      TypeSourceInfo *TSInfo, QualType T);
      
    /// \brief Check for tag mismatches when instantiating an
    /// elaborated type.
    QualType RebuildElaboratedType(SourceLocation KeywordLoc,
                                   ElaboratedTypeKeyword Keyword,
                                   NestedNameSpecifierLoc QualifierLoc,
                                   QualType T);

    TemplateName TransformTemplateName(CXXScopeSpec &SS,
                                       TemplateName Name,
                                       SourceLocation NameLoc,                                     
                                       QualType ObjectType = QualType(),
                                       NamedDecl *FirstQualifierInScope = 0);

    ExprResult TransformPredefinedExpr(PredefinedExpr *E);
    ExprResult TransformDeclRefExpr(DeclRefExpr *E);
    ExprResult TransformCXXDefaultArgExpr(CXXDefaultArgExpr *E);

    ExprResult TransformTemplateParmRefExpr(DeclRefExpr *E,
                                            NonTypeTemplateParmDecl *D);
    ExprResult TransformSubstNonTypeTemplateParmPackExpr(
                                           SubstNonTypeTemplateParmPackExpr *E);

    /// \brief Rebuild a DeclRefExpr for a ParmVarDecl reference.
    ExprResult RebuildParmVarDeclRefExpr(ParmVarDecl *PD, SourceLocation Loc);

    /// \brief Transform a reference to a function parameter pack.
    ExprResult TransformFunctionParmPackRefExpr(DeclRefExpr *E,
                                                ParmVarDecl *PD);

    /// \brief Transform a FunctionParmPackExpr which was built when we couldn't
    /// expand a function parameter pack reference which refers to an expanded
    /// pack.
    ExprResult TransformFunctionParmPackExpr(FunctionParmPackExpr *E);

    QualType TransformFunctionProtoType(TypeLocBuilder &TLB,
                                        FunctionProtoTypeLoc TL);
    QualType TransformFunctionProtoType(TypeLocBuilder &TLB,
                                        FunctionProtoTypeLoc TL,
                                        CXXRecordDecl *ThisContext,
                                        unsigned ThisTypeQuals);

    ParmVarDecl *TransformFunctionTypeParam(ParmVarDecl *OldParm,
                                            int indexAdjustment,
                                        llvm::Optional<unsigned> NumExpansions,
                                            bool ExpectParameterPack);

    /// \brief Transforms a template type parameter type by performing
    /// substitution of the corresponding template type argument.
    QualType TransformTemplateTypeParmType(TypeLocBuilder &TLB,
                                           TemplateTypeParmTypeLoc TL);

    /// \brief Transforms an already-substituted template type parameter pack
    /// into either itself (if we aren't substituting into its pack expansion)
    /// or the appropriate substituted argument.
    QualType TransformSubstTemplateTypeParmPackType(TypeLocBuilder &TLB,
                                           SubstTemplateTypeParmPackTypeLoc TL);

    ExprResult TransformCallExpr(CallExpr *CE) {
      getSema().CallsUndergoingInstantiation.push_back(CE);
      ExprResult Result =
          TreeTransform<TemplateInstantiator>::TransformCallExpr(CE);
      getSema().CallsUndergoingInstantiation.pop_back();
      return Result;
    }

    ExprResult TransformLambdaExpr(LambdaExpr *E) {
      LocalInstantiationScope Scope(SemaRef, /*CombineWithOuterScope=*/true);
      return TreeTransform<TemplateInstantiator>::TransformLambdaExpr(E);
    }

    ExprResult TransformLambdaScope(LambdaExpr *E,
                                    CXXMethodDecl *CallOperator) {
      CallOperator->setInstantiationOfMemberFunction(E->getCallOperator(),
                                                     TSK_ImplicitInstantiation);
      return TreeTransform<TemplateInstantiator>::
         TransformLambdaScope(E, CallOperator);
    }

  private:
    ExprResult transformNonTypeTemplateParmRef(NonTypeTemplateParmDecl *parm,
                                               SourceLocation loc,
                                               TemplateArgument arg);
  };
}

bool TemplateInstantiator::AlreadyTransformed(QualType T) {
  if (T.isNull())
    return true;
  
  if (T->isInstantiationDependentType() || T->isVariablyModifiedType())
    return false;
  
  getSema().MarkDeclarationsReferencedInType(Loc, T);
  return true;
}

Decl *TemplateInstantiator::TransformDecl(SourceLocation Loc, Decl *D) {
  if (!D)
    return 0;

  if (TemplateTemplateParmDecl *TTP = dyn_cast<TemplateTemplateParmDecl>(D)) {
    if (TTP->getDepth() < TemplateArgs.getNumLevels()) {
      // If the corresponding template argument is NULL or non-existent, it's
      // because we are performing instantiation from explicitly-specified
      // template arguments in a function template, but there were some
      // arguments left unspecified.
      if (!TemplateArgs.hasTemplateArgument(TTP->getDepth(),
                                            TTP->getPosition()))
        return D;

      TemplateArgument Arg = TemplateArgs(TTP->getDepth(), TTP->getPosition());
      
      if (TTP->isParameterPack()) {
        assert(Arg.getKind() == TemplateArgument::Pack && 
               "Missing argument pack");
        
        assert(getSema().ArgumentPackSubstitutionIndex >= 0);        
        assert(getSema().ArgumentPackSubstitutionIndex < (int)Arg.pack_size());
        Arg = Arg.pack_begin()[getSema().ArgumentPackSubstitutionIndex];
      }

      TemplateName Template = Arg.getAsTemplate();
      assert(!Template.isNull() && Template.getAsTemplateDecl() &&
             "Wrong kind of template template argument");
      return Template.getAsTemplateDecl();
    }

    // Fall through to find the instantiated declaration for this template
    // template parameter.
  }

  return SemaRef.FindInstantiatedDecl(Loc, cast<NamedDecl>(D), TemplateArgs);
}

Decl *TemplateInstantiator::TransformDefinition(SourceLocation Loc, Decl *D) {
  Decl *Inst = getSema().SubstDecl(D, getSema().CurContext, TemplateArgs);
  if (!Inst)
    return 0;

  getSema().CurrentInstantiationScope->InstantiatedLocal(D, Inst);
  return Inst;
}

NamedDecl *
TemplateInstantiator::TransformFirstQualifierInScope(NamedDecl *D, 
                                                     SourceLocation Loc) {
  // If the first part of the nested-name-specifier was a template type 
  // parameter, instantiate that type parameter down to a tag type.
  if (TemplateTypeParmDecl *TTPD = dyn_cast_or_null<TemplateTypeParmDecl>(D)) {
    const TemplateTypeParmType *TTP 
      = cast<TemplateTypeParmType>(getSema().Context.getTypeDeclType(TTPD));
    
    if (TTP->getDepth() < TemplateArgs.getNumLevels()) {
      // FIXME: This needs testing w/ member access expressions.
      TemplateArgument Arg = TemplateArgs(TTP->getDepth(), TTP->getIndex());
      
      if (TTP->isParameterPack()) {
        assert(Arg.getKind() == TemplateArgument::Pack && 
               "Missing argument pack");
        
        if (getSema().ArgumentPackSubstitutionIndex == -1)
          return 0;
        
        assert(getSema().ArgumentPackSubstitutionIndex < (int)Arg.pack_size());
        Arg = Arg.pack_begin()[getSema().ArgumentPackSubstitutionIndex];
      }

      QualType T = Arg.getAsType();
      if (T.isNull())
        return cast_or_null<NamedDecl>(TransformDecl(Loc, D));
      
      if (const TagType *Tag = T->getAs<TagType>())
        return Tag->getDecl();
      
      // The resulting type is not a tag; complain.
      getSema().Diag(Loc, diag::err_nested_name_spec_non_tag) << T;
      return 0;
    }
  }
  
  return cast_or_null<NamedDecl>(TransformDecl(Loc, D));
}

VarDecl *
TemplateInstantiator::RebuildExceptionDecl(VarDecl *ExceptionDecl,
                                           TypeSourceInfo *Declarator,
                                           SourceLocation StartLoc,
                                           SourceLocation NameLoc,
                                           IdentifierInfo *Name) {
  VarDecl *Var = inherited::RebuildExceptionDecl(ExceptionDecl, Declarator,
                                                 StartLoc, NameLoc, Name);
  if (Var)
    getSema().CurrentInstantiationScope->InstantiatedLocal(ExceptionDecl, Var);
  return Var;
}

VarDecl *TemplateInstantiator::RebuildObjCExceptionDecl(VarDecl *ExceptionDecl, 
                                                        TypeSourceInfo *TSInfo, 
                                                        QualType T) {
  VarDecl *Var = inherited::RebuildObjCExceptionDecl(ExceptionDecl, TSInfo, T);
  if (Var)
    getSema().CurrentInstantiationScope->InstantiatedLocal(ExceptionDecl, Var);
  return Var;
}

QualType
TemplateInstantiator::RebuildElaboratedType(SourceLocation KeywordLoc,
                                            ElaboratedTypeKeyword Keyword,
                                            NestedNameSpecifierLoc QualifierLoc,
                                            QualType T) {
  if (const TagType *TT = T->getAs<TagType>()) {
    TagDecl* TD = TT->getDecl();

    SourceLocation TagLocation = KeywordLoc;

    IdentifierInfo *Id = TD->getIdentifier();

    // TODO: should we even warn on struct/class mismatches for this?  Seems
    // like it's likely to produce a lot of spurious errors.
    if (Id && Keyword != ETK_None && Keyword != ETK_Typename) {
      TagTypeKind Kind = TypeWithKeyword::getTagTypeKindForKeyword(Keyword);
      if (!SemaRef.isAcceptableTagRedeclaration(TD, Kind, /*isDefinition*/false,
                                                TagLocation, *Id)) {
        SemaRef.Diag(TagLocation, diag::err_use_with_wrong_tag)
          << Id
          << FixItHint::CreateReplacement(SourceRange(TagLocation),
                                          TD->getKindName());
        SemaRef.Diag(TD->getLocation(), diag::note_previous_use);
      }
    }
  }

  return TreeTransform<TemplateInstantiator>::RebuildElaboratedType(KeywordLoc,
                                                                    Keyword,
                                                                  QualifierLoc,
                                                                    T);
}

TemplateName TemplateInstantiator::TransformTemplateName(CXXScopeSpec &SS,
                                                         TemplateName Name,
                                                         SourceLocation NameLoc,                                     
                                                         QualType ObjectType,
                                             NamedDecl *FirstQualifierInScope) {
  if (TemplateTemplateParmDecl *TTP
       = dyn_cast_or_null<TemplateTemplateParmDecl>(Name.getAsTemplateDecl())) {
    if (TTP->getDepth() < TemplateArgs.getNumLevels()) {
      // If the corresponding template argument is NULL or non-existent, it's
      // because we are performing instantiation from explicitly-specified
      // template arguments in a function template, but there were some
      // arguments left unspecified.
      if (!TemplateArgs.hasTemplateArgument(TTP->getDepth(),
                                            TTP->getPosition()))
        return Name;
      
      TemplateArgument Arg = TemplateArgs(TTP->getDepth(), TTP->getPosition());
      
      if (TTP->isParameterPack()) {
        assert(Arg.getKind() == TemplateArgument::Pack && 
               "Missing argument pack");
        
        if (getSema().ArgumentPackSubstitutionIndex == -1) {
          // We have the template argument pack to substitute, but we're not
          // actually expanding the enclosing pack expansion yet. So, just
          // keep the entire argument pack.
          return getSema().Context.getSubstTemplateTemplateParmPack(TTP, Arg);
        }
        
        assert(getSema().ArgumentPackSubstitutionIndex < (int)Arg.pack_size());
        Arg = Arg.pack_begin()[getSema().ArgumentPackSubstitutionIndex];
      }
      
      TemplateName Template = Arg.getAsTemplate();
      assert(!Template.isNull() && "Null template template argument");

      // We don't ever want to substitute for a qualified template name, since
      // the qualifier is handled separately. So, look through the qualified
      // template name to its underlying declaration.
      if (QualifiedTemplateName *QTN = Template.getAsQualifiedTemplateName())
        Template = TemplateName(QTN->getTemplateDecl());

      Template = getSema().Context.getSubstTemplateTemplateParm(TTP, Template);
      return Template;
    }
  }
  
  if (SubstTemplateTemplateParmPackStorage *SubstPack
      = Name.getAsSubstTemplateTemplateParmPack()) {
    if (getSema().ArgumentPackSubstitutionIndex == -1)
      return Name;
    
    const TemplateArgument &ArgPack = SubstPack->getArgumentPack();
    assert(getSema().ArgumentPackSubstitutionIndex < (int)ArgPack.pack_size() &&
           "Pack substitution index out-of-range");
    return ArgPack.pack_begin()[getSema().ArgumentPackSubstitutionIndex]
    .getAsTemplate();
  }
  
  return inherited::TransformTemplateName(SS, Name, NameLoc, ObjectType, 
                                          FirstQualifierInScope);  
}

ExprResult 
TemplateInstantiator::TransformPredefinedExpr(PredefinedExpr *E) {
  if (!E->isTypeDependent())
    return SemaRef.Owned(E);

  FunctionDecl *currentDecl = getSema().getCurFunctionDecl();
  assert(currentDecl && "Must have current function declaration when "
                        "instantiating.");

  PredefinedExpr::IdentType IT = E->getIdentType();

  unsigned Length = PredefinedExpr::ComputeName(IT, currentDecl).length();

  llvm::APInt LengthI(32, Length + 1);
  QualType ResTy;
  if (IT == PredefinedExpr::LFunction)
    ResTy = getSema().Context.WCharTy.withConst();
  else
    ResTy = getSema().Context.CharTy.withConst();
  ResTy = getSema().Context.getConstantArrayType(ResTy, LengthI, 
                                                 ArrayType::Normal, 0);
  PredefinedExpr *PE =
    new (getSema().Context) PredefinedExpr(E->getLocation(), ResTy, IT);
  return getSema().Owned(PE);
}

ExprResult
TemplateInstantiator::TransformTemplateParmRefExpr(DeclRefExpr *E,
                                               NonTypeTemplateParmDecl *NTTP) {
  // If the corresponding template argument is NULL or non-existent, it's
  // because we are performing instantiation from explicitly-specified
  // template arguments in a function template, but there were some
  // arguments left unspecified.
  if (!TemplateArgs.hasTemplateArgument(NTTP->getDepth(),
                                        NTTP->getPosition()))
    return SemaRef.Owned(E);

  TemplateArgument Arg = TemplateArgs(NTTP->getDepth(), NTTP->getPosition());
  if (NTTP->isParameterPack()) {
    assert(Arg.getKind() == TemplateArgument::Pack && 
           "Missing argument pack");
    
    if (getSema().ArgumentPackSubstitutionIndex == -1) {
      // We have an argument pack, but we can't select a particular argument
      // out of it yet. Therefore, we'll build an expression to hold on to that
      // argument pack.
      QualType TargetType = SemaRef.SubstType(NTTP->getType(), TemplateArgs,
                                              E->getLocation(), 
                                              NTTP->getDeclName());
      if (TargetType.isNull())
        return ExprError();
      
      return new (SemaRef.Context) SubstNonTypeTemplateParmPackExpr(TargetType,
                                                                    NTTP, 
                                                              E->getLocation(),
                                                                    Arg);
    }
    
    assert(getSema().ArgumentPackSubstitutionIndex < (int)Arg.pack_size());
    Arg = Arg.pack_begin()[getSema().ArgumentPackSubstitutionIndex];
  }

  return transformNonTypeTemplateParmRef(NTTP, E->getLocation(), Arg);
}

ExprResult TemplateInstantiator::transformNonTypeTemplateParmRef(
                                                 NonTypeTemplateParmDecl *parm,
                                                 SourceLocation loc,
                                                 TemplateArgument arg) {
  ExprResult result;
  QualType type;

  // If the argument is a pack expansion, the parameter must actually be a
  // parameter pack, and we should substitute the pattern itself, producing
  // an expression which contains an unexpanded parameter pack.
  if (arg.isPackExpansion()) {
    assert(parm->isParameterPack() && "pack expansion for non-pack");
    arg = arg.getPackExpansionPattern();
  }

  // The template argument itself might be an expression, in which
  // case we just return that expression.
  if (arg.getKind() == TemplateArgument::Expression) {
    Expr *argExpr = arg.getAsExpr();
    result = SemaRef.Owned(argExpr);
    type = argExpr->getType();

  } else if (arg.getKind() == TemplateArgument::Declaration ||
             arg.getKind() == TemplateArgument::NullPtr) {
    ValueDecl *VD;
    if (arg.getKind() == TemplateArgument::Declaration) {
      VD = cast<ValueDecl>(arg.getAsDecl());

      // Find the instantiation of the template argument.  This is
      // required for nested templates.
      VD = cast_or_null<ValueDecl>(
             getSema().FindInstantiatedDecl(loc, VD, TemplateArgs));
      if (!VD)
        return ExprError();
    } else {
      // Propagate NULL template argument.
      VD = 0;
    }
    
    // Derive the type we want the substituted decl to have.  This had
    // better be non-dependent, or these checks will have serious problems.
    if (parm->isExpandedParameterPack()) {
      type = parm->getExpansionType(SemaRef.ArgumentPackSubstitutionIndex);
    } else if (parm->isParameterPack() && 
               isa<PackExpansionType>(parm->getType())) {
      type = SemaRef.SubstType(
                        cast<PackExpansionType>(parm->getType())->getPattern(),
                                     TemplateArgs, loc, parm->getDeclName());
    } else {
      type = SemaRef.SubstType(parm->getType(), TemplateArgs, 
                               loc, parm->getDeclName());
    }
    assert(!type.isNull() && "type substitution failed for param type");
    assert(!type->isDependentType() && "param type still dependent");
    result = SemaRef.BuildExpressionFromDeclTemplateArgument(arg, type, loc);

    if (!result.isInvalid()) type = result.get()->getType();
  } else {
    result = SemaRef.BuildExpressionFromIntegralTemplateArgument(arg, loc);

    // Note that this type can be different from the type of 'result',
    // e.g. if it's an enum type.
    type = arg.getIntegralType();
  }
  if (result.isInvalid()) return ExprError();

  Expr *resultExpr = result.take();
  return SemaRef.Owned(new (SemaRef.Context)
                SubstNonTypeTemplateParmExpr(type,
                                             resultExpr->getValueKind(),
                                             loc, parm, resultExpr));
}
                                                   
ExprResult 
TemplateInstantiator::TransformSubstNonTypeTemplateParmPackExpr(
                                          SubstNonTypeTemplateParmPackExpr *E) {
  if (getSema().ArgumentPackSubstitutionIndex == -1) {
    // We aren't expanding the parameter pack, so just return ourselves.
    return getSema().Owned(E);
  }
  
  const TemplateArgument &ArgPack = E->getArgumentPack();
  unsigned Index = (unsigned)getSema().ArgumentPackSubstitutionIndex;
  assert(Index < ArgPack.pack_size() && "Substitution index out-of-range");
  
  const TemplateArgument &Arg = ArgPack.pack_begin()[Index];
  return transformNonTypeTemplateParmRef(E->getParameterPack(),
                                         E->getParameterPackLocation(),
                                         Arg);
}

ExprResult
TemplateInstantiator::RebuildParmVarDeclRefExpr(ParmVarDecl *PD,
                                                SourceLocation Loc) {
  DeclarationNameInfo NameInfo(PD->getDeclName(), Loc);
  return getSema().BuildDeclarationNameExpr(CXXScopeSpec(), NameInfo, PD);
}

ExprResult
TemplateInstantiator::TransformFunctionParmPackExpr(FunctionParmPackExpr *E) {
  if (getSema().ArgumentPackSubstitutionIndex != -1) {
    // We can expand this parameter pack now.
    ParmVarDecl *D = E->getExpansion(getSema().ArgumentPackSubstitutionIndex);
    ValueDecl *VD = cast_or_null<ValueDecl>(TransformDecl(E->getExprLoc(), D));
    if (!VD)
      return ExprError();
    return RebuildParmVarDeclRefExpr(cast<ParmVarDecl>(VD), E->getExprLoc());
  }

  QualType T = TransformType(E->getType());
  if (T.isNull())
    return ExprError();

  // Transform each of the parameter expansions into the corresponding
  // parameters in the instantiation of the function decl.
  llvm::SmallVector<Decl*, 8> Parms;
  Parms.reserve(E->getNumExpansions());
  for (FunctionParmPackExpr::iterator I = E->begin(), End = E->end();
       I != End; ++I) {
    ParmVarDecl *D =
        cast_or_null<ParmVarDecl>(TransformDecl(E->getExprLoc(), *I));
    if (!D)
      return ExprError();
    Parms.push_back(D);
  }

  return FunctionParmPackExpr::Create(getSema().Context, T,
                                      E->getParameterPack(),
                                      E->getParameterPackLocation(), Parms);
}

ExprResult
TemplateInstantiator::TransformFunctionParmPackRefExpr(DeclRefExpr *E,
                                                       ParmVarDecl *PD) {
  typedef LocalInstantiationScope::DeclArgumentPack DeclArgumentPack;
  llvm::PointerUnion<Decl *, DeclArgumentPack *> *Found
    = getSema().CurrentInstantiationScope->findInstantiationOf(PD);
  assert(Found && "no instantiation for parameter pack");

  Decl *TransformedDecl;
  if (DeclArgumentPack *Pack = Found->dyn_cast<DeclArgumentPack *>()) {
    // If this is a reference to a function parameter pack which we can substitute
    // but can't yet expand, build a FunctionParmPackExpr for it.
    if (getSema().ArgumentPackSubstitutionIndex == -1) {
      QualType T = TransformType(E->getType());
      if (T.isNull())
        return ExprError();
      return FunctionParmPackExpr::Create(getSema().Context, T, PD,
                                          E->getExprLoc(), *Pack);
    }

    TransformedDecl = (*Pack)[getSema().ArgumentPackSubstitutionIndex];
  } else {
    TransformedDecl = Found->get<Decl*>();
  }

  // We have either an unexpanded pack or a specific expansion.
  return RebuildParmVarDeclRefExpr(cast<ParmVarDecl>(TransformedDecl),
                                   E->getExprLoc());
}

ExprResult
TemplateInstantiator::TransformDeclRefExpr(DeclRefExpr *E) {
  NamedDecl *D = E->getDecl();

  // Handle references to non-type template parameters and non-type template
  // parameter packs.
  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
    if (NTTP->getDepth() < TemplateArgs.getNumLevels())
      return TransformTemplateParmRefExpr(E, NTTP);
    
    // We have a non-type template parameter that isn't fully substituted;
    // FindInstantiatedDecl will find it in the local instantiation scope.
  }

  // Handle references to function parameter packs.
  if (ParmVarDecl *PD = dyn_cast<ParmVarDecl>(D))
    if (PD->isParameterPack())
      return TransformFunctionParmPackRefExpr(E, PD);

  return TreeTransform<TemplateInstantiator>::TransformDeclRefExpr(E);
}

ExprResult TemplateInstantiator::TransformCXXDefaultArgExpr(
    CXXDefaultArgExpr *E) {
  assert(!cast<FunctionDecl>(E->getParam()->getDeclContext())->
             getDescribedFunctionTemplate() &&
         "Default arg expressions are never formed in dependent cases.");
  return SemaRef.BuildCXXDefaultArgExpr(E->getUsedLocation(),
                           cast<FunctionDecl>(E->getParam()->getDeclContext()), 
                                        E->getParam());
}

QualType TemplateInstantiator::TransformFunctionProtoType(TypeLocBuilder &TLB,
                                                      FunctionProtoTypeLoc TL) {
  // We need a local instantiation scope for this function prototype.
  LocalInstantiationScope Scope(SemaRef, /*CombineWithOuterScope=*/true);
  return inherited::TransformFunctionProtoType(TLB, TL);
}

QualType TemplateInstantiator::TransformFunctionProtoType(TypeLocBuilder &TLB,
                                 FunctionProtoTypeLoc TL,
                                 CXXRecordDecl *ThisContext,
                                 unsigned ThisTypeQuals) {
  // We need a local instantiation scope for this function prototype.
  LocalInstantiationScope Scope(SemaRef, /*CombineWithOuterScope=*/true);
  return inherited::TransformFunctionProtoType(TLB, TL, ThisContext, 
                                               ThisTypeQuals);  
}

ParmVarDecl *
TemplateInstantiator::TransformFunctionTypeParam(ParmVarDecl *OldParm,
                                                 int indexAdjustment,
                                       llvm::Optional<unsigned> NumExpansions,
                                                 bool ExpectParameterPack) {
  return SemaRef.SubstParmVarDecl(OldParm, TemplateArgs, indexAdjustment,
                                  NumExpansions, ExpectParameterPack);
}

QualType
TemplateInstantiator::TransformTemplateTypeParmType(TypeLocBuilder &TLB,
                                                TemplateTypeParmTypeLoc TL) {
  const TemplateTypeParmType *T = TL.getTypePtr();
  if (T->getDepth() < TemplateArgs.getNumLevels()) {
    // Replace the template type parameter with its corresponding
    // template argument.

    // If the corresponding template argument is NULL or doesn't exist, it's
    // because we are performing instantiation from explicitly-specified
    // template arguments in a function template class, but there were some
    // arguments left unspecified.
    if (!TemplateArgs.hasTemplateArgument(T->getDepth(), T->getIndex())) {
      TemplateTypeParmTypeLoc NewTL
        = TLB.push<TemplateTypeParmTypeLoc>(TL.getType());
      NewTL.setNameLoc(TL.getNameLoc());
      return TL.getType();
    }

    TemplateArgument Arg = TemplateArgs(T->getDepth(), T->getIndex());
    
    if (T->isParameterPack()) {
      assert(Arg.getKind() == TemplateArgument::Pack && 
             "Missing argument pack");
      
      if (getSema().ArgumentPackSubstitutionIndex == -1) {
        // We have the template argument pack, but we're not expanding the
        // enclosing pack expansion yet. Just save the template argument
        // pack for later substitution.
        QualType Result
          = getSema().Context.getSubstTemplateTypeParmPackType(T, Arg);
        SubstTemplateTypeParmPackTypeLoc NewTL
          = TLB.push<SubstTemplateTypeParmPackTypeLoc>(Result);
        NewTL.setNameLoc(TL.getNameLoc());
        return Result;
      }
      
      assert(getSema().ArgumentPackSubstitutionIndex < (int)Arg.pack_size());
      Arg = Arg.pack_begin()[getSema().ArgumentPackSubstitutionIndex];
    }
    
    assert(Arg.getKind() == TemplateArgument::Type &&
           "Template argument kind mismatch");

    QualType Replacement = Arg.getAsType();

    // TODO: only do this uniquing once, at the start of instantiation.
    QualType Result
      = getSema().Context.getSubstTemplateTypeParmType(T, Replacement);
    SubstTemplateTypeParmTypeLoc NewTL
      = TLB.push<SubstTemplateTypeParmTypeLoc>(Result);
    NewTL.setNameLoc(TL.getNameLoc());
    return Result;
  }

  // The template type parameter comes from an inner template (e.g.,
  // the template parameter list of a member template inside the
  // template we are instantiating). Create a new template type
  // parameter with the template "level" reduced by one.
  TemplateTypeParmDecl *NewTTPDecl = 0;
  if (TemplateTypeParmDecl *OldTTPDecl = T->getDecl())
    NewTTPDecl = cast_or_null<TemplateTypeParmDecl>(
                                  TransformDecl(TL.getNameLoc(), OldTTPDecl));

  QualType Result
    = getSema().Context.getTemplateTypeParmType(T->getDepth()
                                                 - TemplateArgs.getNumLevels(),
                                                T->getIndex(),
                                                T->isParameterPack(),
                                                NewTTPDecl);
  TemplateTypeParmTypeLoc NewTL = TLB.push<TemplateTypeParmTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());
  return Result;
}

QualType 
TemplateInstantiator::TransformSubstTemplateTypeParmPackType(
                                                            TypeLocBuilder &TLB,
                                         SubstTemplateTypeParmPackTypeLoc TL) {
  if (getSema().ArgumentPackSubstitutionIndex == -1) {
    // We aren't expanding the parameter pack, so just return ourselves.
    SubstTemplateTypeParmPackTypeLoc NewTL
      = TLB.push<SubstTemplateTypeParmPackTypeLoc>(TL.getType());
    NewTL.setNameLoc(TL.getNameLoc());
    return TL.getType();
  }
  
  const TemplateArgument &ArgPack = TL.getTypePtr()->getArgumentPack();
  unsigned Index = (unsigned)getSema().ArgumentPackSubstitutionIndex;
  assert(Index < ArgPack.pack_size() && "Substitution index out-of-range");
  
  QualType Result = ArgPack.pack_begin()[Index].getAsType();
  Result = getSema().Context.getSubstTemplateTypeParmType(
                                      TL.getTypePtr()->getReplacedParameter(),
                                                          Result);
  SubstTemplateTypeParmTypeLoc NewTL
    = TLB.push<SubstTemplateTypeParmTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());
  return Result;
}

/// \brief Perform substitution on the type T with a given set of template
/// arguments.
///
/// This routine substitutes the given template arguments into the
/// type T and produces the instantiated type.
///
/// \param T the type into which the template arguments will be
/// substituted. If this type is not dependent, it will be returned
/// immediately.
///
/// \param Args the template arguments that will be
/// substituted for the top-level template parameters within T.
///
/// \param Loc the location in the source code where this substitution
/// is being performed. It will typically be the location of the
/// declarator (if we're instantiating the type of some declaration)
/// or the location of the type in the source code (if, e.g., we're
/// instantiating the type of a cast expression).
///
/// \param Entity the name of the entity associated with a declaration
/// being instantiated (if any). May be empty to indicate that there
/// is no such entity (if, e.g., this is a type that occurs as part of
/// a cast expression) or that the entity has no name (e.g., an
/// unnamed function parameter).
///
/// \returns If the instantiation succeeds, the instantiated
/// type. Otherwise, produces diagnostics and returns a NULL type.
TypeSourceInfo *Sema::SubstType(TypeSourceInfo *T,
                                const MultiLevelTemplateArgumentList &Args,
                                SourceLocation Loc,
                                DeclarationName Entity) {
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");
  
  if (!T->getType()->isInstantiationDependentType() && 
      !T->getType()->isVariablyModifiedType())
    return T;

  TemplateInstantiator Instantiator(*this, Args, Loc, Entity);
  return Instantiator.TransformType(T);
}

TypeSourceInfo *Sema::SubstType(TypeLoc TL,
                                const MultiLevelTemplateArgumentList &Args,
                                SourceLocation Loc,
                                DeclarationName Entity) {
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");
  
  if (TL.getType().isNull())
    return 0;

  if (!TL.getType()->isInstantiationDependentType() && 
      !TL.getType()->isVariablyModifiedType()) {
    // FIXME: Make a copy of the TypeLoc data here, so that we can
    // return a new TypeSourceInfo. Inefficient!
    TypeLocBuilder TLB;
    TLB.pushFullCopy(TL);
    return TLB.getTypeSourceInfo(Context, TL.getType());
  }

  TemplateInstantiator Instantiator(*this, Args, Loc, Entity);
  TypeLocBuilder TLB;
  TLB.reserve(TL.getFullDataSize());
  QualType Result = Instantiator.TransformType(TLB, TL);
  if (Result.isNull())
    return 0;

  return TLB.getTypeSourceInfo(Context, Result);
}

/// Deprecated form of the above.
QualType Sema::SubstType(QualType T,
                         const MultiLevelTemplateArgumentList &TemplateArgs,
                         SourceLocation Loc, DeclarationName Entity) {
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");

  // If T is not a dependent type or a variably-modified type, there
  // is nothing to do.
  if (!T->isInstantiationDependentType() && !T->isVariablyModifiedType())
    return T;

  TemplateInstantiator Instantiator(*this, TemplateArgs, Loc, Entity);
  return Instantiator.TransformType(T);
}

static bool NeedsInstantiationAsFunctionType(TypeSourceInfo *T) {
  if (T->getType()->isInstantiationDependentType() || 
      T->getType()->isVariablyModifiedType())
    return true;

  TypeLoc TL = T->getTypeLoc().IgnoreParens();
  if (!isa<FunctionProtoTypeLoc>(TL))
    return false;

  FunctionProtoTypeLoc FP = cast<FunctionProtoTypeLoc>(TL);
  for (unsigned I = 0, E = FP.getNumArgs(); I != E; ++I) {
    ParmVarDecl *P = FP.getArg(I);

    // The parameter's type as written might be dependent even if the
    // decayed type was not dependent.
    if (TypeSourceInfo *TSInfo = P->getTypeSourceInfo())
      if (TSInfo->getType()->isInstantiationDependentType())
        return true;

    // TODO: currently we always rebuild expressions.  When we
    // properly get lazier about this, we should use the same
    // logic to avoid rebuilding prototypes here.
    if (P->hasDefaultArg())
      return true;
  }

  return false;
}

/// A form of SubstType intended specifically for instantiating the
/// type of a FunctionDecl.  Its purpose is solely to force the
/// instantiation of default-argument expressions.
TypeSourceInfo *Sema::SubstFunctionDeclType(TypeSourceInfo *T,
                                const MultiLevelTemplateArgumentList &Args,
                                SourceLocation Loc,
                                DeclarationName Entity,
                                CXXRecordDecl *ThisContext,
                                unsigned ThisTypeQuals) {
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");
  
  if (!NeedsInstantiationAsFunctionType(T))
    return T;

  TemplateInstantiator Instantiator(*this, Args, Loc, Entity);

  TypeLocBuilder TLB;

  TypeLoc TL = T->getTypeLoc();
  TLB.reserve(TL.getFullDataSize());

  QualType Result;
  
  if (FunctionProtoTypeLoc *Proto = dyn_cast<FunctionProtoTypeLoc>(&TL)) {
    Result = Instantiator.TransformFunctionProtoType(TLB, *Proto, ThisContext,
                                                     ThisTypeQuals);
  } else {
    Result = Instantiator.TransformType(TLB, TL);
  }
  if (Result.isNull())
    return 0;

  return TLB.getTypeSourceInfo(Context, Result);
}

ParmVarDecl *Sema::SubstParmVarDecl(ParmVarDecl *OldParm, 
                            const MultiLevelTemplateArgumentList &TemplateArgs,
                                    int indexAdjustment,
                                    llvm::Optional<unsigned> NumExpansions,
                                    bool ExpectParameterPack) {
  TypeSourceInfo *OldDI = OldParm->getTypeSourceInfo();
  TypeSourceInfo *NewDI = 0;
  
  TypeLoc OldTL = OldDI->getTypeLoc();
  if (isa<PackExpansionTypeLoc>(OldTL)) {    
    PackExpansionTypeLoc ExpansionTL = cast<PackExpansionTypeLoc>(OldTL);
    
    // We have a function parameter pack. Substitute into the pattern of the 
    // expansion.
    NewDI = SubstType(ExpansionTL.getPatternLoc(), TemplateArgs, 
                      OldParm->getLocation(), OldParm->getDeclName());
    if (!NewDI)
      return 0;
        
    if (NewDI->getType()->containsUnexpandedParameterPack()) {
      // We still have unexpanded parameter packs, which means that
      // our function parameter is still a function parameter pack.
      // Therefore, make its type a pack expansion type.
      NewDI = CheckPackExpansion(NewDI, ExpansionTL.getEllipsisLoc(),
                                 NumExpansions);
    } else if (ExpectParameterPack) {
      // We expected to get a parameter pack but didn't (because the type
      // itself is not a pack expansion type), so complain. This can occur when
      // the substitution goes through an alias template that "loses" the
      // pack expansion.
      Diag(OldParm->getLocation(), 
           diag::err_function_parameter_pack_without_parameter_packs)
        << NewDI->getType();
      return 0;
    } 
  } else {
    NewDI = SubstType(OldDI, TemplateArgs, OldParm->getLocation(), 
                      OldParm->getDeclName());
  }
  
  if (!NewDI)
    return 0;

  if (NewDI->getType()->isVoidType()) {
    Diag(OldParm->getLocation(), diag::err_param_with_void_type);
    return 0;
  }

  ParmVarDecl *NewParm = CheckParameter(Context.getTranslationUnitDecl(),
                                        OldParm->getInnerLocStart(),
                                        OldParm->getLocation(),
                                        OldParm->getIdentifier(),
                                        NewDI->getType(), NewDI,
                                        OldParm->getStorageClass(),
                                        OldParm->getStorageClassAsWritten());
  if (!NewParm)
    return 0;
                                                
  // Mark the (new) default argument as uninstantiated (if any).
  if (OldParm->hasUninstantiatedDefaultArg()) {
    Expr *Arg = OldParm->getUninstantiatedDefaultArg();
    NewParm->setUninstantiatedDefaultArg(Arg);
  } else if (OldParm->hasUnparsedDefaultArg()) {
    NewParm->setUnparsedDefaultArg();
    UnparsedDefaultArgInstantiations[OldParm].push_back(NewParm);
  } else if (Expr *Arg = OldParm->getDefaultArg())
    // FIXME: if we non-lazily instantiated non-dependent default args for
    // non-dependent parameter types we could remove a bunch of duplicate
    // conversion warnings for such arguments.
    NewParm->setUninstantiatedDefaultArg(Arg);

  NewParm->setHasInheritedDefaultArg(OldParm->hasInheritedDefaultArg());
  
  if (OldParm->isParameterPack() && !NewParm->isParameterPack()) {
    // Add the new parameter to the instantiated parameter pack.
    CurrentInstantiationScope->InstantiatedLocalPackArg(OldParm, NewParm);
  } else {
    // Introduce an Old -> New mapping
    CurrentInstantiationScope->InstantiatedLocal(OldParm, NewParm);  
  }
  
  // FIXME: OldParm may come from a FunctionProtoType, in which case CurContext
  // can be anything, is this right ?
  NewParm->setDeclContext(CurContext);

  NewParm->setScopeInfo(OldParm->getFunctionScopeDepth(),
                        OldParm->getFunctionScopeIndex() + indexAdjustment);
  
  return NewParm;  
}

/// \brief Substitute the given template arguments into the given set of
/// parameters, producing the set of parameter types that would be generated
/// from such a substitution.
bool Sema::SubstParmTypes(SourceLocation Loc, 
                          ParmVarDecl **Params, unsigned NumParams,
                          const MultiLevelTemplateArgumentList &TemplateArgs,
                          SmallVectorImpl<QualType> &ParamTypes,
                          SmallVectorImpl<ParmVarDecl *> *OutParams) {
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");
  
  TemplateInstantiator Instantiator(*this, TemplateArgs, Loc, 
                                    DeclarationName());
  return Instantiator.TransformFunctionTypeParams(Loc, Params, NumParams, 0,
                                                  ParamTypes, OutParams);
}

/// \brief Perform substitution on the base class specifiers of the
/// given class template specialization.
///
/// Produces a diagnostic and returns true on error, returns false and
/// attaches the instantiated base classes to the class template
/// specialization if successful.
bool
Sema::SubstBaseSpecifiers(CXXRecordDecl *Instantiation,
                          CXXRecordDecl *Pattern,
                          const MultiLevelTemplateArgumentList &TemplateArgs) {
  bool Invalid = false;
  SmallVector<CXXBaseSpecifier*, 4> InstantiatedBases;
  for (ClassTemplateSpecializationDecl::base_class_iterator
         Base = Pattern->bases_begin(), BaseEnd = Pattern->bases_end();
       Base != BaseEnd; ++Base) {
    if (!Base->getType()->isDependentType()) {
      InstantiatedBases.push_back(new (Context) CXXBaseSpecifier(*Base));
      continue;
    }

    SourceLocation EllipsisLoc;
    TypeSourceInfo *BaseTypeLoc;
    if (Base->isPackExpansion()) {
      // This is a pack expansion. See whether we should expand it now, or
      // wait until later.
      SmallVector<UnexpandedParameterPack, 2> Unexpanded;
      collectUnexpandedParameterPacks(Base->getTypeSourceInfo()->getTypeLoc(),
                                      Unexpanded);
      bool ShouldExpand = false;
      bool RetainExpansion = false;
      llvm::Optional<unsigned> NumExpansions;
      if (CheckParameterPacksForExpansion(Base->getEllipsisLoc(), 
                                          Base->getSourceRange(),
                                          Unexpanded,
                                          TemplateArgs, ShouldExpand, 
                                          RetainExpansion,
                                          NumExpansions)) {
        Invalid = true;
        continue;
      }
      
      // If we should expand this pack expansion now, do so.
      if (ShouldExpand) {
        for (unsigned I = 0; I != *NumExpansions; ++I) {
            Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(*this, I);
          
          TypeSourceInfo *BaseTypeLoc = SubstType(Base->getTypeSourceInfo(),
                                                  TemplateArgs,
                                              Base->getSourceRange().getBegin(),
                                                  DeclarationName());
          if (!BaseTypeLoc) {
            Invalid = true;
            continue;
          }
          
          if (CXXBaseSpecifier *InstantiatedBase
                = CheckBaseSpecifier(Instantiation,
                                     Base->getSourceRange(),
                                     Base->isVirtual(),
                                     Base->getAccessSpecifierAsWritten(),
                                     BaseTypeLoc,
                                     SourceLocation()))
            InstantiatedBases.push_back(InstantiatedBase);
          else
            Invalid = true;
        }
      
        continue;
      }
      
      // The resulting base specifier will (still) be a pack expansion.
      EllipsisLoc = Base->getEllipsisLoc();
      Sema::ArgumentPackSubstitutionIndexRAII SubstIndex(*this, -1);
      BaseTypeLoc = SubstType(Base->getTypeSourceInfo(),
                              TemplateArgs,
                              Base->getSourceRange().getBegin(),
                              DeclarationName());
    } else {
      BaseTypeLoc = SubstType(Base->getTypeSourceInfo(),
                              TemplateArgs,
                              Base->getSourceRange().getBegin(),
                              DeclarationName());
    }
    
    if (!BaseTypeLoc) {
      Invalid = true;
      continue;
    }

    if (CXXBaseSpecifier *InstantiatedBase
          = CheckBaseSpecifier(Instantiation,
                               Base->getSourceRange(),
                               Base->isVirtual(),
                               Base->getAccessSpecifierAsWritten(),
                               BaseTypeLoc,
                               EllipsisLoc))
      InstantiatedBases.push_back(InstantiatedBase);
    else
      Invalid = true;
  }

  if (!Invalid &&
      AttachBaseSpecifiers(Instantiation, InstantiatedBases.data(),
                           InstantiatedBases.size()))
    Invalid = true;

  return Invalid;
}

// Defined via #include from SemaTemplateInstantiateDecl.cpp
namespace clang {
  namespace sema {
    Attr *instantiateTemplateAttribute(const Attr *At, ASTContext &C, Sema &S,
                            const MultiLevelTemplateArgumentList &TemplateArgs);
  }
}

/// Determine whether we would be unable to instantiate this template (because
/// it either has no definition, or is in the process of being instantiated).
static bool DiagnoseUninstantiableTemplate(Sema &S,
                                           SourceLocation PointOfInstantiation,
                                           TagDecl *Instantiation,
                                           bool InstantiatedFromMember,
                                           TagDecl *Pattern,
                                           TagDecl *PatternDef,
                                           TemplateSpecializationKind TSK,
                                           bool Complain = true) {
  if (PatternDef && !PatternDef->isBeingDefined())
    return false;

  if (!Complain || (PatternDef && PatternDef->isInvalidDecl())) {
    // Say nothing
  } else if (PatternDef) {
    assert(PatternDef->isBeingDefined());
    S.Diag(PointOfInstantiation,
           diag::err_template_instantiate_within_definition)
      << (TSK != TSK_ImplicitInstantiation)
      << S.Context.getTypeDeclType(Instantiation);
    // Not much point in noting the template declaration here, since
    // we're lexically inside it.
    Instantiation->setInvalidDecl();
  } else if (InstantiatedFromMember) {
    S.Diag(PointOfInstantiation,
           diag::err_implicit_instantiate_member_undefined)
      << S.Context.getTypeDeclType(Instantiation);
    S.Diag(Pattern->getLocation(), diag::note_member_of_template_here);
  } else {
    S.Diag(PointOfInstantiation, diag::err_template_instantiate_undefined)
      << (TSK != TSK_ImplicitInstantiation)
      << S.Context.getTypeDeclType(Instantiation);
    S.Diag(Pattern->getLocation(), diag::note_template_decl_here);
  }

  // In general, Instantiation isn't marked invalid to get more than one
  // error for multiple undefined instantiations. But the code that does
  // explicit declaration -> explicit definition conversion can't handle
  // invalid declarations, so mark as invalid in that case.
  if (TSK == TSK_ExplicitInstantiationDeclaration)
    Instantiation->setInvalidDecl();
  return true;
}

/// \brief Instantiate the definition of a class from a given pattern.
///
/// \param PointOfInstantiation The point of instantiation within the
/// source code.
///
/// \param Instantiation is the declaration whose definition is being
/// instantiated. This will be either a class template specialization
/// or a member class of a class template specialization.
///
/// \param Pattern is the pattern from which the instantiation
/// occurs. This will be either the declaration of a class template or
/// the declaration of a member class of a class template.
///
/// \param TemplateArgs The template arguments to be substituted into
/// the pattern.
///
/// \param TSK the kind of implicit or explicit instantiation to perform.
///
/// \param Complain whether to complain if the class cannot be instantiated due
/// to the lack of a definition.
///
/// \returns true if an error occurred, false otherwise.
bool
Sema::InstantiateClass(SourceLocation PointOfInstantiation,
                       CXXRecordDecl *Instantiation, CXXRecordDecl *Pattern,
                       const MultiLevelTemplateArgumentList &TemplateArgs,
                       TemplateSpecializationKind TSK,
                       bool Complain) {
  CXXRecordDecl *PatternDef
    = cast_or_null<CXXRecordDecl>(Pattern->getDefinition());
  if (DiagnoseUninstantiableTemplate(*this, PointOfInstantiation, Instantiation,
                                Instantiation->getInstantiatedFromMemberClass(),
                                     Pattern, PatternDef, TSK, Complain))
    return true;
  Pattern = PatternDef;

  // \brief Record the point of instantiation.
  if (MemberSpecializationInfo *MSInfo 
        = Instantiation->getMemberSpecializationInfo()) {
    MSInfo->setTemplateSpecializationKind(TSK);
    MSInfo->setPointOfInstantiation(PointOfInstantiation);
  } else if (ClassTemplateSpecializationDecl *Spec 
        = dyn_cast<ClassTemplateSpecializationDecl>(Instantiation)) {
    Spec->setTemplateSpecializationKind(TSK);
    Spec->setPointOfInstantiation(PointOfInstantiation);
  }
  
  InstantiatingTemplate Inst(*this, PointOfInstantiation, Instantiation);
  if (Inst)
    return true;

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  ContextRAII SavedContext(*this, Instantiation);
  EnterExpressionEvaluationContext EvalContext(*this, 
                                               Sema::PotentiallyEvaluated);

  // If this is an instantiation of a local class, merge this local
  // instantiation scope with the enclosing scope. Otherwise, every
  // instantiation of a class has its own local instantiation scope.
  bool MergeWithParentScope = !Instantiation->isDefinedOutsideFunctionOrMethod();
  LocalInstantiationScope Scope(*this, MergeWithParentScope);

  // Pull attributes from the pattern onto the instantiation.
  InstantiateAttrs(TemplateArgs, Pattern, Instantiation);

  // Start the definition of this instantiation.
  Instantiation->startDefinition();
  
  Instantiation->setTagKind(Pattern->getTagKind());

  // Do substitution on the base class specifiers.
  if (SubstBaseSpecifiers(Instantiation, Pattern, TemplateArgs))
    Instantiation->setInvalidDecl();

  TemplateDeclInstantiator Instantiator(*this, Instantiation, TemplateArgs);
  SmallVector<Decl*, 4> Fields;
  SmallVector<std::pair<FieldDecl*, FieldDecl*>, 4>
    FieldsWithMemberInitializers;
  // Delay instantiation of late parsed attributes.
  LateInstantiatedAttrVec LateAttrs;
  Instantiator.enableLateAttributeInstantiation(&LateAttrs);

  for (RecordDecl::decl_iterator Member = Pattern->decls_begin(),
         MemberEnd = Pattern->decls_end();
       Member != MemberEnd; ++Member) {
    // Don't instantiate members not belonging in this semantic context.
    // e.g. for:
    // @code
    //    template <int i> class A {
    //      class B *g;
    //    };
    // @endcode
    // 'class B' has the template as lexical context but semantically it is
    // introduced in namespace scope.
    if ((*Member)->getDeclContext() != Pattern)
      continue;

    if ((*Member)->isInvalidDecl()) {
      Instantiation->setInvalidDecl();
      continue;
    }

    Decl *NewMember = Instantiator.Visit(*Member);
    if (NewMember) {
      if (FieldDecl *Field = dyn_cast<FieldDecl>(NewMember)) {
        Fields.push_back(Field);
        FieldDecl *OldField = cast<FieldDecl>(*Member);
        if (OldField->getInClassInitializer())
          FieldsWithMemberInitializers.push_back(std::make_pair(OldField,
                                                                Field));
      } else if (EnumDecl *Enum = dyn_cast<EnumDecl>(NewMember)) {
        // C++11 [temp.inst]p1: The implicit instantiation of a class template
        // specialization causes the implicit instantiation of the definitions
        // of unscoped member enumerations.
        // Record a point of instantiation for this implicit instantiation.
        if (TSK == TSK_ImplicitInstantiation && !Enum->isScoped() &&
            Enum->isCompleteDefinition()) {
          MemberSpecializationInfo *MSInfo =Enum->getMemberSpecializationInfo();
          assert(MSInfo && "no spec info for member enum specialization");
          MSInfo->setTemplateSpecializationKind(TSK_ImplicitInstantiation);
          MSInfo->setPointOfInstantiation(PointOfInstantiation);
        }
      } else if (StaticAssertDecl *SA = dyn_cast<StaticAssertDecl>(NewMember)) {
        if (SA->isFailed()) {
          // A static_assert failed. Bail out; instantiating this
          // class is probably not meaningful.
          Instantiation->setInvalidDecl();
          break;
        }
      }

      if (NewMember->isInvalidDecl())
        Instantiation->setInvalidDecl();
    } else {
      // FIXME: Eventually, a NULL return will mean that one of the
      // instantiations was a semantic disaster, and we'll want to mark the
      // declaration invalid.
      // For now, we expect to skip some members that we can't yet handle.
    }
  }

  // Finish checking fields.
  ActOnFields(0, Instantiation->getLocation(), Instantiation, Fields, 
              SourceLocation(), SourceLocation(), 0);
  CheckCompletedCXXClass(Instantiation);

  // Attach any in-class member initializers now the class is complete.
  if (!FieldsWithMemberInitializers.empty()) {
    // C++11 [expr.prim.general]p4:
    //   Otherwise, if a member-declarator declares a non-static data member 
    //  (9.2) of a class X, the expression this is a prvalue of type "pointer
    //  to X" within the optional brace-or-equal-initializer. It shall not 
    //  appear elsewhere in the member-declarator.
    CXXThisScopeRAII ThisScope(*this, Instantiation, (unsigned)0);
    
    for (unsigned I = 0, N = FieldsWithMemberInitializers.size(); I != N; ++I) {
      FieldDecl *OldField = FieldsWithMemberInitializers[I].first;
      FieldDecl *NewField = FieldsWithMemberInitializers[I].second;
      Expr *OldInit = OldField->getInClassInitializer();

      ExprResult NewInit = SubstInitializer(OldInit, TemplateArgs,
                                            /*CXXDirectInit=*/false);
      if (NewInit.isInvalid())
        NewField->setInvalidDecl();
      else {
        Expr *Init = NewInit.take();
        assert(Init && "no-argument initializer in class");
        assert(!isa<ParenListExpr>(Init) && "call-style init in class");
        ActOnCXXInClassMemberInitializer(NewField, Init->getLocStart(), Init);
      }
    }
  }
  // Instantiate late parsed attributes, and attach them to their decls.
  // See Sema::InstantiateAttrs
  for (LateInstantiatedAttrVec::iterator I = LateAttrs.begin(),
       E = LateAttrs.end(); I != E; ++I) {
    assert(CurrentInstantiationScope == Instantiator.getStartingScope());
    CurrentInstantiationScope = I->Scope;
    Attr *NewAttr =
      instantiateTemplateAttribute(I->TmplAttr, Context, *this, TemplateArgs);
    I->NewDecl->addAttr(NewAttr);
    LocalInstantiationScope::deleteScopes(I->Scope,
                                          Instantiator.getStartingScope());
  }
  Instantiator.disableLateAttributeInstantiation();
  LateAttrs.clear();

  ActOnFinishDelayedMemberInitializers(Instantiation);

  if (TSK == TSK_ImplicitInstantiation) {
    Instantiation->setLocation(Pattern->getLocation());
    Instantiation->setLocStart(Pattern->getInnerLocStart());
    Instantiation->setRBraceLoc(Pattern->getRBraceLoc());
  }

  if (!Instantiation->isInvalidDecl()) {
    // Perform any dependent diagnostics from the pattern.
    PerformDependentDiagnostics(Pattern, TemplateArgs);

    // Instantiate any out-of-line class template partial
    // specializations now.
    for (TemplateDeclInstantiator::delayed_partial_spec_iterator 
              P = Instantiator.delayed_partial_spec_begin(),
           PEnd = Instantiator.delayed_partial_spec_end();
         P != PEnd; ++P) {
      if (!Instantiator.InstantiateClassTemplatePartialSpecialization(
                                                                P->first,
                                                                P->second)) {
        Instantiation->setInvalidDecl();
        break;
      }
    }
  }

  // Exit the scope of this instantiation.
  SavedContext.pop();

  if (!Instantiation->isInvalidDecl()) {
    Consumer.HandleTagDeclDefinition(Instantiation);

    // Always emit the vtable for an explicit instantiation definition
    // of a polymorphic class template specialization.
    if (TSK == TSK_ExplicitInstantiationDefinition)
      MarkVTableUsed(PointOfInstantiation, Instantiation, true);
  }

  return Instantiation->isInvalidDecl();
}

/// \brief Instantiate the definition of an enum from a given pattern.
///
/// \param PointOfInstantiation The point of instantiation within the
///        source code.
/// \param Instantiation is the declaration whose definition is being
///        instantiated. This will be a member enumeration of a class
///        temploid specialization, or a local enumeration within a
///        function temploid specialization.
/// \param Pattern The templated declaration from which the instantiation
///        occurs.
/// \param TemplateArgs The template arguments to be substituted into
///        the pattern.
/// \param TSK The kind of implicit or explicit instantiation to perform.
///
/// \return \c true if an error occurred, \c false otherwise.
bool Sema::InstantiateEnum(SourceLocation PointOfInstantiation,
                           EnumDecl *Instantiation, EnumDecl *Pattern,
                           const MultiLevelTemplateArgumentList &TemplateArgs,
                           TemplateSpecializationKind TSK) {
  EnumDecl *PatternDef = Pattern->getDefinition();
  if (DiagnoseUninstantiableTemplate(*this, PointOfInstantiation, Instantiation,
                                 Instantiation->getInstantiatedFromMemberEnum(),
                                     Pattern, PatternDef, TSK,/*Complain*/true))
    return true;
  Pattern = PatternDef;

  // Record the point of instantiation.
  if (MemberSpecializationInfo *MSInfo
        = Instantiation->getMemberSpecializationInfo()) {
    MSInfo->setTemplateSpecializationKind(TSK);
    MSInfo->setPointOfInstantiation(PointOfInstantiation);
  }

  InstantiatingTemplate Inst(*this, PointOfInstantiation, Instantiation);
  if (Inst)
    return true;

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  ContextRAII SavedContext(*this, Instantiation);
  EnterExpressionEvaluationContext EvalContext(*this,
                                               Sema::PotentiallyEvaluated);

  LocalInstantiationScope Scope(*this, /*MergeWithParentScope*/true);

  // Pull attributes from the pattern onto the instantiation.
  InstantiateAttrs(TemplateArgs, Pattern, Instantiation);

  TemplateDeclInstantiator Instantiator(*this, Instantiation, TemplateArgs);
  Instantiator.InstantiateEnumDefinition(Instantiation, Pattern);

  // Exit the scope of this instantiation.
  SavedContext.pop();

  return Instantiation->isInvalidDecl();
}

namespace {
  /// \brief A partial specialization whose template arguments have matched
  /// a given template-id.
  struct PartialSpecMatchResult {
    ClassTemplatePartialSpecializationDecl *Partial;
    TemplateArgumentList *Args;
  };
}

bool
Sema::InstantiateClassTemplateSpecialization(
                           SourceLocation PointOfInstantiation,
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           TemplateSpecializationKind TSK,
                           bool Complain) {
  // Perform the actual instantiation on the canonical declaration.
  ClassTemplateSpec = cast<ClassTemplateSpecializationDecl>(
                                         ClassTemplateSpec->getCanonicalDecl());

  // Check whether we have already instantiated or specialized this class
  // template specialization.
  if (ClassTemplateSpec->getSpecializationKind() != TSK_Undeclared) {
    if (ClassTemplateSpec->getSpecializationKind() == 
          TSK_ExplicitInstantiationDeclaration &&
        TSK == TSK_ExplicitInstantiationDefinition) {
      // An explicit instantiation definition follows an explicit instantiation
      // declaration (C++0x [temp.explicit]p10); go ahead and perform the
      // explicit instantiation.
      ClassTemplateSpec->setSpecializationKind(TSK);
      
      // If this is an explicit instantiation definition, mark the
      // vtable as used.
      if (TSK == TSK_ExplicitInstantiationDefinition &&
          !ClassTemplateSpec->isInvalidDecl())
        MarkVTableUsed(PointOfInstantiation, ClassTemplateSpec, true);

      return false;
    }
    
    // We can only instantiate something that hasn't already been
    // instantiated or specialized. Fail without any diagnostics: our
    // caller will provide an error message.    
    return true;
  }

  if (ClassTemplateSpec->isInvalidDecl())
    return true;
  
  ClassTemplateDecl *Template = ClassTemplateSpec->getSpecializedTemplate();
  CXXRecordDecl *Pattern = 0;

  // C++ [temp.class.spec.match]p1:
  //   When a class template is used in a context that requires an
  //   instantiation of the class, it is necessary to determine
  //   whether the instantiation is to be generated using the primary
  //   template or one of the partial specializations. This is done by
  //   matching the template arguments of the class template
  //   specialization with the template argument lists of the partial
  //   specializations.
  typedef PartialSpecMatchResult MatchResult;
  SmallVector<MatchResult, 4> Matched;
  SmallVector<ClassTemplatePartialSpecializationDecl *, 4> PartialSpecs;
  Template->getPartialSpecializations(PartialSpecs);
  for (unsigned I = 0, N = PartialSpecs.size(); I != N; ++I) {
    ClassTemplatePartialSpecializationDecl *Partial = PartialSpecs[I];
    TemplateDeductionInfo Info(PointOfInstantiation);
    if (TemplateDeductionResult Result
          = DeduceTemplateArguments(Partial,
                                    ClassTemplateSpec->getTemplateArgs(),
                                    Info)) {
      // FIXME: Store the failed-deduction information for use in
      // diagnostics, later.
      (void)Result;
    } else {
      Matched.push_back(PartialSpecMatchResult());
      Matched.back().Partial = Partial;
      Matched.back().Args = Info.take();
    }
  }

  // If we're dealing with a member template where the template parameters
  // have been instantiated, this provides the original template parameters
  // from which the member template's parameters were instantiated.
  SmallVector<const NamedDecl *, 4> InstantiatedTemplateParameters;
  
  if (Matched.size() >= 1) {
    SmallVector<MatchResult, 4>::iterator Best = Matched.begin();
    if (Matched.size() == 1) {
      //   -- If exactly one matching specialization is found, the
      //      instantiation is generated from that specialization.
      // We don't need to do anything for this.
    } else {
      //   -- If more than one matching specialization is found, the
      //      partial order rules (14.5.4.2) are used to determine
      //      whether one of the specializations is more specialized
      //      than the others. If none of the specializations is more
      //      specialized than all of the other matching
      //      specializations, then the use of the class template is
      //      ambiguous and the program is ill-formed.
      for (SmallVector<MatchResult, 4>::iterator P = Best + 1,
                                                    PEnd = Matched.end();
           P != PEnd; ++P) {
        if (getMoreSpecializedPartialSpecialization(P->Partial, Best->Partial,
                                                    PointOfInstantiation) 
              == P->Partial)
          Best = P;
      }
      
      // Determine if the best partial specialization is more specialized than
      // the others.
      bool Ambiguous = false;
      for (SmallVector<MatchResult, 4>::iterator P = Matched.begin(),
                                                    PEnd = Matched.end();
           P != PEnd; ++P) {
        if (P != Best &&
            getMoreSpecializedPartialSpecialization(P->Partial, Best->Partial,
                                                    PointOfInstantiation)
              != Best->Partial) {
          Ambiguous = true;
          break;
        }
      }
       
      if (Ambiguous) {
        // Partial ordering did not produce a clear winner. Complain.
        ClassTemplateSpec->setInvalidDecl();
        Diag(PointOfInstantiation, diag::err_partial_spec_ordering_ambiguous)
          << ClassTemplateSpec;
        
        // Print the matching partial specializations.
        for (SmallVector<MatchResult, 4>::iterator P = Matched.begin(),
                                                      PEnd = Matched.end();
             P != PEnd; ++P)
          Diag(P->Partial->getLocation(), diag::note_partial_spec_match)
            << getTemplateArgumentBindingsText(
                                            P->Partial->getTemplateParameters(),
                                               *P->Args);

        return true;
      }
    }
    
    // Instantiate using the best class template partial specialization.
    ClassTemplatePartialSpecializationDecl *OrigPartialSpec = Best->Partial;
    while (OrigPartialSpec->getInstantiatedFromMember()) {
      // If we've found an explicit specialization of this class template,
      // stop here and use that as the pattern.
      if (OrigPartialSpec->isMemberSpecialization())
        break;
      
      OrigPartialSpec = OrigPartialSpec->getInstantiatedFromMember();
    }
    
    Pattern = OrigPartialSpec;
    ClassTemplateSpec->setInstantiationOf(Best->Partial, Best->Args);
  } else {
    //   -- If no matches are found, the instantiation is generated
    //      from the primary template.
    ClassTemplateDecl *OrigTemplate = Template;
    while (OrigTemplate->getInstantiatedFromMemberTemplate()) {
      // If we've found an explicit specialization of this class template,
      // stop here and use that as the pattern.
      if (OrigTemplate->isMemberSpecialization())
        break;
      
      OrigTemplate = OrigTemplate->getInstantiatedFromMemberTemplate();
    }
    
    Pattern = OrigTemplate->getTemplatedDecl();
  }

  bool Result = InstantiateClass(PointOfInstantiation, ClassTemplateSpec, 
                                 Pattern,
                                getTemplateInstantiationArgs(ClassTemplateSpec),
                                 TSK,
                                 Complain);

  return Result;
}

/// \brief Instantiates the definitions of all of the member
/// of the given class, which is an instantiation of a class template
/// or a member class of a template.
void
Sema::InstantiateClassMembers(SourceLocation PointOfInstantiation,
                              CXXRecordDecl *Instantiation,
                        const MultiLevelTemplateArgumentList &TemplateArgs,
                              TemplateSpecializationKind TSK) {
  for (DeclContext::decl_iterator D = Instantiation->decls_begin(),
                               DEnd = Instantiation->decls_end();
       D != DEnd; ++D) {
    bool SuppressNew = false;
    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(*D)) {
      if (FunctionDecl *Pattern
            = Function->getInstantiatedFromMemberFunction()) {
        MemberSpecializationInfo *MSInfo 
          = Function->getMemberSpecializationInfo();
        assert(MSInfo && "No member specialization information?");
        if (MSInfo->getTemplateSpecializationKind()
                                                 == TSK_ExplicitSpecialization)
          continue;
        
        if (CheckSpecializationInstantiationRedecl(PointOfInstantiation, TSK, 
                                                   Function, 
                                        MSInfo->getTemplateSpecializationKind(),
                                              MSInfo->getPointOfInstantiation(),
                                                   SuppressNew) ||
            SuppressNew)
          continue;
        
        if (Function->isDefined())
          continue;

        if (TSK == TSK_ExplicitInstantiationDefinition) {
          // C++0x [temp.explicit]p8:
          //   An explicit instantiation definition that names a class template
          //   specialization explicitly instantiates the class template 
          //   specialization and is only an explicit instantiation definition 
          //   of members whose definition is visible at the point of 
          //   instantiation.
          if (!Pattern->isDefined())
            continue;
        
          Function->setTemplateSpecializationKind(TSK, PointOfInstantiation);
                      
          InstantiateFunctionDefinition(PointOfInstantiation, Function);
        } else {
          Function->setTemplateSpecializationKind(TSK, PointOfInstantiation);
        }
      }
    } else if (VarDecl *Var = dyn_cast<VarDecl>(*D)) {
      if (Var->isStaticDataMember()) {
        MemberSpecializationInfo *MSInfo = Var->getMemberSpecializationInfo();
        assert(MSInfo && "No member specialization information?");
        if (MSInfo->getTemplateSpecializationKind()
                                                 == TSK_ExplicitSpecialization)
          continue;
        
        if (CheckSpecializationInstantiationRedecl(PointOfInstantiation, TSK, 
                                                   Var, 
                                        MSInfo->getTemplateSpecializationKind(),
                                              MSInfo->getPointOfInstantiation(),
                                                   SuppressNew) ||
            SuppressNew)
          continue;
        
        if (TSK == TSK_ExplicitInstantiationDefinition) {
          // C++0x [temp.explicit]p8:
          //   An explicit instantiation definition that names a class template
          //   specialization explicitly instantiates the class template 
          //   specialization and is only an explicit instantiation definition 
          //   of members whose definition is visible at the point of 
          //   instantiation.
          if (!Var->getInstantiatedFromStaticDataMember()
                                                     ->getOutOfLineDefinition())
            continue;
          
          Var->setTemplateSpecializationKind(TSK, PointOfInstantiation);
          InstantiateStaticDataMemberDefinition(PointOfInstantiation, Var);
        } else {
          Var->setTemplateSpecializationKind(TSK, PointOfInstantiation);
        }
      }      
    } else if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(*D)) {
      // Always skip the injected-class-name, along with any
      // redeclarations of nested classes, since both would cause us
      // to try to instantiate the members of a class twice.
      if (Record->isInjectedClassName() || Record->getPreviousDecl())
        continue;
      
      MemberSpecializationInfo *MSInfo = Record->getMemberSpecializationInfo();
      assert(MSInfo && "No member specialization information?");
      
      if (MSInfo->getTemplateSpecializationKind()
                                                == TSK_ExplicitSpecialization)
        continue;

      if (CheckSpecializationInstantiationRedecl(PointOfInstantiation, TSK, 
                                                 Record, 
                                        MSInfo->getTemplateSpecializationKind(),
                                              MSInfo->getPointOfInstantiation(),
                                                 SuppressNew) ||
          SuppressNew)
        continue;
      
      CXXRecordDecl *Pattern = Record->getInstantiatedFromMemberClass();
      assert(Pattern && "Missing instantiated-from-template information");
      
      if (!Record->getDefinition()) {
        if (!Pattern->getDefinition()) {
          // C++0x [temp.explicit]p8:
          //   An explicit instantiation definition that names a class template
          //   specialization explicitly instantiates the class template 
          //   specialization and is only an explicit instantiation definition 
          //   of members whose definition is visible at the point of 
          //   instantiation.
          if (TSK == TSK_ExplicitInstantiationDeclaration) {
            MSInfo->setTemplateSpecializationKind(TSK);
            MSInfo->setPointOfInstantiation(PointOfInstantiation);
          }
          
          continue;
        }
        
        InstantiateClass(PointOfInstantiation, Record, Pattern,
                         TemplateArgs,
                         TSK);
      } else {
        if (TSK == TSK_ExplicitInstantiationDefinition &&
            Record->getTemplateSpecializationKind() ==
                TSK_ExplicitInstantiationDeclaration) {
          Record->setTemplateSpecializationKind(TSK);
          MarkVTableUsed(PointOfInstantiation, Record, true);
        }
      }
      
      Pattern = cast_or_null<CXXRecordDecl>(Record->getDefinition());
      if (Pattern)
        InstantiateClassMembers(PointOfInstantiation, Pattern, TemplateArgs, 
                                TSK);
    } else if (EnumDecl *Enum = dyn_cast<EnumDecl>(*D)) {
      MemberSpecializationInfo *MSInfo = Enum->getMemberSpecializationInfo();
      assert(MSInfo && "No member specialization information?");

      if (MSInfo->getTemplateSpecializationKind()
            == TSK_ExplicitSpecialization)
        continue;

      if (CheckSpecializationInstantiationRedecl(
            PointOfInstantiation, TSK, Enum,
            MSInfo->getTemplateSpecializationKind(),
            MSInfo->getPointOfInstantiation(), SuppressNew) ||
          SuppressNew)
        continue;

      if (Enum->getDefinition())
        continue;

      EnumDecl *Pattern = Enum->getInstantiatedFromMemberEnum();
      assert(Pattern && "Missing instantiated-from-template information");

      if (TSK == TSK_ExplicitInstantiationDefinition) {
        if (!Pattern->getDefinition())
          continue;

        InstantiateEnum(PointOfInstantiation, Enum, Pattern, TemplateArgs, TSK);
      } else {
        MSInfo->setTemplateSpecializationKind(TSK);
        MSInfo->setPointOfInstantiation(PointOfInstantiation);
      }
    }
  }
}

/// \brief Instantiate the definitions of all of the members of the
/// given class template specialization, which was named as part of an
/// explicit instantiation.
void
Sema::InstantiateClassTemplateSpecializationMembers(
                                           SourceLocation PointOfInstantiation,
                            ClassTemplateSpecializationDecl *ClassTemplateSpec,
                                               TemplateSpecializationKind TSK) {
  // C++0x [temp.explicit]p7:
  //   An explicit instantiation that names a class template
  //   specialization is an explicit instantion of the same kind
  //   (declaration or definition) of each of its members (not
  //   including members inherited from base classes) that has not
  //   been previously explicitly specialized in the translation unit
  //   containing the explicit instantiation, except as described
  //   below.
  InstantiateClassMembers(PointOfInstantiation, ClassTemplateSpec,
                          getTemplateInstantiationArgs(ClassTemplateSpec),
                          TSK);
}

StmtResult
Sema::SubstStmt(Stmt *S, const MultiLevelTemplateArgumentList &TemplateArgs) {
  if (!S)
    return Owned(S);

  TemplateInstantiator Instantiator(*this, TemplateArgs,
                                    SourceLocation(),
                                    DeclarationName());
  return Instantiator.TransformStmt(S);
}

ExprResult
Sema::SubstExpr(Expr *E, const MultiLevelTemplateArgumentList &TemplateArgs) {
  if (!E)
    return Owned(E);

  TemplateInstantiator Instantiator(*this, TemplateArgs,
                                    SourceLocation(),
                                    DeclarationName());
  return Instantiator.TransformExpr(E);
}

bool Sema::SubstExprs(Expr **Exprs, unsigned NumExprs, bool IsCall,
                      const MultiLevelTemplateArgumentList &TemplateArgs,
                      SmallVectorImpl<Expr *> &Outputs) {
  if (NumExprs == 0)
    return false;
  
  TemplateInstantiator Instantiator(*this, TemplateArgs,
                                    SourceLocation(),
                                    DeclarationName());
  return Instantiator.TransformExprs(Exprs, NumExprs, IsCall, Outputs);
}

NestedNameSpecifierLoc
Sema::SubstNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS,
                        const MultiLevelTemplateArgumentList &TemplateArgs) {  
  if (!NNS)
    return NestedNameSpecifierLoc();
  
  TemplateInstantiator Instantiator(*this, TemplateArgs, NNS.getBeginLoc(),
                                    DeclarationName());
  return Instantiator.TransformNestedNameSpecifierLoc(NNS);
}

/// \brief Do template substitution on declaration name info.
DeclarationNameInfo
Sema::SubstDeclarationNameInfo(const DeclarationNameInfo &NameInfo,
                         const MultiLevelTemplateArgumentList &TemplateArgs) {
  TemplateInstantiator Instantiator(*this, TemplateArgs, NameInfo.getLoc(),
                                    NameInfo.getName());
  return Instantiator.TransformDeclarationNameInfo(NameInfo);
}

TemplateName
Sema::SubstTemplateName(NestedNameSpecifierLoc QualifierLoc,
                        TemplateName Name, SourceLocation Loc,
                        const MultiLevelTemplateArgumentList &TemplateArgs) {
  TemplateInstantiator Instantiator(*this, TemplateArgs, Loc,
                                    DeclarationName());
  CXXScopeSpec SS;
  SS.Adopt(QualifierLoc);
  return Instantiator.TransformTemplateName(SS, Name, Loc);
}

bool Sema::Subst(const TemplateArgumentLoc *Args, unsigned NumArgs,
                 TemplateArgumentListInfo &Result,
                 const MultiLevelTemplateArgumentList &TemplateArgs) {
  TemplateInstantiator Instantiator(*this, TemplateArgs, SourceLocation(),
                                    DeclarationName());
  
  return Instantiator.TransformTemplateArguments(Args, NumArgs, Result);
}


static const Decl* getCanonicalParmVarDecl(const Decl *D) {
  // When storing ParmVarDecls in the local instantiation scope, we always
  // want to use the ParmVarDecl from the canonical function declaration,
  // since the map is then valid for any redeclaration or definition of that
  // function.
  if (const ParmVarDecl *PV = dyn_cast<ParmVarDecl>(D)) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(PV->getDeclContext())) {
      unsigned i = PV->getFunctionScopeIndex();
      return FD->getCanonicalDecl()->getParamDecl(i);
    }
  }
  return D;
}


llvm::PointerUnion<Decl *, LocalInstantiationScope::DeclArgumentPack *> *
LocalInstantiationScope::findInstantiationOf(const Decl *D) {
  D = getCanonicalParmVarDecl(D);
  for (LocalInstantiationScope *Current = this; Current;
       Current = Current->Outer) {

    // Check if we found something within this scope.
    const Decl *CheckD = D;
    do {
      LocalDeclsMap::iterator Found = Current->LocalDecls.find(CheckD);
      if (Found != Current->LocalDecls.end())
        return &Found->second;
      
      // If this is a tag declaration, it's possible that we need to look for
      // a previous declaration.
      if (const TagDecl *Tag = dyn_cast<TagDecl>(CheckD))
        CheckD = Tag->getPreviousDecl();
      else
        CheckD = 0;
    } while (CheckD);
    
    // If we aren't combined with our outer scope, we're done. 
    if (!Current->CombineWithOuterScope)
      break;
  }

  // If we didn't find the decl, then we either have a sema bug, or we have a
  // forward reference to a label declaration.  Return null to indicate that
  // we have an uninstantiated label.
  assert(isa<LabelDecl>(D) && "declaration not instantiated in this scope");
  return 0;
}

void LocalInstantiationScope::InstantiatedLocal(const Decl *D, Decl *Inst) {
  D = getCanonicalParmVarDecl(D);
  llvm::PointerUnion<Decl *, DeclArgumentPack *> &Stored = LocalDecls[D];
  if (Stored.isNull())
    Stored = Inst;
  else if (Stored.is<Decl *>()) {
    assert(Stored.get<Decl *>() == Inst && "Already instantiated this local");
    Stored = Inst;
  } else
    LocalDecls[D].get<DeclArgumentPack *>()->push_back(Inst);
}

void LocalInstantiationScope::InstantiatedLocalPackArg(const Decl *D, 
                                                       Decl *Inst) {
  D = getCanonicalParmVarDecl(D);
  DeclArgumentPack *Pack = LocalDecls[D].get<DeclArgumentPack *>();
  Pack->push_back(Inst);
}

void LocalInstantiationScope::MakeInstantiatedLocalArgPack(const Decl *D) {
  D = getCanonicalParmVarDecl(D);
  llvm::PointerUnion<Decl *, DeclArgumentPack *> &Stored = LocalDecls[D];
  assert(Stored.isNull() && "Already instantiated this local");
  DeclArgumentPack *Pack = new DeclArgumentPack;
  Stored = Pack;
  ArgumentPacks.push_back(Pack);
}

void LocalInstantiationScope::SetPartiallySubstitutedPack(NamedDecl *Pack, 
                                          const TemplateArgument *ExplicitArgs,
                                                    unsigned NumExplicitArgs) {
  assert((!PartiallySubstitutedPack || PartiallySubstitutedPack == Pack) &&
         "Already have a partially-substituted pack");
  assert((!PartiallySubstitutedPack 
          || NumArgsInPartiallySubstitutedPack == NumExplicitArgs) &&
         "Wrong number of arguments in partially-substituted pack");
  PartiallySubstitutedPack = Pack;
  ArgsInPartiallySubstitutedPack = ExplicitArgs;
  NumArgsInPartiallySubstitutedPack = NumExplicitArgs;
}

NamedDecl *LocalInstantiationScope::getPartiallySubstitutedPack(
                                         const TemplateArgument **ExplicitArgs,
                                              unsigned *NumExplicitArgs) const {
  if (ExplicitArgs)
    *ExplicitArgs = 0;
  if (NumExplicitArgs)
    *NumExplicitArgs = 0;
  
  for (const LocalInstantiationScope *Current = this; Current; 
       Current = Current->Outer) {
    if (Current->PartiallySubstitutedPack) {
      if (ExplicitArgs)
        *ExplicitArgs = Current->ArgsInPartiallySubstitutedPack;
      if (NumExplicitArgs)
        *NumExplicitArgs = Current->NumArgsInPartiallySubstitutedPack;
      
      return Current->PartiallySubstitutedPack;
    }

    if (!Current->CombineWithOuterScope)
      break;
  }
  
  return 0;
}
