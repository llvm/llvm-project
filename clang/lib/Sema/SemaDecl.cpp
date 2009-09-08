//===--- SemaDecl.cpp - Semantic Analysis for Declarations ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for declarations.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "SemaInherit.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/SourceManager.h"
// FIXME: layering (ideally, Sema shouldn't be dependent on Lex API's)
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h" 
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <functional>
#include <queue>
using namespace clang;

/// getDeclName - Return a pretty name for the specified decl if possible, or
/// an empty string if not.  This is used for pretty crash reporting. 
std::string Sema::getDeclName(DeclPtrTy d) {
  Decl *D = d.getAs<Decl>();
  if (NamedDecl *DN = dyn_cast_or_null<NamedDecl>(D))
    return DN->getQualifiedNameAsString();
  return "";
}

Sema::DeclGroupPtrTy Sema::ConvertDeclToDeclGroup(DeclPtrTy Ptr) {
  return DeclGroupPtrTy::make(DeclGroupRef(Ptr.getAs<Decl>()));
}

/// \brief If the identifier refers to a type name within this scope,
/// return the declaration of that type.
///
/// This routine performs ordinary name lookup of the identifier II
/// within the given scope, with optional C++ scope specifier SS, to
/// determine whether the name refers to a type. If so, returns an
/// opaque pointer (actually a QualType) corresponding to that
/// type. Otherwise, returns NULL.
///
/// If name lookup results in an ambiguity, this routine will complain
/// and then return NULL.
Sema::TypeTy *Sema::getTypeName(IdentifierInfo &II, SourceLocation NameLoc,
                                Scope *S, const CXXScopeSpec *SS) {
  // C++ [temp.res]p3:
  //   A qualified-id that refers to a type and in which the
  //   nested-name-specifier depends on a template-parameter (14.6.2)
  //   shall be prefixed by the keyword typename to indicate that the
  //   qualified-id denotes a type, forming an
  //   elaborated-type-specifier (7.1.5.3).
  //
  // We therefore do not perform any name lookup if the result would
  // refer to a member of an unknown specialization.
  if (SS && isUnknownSpecialization(*SS))
    return 0;

  LookupResult Result 
    = LookupParsedName(S, SS, &II, LookupOrdinaryName, false, false);

  NamedDecl *IIDecl = 0;
  switch (Result.getKind()) {
  case LookupResult::NotFound:
  case LookupResult::FoundOverloaded:
    return 0;

  case LookupResult::AmbiguousBaseSubobjectTypes:
  case LookupResult::AmbiguousBaseSubobjects:
  case LookupResult::AmbiguousReference: {
    // Look to see if we have a type anywhere in the list of results.
    for (LookupResult::iterator Res = Result.begin(), ResEnd = Result.end();
         Res != ResEnd; ++Res) {
      if (isa<TypeDecl>(*Res) || isa<ObjCInterfaceDecl>(*Res)) {
        if (!IIDecl || 
            (*Res)->getLocation().getRawEncoding() < 
              IIDecl->getLocation().getRawEncoding())
          IIDecl = *Res;
      }
    }

    if (!IIDecl) {
      // None of the entities we found is a type, so there is no way
      // to even assume that the result is a type. In this case, don't
      // complain about the ambiguity. The parser will either try to
      // perform this lookup again (e.g., as an object name), which
      // will produce the ambiguity, or will complain that it expected
      // a type name.
      Result.Destroy();
      return 0;
    }

    // We found a type within the ambiguous lookup; diagnose the
    // ambiguity and then return that type. This might be the right
    // answer, or it might not be, but it suppresses any attempt to
    // perform the name lookup again.
    DiagnoseAmbiguousLookup(Result, DeclarationName(&II), NameLoc);
    break;
  }

  case LookupResult::Found:
    IIDecl = Result.getAsDecl();
    break;
  }

  if (IIDecl) {
    QualType T;
  
    if (TypeDecl *TD = dyn_cast<TypeDecl>(IIDecl)) {
      // Check whether we can use this type
      (void)DiagnoseUseOfDecl(IIDecl, NameLoc);
  
      if (getLangOptions().CPlusPlus) {
        // C++ [temp.local]p2:
        //   Within the scope of a class template specialization or
        //   partial specialization, when the injected-class-name is
        //   not followed by a <, it is equivalent to the
        //   injected-class-name followed by the template-argument s
        //   of the class template specialization or partial
        //   specialization enclosed in <>.
        if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(TD))
          if (RD->isInjectedClassName())
            if (ClassTemplateDecl *Template = RD->getDescribedClassTemplate())
              T = Template->getInjectedClassNameType(Context);
      }

      if (T.isNull())
        T = Context.getTypeDeclType(TD);
    } else if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(IIDecl)) {
      // Check whether we can use this interface.
      (void)DiagnoseUseOfDecl(IIDecl, NameLoc);
      
      T = Context.getObjCInterfaceType(IDecl);
    } else
      return 0;

    if (SS)
      T = getQualifiedNameType(*SS, T);

    return T.getAsOpaquePtr();
  }

  return 0;
}

/// isTagName() - This method is called *for error recovery purposes only*
/// to determine if the specified name is a valid tag name ("struct foo").  If
/// so, this returns the TST for the tag corresponding to it (TST_enum,
/// TST_union, TST_struct, TST_class).  This is used to diagnose cases in C
/// where the user forgot to specify the tag.
DeclSpec::TST Sema::isTagName(IdentifierInfo &II, Scope *S) {
  // Do a tag name lookup in this scope.
  LookupResult R = LookupName(S, &II, LookupTagName, false, false);
  if (R.getKind() == LookupResult::Found)
    if (const TagDecl *TD = dyn_cast<TagDecl>(R.getAsDecl())) {
      switch (TD->getTagKind()) {
      case TagDecl::TK_struct: return DeclSpec::TST_struct;
      case TagDecl::TK_union:  return DeclSpec::TST_union;
      case TagDecl::TK_class:  return DeclSpec::TST_class;
      case TagDecl::TK_enum:   return DeclSpec::TST_enum;
      }
    }
  
  return DeclSpec::TST_unspecified;
}


// Determines the context to return to after temporarily entering a
// context.  This depends in an unnecessarily complicated way on the
// exact ordering of callbacks from the parser.
DeclContext *Sema::getContainingDC(DeclContext *DC) {

  // Functions defined inline within classes aren't parsed until we've
  // finished parsing the top-level class, so the top-level class is
  // the context we'll need to return to.
  if (isa<FunctionDecl>(DC)) {
    DC = DC->getLexicalParent();

    // A function not defined within a class will always return to its
    // lexical context.
    if (!isa<CXXRecordDecl>(DC))
      return DC;

    // A C++ inline method/friend is parsed *after* the topmost class
    // it was declared in is fully parsed ("complete");  the topmost
    // class is the context we need to return to.
    while (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC->getLexicalParent()))
      DC = RD;

    // Return the declaration context of the topmost class the inline method is
    // declared in.
    return DC;
  }

  if (isa<ObjCMethodDecl>(DC))
    return Context.getTranslationUnitDecl();

  return DC->getLexicalParent();
}

void Sema::PushDeclContext(Scope *S, DeclContext *DC) {
  assert(getContainingDC(DC) == CurContext &&
      "The next DeclContext should be lexically contained in the current one.");
  CurContext = DC;
  S->setEntity(DC);
}

void Sema::PopDeclContext() {
  assert(CurContext && "DeclContext imbalance!");

  CurContext = getContainingDC(CurContext);
}

/// EnterDeclaratorContext - Used when we must lookup names in the context
/// of a declarator's nested name specifier.
void Sema::EnterDeclaratorContext(Scope *S, DeclContext *DC) {
  assert(PreDeclaratorDC == 0 && "Previous declarator context not popped?");
  PreDeclaratorDC = static_cast<DeclContext*>(S->getEntity());
  CurContext = DC;
  assert(CurContext && "No context?");
  S->setEntity(CurContext);
}

void Sema::ExitDeclaratorContext(Scope *S) {
  S->setEntity(PreDeclaratorDC);
  PreDeclaratorDC = 0;

  // Reset CurContext to the nearest enclosing context.
  while (!S->getEntity() && S->getParent())
    S = S->getParent();
  CurContext = static_cast<DeclContext*>(S->getEntity());
  assert(CurContext && "No context?");
}

/// \brief Determine whether we allow overloading of the function
/// PrevDecl with another declaration.
///
/// This routine determines whether overloading is possible, not
/// whether some new function is actually an overload. It will return
/// true in C++ (where we can always provide overloads) or, as an
/// extension, in C when the previous function is already an
/// overloaded function declaration or has the "overloadable"
/// attribute.
static bool AllowOverloadingOfFunction(Decl *PrevDecl, ASTContext &Context) {
  if (Context.getLangOptions().CPlusPlus)
    return true;

  if (isa<OverloadedFunctionDecl>(PrevDecl))
    return true;

  return PrevDecl->getAttr<OverloadableAttr>() != 0;
}

/// Add this decl to the scope shadowed decl chains.
void Sema::PushOnScopeChains(NamedDecl *D, Scope *S) {
  // Move up the scope chain until we find the nearest enclosing
  // non-transparent context. The declaration will be introduced into this
  // scope.
  while (S->getEntity() && 
         ((DeclContext *)S->getEntity())->isTransparentContext())
    S = S->getParent();

  S->AddDecl(DeclPtrTy::make(D));

  // Add scoped declarations into their context, so that they can be
  // found later. Declarations without a context won't be inserted
  // into any context.
  CurContext->addDecl(D);

  // C++ [basic.scope]p4:
  //   -- exactly one declaration shall declare a class name or
  //   enumeration name that is not a typedef name and the other
  //   declarations shall all refer to the same object or
  //   enumerator, or all refer to functions and function templates;
  //   in this case the class name or enumeration name is hidden.
  if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    // We are pushing the name of a tag (enum or class).
    if (CurContext->getLookupContext() 
          == TD->getDeclContext()->getLookupContext()) {
      // We're pushing the tag into the current context, which might
      // require some reshuffling in the identifier resolver.
      IdentifierResolver::iterator
        I = IdResolver.begin(TD->getDeclName()),
        IEnd = IdResolver.end();
      if (I != IEnd && isDeclInScope(*I, CurContext, S)) {
        NamedDecl *PrevDecl = *I;
        for (; I != IEnd && isDeclInScope(*I, CurContext, S); 
             PrevDecl = *I, ++I) {
          if (TD->declarationReplaces(*I)) {
            // This is a redeclaration. Remove it from the chain and
            // break out, so that we'll add in the shadowed
            // declaration.
            S->RemoveDecl(DeclPtrTy::make(*I));
            if (PrevDecl == *I) {
              IdResolver.RemoveDecl(*I);
              IdResolver.AddDecl(TD);
              return;
            } else {
              IdResolver.RemoveDecl(*I);
              break;
            }
          }
        }

        // There is already a declaration with the same name in the same
        // scope, which is not a tag declaration. It must be found
        // before we find the new declaration, so insert the new
        // declaration at the end of the chain.
        IdResolver.AddShadowedDecl(TD, PrevDecl);
        
        return;
      }
    }
  } else if ((isa<FunctionDecl>(D) &&
              AllowOverloadingOfFunction(D, Context)) ||
             isa<FunctionTemplateDecl>(D)) {
    // We are pushing the name of a function or function template,
    // which might be an overloaded name.
    IdentifierResolver::iterator Redecl
      = std::find_if(IdResolver.begin(D->getDeclName()),
                     IdResolver.end(),
                     std::bind1st(std::mem_fun(&NamedDecl::declarationReplaces),
                                  D));
    if (Redecl != IdResolver.end() &&
        S->isDeclScope(DeclPtrTy::make(*Redecl))) {
      // There is already a declaration of a function on our
      // IdResolver chain. Replace it with this declaration.
      S->RemoveDecl(DeclPtrTy::make(*Redecl));
      IdResolver.RemoveDecl(*Redecl);
    }
  } else if (isa<ObjCInterfaceDecl>(D)) {
    // We're pushing an Objective-C interface into the current
    // context. If there is already an alias declaration, remove it first.
    for (IdentifierResolver::iterator 
           I = IdResolver.begin(D->getDeclName()), IEnd = IdResolver.end();
         I != IEnd; ++I) {
      if (isa<ObjCCompatibleAliasDecl>(*I)) {
        S->RemoveDecl(DeclPtrTy::make(*I));
        IdResolver.RemoveDecl(*I);
        break;
      }
    }
  }

  IdResolver.AddDecl(D);
}

void Sema::ActOnPopScope(SourceLocation Loc, Scope *S) {
  if (S->decl_empty()) return;
  assert((S->getFlags() & (Scope::DeclScope | Scope::TemplateParamScope)) &&
	 "Scope shouldn't contain decls!");

  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    Decl *TmpD = (*I).getAs<Decl>();
    assert(TmpD && "This decl didn't get pushed??");

    assert(isa<NamedDecl>(TmpD) && "Decl isn't NamedDecl?");
    NamedDecl *D = cast<NamedDecl>(TmpD);

    if (!D->getDeclName()) continue;

    // Remove this name from our lexical scope.
    IdResolver.RemoveDecl(D);
  }
}

/// getObjCInterfaceDecl - Look up a for a class declaration in the scope.
/// return 0 if one not found.
ObjCInterfaceDecl *Sema::getObjCInterfaceDecl(IdentifierInfo *Id) {
  // The third "scope" argument is 0 since we aren't enabling lazy built-in
  // creation from this context.
  NamedDecl *IDecl = LookupName(TUScope, Id, LookupOrdinaryName);
  
  return dyn_cast_or_null<ObjCInterfaceDecl>(IDecl);
}

/// getNonFieldDeclScope - Retrieves the innermost scope, starting
/// from S, where a non-field would be declared. This routine copes
/// with the difference between C and C++ scoping rules in structs and
/// unions. For example, the following code is well-formed in C but
/// ill-formed in C++:
/// @code
/// struct S6 {
///   enum { BAR } e;
/// };
/// 
/// void test_S6() {
///   struct S6 a;
///   a.e = BAR;
/// }
/// @endcode
/// For the declaration of BAR, this routine will return a different
/// scope. The scope S will be the scope of the unnamed enumeration
/// within S6. In C++, this routine will return the scope associated
/// with S6, because the enumeration's scope is a transparent
/// context but structures can contain non-field names. In C, this
/// routine will return the translation unit scope, since the
/// enumeration's scope is a transparent context and structures cannot
/// contain non-field names.
Scope *Sema::getNonFieldDeclScope(Scope *S) {
  while (((S->getFlags() & Scope::DeclScope) == 0) ||
         (S->getEntity() && 
          ((DeclContext *)S->getEntity())->isTransparentContext()) ||
         (S->isClassScope() && !getLangOptions().CPlusPlus))
    S = S->getParent();
  return S;
}

void Sema::InitBuiltinVaListType() {
  if (!Context.getBuiltinVaListType().isNull())
    return;
  
  IdentifierInfo *VaIdent = &Context.Idents.get("__builtin_va_list");
  NamedDecl *VaDecl = LookupName(TUScope, VaIdent, LookupOrdinaryName);
  TypedefDecl *VaTypedef = cast<TypedefDecl>(VaDecl);
  Context.setBuiltinVaListType(Context.getTypedefType(VaTypedef));
}

/// LazilyCreateBuiltin - The specified Builtin-ID was first used at
/// file scope.  lazily create a decl for it. ForRedeclaration is true
/// if we're creating this built-in in anticipation of redeclaring the
/// built-in.
NamedDecl *Sema::LazilyCreateBuiltin(IdentifierInfo *II, unsigned bid,
                                     Scope *S, bool ForRedeclaration,
                                     SourceLocation Loc) {
  Builtin::ID BID = (Builtin::ID)bid;

  if (Context.BuiltinInfo.hasVAListUse(BID))
    InitBuiltinVaListType();

  ASTContext::GetBuiltinTypeError Error;
  QualType R = Context.GetBuiltinType(BID, Error);  
  switch (Error) {
  case ASTContext::GE_None:
    // Okay
    break;

  case ASTContext::GE_Missing_stdio:
    if (ForRedeclaration)
      Diag(Loc, diag::err_implicit_decl_requires_stdio)
        << Context.BuiltinInfo.GetName(BID);
    return 0;

  case ASTContext::GE_Missing_setjmp:
    if (ForRedeclaration)
      Diag(Loc, diag::err_implicit_decl_requires_setjmp)
        << Context.BuiltinInfo.GetName(BID);
    return 0;
  }

  if (!ForRedeclaration && Context.BuiltinInfo.isPredefinedLibFunction(BID)) {
    Diag(Loc, diag::ext_implicit_lib_function_decl)
      << Context.BuiltinInfo.GetName(BID)
      << R;
    if (Context.BuiltinInfo.getHeaderName(BID) &&
        Diags.getDiagnosticLevel(diag::ext_implicit_lib_function_decl)
          != Diagnostic::Ignored)
      Diag(Loc, diag::note_please_include_header)
        << Context.BuiltinInfo.getHeaderName(BID)
        << Context.BuiltinInfo.GetName(BID);
  }

  FunctionDecl *New = FunctionDecl::Create(Context,
                                           Context.getTranslationUnitDecl(),
                                           Loc, II, R, /*DInfo=*/0,
                                           FunctionDecl::Extern, false,
                                           /*hasPrototype=*/true);
  New->setImplicit();

  // Create Decl objects for each parameter, adding them to the
  // FunctionDecl.
  if (FunctionProtoType *FT = dyn_cast<FunctionProtoType>(R)) {
    llvm::SmallVector<ParmVarDecl*, 16> Params;
    for (unsigned i = 0, e = FT->getNumArgs(); i != e; ++i)
      Params.push_back(ParmVarDecl::Create(Context, New, SourceLocation(), 0,
                                           FT->getArgType(i), /*DInfo=*/0,
                                           VarDecl::None, 0));
    New->setParams(Context, Params.data(), Params.size());
  }
  
  AddKnownFunctionAttributes(New);  
  
  // TUScope is the translation-unit scope to insert this function into.
  // FIXME: This is hideous. We need to teach PushOnScopeChains to
  // relate Scopes to DeclContexts, and probably eliminate CurContext
  // entirely, but we're not there yet.
  DeclContext *SavedContext = CurContext;
  CurContext = Context.getTranslationUnitDecl();
  PushOnScopeChains(New, TUScope);
  CurContext = SavedContext;
  return New;
}

/// GetStdNamespace - This method gets the C++ "std" namespace. This is where
/// everything from the standard library is defined.
NamespaceDecl *Sema::GetStdNamespace() {
  if (!StdNamespace) {
    IdentifierInfo *StdIdent = &PP.getIdentifierTable().get("std");
    DeclContext *Global = Context.getTranslationUnitDecl();
    Decl *Std = LookupQualifiedName(Global, StdIdent, LookupNamespaceName);
    StdNamespace = dyn_cast_or_null<NamespaceDecl>(Std);
  }
  return StdNamespace;
}

/// MergeTypeDefDecl - We just parsed a typedef 'New' which has the
/// same name and scope as a previous declaration 'Old'.  Figure out
/// how to resolve this situation, merging decls or emitting
/// diagnostics as appropriate. If there was an error, set New to be invalid.
///
void Sema::MergeTypeDefDecl(TypedefDecl *New, Decl *OldD) {
  // If either decl is known invalid already, set the new one to be invalid and
  // don't bother doing any merging checks.
  if (New->isInvalidDecl() || OldD->isInvalidDecl())
    return New->setInvalidDecl();
  
  // Allow multiple definitions for ObjC built-in typedefs.
  // FIXME: Verify the underlying types are equivalent!
  if (getLangOptions().ObjC1) {
    const IdentifierInfo *TypeID = New->getIdentifier();
    switch (TypeID->getLength()) {
    default: break;
    case 2: 
      if (!TypeID->isStr("id"))
        break;
      Context.ObjCIdRedefinitionType = New->getUnderlyingType();
      // Install the built-in type for 'id', ignoring the current definition.
      New->setTypeForDecl(Context.getObjCIdType().getTypePtr());
      return;
    case 5:
      if (!TypeID->isStr("Class"))
        break;
      Context.ObjCClassRedefinitionType = New->getUnderlyingType();
      // Install the built-in type for 'Class', ignoring the current definition.
      New->setTypeForDecl(Context.getObjCClassType().getTypePtr());
      return;
    case 3:
      if (!TypeID->isStr("SEL"))
        break;
      Context.setObjCSelType(Context.getTypeDeclType(New));
      return;
    case 8:
      if (!TypeID->isStr("Protocol"))
        break;
      Context.setObjCProtoType(New->getUnderlyingType());
      return;
    }
    // Fall through - the typedef name was not a builtin type.
  }
  // Verify the old decl was also a type.
  TypeDecl *Old = dyn_cast<TypeDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind) 
      << New->getDeclName();
    if (OldD->getLocation().isValid())
      Diag(OldD->getLocation(), diag::note_previous_definition);
    return New->setInvalidDecl();
  }

  // Determine the "old" type we'll use for checking and diagnostics.  
  QualType OldType;
  if (TypedefDecl *OldTypedef = dyn_cast<TypedefDecl>(Old))
    OldType = OldTypedef->getUnderlyingType();
  else
    OldType = Context.getTypeDeclType(Old);

  // If the typedef types are not identical, reject them in all languages and
  // with any extensions enabled.

  if (OldType != New->getUnderlyingType() && 
      Context.getCanonicalType(OldType) != 
      Context.getCanonicalType(New->getUnderlyingType())) {
    Diag(New->getLocation(), diag::err_redefinition_different_typedef)
      << New->getUnderlyingType() << OldType;
    if (Old->getLocation().isValid())
      Diag(Old->getLocation(), diag::note_previous_definition);
    return New->setInvalidDecl();
  }
  
  if (getLangOptions().Microsoft)
    return;

  // C++ [dcl.typedef]p2:
  //   In a given non-class scope, a typedef specifier can be used to
  //   redefine the name of any type declared in that scope to refer
  //   to the type to which it already refers.
  if (getLangOptions().CPlusPlus) {
    if (!isa<CXXRecordDecl>(CurContext))
      return;
    Diag(New->getLocation(), diag::err_redefinition)
      << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return New->setInvalidDecl();
  }

  // If we have a redefinition of a typedef in C, emit a warning.  This warning
  // is normally mapped to an error, but can be controlled with
  // -Wtypedef-redefinition.  If either the original or the redefinition is
  // in a system header, don't emit this for compatibility with GCC.
  if (PP.getDiagnostics().getSuppressSystemWarnings() &&
      (Context.getSourceManager().isInSystemHeader(Old->getLocation()) ||
       Context.getSourceManager().isInSystemHeader(New->getLocation())))
    return;
  
  Diag(New->getLocation(), diag::warn_redefinition_of_typedef)
    << New->getDeclName();
  Diag(Old->getLocation(), diag::note_previous_definition);
  return;
}

/// DeclhasAttr - returns true if decl Declaration already has the target
/// attribute.
static bool 
DeclHasAttr(const Decl *decl, const Attr *target) {
  for (const Attr *attr = decl->getAttrs(); attr; attr = attr->getNext())
    if (attr->getKind() == target->getKind())
      return true;

  return false;
}

/// MergeAttributes - append attributes from the Old decl to the New one.
static void MergeAttributes(Decl *New, Decl *Old, ASTContext &C) {
  for (const Attr *attr = Old->getAttrs(); attr; attr = attr->getNext()) {
    if (!DeclHasAttr(New, attr) && attr->isMerged()) {
      Attr *NewAttr = attr->clone(C);
      NewAttr->setInherited(true);
      New->addAttr(NewAttr);
    }
  }
}

/// Used in MergeFunctionDecl to keep track of function parameters in
/// C.
struct GNUCompatibleParamWarning {
  ParmVarDecl *OldParm;
  ParmVarDecl *NewParm;
  QualType PromotedType;
};

/// MergeFunctionDecl - We just parsed a function 'New' from
/// declarator D which has the same name and scope as a previous
/// declaration 'Old'.  Figure out how to resolve this situation,
/// merging decls or emitting diagnostics as appropriate.
///
/// In C++, New and Old must be declarations that are not
/// overloaded. Use IsOverload to determine whether New and Old are
/// overloaded, and to select the Old declaration that New should be
/// merged with.
///
/// Returns true if there was an error, false otherwise.
bool Sema::MergeFunctionDecl(FunctionDecl *New, Decl *OldD) {
  assert(!isa<OverloadedFunctionDecl>(OldD) && 
         "Cannot merge with an overloaded function declaration");

  // Verify the old decl was also a function.
  FunctionDecl *Old = 0;
  if (FunctionTemplateDecl *OldFunctionTemplate 
        = dyn_cast<FunctionTemplateDecl>(OldD))
    Old = OldFunctionTemplate->getTemplatedDecl();
  else 
    Old = dyn_cast<FunctionDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind)
      << New->getDeclName();
    Diag(OldD->getLocation(), diag::note_previous_definition);
    return true;
  }

  // Determine whether the previous declaration was a definition,
  // implicit declaration, or a declaration.
  diag::kind PrevDiag;
  if (Old->isThisDeclarationADefinition())
    PrevDiag = diag::note_previous_definition;
  else if (Old->isImplicit())
    PrevDiag = diag::note_previous_implicit_declaration;
  else 
    PrevDiag = diag::note_previous_declaration;
  
  QualType OldQType = Context.getCanonicalType(Old->getType());
  QualType NewQType = Context.getCanonicalType(New->getType());
  
  if (!isa<CXXMethodDecl>(New) && !isa<CXXMethodDecl>(Old) &&
      New->getStorageClass() == FunctionDecl::Static &&
      Old->getStorageClass() != FunctionDecl::Static) {
    Diag(New->getLocation(), diag::err_static_non_static)
      << New;
    Diag(Old->getLocation(), PrevDiag);
    return true;
  }

  if (getLangOptions().CPlusPlus) {
    // (C++98 13.1p2):
    //   Certain function declarations cannot be overloaded:
    //     -- Function declarations that differ only in the return type 
    //        cannot be overloaded.
    QualType OldReturnType 
      = cast<FunctionType>(OldQType.getTypePtr())->getResultType();
    QualType NewReturnType 
      = cast<FunctionType>(NewQType.getTypePtr())->getResultType();
    if (OldReturnType != NewReturnType) {
      Diag(New->getLocation(), diag::err_ovl_diff_return_type);
      Diag(Old->getLocation(), PrevDiag) << Old << Old->getType();
      return true;
    }

    const CXXMethodDecl* OldMethod = dyn_cast<CXXMethodDecl>(Old);
    const CXXMethodDecl* NewMethod = dyn_cast<CXXMethodDecl>(New);
    if (OldMethod && NewMethod && 
        NewMethod->getLexicalDeclContext()->isRecord()) {
      //    -- Member function declarations with the same name and the 
      //       same parameter types cannot be overloaded if any of them 
      //       is a static member function declaration.
      if (OldMethod->isStatic() || NewMethod->isStatic()) {
        Diag(New->getLocation(), diag::err_ovl_static_nonstatic_member);
        Diag(Old->getLocation(), PrevDiag) << Old << Old->getType();
        return true;
      }

      // C++ [class.mem]p1:
      //   [...] A member shall not be declared twice in the
      //   member-specification, except that a nested class or member
      //   class template can be declared and then later defined.
      unsigned NewDiag;
      if (isa<CXXConstructorDecl>(OldMethod))
        NewDiag = diag::err_constructor_redeclared;
      else if (isa<CXXDestructorDecl>(NewMethod))
        NewDiag = diag::err_destructor_redeclared;
      else if (isa<CXXConversionDecl>(NewMethod))
        NewDiag = diag::err_conv_function_redeclared;
      else
        NewDiag = diag::err_member_redeclared;
      
      Diag(New->getLocation(), NewDiag);
      Diag(Old->getLocation(), PrevDiag) << Old << Old->getType();
    }

    // (C++98 8.3.5p3):
    //   All declarations for a function shall agree exactly in both the
    //   return type and the parameter-type-list.
    if (OldQType == NewQType)
      return MergeCompatibleFunctionDecls(New, Old);

    // Fall through for conflicting redeclarations and redefinitions.
  }

  // C: Function types need to be compatible, not identical. This handles
  // duplicate function decls like "void f(int); void f(enum X);" properly.
  if (!getLangOptions().CPlusPlus &&
      Context.typesAreCompatible(OldQType, NewQType)) {
    const FunctionType *OldFuncType = OldQType->getAsFunctionType();
    const FunctionType *NewFuncType = NewQType->getAsFunctionType();
    const FunctionProtoType *OldProto = 0;
    if (isa<FunctionNoProtoType>(NewFuncType) &&
        (OldProto = dyn_cast<FunctionProtoType>(OldFuncType))) {
      // The old declaration provided a function prototype, but the
      // new declaration does not. Merge in the prototype.
      assert(!OldProto->hasExceptionSpec() && "Exception spec in C");
      llvm::SmallVector<QualType, 16> ParamTypes(OldProto->arg_type_begin(),
                                                 OldProto->arg_type_end());
      NewQType = Context.getFunctionType(NewFuncType->getResultType(),
                                         ParamTypes.data(), ParamTypes.size(),
                                         OldProto->isVariadic(),
                                         OldProto->getTypeQuals());
      New->setType(NewQType);
      New->setHasInheritedPrototype();

      // Synthesize a parameter for each argument type.
      llvm::SmallVector<ParmVarDecl*, 16> Params;
      for (FunctionProtoType::arg_type_iterator 
             ParamType = OldProto->arg_type_begin(), 
             ParamEnd = OldProto->arg_type_end();
           ParamType != ParamEnd; ++ParamType) {
        ParmVarDecl *Param = ParmVarDecl::Create(Context, New,
                                                 SourceLocation(), 0,
                                                 *ParamType, /*DInfo=*/0,
                                                 VarDecl::None, 0);
        Param->setImplicit();
        Params.push_back(Param);
      }

      New->setParams(Context, Params.data(), Params.size());
    } 

    return MergeCompatibleFunctionDecls(New, Old);
  }

  // GNU C permits a K&R definition to follow a prototype declaration
  // if the declared types of the parameters in the K&R definition
  // match the types in the prototype declaration, even when the
  // promoted types of the parameters from the K&R definition differ
  // from the types in the prototype. GCC then keeps the types from
  // the prototype.
  //
  // If a variadic prototype is followed by a non-variadic K&R definition,
  // the K&R definition becomes variadic.  This is sort of an edge case, but
  // it's legal per the standard depending on how you read C99 6.7.5.3p15 and
  // C99 6.9.1p8.
  if (!getLangOptions().CPlusPlus &&
      Old->hasPrototype() && !New->hasPrototype() &&
      New->getType()->getAsFunctionProtoType() &&
      Old->getNumParams() == New->getNumParams()) {
    llvm::SmallVector<QualType, 16> ArgTypes;
    llvm::SmallVector<GNUCompatibleParamWarning, 16> Warnings;
    const FunctionProtoType *OldProto 
      = Old->getType()->getAsFunctionProtoType();
    const FunctionProtoType *NewProto 
      = New->getType()->getAsFunctionProtoType();
    
    // Determine whether this is the GNU C extension.
    QualType MergedReturn = Context.mergeTypes(OldProto->getResultType(),
                                               NewProto->getResultType());
    bool LooseCompatible = !MergedReturn.isNull();
    for (unsigned Idx = 0, End = Old->getNumParams(); 
         LooseCompatible && Idx != End; ++Idx) {
      ParmVarDecl *OldParm = Old->getParamDecl(Idx);
      ParmVarDecl *NewParm = New->getParamDecl(Idx);
      if (Context.typesAreCompatible(OldParm->getType(), 
                                     NewProto->getArgType(Idx))) {
        ArgTypes.push_back(NewParm->getType());
      } else if (Context.typesAreCompatible(OldParm->getType(),
                                            NewParm->getType())) {
        GNUCompatibleParamWarning Warn 
          = { OldParm, NewParm, NewProto->getArgType(Idx) };
        Warnings.push_back(Warn);
        ArgTypes.push_back(NewParm->getType());
      } else
        LooseCompatible = false;
    }

    if (LooseCompatible) {
      for (unsigned Warn = 0; Warn < Warnings.size(); ++Warn) {
        Diag(Warnings[Warn].NewParm->getLocation(),
             diag::ext_param_promoted_not_compatible_with_prototype)
          << Warnings[Warn].PromotedType
          << Warnings[Warn].OldParm->getType();
        Diag(Warnings[Warn].OldParm->getLocation(), 
             diag::note_previous_declaration);
      }

      New->setType(Context.getFunctionType(MergedReturn, &ArgTypes[0],
                                           ArgTypes.size(),
                                           OldProto->isVariadic(), 0));
      return MergeCompatibleFunctionDecls(New, Old);
    }

    // Fall through to diagnose conflicting types.
  }

  // A function that has already been declared has been redeclared or defined
  // with a different type- show appropriate diagnostic
  if (unsigned BuiltinID = Old->getBuiltinID(Context)) {
    // The user has declared a builtin function with an incompatible
    // signature.
    if (Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID)) {
      // The function the user is redeclaring is a library-defined
      // function like 'malloc' or 'printf'. Warn about the
      // redeclaration, then pretend that we don't know about this
      // library built-in.
      Diag(New->getLocation(), diag::warn_redecl_library_builtin) << New;
      Diag(Old->getLocation(), diag::note_previous_builtin_declaration)
        << Old << Old->getType();
      New->getIdentifier()->setBuiltinID(Builtin::NotBuiltin);
      Old->setInvalidDecl();
      return false;
    }

    PrevDiag = diag::note_previous_builtin_declaration;
  }

  Diag(New->getLocation(), diag::err_conflicting_types) << New->getDeclName();
  Diag(Old->getLocation(), PrevDiag) << Old << Old->getType();
  return true;
}

/// \brief Completes the merge of two function declarations that are
/// known to be compatible. 
///
/// This routine handles the merging of attributes and other
/// properties of function declarations form the old declaration to
/// the new declaration, once we know that New is in fact a
/// redeclaration of Old.
///
/// \returns false
bool Sema::MergeCompatibleFunctionDecls(FunctionDecl *New, FunctionDecl *Old) {
  // Merge the attributes
  MergeAttributes(New, Old, Context);

  // Merge the storage class.
  if (Old->getStorageClass() != FunctionDecl::Extern)
    New->setStorageClass(Old->getStorageClass());

  // Merge "inline"
  if (Old->isInline())
    New->setInline(true);

  // If this function declaration by itself qualifies as a C99 inline
  // definition (C99 6.7.4p6), but the previous definition did not,
  // then the function is not a C99 inline definition.
  if (New->isC99InlineDefinition() && !Old->isC99InlineDefinition())
    New->setC99InlineDefinition(false);
  else if (Old->isC99InlineDefinition() && !New->isC99InlineDefinition()) {
    // Mark all preceding definitions as not being C99 inline definitions.
    for (const FunctionDecl *Prev = Old; Prev; 
         Prev = Prev->getPreviousDeclaration())
      const_cast<FunctionDecl *>(Prev)->setC99InlineDefinition(false);
  }

  // Merge "pure" flag.
  if (Old->isPure())
    New->setPure();

  // Merge the "deleted" flag.
  if (Old->isDeleted())
    New->setDeleted();
  
  if (getLangOptions().CPlusPlus)
    return MergeCXXFunctionDecl(New, Old);

  return false;
}

/// MergeVarDecl - We just parsed a variable 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
/// Tentative definition rules (C99 6.9.2p2) are checked by 
/// FinalizeDeclaratorGroup. Unfortunately, we can't analyze tentative 
/// definitions here, since the initializer hasn't been attached.
/// 
void Sema::MergeVarDecl(VarDecl *New, Decl *OldD) {
  // If either decl is invalid, make sure the new one is marked invalid and
  // don't do any other checking.
  if (New->isInvalidDecl() || OldD->isInvalidDecl())
    return New->setInvalidDecl();
  
  // Verify the old decl was also a variable.
  VarDecl *Old = dyn_cast<VarDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind)
      << New->getDeclName();
    Diag(OldD->getLocation(), diag::note_previous_definition);
    return New->setInvalidDecl();
  }

  MergeAttributes(New, Old, Context);

  // Merge the types
  QualType MergedT;
  if (getLangOptions().CPlusPlus) {
    if (Context.hasSameType(New->getType(), Old->getType()))
      MergedT = New->getType();
  } else {
    MergedT = Context.mergeTypes(New->getType(), Old->getType());
  }
  if (MergedT.isNull()) {
    Diag(New->getLocation(), diag::err_redefinition_different_type) 
      << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return New->setInvalidDecl();
  }
  New->setType(MergedT);

  // C99 6.2.2p4: Check if we have a static decl followed by a non-static.
  if (New->getStorageClass() == VarDecl::Static &&
      (Old->getStorageClass() == VarDecl::None || Old->hasExternalStorage())) {
    Diag(New->getLocation(), diag::err_static_non_static) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return New->setInvalidDecl();
  }
  // C99 6.2.2p4: 
  //   For an identifier declared with the storage-class specifier
  //   extern in a scope in which a prior declaration of that
  //   identifier is visible,23) if the prior declaration specifies
  //   internal or external linkage, the linkage of the identifier at
  //   the later declaration is the same as the linkage specified at
  //   the prior declaration. If no prior declaration is visible, or
  //   if the prior declaration specifies no linkage, then the
  //   identifier has external linkage.
  if (New->hasExternalStorage() && Old->hasLinkage())
    /* Okay */;
  else if (New->getStorageClass() != VarDecl::Static &&
           Old->getStorageClass() == VarDecl::Static) {
    Diag(New->getLocation(), diag::err_non_static_static) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return New->setInvalidDecl();
  }

  // Variables with external linkage are analyzed in FinalizeDeclaratorGroup.
  
  // FIXME: The test for external storage here seems wrong? We still
  // need to check for mismatches.
  if (!New->hasExternalStorage() && !New->isFileVarDecl() &&
      // Don't complain about out-of-line definitions of static members.
      !(Old->getLexicalDeclContext()->isRecord() &&
        !New->getLexicalDeclContext()->isRecord())) {
    Diag(New->getLocation(), diag::err_redefinition) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return New->setInvalidDecl();
  }

  if (New->isThreadSpecified() && !Old->isThreadSpecified()) {
    Diag(New->getLocation(), diag::err_thread_non_thread) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
  } else if (!New->isThreadSpecified() && Old->isThreadSpecified()) {
    Diag(New->getLocation(), diag::err_non_thread_thread) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
  }

  // Keep a chain of previous declarations.
  New->setPreviousDeclaration(Old);
}

/// CheckFallThrough - Check that we don't fall off the end of a
/// Statement that should return a value.
///
/// \returns AlwaysFallThrough iff we always fall off the end of the statement,
/// MaybeFallThrough iff we might or might not fall off the end and
/// NeverFallThrough iff we never fall off the end of the statement.  We assume
/// that functions not marked noreturn will return.
Sema::ControlFlowKind Sema::CheckFallThrough(Stmt *Root) {
  llvm::OwningPtr<CFG> cfg (CFG::buildCFG(Root, &Context));
  
  // FIXME: They should never return 0, fix that, delete this code.
  if (cfg == 0)
    return NeverFallThrough;
  // The CFG leaves in dead things, and we don't want to dead code paths to
  // confuse us, so we mark all live things first.
  std::queue<CFGBlock*> workq;
  llvm::BitVector live(cfg->getNumBlockIDs());
  // Prep work queue
  workq.push(&cfg->getEntry());
  // Solve
  while (!workq.empty()) {
    CFGBlock *item = workq.front();
    workq.pop();
    live.set(item->getBlockID());
    for (CFGBlock::succ_iterator I=item->succ_begin(),
           E=item->succ_end();
         I != E;
         ++I) {
      if ((*I) && !live[(*I)->getBlockID()]) {
        live.set((*I)->getBlockID());
        workq.push(*I);
      }
    }
  }

  // Now we know what is live, we check the live precessors of the exit block
  // and look for fall through paths, being careful to ignore normal returns,
  // and exceptional paths.
  bool HasLiveReturn = false;
  bool HasFakeEdge = false;
  bool HasPlainEdge = false;
  for (CFGBlock::succ_iterator I=cfg->getExit().pred_begin(),
         E = cfg->getExit().pred_end();
       I != E;
       ++I) {
    CFGBlock& B = **I;
    if (!live[B.getBlockID()])
      continue;
    if (B.size() == 0) {
      // A labeled empty statement, or the entry block...
      HasPlainEdge = true;
      continue;
    }
    Stmt *S = B[B.size()-1];
    if (isa<ReturnStmt>(S)) {
      HasLiveReturn = true;
      continue;
    }
    if (isa<ObjCAtThrowStmt>(S)) {
      HasFakeEdge = true;
      continue;
    }
    if (isa<CXXThrowExpr>(S)) {
      HasFakeEdge = true;
      continue;
    }
    bool NoReturnEdge = false;
    if (CallExpr *C = dyn_cast<CallExpr>(S)) {
      Expr *CEE = C->getCallee()->IgnoreParenCasts();
      if (CEE->getType().getNoReturnAttr()) {
        NoReturnEdge = true;
        HasFakeEdge = true;
      } else if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CEE)) {
        if (FunctionDecl *FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
          if (FD->hasAttr<NoReturnAttr>()) {
            NoReturnEdge = true;
            HasFakeEdge = true;
          }
        }
      }
    }
    // FIXME: Add noreturn message sends.
    if (NoReturnEdge == false)
      HasPlainEdge = true;
  }
  if (!HasPlainEdge)
    return NeverFallThrough;
  if (HasFakeEdge || HasLiveReturn)
    return MaybeFallThrough;
  // This says AlwaysFallThrough for calls to functions that are not marked
  // noreturn, that don't return.  If people would like this warning to be more
  // accurate, such functions should be marked as noreturn.
  return AlwaysFallThrough;
}

/// CheckFallThroughForFunctionDef - Check that we don't fall off the end of a
/// function that should return a value.  Check that we don't fall off the end
/// of a noreturn function.  We assume that functions and blocks not marked
/// noreturn will return.
void Sema::CheckFallThroughForFunctionDef(Decl *D, Stmt *Body) {
  // FIXME: Would be nice if we had a better way to control cascading errors,
  // but for now, avoid them.  The problem is that when Parse sees:
  //   int foo() { return a; }
  // The return is eaten and the Sema code sees just:
  //   int foo() { }
  // which this code would then warn about.
  if (getDiagnostics().hasErrorOccurred())
    return;
  bool ReturnsVoid = false;
  bool HasNoReturn = false;
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->getResultType()->isVoidType())
      ReturnsVoid = true;
    if (FD->hasAttr<NoReturnAttr>())
      HasNoReturn = true;
  } else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (MD->getResultType()->isVoidType())
      ReturnsVoid = true;
    if (MD->hasAttr<NoReturnAttr>())
      HasNoReturn = true;
  }
    
  // Short circuit for compilation speed.
  if ((Diags.getDiagnosticLevel(diag::warn_maybe_falloff_nonvoid_function)
       == Diagnostic::Ignored || ReturnsVoid)
      && (Diags.getDiagnosticLevel(diag::warn_noreturn_function_has_return_expr)
          == Diagnostic::Ignored || !HasNoReturn)
      && (Diags.getDiagnosticLevel(diag::warn_suggest_noreturn_block)
          == Diagnostic::Ignored || !ReturnsVoid))
    return;
  // FIXME: Funtion try block
  if (CompoundStmt *Compound = dyn_cast<CompoundStmt>(Body)) {
    switch (CheckFallThrough(Body)) {
    case MaybeFallThrough:
      if (HasNoReturn)
        Diag(Compound->getRBracLoc(), diag::warn_falloff_noreturn_function);
      else if (!ReturnsVoid)
        Diag(Compound->getRBracLoc(),diag::warn_maybe_falloff_nonvoid_function);
      break;
    case AlwaysFallThrough:
      if (HasNoReturn)
        Diag(Compound->getRBracLoc(), diag::warn_falloff_noreturn_function);
      else if (!ReturnsVoid)
        Diag(Compound->getRBracLoc(), diag::warn_falloff_nonvoid_function);
      break;
    case NeverFallThrough:
      if (ReturnsVoid)
        Diag(Compound->getLBracLoc(), diag::warn_suggest_noreturn_function);
      break;
    }
  }
}

/// CheckFallThroughForBlock - Check that we don't fall off the end of a block
/// that should return a value.  Check that we don't fall off the end of a
/// noreturn block.  We assume that functions and blocks not marked noreturn
/// will return.
void Sema::CheckFallThroughForBlock(QualType BlockTy, Stmt *Body) {
  // FIXME: Would be nice if we had a better way to control cascading errors,
  // but for now, avoid them.  The problem is that when Parse sees:
  //   int foo() { return a; }
  // The return is eaten and the Sema code sees just:
  //   int foo() { }
  // which this code would then warn about.
  if (getDiagnostics().hasErrorOccurred())
    return;
  bool ReturnsVoid = false;
  bool HasNoReturn = false;
  if (const FunctionType *FT = BlockTy->getPointeeType()->getAsFunctionType()) {
    if (FT->getResultType()->isVoidType())
      ReturnsVoid = true;
    if (FT->getNoReturnAttr())
      HasNoReturn = true;
  }
    
  // Short circuit for compilation speed.
  if (ReturnsVoid
      && !HasNoReturn
      && (Diags.getDiagnosticLevel(diag::warn_suggest_noreturn_block)
          == Diagnostic::Ignored || !ReturnsVoid))
    return;
  // FIXME: Funtion try block
  if (CompoundStmt *Compound = dyn_cast<CompoundStmt>(Body)) {
    switch (CheckFallThrough(Body)) {
    case MaybeFallThrough:
      if (HasNoReturn)
        Diag(Compound->getRBracLoc(), diag::err_noreturn_block_has_return_expr);
      else if (!ReturnsVoid)
        Diag(Compound->getRBracLoc(), diag::err_maybe_falloff_nonvoid_block);
      break;
    case AlwaysFallThrough:
      if (HasNoReturn)
        Diag(Compound->getRBracLoc(), diag::err_noreturn_block_has_return_expr);
      else if (!ReturnsVoid)
        Diag(Compound->getRBracLoc(), diag::err_falloff_nonvoid_block);
      break;
    case NeverFallThrough:
      if (ReturnsVoid)
        Diag(Compound->getLBracLoc(), diag::warn_suggest_noreturn_block);
      break;
    }
  }
}

/// CheckParmsForFunctionDef - Check that the parameters of the given
/// function are appropriate for the definition of a function. This
/// takes care of any checks that cannot be performed on the
/// declaration itself, e.g., that the types of each of the function
/// parameters are complete.
bool Sema::CheckParmsForFunctionDef(FunctionDecl *FD) {
  bool HasInvalidParm = false;
  for (unsigned p = 0, NumParams = FD->getNumParams(); p < NumParams; ++p) {
    ParmVarDecl *Param = FD->getParamDecl(p);

    // C99 6.7.5.3p4: the parameters in a parameter type list in a
    // function declarator that is part of a function definition of
    // that function shall not have incomplete type.
    //
    // This is also C++ [dcl.fct]p6.
    if (!Param->isInvalidDecl() &&
        RequireCompleteType(Param->getLocation(), Param->getType(),
                               diag::err_typecheck_decl_incomplete_type)) {
      Param->setInvalidDecl();
      HasInvalidParm = true;
    }
    
    // C99 6.9.1p5: If the declarator includes a parameter type list, the
    // declaration of each parameter shall include an identifier.
    if (Param->getIdentifier() == 0 && 
        !Param->isImplicit() &&
        !getLangOptions().CPlusPlus)
      Diag(Param->getLocation(), diag::err_parameter_name_omitted);
  }

  return HasInvalidParm;
}

/// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
/// no declarator (e.g. "struct foo;") is parsed.
Sema::DeclPtrTy Sema::ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
  // FIXME: Error on auto/register at file scope
  // FIXME: Error on inline/virtual/explicit
  // FIXME: Error on invalid restrict
  // FIXME: Warn on useless __thread
  // FIXME: Warn on useless const/volatile
  // FIXME: Warn on useless static/extern/typedef/private_extern/mutable
  // FIXME: Warn on useless attributes
  TagDecl *Tag = 0;
  if (DS.getTypeSpecType() == DeclSpec::TST_class ||
      DS.getTypeSpecType() == DeclSpec::TST_struct ||
      DS.getTypeSpecType() == DeclSpec::TST_union ||
      DS.getTypeSpecType() == DeclSpec::TST_enum) {
    if (!DS.getTypeRep()) // We probably had an error
      return DeclPtrTy();

    // Note that the above type specs guarantee that the
    // type rep is a Decl, whereas in many of the others
    // it's a Type.
    Tag = dyn_cast<TagDecl>(static_cast<Decl *>(DS.getTypeRep()));
  }

  if (RecordDecl *Record = dyn_cast_or_null<RecordDecl>(Tag)) {
    if (!Record->getDeclName() && Record->isDefinition() &&
        DS.getStorageClassSpec() != DeclSpec::SCS_typedef) {
      if (getLangOptions().CPlusPlus ||
          Record->getDeclContext()->isRecord())
        return BuildAnonymousStructOrUnion(S, DS, Record);

      Diag(DS.getSourceRange().getBegin(), diag::err_no_declarators)
        << DS.getSourceRange();
    }

    // Microsoft allows unnamed struct/union fields. Don't complain
    // about them.
    // FIXME: Should we support Microsoft's extensions in this area?
    if (Record->getDeclName() && getLangOptions().Microsoft)
      return DeclPtrTy::make(Tag);
  }

  if (!DS.isMissingDeclaratorOk() && 
      DS.getTypeSpecType() != DeclSpec::TST_error) {
    // Warn about typedefs of enums without names, since this is an
    // extension in both Microsoft an GNU.
    if (DS.getStorageClassSpec() == DeclSpec::SCS_typedef &&
        Tag && isa<EnumDecl>(Tag)) {
      Diag(DS.getSourceRange().getBegin(), diag::ext_typedef_without_a_name)
        << DS.getSourceRange();
      return DeclPtrTy::make(Tag);
    }

    Diag(DS.getSourceRange().getBegin(), diag::err_no_declarators)
      << DS.getSourceRange();
    return DeclPtrTy();
  }
  
  return DeclPtrTy::make(Tag);
}

/// InjectAnonymousStructOrUnionMembers - Inject the members of the
/// anonymous struct or union AnonRecord into the owning context Owner
/// and scope S. This routine will be invoked just after we realize
/// that an unnamed union or struct is actually an anonymous union or
/// struct, e.g.,
///
/// @code
/// union {
///   int i;
///   float f;
/// }; // InjectAnonymousStructOrUnionMembers called here to inject i and
///    // f into the surrounding scope.x
/// @endcode
///
/// This routine is recursive, injecting the names of nested anonymous
/// structs/unions into the owning context and scope as well.
bool Sema::InjectAnonymousStructOrUnionMembers(Scope *S, DeclContext *Owner,
                                               RecordDecl *AnonRecord) {
  bool Invalid = false;
  for (RecordDecl::field_iterator F = AnonRecord->field_begin(),
                               FEnd = AnonRecord->field_end();
       F != FEnd; ++F) {
    if ((*F)->getDeclName()) {
      NamedDecl *PrevDecl = LookupQualifiedName(Owner, (*F)->getDeclName(),
                                                LookupOrdinaryName, true);
      if (PrevDecl && !isa<TagDecl>(PrevDecl)) {
        // C++ [class.union]p2:
        //   The names of the members of an anonymous union shall be
        //   distinct from the names of any other entity in the
        //   scope in which the anonymous union is declared.
        unsigned diagKind 
          = AnonRecord->isUnion()? diag::err_anonymous_union_member_redecl
                                 : diag::err_anonymous_struct_member_redecl;
        Diag((*F)->getLocation(), diagKind)
          << (*F)->getDeclName();
        Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
        Invalid = true;
      } else {
        // C++ [class.union]p2:
        //   For the purpose of name lookup, after the anonymous union
        //   definition, the members of the anonymous union are
        //   considered to have been defined in the scope in which the
        //   anonymous union is declared.
        Owner->makeDeclVisibleInContext(*F);
        S->AddDecl(DeclPtrTy::make(*F));
        IdResolver.AddDecl(*F);
      }
    } else if (const RecordType *InnerRecordType
                 = (*F)->getType()->getAs<RecordType>()) {
      RecordDecl *InnerRecord = InnerRecordType->getDecl();
      if (InnerRecord->isAnonymousStructOrUnion())
        Invalid = Invalid || 
          InjectAnonymousStructOrUnionMembers(S, Owner, InnerRecord);
    }
  }

  return Invalid;
}

/// ActOnAnonymousStructOrUnion - Handle the declaration of an
/// anonymous structure or union. Anonymous unions are a C++ feature
/// (C++ [class.union]) and a GNU C extension; anonymous structures
/// are a GNU C and GNU C++ extension. 
Sema::DeclPtrTy Sema::BuildAnonymousStructOrUnion(Scope *S, DeclSpec &DS,
                                                  RecordDecl *Record) {
  DeclContext *Owner = Record->getDeclContext();

  // Diagnose whether this anonymous struct/union is an extension.
  if (Record->isUnion() && !getLangOptions().CPlusPlus)
    Diag(Record->getLocation(), diag::ext_anonymous_union);
  else if (!Record->isUnion())
    Diag(Record->getLocation(), diag::ext_anonymous_struct);
  
  // C and C++ require different kinds of checks for anonymous
  // structs/unions.
  bool Invalid = false;
  if (getLangOptions().CPlusPlus) {
    const char* PrevSpec = 0;
    unsigned DiagID;
    // C++ [class.union]p3:
    //   Anonymous unions declared in a named namespace or in the
    //   global namespace shall be declared static.
    if (DS.getStorageClassSpec() != DeclSpec::SCS_static &&
        (isa<TranslationUnitDecl>(Owner) ||
         (isa<NamespaceDecl>(Owner) && 
          cast<NamespaceDecl>(Owner)->getDeclName()))) {
      Diag(Record->getLocation(), diag::err_anonymous_union_not_static);
      Invalid = true;

      // Recover by adding 'static'.
      DS.SetStorageClassSpec(DeclSpec::SCS_static, SourceLocation(),
                             PrevSpec, DiagID);
    } 
    // C++ [class.union]p3:
    //   A storage class is not allowed in a declaration of an
    //   anonymous union in a class scope.
    else if (DS.getStorageClassSpec() != DeclSpec::SCS_unspecified &&
             isa<RecordDecl>(Owner)) {
      Diag(DS.getStorageClassSpecLoc(), 
           diag::err_anonymous_union_with_storage_spec);
      Invalid = true;

      // Recover by removing the storage specifier.
      DS.SetStorageClassSpec(DeclSpec::SCS_unspecified, SourceLocation(),
                             PrevSpec, DiagID);
    }

    // C++ [class.union]p2: 
    //   The member-specification of an anonymous union shall only
    //   define non-static data members. [Note: nested types and
    //   functions cannot be declared within an anonymous union. ]
    for (DeclContext::decl_iterator Mem = Record->decls_begin(),
                                 MemEnd = Record->decls_end();
         Mem != MemEnd; ++Mem) {
      if (FieldDecl *FD = dyn_cast<FieldDecl>(*Mem)) {
        // C++ [class.union]p3:
        //   An anonymous union shall not have private or protected
        //   members (clause 11).
        if (FD->getAccess() == AS_protected || FD->getAccess() == AS_private) {
          Diag(FD->getLocation(), diag::err_anonymous_record_nonpublic_member)
            << (int)Record->isUnion() << (int)(FD->getAccess() == AS_protected);
          Invalid = true;
        }
      } else if ((*Mem)->isImplicit()) {
        // Any implicit members are fine.
      } else if (isa<TagDecl>(*Mem) && (*Mem)->getDeclContext() != Record) {
        // This is a type that showed up in an
        // elaborated-type-specifier inside the anonymous struct or
        // union, but which actually declares a type outside of the
        // anonymous struct or union. It's okay.
      } else if (RecordDecl *MemRecord = dyn_cast<RecordDecl>(*Mem)) {
        if (!MemRecord->isAnonymousStructOrUnion() &&
            MemRecord->getDeclName()) {
          // This is a nested type declaration.
          Diag(MemRecord->getLocation(), diag::err_anonymous_record_with_type)
            << (int)Record->isUnion();
          Invalid = true;
        }
      } else {
        // We have something that isn't a non-static data
        // member. Complain about it.
        unsigned DK = diag::err_anonymous_record_bad_member;
        if (isa<TypeDecl>(*Mem))
          DK = diag::err_anonymous_record_with_type;
        else if (isa<FunctionDecl>(*Mem))
          DK = diag::err_anonymous_record_with_function;
        else if (isa<VarDecl>(*Mem))
          DK = diag::err_anonymous_record_with_static;
        Diag((*Mem)->getLocation(), DK)
            << (int)Record->isUnion();
          Invalid = true;
      }
    }
  } 

  if (!Record->isUnion() && !Owner->isRecord()) {
    Diag(Record->getLocation(), diag::err_anonymous_struct_not_member)
      << (int)getLangOptions().CPlusPlus;
    Invalid = true;
  }

  // Create a declaration for this anonymous struct/union. 
  NamedDecl *Anon = 0;
  if (RecordDecl *OwningClass = dyn_cast<RecordDecl>(Owner)) {
    Anon = FieldDecl::Create(Context, OwningClass, Record->getLocation(),
                             /*IdentifierInfo=*/0, 
                             Context.getTypeDeclType(Record),
                             // FIXME: Type source info.
                             /*DInfo=*/0,
                             /*BitWidth=*/0, /*Mutable=*/false);
    Anon->setAccess(AS_public);
    if (getLangOptions().CPlusPlus)
      FieldCollector->Add(cast<FieldDecl>(Anon));
  } else {
    VarDecl::StorageClass SC;
    switch (DS.getStorageClassSpec()) {
    default: assert(0 && "Unknown storage class!");
    case DeclSpec::SCS_unspecified:    SC = VarDecl::None; break;
    case DeclSpec::SCS_extern:         SC = VarDecl::Extern; break;
    case DeclSpec::SCS_static:         SC = VarDecl::Static; break;
    case DeclSpec::SCS_auto:           SC = VarDecl::Auto; break;
    case DeclSpec::SCS_register:       SC = VarDecl::Register; break;
    case DeclSpec::SCS_private_extern: SC = VarDecl::PrivateExtern; break;
    case DeclSpec::SCS_mutable:
      // mutable can only appear on non-static class members, so it's always
      // an error here
      Diag(Record->getLocation(), diag::err_mutable_nonmember);
      Invalid = true;
      SC = VarDecl::None;
      break;
    }

    Anon = VarDecl::Create(Context, Owner, Record->getLocation(),
                           /*IdentifierInfo=*/0, 
                           Context.getTypeDeclType(Record),
                           // FIXME: Type source info.
                           /*DInfo=*/0,
                           SC);
  }
  Anon->setImplicit();

  // Add the anonymous struct/union object to the current
  // context. We'll be referencing this object when we refer to one of
  // its members.
  Owner->addDecl(Anon);

  // Inject the members of the anonymous struct/union into the owning
  // context and into the identifier resolver chain for name lookup
  // purposes.
  if (InjectAnonymousStructOrUnionMembers(S, Owner, Record))
    Invalid = true;

  // Mark this as an anonymous struct/union type. Note that we do not
  // do this until after we have already checked and injected the
  // members of this anonymous struct/union type, because otherwise
  // the members could be injected twice: once by DeclContext when it
  // builds its lookup table, and once by
  // InjectAnonymousStructOrUnionMembers. 
  Record->setAnonymousStructOrUnion(true);

  if (Invalid)
    Anon->setInvalidDecl();

  return DeclPtrTy::make(Anon);
}


/// GetNameForDeclarator - Determine the full declaration name for the
/// given Declarator.
DeclarationName Sema::GetNameForDeclarator(Declarator &D) {
  switch (D.getKind()) {
  case Declarator::DK_Abstract:
    assert(D.getIdentifier() == 0 && "abstract declarators have no name");
    return DeclarationName();

  case Declarator::DK_Normal:
    assert (D.getIdentifier() != 0 && "normal declarators have an identifier");
    return DeclarationName(D.getIdentifier());

  case Declarator::DK_Constructor: {
    QualType Ty = GetTypeFromParser(D.getDeclaratorIdType());
    return Context.DeclarationNames.getCXXConstructorName(
                                                Context.getCanonicalType(Ty));
  }

  case Declarator::DK_Destructor: {
    QualType Ty = GetTypeFromParser(D.getDeclaratorIdType());
    return Context.DeclarationNames.getCXXDestructorName(
                                                Context.getCanonicalType(Ty));
  }

  case Declarator::DK_Conversion: {
    // FIXME: We'd like to keep the non-canonical type for diagnostics!
    QualType Ty = GetTypeFromParser(D.getDeclaratorIdType());
    return Context.DeclarationNames.getCXXConversionFunctionName(
                                                Context.getCanonicalType(Ty));
  }

  case Declarator::DK_Operator:
    assert(D.getIdentifier() == 0 && "operator names have no identifier");
    return Context.DeclarationNames.getCXXOperatorName(
                                                D.getOverloadedOperator());
  }

  assert(false && "Unknown name kind");
  return DeclarationName();
}

/// isNearlyMatchingFunction - Determine whether the C++ functions
/// Declaration and Definition are "nearly" matching. This heuristic
/// is used to improve diagnostics in the case where an out-of-line
/// function definition doesn't match any declaration within
/// the class or namespace.
static bool isNearlyMatchingFunction(ASTContext &Context,
                                     FunctionDecl *Declaration,
                                     FunctionDecl *Definition) {
  if (Declaration->param_size() != Definition->param_size())
    return false;
  for (unsigned Idx = 0; Idx < Declaration->param_size(); ++Idx) {
    QualType DeclParamTy = Declaration->getParamDecl(Idx)->getType();
    QualType DefParamTy = Definition->getParamDecl(Idx)->getType();

    DeclParamTy = Context.getCanonicalType(DeclParamTy.getNonReferenceType());
    DefParamTy = Context.getCanonicalType(DefParamTy.getNonReferenceType());
    if (DeclParamTy.getUnqualifiedType() != DefParamTy.getUnqualifiedType())
      return false;
  }

  return true;
}

Sema::DeclPtrTy 
Sema::HandleDeclarator(Scope *S, Declarator &D, 
                       MultiTemplateParamsArg TemplateParamLists,
                       bool IsFunctionDefinition) {
  DeclarationName Name = GetNameForDeclarator(D);

  // All of these full declarators require an identifier.  If it doesn't have
  // one, the ParsedFreeStandingDeclSpec action should be used.
  if (!Name) {
    if (!D.isInvalidType())  // Reject this if we think it is valid.
      Diag(D.getDeclSpec().getSourceRange().getBegin(),
           diag::err_declarator_need_ident)
        << D.getDeclSpec().getSourceRange() << D.getSourceRange();
    return DeclPtrTy();
  }
  
  // The scope passed in may not be a decl scope.  Zip up the scope tree until
  // we find one that is.
  while ((S->getFlags() & Scope::DeclScope) == 0 ||
         (S->getFlags() & Scope::TemplateParamScope) != 0)
    S = S->getParent();
  
  // If this is an out-of-line definition of a member of a class template
  // or class template partial specialization, we may need to rebuild the
  // type specifier in the declarator. See RebuildTypeInCurrentInstantiation()
  // for more information.
  // FIXME: cope with decltype(expr) and typeof(expr) once the rebuilder can
  // handle expressions properly.
  DeclSpec &DS = const_cast<DeclSpec&>(D.getDeclSpec());
  if (D.getCXXScopeSpec().isSet() && !D.getCXXScopeSpec().isInvalid() &&
      isDependentScopeSpecifier(D.getCXXScopeSpec()) &&
      (DS.getTypeSpecType() == DeclSpec::TST_typename ||
       DS.getTypeSpecType() == DeclSpec::TST_typeofType ||
       DS.getTypeSpecType() == DeclSpec::TST_typeofExpr ||
       DS.getTypeSpecType() == DeclSpec::TST_decltype)) {
    if (DeclContext *DC = computeDeclContext(D.getCXXScopeSpec(), true)) {
      // FIXME: Preserve type source info.
      QualType T = GetTypeFromParser(DS.getTypeRep());
      EnterDeclaratorContext(S, DC);
      T = RebuildTypeInCurrentInstantiation(T, D.getIdentifierLoc(), Name);
      ExitDeclaratorContext(S);
      if (T.isNull())
        return DeclPtrTy();
      DS.UpdateTypeRep(T.getAsOpaquePtr());
    }
  }
  
  DeclContext *DC;
  NamedDecl *PrevDecl;
  NamedDecl *New;

  DeclaratorInfo *DInfo = 0;
  QualType R = GetTypeForDeclarator(D, S, &DInfo);

  // See if this is a redefinition of a variable in the same scope.
  if (D.getCXXScopeSpec().isInvalid()) {
    DC = CurContext;
    PrevDecl = 0;
    D.setInvalidType();
  } else if (!D.getCXXScopeSpec().isSet()) {
    LookupNameKind NameKind = LookupOrdinaryName;

    // If the declaration we're planning to build will be a function
    // or object with linkage, then look for another declaration with
    // linkage (C99 6.2.2p4-5 and C++ [basic.link]p6).
    if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef)
      /* Do nothing*/;
    else if (R->isFunctionType()) {
      if (CurContext->isFunctionOrMethod() ||
          D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_static)
        NameKind = LookupRedeclarationWithLinkage;
    } else if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_extern)
      NameKind = LookupRedeclarationWithLinkage;
    else if (CurContext->getLookupContext()->isTranslationUnit() &&
             D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_static)
      NameKind = LookupRedeclarationWithLinkage;

    DC = CurContext;
    PrevDecl = LookupName(S, Name, NameKind, true, 
                          NameKind == LookupRedeclarationWithLinkage,
                          D.getIdentifierLoc());
  } else { // Something like "int foo::x;"
    DC = computeDeclContext(D.getCXXScopeSpec(), true);
    // FIXME: RequireCompleteDeclContext(D.getCXXScopeSpec()); ?
    PrevDecl = LookupQualifiedName(DC, Name, LookupOrdinaryName, true);

    // C++ 7.3.1.2p2:
    // Members (including explicit specializations of templates) of a named
    // namespace can also be defined outside that namespace by explicit
    // qualification of the name being defined, provided that the entity being
    // defined was already declared in the namespace and the definition appears
    // after the point of declaration in a namespace that encloses the
    // declarations namespace.
    //
    // Note that we only check the context at this point. We don't yet
    // have enough information to make sure that PrevDecl is actually
    // the declaration we want to match. For example, given:
    //
    //   class X {
    //     void f();
    //     void f(float);
    //   };
    //
    //   void X::f(int) { } // ill-formed
    //
    // In this case, PrevDecl will point to the overload set
    // containing the two f's declared in X, but neither of them
    // matches. 

    // First check whether we named the global scope.
    if (isa<TranslationUnitDecl>(DC)) {
      Diag(D.getIdentifierLoc(), diag::err_invalid_declarator_global_scope) 
        << Name << D.getCXXScopeSpec().getRange();
    } else if (!CurContext->Encloses(DC)) {
      // The qualifying scope doesn't enclose the original declaration.
      // Emit diagnostic based on current scope.
      SourceLocation L = D.getIdentifierLoc();
      SourceRange R = D.getCXXScopeSpec().getRange();
      if (isa<FunctionDecl>(CurContext))
        Diag(L, diag::err_invalid_declarator_in_function) << Name << R;
      else 
        Diag(L, diag::err_invalid_declarator_scope)
          << Name << cast<NamedDecl>(DC) << R;
      D.setInvalidType();
    }
  }

  if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    if (!D.isInvalidType()) 
      if (DiagnoseTemplateParameterShadow(D.getIdentifierLoc(), PrevDecl))
        D.setInvalidType();
    
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  }

  // In C++, the previous declaration we find might be a tag type
  // (class or enum). In this case, the new declaration will hide the
  // tag type. Note that this does does not apply if we're declaring a
  // typedef (C++ [dcl.typedef]p4).
  if (PrevDecl && PrevDecl->getIdentifierNamespace() == Decl::IDNS_Tag &&
      D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_typedef)
    PrevDecl = 0;

  bool Redeclaration = false;
  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef) {
    if (TemplateParamLists.size()) {
      Diag(D.getIdentifierLoc(), diag::err_template_typedef);
      return DeclPtrTy();
    }
      
    New = ActOnTypedefDeclarator(S, D, DC, R, DInfo, PrevDecl, Redeclaration);
  } else if (R->isFunctionType()) {
    New = ActOnFunctionDeclarator(S, D, DC, R, DInfo, PrevDecl, 
                                  move(TemplateParamLists),
                                  IsFunctionDefinition, Redeclaration);
  } else {
    New = ActOnVariableDeclarator(S, D, DC, R, DInfo, PrevDecl, 
                                  move(TemplateParamLists),
                                  Redeclaration);
  }

  if (New == 0)
    return DeclPtrTy();
  
  // If this has an identifier and is not an invalid redeclaration,
  // add it to the scope stack.
  if (Name && !(Redeclaration && New->isInvalidDecl()))
    PushOnScopeChains(New, S);
  
  return DeclPtrTy::make(New);
}

/// TryToFixInvalidVariablyModifiedType - Helper method to turn variable array
/// types into constant array types in certain situations which would otherwise
/// be errors (for GCC compatibility).
static QualType TryToFixInvalidVariablyModifiedType(QualType T,
                                                    ASTContext &Context,
                                                    bool &SizeIsNegative) {
  // This method tries to turn a variable array into a constant
  // array even when the size isn't an ICE.  This is necessary
  // for compatibility with code that depends on gcc's buggy
  // constant expression folding, like struct {char x[(int)(char*)2];}
  SizeIsNegative = false;

  if (const PointerType* PTy = dyn_cast<PointerType>(T)) {
    QualType Pointee = PTy->getPointeeType();
    QualType FixedType =
        TryToFixInvalidVariablyModifiedType(Pointee, Context, SizeIsNegative);
    if (FixedType.isNull()) return FixedType;
    FixedType = Context.getPointerType(FixedType);
    FixedType.setCVRQualifiers(T.getCVRQualifiers());
    return FixedType;
  }

  const VariableArrayType* VLATy = dyn_cast<VariableArrayType>(T);
  if (!VLATy)
    return QualType();
  // FIXME: We should probably handle this case
  if (VLATy->getElementType()->isVariablyModifiedType())
    return QualType();
  
  Expr::EvalResult EvalResult;
  if (!VLATy->getSizeExpr() ||
      !VLATy->getSizeExpr()->Evaluate(EvalResult, Context) ||
      !EvalResult.Val.isInt())
    return QualType();

  llvm::APSInt &Res = EvalResult.Val.getInt();
  if (Res >= llvm::APSInt(Res.getBitWidth(), Res.isUnsigned())) {
    Expr* ArySizeExpr = VLATy->getSizeExpr();
    // FIXME: here we could "steal" (how?) ArySizeExpr from the VLA,
    // so as to transfer ownership to the ConstantArrayWithExpr.
    // Alternatively, we could "clone" it (how?).
    // Since we don't know how to do things above, we just use the
    // very same Expr*.
    return Context.getConstantArrayWithExprType(VLATy->getElementType(),
                                                Res, ArySizeExpr,
                                                ArrayType::Normal, 0,
                                                VLATy->getBracketsRange());
  }

  SizeIsNegative = true;
  return QualType();
}

/// \brief Register the given locally-scoped external C declaration so
/// that it can be found later for redeclarations
void 
Sema::RegisterLocallyScopedExternCDecl(NamedDecl *ND, NamedDecl *PrevDecl,
                                       Scope *S) {
  assert(ND->getLexicalDeclContext()->isFunctionOrMethod() &&
         "Decl is not a locally-scoped decl!");
  // Note that we have a locally-scoped external with this name.
  LocallyScopedExternalDecls[ND->getDeclName()] = ND;

  if (!PrevDecl)
    return;

  // If there was a previous declaration of this variable, it may be
  // in our identifier chain. Update the identifier chain with the new
  // declaration.
  if (S && IdResolver.ReplaceDecl(PrevDecl, ND)) {
    // The previous declaration was found on the identifer resolver
    // chain, so remove it from its scope.
    while (S && !S->isDeclScope(DeclPtrTy::make(PrevDecl)))
      S = S->getParent();

    if (S)
      S->RemoveDecl(DeclPtrTy::make(PrevDecl));
  }
}

/// \brief Diagnose function specifiers on a declaration of an identifier that
/// does not identify a function.
void Sema::DiagnoseFunctionSpecifiers(Declarator& D) {
  // FIXME: We should probably indicate the identifier in question to avoid
  // confusion for constructs like "inline int a(), b;"
  if (D.getDeclSpec().isInlineSpecified())
    Diag(D.getDeclSpec().getInlineSpecLoc(), 
         diag::err_inline_non_function);

  if (D.getDeclSpec().isVirtualSpecified())
    Diag(D.getDeclSpec().getVirtualSpecLoc(), 
         diag::err_virtual_non_function);

  if (D.getDeclSpec().isExplicitSpecified())
    Diag(D.getDeclSpec().getExplicitSpecLoc(), 
         diag::err_explicit_non_function);
}

NamedDecl*
Sema::ActOnTypedefDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                             QualType R,  DeclaratorInfo *DInfo,
                             Decl* PrevDecl, bool &Redeclaration) {
  // Typedef declarators cannot be qualified (C++ [dcl.meaning]p1).
  if (D.getCXXScopeSpec().isSet()) {
    Diag(D.getIdentifierLoc(), diag::err_qualified_typedef_declarator)
      << D.getCXXScopeSpec().getRange();
    D.setInvalidType();
    // Pretend we didn't see the scope specifier.
    DC = 0;
  }

  if (getLangOptions().CPlusPlus) {
    // Check that there are no default arguments (C++ only).
    CheckExtraCXXDefaultArguments(D);
  }

  DiagnoseFunctionSpecifiers(D);

  if (D.getDeclSpec().isThreadSpecified())
    Diag(D.getDeclSpec().getThreadSpecLoc(), diag::err_invalid_thread);

  TypedefDecl *NewTD = ParseTypedefDecl(S, D, R);
  if (!NewTD) return 0;
  
  if (D.isInvalidType())
    NewTD->setInvalidDecl();

  // Handle attributes prior to checking for duplicates in MergeVarDecl
  ProcessDeclAttributes(S, NewTD, D);
  // Merge the decl with the existing one if appropriate. If the decl is
  // in an outer scope, it isn't the same thing.
  if (PrevDecl && isDeclInScope(PrevDecl, DC, S)) {
    Redeclaration = true;
    MergeTypeDefDecl(NewTD, PrevDecl);
  }

  // C99 6.7.7p2: If a typedef name specifies a variably modified type
  // then it shall have block scope.
  QualType T = NewTD->getUnderlyingType();
  if (T->isVariablyModifiedType()) {
    CurFunctionNeedsScopeChecking = true;
  
    if (S->getFnParent() == 0) {
      bool SizeIsNegative;
      QualType FixedTy =
          TryToFixInvalidVariablyModifiedType(T, Context, SizeIsNegative);
      if (!FixedTy.isNull()) {
        Diag(D.getIdentifierLoc(), diag::warn_illegal_constant_array_size);
        NewTD->setUnderlyingType(FixedTy);
      } else {
        if (SizeIsNegative)
          Diag(D.getIdentifierLoc(), diag::err_typecheck_negative_array_size);
        else if (T->isVariableArrayType())
          Diag(D.getIdentifierLoc(), diag::err_vla_decl_in_file_scope);
        else
          Diag(D.getIdentifierLoc(), diag::err_vm_decl_in_file_scope);
        NewTD->setInvalidDecl();
      }
    }
  }

  // If this is the C FILE type, notify the AST context.
  if (IdentifierInfo *II = NewTD->getIdentifier())
    if (!NewTD->isInvalidDecl() &&
        NewTD->getDeclContext()->getLookupContext()->isTranslationUnit()) {
      if (II->isStr("FILE"))
        Context.setFILEDecl(NewTD);
      else if (II->isStr("jmp_buf"))
        Context.setjmp_bufDecl(NewTD);
      else if (II->isStr("sigjmp_buf"))
        Context.setsigjmp_bufDecl(NewTD);
    }

  return NewTD;
}

/// \brief Determines whether the given declaration is an out-of-scope
/// previous declaration.
///
/// This routine should be invoked when name lookup has found a
/// previous declaration (PrevDecl) that is not in the scope where a
/// new declaration by the same name is being introduced. If the new
/// declaration occurs in a local scope, previous declarations with
/// linkage may still be considered previous declarations (C99
/// 6.2.2p4-5, C++ [basic.link]p6).
///
/// \param PrevDecl the previous declaration found by name
/// lookup
/// 
/// \param DC the context in which the new declaration is being
/// declared.
///
/// \returns true if PrevDecl is an out-of-scope previous declaration
/// for a new delcaration with the same name.
static bool 
isOutOfScopePreviousDeclaration(NamedDecl *PrevDecl, DeclContext *DC,
                                ASTContext &Context) {
  if (!PrevDecl)
    return 0;

  // FIXME: PrevDecl could be an OverloadedFunctionDecl, in which
  // case we need to check each of the overloaded functions.
  if (!PrevDecl->hasLinkage())
    return false;

  if (Context.getLangOptions().CPlusPlus) {
    // C++ [basic.link]p6:
    //   If there is a visible declaration of an entity with linkage
    //   having the same name and type, ignoring entities declared
    //   outside the innermost enclosing namespace scope, the block
    //   scope declaration declares that same entity and receives the
    //   linkage of the previous declaration.
    DeclContext *OuterContext = DC->getLookupContext();
    if (!OuterContext->isFunctionOrMethod())
      // This rule only applies to block-scope declarations.
      return false;
    else {
      DeclContext *PrevOuterContext = PrevDecl->getDeclContext();
      if (PrevOuterContext->isRecord())
        // We found a member function: ignore it.
        return false;
      else {
        // Find the innermost enclosing namespace for the new and
        // previous declarations.
        while (!OuterContext->isFileContext())
          OuterContext = OuterContext->getParent();
        while (!PrevOuterContext->isFileContext())
          PrevOuterContext = PrevOuterContext->getParent();
          
        // The previous declaration is in a different namespace, so it
        // isn't the same function.
        if (OuterContext->getPrimaryContext() != 
            PrevOuterContext->getPrimaryContext())
          return false;
      }
    }
  }

  return true;
}

NamedDecl*
Sema::ActOnVariableDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                              QualType R, DeclaratorInfo *DInfo,
                              NamedDecl* PrevDecl,
                              MultiTemplateParamsArg TemplateParamLists,
                              bool &Redeclaration) {
  DeclarationName Name = GetNameForDeclarator(D);

  // Check that there are no default arguments (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);

  VarDecl *NewVD;
  VarDecl::StorageClass SC;
  switch (D.getDeclSpec().getStorageClassSpec()) {
  default: assert(0 && "Unknown storage class!");
  case DeclSpec::SCS_unspecified:    SC = VarDecl::None; break;
  case DeclSpec::SCS_extern:         SC = VarDecl::Extern; break;
  case DeclSpec::SCS_static:         SC = VarDecl::Static; break;
  case DeclSpec::SCS_auto:           SC = VarDecl::Auto; break;
  case DeclSpec::SCS_register:       SC = VarDecl::Register; break;
  case DeclSpec::SCS_private_extern: SC = VarDecl::PrivateExtern; break;
  case DeclSpec::SCS_mutable:
    // mutable can only appear on non-static class members, so it's always
    // an error here
    Diag(D.getIdentifierLoc(), diag::err_mutable_nonmember);
    D.setInvalidType();
    SC = VarDecl::None;
    break;
  }

  IdentifierInfo *II = Name.getAsIdentifierInfo();
  if (!II) {
    Diag(D.getIdentifierLoc(), diag::err_bad_variable_name)
      << Name.getAsString();
    return 0;
  }

  DiagnoseFunctionSpecifiers(D);

  if (!DC->isRecord() && S->getFnParent() == 0) {
    // C99 6.9p2: The storage-class specifiers auto and register shall not
    // appear in the declaration specifiers in an external declaration.
    if (SC == VarDecl::Auto || SC == VarDecl::Register) {
      
      // If this is a register variable with an asm label specified, then this
      // is a GNU extension.
      if (SC == VarDecl::Register && D.getAsmLabel())
        Diag(D.getIdentifierLoc(), diag::err_unsupported_global_register);
      else
        Diag(D.getIdentifierLoc(), diag::err_typecheck_sclass_fscope);
      D.setInvalidType();
    }
  }
  if (DC->isRecord() && !CurContext->isRecord()) {
    // This is an out-of-line definition of a static data member.
    if (SC == VarDecl::Static) {
      Diag(D.getDeclSpec().getStorageClassSpecLoc(), 
           diag::err_static_out_of_line)
        << CodeModificationHint::CreateRemoval(
                       SourceRange(D.getDeclSpec().getStorageClassSpecLoc()));
    } else if (SC == VarDecl::None)
      SC = VarDecl::Static;
  }
  if (SC == VarDecl::Static) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC)) {
      if (RD->isLocalClass())
        Diag(D.getIdentifierLoc(), 
             diag::err_static_data_member_not_allowed_in_local_class)
          << Name << RD->getDeclName();
    }
  }
        
  // Check that we can declare a template here.
  if (TemplateParamLists.size() && 
      CheckTemplateDeclScope(S, TemplateParamLists))
    return 0;
  
  // Match up the template parameter lists with the scope specifier, then
  // determine whether we have a template or a template specialization.
  if (TemplateParameterList *TemplateParams
      = MatchTemplateParametersToScopeSpecifier(
                                  D.getDeclSpec().getSourceRange().getBegin(),
                                                D.getCXXScopeSpec(),
                        (TemplateParameterList**)TemplateParamLists.get(),
                                                 TemplateParamLists.size())) {
    if (TemplateParams->size() > 0) {
      // There is no such thing as a variable template.
      Diag(D.getIdentifierLoc(), diag::err_template_variable)
        << II
        << SourceRange(TemplateParams->getTemplateLoc(),
                       TemplateParams->getRAngleLoc());
      return 0;
    } else {
      // There is an extraneous 'template<>' for this variable. Complain
      // about it, but allow the declaration of the variable.
      Diag(TemplateParams->getTemplateLoc(), 
           diag::err_template_variable_noparams)
        << II
        << SourceRange(TemplateParams->getTemplateLoc(),
                       TemplateParams->getRAngleLoc());          
    }
  }        
  
  NewVD = VarDecl::Create(Context, DC, D.getIdentifierLoc(), 
                          II, R, DInfo, SC);

  if (D.isInvalidType())
    NewVD->setInvalidDecl();
  
  if (D.getDeclSpec().isThreadSpecified()) {
    if (NewVD->hasLocalStorage())
      Diag(D.getDeclSpec().getThreadSpecLoc(), diag::err_thread_non_global);
    else if (!Context.Target.isTLSSupported())
      Diag(D.getDeclSpec().getThreadSpecLoc(), diag::err_thread_unsupported);
    else
      NewVD->setThreadSpecified(true);
  }

  // Set the lexical context. If the declarator has a C++ scope specifier, the
  // lexical context will be different from the semantic context.
  NewVD->setLexicalDeclContext(CurContext);

  // Handle attributes prior to checking for duplicates in MergeVarDecl
  ProcessDeclAttributes(S, NewVD, D);

  // Handle GNU asm-label extension (encoded as an attribute).
  if (Expr *E = (Expr*) D.getAsmLabel()) {
    // The parser guarantees this is a string.
    StringLiteral *SE = cast<StringLiteral>(E);  
    NewVD->addAttr(::new (Context) AsmLabelAttr(std::string(SE->getStrData(),
                                                        SE->getByteLength())));
  }

  // If name lookup finds a previous declaration that is not in the
  // same scope as the new declaration, this may still be an
  // acceptable redeclaration.
  if (PrevDecl && !isDeclInScope(PrevDecl, DC, S) &&
      !(NewVD->hasLinkage() &&
        isOutOfScopePreviousDeclaration(PrevDecl, DC, Context)))
    PrevDecl = 0;     

  // Merge the decl with the existing one if appropriate.
  if (PrevDecl) {
    if (isa<FieldDecl>(PrevDecl) && D.getCXXScopeSpec().isSet()) {
      // The user tried to define a non-static data member
      // out-of-line (C++ [dcl.meaning]p1).
      Diag(NewVD->getLocation(), diag::err_nonstatic_member_out_of_line)
        << D.getCXXScopeSpec().getRange();
      PrevDecl = 0;
      NewVD->setInvalidDecl();
    }
  } else if (D.getCXXScopeSpec().isSet()) {
    // No previous declaration in the qualifying scope.
    Diag(D.getIdentifierLoc(), diag::err_typecheck_no_member)
      << Name << D.getCXXScopeSpec().getRange();
    NewVD->setInvalidDecl();
  }

  CheckVariableDeclaration(NewVD, PrevDecl, Redeclaration);

  // attributes declared post-definition are currently ignored
  if (PrevDecl) {
    const VarDecl *Def = 0, *PrevVD = dyn_cast<VarDecl>(PrevDecl);
    if (PrevVD->getDefinition(Def) && D.hasAttributes()) {
      Diag(NewVD->getLocation(), diag::warn_attribute_precede_definition);
      Diag(Def->getLocation(), diag::note_previous_definition);
    }
  }

  // If this is a locally-scoped extern C variable, update the map of
  // such variables.
  if (CurContext->isFunctionOrMethod() && NewVD->isExternC(Context) &&
      !NewVD->isInvalidDecl())
    RegisterLocallyScopedExternCDecl(NewVD, PrevDecl, S);

  return NewVD;
}

/// \brief Perform semantic checking on a newly-created variable
/// declaration.
///
/// This routine performs all of the type-checking required for a
/// variable declaration once it has been built. It is used both to
/// check variables after they have been parsed and their declarators
/// have been translated into a declaration, and to check variables
/// that have been instantiated from a template.
///
/// Sets NewVD->isInvalidDecl() if an error was encountered.
void Sema::CheckVariableDeclaration(VarDecl *NewVD, NamedDecl *PrevDecl,
                                    bool &Redeclaration) {
  // If the decl is already known invalid, don't check it.
  if (NewVD->isInvalidDecl())
    return;
  
  QualType T = NewVD->getType();

  if (T->isObjCInterfaceType()) {
    Diag(NewVD->getLocation(), diag::err_statically_allocated_object);
    return NewVD->setInvalidDecl();
  }
  
  // The variable can not have an abstract class type.
  if (RequireNonAbstractType(NewVD->getLocation(), T,
                             diag::err_abstract_type_in_decl, 
                             AbstractVariableType))
    return NewVD->setInvalidDecl();

  // Emit an error if an address space was applied to decl with local storage.
  // This includes arrays of objects with address space qualifiers, but not
  // automatic variables that point to other address spaces.
  // ISO/IEC TR 18037 S5.1.2
  if (NewVD->hasLocalStorage() && (T.getAddressSpace() != 0)) {
    Diag(NewVD->getLocation(), diag::err_as_qualified_auto_decl);
    return NewVD->setInvalidDecl();
  }

  if (NewVD->hasLocalStorage() && T.isObjCGCWeak()
      && !NewVD->hasAttr<BlocksAttr>())
    Diag(NewVD->getLocation(), diag::warn_attribute_weak_on_local);

  bool isVM = T->isVariablyModifiedType();
  if (isVM || NewVD->hasAttr<CleanupAttr>() ||
      NewVD->hasAttr<BlocksAttr>())
    CurFunctionNeedsScopeChecking = true;
  
  if ((isVM && NewVD->hasLinkage()) ||
      (T->isVariableArrayType() && NewVD->hasGlobalStorage())) {
    bool SizeIsNegative;
    QualType FixedTy =
        TryToFixInvalidVariablyModifiedType(T, Context, SizeIsNegative);
    
    if (FixedTy.isNull() && T->isVariableArrayType()) {
      const VariableArrayType *VAT = Context.getAsVariableArrayType(T);
      // FIXME: This won't give the correct result for 
      // int a[10][n];      
      SourceRange SizeRange = VAT->getSizeExpr()->getSourceRange();
      
      if (NewVD->isFileVarDecl())
        Diag(NewVD->getLocation(), diag::err_vla_decl_in_file_scope)
        << SizeRange;
      else if (NewVD->getStorageClass() == VarDecl::Static)
        Diag(NewVD->getLocation(), diag::err_vla_decl_has_static_storage)
        << SizeRange;
      else
        Diag(NewVD->getLocation(), diag::err_vla_decl_has_extern_linkage)
        << SizeRange;
      return NewVD->setInvalidDecl();
    } 
    
    if (FixedTy.isNull()) {
      if (NewVD->isFileVarDecl())
        Diag(NewVD->getLocation(), diag::err_vm_decl_in_file_scope);
      else
        Diag(NewVD->getLocation(), diag::err_vm_decl_has_extern_linkage);
      return NewVD->setInvalidDecl();
    }
    
    Diag(NewVD->getLocation(), diag::warn_illegal_constant_array_size);
    NewVD->setType(FixedTy);
  }

  if (!PrevDecl && NewVD->isExternC(Context)) {
    // Since we did not find anything by this name and we're declaring
    // an extern "C" variable, look for a non-visible extern "C"
    // declaration with the same name.
    llvm::DenseMap<DeclarationName, NamedDecl *>::iterator Pos
      = LocallyScopedExternalDecls.find(NewVD->getDeclName());
    if (Pos != LocallyScopedExternalDecls.end())
      PrevDecl = Pos->second;
  }

  if (T->isVoidType() && !NewVD->hasExternalStorage()) {
    Diag(NewVD->getLocation(), diag::err_typecheck_decl_incomplete_type)
      << T;
    return NewVD->setInvalidDecl();
  }

  if (!NewVD->hasLocalStorage() && NewVD->hasAttr<BlocksAttr>()) {
    Diag(NewVD->getLocation(), diag::err_block_on_nonlocal);
    return NewVD->setInvalidDecl();
  }
    
  if (isVM && NewVD->hasAttr<BlocksAttr>()) {
    Diag(NewVD->getLocation(), diag::err_block_on_vm);
    return NewVD->setInvalidDecl();
  }

  if (PrevDecl) {
    Redeclaration = true;
    MergeVarDecl(NewVD, PrevDecl);
  }
}

NamedDecl* 
Sema::ActOnFunctionDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                              QualType R, DeclaratorInfo *DInfo,
                              NamedDecl* PrevDecl,
                              MultiTemplateParamsArg TemplateParamLists,
                              bool IsFunctionDefinition, bool &Redeclaration) {
  assert(R.getTypePtr()->isFunctionType());

  DeclarationName Name = GetNameForDeclarator(D);
  FunctionDecl::StorageClass SC = FunctionDecl::None;
  switch (D.getDeclSpec().getStorageClassSpec()) {
  default: assert(0 && "Unknown storage class!");
  case DeclSpec::SCS_auto:
  case DeclSpec::SCS_register:
  case DeclSpec::SCS_mutable:
    Diag(D.getDeclSpec().getStorageClassSpecLoc(), 
         diag::err_typecheck_sclass_func);
    D.setInvalidType();
    break;
  case DeclSpec::SCS_unspecified: SC = FunctionDecl::None; break;
  case DeclSpec::SCS_extern:      SC = FunctionDecl::Extern; break;
  case DeclSpec::SCS_static: {
    if (CurContext->getLookupContext()->isFunctionOrMethod()) {
      // C99 6.7.1p5:
      //   The declaration of an identifier for a function that has
      //   block scope shall have no explicit storage-class specifier
      //   other than extern
      // See also (C++ [dcl.stc]p4).
      Diag(D.getDeclSpec().getStorageClassSpecLoc(), 
           diag::err_static_block_func);
      SC = FunctionDecl::None;
    } else
      SC = FunctionDecl::Static; 
    break;
  }
  case DeclSpec::SCS_private_extern: SC = FunctionDecl::PrivateExtern;break;
  }

  if (D.getDeclSpec().isThreadSpecified())
    Diag(D.getDeclSpec().getThreadSpecLoc(), diag::err_invalid_thread);

  bool isFriend = D.getDeclSpec().isFriendSpecified();
  bool isInline = D.getDeclSpec().isInlineSpecified();
  bool isVirtual = D.getDeclSpec().isVirtualSpecified();
  bool isExplicit = D.getDeclSpec().isExplicitSpecified();

  // Check that the return type is not an abstract class type.
  // For record types, this is done by the AbstractClassUsageDiagnoser once
  // the class has been completely parsed. 
  if (!DC->isRecord() &&
      RequireNonAbstractType(D.getIdentifierLoc(), 
                             R->getAsFunctionType()->getResultType(),
                             diag::err_abstract_type_in_decl, 
                             AbstractReturnType))
    D.setInvalidType();
  
  // Do not allow returning a objc interface by-value.
  if (R->getAsFunctionType()->getResultType()->isObjCInterfaceType()) {
    Diag(D.getIdentifierLoc(),
         diag::err_object_cannot_be_passed_returned_by_value) << 0
      << R->getAsFunctionType()->getResultType();
    D.setInvalidType();
  }

  // Check that we can declare a template here.
  if (TemplateParamLists.size() && 
      CheckTemplateDeclScope(S, TemplateParamLists))
    return 0;
  
  bool isVirtualOkay = false;
  FunctionDecl *NewFD;
  if (isFriend) {
    // DC is the namespace in which the function is being declared.
    assert(DC->isFileContext() || D.getCXXScopeSpec().isSet());

    // C++ [class.friend]p5
    //   A function can be defined in a friend declaration of a
    //   class . . . . Such a function is implicitly inline.
    isInline |= IsFunctionDefinition;

    NewFD = FriendFunctionDecl::Create(Context, DC,
                                       D.getIdentifierLoc(), Name, R, DInfo,
                                       isInline,
                                       D.getDeclSpec().getFriendSpecLoc());
    
  } else if (D.getKind() == Declarator::DK_Constructor) {
    // This is a C++ constructor declaration.
    assert(DC->isRecord() &&
           "Constructors can only be declared in a member context");

    R = CheckConstructorDeclarator(D, R, SC);

    // Create the new declaration
    NewFD = CXXConstructorDecl::Create(Context, 
                                       cast<CXXRecordDecl>(DC),
                                       D.getIdentifierLoc(), Name, R, DInfo,
                                       isExplicit, isInline,
                                       /*isImplicitlyDeclared=*/false);
  } else if (D.getKind() == Declarator::DK_Destructor) {
    // This is a C++ destructor declaration.
    if (DC->isRecord()) {
      R = CheckDestructorDeclarator(D, SC);
      
      NewFD = CXXDestructorDecl::Create(Context,
                                        cast<CXXRecordDecl>(DC),
                                        D.getIdentifierLoc(), Name, R, 
                                        isInline,
                                        /*isImplicitlyDeclared=*/false);

      isVirtualOkay = true;
    } else {
      Diag(D.getIdentifierLoc(), diag::err_destructor_not_member);

      // Create a FunctionDecl to satisfy the function definition parsing
      // code path.
      NewFD = FunctionDecl::Create(Context, DC, D.getIdentifierLoc(),
                                   Name, R, DInfo, SC, isInline, 
                                   /*hasPrototype=*/true);
      D.setInvalidType();
    }
  } else if (D.getKind() == Declarator::DK_Conversion) {
    if (!DC->isRecord()) {
      Diag(D.getIdentifierLoc(),
           diag::err_conv_function_not_member);
      return 0;
    }
    
    CheckConversionDeclarator(D, R, SC);
    NewFD = CXXConversionDecl::Create(Context, cast<CXXRecordDecl>(DC),
                                      D.getIdentifierLoc(), Name, R, DInfo,
                                      isInline, isExplicit);
      
    isVirtualOkay = true;
  } else if (DC->isRecord()) {
    // If the of the function is the same as the name of the record, then this
    // must be an invalid constructor that has a return type. 
    // (The parser checks for a return type and makes the declarator a 
    // constructor if it has no return type).
    // must have an invalid constructor that has a return type 
    if (Name.getAsIdentifierInfo() == cast<CXXRecordDecl>(DC)->getIdentifier()){
      Diag(D.getIdentifierLoc(), diag::err_constructor_return_type)
        << SourceRange(D.getDeclSpec().getTypeSpecTypeLoc())
        << SourceRange(D.getIdentifierLoc());
      return 0;
    }
    
    // This is a C++ method declaration.
    NewFD = CXXMethodDecl::Create(Context, cast<CXXRecordDecl>(DC),
                                  D.getIdentifierLoc(), Name, R, DInfo,
                                  (SC == FunctionDecl::Static), isInline);

    isVirtualOkay = (SC != FunctionDecl::Static);
  } else {
    // Determine whether the function was written with a
    // prototype. This true when:
    //   - we're in C++ (where every function has a prototype),
    //   - there is a prototype in the declarator, or
    //   - the type R of the function is some kind of typedef or other reference
    //     to a type name (which eventually refers to a function type).
    bool HasPrototype = 
       getLangOptions().CPlusPlus ||
       (D.getNumTypeObjects() && D.getTypeObject(0).Fun.hasPrototype) ||
       (!isa<FunctionType>(R.getTypePtr()) && R->isFunctionProtoType());
    
    NewFD = FunctionDecl::Create(Context, DC,
                                 D.getIdentifierLoc(),
                                 Name, R, DInfo, SC, isInline, HasPrototype);
  }

  if (D.isInvalidType())
    NewFD->setInvalidDecl();
  
  // Set the lexical context. If the declarator has a C++
  // scope specifier, the lexical context will be different
  // from the semantic context.
  NewFD->setLexicalDeclContext(CurContext);

  // Match up the template parameter lists with the scope specifier, then
  // determine whether we have a template or a template specialization.
  FunctionTemplateDecl *FunctionTemplate = 0;
  if (TemplateParameterList *TemplateParams
        = MatchTemplateParametersToScopeSpecifier(
                                  D.getDeclSpec().getSourceRange().getBegin(),
                                  D.getCXXScopeSpec(),
                           (TemplateParameterList**)TemplateParamLists.get(),
                                                  TemplateParamLists.size())) {
    if (TemplateParams->size() > 0) {
      // This is a function template
      FunctionTemplate = FunctionTemplateDecl::Create(Context, CurContext,
                                                      NewFD->getLocation(),
                                                      Name, TemplateParams,
                                                      NewFD);
      NewFD->setDescribedFunctionTemplate(FunctionTemplate);
    } else {
      // FIXME: Handle function template specializations
    }
          
    // FIXME: Free this memory properly.
    TemplateParamLists.release();
  }        
  
  // C++ [dcl.fct.spec]p5:
  //   The virtual specifier shall only be used in declarations of
  //   nonstatic class member functions that appear within a
  //   member-specification of a class declaration; see 10.3.
  //
  if (isVirtual && !NewFD->isInvalidDecl()) {
    if (!isVirtualOkay) {
       Diag(D.getDeclSpec().getVirtualSpecLoc(), 
           diag::err_virtual_non_function);
    } else if (!CurContext->isRecord()) {
      // 'virtual' was specified outside of the class.
      Diag(D.getDeclSpec().getVirtualSpecLoc(), diag::err_virtual_out_of_class)
        << CodeModificationHint::CreateRemoval(
                             SourceRange(D.getDeclSpec().getVirtualSpecLoc()));
    } else {
      // Okay: Add virtual to the method.
      cast<CXXMethodDecl>(NewFD)->setVirtualAsWritten(true);
      CXXRecordDecl *CurClass = cast<CXXRecordDecl>(DC);
      CurClass->setAggregate(false);
      CurClass->setPOD(false);
      CurClass->setEmpty(false);
      CurClass->setPolymorphic(true);
      CurClass->setHasTrivialConstructor(false);
      CurClass->setHasTrivialCopyConstructor(false);
      CurClass->setHasTrivialCopyAssignment(false);
    }
  }

  if (CXXMethodDecl *NewMD = dyn_cast<CXXMethodDecl>(NewFD)) {
    // Look for virtual methods in base classes that this method might override.

    BasePaths Paths;
    if (LookupInBases(cast<CXXRecordDecl>(DC), 
                      MemberLookupCriteria(NewMD), Paths)) {
      for (BasePaths::decl_iterator I = Paths.found_decls_begin(), 
           E = Paths.found_decls_end(); I != E; ++I) {
        if (CXXMethodDecl *OldMD = dyn_cast<CXXMethodDecl>(*I)) {
          if (!CheckOverridingFunctionReturnType(NewMD, OldMD) &&
              !CheckOverridingFunctionExceptionSpec(NewMD, OldMD))
            NewMD->addOverriddenMethod(OldMD);
        }
      }
    }
  }
  
  if (SC == FunctionDecl::Static && isa<CXXMethodDecl>(NewFD) && 
      !CurContext->isRecord()) {
    // C++ [class.static]p1:
    //   A data or function member of a class may be declared static
    //   in a class definition, in which case it is a static member of
    //   the class.

    // Complain about the 'static' specifier if it's on an out-of-line
    // member function definition.
    Diag(D.getDeclSpec().getStorageClassSpecLoc(), 
         diag::err_static_out_of_line)
      << CodeModificationHint::CreateRemoval(
                      SourceRange(D.getDeclSpec().getStorageClassSpecLoc()));
  }

  // Handle GNU asm-label extension (encoded as an attribute).
  if (Expr *E = (Expr*) D.getAsmLabel()) {
    // The parser guarantees this is a string.
    StringLiteral *SE = cast<StringLiteral>(E);  
    NewFD->addAttr(::new (Context) AsmLabelAttr(std::string(SE->getStrData(),
                                                        SE->getByteLength())));
  }

  // Copy the parameter declarations from the declarator D to the function
  // declaration NewFD, if they are available.  First scavenge them into Params.
  llvm::SmallVector<ParmVarDecl*, 16> Params;
  if (D.getNumTypeObjects() > 0) {
    DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;

    // Check for C99 6.7.5.3p10 - foo(void) is a non-varargs
    // function that takes no arguments, not a function that takes a
    // single void argument.
    // We let through "const void" here because Sema::GetTypeForDeclarator
    // already checks for that case.
    if (FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
        FTI.ArgInfo[0].Param &&
        FTI.ArgInfo[0].Param.getAs<ParmVarDecl>()->getType()->isVoidType()) {
      // Empty arg list, don't push any params.
      ParmVarDecl *Param = FTI.ArgInfo[0].Param.getAs<ParmVarDecl>();

      // In C++, the empty parameter-type-list must be spelled "void"; a
      // typedef of void is not permitted.
      if (getLangOptions().CPlusPlus &&
          Param->getType().getUnqualifiedType() != Context.VoidTy)
        Diag(Param->getLocation(), diag::err_param_typedef_of_void);
      // FIXME: Leaks decl?
    } else if (FTI.NumArgs > 0 && FTI.ArgInfo[0].Param != 0) {
      for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
        ParmVarDecl *Param = FTI.ArgInfo[i].Param.getAs<ParmVarDecl>();
        assert(Param->getDeclContext() != NewFD && "Was set before ?");
        Param->setDeclContext(NewFD);
        Params.push_back(Param);
      }
    }
  
  } else if (const FunctionProtoType *FT = R->getAsFunctionProtoType()) {
    // When we're declaring a function with a typedef, typeof, etc as in the
    // following example, we'll need to synthesize (unnamed)
    // parameters for use in the declaration.
    //
    // @code
    // typedef void fn(int);
    // fn f;
    // @endcode
    
    // Synthesize a parameter for each argument type.
    for (FunctionProtoType::arg_type_iterator AI = FT->arg_type_begin(),
         AE = FT->arg_type_end(); AI != AE; ++AI) {
      ParmVarDecl *Param = ParmVarDecl::Create(Context, DC,
                                               SourceLocation(), 0,
                                               *AI, /*DInfo=*/0,
                                               VarDecl::None, 0);
      Param->setImplicit();
      Params.push_back(Param);
    }
  } else {
    assert(R->isFunctionNoProtoType() && NewFD->getNumParams() == 0 &&
           "Should not need args for typedef of non-prototype fn");
  }
  // Finally, we know we have the right number of parameters, install them.
  NewFD->setParams(Context, Params.data(), Params.size());
    
  // If name lookup finds a previous declaration that is not in the
  // same scope as the new declaration, this may still be an
  // acceptable redeclaration.
  if (PrevDecl && !isDeclInScope(PrevDecl, DC, S) &&
      !(NewFD->hasLinkage() &&
        isOutOfScopePreviousDeclaration(PrevDecl, DC, Context)))
    PrevDecl = 0;

  // Perform semantic checking on the function declaration.
  bool OverloadableAttrRequired = false; // FIXME: HACK!
  CheckFunctionDeclaration(NewFD, PrevDecl, Redeclaration,
                           /*FIXME:*/OverloadableAttrRequired);

  if (D.getCXXScopeSpec().isSet() && !NewFD->isInvalidDecl()) {
    // An out-of-line member function declaration must also be a
    // definition (C++ [dcl.meaning]p1).
    if (!IsFunctionDefinition && !isFriend) {
      Diag(NewFD->getLocation(), diag::err_out_of_line_declaration)
        << D.getCXXScopeSpec().getRange();
      NewFD->setInvalidDecl();
    } else if (!Redeclaration && (!PrevDecl || !isa<UsingDecl>(PrevDecl))) {
      // The user tried to provide an out-of-line definition for a
      // function that is a member of a class or namespace, but there
      // was no such member function declared (C++ [class.mfct]p2, 
      // C++ [namespace.memdef]p2). For example:
      // 
      // class X {
      //   void f() const;
      // }; 
      //
      // void X::f() { } // ill-formed
      //
      // Complain about this problem, and attempt to suggest close
      // matches (e.g., those that differ only in cv-qualifiers and
      // whether the parameter types are references).
      Diag(D.getIdentifierLoc(), diag::err_member_def_does_not_match)
        << cast<NamedDecl>(DC) << D.getCXXScopeSpec().getRange();
      NewFD->setInvalidDecl();
      
      LookupResult Prev = LookupQualifiedName(DC, Name, LookupOrdinaryName, 
                                              true);
      assert(!Prev.isAmbiguous() && 
             "Cannot have an ambiguity in previous-declaration lookup");
      for (LookupResult::iterator Func = Prev.begin(), FuncEnd = Prev.end();
           Func != FuncEnd; ++Func) {
        if (isa<FunctionDecl>(*Func) &&
            isNearlyMatchingFunction(Context, cast<FunctionDecl>(*Func), NewFD))
          Diag((*Func)->getLocation(), diag::note_member_def_close_match);
      }
      
      PrevDecl = 0;
    }
  }

  // Handle attributes. We need to have merged decls when handling attributes
  // (for example to check for conflicts, etc).
  // FIXME: This needs to happen before we merge declarations. Then,
  // let attribute merging cope with attribute conflicts.
  ProcessDeclAttributes(S, NewFD, D);

  // attributes declared post-definition are currently ignored
  if (PrevDecl) {
    const FunctionDecl *Def, *PrevFD = dyn_cast<FunctionDecl>(PrevDecl);
    if (PrevFD && PrevFD->getBody(Def) && D.hasAttributes()) {
      Diag(NewFD->getLocation(), diag::warn_attribute_precede_definition);
      Diag(Def->getLocation(), diag::note_previous_definition);
    }
  }

  AddKnownFunctionAttributes(NewFD);

  if (OverloadableAttrRequired && !NewFD->getAttr<OverloadableAttr>()) {
    // If a function name is overloadable in C, then every function
    // with that name must be marked "overloadable".
    Diag(NewFD->getLocation(), diag::err_attribute_overloadable_missing)
      << Redeclaration << NewFD;
    if (PrevDecl)
      Diag(PrevDecl->getLocation(), 
           diag::note_attribute_overloadable_prev_overload);
    NewFD->addAttr(::new (Context) OverloadableAttr());
  }

  // If this is a locally-scoped extern C function, update the
  // map of such names.
  if (CurContext->isFunctionOrMethod() && NewFD->isExternC(Context)
      && !NewFD->isInvalidDecl())
    RegisterLocallyScopedExternCDecl(NewFD, PrevDecl, S);

  // Set this FunctionDecl's range up to the right paren.
  NewFD->setLocEnd(D.getSourceRange().getEnd());

  if (FunctionTemplate && NewFD->isInvalidDecl())
    FunctionTemplate->setInvalidDecl();
  
  if (FunctionTemplate)
    return FunctionTemplate;
  
  return NewFD;
}

/// \brief Perform semantic checking of a new function declaration.
///
/// Performs semantic analysis of the new function declaration
/// NewFD. This routine performs all semantic checking that does not
/// require the actual declarator involved in the declaration, and is
/// used both for the declaration of functions as they are parsed
/// (called via ActOnDeclarator) and for the declaration of functions
/// that have been instantiated via C++ template instantiation (called
/// via InstantiateDecl).
///
/// This sets NewFD->isInvalidDecl() to true if there was an error.
void Sema::CheckFunctionDeclaration(FunctionDecl *NewFD, NamedDecl *&PrevDecl,
                                    bool &Redeclaration,
                                    bool &OverloadableAttrRequired) {
  // If NewFD is already known erroneous, don't do any of this checking.
  if (NewFD->isInvalidDecl())
    return;

  if (NewFD->getResultType()->isVariablyModifiedType()) {
    // Functions returning a variably modified type violate C99 6.7.5.2p2
    // because all functions have linkage.
    Diag(NewFD->getLocation(), diag::err_vm_func_decl);
    return NewFD->setInvalidDecl();
  }

  if (NewFD->isMain(Context)) CheckMain(NewFD);

  // Semantic checking for this function declaration (in isolation).
  if (getLangOptions().CPlusPlus) {
    // C++-specific checks.
    if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(NewFD)) {
      CheckConstructor(Constructor);
    } else if (isa<CXXDestructorDecl>(NewFD)) {
      CXXRecordDecl *Record = cast<CXXRecordDecl>(NewFD->getParent());
      QualType ClassType = Context.getTypeDeclType(Record);
      if (!ClassType->isDependentType()) {
        DeclarationName Name 
          = Context.DeclarationNames.getCXXDestructorName(
                                        Context.getCanonicalType(ClassType));
        if (NewFD->getDeclName() != Name) {
          Diag(NewFD->getLocation(), diag::err_destructor_name);
          return NewFD->setInvalidDecl();  
        }
      }
      Record->setUserDeclaredDestructor(true);
      // C++ [class]p4: A POD-struct is an aggregate class that has [...] no
      // user-defined destructor.
      Record->setPOD(false);
      
      // C++ [class.dtor]p3: A destructor is trivial if it is an implicitly-
      // declared destructor.
      // FIXME: C++0x: don't do this for "= default" destructors
      Record->setHasTrivialDestructor(false);
    } else if (CXXConversionDecl *Conversion 
               = dyn_cast<CXXConversionDecl>(NewFD))
      ActOnConversionDeclarator(Conversion);
    
    // Extra checking for C++ overloaded operators (C++ [over.oper]).
    if (NewFD->isOverloadedOperator() &&
        CheckOverloadedOperatorDeclaration(NewFD))
      return NewFD->setInvalidDecl();
  }

  // C99 6.7.4p6:
  //   [... ] For a function with external linkage, the following
  //   restrictions apply: [...] If all of the file scope declarations
  //   for a function in a translation unit include the inline
  //   function specifier without extern, then the definition in that
  //   translation unit is an inline definition. An inline definition
  //   does not provide an external definition for the function, and
  //   does not forbid an external definition in another translation
  //   unit.
  //
  // Here we determine whether this function, in isolation, would be a
  // C99 inline definition. MergeCompatibleFunctionDecls looks at
  // previous declarations.
  if (NewFD->isInline() && getLangOptions().C99 && 
      NewFD->getStorageClass() == FunctionDecl::None &&
      NewFD->getDeclContext()->getLookupContext()->isTranslationUnit())
    NewFD->setC99InlineDefinition(true);

  // Check for a previous declaration of this name.
  if (!PrevDecl && NewFD->isExternC(Context)) {
    // Since we did not find anything by this name and we're declaring
    // an extern "C" function, look for a non-visible extern "C"
    // declaration with the same name.
    llvm::DenseMap<DeclarationName, NamedDecl *>::iterator Pos
      = LocallyScopedExternalDecls.find(NewFD->getDeclName());
    if (Pos != LocallyScopedExternalDecls.end())
      PrevDecl = Pos->second;
  }

  // Merge or overload the declaration with an existing declaration of
  // the same name, if appropriate.
  if (PrevDecl) {
    // Determine whether NewFD is an overload of PrevDecl or
    // a declaration that requires merging. If it's an overload,
    // there's no more work to do here; we'll just add the new
    // function to the scope.
    OverloadedFunctionDecl::function_iterator MatchedDecl;

    if (!getLangOptions().CPlusPlus &&
        AllowOverloadingOfFunction(PrevDecl, Context)) {
      OverloadableAttrRequired = true;

      // Functions marked "overloadable" must have a prototype (that
      // we can't get through declaration merging).
      if (!NewFD->getType()->getAsFunctionProtoType()) {
        Diag(NewFD->getLocation(), diag::err_attribute_overloadable_no_prototype)
          << NewFD;
        Redeclaration = true;

        // Turn this into a variadic function with no parameters.
        QualType R = Context.getFunctionType(
                       NewFD->getType()->getAsFunctionType()->getResultType(),
                       0, 0, true, 0);
        NewFD->setType(R);
        return NewFD->setInvalidDecl();
      }
    }

    if (PrevDecl && 
        (!AllowOverloadingOfFunction(PrevDecl, Context) || 
         !IsOverload(NewFD, PrevDecl, MatchedDecl)) &&
        !isa<UsingDecl>(PrevDecl)) {
      Redeclaration = true;
      Decl *OldDecl = PrevDecl;

      // If PrevDecl was an overloaded function, extract the
      // FunctionDecl that matched.
      if (isa<OverloadedFunctionDecl>(PrevDecl))
        OldDecl = *MatchedDecl;

      // NewFD and OldDecl represent declarations that need to be
      // merged. 
      if (MergeFunctionDecl(NewFD, OldDecl))
        return NewFD->setInvalidDecl();

      if (FunctionTemplateDecl *OldTemplateDecl
            = dyn_cast<FunctionTemplateDecl>(OldDecl))
        NewFD->setPreviousDeclaration(OldTemplateDecl->getTemplatedDecl());
      else {
        if (isa<CXXMethodDecl>(NewFD)) // Set access for out-of-line definitions
          NewFD->setAccess(OldDecl->getAccess());
        NewFD->setPreviousDeclaration(cast<FunctionDecl>(OldDecl));
      }
    }
  }

  // In C++, check default arguments now that we have merged decls. Unless
  // the lexical context is the class, because in this case this is done
  // during delayed parsing anyway.
  if (getLangOptions().CPlusPlus && !CurContext->isRecord())
    CheckCXXDefaultArguments(NewFD);
}

void Sema::CheckMain(FunctionDecl* FD) {
  // C++ [basic.start.main]p3:  A program that declares main to be inline
  //   or static is ill-formed.
  // C99 6.7.4p4:  In a hosted environment, the inline function specifier
  //   shall not appear in a declaration of main.
  // static main is not an error under C99, but we should warn about it.
  bool isInline = FD->isInline();
  bool isStatic = FD->getStorageClass() == FunctionDecl::Static;
  if (isInline || isStatic) {
    unsigned diagID = diag::warn_unusual_main_decl;
    if (isInline || getLangOptions().CPlusPlus)
      diagID = diag::err_unusual_main_decl;

    int which = isStatic + (isInline << 1) - 1;
    Diag(FD->getLocation(), diagID) << which;
  }

  QualType T = FD->getType();
  assert(T->isFunctionType() && "function decl is not of function type");
  const FunctionType* FT = T->getAsFunctionType();
  
  if (!Context.hasSameUnqualifiedType(FT->getResultType(), Context.IntTy)) {
    // TODO: add a replacement fixit to turn the return type into 'int'.
    Diag(FD->getTypeSpecStartLoc(), diag::err_main_returns_nonint);
    FD->setInvalidDecl(true);
  }

  // Treat protoless main() as nullary.
  if (isa<FunctionNoProtoType>(FT)) return;

  const FunctionProtoType* FTP = cast<const FunctionProtoType>(FT);
  unsigned nparams = FTP->getNumArgs();
  assert(FD->getNumParams() == nparams);

  if (nparams > 3) {
    Diag(FD->getLocation(), diag::err_main_surplus_args) << nparams;
    FD->setInvalidDecl(true);
    nparams = 3;
  }

  // FIXME: a lot of the following diagnostics would be improved
  // if we had some location information about types.

  QualType CharPP =
    Context.getPointerType(Context.getPointerType(Context.CharTy));
  QualType Expected[] = { Context.IntTy, CharPP, CharPP };

  for (unsigned i = 0; i < nparams; ++i) {
    QualType AT = FTP->getArgType(i);

    bool mismatch = true;

    if (Context.hasSameUnqualifiedType(AT, Expected[i]))
      mismatch = false;
    else if (Expected[i] == CharPP) {
      // As an extension, the following forms are okay:
      //   char const **
      //   char const * const *
      //   char * const *

      QualifierSet qs;
      const PointerType* PT;
      if ((PT = qs.strip(AT)->getAs<PointerType>()) &&
          (PT = qs.strip(PT->getPointeeType())->getAs<PointerType>()) &&
          (QualType(qs.strip(PT->getPointeeType()), 0) == Context.CharTy)) {
        qs.removeConst();
        mismatch = !qs.empty();
      }
    }

    if (mismatch) {
      Diag(FD->getLocation(), diag::err_main_arg_wrong) << i << Expected[i];
      // TODO: suggest replacing given type with expected type
      FD->setInvalidDecl(true);
    }
  }

  if (nparams == 1 && !FD->isInvalidDecl()) {
    Diag(FD->getLocation(), diag::warn_main_one_arg);
  }
}

bool Sema::CheckForConstantInitializer(Expr *Init, QualType DclT) {
  // FIXME: Need strict checking.  In C89, we need to check for
  // any assignment, increment, decrement, function-calls, or
  // commas outside of a sizeof.  In C99, it's the same list,
  // except that the aforementioned are allowed in unevaluated
  // expressions.  Everything else falls under the
  // "may accept other forms of constant expressions" exception.
  // (We never end up here for C++, so the constant expression
  // rules there don't matter.)
  if (Init->isConstantInitializer(Context))
    return false;
  Diag(Init->getExprLoc(), diag::err_init_element_not_constant)
    << Init->getSourceRange();
  return true;
}

void Sema::AddInitializerToDecl(DeclPtrTy dcl, ExprArg init) {
  AddInitializerToDecl(dcl, move(init), /*DirectInit=*/false);
}

/// AddInitializerToDecl - Adds the initializer Init to the
/// declaration dcl. If DirectInit is true, this is C++ direct
/// initialization rather than copy initialization.
void Sema::AddInitializerToDecl(DeclPtrTy dcl, ExprArg init, bool DirectInit) {
  Decl *RealDecl = dcl.getAs<Decl>();
  // If there is no declaration, there was an error parsing it.  Just ignore
  // the initializer.
  if (RealDecl == 0)
    return;
  
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(RealDecl)) {
    // With declarators parsed the way they are, the parser cannot
    // distinguish between a normal initializer and a pure-specifier.
    // Thus this grotesque test.
    IntegerLiteral *IL;
    Expr *Init = static_cast<Expr *>(init.get());
    if ((IL = dyn_cast<IntegerLiteral>(Init)) && IL->getValue() == 0 &&
        Context.getCanonicalType(IL->getType()) == Context.IntTy) {
      if (Method->isVirtualAsWritten()) {
        Method->setPure();

        // A class is abstract if at least one function is pure virtual.
        cast<CXXRecordDecl>(CurContext)->setAbstract(true);
      } else if (!Method->isInvalidDecl()) {
        Diag(Method->getLocation(), diag::err_non_virtual_pure)
          << Method->getDeclName() << Init->getSourceRange();
        Method->setInvalidDecl();
      }
    } else {
      Diag(Method->getLocation(), diag::err_member_function_initialization)
        << Method->getDeclName() << Init->getSourceRange();
      Method->setInvalidDecl();
    }
    return;
  }

  VarDecl *VDecl = dyn_cast<VarDecl>(RealDecl);
  if (!VDecl) {
    if (getLangOptions().CPlusPlus &&
        RealDecl->getLexicalDeclContext()->isRecord() &&
        isa<NamedDecl>(RealDecl))
      Diag(RealDecl->getLocation(), diag::err_member_initialization)
        << cast<NamedDecl>(RealDecl)->getDeclName();
    else
      Diag(RealDecl->getLocation(), diag::err_illegal_initializer);
    RealDecl->setInvalidDecl();
    return;
  }

  if (!VDecl->getType()->isArrayType() &&
      RequireCompleteType(VDecl->getLocation(), VDecl->getType(),
                          diag::err_typecheck_decl_incomplete_type)) {
    RealDecl->setInvalidDecl();
    return;
  }

  const VarDecl *Def = 0;
  if (VDecl->getDefinition(Def)) {
    Diag(VDecl->getLocation(), diag::err_redefinition) 
      << VDecl->getDeclName();
    Diag(Def->getLocation(), diag::note_previous_definition);
    VDecl->setInvalidDecl();
    return;
  }

  // Take ownership of the expression, now that we're sure we have somewhere
  // to put it.
  Expr *Init = init.takeAs<Expr>();
  assert(Init && "missing initializer");

  // Get the decls type and save a reference for later, since
  // CheckInitializerTypes may change it.
  QualType DclT = VDecl->getType(), SavT = DclT;
  if (VDecl->isBlockVarDecl()) {
    if (VDecl->hasExternalStorage()) { // C99 6.7.8p5
      Diag(VDecl->getLocation(), diag::err_block_extern_cant_init);
      VDecl->setInvalidDecl();
    } else if (!VDecl->isInvalidDecl()) {
      if (CheckInitializerTypes(Init, DclT, VDecl->getLocation(),
                                VDecl->getDeclName(), DirectInit))
        VDecl->setInvalidDecl();
      
      // C++ 3.6.2p2, allow dynamic initialization of static initializers.
      // Don't check invalid declarations to avoid emitting useless diagnostics.
      if (!getLangOptions().CPlusPlus && !VDecl->isInvalidDecl()) {
        if (VDecl->getStorageClass() == VarDecl::Static) // C99 6.7.8p4.
          CheckForConstantInitializer(Init, DclT);
      }
    }
  } else if (VDecl->isStaticDataMember() && 
             VDecl->getLexicalDeclContext()->isRecord()) {
    // This is an in-class initialization for a static data member, e.g.,
    //
    // struct S {
    //   static const int value = 17;
    // };

    // Attach the initializer
    VDecl->setInit(Context, Init);

    // C++ [class.mem]p4:
    //   A member-declarator can contain a constant-initializer only
    //   if it declares a static member (9.4) of const integral or
    //   const enumeration type, see 9.4.2.
    QualType T = VDecl->getType();
    if (!T->isDependentType() && 
        (!Context.getCanonicalType(T).isConstQualified() ||
         !T->isIntegralType())) {
      Diag(VDecl->getLocation(), diag::err_member_initialization)
        << VDecl->getDeclName() << Init->getSourceRange();
      VDecl->setInvalidDecl();
    } else {
      // C++ [class.static.data]p4:
      //   If a static data member is of const integral or const
      //   enumeration type, its declaration in the class definition
      //   can specify a constant-initializer which shall be an
      //   integral constant expression (5.19).
      if (!Init->isTypeDependent() &&
          !Init->getType()->isIntegralType()) {
        // We have a non-dependent, non-integral or enumeration type.
        Diag(Init->getSourceRange().getBegin(), 
             diag::err_in_class_initializer_non_integral_type)
          << Init->getType() << Init->getSourceRange();
        VDecl->setInvalidDecl();
      } else if (!Init->isTypeDependent() && !Init->isValueDependent()) {
        // Check whether the expression is a constant expression.
        llvm::APSInt Value;
        SourceLocation Loc;
        if (!Init->isIntegerConstantExpr(Value, Context, &Loc)) {
          Diag(Loc, diag::err_in_class_initializer_non_constant)
            << Init->getSourceRange();
          VDecl->setInvalidDecl();
        } else if (!VDecl->getType()->isDependentType())
          ImpCastExprToType(Init, VDecl->getType());
      }
    }
  } else if (VDecl->isFileVarDecl()) {
    if (VDecl->getStorageClass() == VarDecl::Extern)
      Diag(VDecl->getLocation(), diag::warn_extern_init);
    if (!VDecl->isInvalidDecl())
      if (CheckInitializerTypes(Init, DclT, VDecl->getLocation(),
                                VDecl->getDeclName(), DirectInit))
        VDecl->setInvalidDecl();
    
    // C++ 3.6.2p2, allow dynamic initialization of static initializers.
    // Don't check invalid declarations to avoid emitting useless diagnostics.
    if (!getLangOptions().CPlusPlus && !VDecl->isInvalidDecl()) {
      // C99 6.7.8p4. All file scoped initializers need to be constant.
      CheckForConstantInitializer(Init, DclT);
    }
  }
  // If the type changed, it means we had an incomplete type that was
  // completed by the initializer. For example: 
  //   int ary[] = { 1, 3, 5 };
  // "ary" transitions from a VariableArrayType to a ConstantArrayType.
  if (!VDecl->isInvalidDecl() && (DclT != SavT)) {
    VDecl->setType(DclT);
    Init->setType(DclT);
  }
    
  Init = MaybeCreateCXXExprWithTemporaries(Init, 
                                           /*ShouldDestroyTemporaries=*/true);
  // Attach the initializer to the decl.
  VDecl->setInit(Context, Init);

  // If the previous declaration of VDecl was a tentative definition,
  // remove it from the set of tentative definitions.
  if (VDecl->getPreviousDeclaration() &&
      VDecl->getPreviousDeclaration()->isTentativeDefinition(Context)) {
    bool Deleted = TentativeDefinitions.erase(VDecl->getDeclName());
    assert(Deleted && "Unrecorded tentative definition?"); Deleted=Deleted;
  }

  return;
}

void Sema::ActOnUninitializedDecl(DeclPtrTy dcl, 
                                  bool TypeContainsUndeducedAuto) {
  Decl *RealDecl = dcl.getAs<Decl>();

  // If there is no declaration, there was an error parsing it. Just ignore it.
  if (RealDecl == 0)
    return;

  if (VarDecl *Var = dyn_cast<VarDecl>(RealDecl)) {
    QualType Type = Var->getType();

    // Record tentative definitions.
    if (Var->isTentativeDefinition(Context)) {
      std::pair<llvm::DenseMap<DeclarationName, VarDecl *>::iterator, bool>
        InsertPair = 
           TentativeDefinitions.insert(std::make_pair(Var->getDeclName(), Var));
      
      // Keep the latest definition in the map.  If we see 'int i; int i;' we
      // want the second one in the map.
      InsertPair.first->second = Var;

      // However, for the list, we don't care about the order, just make sure
      // that there are no dupes for a given declaration name.
      if (InsertPair.second)
        TentativeDefinitionList.push_back(Var->getDeclName());
    }

    // C++ [dcl.init.ref]p3:
    //   The initializer can be omitted for a reference only in a
    //   parameter declaration (8.3.5), in the declaration of a
    //   function return type, in the declaration of a class member
    //   within its class declaration (9.2), and where the extern
    //   specifier is explicitly used.
    if (Type->isReferenceType() && !Var->hasExternalStorage()) {
      Diag(Var->getLocation(), diag::err_reference_var_requires_init)
        << Var->getDeclName()
        << SourceRange(Var->getLocation(), Var->getLocation());
      Var->setInvalidDecl();
      return;
    }

    // C++0x [dcl.spec.auto]p3
    if (TypeContainsUndeducedAuto) {
      Diag(Var->getLocation(), diag::err_auto_var_requires_init)
        << Var->getDeclName() << Type;
      Var->setInvalidDecl();
      return;
    }
    
    // C++ [dcl.init]p9:
    //
    //   If no initializer is specified for an object, and the object
    //   is of (possibly cv-qualified) non-POD class type (or array
    //   thereof), the object shall be default-initialized; if the
    //   object is of const-qualified type, the underlying class type
    //   shall have a user-declared default constructor.
    if (getLangOptions().CPlusPlus) {
      QualType InitType = Type;
      if (const ArrayType *Array = Context.getAsArrayType(Type))
        InitType = Array->getElementType();
      if ((!Var->hasExternalStorage() && !Var->isExternC(Context)) &&
          InitType->isRecordType() && !InitType->isDependentType()) {
        CXXRecordDecl *RD = 
          cast<CXXRecordDecl>(InitType->getAs<RecordType>()->getDecl());
        CXXConstructorDecl *Constructor = 0;
        if (!RequireCompleteType(Var->getLocation(), InitType, 
                                    diag::err_invalid_incomplete_type_use))
          Constructor
            = PerformInitializationByConstructor(InitType, 0, 0, 
                                                 Var->getLocation(),
                                               SourceRange(Var->getLocation(),
                                                           Var->getLocation()),
                                                 Var->getDeclName(),
                                                 IK_Default);
        if (!Constructor)
          Var->setInvalidDecl();
        else { 
          if (!RD->hasTrivialConstructor() || !RD->hasTrivialDestructor())
            InitializeVarWithConstructor(Var, Constructor, InitType, 0, 0);
          FinalizeVarWithDestructor(Var, InitType);
        }
      }
    }

#if 0
    // FIXME: Temporarily disabled because we are not properly parsing
    // linkage specifications on declarations, e.g., 
    //
    //   extern "C" const CGPoint CGPointerZero;
    //
    // C++ [dcl.init]p9:
    //
    //     If no initializer is specified for an object, and the
    //     object is of (possibly cv-qualified) non-POD class type (or
    //     array thereof), the object shall be default-initialized; if
    //     the object is of const-qualified type, the underlying class
    //     type shall have a user-declared default
    //     constructor. Otherwise, if no initializer is specified for
    //     an object, the object and its subobjects, if any, have an
    //     indeterminate initial value; if the object or any of its
    //     subobjects are of const-qualified type, the program is
    //     ill-formed.
    //
    // This isn't technically an error in C, so we don't diagnose it.
    //
    // FIXME: Actually perform the POD/user-defined default
    // constructor check.
    if (getLangOptions().CPlusPlus &&
        Context.getCanonicalType(Type).isConstQualified() &&
        !Var->hasExternalStorage())
      Diag(Var->getLocation(),  diag::err_const_var_requires_init)
        << Var->getName()
        << SourceRange(Var->getLocation(), Var->getLocation());
#endif
  }
}

Sema::DeclGroupPtrTy Sema::FinalizeDeclaratorGroup(Scope *S, const DeclSpec &DS,
                                                   DeclPtrTy *Group,
                                                   unsigned NumDecls) {
  llvm::SmallVector<Decl*, 8> Decls;

  if (DS.isTypeSpecOwned())
    Decls.push_back((Decl*)DS.getTypeRep());

  for (unsigned i = 0; i != NumDecls; ++i)
    if (Decl *D = Group[i].getAs<Decl>())
      Decls.push_back(D);
  
  // Perform semantic analysis that depends on having fully processed both
  // the declarator and initializer.
  for (unsigned i = 0, e = Decls.size(); i != e; ++i) {
    VarDecl *IDecl = dyn_cast<VarDecl>(Decls[i]);
    if (!IDecl)
      continue;
    QualType T = IDecl->getType();
    
    // Block scope. C99 6.7p7: If an identifier for an object is declared with
    // no linkage (C99 6.2.2p6), the type for the object shall be complete...
    if (IDecl->isBlockVarDecl() && !IDecl->hasExternalStorage()) {
      if (!IDecl->isInvalidDecl() &&
          RequireCompleteType(IDecl->getLocation(), T, 
                              diag::err_typecheck_decl_incomplete_type))
        IDecl->setInvalidDecl();
    }
    // File scope. C99 6.9.2p2: A declaration of an identifier for an
    // object that has file scope without an initializer, and without a
    // storage-class specifier or with the storage-class specifier "static",
    // constitutes a tentative definition. Note: A tentative definition with
    // external linkage is valid (C99 6.2.2p5).
    if (IDecl->isTentativeDefinition(Context) && !IDecl->isInvalidDecl()) {
      if (const IncompleteArrayType *ArrayT
          = Context.getAsIncompleteArrayType(T)) {
        if (RequireCompleteType(IDecl->getLocation(),
                                ArrayT->getElementType(),
                                diag::err_illegal_decl_array_incomplete_type))
          IDecl->setInvalidDecl();
      } else if (IDecl->getStorageClass() == VarDecl::Static) {
        // C99 6.9.2p3: If the declaration of an identifier for an object is
        // a tentative definition and has internal linkage (C99 6.2.2p3), the
        // declared type shall not be an incomplete type.
        // NOTE: code such as the following
        //     static struct s;
        //     struct s { int a; };
        // is accepted by gcc. Hence here we issue a warning instead of
        // an error and we do not invalidate the static declaration.
        // NOTE: to avoid multiple warnings, only check the first declaration.
        if (IDecl->getPreviousDeclaration() == 0)
          RequireCompleteType(IDecl->getLocation(), T,
                              diag::ext_typecheck_decl_incomplete_type);
      }
    }
  }
  return DeclGroupPtrTy::make(DeclGroupRef::Create(Context,
                                                   Decls.data(), Decls.size()));
}


/// ActOnParamDeclarator - Called from Parser::ParseFunctionDeclarator()
/// to introduce parameters into function prototype scope.
Sema::DeclPtrTy 
Sema::ActOnParamDeclarator(Scope *S, Declarator &D) {
  const DeclSpec &DS = D.getDeclSpec();

  // Verify C99 6.7.5.3p2: The only SCS allowed is 'register'.
  VarDecl::StorageClass StorageClass = VarDecl::None;
  if (DS.getStorageClassSpec() == DeclSpec::SCS_register) {
    StorageClass = VarDecl::Register;
  } else if (DS.getStorageClassSpec() != DeclSpec::SCS_unspecified) {
    Diag(DS.getStorageClassSpecLoc(),
         diag::err_invalid_storage_class_in_func_decl);
    D.getMutableDeclSpec().ClearStorageClassSpecs();
  }

  if (D.getDeclSpec().isThreadSpecified())
    Diag(D.getDeclSpec().getThreadSpecLoc(), diag::err_invalid_thread);

  DiagnoseFunctionSpecifiers(D);

  // Check that there are no default arguments inside the type of this
  // parameter (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);
 
  DeclaratorInfo *DInfo = 0;
  TagDecl *OwnedDecl = 0;
  QualType parmDeclType = GetTypeForDeclarator(D, S, &DInfo, /*Skip=*/0,
                                               &OwnedDecl);
  
  if (getLangOptions().CPlusPlus && OwnedDecl && OwnedDecl->isDefinition()) {
    // C++ [dcl.fct]p6:
    //   Types shall not be defined in return or parameter types.
    Diag(OwnedDecl->getLocation(), diag::err_type_defined_in_param_type)
      << Context.getTypeDeclType(OwnedDecl);
  }

  // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  // Can this happen for params?  We already checked that they don't conflict
  // among each other.  Here they can only shadow globals, which is ok.
  IdentifierInfo *II = D.getIdentifier();
  if (II) {
    if (NamedDecl *PrevDecl = LookupName(S, II, LookupOrdinaryName)) {
      if (PrevDecl->isTemplateParameter()) {
        // Maybe we will complain about the shadowed template parameter.
        DiagnoseTemplateParameterShadow(D.getIdentifierLoc(), PrevDecl);
        // Just pretend that we didn't see the previous declaration.
        PrevDecl = 0;
      } else if (S->isDeclScope(DeclPtrTy::make(PrevDecl))) {
        Diag(D.getIdentifierLoc(), diag::err_param_redefinition) << II;

        // Recover by removing the name
        II = 0;
        D.SetIdentifier(0, D.getIdentifierLoc());
      }
    }
  }

  // Parameters can not be abstract class types.
  // For record types, this is done by the AbstractClassUsageDiagnoser once
  // the class has been completely parsed. 
  if (!CurContext->isRecord() && 
      RequireNonAbstractType(D.getIdentifierLoc(), parmDeclType, 
                             diag::err_abstract_type_in_decl,
                             AbstractParamType))
    D.setInvalidType(true);

  QualType T = adjustParameterType(parmDeclType);
  
  ParmVarDecl *New;
  if (T == parmDeclType) // parameter type did not need adjustment
    New = ParmVarDecl::Create(Context, CurContext, 
                              D.getIdentifierLoc(), II,
                              parmDeclType, DInfo, StorageClass, 
                              0);
  else // keep track of both the adjusted and unadjusted types
    New = OriginalParmVarDecl::Create(Context, CurContext, 
                                      D.getIdentifierLoc(), II, T, DInfo,
                                      parmDeclType, StorageClass, 0);
  
  if (D.isInvalidType())
    New->setInvalidDecl();

  // Parameter declarators cannot be interface types. All ObjC objects are
  // passed by reference.
  if (T->isObjCInterfaceType()) {
    Diag(D.getIdentifierLoc(),
         diag::err_object_cannot_be_passed_returned_by_value) << 1 << T;
    New->setInvalidDecl();
  }
  
  // Parameter declarators cannot be qualified (C++ [dcl.meaning]p1).
  if (D.getCXXScopeSpec().isSet()) {
    Diag(D.getIdentifierLoc(), diag::err_qualified_param_declarator)
      << D.getCXXScopeSpec().getRange();
    New->setInvalidDecl();
  }

  // Add the parameter declaration into this scope.
  S->AddDecl(DeclPtrTy::make(New));
  if (II)
    IdResolver.AddDecl(New);

  ProcessDeclAttributes(S, New, D);

  if (New->hasAttr<BlocksAttr>()) {
    Diag(New->getLocation(), diag::err_block_on_nonlocal);
  }
  return DeclPtrTy::make(New);
}

void Sema::ActOnFinishKNRParamDeclarations(Scope *S, Declarator &D,
                                           SourceLocation LocAfterDecls) {
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;

  // Verify 6.9.1p6: 'every identifier in the identifier list shall be declared'
  // for a K&R function.
  if (!FTI.hasPrototype) {
    for (int i = FTI.NumArgs; i != 0; /* decrement in loop */) {
      --i;
      if (FTI.ArgInfo[i].Param == 0) {
        std::string Code = "  int ";
        Code += FTI.ArgInfo[i].Ident->getName();
        Code += ";\n";
        Diag(FTI.ArgInfo[i].IdentLoc, diag::ext_param_not_declared)
          << FTI.ArgInfo[i].Ident
          << CodeModificationHint::CreateInsertion(LocAfterDecls, Code);

        // Implicitly declare the argument as type 'int' for lack of a better
        // type.
        DeclSpec DS;
        const char* PrevSpec; // unused
        unsigned DiagID; // unused
        DS.SetTypeSpecType(DeclSpec::TST_int, FTI.ArgInfo[i].IdentLoc, 
                           PrevSpec, DiagID);
        Declarator ParamD(DS, Declarator::KNRTypeListContext);
        ParamD.SetIdentifier(FTI.ArgInfo[i].Ident, FTI.ArgInfo[i].IdentLoc);
        FTI.ArgInfo[i].Param = ActOnParamDeclarator(S, ParamD);
      }
    }
  } 
}

Sema::DeclPtrTy Sema::ActOnStartOfFunctionDef(Scope *FnBodyScope,
                                              Declarator &D) {
  assert(getCurFunctionDecl() == 0 && "Function parsing confused");
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;

  if (FTI.hasPrototype) {
    // FIXME: Diagnose arguments without names in C. 
  }
  
  Scope *ParentScope = FnBodyScope->getParent();

  DeclPtrTy DP = HandleDeclarator(ParentScope, D, 
                                  MultiTemplateParamsArg(*this),
                                  /*IsFunctionDefinition=*/true);
  return ActOnStartOfFunctionDef(FnBodyScope, DP);
}

Sema::DeclPtrTy Sema::ActOnStartOfFunctionDef(Scope *FnBodyScope, DeclPtrTy D) {
  if (!D)
    return D;
  FunctionDecl *FD = 0;
  
  if (FunctionTemplateDecl *FunTmpl 
        = dyn_cast<FunctionTemplateDecl>(D.getAs<Decl>()))
    FD = FunTmpl->getTemplatedDecl();
  else
    FD = cast<FunctionDecl>(D.getAs<Decl>());

  CurFunctionNeedsScopeChecking = false;
  
  // See if this is a redefinition.
  const FunctionDecl *Definition;
  if (FD->getBody(Definition)) {
    Diag(FD->getLocation(), diag::err_redefinition) << FD->getDeclName();
    Diag(Definition->getLocation(), diag::note_previous_definition);
  }

  // Builtin functions cannot be defined.
  if (unsigned BuiltinID = FD->getBuiltinID(Context)) {
    if (!Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID)) {
      Diag(FD->getLocation(), diag::err_builtin_definition) << FD;
      FD->setInvalidDecl();
    }
  }

  // The return type of a function definition must be complete
  // (C99 6.9.1p3, C++ [dcl.fct]p6).
  QualType ResultType = FD->getResultType();
  if (!ResultType->isDependentType() && !ResultType->isVoidType() &&
      !FD->isInvalidDecl() &&
      RequireCompleteType(FD->getLocation(), ResultType,
                          diag::err_func_def_incomplete_result))
    FD->setInvalidDecl();

  // GNU warning -Wmissing-prototypes:
  //   Warn if a global function is defined without a previous
  //   prototype declaration. This warning is issued even if the
  //   definition itself provides a prototype. The aim is to detect
  //   global functions that fail to be declared in header files.
  if (!FD->isInvalidDecl() && FD->isGlobal() && !isa<CXXMethodDecl>(FD) &&
      !FD->isMain(Context)) {
    bool MissingPrototype = true;
    for (const FunctionDecl *Prev = FD->getPreviousDeclaration();
         Prev; Prev = Prev->getPreviousDeclaration()) {
      // Ignore any declarations that occur in function or method
      // scope, because they aren't visible from the header.
      if (Prev->getDeclContext()->isFunctionOrMethod())
        continue;

      MissingPrototype = !Prev->getType()->isFunctionProtoType();
      break;
    }

    if (MissingPrototype)
      Diag(FD->getLocation(), diag::warn_missing_prototype) << FD;
  }

  if (FnBodyScope)
    PushDeclContext(FnBodyScope, FD);

  // Check the validity of our function parameters
  CheckParmsForFunctionDef(FD);

  // Introduce our parameters into the function scope
  for (unsigned p = 0, NumParams = FD->getNumParams(); p < NumParams; ++p) {
    ParmVarDecl *Param = FD->getParamDecl(p);
    Param->setOwningFunction(FD);

    // If this has an identifier, add it to the scope stack.
    if (Param->getIdentifier() && FnBodyScope)
      PushOnScopeChains(Param, FnBodyScope);
  }

  // Checking attributes of current function definition
  // dllimport attribute.
  if (FD->getAttr<DLLImportAttr>() && 
      (!FD->getAttr<DLLExportAttr>())) {
    // dllimport attribute cannot be applied to definition.
    if (!(FD->getAttr<DLLImportAttr>())->isInherited()) {
      Diag(FD->getLocation(),
           diag::err_attribute_can_be_applied_only_to_symbol_declaration)
        << "dllimport";
      FD->setInvalidDecl();
      return DeclPtrTy::make(FD);
    } else {
      // If a symbol previously declared dllimport is later defined, the
      // attribute is ignored in subsequent references, and a warning is
      // emitted.
      Diag(FD->getLocation(),
           diag::warn_redeclaration_without_attribute_prev_attribute_ignored)
        << FD->getNameAsCString() << "dllimport";
    }
  }
  return DeclPtrTy::make(FD);
}

Sema::DeclPtrTy Sema::ActOnFinishFunctionBody(DeclPtrTy D, StmtArg BodyArg) {
  return ActOnFinishFunctionBody(D, move(BodyArg), false);
}

Sema::DeclPtrTy Sema::ActOnFinishFunctionBody(DeclPtrTy D, StmtArg BodyArg,
                                              bool IsInstantiation) {
  Decl *dcl = D.getAs<Decl>();
  Stmt *Body = BodyArg.takeAs<Stmt>();

  FunctionDecl *FD = 0;
  FunctionTemplateDecl *FunTmpl = dyn_cast_or_null<FunctionTemplateDecl>(dcl);
  if (FunTmpl)
    FD = FunTmpl->getTemplatedDecl();
  else
    FD = dyn_cast_or_null<FunctionDecl>(dcl);

  if (FD) {
    FD->setBody(Body);
    if (FD->isMain(Context))
      // C and C++ allow for main to automagically return 0.
      // Implements C++ [basic.start.main]p5 and C99 5.1.2.2.3.
      FD->setHasImplicitReturnZero(true);
    else
      CheckFallThroughForFunctionDef(FD, Body);
    
    if (!FD->isInvalidDecl())
      DiagnoseUnusedParameters(FD->param_begin(), FD->param_end());
    
    // C++ [basic.def.odr]p2:
    //   [...] A virtual member function is used if it is not pure. [...]
    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(FD))
      if (Method->isVirtual() && !Method->isPure())
        MarkDeclarationReferenced(Method->getLocation(), Method);
    
    assert(FD == getCurFunctionDecl() && "Function parsing confused");
  } else if (ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(dcl)) {
    assert(MD == getCurMethodDecl() && "Method parsing confused");
    MD->setBody(Body);
    CheckFallThroughForFunctionDef(MD, Body);
    MD->setEndLoc(Body->getLocEnd());
    
    if (!MD->isInvalidDecl())
      DiagnoseUnusedParameters(MD->param_begin(), MD->param_end());
  } else {
    Body->Destroy(Context);
    return DeclPtrTy();
  }
  if (!IsInstantiation)
    PopDeclContext();

  // Verify and clean out per-function state.

  assert(&getLabelMap() == &FunctionLabelMap && "Didn't pop block right?");
  
  // Check goto/label use.
  for (llvm::DenseMap<IdentifierInfo*, LabelStmt*>::iterator
       I = FunctionLabelMap.begin(), E = FunctionLabelMap.end(); I != E; ++I) {
    LabelStmt *L = I->second;
    
    // Verify that we have no forward references left.  If so, there was a goto
    // or address of a label taken, but no definition of it.  Label fwd
    // definitions are indicated with a null substmt.
    if (L->getSubStmt() != 0)
      continue;
    
    // Emit error.
    Diag(L->getIdentLoc(), diag::err_undeclared_label_use) << L->getName();
    
    // At this point, we have gotos that use the bogus label.  Stitch it into
    // the function body so that they aren't leaked and that the AST is well
    // formed.
    if (Body == 0) {
      // The whole function wasn't parsed correctly, just delete this.
      L->Destroy(Context);
      continue;
    }
    
    // Otherwise, the body is valid: we want to stitch the label decl into the
    // function somewhere so that it is properly owned and so that the goto
    // has a valid target.  Do this by creating a new compound stmt with the
    // label in it.

    // Give the label a sub-statement.
    L->setSubStmt(new (Context) NullStmt(L->getIdentLoc()));

    CompoundStmt *Compound = isa<CXXTryStmt>(Body) ?
                               cast<CXXTryStmt>(Body)->getTryBlock() :
                               cast<CompoundStmt>(Body);
    std::vector<Stmt*> Elements(Compound->body_begin(), Compound->body_end());
    Elements.push_back(L);
    Compound->setStmts(Context, &Elements[0], Elements.size());
  }
  FunctionLabelMap.clear();

  if (!Body) return D;

  // Verify that that gotos and switch cases don't jump into scopes illegally.
  if (CurFunctionNeedsScopeChecking)
    DiagnoseInvalidJumps(Body);
  
  // C++ constructors that have function-try-blocks can't have return 
  // statements in the handlers of that block. (C++ [except.handle]p14) 
  // Verify this.
  if (FD && isa<CXXConstructorDecl>(FD) && isa<CXXTryStmt>(Body))
    DiagnoseReturnInConstructorExceptionHandler(cast<CXXTryStmt>(Body));
  
  if (CXXDestructorDecl *Destructor = dyn_cast<CXXDestructorDecl>(dcl))
    Destructor->computeBaseOrMembersToDestroy(Context);
  return D;
}

/// ImplicitlyDefineFunction - An undeclared identifier was used in a function
/// call, forming a call to an implicitly defined function (per C99 6.5.1p2).
NamedDecl *Sema::ImplicitlyDefineFunction(SourceLocation Loc, 
                                          IdentifierInfo &II, Scope *S) {
  // Before we produce a declaration for an implicitly defined
  // function, see whether there was a locally-scoped declaration of
  // this name as a function or variable. If so, use that
  // (non-visible) declaration, and complain about it.
  llvm::DenseMap<DeclarationName, NamedDecl *>::iterator Pos
    = LocallyScopedExternalDecls.find(&II);
  if (Pos != LocallyScopedExternalDecls.end()) {
    Diag(Loc, diag::warn_use_out_of_scope_declaration) << Pos->second;
    Diag(Pos->second->getLocation(), diag::note_previous_declaration);
    return Pos->second;
  }

  // Extension in C99.  Legal in C90, but warn about it.
  if (getLangOptions().C99)
    Diag(Loc, diag::ext_implicit_function_decl) << &II;
  else
    Diag(Loc, diag::warn_implicit_function_decl) << &II;
  
  // FIXME: handle stuff like:
  // void foo() { extern float X(); }
  // void bar() { X(); }  <-- implicit decl for X in another scope.

  // Set a Declarator for the implicit definition: int foo();
  const char *Dummy;
  DeclSpec DS;
  unsigned DiagID;
  bool Error = DS.SetTypeSpecType(DeclSpec::TST_int, Loc, Dummy, DiagID);
  Error = Error; // Silence warning.
  assert(!Error && "Error setting up implicit decl!");
  Declarator D(DS, Declarator::BlockContext);
  D.AddTypeInfo(DeclaratorChunk::getFunction(false, false, SourceLocation(), 0,
                                             0, 0, false, SourceLocation(),
                                             false, 0,0,0, Loc, Loc, D),
                SourceLocation());
  D.SetIdentifier(&II, Loc);

  // Insert this function into translation-unit scope.

  DeclContext *PrevDC = CurContext;
  CurContext = Context.getTranslationUnitDecl();
 
  FunctionDecl *FD = 
 dyn_cast<FunctionDecl>(ActOnDeclarator(TUScope, D).getAs<Decl>());
  FD->setImplicit();

  CurContext = PrevDC;

  AddKnownFunctionAttributes(FD);

  return FD;
}

/// \brief Adds any function attributes that we know a priori based on
/// the declaration of this function.
///
/// These attributes can apply both to implicitly-declared builtins
/// (like __builtin___printf_chk) or to library-declared functions
/// like NSLog or printf.
void Sema::AddKnownFunctionAttributes(FunctionDecl *FD) {
  if (FD->isInvalidDecl())
    return;

  // If this is a built-in function, map its builtin attributes to
  // actual attributes.
  if (unsigned BuiltinID = FD->getBuiltinID(Context)) {
    // Handle printf-formatting attributes.
    unsigned FormatIdx;
    bool HasVAListArg;
    if (Context.BuiltinInfo.isPrintfLike(BuiltinID, FormatIdx, HasVAListArg)) {
      if (!FD->getAttr<FormatAttr>())
        FD->addAttr(::new (Context) FormatAttr("printf", FormatIdx + 1,
                                             HasVAListArg ? 0 : FormatIdx + 2));
    }

    // Mark const if we don't care about errno and that is the only
    // thing preventing the function from being const. This allows
    // IRgen to use LLVM intrinsics for such functions.
    if (!getLangOptions().MathErrno &&
        Context.BuiltinInfo.isConstWithoutErrno(BuiltinID)) {
      if (!FD->getAttr<ConstAttr>())
        FD->addAttr(::new (Context) ConstAttr());
    }

    if (Context.BuiltinInfo.isNoReturn(BuiltinID))
      FD->addAttr(::new (Context) NoReturnAttr());
  }

  IdentifierInfo *Name = FD->getIdentifier();
  if (!Name)
    return;
  if ((!getLangOptions().CPlusPlus && 
       FD->getDeclContext()->isTranslationUnit()) ||
      (isa<LinkageSpecDecl>(FD->getDeclContext()) &&
       cast<LinkageSpecDecl>(FD->getDeclContext())->getLanguage() == 
       LinkageSpecDecl::lang_c)) {
    // Okay: this could be a libc/libm/Objective-C function we know
    // about.
  } else
    return;

  if (Name->isStr("NSLog") || Name->isStr("NSLogv")) {
    // FIXME: NSLog and NSLogv should be target specific
    if (const FormatAttr *Format = FD->getAttr<FormatAttr>()) {
      // FIXME: We known better than our headers.
      const_cast<FormatAttr *>(Format)->setType("printf");
    } else 
      FD->addAttr(::new (Context) FormatAttr("printf", 1,
                                             Name->isStr("NSLogv") ? 0 : 2));
  } else if (Name->isStr("asprintf") || Name->isStr("vasprintf")) {
    // FIXME: asprintf and vasprintf aren't C99 functions. Should they be
    // target-specific builtins, perhaps? 
    if (!FD->getAttr<FormatAttr>())
      FD->addAttr(::new (Context) FormatAttr("printf", 2,
                                             Name->isStr("vasprintf") ? 0 : 3));
  }
}

TypedefDecl *Sema::ParseTypedefDecl(Scope *S, Declarator &D, QualType T) {
  assert(D.getIdentifier() && "Wrong callback for declspec without declarator");
  assert(!T.isNull() && "GetTypeForDeclarator() returned null type");
  
  // Scope manipulation handled by caller.
  TypedefDecl *NewTD = TypedefDecl::Create(Context, CurContext,
                                           D.getIdentifierLoc(),
                                           D.getIdentifier(), 
                                           T);
  
  if (TagType *TT = dyn_cast<TagType>(T)) {
    TagDecl *TD = TT->getDecl();
    
    // If the TagDecl that the TypedefDecl points to is an anonymous decl
    // keep track of the TypedefDecl.
    if (!TD->getIdentifier() && !TD->getTypedefForAnonDecl())
      TD->setTypedefForAnonDecl(NewTD);
  }

  if (D.isInvalidType())
    NewTD->setInvalidDecl();
  return NewTD;
}


/// \brief Determine whether a tag with a given kind is acceptable
/// as a redeclaration of the given tag declaration.
///
/// \returns true if the new tag kind is acceptable, false otherwise.
bool Sema::isAcceptableTagRedeclaration(const TagDecl *Previous, 
                                        TagDecl::TagKind NewTag,
                                        SourceLocation NewTagLoc,
                                        const IdentifierInfo &Name) {
  // C++ [dcl.type.elab]p3:
  //   The class-key or enum keyword present in the
  //   elaborated-type-specifier shall agree in kind with the
  //   declaration to which the name in theelaborated-type-specifier
  //   refers. This rule also applies to the form of
  //   elaborated-type-specifier that declares a class-name or
  //   friend class since it can be construed as referring to the
  //   definition of the class. Thus, in any
  //   elaborated-type-specifier, the enum keyword shall be used to
  //   refer to an enumeration (7.2), the union class-keyshall be
  //   used to refer to a union (clause 9), and either the class or
  //   struct class-key shall be used to refer to a class (clause 9)
  //   declared using the class or struct class-key.
  TagDecl::TagKind OldTag = Previous->getTagKind();
  if (OldTag == NewTag)
    return true;
 
  if ((OldTag == TagDecl::TK_struct || OldTag == TagDecl::TK_class) &&
      (NewTag == TagDecl::TK_struct || NewTag == TagDecl::TK_class)) {
    // Warn about the struct/class tag mismatch.
    bool isTemplate = false;
    if (const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Previous))
      isTemplate = Record->getDescribedClassTemplate();

    Diag(NewTagLoc, diag::warn_struct_class_tag_mismatch)
      << (NewTag == TagDecl::TK_class)
      << isTemplate << &Name
      << CodeModificationHint::CreateReplacement(SourceRange(NewTagLoc),
                              OldTag == TagDecl::TK_class? "class" : "struct");
    Diag(Previous->getLocation(), diag::note_previous_use);
    return true;
  }
  return false;
}

/// ActOnTag - This is invoked when we see 'struct foo' or 'struct {'.  In the
/// former case, Name will be non-null.  In the later case, Name will be null.
/// TagSpec indicates what kind of tag this is. TUK indicates whether this is a
/// reference/declaration/definition of a tag.
Sema::DeclPtrTy Sema::ActOnTag(Scope *S, unsigned TagSpec, TagUseKind TUK,
                               SourceLocation KWLoc, const CXXScopeSpec &SS,
                               IdentifierInfo *Name, SourceLocation NameLoc,
                               AttributeList *Attr, AccessSpecifier AS,
                               MultiTemplateParamsArg TemplateParameterLists,
                               bool &OwnedDecl) {
  // If this is not a definition, it must have a name.
  assert((Name != 0 || TUK == TUK_Definition) &&
         "Nameless record must be a definition!");

  OwnedDecl = false;
  TagDecl::TagKind Kind;
  switch (TagSpec) {
  default: assert(0 && "Unknown tag type!");
  case DeclSpec::TST_struct: Kind = TagDecl::TK_struct; break;
  case DeclSpec::TST_union:  Kind = TagDecl::TK_union; break;
  case DeclSpec::TST_class:  Kind = TagDecl::TK_class; break;
  case DeclSpec::TST_enum:   Kind = TagDecl::TK_enum; break;
  }
  
  if (TUK != TUK_Reference) {
    if (TemplateParameterList *TemplateParams
          = MatchTemplateParametersToScopeSpecifier(KWLoc, SS,
                        (TemplateParameterList**)TemplateParameterLists.get(),
                                              TemplateParameterLists.size())) {
      if (TemplateParams->size() > 0) {
        // This is a declaration or definition of a class template (which may
        // be a member of another template).
        OwnedDecl = false;
        DeclResult Result = CheckClassTemplate(S, TagSpec, TUK, KWLoc,
                                               SS, Name, NameLoc, Attr,
                                               move(TemplateParameterLists),
                                               AS);
        return Result.get();
      } else {
        // FIXME: diagnose the extraneous 'template<>', once we recover
        // slightly better in ParseTemplate.cpp from bogus template
        // parameters.
      }
    }        
  }  
  
  DeclContext *SearchDC = CurContext;
  DeclContext *DC = CurContext;
  NamedDecl *PrevDecl = 0;

  bool Invalid = false;

  if (Name && SS.isNotEmpty()) {
    // We have a nested-name tag ('struct foo::bar').

    // Check for invalid 'foo::'.
    if (SS.isInvalid()) {
      Name = 0;
      goto CreateNewDecl;
    }

    if (RequireCompleteDeclContext(SS))
      return DeclPtrTy::make((Decl *)0);

    DC = computeDeclContext(SS, true);
    SearchDC = DC;
    // Look-up name inside 'foo::'.
    PrevDecl 
      = dyn_cast_or_null<TagDecl>(
               LookupQualifiedName(DC, Name, LookupTagName, true).getAsDecl());

    // A tag 'foo::bar' must already exist.
    if (PrevDecl == 0) {
      Diag(NameLoc, diag::err_not_tag_in_scope) << Name << SS.getRange();
      Name = 0;
      Invalid = true;
      goto CreateNewDecl;
    }
  } else if (Name) {
    // If this is a named struct, check to see if there was a previous forward
    // declaration or definition.
    // FIXME: We're looking into outer scopes here, even when we
    // shouldn't be. Doing so can result in ambiguities that we
    // shouldn't be diagnosing.
    LookupResult R = LookupName(S, Name, LookupTagName,
                                /*RedeclarationOnly=*/(TUK != TUK_Reference));
    if (R.isAmbiguous()) {
      DiagnoseAmbiguousLookup(R, Name, NameLoc);
      // FIXME: This is not best way to recover from case like:
      //
      // struct S s;
      //
      // causes needless "incomplete type" error later.
      Name = 0;
      PrevDecl = 0;
      Invalid = true;
    } else
      PrevDecl = R;

    if (!getLangOptions().CPlusPlus && TUK != TUK_Reference) {
      // FIXME: This makes sure that we ignore the contexts associated
      // with C structs, unions, and enums when looking for a matching
      // tag declaration or definition. See the similar lookup tweak
      // in Sema::LookupName; is there a better way to deal with this?
      while (isa<RecordDecl>(SearchDC) || isa<EnumDecl>(SearchDC))
        SearchDC = SearchDC->getParent();
    }
  }

  if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    DiagnoseTemplateParameterShadow(NameLoc, PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  }

  if (PrevDecl) {
    // Check whether the previous declaration is usable.
    (void)DiagnoseUseOfDecl(PrevDecl, NameLoc);
    
    if (TagDecl *PrevTagDecl = dyn_cast<TagDecl>(PrevDecl)) {
      // If this is a use of a previous tag, or if the tag is already declared
      // in the same scope (so that the definition/declaration completes or
      // rementions the tag), reuse the decl.
      if (TUK == TUK_Reference || TUK == TUK_Friend ||
          isDeclInScope(PrevDecl, SearchDC, S)) {
        // Make sure that this wasn't declared as an enum and now used as a
        // struct or something similar.
        if (!isAcceptableTagRedeclaration(PrevTagDecl, Kind, KWLoc, *Name)) {
          bool SafeToContinue 
            = (PrevTagDecl->getTagKind() != TagDecl::TK_enum &&
               Kind != TagDecl::TK_enum);
          if (SafeToContinue)
            Diag(KWLoc, diag::err_use_with_wrong_tag) 
              << Name
              << CodeModificationHint::CreateReplacement(SourceRange(KWLoc),
                                                  PrevTagDecl->getKindName());
          else
            Diag(KWLoc, diag::err_use_with_wrong_tag) << Name;
          Diag(PrevDecl->getLocation(), diag::note_previous_use);

          if (SafeToContinue) 
            Kind = PrevTagDecl->getTagKind();
          else {
            // Recover by making this an anonymous redefinition.
            Name = 0;
            PrevDecl = 0;
            Invalid = true;
          }
        }

        if (!Invalid) {
          // If this is a use, just return the declaration we found.

          // FIXME: In the future, return a variant or some other clue
          // for the consumer of this Decl to know it doesn't own it.
          // For our current ASTs this shouldn't be a problem, but will
          // need to be changed with DeclGroups.
          if (TUK == TUK_Reference)
            return DeclPtrTy::make(PrevDecl);

          // If this is a friend, make sure we create the new
          // declaration in the appropriate semantic context.
          if (TUK == TUK_Friend)
            SearchDC = PrevDecl->getDeclContext();

          // Diagnose attempts to redefine a tag.
          if (TUK == TUK_Definition) {
            if (TagDecl *Def = PrevTagDecl->getDefinition(Context)) {
              Diag(NameLoc, diag::err_redefinition) << Name;
              Diag(Def->getLocation(), diag::note_previous_definition);
              // If this is a redefinition, recover by making this
              // struct be anonymous, which will make any later
              // references get the previous definition.
              Name = 0;
              PrevDecl = 0;
              Invalid = true;
            } else {
              // If the type is currently being defined, complain
              // about a nested redefinition.
              TagType *Tag = cast<TagType>(Context.getTagDeclType(PrevTagDecl));
              if (Tag->isBeingDefined()) {
                Diag(NameLoc, diag::err_nested_redefinition) << Name;
                Diag(PrevTagDecl->getLocation(), 
                     diag::note_previous_definition);
                Name = 0;
                PrevDecl = 0;
                Invalid = true;
              }
            }

            // Okay, this is definition of a previously declared or referenced
            // tag PrevDecl. We're going to create a new Decl for it.
          }
        }
        // If we get here we have (another) forward declaration or we
        // have a definition.  Just create a new decl.

      } else {
        // If we get here, this is a definition of a new tag type in a nested
        // scope, e.g. "struct foo; void bar() { struct foo; }", just create a 
        // new decl/type.  We set PrevDecl to NULL so that the entities
        // have distinct types.
        PrevDecl = 0;
      }
      // If we get here, we're going to create a new Decl. If PrevDecl
      // is non-NULL, it's a definition of the tag declared by
      // PrevDecl. If it's NULL, we have a new definition.
    } else {
      // PrevDecl is a namespace, template, or anything else
      // that lives in the IDNS_Tag identifier namespace.
      if (isDeclInScope(PrevDecl, SearchDC, S)) {
        // The tag name clashes with a namespace name, issue an error and
        // recover by making this tag be anonymous.
        Diag(NameLoc, diag::err_redefinition_different_kind) << Name;
        Diag(PrevDecl->getLocation(), diag::note_previous_definition);
        Name = 0;
        PrevDecl = 0;
        Invalid = true;
      } else {
        // The existing declaration isn't relevant to us; we're in a
        // new scope, so clear out the previous declaration.
        PrevDecl = 0;
      }
    }
  } else if (TUK == TUK_Reference && SS.isEmpty() && Name &&
             (Kind != TagDecl::TK_enum || !getLangOptions().CPlusPlus)) {
    // C++ [basic.scope.pdecl]p5:
    //   -- for an elaborated-type-specifier of the form 
    //
    //          class-key identifier
    //
    //      if the elaborated-type-specifier is used in the
    //      decl-specifier-seq or parameter-declaration-clause of a
    //      function defined in namespace scope, the identifier is
    //      declared as a class-name in the namespace that contains
    //      the declaration; otherwise, except as a friend
    //      declaration, the identifier is declared in the smallest
    //      non-class, non-function-prototype scope that contains the
    //      declaration.
    //
    // C99 6.7.2.3p8 has a similar (but not identical!) provision for
    // C structs and unions.
    //
    // GNU C also supports this behavior as part of its incomplete
    // enum types extension, while GNU C++ does not.
    //
    // Find the context where we'll be declaring the tag.
    // FIXME: We would like to maintain the current DeclContext as the
    // lexical context, 
    while (SearchDC->isRecord())
      SearchDC = SearchDC->getParent();

    // Find the scope where we'll be declaring the tag.
    while (S->isClassScope() || 
           (getLangOptions().CPlusPlus && S->isFunctionPrototypeScope()) ||
           ((S->getFlags() & Scope::DeclScope) == 0) ||
           (S->getEntity() && 
            ((DeclContext *)S->getEntity())->isTransparentContext()))
      S = S->getParent();

  } else if (TUK == TUK_Friend && SS.isEmpty() && Name) {
    // C++ [namespace.memdef]p3:
    //   If a friend declaration in a non-local class first declares a
    //   class or function, the friend class or function is a member of
    //   the innermost enclosing namespace.
    while (!SearchDC->isFileContext())
      SearchDC = SearchDC->getParent();

    // The entity of a decl scope is a DeclContext; see PushDeclContext.
    while (S->getEntity() != SearchDC)
      S = S->getParent();
  }

CreateNewDecl:
  
  // If there is an identifier, use the location of the identifier as the
  // location of the decl, otherwise use the location of the struct/union
  // keyword.
  SourceLocation Loc = NameLoc.isValid() ? NameLoc : KWLoc;
  
  // Otherwise, create a new declaration. If there is a previous
  // declaration of the same entity, the two will be linked via
  // PrevDecl.
  TagDecl *New;

  if (Kind == TagDecl::TK_enum) {
    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // enum X { A, B, C } D;    D should chain to X.
    New = EnumDecl::Create(Context, SearchDC, Loc, Name, KWLoc,
                           cast_or_null<EnumDecl>(PrevDecl));
    // If this is an undefined enum, warn.
    if (TUK != TUK_Definition && !Invalid)  {
      unsigned DK = getLangOptions().CPlusPlus? diag::err_forward_ref_enum
                                              : diag::ext_forward_ref_enum;
      Diag(Loc, DK);
    }
  } else {
    // struct/union/class

    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // struct X { int A; } D;    D should chain to X.
    if (getLangOptions().CPlusPlus)
      // FIXME: Look for a way to use RecordDecl for simple structs.
      New = CXXRecordDecl::Create(Context, Kind, SearchDC, Loc, Name, KWLoc,
                                  cast_or_null<CXXRecordDecl>(PrevDecl));
    else
      New = RecordDecl::Create(Context, Kind, SearchDC, Loc, Name, KWLoc,
                               cast_or_null<RecordDecl>(PrevDecl));
  }

  if (Kind != TagDecl::TK_enum) {
    // Handle #pragma pack: if the #pragma pack stack has non-default
    // alignment, make up a packed attribute for this decl. These
    // attributes are checked when the ASTContext lays out the
    // structure.
    //
    // It is important for implementing the correct semantics that this
    // happen here (in act on tag decl). The #pragma pack stack is
    // maintained as a result of parser callbacks which can occur at
    // many points during the parsing of a struct declaration (because
    // the #pragma tokens are effectively skipped over during the
    // parsing of the struct).
    if (unsigned Alignment = getPragmaPackAlignment())
      New->addAttr(::new (Context) PragmaPackAttr(Alignment * 8));
  }

  if (getLangOptions().CPlusPlus && SS.isEmpty() && Name && !Invalid) {
    // C++ [dcl.typedef]p3:
    //   [...] Similarly, in a given scope, a class or enumeration
    //   shall not be declared with the same name as a typedef-name
    //   that is declared in that scope and refers to a type other
    //   than the class or enumeration itself.
    LookupResult Lookup = LookupName(S, Name, LookupOrdinaryName, true);
    TypedefDecl *PrevTypedef = 0;
    if (Lookup.getKind() == LookupResult::Found)
      PrevTypedef = dyn_cast<TypedefDecl>(Lookup.getAsDecl());

    if (PrevTypedef && isDeclInScope(PrevTypedef, SearchDC, S) &&
        Context.getCanonicalType(Context.getTypeDeclType(PrevTypedef)) !=
          Context.getCanonicalType(Context.getTypeDeclType(New))) {
      Diag(Loc, diag::err_tag_definition_of_typedef)
        << Context.getTypeDeclType(New)
        << PrevTypedef->getUnderlyingType();
      Diag(PrevTypedef->getLocation(), diag::note_previous_definition);
      Invalid = true;
    }
  }

  if (Invalid)
    New->setInvalidDecl();

  if (Attr)
    ProcessDeclAttributeList(S, New, Attr);

  // If we're declaring or defining a tag in function prototype scope
  // in C, note that this type can only be used within the function.
  if (Name && S->isFunctionPrototypeScope() && !getLangOptions().CPlusPlus)
    Diag(Loc, diag::warn_decl_in_param_list) << Context.getTagDeclType(New);

  // Set the lexical context. If the tag has a C++ scope specifier, the
  // lexical context will be different from the semantic context.
  New->setLexicalDeclContext(CurContext);

  // Set the access specifier.
  if (!Invalid && TUK != TUK_Friend)
    SetMemberAccessSpecifier(New, PrevDecl, AS);

  if (TUK == TUK_Definition)
    New->startDefinition();
  
  // If this has an identifier, add it to the scope stack.
  if (Name && TUK != TUK_Friend) {
    S = getNonFieldDeclScope(S);
    PushOnScopeChains(New, S);
  } else {
    CurContext->addDecl(New);
  }

  // If this is the C FILE type, notify the AST context.
  if (IdentifierInfo *II = New->getIdentifier())
    if (!New->isInvalidDecl() &&
        New->getDeclContext()->getLookupContext()->isTranslationUnit() &&
        II->isStr("FILE"))
      Context.setFILEDecl(New);
  
  OwnedDecl = true;
  return DeclPtrTy::make(New);
}

void Sema::ActOnTagStartDefinition(Scope *S, DeclPtrTy TagD) {
  AdjustDeclIfTemplate(TagD);
  TagDecl *Tag = cast<TagDecl>(TagD.getAs<Decl>());

  // Enter the tag context.
  PushDeclContext(S, Tag);

  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Tag)) {
    FieldCollector->StartClass();

    if (Record->getIdentifier()) {
      // C++ [class]p2: 
      //   [...] The class-name is also inserted into the scope of the
      //   class itself; this is known as the injected-class-name. For
      //   purposes of access checking, the injected-class-name is treated
      //   as if it were a public member name.
      CXXRecordDecl *InjectedClassName
        = CXXRecordDecl::Create(Context, Record->getTagKind(),
                                CurContext, Record->getLocation(),
                                Record->getIdentifier(),
                                Record->getTagKeywordLoc(),
                                Record);
      InjectedClassName->setImplicit();
      InjectedClassName->setAccess(AS_public);
      if (ClassTemplateDecl *Template = Record->getDescribedClassTemplate())
        InjectedClassName->setDescribedClassTemplate(Template);
      PushOnScopeChains(InjectedClassName, S);
      assert(InjectedClassName->isInjectedClassName() && 
             "Broken injected-class-name");
    }
  }
}

void Sema::ActOnTagFinishDefinition(Scope *S, DeclPtrTy TagD,
                                    SourceLocation RBraceLoc) {
  AdjustDeclIfTemplate(TagD);
  TagDecl *Tag = cast<TagDecl>(TagD.getAs<Decl>());
  Tag->setRBraceLoc(RBraceLoc);

  if (isa<CXXRecordDecl>(Tag))
    FieldCollector->FinishClass();

  // Exit this scope of this tag's definition.
  PopDeclContext();

  // Notify the consumer that we've defined a tag.
  Consumer.HandleTagDeclDefinition(Tag);
}

// Note that FieldName may be null for anonymous bitfields.
bool Sema::VerifyBitField(SourceLocation FieldLoc, IdentifierInfo *FieldName, 
                          QualType FieldTy, const Expr *BitWidth,
                          bool *ZeroWidth) {
  // Default to true; that shouldn't confuse checks for emptiness
  if (ZeroWidth)
    *ZeroWidth = true;

  // C99 6.7.2.1p4 - verify the field type.
  // C++ 9.6p3: A bit-field shall have integral or enumeration type.
  if (!FieldTy->isDependentType() && !FieldTy->isIntegralType()) {
    // Handle incomplete types with specific error.
    if (RequireCompleteType(FieldLoc, FieldTy, diag::err_field_incomplete))
      return true;
    if (FieldName)
      return Diag(FieldLoc, diag::err_not_integral_type_bitfield)
        << FieldName << FieldTy << BitWidth->getSourceRange();
    return Diag(FieldLoc, diag::err_not_integral_type_anon_bitfield)
      << FieldTy << BitWidth->getSourceRange();
  }

  // If the bit-width is type- or value-dependent, don't try to check
  // it now.
  if (BitWidth->isValueDependent() || BitWidth->isTypeDependent())
    return false;

  llvm::APSInt Value;
  if (VerifyIntegerConstantExpression(BitWidth, &Value))
    return true;

  if (Value != 0 && ZeroWidth)
    *ZeroWidth = false;

  // Zero-width bitfield is ok for anonymous field.
  if (Value == 0 && FieldName)
    return Diag(FieldLoc, diag::err_bitfield_has_zero_width) << FieldName;
  
  if (Value.isSigned() && Value.isNegative()) {
    if (FieldName)
      return Diag(FieldLoc, diag::err_bitfield_has_negative_width) 
               << FieldName << Value.toString(10);
    return Diag(FieldLoc, diag::err_anon_bitfield_has_negative_width)
      << Value.toString(10);
  }

  if (!FieldTy->isDependentType()) {
    uint64_t TypeSize = Context.getTypeSize(FieldTy);
    if (Value.getZExtValue() > TypeSize) {
      if (FieldName)
        return Diag(FieldLoc, diag::err_bitfield_width_exceeds_type_size)
          << FieldName << (unsigned)TypeSize;
      return Diag(FieldLoc, diag::err_anon_bitfield_width_exceeds_type_size)
        << (unsigned)TypeSize;
    }
  }

  return false;
}

/// ActOnField - Each field of a struct/union/class is passed into this in order
/// to create a FieldDecl object for it.
Sema::DeclPtrTy Sema::ActOnField(Scope *S, DeclPtrTy TagD,
                                 SourceLocation DeclStart, 
                                 Declarator &D, ExprTy *BitfieldWidth) {
  FieldDecl *Res = HandleField(S, cast_or_null<RecordDecl>(TagD.getAs<Decl>()),
                               DeclStart, D, static_cast<Expr*>(BitfieldWidth),
                               AS_public);
  return DeclPtrTy::make(Res);
}

/// HandleField - Analyze a field of a C struct or a C++ data member.
///
FieldDecl *Sema::HandleField(Scope *S, RecordDecl *Record,
                             SourceLocation DeclStart,
                             Declarator &D, Expr *BitWidth,
                             AccessSpecifier AS) {
  IdentifierInfo *II = D.getIdentifier();
  SourceLocation Loc = DeclStart;
  if (II) Loc = D.getIdentifierLoc();
 
  DeclaratorInfo *DInfo = 0;
  QualType T = GetTypeForDeclarator(D, S, &DInfo);
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);

  DiagnoseFunctionSpecifiers(D);

  if (D.getDeclSpec().isThreadSpecified())
    Diag(D.getDeclSpec().getThreadSpecLoc(), diag::err_invalid_thread);

  NamedDecl *PrevDecl = LookupName(S, II, LookupMemberName, true);

  if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    DiagnoseTemplateParameterShadow(D.getIdentifierLoc(), PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  }

  if (PrevDecl && !isDeclInScope(PrevDecl, Record, S))
    PrevDecl = 0;

  bool Mutable
    = (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_mutable);
  SourceLocation TSSL = D.getSourceRange().getBegin();
  FieldDecl *NewFD
    = CheckFieldDecl(II, T, DInfo, Record, Loc, Mutable, BitWidth, TSSL,
                     AS, PrevDecl, &D);
  if (NewFD->isInvalidDecl() && PrevDecl) {
    // Don't introduce NewFD into scope; there's already something
    // with the same name in the same scope.
  } else if (II) {
    PushOnScopeChains(NewFD, S);
  } else
    Record->addDecl(NewFD);

  return NewFD;
}

/// \brief Build a new FieldDecl and check its well-formedness.
///
/// This routine builds a new FieldDecl given the fields name, type,
/// record, etc. \p PrevDecl should refer to any previous declaration
/// with the same name and in the same scope as the field to be
/// created.
///
/// \returns a new FieldDecl.
///
/// \todo The Declarator argument is a hack. It will be removed once 
FieldDecl *Sema::CheckFieldDecl(DeclarationName Name, QualType T,
                                DeclaratorInfo *DInfo, 
                                RecordDecl *Record, SourceLocation Loc,
                                bool Mutable, Expr *BitWidth, 
                                SourceLocation TSSL,
                                AccessSpecifier AS, NamedDecl *PrevDecl,
                                Declarator *D) {
  IdentifierInfo *II = Name.getAsIdentifierInfo();
  bool InvalidDecl = false;
  if (D) InvalidDecl = D->isInvalidType();

  // If we receive a broken type, recover by assuming 'int' and
  // marking this declaration as invalid.
  if (T.isNull()) {
    InvalidDecl = true;
    T = Context.IntTy;
  }

  // C99 6.7.2.1p8: A member of a structure or union may have any type other
  // than a variably modified type.
  if (T->isVariablyModifiedType()) {
    bool SizeIsNegative;
    QualType FixedTy = TryToFixInvalidVariablyModifiedType(T, Context,
                                                           SizeIsNegative);
    if (!FixedTy.isNull()) {
      Diag(Loc, diag::warn_illegal_constant_array_size);
      T = FixedTy;
    } else {
      if (SizeIsNegative)
        Diag(Loc, diag::err_typecheck_negative_array_size);
      else
        Diag(Loc, diag::err_typecheck_field_variable_size);
      InvalidDecl = true;
    }
  }
  
  // Fields can not have abstract class types
  if (RequireNonAbstractType(Loc, T, diag::err_abstract_type_in_decl, 
                             AbstractFieldType))
    InvalidDecl = true;
  
  bool ZeroWidth = false;
  // If this is declared as a bit-field, check the bit-field.
  if (BitWidth && VerifyBitField(Loc, II, T, BitWidth, &ZeroWidth)) {
    InvalidDecl = true;
    DeleteExpr(BitWidth);
    BitWidth = 0;
    ZeroWidth = false;
  }
  
  FieldDecl *NewFD = FieldDecl::Create(Context, Record, Loc, II, T, DInfo,
                                       BitWidth, Mutable);
  if (InvalidDecl)
    NewFD->setInvalidDecl();

  if (PrevDecl && !isa<TagDecl>(PrevDecl)) {
    Diag(Loc, diag::err_duplicate_member) << II;
    Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
    NewFD->setInvalidDecl();
  }

  if (getLangOptions().CPlusPlus) {
    QualType EltTy = Context.getBaseElementType(T);

    CXXRecordDecl* CXXRecord = cast<CXXRecordDecl>(Record);

    if (!T->isPODType())
      CXXRecord->setPOD(false);
    if (!ZeroWidth)
      CXXRecord->setEmpty(false);

    if (const RecordType *RT = EltTy->getAs<RecordType>()) {
      CXXRecordDecl* RDecl = cast<CXXRecordDecl>(RT->getDecl());

      if (!RDecl->hasTrivialConstructor())
        CXXRecord->setHasTrivialConstructor(false);
      if (!RDecl->hasTrivialCopyConstructor())
        CXXRecord->setHasTrivialCopyConstructor(false);
      if (!RDecl->hasTrivialCopyAssignment())
        CXXRecord->setHasTrivialCopyAssignment(false);
      if (!RDecl->hasTrivialDestructor())
        CXXRecord->setHasTrivialDestructor(false);

      // C++ 9.5p1: An object of a class with a non-trivial
      // constructor, a non-trivial copy constructor, a non-trivial
      // destructor, or a non-trivial copy assignment operator
      // cannot be a member of a union, nor can an array of such
      // objects.
      // TODO: C++0x alters this restriction significantly.
      if (Record->isUnion()) {
        // We check for copy constructors before constructors
        // because otherwise we'll never get complaints about
        // copy constructors.

        const CXXSpecialMember invalid = (CXXSpecialMember) -1;

        CXXSpecialMember member;
        if (!RDecl->hasTrivialCopyConstructor())
          member = CXXCopyConstructor;
        else if (!RDecl->hasTrivialConstructor())
          member = CXXDefaultConstructor;
        else if (!RDecl->hasTrivialCopyAssignment())
          member = CXXCopyAssignment;
        else if (!RDecl->hasTrivialDestructor())
          member = CXXDestructor;
        else
          member = invalid;

        if (member != invalid) {
          Diag(Loc, diag::err_illegal_union_member) << Name << member;
          DiagnoseNontrivial(RT, member);
          NewFD->setInvalidDecl();
        }
      }
    }
  }

  // FIXME: We need to pass in the attributes given an AST
  // representation, not a parser representation.
  if (D)
    // FIXME: What to pass instead of TUScope?
    ProcessDeclAttributes(TUScope, NewFD, *D);

  if (T.isObjCGCWeak())
    Diag(Loc, diag::warn_attribute_weak_on_field);

  NewFD->setAccess(AS);

  // C++ [dcl.init.aggr]p1:
  //   An aggregate is an array or a class (clause 9) with [...] no
  //   private or protected non-static data members (clause 11).
  // A POD must be an aggregate.
  if (getLangOptions().CPlusPlus &&
      (AS == AS_private || AS == AS_protected)) {
    CXXRecordDecl *CXXRecord = cast<CXXRecordDecl>(Record);
    CXXRecord->setAggregate(false);
    CXXRecord->setPOD(false);
  }

  return NewFD;
}

/// DiagnoseNontrivial - Given that a class has a non-trivial
/// special member, figure out why.
void Sema::DiagnoseNontrivial(const RecordType* T, CXXSpecialMember member) {
  QualType QT(T, 0U);
  CXXRecordDecl* RD = cast<CXXRecordDecl>(T->getDecl());

  // Check whether the member was user-declared.
  switch (member) {
  case CXXDefaultConstructor:
    if (RD->hasUserDeclaredConstructor()) {
      typedef CXXRecordDecl::ctor_iterator ctor_iter;
      for (ctor_iter ci = RD->ctor_begin(), ce = RD->ctor_end(); ci != ce; ++ci)
        if (!ci->isImplicitlyDefined(Context)) {
          SourceLocation CtorLoc = ci->getLocation();
          Diag(CtorLoc, diag::note_nontrivial_user_defined) << QT << member;
          return;
        }

      assert(0 && "found no user-declared constructors");
      return;
    }
    break;

  case CXXCopyConstructor:
    if (RD->hasUserDeclaredCopyConstructor()) {
      SourceLocation CtorLoc =
        RD->getCopyConstructor(Context, 0)->getLocation();
      Diag(CtorLoc, diag::note_nontrivial_user_defined) << QT << member;
      return;
    }
    break;

  case CXXCopyAssignment:
    if (RD->hasUserDeclaredCopyAssignment()) {
      // FIXME: this should use the location of the copy
      // assignment, not the type.
      SourceLocation TyLoc = RD->getSourceRange().getBegin();
      Diag(TyLoc, diag::note_nontrivial_user_defined) << QT << member;
      return;
    }
    break;

  case CXXDestructor:
    if (RD->hasUserDeclaredDestructor()) {
      SourceLocation DtorLoc = RD->getDestructor(Context)->getLocation();
      Diag(DtorLoc, diag::note_nontrivial_user_defined) << QT << member;
      return;
    }
    break;
  }

  typedef CXXRecordDecl::base_class_iterator base_iter;

  // Virtual bases and members inhibit trivial copying/construction,
  // but not trivial destruction.
  if (member != CXXDestructor) {
    // Check for virtual bases.  vbases includes indirect virtual bases,
    // so we just iterate through the direct bases.
    for (base_iter bi = RD->bases_begin(), be = RD->bases_end(); bi != be; ++bi)
      if (bi->isVirtual()) {
        SourceLocation BaseLoc = bi->getSourceRange().getBegin();
        Diag(BaseLoc, diag::note_nontrivial_has_virtual) << QT << 1;
        return;
      }

    // Check for virtual methods.
    typedef CXXRecordDecl::method_iterator meth_iter;
    for (meth_iter mi = RD->method_begin(), me = RD->method_end(); mi != me;
         ++mi) {
      if (mi->isVirtual()) {
        SourceLocation MLoc = mi->getSourceRange().getBegin();
        Diag(MLoc, diag::note_nontrivial_has_virtual) << QT << 0;
        return;
      }
    }
  }
  
  bool (CXXRecordDecl::*hasTrivial)() const;
  switch (member) {
  case CXXDefaultConstructor:
    hasTrivial = &CXXRecordDecl::hasTrivialConstructor; break;
  case CXXCopyConstructor:
    hasTrivial = &CXXRecordDecl::hasTrivialCopyConstructor; break;
  case CXXCopyAssignment:
    hasTrivial = &CXXRecordDecl::hasTrivialCopyAssignment; break;
  case CXXDestructor:
    hasTrivial = &CXXRecordDecl::hasTrivialDestructor; break;
  default:
    assert(0 && "unexpected special member"); return;
  }

  // Check for nontrivial bases (and recurse).
  for (base_iter bi = RD->bases_begin(), be = RD->bases_end(); bi != be; ++bi) {
    const RecordType *BaseRT = bi->getType()->getAs<RecordType>();
    assert(BaseRT);
    CXXRecordDecl *BaseRecTy = cast<CXXRecordDecl>(BaseRT->getDecl());
    if (!(BaseRecTy->*hasTrivial)()) {
      SourceLocation BaseLoc = bi->getSourceRange().getBegin();
      Diag(BaseLoc, diag::note_nontrivial_has_nontrivial) << QT << 1 << member;
      DiagnoseNontrivial(BaseRT, member);
      return;
    }
  }
  
  // Check for nontrivial members (and recurse).
  typedef RecordDecl::field_iterator field_iter;
  for (field_iter fi = RD->field_begin(), fe = RD->field_end(); fi != fe;
       ++fi) {
    QualType EltTy = Context.getBaseElementType((*fi)->getType());
    if (const RecordType *EltRT = EltTy->getAs<RecordType>()) {
      CXXRecordDecl* EltRD = cast<CXXRecordDecl>(EltRT->getDecl());

      if (!(EltRD->*hasTrivial)()) {
        SourceLocation FLoc = (*fi)->getLocation();
        Diag(FLoc, diag::note_nontrivial_has_nontrivial) << QT << 0 << member;
        DiagnoseNontrivial(EltRT, member);
        return;
      }
    }
  }

  assert(0 && "found no explanation for non-trivial member");
}

/// TranslateIvarVisibility - Translate visibility from a token ID to an 
///  AST enum value.
static ObjCIvarDecl::AccessControl
TranslateIvarVisibility(tok::ObjCKeywordKind ivarVisibility) {
  switch (ivarVisibility) {
  default: assert(0 && "Unknown visitibility kind");
  case tok::objc_private: return ObjCIvarDecl::Private;
  case tok::objc_public: return ObjCIvarDecl::Public;
  case tok::objc_protected: return ObjCIvarDecl::Protected;
  case tok::objc_package: return ObjCIvarDecl::Package;
  }
}

/// ActOnIvar - Each ivar field of an objective-c class is passed into this 
/// in order to create an IvarDecl object for it.
Sema::DeclPtrTy Sema::ActOnIvar(Scope *S,
                                SourceLocation DeclStart, 
                                DeclPtrTy IntfDecl,
                                Declarator &D, ExprTy *BitfieldWidth,
                                tok::ObjCKeywordKind Visibility) {
  
  IdentifierInfo *II = D.getIdentifier();
  Expr *BitWidth = (Expr*)BitfieldWidth;
  SourceLocation Loc = DeclStart;
  if (II) Loc = D.getIdentifierLoc();
  
  // FIXME: Unnamed fields can be handled in various different ways, for
  // example, unnamed unions inject all members into the struct namespace!
  
  DeclaratorInfo *DInfo = 0;
  QualType T = GetTypeForDeclarator(D, S, &DInfo);
  
  if (BitWidth) {
    // 6.7.2.1p3, 6.7.2.1p4
    if (VerifyBitField(Loc, II, T, BitWidth)) {
      D.setInvalidType();
      DeleteExpr(BitWidth);
      BitWidth = 0;
    }
  } else {
    // Not a bitfield.
    
    // validate II.
    
  }
  
  // C99 6.7.2.1p8: A member of a structure or union may have any type other
  // than a variably modified type.
  if (T->isVariablyModifiedType()) {
    Diag(Loc, diag::err_typecheck_ivar_variable_size);
    D.setInvalidType();
  }
  
  // Get the visibility (access control) for this ivar.
  ObjCIvarDecl::AccessControl ac = 
    Visibility != tok::objc_not_keyword ? TranslateIvarVisibility(Visibility)
                                        : ObjCIvarDecl::None;
  // Must set ivar's DeclContext to its enclosing interface.
  Decl *EnclosingDecl = IntfDecl.getAs<Decl>();
  DeclContext *EnclosingContext;
  if (ObjCImplementationDecl *IMPDecl = 
      dyn_cast<ObjCImplementationDecl>(EnclosingDecl)) {
    // Case of ivar declared in an implementation. Context is that of its class.
    ObjCInterfaceDecl* IDecl = IMPDecl->getClassInterface();
    assert(IDecl && "No class- ActOnIvar");
    EnclosingContext = cast_or_null<DeclContext>(IDecl);
  } else
    EnclosingContext = dyn_cast<DeclContext>(EnclosingDecl);
  assert(EnclosingContext && "null DeclContext for ivar - ActOnIvar");
  
  // Construct the decl.
  ObjCIvarDecl *NewID = ObjCIvarDecl::Create(Context, 
                                             EnclosingContext, Loc, II, T,
                                             DInfo, ac, (Expr *)BitfieldWidth);
  
  if (II) {
    NamedDecl *PrevDecl = LookupName(S, II, LookupMemberName, true);
    if (PrevDecl && isDeclInScope(PrevDecl, EnclosingContext, S)
        && !isa<TagDecl>(PrevDecl)) {
      Diag(Loc, diag::err_duplicate_member) << II;
      Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
      NewID->setInvalidDecl();
    }
  }

  // Process attributes attached to the ivar.
  ProcessDeclAttributes(S, NewID, D);
  
  if (D.isInvalidType())
    NewID->setInvalidDecl();

  if (II) {
    // FIXME: When interfaces are DeclContexts, we'll need to add
    // these to the interface.
    S->AddDecl(DeclPtrTy::make(NewID));
    IdResolver.AddDecl(NewID);
  }

  return DeclPtrTy::make(NewID);
}

void Sema::ActOnFields(Scope* S,
                       SourceLocation RecLoc, DeclPtrTy RecDecl,
                       DeclPtrTy *Fields, unsigned NumFields,
                       SourceLocation LBrac, SourceLocation RBrac,
                       AttributeList *Attr) {
  Decl *EnclosingDecl = RecDecl.getAs<Decl>();
  assert(EnclosingDecl && "missing record or interface decl");
  
  // If the decl this is being inserted into is invalid, then it may be a
  // redeclaration or some other bogus case.  Don't try to add fields to it.
  if (EnclosingDecl->isInvalidDecl()) {
    // FIXME: Deallocate fields?
    return;
  }

  
  // Verify that all the fields are okay.
  unsigned NumNamedMembers = 0;
  llvm::SmallVector<FieldDecl*, 32> RecFields;

  RecordDecl *Record = dyn_cast<RecordDecl>(EnclosingDecl);
  for (unsigned i = 0; i != NumFields; ++i) {
    FieldDecl *FD = cast<FieldDecl>(Fields[i].getAs<Decl>());
    
    // Get the type for the field.
    Type *FDTy = FD->getType().getTypePtr();

    if (!FD->isAnonymousStructOrUnion()) {
      // Remember all fields written by the user.
      RecFields.push_back(FD);
    }
    
    // If the field is already invalid for some reason, don't emit more
    // diagnostics about it.
    if (FD->isInvalidDecl())
      continue;
      
    // C99 6.7.2.1p2:
    //   A structure or union shall not contain a member with
    //   incomplete or function type (hence, a structure shall not
    //   contain an instance of itself, but may contain a pointer to
    //   an instance of itself), except that the last member of a
    //   structure with more than one named member may have incomplete
    //   array type; such a structure (and any union containing,
    //   possibly recursively, a member that is such a structure)
    //   shall not be a member of a structure or an element of an
    //   array.
    if (FDTy->isFunctionType()) {
      // Field declared as a function.
      Diag(FD->getLocation(), diag::err_field_declared_as_function)
        << FD->getDeclName();
      FD->setInvalidDecl();
      EnclosingDecl->setInvalidDecl();
      continue;
    } else if (FDTy->isIncompleteArrayType() && i == NumFields - 1 &&
               Record && Record->isStruct()) {
      // Flexible array member.
      if (NumNamedMembers < 1) {
        Diag(FD->getLocation(), diag::err_flexible_array_empty_struct)
          << FD->getDeclName();
        FD->setInvalidDecl();
        EnclosingDecl->setInvalidDecl();
        continue;
      }
      // Okay, we have a legal flexible array member at the end of the struct.
      if (Record)
        Record->setHasFlexibleArrayMember(true);
    } else if (!FDTy->isDependentType() &&
               RequireCompleteType(FD->getLocation(), FD->getType(), 
                                   diag::err_field_incomplete)) {
      // Incomplete type
      FD->setInvalidDecl();
      EnclosingDecl->setInvalidDecl();
      continue;
    } else if (const RecordType *FDTTy = FDTy->getAs<RecordType>()) {
      if (FDTTy->getDecl()->hasFlexibleArrayMember()) {
        // If this is a member of a union, then entire union becomes "flexible".
        if (Record && Record->isUnion()) {
          Record->setHasFlexibleArrayMember(true);
        } else {
          // If this is a struct/class and this is not the last element, reject
          // it.  Note that GCC supports variable sized arrays in the middle of
          // structures.
          if (i != NumFields-1)
            Diag(FD->getLocation(), diag::ext_variable_sized_type_in_struct)
              << FD->getDeclName() << FD->getType();
          else {
            // We support flexible arrays at the end of structs in
            // other structs as an extension.
            Diag(FD->getLocation(), diag::ext_flexible_array_in_struct)
              << FD->getDeclName();
            if (Record)
              Record->setHasFlexibleArrayMember(true);
          }
        }
      }
      if (Record && FDTTy->getDecl()->hasObjectMember())
        Record->setHasObjectMember(true);
    } else if (FDTy->isObjCInterfaceType()) {
      /// A field cannot be an Objective-c object
      Diag(FD->getLocation(), diag::err_statically_allocated_object);
      FD->setInvalidDecl();
      EnclosingDecl->setInvalidDecl();
      continue;
    } else if (getLangOptions().ObjC1 &&
               getLangOptions().getGCMode() != LangOptions::NonGC &&
               Record &&
               (FD->getType()->isObjCObjectPointerType() ||
                FD->getType().isObjCGCStrong()))
      Record->setHasObjectMember(true);
    // Keep track of the number of named members.
    if (FD->getIdentifier())
      ++NumNamedMembers;
  }

  // Okay, we successfully defined 'Record'.
  if (Record) {
    Record->completeDefinition(Context);
  } else {
    ObjCIvarDecl **ClsFields =
      reinterpret_cast<ObjCIvarDecl**>(RecFields.data());
    if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(EnclosingDecl)) {
      ID->setIVarList(ClsFields, RecFields.size(), Context);
      ID->setLocEnd(RBrac);
      // Add ivar's to class's DeclContext.
      for (unsigned i = 0, e = RecFields.size(); i != e; ++i) {
        ClsFields[i]->setLexicalDeclContext(ID);
        ID->addDecl(ClsFields[i]);
      }
      // Must enforce the rule that ivars in the base classes may not be
      // duplicates.
      if (ID->getSuperClass()) {
        for (ObjCInterfaceDecl::ivar_iterator IVI = ID->ivar_begin(), 
             IVE = ID->ivar_end(); IVI != IVE; ++IVI) {
          ObjCIvarDecl* Ivar = (*IVI);

          if (IdentifierInfo *II = Ivar->getIdentifier()) {
            ObjCIvarDecl* prevIvar =
              ID->getSuperClass()->lookupInstanceVariable(II);
            if (prevIvar) {
              Diag(Ivar->getLocation(), diag::err_duplicate_member) << II;
              Diag(prevIvar->getLocation(), diag::note_previous_declaration);
            }
          }
        }
      }
    } else if (ObjCImplementationDecl *IMPDecl = 
                  dyn_cast<ObjCImplementationDecl>(EnclosingDecl)) {
      assert(IMPDecl && "ActOnFields - missing ObjCImplementationDecl");
      for (unsigned I = 0, N = RecFields.size(); I != N; ++I)
        // Ivar declared in @implementation never belongs to the implementation.
        // Only it is in implementation's lexical context.
        ClsFields[I]->setLexicalDeclContext(IMPDecl);
      CheckImplementationIvars(IMPDecl, ClsFields, RecFields.size(), RBrac);
    }
  }

  if (Attr)
    ProcessDeclAttributeList(S, Record, Attr);
}

EnumConstantDecl *Sema::CheckEnumConstant(EnumDecl *Enum,
                                          EnumConstantDecl *LastEnumConst,
                                          SourceLocation IdLoc,
                                          IdentifierInfo *Id,
                                          ExprArg val) {
  Expr *Val = (Expr *)val.get();

  llvm::APSInt EnumVal(32);
  QualType EltTy;
  if (Val && !Val->isTypeDependent()) {
    // Make sure to promote the operand type to int.
    UsualUnaryConversions(Val);
    if (Val != val.get()) {
      val.release();
      val = Val;
    }

    // C99 6.7.2.2p2: Make sure we have an integer constant expression.
    SourceLocation ExpLoc;
    if (!Val->isValueDependent() &&
        VerifyIntegerConstantExpression(Val, &EnumVal)) {
      Val = 0;
    } else {
      EltTy = Val->getType();
    }
  }
  
  if (!Val) {
    if (LastEnumConst) {
      // Assign the last value + 1.
      EnumVal = LastEnumConst->getInitVal();
      ++EnumVal;

      // Check for overflow on increment.
      if (EnumVal < LastEnumConst->getInitVal())
        Diag(IdLoc, diag::warn_enum_value_overflow);
      
      EltTy = LastEnumConst->getType();
    } else {
      // First value, set to zero.
      EltTy = Context.IntTy;
      EnumVal.zextOrTrunc(static_cast<uint32_t>(Context.getTypeSize(EltTy)));
    }
  }
  
  val.release();
  return EnumConstantDecl::Create(Context, Enum, IdLoc, Id, EltTy,
                                  Val, EnumVal);  
}


Sema::DeclPtrTy Sema::ActOnEnumConstant(Scope *S, DeclPtrTy theEnumDecl,
                                        DeclPtrTy lastEnumConst,
                                        SourceLocation IdLoc,
                                        IdentifierInfo *Id,
                                        SourceLocation EqualLoc, ExprTy *val) {
  EnumDecl *TheEnumDecl = cast<EnumDecl>(theEnumDecl.getAs<Decl>());
  EnumConstantDecl *LastEnumConst =
    cast_or_null<EnumConstantDecl>(lastEnumConst.getAs<Decl>());
  Expr *Val = static_cast<Expr*>(val);

  // The scope passed in may not be a decl scope.  Zip up the scope tree until
  // we find one that is.
  S = getNonFieldDeclScope(S);
  
  // Verify that there isn't already something declared with this name in this
  // scope.
  NamedDecl *PrevDecl = LookupName(S, Id, LookupOrdinaryName);
  if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    DiagnoseTemplateParameterShadow(IdLoc, PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  }

  if (PrevDecl) {
    // When in C++, we may get a TagDecl with the same name; in this case the
    // enum constant will 'hide' the tag.
    assert((getLangOptions().CPlusPlus || !isa<TagDecl>(PrevDecl)) &&
           "Received TagDecl when not in C++!");
    if (!isa<TagDecl>(PrevDecl) && isDeclInScope(PrevDecl, CurContext, S)) {
      if (isa<EnumConstantDecl>(PrevDecl))
        Diag(IdLoc, diag::err_redefinition_of_enumerator) << Id;
      else
        Diag(IdLoc, diag::err_redefinition) << Id;
      Diag(PrevDecl->getLocation(), diag::note_previous_definition);
      if (Val) Val->Destroy(Context);
      return DeclPtrTy();
    }
  }

  EnumConstantDecl *New = CheckEnumConstant(TheEnumDecl, LastEnumConst,
                                            IdLoc, Id, Owned(Val));

  // Register this decl in the current scope stack.
  if (New)
    PushOnScopeChains(New, S);

  return DeclPtrTy::make(New);
}

void Sema::ActOnEnumBody(SourceLocation EnumLoc, SourceLocation LBraceLoc,
                         SourceLocation RBraceLoc, DeclPtrTy EnumDeclX,
                         DeclPtrTy *Elements, unsigned NumElements,
                         Scope *S, AttributeList *Attr) {
  EnumDecl *Enum = cast<EnumDecl>(EnumDeclX.getAs<Decl>());
  QualType EnumType = Context.getTypeDeclType(Enum);

  if (Attr)
    ProcessDeclAttributeList(S, Enum, Attr);
  
  // TODO: If the result value doesn't fit in an int, it must be a long or long
  // long value.  ISO C does not support this, but GCC does as an extension,
  // emit a warning.
  unsigned IntWidth = Context.Target.getIntWidth();
  unsigned CharWidth = Context.Target.getCharWidth();
  unsigned ShortWidth = Context.Target.getShortWidth();
  
  // Verify that all the values are okay, compute the size of the values, and
  // reverse the list.
  unsigned NumNegativeBits = 0;
  unsigned NumPositiveBits = 0;
  
  // Keep track of whether all elements have type int.
  bool AllElementsInt = true;
  
  for (unsigned i = 0; i != NumElements; ++i) {
    EnumConstantDecl *ECD =
      cast_or_null<EnumConstantDecl>(Elements[i].getAs<Decl>());
    if (!ECD) continue;  // Already issued a diagnostic.
    
    // If the enum value doesn't fit in an int, emit an extension warning.
    const llvm::APSInt &InitVal = ECD->getInitVal();
    assert(InitVal.getBitWidth() >= IntWidth &&
           "Should have promoted value to int");
    if (InitVal.getBitWidth() > IntWidth) {
      llvm::APSInt V(InitVal);
      V.trunc(IntWidth);
      V.extend(InitVal.getBitWidth());
      if (V != InitVal)
        Diag(ECD->getLocation(), diag::ext_enum_value_not_int)
          << InitVal.toString(10);
    }
    
    // Keep track of the size of positive and negative values.
    if (InitVal.isUnsigned() || InitVal.isNonNegative())
      NumPositiveBits = std::max(NumPositiveBits,
                                 (unsigned)InitVal.getActiveBits());
    else
      NumNegativeBits = std::max(NumNegativeBits,
                                 (unsigned)InitVal.getMinSignedBits());

    // Keep track of whether every enum element has type int (very commmon).
    if (AllElementsInt)
      AllElementsInt = ECD->getType() == Context.IntTy; 
  }
  
  // Figure out the type that should be used for this enum.
  // FIXME: Support -fshort-enums.
  QualType BestType;
  unsigned BestWidth;

  bool Packed = Enum->getAttr<PackedAttr>() ? true : false;

  if (NumNegativeBits) {
    // If there is a negative value, figure out the smallest integer type (of 
    // int/long/longlong) that fits.
    // If it's packed, check also if it fits a char or a short.
    if (Packed && NumNegativeBits <= CharWidth && NumPositiveBits < CharWidth) {
        BestType = Context.SignedCharTy;
        BestWidth = CharWidth;
    } else if (Packed && NumNegativeBits <= ShortWidth && 
               NumPositiveBits < ShortWidth) {
        BestType = Context.ShortTy;
        BestWidth = ShortWidth;
    }
    else if (NumNegativeBits <= IntWidth && NumPositiveBits < IntWidth) {
      BestType = Context.IntTy;
      BestWidth = IntWidth;
    } else {
      BestWidth = Context.Target.getLongWidth();
      
      if (NumNegativeBits <= BestWidth && NumPositiveBits < BestWidth)
        BestType = Context.LongTy;
      else {
        BestWidth = Context.Target.getLongLongWidth();
        
        if (NumNegativeBits > BestWidth || NumPositiveBits >= BestWidth)
          Diag(Enum->getLocation(), diag::warn_enum_too_large);
        BestType = Context.LongLongTy;
      }
    }
  } else {
    // If there is no negative value, figure out which of uint, ulong, ulonglong
    // fits.
    // If it's packed, check also if it fits a char or a short.
    if (Packed && NumPositiveBits <= CharWidth) {
        BestType = Context.UnsignedCharTy;
        BestWidth = CharWidth;
    } else if (Packed && NumPositiveBits <= ShortWidth) {
        BestType = Context.UnsignedShortTy;
        BestWidth = ShortWidth;
    }
    else if (NumPositiveBits <= IntWidth) {
      BestType = Context.UnsignedIntTy;
      BestWidth = IntWidth;
    } else if (NumPositiveBits <=
               (BestWidth = Context.Target.getLongWidth())) {
      BestType = Context.UnsignedLongTy;
    } else {
      BestWidth = Context.Target.getLongLongWidth();
      assert(NumPositiveBits <= BestWidth &&
             "How could an initializer get larger than ULL?");
      BestType = Context.UnsignedLongLongTy;
    }
  }
  
  // Loop over all of the enumerator constants, changing their types to match
  // the type of the enum if needed.
  for (unsigned i = 0; i != NumElements; ++i) {
    EnumConstantDecl *ECD =
      cast_or_null<EnumConstantDecl>(Elements[i].getAs<Decl>());
    if (!ECD) continue;  // Already issued a diagnostic.

    // Standard C says the enumerators have int type, but we allow, as an
    // extension, the enumerators to be larger than int size.  If each
    // enumerator value fits in an int, type it as an int, otherwise type it the
    // same as the enumerator decl itself.  This means that in "enum { X = 1U }"
    // that X has type 'int', not 'unsigned'.
    if (ECD->getType() == Context.IntTy) {
      // Make sure the init value is signed.
      llvm::APSInt IV = ECD->getInitVal();
      IV.setIsSigned(true);
      ECD->setInitVal(IV);

      if (getLangOptions().CPlusPlus)
        // C++ [dcl.enum]p4: Following the closing brace of an
        // enum-specifier, each enumerator has the type of its
        // enumeration. 
        ECD->setType(EnumType);
      continue;  // Already int type.
    }

    // Determine whether the value fits into an int.
    llvm::APSInt InitVal = ECD->getInitVal();
    bool FitsInInt;
    if (InitVal.isUnsigned() || !InitVal.isNegative())
      FitsInInt = InitVal.getActiveBits() < IntWidth;
    else
      FitsInInt = InitVal.getMinSignedBits() <= IntWidth;

    // If it fits into an integer type, force it.  Otherwise force it to match
    // the enum decl type.
    QualType NewTy;
    unsigned NewWidth;
    bool NewSign;
    if (FitsInInt) {
      NewTy = Context.IntTy;
      NewWidth = IntWidth;
      NewSign = true;
    } else if (ECD->getType() == BestType) {
      // Already the right type!
      if (getLangOptions().CPlusPlus)
        // C++ [dcl.enum]p4: Following the closing brace of an
        // enum-specifier, each enumerator has the type of its
        // enumeration. 
        ECD->setType(EnumType);
      continue;
    } else {
      NewTy = BestType;
      NewWidth = BestWidth;
      NewSign = BestType->isSignedIntegerType();
    }

    // Adjust the APSInt value.
    InitVal.extOrTrunc(NewWidth);
    InitVal.setIsSigned(NewSign);
    ECD->setInitVal(InitVal);
    
    // Adjust the Expr initializer and type.
    if (ECD->getInitExpr())
      ECD->setInitExpr(new (Context) ImplicitCastExpr(NewTy, 
                                                      CastExpr::CK_Unknown, 
                                                      ECD->getInitExpr(), 
                                                      /*isLvalue=*/false));
    if (getLangOptions().CPlusPlus)
      // C++ [dcl.enum]p4: Following the closing brace of an
      // enum-specifier, each enumerator has the type of its
      // enumeration. 
      ECD->setType(EnumType);
    else
      ECD->setType(NewTy);
  }
  
  Enum->completeDefinition(Context, BestType);
}

Sema::DeclPtrTy Sema::ActOnFileScopeAsmDecl(SourceLocation Loc,
                                            ExprArg expr) {
  StringLiteral *AsmString = cast<StringLiteral>(expr.takeAs<Expr>());

  FileScopeAsmDecl *New = FileScopeAsmDecl::Create(Context, CurContext,
                                                   Loc, AsmString);
  CurContext->addDecl(New);
  return DeclPtrTy::make(New);
}

void Sema::ActOnPragmaWeakID(IdentifierInfo* Name,
                             SourceLocation PragmaLoc,
                             SourceLocation NameLoc) {
  Decl *PrevDecl = LookupName(TUScope, Name, LookupOrdinaryName);

  if (PrevDecl) {
    PrevDecl->addAttr(::new (Context) WeakAttr());
  } else {
    (void)WeakUndeclaredIdentifiers.insert(
      std::pair<IdentifierInfo*,WeakInfo>
        (Name, WeakInfo((IdentifierInfo*)0, NameLoc)));
  }
}

void Sema::ActOnPragmaWeakAlias(IdentifierInfo* Name,
                                IdentifierInfo* AliasName,
                                SourceLocation PragmaLoc,
                                SourceLocation NameLoc,
                                SourceLocation AliasNameLoc) {
  Decl *PrevDecl = LookupName(TUScope, AliasName, LookupOrdinaryName);
  WeakInfo W = WeakInfo(Name, NameLoc);

  if (PrevDecl) {
    if (!PrevDecl->hasAttr<AliasAttr>())
      if (NamedDecl *ND = dyn_cast<NamedDecl>(PrevDecl))
        DeclApplyPragmaWeak(TUScope, ND, W);
  } else {
    (void)WeakUndeclaredIdentifiers.insert(
      std::pair<IdentifierInfo*,WeakInfo>(AliasName, W));
  }
}
