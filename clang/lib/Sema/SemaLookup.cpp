//===--------------------- SemaLookup.cpp - Name Lookup  ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements name lookup for C, C++, Objective-C, and
//  Objective-C++.
//
//===----------------------------------------------------------------------===//
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include <limits>
#include <list>
#include <set>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>

using namespace clang;
using namespace sema;

namespace {
  class UnqualUsingEntry {
    const DeclContext *Nominated;
    const DeclContext *CommonAncestor;

  public:
    UnqualUsingEntry(const DeclContext *Nominated,
                     const DeclContext *CommonAncestor)
      : Nominated(Nominated), CommonAncestor(CommonAncestor) {
    }

    const DeclContext *getCommonAncestor() const {
      return CommonAncestor;
    }

    const DeclContext *getNominatedNamespace() const {
      return Nominated;
    }

    // Sort by the pointer value of the common ancestor.
    struct Comparator {
      bool operator()(const UnqualUsingEntry &L, const UnqualUsingEntry &R) {
        return L.getCommonAncestor() < R.getCommonAncestor();
      }

      bool operator()(const UnqualUsingEntry &E, const DeclContext *DC) {
        return E.getCommonAncestor() < DC;
      }

      bool operator()(const DeclContext *DC, const UnqualUsingEntry &E) {
        return DC < E.getCommonAncestor();
      }
    };
  };

  /// A collection of using directives, as used by C++ unqualified
  /// lookup.
  class UnqualUsingDirectiveSet {
    typedef llvm::SmallVector<UnqualUsingEntry, 8> ListTy;

    ListTy list;
    llvm::SmallPtrSet<DeclContext*, 8> visited;

  public:
    UnqualUsingDirectiveSet() {}

    void visitScopeChain(Scope *S, Scope *InnermostFileScope) {
      // C++ [namespace.udir]p1:
      //   During unqualified name lookup, the names appear as if they
      //   were declared in the nearest enclosing namespace which contains
      //   both the using-directive and the nominated namespace.
      DeclContext *InnermostFileDC
        = static_cast<DeclContext*>(InnermostFileScope->getEntity());
      assert(InnermostFileDC && InnermostFileDC->isFileContext());

      for (; S; S = S->getParent()) {
        if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity())) {
          DeclContext *EffectiveDC = (Ctx->isFileContext() ? Ctx : InnermostFileDC);
          visit(Ctx, EffectiveDC);
        } else {
          Scope::udir_iterator I = S->using_directives_begin(),
                             End = S->using_directives_end();

          for (; I != End; ++I)
            visit(*I, InnermostFileDC);
        }
      }
    }

    // Visits a context and collect all of its using directives
    // recursively.  Treats all using directives as if they were
    // declared in the context.
    //
    // A given context is only every visited once, so it is important
    // that contexts be visited from the inside out in order to get
    // the effective DCs right.
    void visit(DeclContext *DC, DeclContext *EffectiveDC) {
      if (!visited.insert(DC))
        return;

      addUsingDirectives(DC, EffectiveDC);
    }

    // Visits a using directive and collects all of its using
    // directives recursively.  Treats all using directives as if they
    // were declared in the effective DC.
    void visit(UsingDirectiveDecl *UD, DeclContext *EffectiveDC) {
      DeclContext *NS = UD->getNominatedNamespace();
      if (!visited.insert(NS))
        return;

      addUsingDirective(UD, EffectiveDC);
      addUsingDirectives(NS, EffectiveDC);
    }

    // Adds all the using directives in a context (and those nominated
    // by its using directives, transitively) as if they appeared in
    // the given effective context.
    void addUsingDirectives(DeclContext *DC, DeclContext *EffectiveDC) {
      llvm::SmallVector<DeclContext*,4> queue;
      while (true) {
        DeclContext::udir_iterator I, End;
        for (llvm::tie(I, End) = DC->getUsingDirectives(); I != End; ++I) {
          UsingDirectiveDecl *UD = *I;
          DeclContext *NS = UD->getNominatedNamespace();
          if (visited.insert(NS)) {
            addUsingDirective(UD, EffectiveDC);
            queue.push_back(NS);
          }
        }

        if (queue.empty())
          return;

        DC = queue.back();
        queue.pop_back();
      }
    }

    // Add a using directive as if it had been declared in the given
    // context.  This helps implement C++ [namespace.udir]p3:
    //   The using-directive is transitive: if a scope contains a
    //   using-directive that nominates a second namespace that itself
    //   contains using-directives, the effect is as if the
    //   using-directives from the second namespace also appeared in
    //   the first.
    void addUsingDirective(UsingDirectiveDecl *UD, DeclContext *EffectiveDC) {
      // Find the common ancestor between the effective context and
      // the nominated namespace.
      DeclContext *Common = UD->getNominatedNamespace();
      while (!Common->Encloses(EffectiveDC))
        Common = Common->getParent();
      Common = Common->getPrimaryContext();

      list.push_back(UnqualUsingEntry(UD->getNominatedNamespace(), Common));
    }

    void done() {
      std::sort(list.begin(), list.end(), UnqualUsingEntry::Comparator());
    }

    typedef ListTy::const_iterator const_iterator;

    const_iterator begin() const { return list.begin(); }
    const_iterator end() const { return list.end(); }

    std::pair<const_iterator,const_iterator>
    getNamespacesFor(DeclContext *DC) const {
      return std::equal_range(begin(), end(), DC->getPrimaryContext(),
                              UnqualUsingEntry::Comparator());
    }
  };
}

// Retrieve the set of identifier namespaces that correspond to a
// specific kind of name lookup.
static inline unsigned getIDNS(Sema::LookupNameKind NameKind,
                               bool CPlusPlus,
                               bool Redeclaration) {
  unsigned IDNS = 0;
  switch (NameKind) {
  case Sema::LookupOrdinaryName:
  case Sema::LookupRedeclarationWithLinkage:
    IDNS = Decl::IDNS_Ordinary;
    if (CPlusPlus) {
      IDNS |= Decl::IDNS_Tag | Decl::IDNS_Member | Decl::IDNS_Namespace;
      if (Redeclaration)
        IDNS |= Decl::IDNS_TagFriend | Decl::IDNS_OrdinaryFriend;
    }
    break;

  case Sema::LookupOperatorName:
    // Operator lookup is its own crazy thing;  it is not the same
    // as (e.g.) looking up an operator name for redeclaration.
    assert(!Redeclaration && "cannot do redeclaration operator lookup");
    IDNS = Decl::IDNS_NonMemberOperator;
    break;

  case Sema::LookupTagName:
    if (CPlusPlus) {
      IDNS = Decl::IDNS_Type;

      // When looking for a redeclaration of a tag name, we add:
      // 1) TagFriend to find undeclared friend decls
      // 2) Namespace because they can't "overload" with tag decls.
      // 3) Tag because it includes class templates, which can't
      //    "overload" with tag decls.
      if (Redeclaration)
        IDNS |= Decl::IDNS_Tag | Decl::IDNS_TagFriend | Decl::IDNS_Namespace;
    } else {
      IDNS = Decl::IDNS_Tag;
    }
    break;
  case Sema::LookupLabel:
    IDNS = Decl::IDNS_Label;
    break;
      
  case Sema::LookupMemberName:
    IDNS = Decl::IDNS_Member;
    if (CPlusPlus)
      IDNS |= Decl::IDNS_Tag | Decl::IDNS_Ordinary;
    break;

  case Sema::LookupNestedNameSpecifierName:
    IDNS = Decl::IDNS_Type | Decl::IDNS_Namespace;
    break;

  case Sema::LookupNamespaceName:
    IDNS = Decl::IDNS_Namespace;
    break;

  case Sema::LookupUsingDeclName:
    IDNS = Decl::IDNS_Ordinary | Decl::IDNS_Tag
         | Decl::IDNS_Member | Decl::IDNS_Using;
    break;

  case Sema::LookupObjCProtocolName:
    IDNS = Decl::IDNS_ObjCProtocol;
    break;

  case Sema::LookupAnyName:
    IDNS = Decl::IDNS_Ordinary | Decl::IDNS_Tag | Decl::IDNS_Member
      | Decl::IDNS_Using | Decl::IDNS_Namespace | Decl::IDNS_ObjCProtocol
      | Decl::IDNS_Type;
    break;
  }
  return IDNS;
}

void LookupResult::configure() {
  IDNS = getIDNS(LookupKind, SemaRef.getLangOptions().CPlusPlus,
                 isForRedeclaration());

  // If we're looking for one of the allocation or deallocation
  // operators, make sure that the implicitly-declared new and delete
  // operators can be found.
  if (!isForRedeclaration()) {
    switch (NameInfo.getName().getCXXOverloadedOperator()) {
    case OO_New:
    case OO_Delete:
    case OO_Array_New:
    case OO_Array_Delete:
      SemaRef.DeclareGlobalNewDelete();
      break;

    default:
      break;
    }
  }
}

void LookupResult::sanity() const {
  assert(ResultKind != NotFound || Decls.size() == 0);
  assert(ResultKind != Found || Decls.size() == 1);
  assert(ResultKind != FoundOverloaded || Decls.size() > 1 ||
         (Decls.size() == 1 &&
          isa<FunctionTemplateDecl>((*begin())->getUnderlyingDecl())));
  assert(ResultKind != FoundUnresolvedValue || sanityCheckUnresolved());
  assert(ResultKind != Ambiguous || Decls.size() > 1 ||
         (Decls.size() == 1 && (Ambiguity == AmbiguousBaseSubobjects ||
                                Ambiguity == AmbiguousBaseSubobjectTypes)));
  assert((Paths != NULL) == (ResultKind == Ambiguous &&
                             (Ambiguity == AmbiguousBaseSubobjectTypes ||
                              Ambiguity == AmbiguousBaseSubobjects)));
}

// Necessary because CXXBasePaths is not complete in Sema.h
void LookupResult::deletePaths(CXXBasePaths *Paths) {
  delete Paths;
}

/// Resolves the result kind of this lookup.
void LookupResult::resolveKind() {
  unsigned N = Decls.size();

  // Fast case: no possible ambiguity.
  if (N == 0) {
    assert(ResultKind == NotFound || ResultKind == NotFoundInCurrentInstantiation);
    return;
  }

  // If there's a single decl, we need to examine it to decide what
  // kind of lookup this is.
  if (N == 1) {
    NamedDecl *D = (*Decls.begin())->getUnderlyingDecl();
    if (isa<FunctionTemplateDecl>(D))
      ResultKind = FoundOverloaded;
    else if (isa<UnresolvedUsingValueDecl>(D))
      ResultKind = FoundUnresolvedValue;
    return;
  }

  // Don't do any extra resolution if we've already resolved as ambiguous.
  if (ResultKind == Ambiguous) return;

  llvm::SmallPtrSet<NamedDecl*, 16> Unique;
  llvm::SmallPtrSet<QualType, 16> UniqueTypes;

  bool Ambiguous = false;
  bool HasTag = false, HasFunction = false, HasNonFunction = false;
  bool HasFunctionTemplate = false, HasUnresolved = false;

  unsigned UniqueTagIndex = 0;

  unsigned I = 0;
  while (I < N) {
    NamedDecl *D = Decls[I]->getUnderlyingDecl();
    D = cast<NamedDecl>(D->getCanonicalDecl());

    // Redeclarations of types via typedef can occur both within a scope
    // and, through using declarations and directives, across scopes. There is
    // no ambiguity if they all refer to the same type, so unique based on the
    // canonical type.
    if (TypeDecl *TD = dyn_cast<TypeDecl>(D)) {
      if (!TD->getDeclContext()->isRecord()) {
        QualType T = SemaRef.Context.getTypeDeclType(TD);
        if (!UniqueTypes.insert(SemaRef.Context.getCanonicalType(T))) {
          // The type is not unique; pull something off the back and continue
          // at this index.
          Decls[I] = Decls[--N];
          continue;
        }
      }
    }

    if (!Unique.insert(D)) {
      // If it's not unique, pull something off the back (and
      // continue at this index).
      Decls[I] = Decls[--N];
      continue;
    }

    // Otherwise, do some decl type analysis and then continue.

    if (isa<UnresolvedUsingValueDecl>(D)) {
      HasUnresolved = true;
    } else if (isa<TagDecl>(D)) {
      if (HasTag)
        Ambiguous = true;
      UniqueTagIndex = I;
      HasTag = true;
    } else if (isa<FunctionTemplateDecl>(D)) {
      HasFunction = true;
      HasFunctionTemplate = true;
    } else if (isa<FunctionDecl>(D)) {
      HasFunction = true;
    } else {
      if (HasNonFunction)
        Ambiguous = true;
      HasNonFunction = true;
    }
    I++;
  }

  // C++ [basic.scope.hiding]p2:
  //   A class name or enumeration name can be hidden by the name of
  //   an object, function, or enumerator declared in the same
  //   scope. If a class or enumeration name and an object, function,
  //   or enumerator are declared in the same scope (in any order)
  //   with the same name, the class or enumeration name is hidden
  //   wherever the object, function, or enumerator name is visible.
  // But it's still an error if there are distinct tag types found,
  // even if they're not visible. (ref?)
  if (HideTags && HasTag && !Ambiguous &&
      (HasFunction || HasNonFunction || HasUnresolved)) {
    if (Decls[UniqueTagIndex]->getDeclContext()->getRedeclContext()->Equals(
         Decls[UniqueTagIndex? 0 : N-1]->getDeclContext()->getRedeclContext()))
      Decls[UniqueTagIndex] = Decls[--N];
    else
      Ambiguous = true;
  }

  Decls.set_size(N);

  if (HasNonFunction && (HasFunction || HasUnresolved))
    Ambiguous = true;

  if (Ambiguous)
    setAmbiguous(LookupResult::AmbiguousReference);
  else if (HasUnresolved)
    ResultKind = LookupResult::FoundUnresolvedValue;
  else if (N > 1 || HasFunctionTemplate)
    ResultKind = LookupResult::FoundOverloaded;
  else
    ResultKind = LookupResult::Found;
}

void LookupResult::addDeclsFromBasePaths(const CXXBasePaths &P) {
  CXXBasePaths::const_paths_iterator I, E;
  DeclContext::lookup_iterator DI, DE;
  for (I = P.begin(), E = P.end(); I != E; ++I)
    for (llvm::tie(DI,DE) = I->Decls; DI != DE; ++DI)
      addDecl(*DI);
}

void LookupResult::setAmbiguousBaseSubobjects(CXXBasePaths &P) {
  Paths = new CXXBasePaths;
  Paths->swap(P);
  addDeclsFromBasePaths(*Paths);
  resolveKind();
  setAmbiguous(AmbiguousBaseSubobjects);
}

void LookupResult::setAmbiguousBaseSubobjectTypes(CXXBasePaths &P) {
  Paths = new CXXBasePaths;
  Paths->swap(P);
  addDeclsFromBasePaths(*Paths);
  resolveKind();
  setAmbiguous(AmbiguousBaseSubobjectTypes);
}

void LookupResult::print(llvm::raw_ostream &Out) {
  Out << Decls.size() << " result(s)";
  if (isAmbiguous()) Out << ", ambiguous";
  if (Paths) Out << ", base paths present";

  for (iterator I = begin(), E = end(); I != E; ++I) {
    Out << "\n";
    (*I)->print(Out, 2);
  }
}

/// \brief Lookup a builtin function, when name lookup would otherwise
/// fail.
static bool LookupBuiltin(Sema &S, LookupResult &R) {
  Sema::LookupNameKind NameKind = R.getLookupKind();

  // If we didn't find a use of this identifier, and if the identifier
  // corresponds to a compiler builtin, create the decl object for the builtin
  // now, injecting it into translation unit scope, and return it.
  if (NameKind == Sema::LookupOrdinaryName ||
      NameKind == Sema::LookupRedeclarationWithLinkage) {
    IdentifierInfo *II = R.getLookupName().getAsIdentifierInfo();
    if (II) {
      // If this is a builtin on this (or all) targets, create the decl.
      if (unsigned BuiltinID = II->getBuiltinID()) {
        // In C++, we don't have any predefined library functions like
        // 'malloc'. Instead, we'll just error.
        if (S.getLangOptions().CPlusPlus &&
            S.Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID))
          return false;

        if (NamedDecl *D = S.LazilyCreateBuiltin((IdentifierInfo *)II,
                                                 BuiltinID, S.TUScope,
                                                 R.isForRedeclaration(),
                                                 R.getNameLoc())) {
          R.addDecl(D);
          return true;
        }

        if (R.isForRedeclaration()) {
          // If we're redeclaring this function anyway, forget that
          // this was a builtin at all.
          S.Context.BuiltinInfo.ForgetBuiltin(BuiltinID, S.Context.Idents);
        }

        return false;
      }
    }
  }

  return false;
}

/// \brief Determine whether we can declare a special member function within
/// the class at this point.
static bool CanDeclareSpecialMemberFunction(ASTContext &Context,
                                            const CXXRecordDecl *Class) {
  // Don't do it if the class is invalid.
  if (Class->isInvalidDecl())
    return false;

  // We need to have a definition for the class.
  if (!Class->getDefinition() || Class->isDependentContext())
    return false;

  // We can't be in the middle of defining the class.
  if (const RecordType *RecordTy
                        = Context.getTypeDeclType(Class)->getAs<RecordType>())
    return !RecordTy->isBeingDefined();

  return false;
}

void Sema::ForceDeclarationOfImplicitMembers(CXXRecordDecl *Class) {
  if (!CanDeclareSpecialMemberFunction(Context, Class))
    return;

  // If the default constructor has not yet been declared, do so now.
  if (!Class->hasDeclaredDefaultConstructor())
    DeclareImplicitDefaultConstructor(Class);

  // If the copy constructor has not yet been declared, do so now.
  if (!Class->hasDeclaredCopyConstructor())
    DeclareImplicitCopyConstructor(Class);

  // If the copy assignment operator has not yet been declared, do so now.
  if (!Class->hasDeclaredCopyAssignment())
    DeclareImplicitCopyAssignment(Class);

  // If the destructor has not yet been declared, do so now.
  if (!Class->hasDeclaredDestructor())
    DeclareImplicitDestructor(Class);
}

/// \brief Determine whether this is the name of an implicitly-declared
/// special member function.
static bool isImplicitlyDeclaredMemberFunctionName(DeclarationName Name) {
  switch (Name.getNameKind()) {
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
    return true;

  case DeclarationName::CXXOperatorName:
    return Name.getCXXOverloadedOperator() == OO_Equal;

  default:
    break;
  }

  return false;
}

/// \brief If there are any implicit member functions with the given name
/// that need to be declared in the given declaration context, do so.
static void DeclareImplicitMemberFunctionsWithName(Sema &S,
                                                   DeclarationName Name,
                                                   const DeclContext *DC) {
  if (!DC)
    return;

  switch (Name.getNameKind()) {
  case DeclarationName::CXXConstructorName:
    if (const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(DC))
      if (Record->getDefinition() &&
          CanDeclareSpecialMemberFunction(S.Context, Record)) {
        if (!Record->hasDeclaredDefaultConstructor())
          S.DeclareImplicitDefaultConstructor(
                                           const_cast<CXXRecordDecl *>(Record));
        if (!Record->hasDeclaredCopyConstructor())
          S.DeclareImplicitCopyConstructor(const_cast<CXXRecordDecl *>(Record));
      }
    break;

  case DeclarationName::CXXDestructorName:
    if (const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(DC))
      if (Record->getDefinition() && !Record->hasDeclaredDestructor() &&
          CanDeclareSpecialMemberFunction(S.Context, Record))
        S.DeclareImplicitDestructor(const_cast<CXXRecordDecl *>(Record));
    break;

  case DeclarationName::CXXOperatorName:
    if (Name.getCXXOverloadedOperator() != OO_Equal)
      break;

    if (const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(DC))
      if (Record->getDefinition() && !Record->hasDeclaredCopyAssignment() &&
          CanDeclareSpecialMemberFunction(S.Context, Record))
        S.DeclareImplicitCopyAssignment(const_cast<CXXRecordDecl *>(Record));
    break;

  default:
    break;
  }
}

// Adds all qualifying matches for a name within a decl context to the
// given lookup result.  Returns true if any matches were found.
static bool LookupDirect(Sema &S, LookupResult &R, const DeclContext *DC) {
  bool Found = false;

  // Lazily declare C++ special member functions.
  if (S.getLangOptions().CPlusPlus)
    DeclareImplicitMemberFunctionsWithName(S, R.getLookupName(), DC);

  // Perform lookup into this declaration context.
  DeclContext::lookup_const_iterator I, E;
  for (llvm::tie(I, E) = DC->lookup(R.getLookupName()); I != E; ++I) {
    NamedDecl *D = *I;
    if (R.isAcceptableDecl(D)) {
      R.addDecl(D);
      Found = true;
    }
  }

  if (!Found && DC->isTranslationUnit() && LookupBuiltin(S, R))
    return true;

  if (R.getLookupName().getNameKind()
        != DeclarationName::CXXConversionFunctionName ||
      R.getLookupName().getCXXNameType()->isDependentType() ||
      !isa<CXXRecordDecl>(DC))
    return Found;

  // C++ [temp.mem]p6:
  //   A specialization of a conversion function template is not found by
  //   name lookup. Instead, any conversion function templates visible in the
  //   context of the use are considered. [...]
  const CXXRecordDecl *Record = cast<CXXRecordDecl>(DC);
  if (!Record->isDefinition())
    return Found;

  const UnresolvedSetImpl *Unresolved = Record->getConversionFunctions();
  for (UnresolvedSetImpl::iterator U = Unresolved->begin(),
         UEnd = Unresolved->end(); U != UEnd; ++U) {
    FunctionTemplateDecl *ConvTemplate = dyn_cast<FunctionTemplateDecl>(*U);
    if (!ConvTemplate)
      continue;

    // When we're performing lookup for the purposes of redeclaration, just
    // add the conversion function template. When we deduce template
    // arguments for specializations, we'll end up unifying the return
    // type of the new declaration with the type of the function template.
    if (R.isForRedeclaration()) {
      R.addDecl(ConvTemplate);
      Found = true;
      continue;
    }

    // C++ [temp.mem]p6:
    //   [...] For each such operator, if argument deduction succeeds
    //   (14.9.2.3), the resulting specialization is used as if found by
    //   name lookup.
    //
    // When referencing a conversion function for any purpose other than
    // a redeclaration (such that we'll be building an expression with the
    // result), perform template argument deduction and place the
    // specialization into the result set. We do this to avoid forcing all
    // callers to perform special deduction for conversion functions.
    TemplateDeductionInfo Info(R.getSema().Context, R.getNameLoc());
    FunctionDecl *Specialization = 0;

    const FunctionProtoType *ConvProto
      = ConvTemplate->getTemplatedDecl()->getType()->getAs<FunctionProtoType>();
    assert(ConvProto && "Nonsensical conversion function template type");

    // Compute the type of the function that we would expect the conversion
    // function to have, if it were to match the name given.
    // FIXME: Calling convention!
    FunctionProtoType::ExtProtoInfo EPI = ConvProto->getExtProtoInfo();
    EPI.ExtInfo = EPI.ExtInfo.withCallingConv(CC_Default);
    EPI.HasExceptionSpec = false;
    EPI.HasAnyExceptionSpec = false;
    EPI.NumExceptions = 0;
    QualType ExpectedType
      = R.getSema().Context.getFunctionType(R.getLookupName().getCXXNameType(),
                                            0, 0, EPI);

    // Perform template argument deduction against the type that we would
    // expect the function to have.
    if (R.getSema().DeduceTemplateArguments(ConvTemplate, 0, ExpectedType,
                                            Specialization, Info)
          == Sema::TDK_Success) {
      R.addDecl(Specialization);
      Found = true;
    }
  }

  return Found;
}

// Performs C++ unqualified lookup into the given file context.
static bool
CppNamespaceLookup(Sema &S, LookupResult &R, ASTContext &Context,
                   DeclContext *NS, UnqualUsingDirectiveSet &UDirs) {

  assert(NS && NS->isFileContext() && "CppNamespaceLookup() requires namespace!");

  // Perform direct name lookup into the LookupCtx.
  bool Found = LookupDirect(S, R, NS);

  // Perform direct name lookup into the namespaces nominated by the
  // using directives whose common ancestor is this namespace.
  UnqualUsingDirectiveSet::const_iterator UI, UEnd;
  llvm::tie(UI, UEnd) = UDirs.getNamespacesFor(NS);

  for (; UI != UEnd; ++UI)
    if (LookupDirect(S, R, UI->getNominatedNamespace()))
      Found = true;

  R.resolveKind();

  return Found;
}

static bool isNamespaceOrTranslationUnitScope(Scope *S) {
  if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity()))
    return Ctx->isFileContext();
  return false;
}

// Find the next outer declaration context from this scope. This
// routine actually returns the semantic outer context, which may
// differ from the lexical context (encoded directly in the Scope
// stack) when we are parsing a member of a class template. In this
// case, the second element of the pair will be true, to indicate that
// name lookup should continue searching in this semantic context when
// it leaves the current template parameter scope.
static std::pair<DeclContext *, bool> findOuterContext(Scope *S) {
  DeclContext *DC = static_cast<DeclContext *>(S->getEntity());
  DeclContext *Lexical = 0;
  for (Scope *OuterS = S->getParent(); OuterS;
       OuterS = OuterS->getParent()) {
    if (OuterS->getEntity()) {
      Lexical = static_cast<DeclContext *>(OuterS->getEntity());
      break;
    }
  }

  // C++ [temp.local]p8:
  //   In the definition of a member of a class template that appears
  //   outside of the namespace containing the class template
  //   definition, the name of a template-parameter hides the name of
  //   a member of this namespace.
  //
  // Example:
  //
  //   namespace N {
  //     class C { };
  //
  //     template<class T> class B {
  //       void f(T);
  //     };
  //   }
  //
  //   template<class C> void N::B<C>::f(C) {
  //     C b;  // C is the template parameter, not N::C
  //   }
  //
  // In this example, the lexical context we return is the
  // TranslationUnit, while the semantic context is the namespace N.
  if (!Lexical || !DC || !S->getParent() ||
      !S->getParent()->isTemplateParamScope())
    return std::make_pair(Lexical, false);

  // Find the outermost template parameter scope.
  // For the example, this is the scope for the template parameters of
  // template<class C>.
  Scope *OutermostTemplateScope = S->getParent();
  while (OutermostTemplateScope->getParent() &&
         OutermostTemplateScope->getParent()->isTemplateParamScope())
    OutermostTemplateScope = OutermostTemplateScope->getParent();

  // Find the namespace context in which the original scope occurs. In
  // the example, this is namespace N.
  DeclContext *Semantic = DC;
  while (!Semantic->isFileContext())
    Semantic = Semantic->getParent();

  // Find the declaration context just outside of the template
  // parameter scope. This is the context in which the template is
  // being lexically declaration (a namespace context). In the
  // example, this is the global scope.
  if (Lexical->isFileContext() && !Lexical->Equals(Semantic) &&
      Lexical->Encloses(Semantic))
    return std::make_pair(Semantic, true);

  return std::make_pair(Lexical, false);
}

bool Sema::CppLookupName(LookupResult &R, Scope *S) {
  assert(getLangOptions().CPlusPlus && "Can perform only C++ lookup");

  DeclarationName Name = R.getLookupName();

  // If this is the name of an implicitly-declared special member function,
  // go through the scope stack to implicitly declare
  if (isImplicitlyDeclaredMemberFunctionName(Name)) {
    for (Scope *PreS = S; PreS; PreS = PreS->getParent())
      if (DeclContext *DC = static_cast<DeclContext *>(PreS->getEntity()))
        DeclareImplicitMemberFunctionsWithName(*this, Name, DC);
  }

  // Implicitly declare member functions with the name we're looking for, if in
  // fact we are in a scope where it matters.

  Scope *Initial = S;
  IdentifierResolver::iterator
    I = IdResolver.begin(Name),
    IEnd = IdResolver.end();

  // First we lookup local scope.
  // We don't consider using-directives, as per 7.3.4.p1 [namespace.udir]
  // ...During unqualified name lookup (3.4.1), the names appear as if
  // they were declared in the nearest enclosing namespace which contains
  // both the using-directive and the nominated namespace.
  // [Note: in this context, "contains" means "contains directly or
  // indirectly".
  //
  // For example:
  // namespace A { int i; }
  // void foo() {
  //   int i;
  //   {
  //     using namespace A;
  //     ++i; // finds local 'i', A::i appears at global scope
  //   }
  // }
  //
  DeclContext *OutsideOfTemplateParamDC = 0;
  for (; S && !isNamespaceOrTranslationUnitScope(S); S = S->getParent()) {
    DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity());

    // Check whether the IdResolver has anything in this scope.
    bool Found = false;
    for (; I != IEnd && S->isDeclScope(*I); ++I) {
      if (R.isAcceptableDecl(*I)) {
        Found = true;
        R.addDecl(*I);
      }
    }
    if (Found) {
      R.resolveKind();
      if (S->isClassScope())
        if (CXXRecordDecl *Record = dyn_cast_or_null<CXXRecordDecl>(Ctx))
          R.setNamingClass(Record);
      return true;
    }

    if (!Ctx && S->isTemplateParamScope() && OutsideOfTemplateParamDC &&
        S->getParent() && !S->getParent()->isTemplateParamScope()) {
      // We've just searched the last template parameter scope and
      // found nothing, so look into the the contexts between the
      // lexical and semantic declaration contexts returned by
      // findOuterContext(). This implements the name lookup behavior
      // of C++ [temp.local]p8.
      Ctx = OutsideOfTemplateParamDC;
      OutsideOfTemplateParamDC = 0;
    }

    if (Ctx) {
      DeclContext *OuterCtx;
      bool SearchAfterTemplateScope;
      llvm::tie(OuterCtx, SearchAfterTemplateScope) = findOuterContext(S);
      if (SearchAfterTemplateScope)
        OutsideOfTemplateParamDC = OuterCtx;

      for (; Ctx && !Ctx->Equals(OuterCtx); Ctx = Ctx->getLookupParent()) {
        // We do not directly look into transparent contexts, since
        // those entities will be found in the nearest enclosing
        // non-transparent context.
        if (Ctx->isTransparentContext())
          continue;

        // We do not look directly into function or method contexts,
        // since all of the local variables and parameters of the
        // function/method are present within the Scope.
        if (Ctx->isFunctionOrMethod()) {
          // If we have an Objective-C instance method, look for ivars
          // in the corresponding interface.
          if (ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(Ctx)) {
            if (Method->isInstanceMethod() && Name.getAsIdentifierInfo())
              if (ObjCInterfaceDecl *Class = Method->getClassInterface()) {
                ObjCInterfaceDecl *ClassDeclared;
                if (ObjCIvarDecl *Ivar = Class->lookupInstanceVariable(
                                                 Name.getAsIdentifierInfo(),
                                                             ClassDeclared)) {
                  if (R.isAcceptableDecl(Ivar)) {
                    R.addDecl(Ivar);
                    R.resolveKind();
                    return true;
                  }
                }
              }
          }

          continue;
        }

        // Perform qualified name lookup into this context.
        // FIXME: In some cases, we know that every name that could be found by
        // this qualified name lookup will also be on the identifier chain. For
        // example, inside a class without any base classes, we never need to
        // perform qualified lookup because all of the members are on top of the
        // identifier chain.
        if (LookupQualifiedName(R, Ctx, /*InUnqualifiedLookup=*/true))
          return true;
      }
    }
  }

  // Stop if we ran out of scopes.
  // FIXME:  This really, really shouldn't be happening.
  if (!S) return false;

  // If we are looking for members, no need to look into global/namespace scope.
  if (R.getLookupKind() == LookupMemberName)
    return false;

  // Collect UsingDirectiveDecls in all scopes, and recursively all
  // nominated namespaces by those using-directives.
  //
  // FIXME: Cache this sorted list in Scope structure, and DeclContext, so we
  // don't build it for each lookup!

  UnqualUsingDirectiveSet UDirs;
  UDirs.visitScopeChain(Initial, S);
  UDirs.done();

  // Lookup namespace scope, and global scope.
  // Unqualified name lookup in C++ requires looking into scopes
  // that aren't strictly lexical, and therefore we walk through the
  // context as well as walking through the scopes.

  for (; S; S = S->getParent()) {
    // Check whether the IdResolver has anything in this scope.
    bool Found = false;
    for (; I != IEnd && S->isDeclScope(*I); ++I) {
      if (R.isAcceptableDecl(*I)) {
        // We found something.  Look for anything else in our scope
        // with this same name and in an acceptable identifier
        // namespace, so that we can construct an overload set if we
        // need to.
        Found = true;
        R.addDecl(*I);
      }
    }

    if (Found && S->isTemplateParamScope()) {
      R.resolveKind();
      return true;
    }

    DeclContext *Ctx = static_cast<DeclContext *>(S->getEntity());
    if (!Ctx && S->isTemplateParamScope() && OutsideOfTemplateParamDC &&
        S->getParent() && !S->getParent()->isTemplateParamScope()) {
      // We've just searched the last template parameter scope and
      // found nothing, so look into the the contexts between the
      // lexical and semantic declaration contexts returned by
      // findOuterContext(). This implements the name lookup behavior
      // of C++ [temp.local]p8.
      Ctx = OutsideOfTemplateParamDC;
      OutsideOfTemplateParamDC = 0;
    }

    if (Ctx) {
      DeclContext *OuterCtx;
      bool SearchAfterTemplateScope;
      llvm::tie(OuterCtx, SearchAfterTemplateScope) = findOuterContext(S);
      if (SearchAfterTemplateScope)
        OutsideOfTemplateParamDC = OuterCtx;

      for (; Ctx && !Ctx->Equals(OuterCtx); Ctx = Ctx->getLookupParent()) {
        // We do not directly look into transparent contexts, since
        // those entities will be found in the nearest enclosing
        // non-transparent context.
        if (Ctx->isTransparentContext())
          continue;

        // If we have a context, and it's not a context stashed in the
        // template parameter scope for an out-of-line definition, also
        // look into that context.
        if (!(Found && S && S->isTemplateParamScope())) {
          assert(Ctx->isFileContext() &&
              "We should have been looking only at file context here already.");

          // Look into context considering using-directives.
          if (CppNamespaceLookup(*this, R, Context, Ctx, UDirs))
            Found = true;
        }

        if (Found) {
          R.resolveKind();
          return true;
        }

        if (R.isForRedeclaration() && !Ctx->isTransparentContext())
          return false;
      }
    }

    if (R.isForRedeclaration() && Ctx && !Ctx->isTransparentContext())
      return false;
  }

  return !R.empty();
}

/// @brief Perform unqualified name lookup starting from a given
/// scope.
///
/// Unqualified name lookup (C++ [basic.lookup.unqual], C99 6.2.1) is
/// used to find names within the current scope. For example, 'x' in
/// @code
/// int x;
/// int f() {
///   return x; // unqualified name look finds 'x' in the global scope
/// }
/// @endcode
///
/// Different lookup criteria can find different names. For example, a
/// particular scope can have both a struct and a function of the same
/// name, and each can be found by certain lookup criteria. For more
/// information about lookup criteria, see the documentation for the
/// class LookupCriteria.
///
/// @param S        The scope from which unqualified name lookup will
/// begin. If the lookup criteria permits, name lookup may also search
/// in the parent scopes.
///
/// @param Name     The name of the entity that we are searching for.
///
/// @param Loc      If provided, the source location where we're performing
/// name lookup. At present, this is only used to produce diagnostics when
/// C library functions (like "malloc") are implicitly declared.
///
/// @returns The result of name lookup, which includes zero or more
/// declarations and possibly additional information used to diagnose
/// ambiguities.
bool Sema::LookupName(LookupResult &R, Scope *S, bool AllowBuiltinCreation) {
  DeclarationName Name = R.getLookupName();
  if (!Name) return false;

  LookupNameKind NameKind = R.getLookupKind();

  if (!getLangOptions().CPlusPlus) {
    // Unqualified name lookup in C/Objective-C is purely lexical, so
    // search in the declarations attached to the name.
    if (NameKind == Sema::LookupRedeclarationWithLinkage) {
      // Find the nearest non-transparent declaration scope.
      while (!(S->getFlags() & Scope::DeclScope) ||
             (S->getEntity() &&
              static_cast<DeclContext *>(S->getEntity())
                ->isTransparentContext()))
        S = S->getParent();
    }

    unsigned IDNS = R.getIdentifierNamespace();

    // Scan up the scope chain looking for a decl that matches this
    // identifier that is in the appropriate namespace.  This search
    // should not take long, as shadowing of names is uncommon, and
    // deep shadowing is extremely uncommon.
    bool LeftStartingScope = false;

    for (IdentifierResolver::iterator I = IdResolver.begin(Name),
                                   IEnd = IdResolver.end();
         I != IEnd; ++I)
      if ((*I)->isInIdentifierNamespace(IDNS)) {
        if (NameKind == LookupRedeclarationWithLinkage) {
          // Determine whether this (or a previous) declaration is
          // out-of-scope.
          if (!LeftStartingScope && !S->isDeclScope(*I))
            LeftStartingScope = true;

          // If we found something outside of our starting scope that
          // does not have linkage, skip it.
          if (LeftStartingScope && !((*I)->hasLinkage()))
            continue;
        }

        R.addDecl(*I);

        if ((*I)->getAttr<OverloadableAttr>()) {
          // If this declaration has the "overloadable" attribute, we
          // might have a set of overloaded functions.

          // Figure out what scope the identifier is in.
          while (!(S->getFlags() & Scope::DeclScope) ||
                 !S->isDeclScope(*I))
            S = S->getParent();

          // Find the last declaration in this scope (with the same
          // name, naturally).
          IdentifierResolver::iterator LastI = I;
          for (++LastI; LastI != IEnd; ++LastI) {
            if (!S->isDeclScope(*LastI))
              break;
            R.addDecl(*LastI);
          }
        }

        R.resolveKind();

        return true;
      }
  } else {
    // Perform C++ unqualified name lookup.
    if (CppLookupName(R, S))
      return true;
  }

  // If we didn't find a use of this identifier, and if the identifier
  // corresponds to a compiler builtin, create the decl object for the builtin
  // now, injecting it into translation unit scope, and return it.
  if (AllowBuiltinCreation)
    return LookupBuiltin(*this, R);

  // If we didn't find a use of this identifier, the ExternalSource 
  // may be able to handle the situation. 
  // Note: some lookup failures are expected!
  // See e.g. R.isForRedeclaration().
  return (ExternalSource && ExternalSource->LookupUnqualified(R, S));
}

/// @brief Perform qualified name lookup in the namespaces nominated by
/// using directives by the given context.
///
/// C++98 [namespace.qual]p2:
///   Given X::m (where X is a user-declared namespace), or given ::m
///   (where X is the global namespace), let S be the set of all
///   declarations of m in X and in the transitive closure of all
///   namespaces nominated by using-directives in X and its used
///   namespaces, except that using-directives are ignored in any
///   namespace, including X, directly containing one or more
///   declarations of m. No namespace is searched more than once in
///   the lookup of a name. If S is the empty set, the program is
///   ill-formed. Otherwise, if S has exactly one member, or if the
///   context of the reference is a using-declaration
///   (namespace.udecl), S is the required set of declarations of
///   m. Otherwise if the use of m is not one that allows a unique
///   declaration to be chosen from S, the program is ill-formed.
/// C++98 [namespace.qual]p5:
///   During the lookup of a qualified namespace member name, if the
///   lookup finds more than one declaration of the member, and if one
///   declaration introduces a class name or enumeration name and the
///   other declarations either introduce the same object, the same
///   enumerator or a set of functions, the non-type name hides the
///   class or enumeration name if and only if the declarations are
///   from the same namespace; otherwise (the declarations are from
///   different namespaces), the program is ill-formed.
static bool LookupQualifiedNameInUsingDirectives(Sema &S, LookupResult &R,
                                                 DeclContext *StartDC) {
  assert(StartDC->isFileContext() && "start context is not a file context");

  DeclContext::udir_iterator I = StartDC->using_directives_begin();
  DeclContext::udir_iterator E = StartDC->using_directives_end();

  if (I == E) return false;

  // We have at least added all these contexts to the queue.
  llvm::DenseSet<DeclContext*> Visited;
  Visited.insert(StartDC);

  // We have not yet looked into these namespaces, much less added
  // their "using-children" to the queue.
  llvm::SmallVector<NamespaceDecl*, 8> Queue;

  // We have already looked into the initial namespace; seed the queue
  // with its using-children.
  for (; I != E; ++I) {
    NamespaceDecl *ND = (*I)->getNominatedNamespace()->getOriginalNamespace();
    if (Visited.insert(ND).second)
      Queue.push_back(ND);
  }

  // The easiest way to implement the restriction in [namespace.qual]p5
  // is to check whether any of the individual results found a tag
  // and, if so, to declare an ambiguity if the final result is not
  // a tag.
  bool FoundTag = false;
  bool FoundNonTag = false;

  LookupResult LocalR(LookupResult::Temporary, R);

  bool Found = false;
  while (!Queue.empty()) {
    NamespaceDecl *ND = Queue.back();
    Queue.pop_back();

    // We go through some convolutions here to avoid copying results
    // between LookupResults.
    bool UseLocal = !R.empty();
    LookupResult &DirectR = UseLocal ? LocalR : R;
    bool FoundDirect = LookupDirect(S, DirectR, ND);

    if (FoundDirect) {
      // First do any local hiding.
      DirectR.resolveKind();

      // If the local result is a tag, remember that.
      if (DirectR.isSingleTagDecl())
        FoundTag = true;
      else
        FoundNonTag = true;

      // Append the local results to the total results if necessary.
      if (UseLocal) {
        R.addAllDecls(LocalR);
        LocalR.clear();
      }
    }

    // If we find names in this namespace, ignore its using directives.
    if (FoundDirect) {
      Found = true;
      continue;
    }

    for (llvm::tie(I,E) = ND->getUsingDirectives(); I != E; ++I) {
      NamespaceDecl *Nom = (*I)->getNominatedNamespace();
      if (Visited.insert(Nom).second)
        Queue.push_back(Nom);
    }
  }

  if (Found) {
    if (FoundTag && FoundNonTag)
      R.setAmbiguousQualifiedTagHiding();
    else
      R.resolveKind();
  }

  return Found;
}

/// \brief Callback that looks for any member of a class with the given name.
static bool LookupAnyMember(const CXXBaseSpecifier *Specifier,
                            CXXBasePath &Path,
                            void *Name) {
  RecordDecl *BaseRecord = Specifier->getType()->getAs<RecordType>()->getDecl();

  DeclarationName N = DeclarationName::getFromOpaquePtr(Name);
  Path.Decls = BaseRecord->lookup(N);
  return Path.Decls.first != Path.Decls.second;
}

/// \brief Determine whether the given set of member declarations contains only
/// static members, nested types, and enumerators.
template<typename InputIterator>
static bool HasOnlyStaticMembers(InputIterator First, InputIterator Last) {
  Decl *D = (*First)->getUnderlyingDecl();
  if (isa<VarDecl>(D) || isa<TypeDecl>(D) || isa<EnumConstantDecl>(D))
    return true;

  if (isa<CXXMethodDecl>(D)) {
    // Determine whether all of the methods are static.
    bool AllMethodsAreStatic = true;
    for(; First != Last; ++First) {
      D = (*First)->getUnderlyingDecl();

      if (!isa<CXXMethodDecl>(D)) {
        assert(isa<TagDecl>(D) && "Non-function must be a tag decl");
        break;
      }

      if (!cast<CXXMethodDecl>(D)->isStatic()) {
        AllMethodsAreStatic = false;
        break;
      }
    }

    if (AllMethodsAreStatic)
      return true;
  }

  return false;
}

/// \brief Perform qualified name lookup into a given context.
///
/// Qualified name lookup (C++ [basic.lookup.qual]) is used to find
/// names when the context of those names is explicit specified, e.g.,
/// "std::vector" or "x->member", or as part of unqualified name lookup.
///
/// Different lookup criteria can find different names. For example, a
/// particular scope can have both a struct and a function of the same
/// name, and each can be found by certain lookup criteria. For more
/// information about lookup criteria, see the documentation for the
/// class LookupCriteria.
///
/// \param R captures both the lookup criteria and any lookup results found.
///
/// \param LookupCtx The context in which qualified name lookup will
/// search. If the lookup criteria permits, name lookup may also search
/// in the parent contexts or (for C++ classes) base classes.
///
/// \param InUnqualifiedLookup true if this is qualified name lookup that
/// occurs as part of unqualified name lookup.
///
/// \returns true if lookup succeeded, false if it failed.
bool Sema::LookupQualifiedName(LookupResult &R, DeclContext *LookupCtx,
                               bool InUnqualifiedLookup) {
  assert(LookupCtx && "Sema::LookupQualifiedName requires a lookup context");

  if (!R.getLookupName())
    return false;

  // Make sure that the declaration context is complete.
  assert((!isa<TagDecl>(LookupCtx) ||
          LookupCtx->isDependentContext() ||
          cast<TagDecl>(LookupCtx)->isDefinition() ||
          Context.getTypeDeclType(cast<TagDecl>(LookupCtx))->getAs<TagType>()
            ->isBeingDefined()) &&
         "Declaration context must already be complete!");

  // Perform qualified name lookup into the LookupCtx.
  if (LookupDirect(*this, R, LookupCtx)) {
    R.resolveKind();
    if (isa<CXXRecordDecl>(LookupCtx))
      R.setNamingClass(cast<CXXRecordDecl>(LookupCtx));
    return true;
  }

  // Don't descend into implied contexts for redeclarations.
  // C++98 [namespace.qual]p6:
  //   In a declaration for a namespace member in which the
  //   declarator-id is a qualified-id, given that the qualified-id
  //   for the namespace member has the form
  //     nested-name-specifier unqualified-id
  //   the unqualified-id shall name a member of the namespace
  //   designated by the nested-name-specifier.
  // See also [class.mfct]p5 and [class.static.data]p2.
  if (R.isForRedeclaration())
    return false;

  // If this is a namespace, look it up in the implied namespaces.
  if (LookupCtx->isFileContext())
    return LookupQualifiedNameInUsingDirectives(*this, R, LookupCtx);

  // If this isn't a C++ class, we aren't allowed to look into base
  // classes, we're done.
  CXXRecordDecl *LookupRec = dyn_cast<CXXRecordDecl>(LookupCtx);
  if (!LookupRec || !LookupRec->getDefinition())
    return false;

  // If we're performing qualified name lookup into a dependent class,
  // then we are actually looking into a current instantiation. If we have any
  // dependent base classes, then we either have to delay lookup until
  // template instantiation time (at which point all bases will be available)
  // or we have to fail.
  if (!InUnqualifiedLookup && LookupRec->isDependentContext() &&
      LookupRec->hasAnyDependentBases()) {
    R.setNotFoundInCurrentInstantiation();
    return false;
  }

  // Perform lookup into our base classes.
  CXXBasePaths Paths;
  Paths.setOrigin(LookupRec);

  // Look for this member in our base classes
  CXXRecordDecl::BaseMatchesCallback *BaseCallback = 0;
  switch (R.getLookupKind()) {
    case LookupOrdinaryName:
    case LookupMemberName:
    case LookupRedeclarationWithLinkage:
      BaseCallback = &CXXRecordDecl::FindOrdinaryMember;
      break;

    case LookupTagName:
      BaseCallback = &CXXRecordDecl::FindTagMember;
      break;

    case LookupAnyName:
      BaseCallback = &LookupAnyMember;
      break;

    case LookupUsingDeclName:
      // This lookup is for redeclarations only.

    case LookupOperatorName:
    case LookupNamespaceName:
    case LookupObjCProtocolName:
    case LookupLabel:
      // These lookups will never find a member in a C++ class (or base class).
      return false;

    case LookupNestedNameSpecifierName:
      BaseCallback = &CXXRecordDecl::FindNestedNameSpecifierMember;
      break;
  }

  if (!LookupRec->lookupInBases(BaseCallback,
                                R.getLookupName().getAsOpaquePtr(), Paths))
    return false;

  R.setNamingClass(LookupRec);

  // C++ [class.member.lookup]p2:
  //   [...] If the resulting set of declarations are not all from
  //   sub-objects of the same type, or the set has a nonstatic member
  //   and includes members from distinct sub-objects, there is an
  //   ambiguity and the program is ill-formed. Otherwise that set is
  //   the result of the lookup.
  QualType SubobjectType;
  int SubobjectNumber = 0;
  AccessSpecifier SubobjectAccess = AS_none;

  for (CXXBasePaths::paths_iterator Path = Paths.begin(), PathEnd = Paths.end();
       Path != PathEnd; ++Path) {
    const CXXBasePathElement &PathElement = Path->back();

    // Pick the best (i.e. most permissive i.e. numerically lowest) access
    // across all paths.
    SubobjectAccess = std::min(SubobjectAccess, Path->Access);

    // Determine whether we're looking at a distinct sub-object or not.
    if (SubobjectType.isNull()) {
      // This is the first subobject we've looked at. Record its type.
      SubobjectType = Context.getCanonicalType(PathElement.Base->getType());
      SubobjectNumber = PathElement.SubobjectNumber;
      continue;
    }

    if (SubobjectType
                 != Context.getCanonicalType(PathElement.Base->getType())) {
      // We found members of the given name in two subobjects of
      // different types. If the declaration sets aren't the same, this
      // this lookup is ambiguous.
      if (HasOnlyStaticMembers(Path->Decls.first, Path->Decls.second)) {
        CXXBasePaths::paths_iterator FirstPath = Paths.begin();
        DeclContext::lookup_iterator FirstD = FirstPath->Decls.first;
        DeclContext::lookup_iterator CurrentD = Path->Decls.first;

        while (FirstD != FirstPath->Decls.second &&
               CurrentD != Path->Decls.second) {
         if ((*FirstD)->getUnderlyingDecl()->getCanonicalDecl() !=
             (*CurrentD)->getUnderlyingDecl()->getCanonicalDecl())
           break;

          ++FirstD;
          ++CurrentD;
        }

        if (FirstD == FirstPath->Decls.second &&
            CurrentD == Path->Decls.second)
          continue;
      }

      R.setAmbiguousBaseSubobjectTypes(Paths);
      return true;
    }

    if (SubobjectNumber != PathElement.SubobjectNumber) {
      // We have a different subobject of the same type.

      // C++ [class.member.lookup]p5:
      //   A static member, a nested type or an enumerator defined in
      //   a base class T can unambiguously be found even if an object
      //   has more than one base class subobject of type T.
      if (HasOnlyStaticMembers(Path->Decls.first, Path->Decls.second))
        continue;

      // We have found a nonstatic member name in multiple, distinct
      // subobjects. Name lookup is ambiguous.
      R.setAmbiguousBaseSubobjects(Paths);
      return true;
    }
  }

  // Lookup in a base class succeeded; return these results.

  DeclContext::lookup_iterator I, E;
  for (llvm::tie(I,E) = Paths.front().Decls; I != E; ++I) {
    NamedDecl *D = *I;
    AccessSpecifier AS = CXXRecordDecl::MergeAccess(SubobjectAccess,
                                                    D->getAccess());
    R.addDecl(D, AS);
  }
  R.resolveKind();
  return true;
}

/// @brief Performs name lookup for a name that was parsed in the
/// source code, and may contain a C++ scope specifier.
///
/// This routine is a convenience routine meant to be called from
/// contexts that receive a name and an optional C++ scope specifier
/// (e.g., "N::M::x"). It will then perform either qualified or
/// unqualified name lookup (with LookupQualifiedName or LookupName,
/// respectively) on the given name and return those results.
///
/// @param S        The scope from which unqualified name lookup will
/// begin.
///
/// @param SS       An optional C++ scope-specifier, e.g., "::N::M".
///
/// @param EnteringContext Indicates whether we are going to enter the
/// context of the scope-specifier SS (if present).
///
/// @returns True if any decls were found (but possibly ambiguous)
bool Sema::LookupParsedName(LookupResult &R, Scope *S, CXXScopeSpec *SS,
                            bool AllowBuiltinCreation, bool EnteringContext) {
  if (SS && SS->isInvalid()) {
    // When the scope specifier is invalid, don't even look for
    // anything.
    return false;
  }

  if (SS && SS->isSet()) {
    if (DeclContext *DC = computeDeclContext(*SS, EnteringContext)) {
      // We have resolved the scope specifier to a particular declaration
      // contex, and will perform name lookup in that context.
      if (!DC->isDependentContext() && RequireCompleteDeclContext(*SS, DC))
        return false;

      R.setContextRange(SS->getRange());

      return LookupQualifiedName(R, DC);
    }

    // We could not resolve the scope specified to a specific declaration
    // context, which means that SS refers to an unknown specialization.
    // Name lookup can't find anything in this case.
    return false;
  }

  // Perform unqualified name lookup starting in the given scope.
  return LookupName(R, S, AllowBuiltinCreation);
}


/// @brief Produce a diagnostic describing the ambiguity that resulted
/// from name lookup.
///
/// @param Result       The ambiguous name lookup result.
///
/// @param Name         The name of the entity that name lookup was
/// searching for.
///
/// @param NameLoc      The location of the name within the source code.
///
/// @param LookupRange  A source range that provides more
/// source-location information concerning the lookup itself. For
/// example, this range might highlight a nested-name-specifier that
/// precedes the name.
///
/// @returns true
bool Sema::DiagnoseAmbiguousLookup(LookupResult &Result) {
  assert(Result.isAmbiguous() && "Lookup result must be ambiguous");

  DeclarationName Name = Result.getLookupName();
  SourceLocation NameLoc = Result.getNameLoc();
  SourceRange LookupRange = Result.getContextRange();

  switch (Result.getAmbiguityKind()) {
  case LookupResult::AmbiguousBaseSubobjects: {
    CXXBasePaths *Paths = Result.getBasePaths();
    QualType SubobjectType = Paths->front().back().Base->getType();
    Diag(NameLoc, diag::err_ambiguous_member_multiple_subobjects)
      << Name << SubobjectType << getAmbiguousPathsDisplayString(*Paths)
      << LookupRange;

    DeclContext::lookup_iterator Found = Paths->front().Decls.first;
    while (isa<CXXMethodDecl>(*Found) &&
           cast<CXXMethodDecl>(*Found)->isStatic())
      ++Found;

    Diag((*Found)->getLocation(), diag::note_ambiguous_member_found);

    return true;
  }

  case LookupResult::AmbiguousBaseSubobjectTypes: {
    Diag(NameLoc, diag::err_ambiguous_member_multiple_subobject_types)
      << Name << LookupRange;

    CXXBasePaths *Paths = Result.getBasePaths();
    std::set<Decl *> DeclsPrinted;
    for (CXXBasePaths::paths_iterator Path = Paths->begin(),
                                      PathEnd = Paths->end();
         Path != PathEnd; ++Path) {
      Decl *D = *Path->Decls.first;
      if (DeclsPrinted.insert(D).second)
        Diag(D->getLocation(), diag::note_ambiguous_member_found);
    }

    return true;
  }

  case LookupResult::AmbiguousTagHiding: {
    Diag(NameLoc, diag::err_ambiguous_tag_hiding) << Name << LookupRange;

    llvm::SmallPtrSet<NamedDecl*,8> TagDecls;

    LookupResult::iterator DI, DE = Result.end();
    for (DI = Result.begin(); DI != DE; ++DI)
      if (TagDecl *TD = dyn_cast<TagDecl>(*DI)) {
        TagDecls.insert(TD);
        Diag(TD->getLocation(), diag::note_hidden_tag);
      }

    for (DI = Result.begin(); DI != DE; ++DI)
      if (!isa<TagDecl>(*DI))
        Diag((*DI)->getLocation(), diag::note_hiding_object);

    // For recovery purposes, go ahead and implement the hiding.
    LookupResult::Filter F = Result.makeFilter();
    while (F.hasNext()) {
      if (TagDecls.count(F.next()))
        F.erase();
    }
    F.done();

    return true;
  }

  case LookupResult::AmbiguousReference: {
    Diag(NameLoc, diag::err_ambiguous_reference) << Name << LookupRange;

    LookupResult::iterator DI = Result.begin(), DE = Result.end();
    for (; DI != DE; ++DI)
      Diag((*DI)->getLocation(), diag::note_ambiguous_candidate) << *DI;

    return true;
  }
  }

  llvm_unreachable("unknown ambiguity kind");
  return true;
}

namespace {
  struct AssociatedLookup {
    AssociatedLookup(Sema &S,
                     Sema::AssociatedNamespaceSet &Namespaces,
                     Sema::AssociatedClassSet &Classes)
      : S(S), Namespaces(Namespaces), Classes(Classes) {
    }

    Sema &S;
    Sema::AssociatedNamespaceSet &Namespaces;
    Sema::AssociatedClassSet &Classes;
  };
}

static void
addAssociatedClassesAndNamespaces(AssociatedLookup &Result, QualType T);

static void CollectEnclosingNamespace(Sema::AssociatedNamespaceSet &Namespaces,
                                      DeclContext *Ctx) {
  // Add the associated namespace for this class.

  // We don't use DeclContext::getEnclosingNamespaceContext() as this may
  // be a locally scoped record.

  // We skip out of inline namespaces. The innermost non-inline namespace
  // contains all names of all its nested inline namespaces anyway, so we can
  // replace the entire inline namespace tree with its root.
  while (Ctx->isRecord() || Ctx->isTransparentContext() ||
         Ctx->isInlineNamespace())
    Ctx = Ctx->getParent();

  if (Ctx->isFileContext())
    Namespaces.insert(Ctx->getPrimaryContext());
}

// \brief Add the associated classes and namespaces for argument-dependent
// lookup that involves a template argument (C++ [basic.lookup.koenig]p2).
static void
addAssociatedClassesAndNamespaces(AssociatedLookup &Result,
                                  const TemplateArgument &Arg) {
  // C++ [basic.lookup.koenig]p2, last bullet:
  //   -- [...] ;
  switch (Arg.getKind()) {
    case TemplateArgument::Null:
      break;

    case TemplateArgument::Type:
      // [...] the namespaces and classes associated with the types of the
      // template arguments provided for template type parameters (excluding
      // template template parameters)
      addAssociatedClassesAndNamespaces(Result, Arg.getAsType());
      break;

    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion: {
      // [...] the namespaces in which any template template arguments are
      // defined; and the classes in which any member templates used as
      // template template arguments are defined.
      TemplateName Template = Arg.getAsTemplateOrTemplatePattern();
      if (ClassTemplateDecl *ClassTemplate
                 = dyn_cast<ClassTemplateDecl>(Template.getAsTemplateDecl())) {
        DeclContext *Ctx = ClassTemplate->getDeclContext();
        if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
          Result.Classes.insert(EnclosingClass);
        // Add the associated namespace for this class.
        CollectEnclosingNamespace(Result.Namespaces, Ctx);
      }
      break;
    }

    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
    case TemplateArgument::Expression:
      // [Note: non-type template arguments do not contribute to the set of
      //  associated namespaces. ]
      break;

    case TemplateArgument::Pack:
      for (TemplateArgument::pack_iterator P = Arg.pack_begin(),
                                        PEnd = Arg.pack_end();
           P != PEnd; ++P)
        addAssociatedClassesAndNamespaces(Result, *P);
      break;
  }
}

// \brief Add the associated classes and namespaces for
// argument-dependent lookup with an argument of class type
// (C++ [basic.lookup.koenig]p2).
static void
addAssociatedClassesAndNamespaces(AssociatedLookup &Result,
                                  CXXRecordDecl *Class) {

  // Just silently ignore anything whose name is __va_list_tag.
  if (Class->getDeclName() == Result.S.VAListTagName)
    return;

  // C++ [basic.lookup.koenig]p2:
  //   [...]
  //     -- If T is a class type (including unions), its associated
  //        classes are: the class itself; the class of which it is a
  //        member, if any; and its direct and indirect base
  //        classes. Its associated namespaces are the namespaces in
  //        which its associated classes are defined.

  // Add the class of which it is a member, if any.
  DeclContext *Ctx = Class->getDeclContext();
  if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
    Result.Classes.insert(EnclosingClass);
  // Add the associated namespace for this class.
  CollectEnclosingNamespace(Result.Namespaces, Ctx);

  // Add the class itself. If we've already seen this class, we don't
  // need to visit base classes.
  if (!Result.Classes.insert(Class))
    return;

  // -- If T is a template-id, its associated namespaces and classes are
  //    the namespace in which the template is defined; for member
  //    templates, the member template's class; the namespaces and classes
  //    associated with the types of the template arguments provided for
  //    template type parameters (excluding template template parameters); the
  //    namespaces in which any template template arguments are defined; and
  //    the classes in which any member templates used as template template
  //    arguments are defined. [Note: non-type template arguments do not
  //    contribute to the set of associated namespaces. ]
  if (ClassTemplateSpecializationDecl *Spec
        = dyn_cast<ClassTemplateSpecializationDecl>(Class)) {
    DeclContext *Ctx = Spec->getSpecializedTemplate()->getDeclContext();
    if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
      Result.Classes.insert(EnclosingClass);
    // Add the associated namespace for this class.
    CollectEnclosingNamespace(Result.Namespaces, Ctx);

    const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
    for (unsigned I = 0, N = TemplateArgs.size(); I != N; ++I)
      addAssociatedClassesAndNamespaces(Result, TemplateArgs[I]);
  }

  // Only recurse into base classes for complete types.
  if (!Class->hasDefinition()) {
    // FIXME: we might need to instantiate templates here
    return;
  }

  // Add direct and indirect base classes along with their associated
  // namespaces.
  llvm::SmallVector<CXXRecordDecl *, 32> Bases;
  Bases.push_back(Class);
  while (!Bases.empty()) {
    // Pop this class off the stack.
    Class = Bases.back();
    Bases.pop_back();

    // Visit the base classes.
    for (CXXRecordDecl::base_class_iterator Base = Class->bases_begin(),
                                         BaseEnd = Class->bases_end();
         Base != BaseEnd; ++Base) {
      const RecordType *BaseType = Base->getType()->getAs<RecordType>();
      // In dependent contexts, we do ADL twice, and the first time around,
      // the base type might be a dependent TemplateSpecializationType, or a
      // TemplateTypeParmType. If that happens, simply ignore it.
      // FIXME: If we want to support export, we probably need to add the
      // namespace of the template in a TemplateSpecializationType, or even
      // the classes and namespaces of known non-dependent arguments.
      if (!BaseType)
        continue;
      CXXRecordDecl *BaseDecl = cast<CXXRecordDecl>(BaseType->getDecl());
      if (Result.Classes.insert(BaseDecl)) {
        // Find the associated namespace for this base class.
        DeclContext *BaseCtx = BaseDecl->getDeclContext();
        CollectEnclosingNamespace(Result.Namespaces, BaseCtx);

        // Make sure we visit the bases of this base class.
        if (BaseDecl->bases_begin() != BaseDecl->bases_end())
          Bases.push_back(BaseDecl);
      }
    }
  }
}

// \brief Add the associated classes and namespaces for
// argument-dependent lookup with an argument of type T
// (C++ [basic.lookup.koenig]p2).
static void
addAssociatedClassesAndNamespaces(AssociatedLookup &Result, QualType Ty) {
  // C++ [basic.lookup.koenig]p2:
  //
  //   For each argument type T in the function call, there is a set
  //   of zero or more associated namespaces and a set of zero or more
  //   associated classes to be considered. The sets of namespaces and
  //   classes is determined entirely by the types of the function
  //   arguments (and the namespace of any template template
  //   argument). Typedef names and using-declarations used to specify
  //   the types do not contribute to this set. The sets of namespaces
  //   and classes are determined in the following way:

  llvm::SmallVector<const Type *, 16> Queue;
  const Type *T = Ty->getCanonicalTypeInternal().getTypePtr();

  while (true) {
    switch (T->getTypeClass()) {

#define TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
      // T is canonical.  We can also ignore dependent types because
      // we don't need to do ADL at the definition point, but if we
      // wanted to implement template export (or if we find some other
      // use for associated classes and namespaces...) this would be
      // wrong.
      break;

    //    -- If T is a pointer to U or an array of U, its associated
    //       namespaces and classes are those associated with U.
    case Type::Pointer:
      T = cast<PointerType>(T)->getPointeeType().getTypePtr();
      continue;
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
      T = cast<ArrayType>(T)->getElementType().getTypePtr();
      continue;

    //     -- If T is a fundamental type, its associated sets of
    //        namespaces and classes are both empty.
    case Type::Builtin:
      break;

    //     -- If T is a class type (including unions), its associated
    //        classes are: the class itself; the class of which it is a
    //        member, if any; and its direct and indirect base
    //        classes. Its associated namespaces are the namespaces in
    //        which its associated classes are defined.
    case Type::Record: {
      CXXRecordDecl *Class
        = cast<CXXRecordDecl>(cast<RecordType>(T)->getDecl());
      addAssociatedClassesAndNamespaces(Result, Class);
      break;
    }

    //     -- If T is an enumeration type, its associated namespace is
    //        the namespace in which it is defined. If it is class
    //        member, its associated class is the member's class; else
    //        it has no associated class.
    case Type::Enum: {
      EnumDecl *Enum = cast<EnumType>(T)->getDecl();

      DeclContext *Ctx = Enum->getDeclContext();
      if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
        Result.Classes.insert(EnclosingClass);

      // Add the associated namespace for this class.
      CollectEnclosingNamespace(Result.Namespaces, Ctx);

      break;
    }

    //     -- If T is a function type, its associated namespaces and
    //        classes are those associated with the function parameter
    //        types and those associated with the return type.
    case Type::FunctionProto: {
      const FunctionProtoType *Proto = cast<FunctionProtoType>(T);
      for (FunctionProtoType::arg_type_iterator Arg = Proto->arg_type_begin(),
                                             ArgEnd = Proto->arg_type_end();
             Arg != ArgEnd; ++Arg)
        Queue.push_back(Arg->getTypePtr());
      // fallthrough
    }
    case Type::FunctionNoProto: {
      const FunctionType *FnType = cast<FunctionType>(T);
      T = FnType->getResultType().getTypePtr();
      continue;
    }

    //     -- If T is a pointer to a member function of a class X, its
    //        associated namespaces and classes are those associated
    //        with the function parameter types and return type,
    //        together with those associated with X.
    //
    //     -- If T is a pointer to a data member of class X, its
    //        associated namespaces and classes are those associated
    //        with the member type together with those associated with
    //        X.
    case Type::MemberPointer: {
      const MemberPointerType *MemberPtr = cast<MemberPointerType>(T);

      // Queue up the class type into which this points.
      Queue.push_back(MemberPtr->getClass());

      // And directly continue with the pointee type.
      T = MemberPtr->getPointeeType().getTypePtr();
      continue;
    }

    // As an extension, treat this like a normal pointer.
    case Type::BlockPointer:
      T = cast<BlockPointerType>(T)->getPointeeType().getTypePtr();
      continue;

    // References aren't covered by the standard, but that's such an
    // obvious defect that we cover them anyway.
    case Type::LValueReference:
    case Type::RValueReference:
      T = cast<ReferenceType>(T)->getPointeeType().getTypePtr();
      continue;

    // These are fundamental types.
    case Type::Vector:
    case Type::ExtVector:
    case Type::Complex:
      break;

    // These are ignored by ADL.
    case Type::ObjCObject:
    case Type::ObjCInterface:
    case Type::ObjCObjectPointer:
      break;
    }

    if (Queue.empty()) break;
    T = Queue.back();
    Queue.pop_back();
  }
}

/// \brief Find the associated classes and namespaces for
/// argument-dependent lookup for a call with the given set of
/// arguments.
///
/// This routine computes the sets of associated classes and associated
/// namespaces searched by argument-dependent lookup
/// (C++ [basic.lookup.argdep]) for a given set of arguments.
void
Sema::FindAssociatedClassesAndNamespaces(Expr **Args, unsigned NumArgs,
                                 AssociatedNamespaceSet &AssociatedNamespaces,
                                 AssociatedClassSet &AssociatedClasses) {
  AssociatedNamespaces.clear();
  AssociatedClasses.clear();

  AssociatedLookup Result(*this, AssociatedNamespaces, AssociatedClasses);

  // C++ [basic.lookup.koenig]p2:
  //   For each argument type T in the function call, there is a set
  //   of zero or more associated namespaces and a set of zero or more
  //   associated classes to be considered. The sets of namespaces and
  //   classes is determined entirely by the types of the function
  //   arguments (and the namespace of any template template
  //   argument).
  for (unsigned ArgIdx = 0; ArgIdx != NumArgs; ++ArgIdx) {
    Expr *Arg = Args[ArgIdx];

    if (Arg->getType() != Context.OverloadTy) {
      addAssociatedClassesAndNamespaces(Result, Arg->getType());
      continue;
    }

    // [...] In addition, if the argument is the name or address of a
    // set of overloaded functions and/or function templates, its
    // associated classes and namespaces are the union of those
    // associated with each of the members of the set: the namespace
    // in which the function or function template is defined and the
    // classes and namespaces associated with its (non-dependent)
    // parameter types and return type.
    Arg = Arg->IgnoreParens();
    if (UnaryOperator *unaryOp = dyn_cast<UnaryOperator>(Arg))
      if (unaryOp->getOpcode() == UO_AddrOf)
        Arg = unaryOp->getSubExpr();

    UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(Arg);
    if (!ULE) continue;

    for (UnresolvedSetIterator I = ULE->decls_begin(), E = ULE->decls_end();
           I != E; ++I) {
      // Look through any using declarations to find the underlying function.
      NamedDecl *Fn = (*I)->getUnderlyingDecl();

      FunctionDecl *FDecl = dyn_cast<FunctionDecl>(Fn);
      if (!FDecl)
        FDecl = cast<FunctionTemplateDecl>(Fn)->getTemplatedDecl();

      // Add the classes and namespaces associated with the parameter
      // types and return type of this function.
      addAssociatedClassesAndNamespaces(Result, FDecl->getType());
    }
  }
}

/// IsAcceptableNonMemberOperatorCandidate - Determine whether Fn is
/// an acceptable non-member overloaded operator for a call whose
/// arguments have types T1 (and, if non-empty, T2). This routine
/// implements the check in C++ [over.match.oper]p3b2 concerning
/// enumeration types.
static bool
IsAcceptableNonMemberOperatorCandidate(FunctionDecl *Fn,
                                       QualType T1, QualType T2,
                                       ASTContext &Context) {
  if (T1->isDependentType() || (!T2.isNull() && T2->isDependentType()))
    return true;

  if (T1->isRecordType() || (!T2.isNull() && T2->isRecordType()))
    return true;

  const FunctionProtoType *Proto = Fn->getType()->getAs<FunctionProtoType>();
  if (Proto->getNumArgs() < 1)
    return false;

  if (T1->isEnumeralType()) {
    QualType ArgType = Proto->getArgType(0).getNonReferenceType();
    if (Context.hasSameUnqualifiedType(T1, ArgType))
      return true;
  }

  if (Proto->getNumArgs() < 2)
    return false;

  if (!T2.isNull() && T2->isEnumeralType()) {
    QualType ArgType = Proto->getArgType(1).getNonReferenceType();
    if (Context.hasSameUnqualifiedType(T2, ArgType))
      return true;
  }

  return false;
}

NamedDecl *Sema::LookupSingleName(Scope *S, DeclarationName Name,
                                  SourceLocation Loc,
                                  LookupNameKind NameKind,
                                  RedeclarationKind Redecl) {
  LookupResult R(*this, Name, Loc, NameKind, Redecl);
  LookupName(R, S);
  return R.getAsSingle<NamedDecl>();
}

/// \brief Find the protocol with the given name, if any.
ObjCProtocolDecl *Sema::LookupProtocol(IdentifierInfo *II,
                                       SourceLocation IdLoc) {
  Decl *D = LookupSingleName(TUScope, II, IdLoc,
                             LookupObjCProtocolName);
  return cast_or_null<ObjCProtocolDecl>(D);
}

void Sema::LookupOverloadedOperatorName(OverloadedOperatorKind Op, Scope *S,
                                        QualType T1, QualType T2,
                                        UnresolvedSetImpl &Functions) {
  // C++ [over.match.oper]p3:
  //     -- The set of non-member candidates is the result of the
  //        unqualified lookup of operator@ in the context of the
  //        expression according to the usual rules for name lookup in
  //        unqualified function calls (3.4.2) except that all member
  //        functions are ignored. However, if no operand has a class
  //        type, only those non-member functions in the lookup set
  //        that have a first parameter of type T1 or "reference to
  //        (possibly cv-qualified) T1", when T1 is an enumeration
  //        type, or (if there is a right operand) a second parameter
  //        of type T2 or "reference to (possibly cv-qualified) T2",
  //        when T2 is an enumeration type, are candidate functions.
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);
  LookupResult Operators(*this, OpName, SourceLocation(), LookupOperatorName);
  LookupName(Operators, S);

  assert(!Operators.isAmbiguous() && "Operator lookup cannot be ambiguous");

  if (Operators.empty())
    return;

  for (LookupResult::iterator Op = Operators.begin(), OpEnd = Operators.end();
       Op != OpEnd; ++Op) {
    NamedDecl *Found = (*Op)->getUnderlyingDecl();
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(Found)) {
      if (IsAcceptableNonMemberOperatorCandidate(FD, T1, T2, Context))
        Functions.addDecl(*Op, Op.getAccess()); // FIXME: canonical FD
    } else if (FunctionTemplateDecl *FunTmpl
                 = dyn_cast<FunctionTemplateDecl>(Found)) {
      // FIXME: friend operators?
      // FIXME: do we need to check IsAcceptableNonMemberOperatorCandidate,
      // later?
      if (!FunTmpl->getDeclContext()->isRecord())
        Functions.addDecl(*Op, Op.getAccess());
    }
  }
}

/// \brief Look up the constructors for the given class.
DeclContext::lookup_result Sema::LookupConstructors(CXXRecordDecl *Class) {
  // If the copy constructor has not yet been declared, do so now.
  if (CanDeclareSpecialMemberFunction(Context, Class)) {
    if (!Class->hasDeclaredDefaultConstructor())
      DeclareImplicitDefaultConstructor(Class);
    if (!Class->hasDeclaredCopyConstructor())
      DeclareImplicitCopyConstructor(Class);
  }

  CanQualType T = Context.getCanonicalType(Context.getTypeDeclType(Class));
  DeclarationName Name = Context.DeclarationNames.getCXXConstructorName(T);
  return Class->lookup(Name);
}

/// \brief Look for the destructor of the given class.
///
/// During semantic analysis, this routine should be used in lieu of
/// CXXRecordDecl::getDestructor().
///
/// \returns The destructor for this class.
CXXDestructorDecl *Sema::LookupDestructor(CXXRecordDecl *Class) {
  // If the destructor has not yet been declared, do so now.
  if (CanDeclareSpecialMemberFunction(Context, Class) &&
      !Class->hasDeclaredDestructor())
    DeclareImplicitDestructor(Class);

  return Class->getDestructor();
}

void ADLResult::insert(NamedDecl *New) {
  NamedDecl *&Old = Decls[cast<NamedDecl>(New->getCanonicalDecl())];

  // If we haven't yet seen a decl for this key, or the last decl
  // was exactly this one, we're done.
  if (Old == 0 || Old == New) {
    Old = New;
    return;
  }

  // Otherwise, decide which is a more recent redeclaration.
  FunctionDecl *OldFD, *NewFD;
  if (isa<FunctionTemplateDecl>(New)) {
    OldFD = cast<FunctionTemplateDecl>(Old)->getTemplatedDecl();
    NewFD = cast<FunctionTemplateDecl>(New)->getTemplatedDecl();
  } else {
    OldFD = cast<FunctionDecl>(Old);
    NewFD = cast<FunctionDecl>(New);
  }

  FunctionDecl *Cursor = NewFD;
  while (true) {
    Cursor = Cursor->getPreviousDeclaration();

    // If we got to the end without finding OldFD, OldFD is the newer
    // declaration;  leave things as they are.
    if (!Cursor) return;

    // If we do find OldFD, then NewFD is newer.
    if (Cursor == OldFD) break;

    // Otherwise, keep looking.
  }

  Old = New;
}

void Sema::ArgumentDependentLookup(DeclarationName Name, bool Operator,
                                   Expr **Args, unsigned NumArgs,
                                   ADLResult &Result) {
  // Find all of the associated namespaces and classes based on the
  // arguments we have.
  AssociatedNamespaceSet AssociatedNamespaces;
  AssociatedClassSet AssociatedClasses;
  FindAssociatedClassesAndNamespaces(Args, NumArgs,
                                     AssociatedNamespaces,
                                     AssociatedClasses);

  QualType T1, T2;
  if (Operator) {
    T1 = Args[0]->getType();
    if (NumArgs >= 2)
      T2 = Args[1]->getType();
  }

  // C++ [basic.lookup.argdep]p3:
  //   Let X be the lookup set produced by unqualified lookup (3.4.1)
  //   and let Y be the lookup set produced by argument dependent
  //   lookup (defined as follows). If X contains [...] then Y is
  //   empty. Otherwise Y is the set of declarations found in the
  //   namespaces associated with the argument types as described
  //   below. The set of declarations found by the lookup of the name
  //   is the union of X and Y.
  //
  // Here, we compute Y and add its members to the overloaded
  // candidate set.
  for (AssociatedNamespaceSet::iterator NS = AssociatedNamespaces.begin(),
                                     NSEnd = AssociatedNamespaces.end();
       NS != NSEnd; ++NS) {
    //   When considering an associated namespace, the lookup is the
    //   same as the lookup performed when the associated namespace is
    //   used as a qualifier (3.4.3.2) except that:
    //
    //     -- Any using-directives in the associated namespace are
    //        ignored.
    //
    //     -- Any namespace-scope friend functions declared in
    //        associated classes are visible within their respective
    //        namespaces even if they are not visible during an ordinary
    //        lookup (11.4).
    DeclContext::lookup_iterator I, E;
    for (llvm::tie(I, E) = (*NS)->lookup(Name); I != E; ++I) {
      NamedDecl *D = *I;
      // If the only declaration here is an ordinary friend, consider
      // it only if it was declared in an associated classes.
      if (D->getIdentifierNamespace() == Decl::IDNS_OrdinaryFriend) {
        DeclContext *LexDC = D->getLexicalDeclContext();
        if (!AssociatedClasses.count(cast<CXXRecordDecl>(LexDC)))
          continue;
      }

      if (isa<UsingShadowDecl>(D))
        D = cast<UsingShadowDecl>(D)->getTargetDecl();

      if (isa<FunctionDecl>(D)) {
        if (Operator &&
            !IsAcceptableNonMemberOperatorCandidate(cast<FunctionDecl>(D),
                                                    T1, T2, Context))
          continue;
      } else if (!isa<FunctionTemplateDecl>(D))
        continue;

      Result.insert(D);
    }
  }
}

//----------------------------------------------------------------------------
// Search for all visible declarations.
//----------------------------------------------------------------------------
VisibleDeclConsumer::~VisibleDeclConsumer() { }

namespace {

class ShadowContextRAII;

class VisibleDeclsRecord {
public:
  /// \brief An entry in the shadow map, which is optimized to store a
  /// single declaration (the common case) but can also store a list
  /// of declarations.
  class ShadowMapEntry {
    typedef llvm::SmallVector<NamedDecl *, 4> DeclVector;

    /// \brief Contains either the solitary NamedDecl * or a vector
    /// of declarations.
    llvm::PointerUnion<NamedDecl *, DeclVector*> DeclOrVector;

  public:
    ShadowMapEntry() : DeclOrVector() { }

    void Add(NamedDecl *ND);
    void Destroy();

    // Iteration.
    typedef NamedDecl * const *iterator;
    iterator begin();
    iterator end();
  };

private:
  /// \brief A mapping from declaration names to the declarations that have
  /// this name within a particular scope.
  typedef llvm::DenseMap<DeclarationName, ShadowMapEntry> ShadowMap;

  /// \brief A list of shadow maps, which is used to model name hiding.
  std::list<ShadowMap> ShadowMaps;

  /// \brief The declaration contexts we have already visited.
  llvm::SmallPtrSet<DeclContext *, 8> VisitedContexts;

  friend class ShadowContextRAII;

public:
  /// \brief Determine whether we have already visited this context
  /// (and, if not, note that we are going to visit that context now).
  bool visitedContext(DeclContext *Ctx) {
    return !VisitedContexts.insert(Ctx);
  }

  bool alreadyVisitedContext(DeclContext *Ctx) {
    return VisitedContexts.count(Ctx);
  }

  /// \brief Determine whether the given declaration is hidden in the
  /// current scope.
  ///
  /// \returns the declaration that hides the given declaration, or
  /// NULL if no such declaration exists.
  NamedDecl *checkHidden(NamedDecl *ND);

  /// \brief Add a declaration to the current shadow map.
  void add(NamedDecl *ND) { ShadowMaps.back()[ND->getDeclName()].Add(ND); }
};

/// \brief RAII object that records when we've entered a shadow context.
class ShadowContextRAII {
  VisibleDeclsRecord &Visible;

  typedef VisibleDeclsRecord::ShadowMap ShadowMap;

public:
  ShadowContextRAII(VisibleDeclsRecord &Visible) : Visible(Visible) {
    Visible.ShadowMaps.push_back(ShadowMap());
  }

  ~ShadowContextRAII() {
    for (ShadowMap::iterator E = Visible.ShadowMaps.back().begin(),
                          EEnd = Visible.ShadowMaps.back().end();
         E != EEnd;
         ++E)
      E->second.Destroy();

    Visible.ShadowMaps.pop_back();
  }
};

} // end anonymous namespace

void VisibleDeclsRecord::ShadowMapEntry::Add(NamedDecl *ND) {
  if (DeclOrVector.isNull()) {
    // 0 - > 1 elements: just set the single element information.
    DeclOrVector = ND;
    return;
  }

  if (NamedDecl *PrevND = DeclOrVector.dyn_cast<NamedDecl *>()) {
    // 1 -> 2 elements: create the vector of results and push in the
    // existing declaration.
    DeclVector *Vec = new DeclVector;
    Vec->push_back(PrevND);
    DeclOrVector = Vec;
  }

  // Add the new element to the end of the vector.
  DeclOrVector.get<DeclVector*>()->push_back(ND);
}

void VisibleDeclsRecord::ShadowMapEntry::Destroy() {
  if (DeclVector *Vec = DeclOrVector.dyn_cast<DeclVector *>()) {
    delete Vec;
    DeclOrVector = ((NamedDecl *)0);
  }
}

VisibleDeclsRecord::ShadowMapEntry::iterator
VisibleDeclsRecord::ShadowMapEntry::begin() {
  if (DeclOrVector.isNull())
    return 0;

  if (DeclOrVector.is<NamedDecl *>())
    return DeclOrVector.getAddrOf<NamedDecl *>();

  return DeclOrVector.get<DeclVector *>()->begin();
}

VisibleDeclsRecord::ShadowMapEntry::iterator
VisibleDeclsRecord::ShadowMapEntry::end() {
  if (DeclOrVector.isNull())
    return 0;

  if (DeclOrVector.dyn_cast<NamedDecl *>())
    return &reinterpret_cast<NamedDecl*&>(DeclOrVector) + 1;

  return DeclOrVector.get<DeclVector *>()->end();
}

NamedDecl *VisibleDeclsRecord::checkHidden(NamedDecl *ND) {
  // Look through using declarations.
  ND = ND->getUnderlyingDecl();

  unsigned IDNS = ND->getIdentifierNamespace();
  std::list<ShadowMap>::reverse_iterator SM = ShadowMaps.rbegin();
  for (std::list<ShadowMap>::reverse_iterator SMEnd = ShadowMaps.rend();
       SM != SMEnd; ++SM) {
    ShadowMap::iterator Pos = SM->find(ND->getDeclName());
    if (Pos == SM->end())
      continue;

    for (ShadowMapEntry::iterator I = Pos->second.begin(),
                               IEnd = Pos->second.end();
         I != IEnd; ++I) {
      // A tag declaration does not hide a non-tag declaration.
      if ((*I)->hasTagIdentifierNamespace() &&
          (IDNS & (Decl::IDNS_Member | Decl::IDNS_Ordinary |
                   Decl::IDNS_ObjCProtocol)))
        continue;

      // Protocols are in distinct namespaces from everything else.
      if ((((*I)->getIdentifierNamespace() & Decl::IDNS_ObjCProtocol)
           || (IDNS & Decl::IDNS_ObjCProtocol)) &&
          (*I)->getIdentifierNamespace() != IDNS)
        continue;

      // Functions and function templates in the same scope overload
      // rather than hide.  FIXME: Look for hiding based on function
      // signatures!
      if ((*I)->isFunctionOrFunctionTemplate() &&
          ND->isFunctionOrFunctionTemplate() &&
          SM == ShadowMaps.rbegin())
        continue;

      // We've found a declaration that hides this one.
      return *I;
    }
  }

  return 0;
}

static void LookupVisibleDecls(DeclContext *Ctx, LookupResult &Result,
                               bool QualifiedNameLookup,
                               bool InBaseClass,
                               VisibleDeclConsumer &Consumer,
                               VisibleDeclsRecord &Visited) {
  if (!Ctx)
    return;

  // Make sure we don't visit the same context twice.
  if (Visited.visitedContext(Ctx->getPrimaryContext()))
    return;

  if (CXXRecordDecl *Class = dyn_cast<CXXRecordDecl>(Ctx))
    Result.getSema().ForceDeclarationOfImplicitMembers(Class);

  // Enumerate all of the results in this context.
  for (DeclContext *CurCtx = Ctx->getPrimaryContext(); CurCtx;
       CurCtx = CurCtx->getNextContext()) {
    for (DeclContext::decl_iterator D = CurCtx->decls_begin(),
                                 DEnd = CurCtx->decls_end();
         D != DEnd; ++D) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D)) {
        if (Result.isAcceptableDecl(ND)) {
          Consumer.FoundDecl(ND, Visited.checkHidden(ND), InBaseClass);
          Visited.add(ND);
        }
      } else if (ObjCForwardProtocolDecl *ForwardProto
                                      = dyn_cast<ObjCForwardProtocolDecl>(*D)) {
        for (ObjCForwardProtocolDecl::protocol_iterator
                  P = ForwardProto->protocol_begin(),
               PEnd = ForwardProto->protocol_end();
             P != PEnd;
             ++P) {
          if (Result.isAcceptableDecl(*P)) {
            Consumer.FoundDecl(*P, Visited.checkHidden(*P), InBaseClass);
            Visited.add(*P);
          }
        }
      } else if (ObjCClassDecl *Class = dyn_cast<ObjCClassDecl>(*D)) {
        for (ObjCClassDecl::iterator I = Class->begin(), IEnd = Class->end();
             I != IEnd; ++I) {
          ObjCInterfaceDecl *IFace = I->getInterface();
          if (Result.isAcceptableDecl(IFace)) {
            Consumer.FoundDecl(IFace, Visited.checkHidden(IFace), InBaseClass);
            Visited.add(IFace);
          }
        }
      }
      
      // Visit transparent contexts and inline namespaces inside this context.
      if (DeclContext *InnerCtx = dyn_cast<DeclContext>(*D)) {
        if (InnerCtx->isTransparentContext() || InnerCtx->isInlineNamespace())
          LookupVisibleDecls(InnerCtx, Result, QualifiedNameLookup, InBaseClass,
                             Consumer, Visited);
      }
    }
  }

  // Traverse using directives for qualified name lookup.
  if (QualifiedNameLookup) {
    ShadowContextRAII Shadow(Visited);
    DeclContext::udir_iterator I, E;
    for (llvm::tie(I, E) = Ctx->getUsingDirectives(); I != E; ++I) {
      LookupVisibleDecls((*I)->getNominatedNamespace(), Result,
                         QualifiedNameLookup, InBaseClass, Consumer, Visited);
    }
  }

  // Traverse the contexts of inherited C++ classes.
  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Ctx)) {
    if (!Record->hasDefinition())
      return;

    for (CXXRecordDecl::base_class_iterator B = Record->bases_begin(),
                                         BEnd = Record->bases_end();
         B != BEnd; ++B) {
      QualType BaseType = B->getType();

      // Don't look into dependent bases, because name lookup can't look
      // there anyway.
      if (BaseType->isDependentType())
        continue;

      const RecordType *Record = BaseType->getAs<RecordType>();
      if (!Record)
        continue;

      // FIXME: It would be nice to be able to determine whether referencing
      // a particular member would be ambiguous. For example, given
      //
      //   struct A { int member; };
      //   struct B { int member; };
      //   struct C : A, B { };
      //
      //   void f(C *c) { c->### }
      //
      // accessing 'member' would result in an ambiguity. However, we
      // could be smart enough to qualify the member with the base
      // class, e.g.,
      //
      //   c->B::member
      //
      // or
      //
      //   c->A::member

      // Find results in this base class (and its bases).
      ShadowContextRAII Shadow(Visited);
      LookupVisibleDecls(Record->getDecl(), Result, QualifiedNameLookup,
                         true, Consumer, Visited);
    }
  }

  // Traverse the contexts of Objective-C classes.
  if (ObjCInterfaceDecl *IFace = dyn_cast<ObjCInterfaceDecl>(Ctx)) {
    // Traverse categories.
    for (ObjCCategoryDecl *Category = IFace->getCategoryList();
         Category; Category = Category->getNextClassCategory()) {
      ShadowContextRAII Shadow(Visited);
      LookupVisibleDecls(Category, Result, QualifiedNameLookup, false,
                         Consumer, Visited);
    }

    // Traverse protocols.
    for (ObjCInterfaceDecl::all_protocol_iterator
         I = IFace->all_referenced_protocol_begin(),
         E = IFace->all_referenced_protocol_end(); I != E; ++I) {
      ShadowContextRAII Shadow(Visited);
      LookupVisibleDecls(*I, Result, QualifiedNameLookup, false, Consumer,
                         Visited);
    }

    // Traverse the superclass.
    if (IFace->getSuperClass()) {
      ShadowContextRAII Shadow(Visited);
      LookupVisibleDecls(IFace->getSuperClass(), Result, QualifiedNameLookup,
                         true, Consumer, Visited);
    }

    // If there is an implementation, traverse it. We do this to find
    // synthesized ivars.
    if (IFace->getImplementation()) {
      ShadowContextRAII Shadow(Visited);
      LookupVisibleDecls(IFace->getImplementation(), Result,
                         QualifiedNameLookup, true, Consumer, Visited);
    }
  } else if (ObjCProtocolDecl *Protocol = dyn_cast<ObjCProtocolDecl>(Ctx)) {
    for (ObjCProtocolDecl::protocol_iterator I = Protocol->protocol_begin(),
           E = Protocol->protocol_end(); I != E; ++I) {
      ShadowContextRAII Shadow(Visited);
      LookupVisibleDecls(*I, Result, QualifiedNameLookup, false, Consumer,
                         Visited);
    }
  } else if (ObjCCategoryDecl *Category = dyn_cast<ObjCCategoryDecl>(Ctx)) {
    for (ObjCCategoryDecl::protocol_iterator I = Category->protocol_begin(),
           E = Category->protocol_end(); I != E; ++I) {
      ShadowContextRAII Shadow(Visited);
      LookupVisibleDecls(*I, Result, QualifiedNameLookup, false, Consumer,
                         Visited);
    }

    // If there is an implementation, traverse it.
    if (Category->getImplementation()) {
      ShadowContextRAII Shadow(Visited);
      LookupVisibleDecls(Category->getImplementation(), Result,
                         QualifiedNameLookup, true, Consumer, Visited);
    }
  }
}

static void LookupVisibleDecls(Scope *S, LookupResult &Result,
                               UnqualUsingDirectiveSet &UDirs,
                               VisibleDeclConsumer &Consumer,
                               VisibleDeclsRecord &Visited) {
  if (!S)
    return;

  if (!S->getEntity() ||
      (!S->getParent() &&
       !Visited.alreadyVisitedContext((DeclContext *)S->getEntity())) ||
      ((DeclContext *)S->getEntity())->isFunctionOrMethod()) {
    // Walk through the declarations in this Scope.
    for (Scope::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
         D != DEnd; ++D) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D))
        if (Result.isAcceptableDecl(ND)) {
          Consumer.FoundDecl(ND, Visited.checkHidden(ND), false);
          Visited.add(ND);
        }
    }
  }

  // FIXME: C++ [temp.local]p8
  DeclContext *Entity = 0;
  if (S->getEntity()) {
    // Look into this scope's declaration context, along with any of its
    // parent lookup contexts (e.g., enclosing classes), up to the point
    // where we hit the context stored in the next outer scope.
    Entity = (DeclContext *)S->getEntity();
    DeclContext *OuterCtx = findOuterContext(S).first; // FIXME

    for (DeclContext *Ctx = Entity; Ctx && !Ctx->Equals(OuterCtx);
         Ctx = Ctx->getLookupParent()) {
      if (ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(Ctx)) {
        if (Method->isInstanceMethod()) {
          // For instance methods, look for ivars in the method's interface.
          LookupResult IvarResult(Result.getSema(), Result.getLookupName(),
                                  Result.getNameLoc(), Sema::LookupMemberName);
          if (ObjCInterfaceDecl *IFace = Method->getClassInterface()) {
            LookupVisibleDecls(IFace, IvarResult, /*QualifiedNameLookup=*/false,
                               /*InBaseClass=*/false, Consumer, Visited);

            // Look for properties from which we can synthesize ivars, if
            // permitted.
            if (Result.getSema().getLangOptions().ObjCNonFragileABI2 &&
                IFace->getImplementation() &&
                Result.getLookupKind() == Sema::LookupOrdinaryName) {
              for (ObjCInterfaceDecl::prop_iterator
                        P = IFace->prop_begin(),
                     PEnd = IFace->prop_end();
                   P != PEnd; ++P) {
                if (Result.getSema().canSynthesizeProvisionalIvar(*P) &&
                    !IFace->lookupInstanceVariable((*P)->getIdentifier())) {
                  Consumer.FoundDecl(*P, Visited.checkHidden(*P), false);
                  Visited.add(*P);
                }
              }
            }
          }
        }

        // We've already performed all of the name lookup that we need
        // to for Objective-C methods; the next context will be the
        // outer scope.
        break;
      }

      if (Ctx->isFunctionOrMethod())
        continue;

      LookupVisibleDecls(Ctx, Result, /*QualifiedNameLookup=*/false,
                         /*InBaseClass=*/false, Consumer, Visited);
    }
  } else if (!S->getParent()) {
    // Look into the translation unit scope. We walk through the translation
    // unit's declaration context, because the Scope itself won't have all of
    // the declarations if we loaded a precompiled header.
    // FIXME: We would like the translation unit's Scope object to point to the
    // translation unit, so we don't need this special "if" branch. However,
    // doing so would force the normal C++ name-lookup code to look into the
    // translation unit decl when the IdentifierInfo chains would suffice.
    // Once we fix that problem (which is part of a more general "don't look
    // in DeclContexts unless we have to" optimization), we can eliminate this.
    Entity = Result.getSema().Context.getTranslationUnitDecl();
    LookupVisibleDecls(Entity, Result, /*QualifiedNameLookup=*/false,
                       /*InBaseClass=*/false, Consumer, Visited);
  }

  if (Entity) {
    // Lookup visible declarations in any namespaces found by using
    // directives.
    UnqualUsingDirectiveSet::const_iterator UI, UEnd;
    llvm::tie(UI, UEnd) = UDirs.getNamespacesFor(Entity);
    for (; UI != UEnd; ++UI)
      LookupVisibleDecls(const_cast<DeclContext *>(UI->getNominatedNamespace()),
                         Result, /*QualifiedNameLookup=*/false,
                         /*InBaseClass=*/false, Consumer, Visited);
  }

  // Lookup names in the parent scope.
  ShadowContextRAII Shadow(Visited);
  LookupVisibleDecls(S->getParent(), Result, UDirs, Consumer, Visited);
}

void Sema::LookupVisibleDecls(Scope *S, LookupNameKind Kind,
                              VisibleDeclConsumer &Consumer,
                              bool IncludeGlobalScope) {
  // Determine the set of using directives available during
  // unqualified name lookup.
  Scope *Initial = S;
  UnqualUsingDirectiveSet UDirs;
  if (getLangOptions().CPlusPlus) {
    // Find the first namespace or translation-unit scope.
    while (S && !isNamespaceOrTranslationUnitScope(S))
      S = S->getParent();

    UDirs.visitScopeChain(Initial, S);
  }
  UDirs.done();

  // Look for visible declarations.
  LookupResult Result(*this, DeclarationName(), SourceLocation(), Kind);
  VisibleDeclsRecord Visited;
  if (!IncludeGlobalScope)
    Visited.visitedContext(Context.getTranslationUnitDecl());
  ShadowContextRAII Shadow(Visited);
  ::LookupVisibleDecls(Initial, Result, UDirs, Consumer, Visited);
}

void Sema::LookupVisibleDecls(DeclContext *Ctx, LookupNameKind Kind,
                              VisibleDeclConsumer &Consumer,
                              bool IncludeGlobalScope) {
  LookupResult Result(*this, DeclarationName(), SourceLocation(), Kind);
  VisibleDeclsRecord Visited;
  if (!IncludeGlobalScope)
    Visited.visitedContext(Context.getTranslationUnitDecl());
  ShadowContextRAII Shadow(Visited);
  ::LookupVisibleDecls(Ctx, Result, /*QualifiedNameLookup=*/true,
                       /*InBaseClass=*/false, Consumer, Visited);
}

/// LookupOrCreateLabel - Do a name lookup of a label with the specified name.
/// If GnuLabelLoc is a valid source location, then this is a definition
/// of an __label__ label name, otherwise it is a normal label definition
/// or use.
LabelDecl *Sema::LookupOrCreateLabel(IdentifierInfo *II, SourceLocation Loc,
                                     SourceLocation GnuLabelLoc) {
  // Do a lookup to see if we have a label with this name already.
  NamedDecl *Res = 0;

  if (GnuLabelLoc.isValid()) {
    // Local label definitions always shadow existing labels.
    Res = LabelDecl::Create(Context, CurContext, Loc, II, GnuLabelLoc);
    Scope *S = CurScope;
    PushOnScopeChains(Res, S, true);
    return cast<LabelDecl>(Res);
  }

  // Not a GNU local label.
  Res = LookupSingleName(CurScope, II, Loc, LookupLabel, NotForRedeclaration);
  // If we found a label, check to see if it is in the same context as us.
  // When in a Block, we don't want to reuse a label in an enclosing function.
  if (Res && Res->getDeclContext() != CurContext)
    Res = 0;
  if (Res == 0) {
    // If not forward referenced or defined already, create the backing decl.
    Res = LabelDecl::Create(Context, CurContext, Loc, II);
    Scope *S = CurScope->getFnParent();
    assert(S && "Not in a function?");
    PushOnScopeChains(Res, S, true);
  }
  return cast<LabelDecl>(Res);
}

//===----------------------------------------------------------------------===//
// Typo correction
//===----------------------------------------------------------------------===//

namespace {
class TypoCorrectionConsumer : public VisibleDeclConsumer {
  /// \brief The name written that is a typo in the source.
  llvm::StringRef Typo;

  /// \brief The results found that have the smallest edit distance
  /// found (so far) with the typo name.
  ///
  /// The boolean value indicates whether there is a keyword with this name.
  llvm::StringMap<bool, llvm::BumpPtrAllocator> BestResults;

  /// \brief The best edit distance found so far.
  unsigned BestEditDistance;

public:
  explicit TypoCorrectionConsumer(IdentifierInfo *Typo)
    : Typo(Typo->getName()),
      BestEditDistance((std::numeric_limits<unsigned>::max)()) { }

  virtual void FoundDecl(NamedDecl *ND, NamedDecl *Hiding, bool InBaseClass);
  void FoundName(llvm::StringRef Name);
  void addKeywordResult(ASTContext &Context, llvm::StringRef Keyword);

  typedef llvm::StringMap<bool, llvm::BumpPtrAllocator>::iterator iterator;
  iterator begin() { return BestResults.begin(); }
  iterator end()  { return BestResults.end(); }
  void erase(iterator I) { BestResults.erase(I); }
  unsigned size() const { return BestResults.size(); }
  bool empty() const { return BestResults.empty(); }

  bool &operator[](llvm::StringRef Name) {
    return BestResults[Name];
  }

  unsigned getBestEditDistance() const { return BestEditDistance; }
};

}

void TypoCorrectionConsumer::FoundDecl(NamedDecl *ND, NamedDecl *Hiding,
                                       bool InBaseClass) {
  // Don't consider hidden names for typo correction.
  if (Hiding)
    return;

  // Only consider entities with identifiers for names, ignoring
  // special names (constructors, overloaded operators, selectors,
  // etc.).
  IdentifierInfo *Name = ND->getIdentifier();
  if (!Name)
    return;

  FoundName(Name->getName());
}

void TypoCorrectionConsumer::FoundName(llvm::StringRef Name) {
  using namespace std;

  // Use a simple length-based heuristic to determine the minimum possible
  // edit distance. If the minimum isn't good enough, bail out early.
  unsigned MinED = abs((int)Name.size() - (int)Typo.size());
  if (MinED > BestEditDistance || (MinED && Typo.size() / MinED < 3))
    return;

  // Compute an upper bound on the allowable edit distance, so that the
  // edit-distance algorithm can short-circuit.
  unsigned UpperBound = min(unsigned((Typo.size() + 2) / 3), BestEditDistance);

  // Compute the edit distance between the typo and the name of this
  // entity. If this edit distance is not worse than the best edit
  // distance we've seen so far, add it to the list of results.
  unsigned ED = Typo.edit_distance(Name, true, UpperBound);
  if (ED == 0)
    return;

  if (ED < BestEditDistance) {
    // This result is better than any we've seen before; clear out
    // the previous results.
    BestResults.clear();
    BestEditDistance = ED;
  } else if (ED > BestEditDistance) {
    // This result is worse than the best results we've seen so far;
    // ignore it.
    return;
  }

  // Add this name to the list of results. By not assigning a value, we
  // keep the current value if we've seen this name before (either as a
  // keyword or as a declaration), or get the default value (not a keyword)
  // if we haven't seen it before.
  (void)BestResults[Name];
}

void TypoCorrectionConsumer::addKeywordResult(ASTContext &Context,
                                              llvm::StringRef Keyword) {
  // Compute the edit distance between the typo and this keyword.
  // If this edit distance is not worse than the best edit
  // distance we've seen so far, add it to the list of results.
  unsigned ED = Typo.edit_distance(Keyword);
  if (ED < BestEditDistance) {
    BestResults.clear();
    BestEditDistance = ED;
  } else if (ED > BestEditDistance) {
    // This result is worse than the best results we've seen so far;
    // ignore it.
    return;
  }

  BestResults[Keyword] = true;
}

/// \brief Perform name lookup for a possible result for typo correction.
static void LookupPotentialTypoResult(Sema &SemaRef,
                                      LookupResult &Res,
                                      IdentifierInfo *Name,
                                      Scope *S, CXXScopeSpec *SS,
                                      DeclContext *MemberContext,
                                      bool EnteringContext,
                                      Sema::CorrectTypoContext CTC) {
  Res.suppressDiagnostics();
  Res.clear();
  Res.setLookupName(Name);
  if (MemberContext) {
    if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(MemberContext)) {
      if (CTC == Sema::CTC_ObjCIvarLookup) {
        if (ObjCIvarDecl *Ivar = Class->lookupInstanceVariable(Name)) {
          Res.addDecl(Ivar);
          Res.resolveKind();
          return;
        }
      }

      if (ObjCPropertyDecl *Prop = Class->FindPropertyDeclaration(Name)) {
        Res.addDecl(Prop);
        Res.resolveKind();
        return;
      }
    }

    SemaRef.LookupQualifiedName(Res, MemberContext);
    return;
  }

  SemaRef.LookupParsedName(Res, S, SS, /*AllowBuiltinCreation=*/false,
                           EnteringContext);

  // Fake ivar lookup; this should really be part of
  // LookupParsedName.
  if (ObjCMethodDecl *Method = SemaRef.getCurMethodDecl()) {
    if (Method->isInstanceMethod() && Method->getClassInterface() &&
        (Res.empty() ||
         (Res.isSingleResult() &&
          Res.getFoundDecl()->isDefinedOutsideFunctionOrMethod()))) {
       if (ObjCIvarDecl *IV
             = Method->getClassInterface()->lookupInstanceVariable(Name)) {
         Res.addDecl(IV);
         Res.resolveKind();
       }
     }
  }
}

/// \brief Try to "correct" a typo in the source code by finding
/// visible declarations whose names are similar to the name that was
/// present in the source code.
///
/// \param Res the \c LookupResult structure that contains the name
/// that was present in the source code along with the name-lookup
/// criteria used to search for the name. On success, this structure
/// will contain the results of name lookup.
///
/// \param S the scope in which name lookup occurs.
///
/// \param SS the nested-name-specifier that precedes the name we're
/// looking for, if present.
///
/// \param MemberContext if non-NULL, the context in which to look for
/// a member access expression.
///
/// \param EnteringContext whether we're entering the context described by
/// the nested-name-specifier SS.
///
/// \param CTC The context in which typo correction occurs, which impacts the
/// set of keywords permitted.
///
/// \param OPT when non-NULL, the search for visible declarations will
/// also walk the protocols in the qualified interfaces of \p OPT.
///
/// \returns the corrected name if the typo was corrected, otherwise returns an
/// empty \c DeclarationName. When a typo was corrected, the result structure
/// may contain the results of name lookup for the correct name or it may be
/// empty.
DeclarationName Sema::CorrectTypo(LookupResult &Res, Scope *S, CXXScopeSpec *SS,
                                  DeclContext *MemberContext,
                                  bool EnteringContext,
                                  CorrectTypoContext CTC,
                                  const ObjCObjectPointerType *OPT) {
  if (Diags.hasFatalErrorOccurred() || !getLangOptions().SpellChecking)
    return DeclarationName();

  // We only attempt to correct typos for identifiers.
  IdentifierInfo *Typo = Res.getLookupName().getAsIdentifierInfo();
  if (!Typo)
    return DeclarationName();

  // If the scope specifier itself was invalid, don't try to correct
  // typos.
  if (SS && SS->isInvalid())
    return DeclarationName();

  // Never try to correct typos during template deduction or
  // instantiation.
  if (!ActiveTemplateInstantiations.empty())
    return DeclarationName();

  TypoCorrectionConsumer Consumer(Typo);

  // Perform name lookup to find visible, similarly-named entities.
  bool IsUnqualifiedLookup = false;
  if (MemberContext) {
    LookupVisibleDecls(MemberContext, Res.getLookupKind(), Consumer);

    // Look in qualified interfaces.
    if (OPT) {
      for (ObjCObjectPointerType::qual_iterator
             I = OPT->qual_begin(), E = OPT->qual_end();
           I != E; ++I)
        LookupVisibleDecls(*I, Res.getLookupKind(), Consumer);
    }
  } else if (SS && SS->isSet()) {
    DeclContext *DC = computeDeclContext(*SS, EnteringContext);
    if (!DC)
      return DeclarationName();

    // Provide a stop gap for files that are just seriously broken.  Trying
    // to correct all typos can turn into a HUGE performance penalty, causing
    // some files to take minutes to get rejected by the parser.
    if (TyposCorrected + UnqualifiedTyposCorrected.size() >= 20)
      return DeclarationName();
    ++TyposCorrected;

    LookupVisibleDecls(DC, Res.getLookupKind(), Consumer);
  } else {
    IsUnqualifiedLookup = true;
    UnqualifiedTyposCorrectedMap::iterator Cached
      = UnqualifiedTyposCorrected.find(Typo);
    if (Cached == UnqualifiedTyposCorrected.end()) {
      // Provide a stop gap for files that are just seriously broken.  Trying
      // to correct all typos can turn into a HUGE performance penalty, causing
      // some files to take minutes to get rejected by the parser.
      if (TyposCorrected + UnqualifiedTyposCorrected.size() >= 20)
        return DeclarationName();

      // For unqualified lookup, look through all of the names that we have
      // seen in this translation unit.
      for (IdentifierTable::iterator I = Context.Idents.begin(),
                                  IEnd = Context.Idents.end();
           I != IEnd; ++I)
        Consumer.FoundName(I->getKey());

      // Walk through identifiers in external identifier sources.
      if (IdentifierInfoLookup *External
                              = Context.Idents.getExternalIdentifierLookup()) {
        llvm::OwningPtr<IdentifierIterator> Iter(External->getIdentifiers());
        do {
          llvm::StringRef Name = Iter->Next();
          if (Name.empty())
            break;

          Consumer.FoundName(Name);
        } while (true);
      }
    } else {
      // Use the cached value, unless it's a keyword. In the keyword case, we'll
      // end up adding the keyword below.
      if (Cached->second.first.empty())
        return DeclarationName();

      if (!Cached->second.second)
        Consumer.FoundName(Cached->second.first);
    }
  }

  // Add context-dependent keywords.
  bool WantTypeSpecifiers = false;
  bool WantExpressionKeywords = false;
  bool WantCXXNamedCasts = false;
  bool WantRemainingKeywords = false;
  switch (CTC) {
    case CTC_Unknown:
      WantTypeSpecifiers = true;
      WantExpressionKeywords = true;
      WantCXXNamedCasts = true;
      WantRemainingKeywords = true;

      if (ObjCMethodDecl *Method = getCurMethodDecl())
        if (Method->getClassInterface() &&
            Method->getClassInterface()->getSuperClass())
          Consumer.addKeywordResult(Context, "super");

      break;

    case CTC_NoKeywords:
      break;

    case CTC_Type:
      WantTypeSpecifiers = true;
      break;

    case CTC_ObjCMessageReceiver:
      Consumer.addKeywordResult(Context, "super");
      // Fall through to handle message receivers like expressions.

    case CTC_Expression:
      if (getLangOptions().CPlusPlus)
        WantTypeSpecifiers = true;
      WantExpressionKeywords = true;
      // Fall through to get C++ named casts.

    case CTC_CXXCasts:
      WantCXXNamedCasts = true;
      break;

    case CTC_ObjCPropertyLookup:
      // FIXME: Add "isa"?
      break;

    case CTC_MemberLookup:
      if (getLangOptions().CPlusPlus)
        Consumer.addKeywordResult(Context, "template");
      break;

    case CTC_ObjCIvarLookup:
      break;
  }

  if (WantTypeSpecifiers) {
    // Add type-specifier keywords to the set of results.
    const char *CTypeSpecs[] = {
      "char", "const", "double", "enum", "float", "int", "long", "short",
      "signed", "struct", "union", "unsigned", "void", "volatile", "_Bool",
      "_Complex", "_Imaginary",
      // storage-specifiers as well
      "extern", "inline", "static", "typedef"
    };

    const unsigned NumCTypeSpecs = sizeof(CTypeSpecs) / sizeof(CTypeSpecs[0]);
    for (unsigned I = 0; I != NumCTypeSpecs; ++I)
      Consumer.addKeywordResult(Context, CTypeSpecs[I]);

    if (getLangOptions().C99)
      Consumer.addKeywordResult(Context, "restrict");
    if (getLangOptions().Bool || getLangOptions().CPlusPlus)
      Consumer.addKeywordResult(Context, "bool");

    if (getLangOptions().CPlusPlus) {
      Consumer.addKeywordResult(Context, "class");
      Consumer.addKeywordResult(Context, "typename");
      Consumer.addKeywordResult(Context, "wchar_t");

      if (getLangOptions().CPlusPlus0x) {
        Consumer.addKeywordResult(Context, "char16_t");
        Consumer.addKeywordResult(Context, "char32_t");
        Consumer.addKeywordResult(Context, "constexpr");
        Consumer.addKeywordResult(Context, "decltype");
        Consumer.addKeywordResult(Context, "thread_local");
      }
    }

    if (getLangOptions().GNUMode)
      Consumer.addKeywordResult(Context, "typeof");
  }

  if (WantCXXNamedCasts && getLangOptions().CPlusPlus) {
    Consumer.addKeywordResult(Context, "const_cast");
    Consumer.addKeywordResult(Context, "dynamic_cast");
    Consumer.addKeywordResult(Context, "reinterpret_cast");
    Consumer.addKeywordResult(Context, "static_cast");
  }

  if (WantExpressionKeywords) {
    Consumer.addKeywordResult(Context, "sizeof");
    if (getLangOptions().Bool || getLangOptions().CPlusPlus) {
      Consumer.addKeywordResult(Context, "false");
      Consumer.addKeywordResult(Context, "true");
    }

    if (getLangOptions().CPlusPlus) {
      const char *CXXExprs[] = {
        "delete", "new", "operator", "throw", "typeid"
      };
      const unsigned NumCXXExprs = sizeof(CXXExprs) / sizeof(CXXExprs[0]);
      for (unsigned I = 0; I != NumCXXExprs; ++I)
        Consumer.addKeywordResult(Context, CXXExprs[I]);

      if (isa<CXXMethodDecl>(CurContext) &&
          cast<CXXMethodDecl>(CurContext)->isInstance())
        Consumer.addKeywordResult(Context, "this");

      if (getLangOptions().CPlusPlus0x) {
        Consumer.addKeywordResult(Context, "alignof");
        Consumer.addKeywordResult(Context, "nullptr");
      }
    }
  }

  if (WantRemainingKeywords) {
    if (getCurFunctionOrMethodDecl() || getCurBlock()) {
      // Statements.
      const char *CStmts[] = {
        "do", "else", "for", "goto", "if", "return", "switch", "while" };
      const unsigned NumCStmts = sizeof(CStmts) / sizeof(CStmts[0]);
      for (unsigned I = 0; I != NumCStmts; ++I)
        Consumer.addKeywordResult(Context, CStmts[I]);

      if (getLangOptions().CPlusPlus) {
        Consumer.addKeywordResult(Context, "catch");
        Consumer.addKeywordResult(Context, "try");
      }

      if (S && S->getBreakParent())
        Consumer.addKeywordResult(Context, "break");

      if (S && S->getContinueParent())
        Consumer.addKeywordResult(Context, "continue");

      if (!getCurFunction()->SwitchStack.empty()) {
        Consumer.addKeywordResult(Context, "case");
        Consumer.addKeywordResult(Context, "default");
      }
    } else {
      if (getLangOptions().CPlusPlus) {
        Consumer.addKeywordResult(Context, "namespace");
        Consumer.addKeywordResult(Context, "template");
      }

      if (S && S->isClassScope()) {
        Consumer.addKeywordResult(Context, "explicit");
        Consumer.addKeywordResult(Context, "friend");
        Consumer.addKeywordResult(Context, "mutable");
        Consumer.addKeywordResult(Context, "private");
        Consumer.addKeywordResult(Context, "protected");
        Consumer.addKeywordResult(Context, "public");
        Consumer.addKeywordResult(Context, "virtual");
      }
    }

    if (getLangOptions().CPlusPlus) {
      Consumer.addKeywordResult(Context, "using");

      if (getLangOptions().CPlusPlus0x)
        Consumer.addKeywordResult(Context, "static_assert");
    }
  }

  // If we haven't found anything, we're done.
  if (Consumer.empty()) {
    // If this was an unqualified lookup, note that no correction was found.
    if (IsUnqualifiedLookup)
      (void)UnqualifiedTyposCorrected[Typo];

    return DeclarationName();
  }

  // Make sure that the user typed at least 3 characters for each correction
  // made. Otherwise, we don't even both looking at the results.

  // We also suppress exact matches; those should be handled by a
  // different mechanism (e.g., one that introduces qualification in
  // C++).
  unsigned ED = Consumer.getBestEditDistance();
  if (ED > 0 && Typo->getName().size() / ED < 3) {
    // If this was an unqualified lookup, note that no correction was found.
    if (IsUnqualifiedLookup)
      (void)UnqualifiedTyposCorrected[Typo];

    return DeclarationName();
  }

  // Weed out any names that could not be found by name lookup.
  bool LastLookupWasAccepted = false;
  for (TypoCorrectionConsumer::iterator I = Consumer.begin(),
                                     IEnd = Consumer.end();
       I != IEnd; /* Increment in loop. */) {
    // Keywords are always found.
    if (I->second) {
      ++I;
      continue;
    }

    // Perform name lookup on this name.
    IdentifierInfo *Name = &Context.Idents.get(I->getKey());
    LookupPotentialTypoResult(*this, Res, Name, S, SS, MemberContext,
                              EnteringContext, CTC);

    switch (Res.getResultKind()) {
    case LookupResult::NotFound:
    case LookupResult::NotFoundInCurrentInstantiation:
    case LookupResult::Ambiguous:
      // We didn't find this name in our scope, or didn't like what we found;
      // ignore it.
      Res.suppressDiagnostics();
      {
        TypoCorrectionConsumer::iterator Next = I;
        ++Next;
        Consumer.erase(I);
        I = Next;
      }
      LastLookupWasAccepted = false;
      break;

    case LookupResult::Found:
    case LookupResult::FoundOverloaded:
    case LookupResult::FoundUnresolvedValue:
      ++I;
      LastLookupWasAccepted = true;
      break;
    }

    if (Res.isAmbiguous()) {
      // We don't deal with ambiguities.
      Res.suppressDiagnostics();
      Res.clear();
      return DeclarationName();
    }
  }

  // If only a single name remains, return that result.
  if (Consumer.size() == 1) {
    IdentifierInfo *Name = &Context.Idents.get(Consumer.begin()->getKey());
    if (Consumer.begin()->second) {
      Res.suppressDiagnostics();
      Res.clear();

      // Don't correct to a keyword that's the same as the typo; the keyword
      // wasn't actually in scope.
      if (ED == 0) {
        Res.setLookupName(Typo);
        return DeclarationName();
      }

    } else if (!LastLookupWasAccepted) {
      // Perform name lookup on this name.
      LookupPotentialTypoResult(*this, Res, Name, S, SS, MemberContext,
                                EnteringContext, CTC);
    }

    // Record the correction for unqualified lookup.
    if (IsUnqualifiedLookup)
      UnqualifiedTyposCorrected[Typo]
        = std::make_pair(Name->getName(), Consumer.begin()->second);

    return &Context.Idents.get(Consumer.begin()->getKey());
  }
  else if (Consumer.size() > 1 && CTC == CTC_ObjCMessageReceiver
           && Consumer["super"]) {
    // Prefix 'super' when we're completing in a message-receiver
    // context.
    Res.suppressDiagnostics();
    Res.clear();

    // Don't correct to a keyword that's the same as the typo; the keyword
    // wasn't actually in scope.
    if (ED == 0) {
      Res.setLookupName(Typo);
      return DeclarationName();
    }

    // Record the correction for unqualified lookup.
    if (IsUnqualifiedLookup)
      UnqualifiedTyposCorrected[Typo]
        = std::make_pair("super", Consumer.begin()->second);

    return &Context.Idents.get("super");
  }

  Res.suppressDiagnostics();
  Res.setLookupName(Typo);
  Res.clear();
  // Record the correction for unqualified lookup.
  if (IsUnqualifiedLookup)
    (void)UnqualifiedTyposCorrected[Typo];

  return DeclarationName();
}
