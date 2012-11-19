//===--- DeclCXX.cpp - C++ Declaration AST Node Implementation ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C++ related Decl classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

void AccessSpecDecl::anchor() { }

AccessSpecDecl *AccessSpecDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(AccessSpecDecl));
  return new (Mem) AccessSpecDecl(EmptyShell());
}

CXXRecordDecl::DefinitionData::DefinitionData(CXXRecordDecl *D)
  : UserDeclaredConstructor(false), UserDeclaredCopyConstructor(false),
    UserDeclaredMoveConstructor(false), UserDeclaredCopyAssignment(false),
    UserDeclaredMoveAssignment(false), UserDeclaredDestructor(false),
    Aggregate(true), PlainOldData(true), Empty(true), Polymorphic(false),
    Abstract(false), IsStandardLayout(true), HasNoNonEmptyBases(true),
    HasPrivateFields(false), HasProtectedFields(false), HasPublicFields(false),
    HasMutableFields(false), HasOnlyCMembers(true),
    HasInClassInitializer(false),
    HasTrivialDefaultConstructor(true),
    HasConstexprNonCopyMoveConstructor(false),
    DefaultedDefaultConstructorIsConstexpr(true),
    HasConstexprDefaultConstructor(false), HasTrivialCopyConstructor(true),
    HasTrivialMoveConstructor(true), HasTrivialCopyAssignment(true),
    HasTrivialMoveAssignment(true), HasTrivialDestructor(true),
    HasIrrelevantDestructor(true),
    HasNonLiteralTypeFieldsOrBases(false), ComputedVisibleConversions(false),
    UserProvidedDefaultConstructor(false), DeclaredDefaultConstructor(false),
    DeclaredCopyConstructor(false), DeclaredMoveConstructor(false),
    DeclaredCopyAssignment(false), DeclaredMoveAssignment(false),
    DeclaredDestructor(false), FailedImplicitMoveConstructor(false),
    FailedImplicitMoveAssignment(false), IsLambda(false), NumBases(0),
    NumVBases(0), Bases(), VBases(), Definition(D), FirstFriend(0) {
}

CXXBaseSpecifier *CXXRecordDecl::DefinitionData::getBasesSlowCase() const {
  return Bases.get(Definition->getASTContext().getExternalSource());
}

CXXBaseSpecifier *CXXRecordDecl::DefinitionData::getVBasesSlowCase() const {
  return VBases.get(Definition->getASTContext().getExternalSource());
}

CXXRecordDecl::CXXRecordDecl(Kind K, TagKind TK, DeclContext *DC,
                             SourceLocation StartLoc, SourceLocation IdLoc,
                             IdentifierInfo *Id, CXXRecordDecl *PrevDecl)
  : RecordDecl(K, TK, DC, StartLoc, IdLoc, Id, PrevDecl),
    DefinitionData(PrevDecl ? PrevDecl->DefinitionData : 0),
    TemplateOrInstantiation() { }

CXXRecordDecl *CXXRecordDecl::Create(const ASTContext &C, TagKind TK,
                                     DeclContext *DC, SourceLocation StartLoc,
                                     SourceLocation IdLoc, IdentifierInfo *Id,
                                     CXXRecordDecl* PrevDecl,
                                     bool DelayTypeCreation) {
  CXXRecordDecl* R = new (C) CXXRecordDecl(CXXRecord, TK, DC, StartLoc, IdLoc,
                                           Id, PrevDecl);

  // FIXME: DelayTypeCreation seems like such a hack
  if (!DelayTypeCreation)
    C.getTypeDeclType(R, PrevDecl);
  return R;
}

CXXRecordDecl *CXXRecordDecl::CreateLambda(const ASTContext &C, DeclContext *DC,
                                           TypeSourceInfo *Info, SourceLocation Loc,
                                           bool Dependent) {
  CXXRecordDecl* R = new (C) CXXRecordDecl(CXXRecord, TTK_Class, DC, Loc, Loc,
                                           0, 0);
  R->IsBeingDefined = true;
  R->DefinitionData = new (C) struct LambdaDefinitionData(R, Info, Dependent);
  C.getTypeDeclType(R, /*PrevDecl=*/0);
  return R;
}

CXXRecordDecl *
CXXRecordDecl::CreateDeserialized(const ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(CXXRecordDecl));
  return new (Mem) CXXRecordDecl(CXXRecord, TTK_Struct, 0, SourceLocation(),
                                 SourceLocation(), 0, 0);
}

void
CXXRecordDecl::setBases(CXXBaseSpecifier const * const *Bases,
                        unsigned NumBases) {
  ASTContext &C = getASTContext();

  if (!data().Bases.isOffset() && data().NumBases > 0)
    C.Deallocate(data().getBases());

  if (NumBases) {
    // C++ [dcl.init.aggr]p1:
    //   An aggregate is [...] a class with [...] no base classes [...].
    data().Aggregate = false;

    // C++ [class]p4:
    //   A POD-struct is an aggregate class...
    data().PlainOldData = false;
  }

  // The set of seen virtual base types.
  llvm::SmallPtrSet<CanQualType, 8> SeenVBaseTypes;
  
  // The virtual bases of this class.
  SmallVector<const CXXBaseSpecifier *, 8> VBases;

  data().Bases = new(C) CXXBaseSpecifier [NumBases];
  data().NumBases = NumBases;
  for (unsigned i = 0; i < NumBases; ++i) {
    data().getBases()[i] = *Bases[i];
    // Keep track of inherited vbases for this base class.
    const CXXBaseSpecifier *Base = Bases[i];
    QualType BaseType = Base->getType();
    // Skip dependent types; we can't do any checking on them now.
    if (BaseType->isDependentType())
      continue;
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(BaseType->getAs<RecordType>()->getDecl());

    // A class with a non-empty base class is not empty.
    // FIXME: Standard ref?
    if (!BaseClassDecl->isEmpty()) {
      if (!data().Empty) {
        // C++0x [class]p7:
        //   A standard-layout class is a class that:
        //    [...]
        //    -- either has no non-static data members in the most derived
        //       class and at most one base class with non-static data members,
        //       or has no base classes with non-static data members, and
        // If this is the second non-empty base, then neither of these two
        // clauses can be true.
        data().IsStandardLayout = false;
      }

      data().Empty = false;
      data().HasNoNonEmptyBases = false;
    }
    
    // C++ [class.virtual]p1:
    //   A class that declares or inherits a virtual function is called a 
    //   polymorphic class.
    if (BaseClassDecl->isPolymorphic())
      data().Polymorphic = true;

    // C++0x [class]p7:
    //   A standard-layout class is a class that: [...]
    //    -- has no non-standard-layout base classes
    if (!BaseClassDecl->isStandardLayout())
      data().IsStandardLayout = false;

    // Record if this base is the first non-literal field or base.
    if (!hasNonLiteralTypeFieldsOrBases() && !BaseType->isLiteralType())
      data().HasNonLiteralTypeFieldsOrBases = true;
    
    // Now go through all virtual bases of this base and add them.
    for (CXXRecordDecl::base_class_iterator VBase =
          BaseClassDecl->vbases_begin(),
         E = BaseClassDecl->vbases_end(); VBase != E; ++VBase) {
      // Add this base if it's not already in the list.
      if (SeenVBaseTypes.insert(C.getCanonicalType(VBase->getType())))
        VBases.push_back(VBase);
    }

    if (Base->isVirtual()) {
      // Add this base if it's not already in the list.
      if (SeenVBaseTypes.insert(C.getCanonicalType(BaseType)))
          VBases.push_back(Base);
      
      // C++0x [meta.unary.prop] is_empty:
      //    T is a class type, but not a union type, with ... no virtual base
      //    classes
      data().Empty = false;
      
      // C++ [class.ctor]p5:
      //   A default constructor is trivial [...] if:
      //    -- its class has [...] no virtual bases
      data().HasTrivialDefaultConstructor = false;

      // C++0x [class.copy]p13:
      //   A copy/move constructor for class X is trivial if it is neither
      //   user-provided nor deleted and if
      //    -- class X has no virtual functions and no virtual base classes, and
      data().HasTrivialCopyConstructor = false;
      data().HasTrivialMoveConstructor = false;

      // C++0x [class.copy]p27:
      //   A copy/move assignment operator for class X is trivial if it is
      //   neither user-provided nor deleted and if
      //    -- class X has no virtual functions and no virtual base classes, and
      data().HasTrivialCopyAssignment = false;
      data().HasTrivialMoveAssignment = false;

      // C++0x [class]p7:
      //   A standard-layout class is a class that: [...]
      //    -- has [...] no virtual base classes
      data().IsStandardLayout = false;

      // C++11 [dcl.constexpr]p4:
      //   In the definition of a constexpr constructor [...]
      //    -- the class shall not have any virtual base classes
      data().DefaultedDefaultConstructorIsConstexpr = false;
    } else {
      // C++ [class.ctor]p5:
      //   A default constructor is trivial [...] if:
      //    -- all the direct base classes of its class have trivial default
      //       constructors.
      if (!BaseClassDecl->hasTrivialDefaultConstructor())
        data().HasTrivialDefaultConstructor = false;
      
      // C++0x [class.copy]p13:
      //   A copy/move constructor for class X is trivial if [...]
      //    [...]
      //    -- the constructor selected to copy/move each direct base class
      //       subobject is trivial, and
      // FIXME: C++0x: We need to only consider the selected constructor
      // instead of all of them. For now, we treat a move constructor as being
      // non-trivial if it calls anything other than a trivial move constructor.
      if (!BaseClassDecl->hasTrivialCopyConstructor())
        data().HasTrivialCopyConstructor = false;
      if (!BaseClassDecl->hasTrivialMoveConstructor() ||
          !(BaseClassDecl->hasDeclaredMoveConstructor() ||
            BaseClassDecl->needsImplicitMoveConstructor()))
        data().HasTrivialMoveConstructor = false;

      // C++0x [class.copy]p27:
      //   A copy/move assignment operator for class X is trivial if [...]
      //    [...]
      //    -- the assignment operator selected to copy/move each direct base
      //       class subobject is trivial, and
      // FIXME: C++0x: We need to only consider the selected operator instead
      // of all of them.
      if (!BaseClassDecl->hasTrivialCopyAssignment())
        data().HasTrivialCopyAssignment = false;
      if (!BaseClassDecl->hasTrivialMoveAssignment() ||
          !(BaseClassDecl->hasDeclaredMoveAssignment() ||
            BaseClassDecl->needsImplicitMoveAssignment()))
        data().HasTrivialMoveAssignment = false;

      // C++11 [class.ctor]p6:
      //   If that user-written default constructor would satisfy the
      //   requirements of a constexpr constructor, the implicitly-defined
      //   default constructor is constexpr.
      if (!BaseClassDecl->hasConstexprDefaultConstructor())
        data().DefaultedDefaultConstructorIsConstexpr = false;
    }
    
    // C++ [class.ctor]p3:
    //   A destructor is trivial if all the direct base classes of its class
    //   have trivial destructors.
    if (!BaseClassDecl->hasTrivialDestructor())
      data().HasTrivialDestructor = false;

    if (!BaseClassDecl->hasIrrelevantDestructor())
      data().HasIrrelevantDestructor = false;

    // A class has an Objective-C object member if... or any of its bases
    // has an Objective-C object member.
    if (BaseClassDecl->hasObjectMember())
      setHasObjectMember(true);

    // Keep track of the presence of mutable fields.
    if (BaseClassDecl->hasMutableFields())
      data().HasMutableFields = true;
  }
  
  if (VBases.empty())
    return;

  // Create base specifier for any direct or indirect virtual bases.
  data().VBases = new (C) CXXBaseSpecifier[VBases.size()];
  data().NumVBases = VBases.size();
  for (int I = 0, E = VBases.size(); I != E; ++I)
    data().getVBases()[I] = *VBases[I];
}

/// Callback function for CXXRecordDecl::forallBases that acknowledges
/// that it saw a base class.
static bool SawBase(const CXXRecordDecl *, void *) {
  return true;
}

bool CXXRecordDecl::hasAnyDependentBases() const {
  if (!isDependentContext())
    return false;

  return !forallBases(SawBase, 0);
}

bool CXXRecordDecl::hasConstCopyConstructor() const {
  return getCopyConstructor(Qualifiers::Const) != 0;
}

bool CXXRecordDecl::isTriviallyCopyable() const {
  // C++0x [class]p5:
  //   A trivially copyable class is a class that:
  //   -- has no non-trivial copy constructors,
  if (!hasTrivialCopyConstructor()) return false;
  //   -- has no non-trivial move constructors,
  if (!hasTrivialMoveConstructor()) return false;
  //   -- has no non-trivial copy assignment operators,
  if (!hasTrivialCopyAssignment()) return false;
  //   -- has no non-trivial move assignment operators, and
  if (!hasTrivialMoveAssignment()) return false;
  //   -- has a trivial destructor.
  if (!hasTrivialDestructor()) return false;

  return true;
}

/// \brief Perform a simplistic form of overload resolution that only considers
/// cv-qualifiers on a single parameter, and return the best overload candidate
/// (if there is one).
static CXXMethodDecl *
GetBestOverloadCandidateSimple(
  const SmallVectorImpl<std::pair<CXXMethodDecl *, Qualifiers> > &Cands) {
  if (Cands.empty())
    return 0;
  if (Cands.size() == 1)
    return Cands[0].first;
  
  unsigned Best = 0, N = Cands.size();
  for (unsigned I = 1; I != N; ++I)
    if (Cands[Best].second.compatiblyIncludes(Cands[I].second))
      Best = I;
  
  for (unsigned I = 0; I != N; ++I)
    if (I != Best && Cands[Best].second.compatiblyIncludes(Cands[I].second))
      return 0;
  
  return Cands[Best].first;
}

CXXConstructorDecl *CXXRecordDecl::getCopyConstructor(unsigned TypeQuals) const{
  ASTContext &Context = getASTContext();
  QualType ClassType
    = Context.getTypeDeclType(const_cast<CXXRecordDecl*>(this));
  DeclarationName ConstructorName
    = Context.DeclarationNames.getCXXConstructorName(
                                          Context.getCanonicalType(ClassType));
  unsigned FoundTQs;
  SmallVector<std::pair<CXXMethodDecl *, Qualifiers>, 4> Found;
  DeclContext::lookup_const_iterator Con, ConEnd;
  for (llvm::tie(Con, ConEnd) = this->lookup(ConstructorName);
       Con != ConEnd; ++Con) {
    // C++ [class.copy]p2:
    //   A non-template constructor for class X is a copy constructor if [...]
    if (isa<FunctionTemplateDecl>(*Con))
      continue;

    CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(*Con);
    if (Constructor->isCopyConstructor(FoundTQs)) {
      if (((TypeQuals & Qualifiers::Const) == (FoundTQs & Qualifiers::Const)) ||
          (!(TypeQuals & Qualifiers::Const) && (FoundTQs & Qualifiers::Const)))
        Found.push_back(std::make_pair(
                                 const_cast<CXXConstructorDecl *>(Constructor), 
                                       Qualifiers::fromCVRMask(FoundTQs)));
    }
  }
  
  return cast_or_null<CXXConstructorDecl>(
                                        GetBestOverloadCandidateSimple(Found));
}

CXXConstructorDecl *CXXRecordDecl::getMoveConstructor() const {
  for (ctor_iterator I = ctor_begin(), E = ctor_end(); I != E; ++I)
    if (I->isMoveConstructor())
      return *I;

  return 0;
}

CXXMethodDecl *CXXRecordDecl::getCopyAssignmentOperator(bool ArgIsConst) const {
  ASTContext &Context = getASTContext();
  QualType Class = Context.getTypeDeclType(const_cast<CXXRecordDecl *>(this));
  DeclarationName Name = Context.DeclarationNames.getCXXOperatorName(OO_Equal);
  
  SmallVector<std::pair<CXXMethodDecl *, Qualifiers>, 4> Found;
  DeclContext::lookup_const_iterator Op, OpEnd;
  for (llvm::tie(Op, OpEnd) = this->lookup(Name); Op != OpEnd; ++Op) {
    // C++ [class.copy]p9:
    //   A user-declared copy assignment operator is a non-static non-template
    //   member function of class X with exactly one parameter of type X, X&,
    //   const X&, volatile X& or const volatile X&.
    const CXXMethodDecl* Method = dyn_cast<CXXMethodDecl>(*Op);
    if (!Method || Method->isStatic() || Method->getPrimaryTemplate())
      continue;
    
    const FunctionProtoType *FnType 
      = Method->getType()->getAs<FunctionProtoType>();
    assert(FnType && "Overloaded operator has no prototype.");
    // Don't assert on this; an invalid decl might have been left in the AST.
    if (FnType->getNumArgs() != 1 || FnType->isVariadic())
      continue;
    
    QualType ArgType = FnType->getArgType(0);
    Qualifiers Quals;
    if (const LValueReferenceType *Ref = ArgType->getAs<LValueReferenceType>()) {
      ArgType = Ref->getPointeeType();
      // If we have a const argument and we have a reference to a non-const,
      // this function does not match.
      if (ArgIsConst && !ArgType.isConstQualified())
        continue;
      
      Quals = ArgType.getQualifiers();
    } else {
      // By-value copy-assignment operators are treated like const X&
      // copy-assignment operators.
      Quals = Qualifiers::fromCVRMask(Qualifiers::Const);
    }
    
    if (!Context.hasSameUnqualifiedType(ArgType, Class))
      continue;

    // Save this copy-assignment operator. It might be "the one".
    Found.push_back(std::make_pair(const_cast<CXXMethodDecl *>(Method), Quals));
  }
  
  // Use a simplistic form of overload resolution to find the candidate.
  return GetBestOverloadCandidateSimple(Found);
}

CXXMethodDecl *CXXRecordDecl::getMoveAssignmentOperator() const {
  for (method_iterator I = method_begin(), E = method_end(); I != E; ++I)
    if (I->isMoveAssignmentOperator())
      return *I;

  return 0;
}

void CXXRecordDecl::markedVirtualFunctionPure() {
  // C++ [class.abstract]p2: 
  //   A class is abstract if it has at least one pure virtual function.
  data().Abstract = true;
}

void CXXRecordDecl::markedConstructorConstexpr(CXXConstructorDecl *CD) {
  if (!CD->isCopyOrMoveConstructor())
    data().HasConstexprNonCopyMoveConstructor = true;

  if (CD->isDefaultConstructor())
    data().HasConstexprDefaultConstructor = true;
}

void CXXRecordDecl::addedMember(Decl *D) {
  if (!D->isImplicit() &&
      !isa<FieldDecl>(D) &&
      !isa<IndirectFieldDecl>(D) &&
      (!isa<TagDecl>(D) || cast<TagDecl>(D)->getTagKind() == TTK_Class ||
        cast<TagDecl>(D)->getTagKind() == TTK_Interface))
    data().HasOnlyCMembers = false;

  // Ignore friends and invalid declarations.
  if (D->getFriendObjectKind() || D->isInvalidDecl())
    return;
  
  FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(D);
  if (FunTmpl)
    D = FunTmpl->getTemplatedDecl();
  
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    if (Method->isVirtual()) {
      // C++ [dcl.init.aggr]p1:
      //   An aggregate is an array or a class with [...] no virtual functions.
      data().Aggregate = false;
      
      // C++ [class]p4:
      //   A POD-struct is an aggregate class...
      data().PlainOldData = false;
      
      // Virtual functions make the class non-empty.
      // FIXME: Standard ref?
      data().Empty = false;

      // C++ [class.virtual]p1:
      //   A class that declares or inherits a virtual function is called a 
      //   polymorphic class.
      data().Polymorphic = true;
      
      // C++0x [class.ctor]p5
      //   A default constructor is trivial [...] if:
      //    -- its class has no virtual functions [...]
      data().HasTrivialDefaultConstructor = false;

      // C++0x [class.copy]p13:
      //   A copy/move constructor for class X is trivial if [...]
      //    -- class X has no virtual functions [...]
      data().HasTrivialCopyConstructor = false;
      data().HasTrivialMoveConstructor = false;

      // C++0x [class.copy]p27:
      //   A copy/move assignment operator for class X is trivial if [...]
      //    -- class X has no virtual functions [...]
      data().HasTrivialCopyAssignment = false;
      data().HasTrivialMoveAssignment = false;
            
      // C++0x [class]p7:
      //   A standard-layout class is a class that: [...]
      //    -- has no virtual functions
      data().IsStandardLayout = false;
    }
  }
  
  if (D->isImplicit()) {
    // Notify that an implicit member was added after the definition
    // was completed.
    if (!isBeingDefined())
      if (ASTMutationListener *L = getASTMutationListener())
        L->AddedCXXImplicitMember(data().Definition, D);

    // If this is a special member function, note that it was added and then
    // return early.
    if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
      if (Constructor->isDefaultConstructor()) {
        data().DeclaredDefaultConstructor = true;
        if (Constructor->isConstexpr()) {
          data().HasConstexprDefaultConstructor = true;
          data().HasConstexprNonCopyMoveConstructor = true;
        }
      } else if (Constructor->isCopyConstructor()) {
        data().DeclaredCopyConstructor = true;
      } else if (Constructor->isMoveConstructor()) {
        data().DeclaredMoveConstructor = true;
      } else
        goto NotASpecialMember;
      return;
    } else if (isa<CXXDestructorDecl>(D)) {
      data().DeclaredDestructor = true;
      return;
    } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
      if (Method->isCopyAssignmentOperator())
        data().DeclaredCopyAssignment = true;
      else if (Method->isMoveAssignmentOperator())
        data().DeclaredMoveAssignment = true;
      else
        goto NotASpecialMember;
      return;
    }

NotASpecialMember:;
    // Any other implicit declarations are handled like normal declarations.
  }
  
  // Handle (user-declared) constructors.
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
    // Note that we have a user-declared constructor.
    data().UserDeclaredConstructor = true;

    // Technically, "user-provided" is only defined for special member
    // functions, but the intent of the standard is clearly that it should apply
    // to all functions.
    bool UserProvided = Constructor->isUserProvided();

    if (Constructor->isDefaultConstructor()) {
      data().DeclaredDefaultConstructor = true;
      if (UserProvided) {
        // C++0x [class.ctor]p5:
        //   A default constructor is trivial if it is not user-provided [...]
        data().HasTrivialDefaultConstructor = false;
        data().UserProvidedDefaultConstructor = true;
      }
      if (Constructor->isConstexpr()) {
        data().HasConstexprDefaultConstructor = true;
        data().HasConstexprNonCopyMoveConstructor = true;
      }
    }

    // Note when we have a user-declared copy or move constructor, which will
    // suppress the implicit declaration of those constructors.
    if (!FunTmpl) {
      if (Constructor->isCopyConstructor()) {
        data().UserDeclaredCopyConstructor = true;
        data().DeclaredCopyConstructor = true;

        // C++0x [class.copy]p13:
        //   A copy/move constructor for class X is trivial if it is not
        //   user-provided [...]
        if (UserProvided)
          data().HasTrivialCopyConstructor = false;
      } else if (Constructor->isMoveConstructor()) {
        data().UserDeclaredMoveConstructor = true;
        data().DeclaredMoveConstructor = true;

        // C++0x [class.copy]p13:
        //   A copy/move constructor for class X is trivial if it is not
        //   user-provided [...]
        if (UserProvided)
          data().HasTrivialMoveConstructor = false;
      }
    }
    if (Constructor->isConstexpr() && !Constructor->isCopyOrMoveConstructor()) {
      // Record if we see any constexpr constructors which are neither copy
      // nor move constructors.
      data().HasConstexprNonCopyMoveConstructor = true;
    }

    // C++ [dcl.init.aggr]p1:
    //   An aggregate is an array or a class with no user-declared
    //   constructors [...].
    // C++0x [dcl.init.aggr]p1:
    //   An aggregate is an array or a class with no user-provided
    //   constructors [...].
    if (!getASTContext().getLangOpts().CPlusPlus0x || UserProvided)
      data().Aggregate = false;

    // C++ [class]p4:
    //   A POD-struct is an aggregate class [...]
    // Since the POD bit is meant to be C++03 POD-ness, clear it even if the
    // type is technically an aggregate in C++0x since it wouldn't be in 03.
    data().PlainOldData = false;

    return;
  }

  // Handle (user-declared) destructors.
  if (CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(D)) {
    data().DeclaredDestructor = true;
    data().UserDeclaredDestructor = true;
    data().HasIrrelevantDestructor = false;

    // C++ [class]p4: 
    //   A POD-struct is an aggregate class that has [...] no user-defined 
    //   destructor.
    // This bit is the C++03 POD bit, not the 0x one.
    data().PlainOldData = false;
    
    // C++11 [class.dtor]p5: 
    //   A destructor is trivial if it is not user-provided and if
    //    -- the destructor is not virtual.
    if (DD->isUserProvided() || DD->isVirtual())
      data().HasTrivialDestructor = false;

    return;
  }
  
  // Handle (user-declared) member functions.
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    if (Method->isCopyAssignmentOperator()) {
      // C++ [class]p4:
      //   A POD-struct is an aggregate class that [...] has no user-defined
      //   copy assignment operator [...].
      // This is the C++03 bit only.
      data().PlainOldData = false;

      // This is a copy assignment operator.

      // Suppress the implicit declaration of a copy constructor.
      data().UserDeclaredCopyAssignment = true;
      data().DeclaredCopyAssignment = true;

      // C++0x [class.copy]p27:
      //   A copy/move assignment operator for class X is trivial if it is
      //   neither user-provided nor deleted [...]
      if (Method->isUserProvided())
        data().HasTrivialCopyAssignment = false;

      return;
    }
    
    if (Method->isMoveAssignmentOperator()) {
      // This is an extension in C++03 mode, but we'll keep consistency by
      // taking a move assignment operator to induce non-POD-ness
      data().PlainOldData = false;

      // This is a move assignment operator.
      data().UserDeclaredMoveAssignment = true;
      data().DeclaredMoveAssignment = true;

      // C++0x [class.copy]p27:
      //   A copy/move assignment operator for class X is trivial if it is
      //   neither user-provided nor deleted [...]
      if (Method->isUserProvided())
        data().HasTrivialMoveAssignment = false;
    }

    // Keep the list of conversion functions up-to-date.
    if (CXXConversionDecl *Conversion = dyn_cast<CXXConversionDecl>(D)) {
      // We don't record specializations.
      if (Conversion->getPrimaryTemplate())
        return;
      
      // FIXME: We intentionally don't use the decl's access here because it
      // hasn't been set yet.  That's really just a misdesign in Sema.

      if (FunTmpl) {
        if (FunTmpl->getPreviousDecl())
          data().Conversions.replace(FunTmpl->getPreviousDecl(),
                                     FunTmpl);
        else
          data().Conversions.addDecl(FunTmpl);
      } else {
        if (Conversion->getPreviousDecl())
          data().Conversions.replace(Conversion->getPreviousDecl(),
                                     Conversion);
        else
          data().Conversions.addDecl(Conversion);        
      }
    }
    
    return;
  }
  
  // Handle non-static data members.
  if (FieldDecl *Field = dyn_cast<FieldDecl>(D)) {
    // C++ [class.bit]p2:
    //   A declaration for a bit-field that omits the identifier declares an 
    //   unnamed bit-field. Unnamed bit-fields are not members and cannot be 
    //   initialized.
    if (Field->isUnnamedBitfield())
      return;
    
    // C++ [dcl.init.aggr]p1:
    //   An aggregate is an array or a class (clause 9) with [...] no
    //   private or protected non-static data members (clause 11).
    //
    // A POD must be an aggregate.    
    if (D->getAccess() == AS_private || D->getAccess() == AS_protected) {
      data().Aggregate = false;
      data().PlainOldData = false;
    }

    // C++0x [class]p7:
    //   A standard-layout class is a class that:
    //    [...]
    //    -- has the same access control for all non-static data members,
    switch (D->getAccess()) {
    case AS_private:    data().HasPrivateFields = true;   break;
    case AS_protected:  data().HasProtectedFields = true; break;
    case AS_public:     data().HasPublicFields = true;    break;
    case AS_none:       llvm_unreachable("Invalid access specifier");
    };
    if ((data().HasPrivateFields + data().HasProtectedFields +
         data().HasPublicFields) > 1)
      data().IsStandardLayout = false;

    // Keep track of the presence of mutable fields.
    if (Field->isMutable())
      data().HasMutableFields = true;
    
    // C++0x [class]p9:
    //   A POD struct is a class that is both a trivial class and a 
    //   standard-layout class, and has no non-static data members of type 
    //   non-POD struct, non-POD union (or array of such types).
    //
    // Automatic Reference Counting: the presence of a member of Objective-C pointer type
    // that does not explicitly have no lifetime makes the class a non-POD.
    // However, we delay setting PlainOldData to false in this case so that
    // Sema has a chance to diagnostic causes where the same class will be
    // non-POD with Automatic Reference Counting but a POD without ARC.
    // In this case, the class will become a non-POD class when we complete
    // the definition.
    ASTContext &Context = getASTContext();
    QualType T = Context.getBaseElementType(Field->getType());
    if (T->isObjCRetainableType() || T.isObjCGCStrong()) {
      if (!Context.getLangOpts().ObjCAutoRefCount ||
          T.getObjCLifetime() != Qualifiers::OCL_ExplicitNone)
        setHasObjectMember(true);
    } else if (!T.isPODType(Context))
      data().PlainOldData = false;
    
    if (T->isReferenceType()) {
      data().HasTrivialDefaultConstructor = false;

      // C++0x [class]p7:
      //   A standard-layout class is a class that:
      //    -- has no non-static data members of type [...] reference,
      data().IsStandardLayout = false;
    }

    // Record if this field is the first non-literal or volatile field or base.
    if (!T->isLiteralType() || T.isVolatileQualified())
      data().HasNonLiteralTypeFieldsOrBases = true;

    if (Field->hasInClassInitializer()) {
      data().HasInClassInitializer = true;

      // C++11 [class]p5:
      //   A default constructor is trivial if [...] no non-static data member
      //   of its class has a brace-or-equal-initializer.
      data().HasTrivialDefaultConstructor = false;

      // C++11 [dcl.init.aggr]p1:
      //   An aggregate is a [...] class with [...] no
      //   brace-or-equal-initializers for non-static data members.
      data().Aggregate = false;

      // C++11 [class]p10:
      //   A POD struct is [...] a trivial class.
      data().PlainOldData = false;
    }

    if (const RecordType *RecordTy = T->getAs<RecordType>()) {
      CXXRecordDecl* FieldRec = cast<CXXRecordDecl>(RecordTy->getDecl());
      if (FieldRec->getDefinition()) {
        // C++0x [class.ctor]p5:
        //   A default constructor is trivial [...] if:
        //    -- for all the non-static data members of its class that are of
        //       class type (or array thereof), each such class has a trivial
        //       default constructor.
        if (!FieldRec->hasTrivialDefaultConstructor())
          data().HasTrivialDefaultConstructor = false;

        // C++0x [class.copy]p13:
        //   A copy/move constructor for class X is trivial if [...]
        //    [...]
        //    -- for each non-static data member of X that is of class type (or
        //       an array thereof), the constructor selected to copy/move that
        //       member is trivial;
        // FIXME: C++0x: We don't correctly model 'selected' constructors.
        if (!FieldRec->hasTrivialCopyConstructor())
          data().HasTrivialCopyConstructor = false;
        if (!FieldRec->hasTrivialMoveConstructor() ||
            !(FieldRec->hasDeclaredMoveConstructor() ||
              FieldRec->needsImplicitMoveConstructor()))
          data().HasTrivialMoveConstructor = false;

        // C++0x [class.copy]p27:
        //   A copy/move assignment operator for class X is trivial if [...]
        //    [...]
        //    -- for each non-static data member of X that is of class type (or
        //       an array thereof), the assignment operator selected to
        //       copy/move that member is trivial;
        // FIXME: C++0x: We don't correctly model 'selected' operators.
        if (!FieldRec->hasTrivialCopyAssignment())
          data().HasTrivialCopyAssignment = false;
        if (!FieldRec->hasTrivialMoveAssignment() ||
            !(FieldRec->hasDeclaredMoveAssignment() ||
              FieldRec->needsImplicitMoveAssignment()))
          data().HasTrivialMoveAssignment = false;

        if (!FieldRec->hasTrivialDestructor())
          data().HasTrivialDestructor = false;
        if (!FieldRec->hasIrrelevantDestructor())
          data().HasIrrelevantDestructor = false;
        if (FieldRec->hasObjectMember())
          setHasObjectMember(true);

        // C++0x [class]p7:
        //   A standard-layout class is a class that:
        //    -- has no non-static data members of type non-standard-layout
        //       class (or array of such types) [...]
        if (!FieldRec->isStandardLayout())
          data().IsStandardLayout = false;

        // C++0x [class]p7:
        //   A standard-layout class is a class that:
        //    [...]
        //    -- has no base classes of the same type as the first non-static
        //       data member.
        // We don't want to expend bits in the state of the record decl
        // tracking whether this is the first non-static data member so we
        // cheat a bit and use some of the existing state: the empty bit.
        // Virtual bases and virtual methods make a class non-empty, but they
        // also make it non-standard-layout so we needn't check here.
        // A non-empty base class may leave the class standard-layout, but not
        // if we have arrived here, and have at least on non-static data
        // member. If IsStandardLayout remains true, then the first non-static
        // data member must come through here with Empty still true, and Empty
        // will subsequently be set to false below.
        if (data().IsStandardLayout && data().Empty) {
          for (CXXRecordDecl::base_class_const_iterator BI = bases_begin(),
                                                        BE = bases_end();
               BI != BE; ++BI) {
            if (Context.hasSameUnqualifiedType(BI->getType(), T)) {
              data().IsStandardLayout = false;
              break;
            }
          }
        }
        
        // Keep track of the presence of mutable fields.
        if (FieldRec->hasMutableFields())
          data().HasMutableFields = true;

        // C++11 [class.copy]p13:
        //   If the implicitly-defined constructor would satisfy the
        //   requirements of a constexpr constructor, the implicitly-defined
        //   constructor is constexpr.
        // C++11 [dcl.constexpr]p4:
        //    -- every constructor involved in initializing non-static data
        //       members [...] shall be a constexpr constructor
        if (!Field->hasInClassInitializer() &&
            !FieldRec->hasConstexprDefaultConstructor() && !isUnion())
          // The standard requires any in-class initializer to be a constant
          // expression. We consider this to be a defect.
          data().DefaultedDefaultConstructorIsConstexpr = false;
      }
    } else {
      // Base element type of field is a non-class type.
      if (!T->isLiteralType() ||
          (!Field->hasInClassInitializer() && !isUnion()))
        data().DefaultedDefaultConstructorIsConstexpr = false;
    }

    // C++0x [class]p7:
    //   A standard-layout class is a class that:
    //    [...]
    //    -- either has no non-static data members in the most derived
    //       class and at most one base class with non-static data members,
    //       or has no base classes with non-static data members, and
    // At this point we know that we have a non-static data member, so the last
    // clause holds.
    if (!data().HasNoNonEmptyBases)
      data().IsStandardLayout = false;

    // If this is not a zero-length bit-field, then the class is not empty.
    if (data().Empty) {
      if (!Field->isBitField() ||
          (!Field->getBitWidth()->isTypeDependent() &&
           !Field->getBitWidth()->isValueDependent() &&
           Field->getBitWidthValue(Context) != 0))
        data().Empty = false;
    }
  }
  
  // Handle using declarations of conversion functions.
  if (UsingShadowDecl *Shadow = dyn_cast<UsingShadowDecl>(D))
    if (Shadow->getDeclName().getNameKind()
          == DeclarationName::CXXConversionFunctionName)
      data().Conversions.addDecl(Shadow, Shadow->getAccess());
}

bool CXXRecordDecl::isCLike() const {
  if (getTagKind() == TTK_Class || getTagKind() == TTK_Interface ||
      !TemplateOrInstantiation.isNull())
    return false;
  if (!hasDefinition())
    return true;

  return isPOD() && data().HasOnlyCMembers;
}

void CXXRecordDecl::getCaptureFields(
       llvm::DenseMap<const VarDecl *, FieldDecl *> &Captures,
       FieldDecl *&ThisCapture) const {
  Captures.clear();
  ThisCapture = 0;

  LambdaDefinitionData &Lambda = getLambdaData();
  RecordDecl::field_iterator Field = field_begin();
  for (LambdaExpr::Capture *C = Lambda.Captures, *CEnd = C + Lambda.NumCaptures;
       C != CEnd; ++C, ++Field) {
    if (C->capturesThis()) {
      ThisCapture = *Field;
      continue;
    }

    Captures[C->getCapturedVar()] = *Field;
  }
}


static CanQualType GetConversionType(ASTContext &Context, NamedDecl *Conv) {
  QualType T;
  if (isa<UsingShadowDecl>(Conv))
    Conv = cast<UsingShadowDecl>(Conv)->getTargetDecl();
  if (FunctionTemplateDecl *ConvTemp = dyn_cast<FunctionTemplateDecl>(Conv))
    T = ConvTemp->getTemplatedDecl()->getResultType();
  else 
    T = cast<CXXConversionDecl>(Conv)->getConversionType();
  return Context.getCanonicalType(T);
}

/// Collect the visible conversions of a base class.
///
/// \param Record a base class of the class we're considering
/// \param InVirtual whether this base class is a virtual base (or a base
///   of a virtual base)
/// \param Access the access along the inheritance path to this base
/// \param ParentHiddenTypes the conversions provided by the inheritors
///   of this base
/// \param Output the set to which to add conversions from non-virtual bases
/// \param VOutput the set to which to add conversions from virtual bases
/// \param HiddenVBaseCs the set of conversions which were hidden in a
///   virtual base along some inheritance path
static void CollectVisibleConversions(ASTContext &Context,
                                      CXXRecordDecl *Record,
                                      bool InVirtual,
                                      AccessSpecifier Access,
                  const llvm::SmallPtrSet<CanQualType, 8> &ParentHiddenTypes,
                                      UnresolvedSetImpl &Output,
                                      UnresolvedSetImpl &VOutput,
                           llvm::SmallPtrSet<NamedDecl*, 8> &HiddenVBaseCs) {
  // The set of types which have conversions in this class or its
  // subclasses.  As an optimization, we don't copy the derived set
  // unless it might change.
  const llvm::SmallPtrSet<CanQualType, 8> *HiddenTypes = &ParentHiddenTypes;
  llvm::SmallPtrSet<CanQualType, 8> HiddenTypesBuffer;

  // Collect the direct conversions and figure out which conversions
  // will be hidden in the subclasses.
  UnresolvedSetImpl &Cs = *Record->getConversionFunctions();
  if (!Cs.empty()) {
    HiddenTypesBuffer = ParentHiddenTypes;
    HiddenTypes = &HiddenTypesBuffer;

    for (UnresolvedSetIterator I = Cs.begin(), E = Cs.end(); I != E; ++I) {
      CanQualType ConvType(GetConversionType(Context, I.getDecl()));
      bool Hidden = ParentHiddenTypes.count(ConvType);
      if (!Hidden)
        HiddenTypesBuffer.insert(ConvType);

      // If this conversion is hidden and we're in a virtual base,
      // remember that it's hidden along some inheritance path.
      if (Hidden && InVirtual)
        HiddenVBaseCs.insert(cast<NamedDecl>(I.getDecl()->getCanonicalDecl()));

      // If this conversion isn't hidden, add it to the appropriate output.
      else if (!Hidden) {
        AccessSpecifier IAccess
          = CXXRecordDecl::MergeAccess(Access, I.getAccess());

        if (InVirtual)
          VOutput.addDecl(I.getDecl(), IAccess);
        else
          Output.addDecl(I.getDecl(), IAccess);
      }
    }
  }

  // Collect information recursively from any base classes.
  for (CXXRecordDecl::base_class_iterator
         I = Record->bases_begin(), E = Record->bases_end(); I != E; ++I) {
    const RecordType *RT = I->getType()->getAs<RecordType>();
    if (!RT) continue;

    AccessSpecifier BaseAccess
      = CXXRecordDecl::MergeAccess(Access, I->getAccessSpecifier());
    bool BaseInVirtual = InVirtual || I->isVirtual();

    CXXRecordDecl *Base = cast<CXXRecordDecl>(RT->getDecl());
    CollectVisibleConversions(Context, Base, BaseInVirtual, BaseAccess,
                              *HiddenTypes, Output, VOutput, HiddenVBaseCs);
  }
}

/// Collect the visible conversions of a class.
///
/// This would be extremely straightforward if it weren't for virtual
/// bases.  It might be worth special-casing that, really.
static void CollectVisibleConversions(ASTContext &Context,
                                      CXXRecordDecl *Record,
                                      UnresolvedSetImpl &Output) {
  // The collection of all conversions in virtual bases that we've
  // found.  These will be added to the output as long as they don't
  // appear in the hidden-conversions set.
  UnresolvedSet<8> VBaseCs;
  
  // The set of conversions in virtual bases that we've determined to
  // be hidden.
  llvm::SmallPtrSet<NamedDecl*, 8> HiddenVBaseCs;

  // The set of types hidden by classes derived from this one.
  llvm::SmallPtrSet<CanQualType, 8> HiddenTypes;

  // Go ahead and collect the direct conversions and add them to the
  // hidden-types set.
  UnresolvedSetImpl &Cs = *Record->getConversionFunctions();
  Output.append(Cs.begin(), Cs.end());
  for (UnresolvedSetIterator I = Cs.begin(), E = Cs.end(); I != E; ++I)
    HiddenTypes.insert(GetConversionType(Context, I.getDecl()));

  // Recursively collect conversions from base classes.
  for (CXXRecordDecl::base_class_iterator
         I = Record->bases_begin(), E = Record->bases_end(); I != E; ++I) {
    const RecordType *RT = I->getType()->getAs<RecordType>();
    if (!RT) continue;

    CollectVisibleConversions(Context, cast<CXXRecordDecl>(RT->getDecl()),
                              I->isVirtual(), I->getAccessSpecifier(),
                              HiddenTypes, Output, VBaseCs, HiddenVBaseCs);
  }

  // Add any unhidden conversions provided by virtual bases.
  for (UnresolvedSetIterator I = VBaseCs.begin(), E = VBaseCs.end();
         I != E; ++I) {
    if (!HiddenVBaseCs.count(cast<NamedDecl>(I.getDecl()->getCanonicalDecl())))
      Output.addDecl(I.getDecl(), I.getAccess());
  }
}

/// getVisibleConversionFunctions - get all conversion functions visible
/// in current class; including conversion function templates.
const UnresolvedSetImpl *CXXRecordDecl::getVisibleConversionFunctions() {
  // If root class, all conversions are visible.
  if (bases_begin() == bases_end())
    return &data().Conversions;
  // If visible conversion list is already evaluated, return it.
  if (data().ComputedVisibleConversions)
    return &data().VisibleConversions;
  CollectVisibleConversions(getASTContext(), this, data().VisibleConversions);
  data().ComputedVisibleConversions = true;
  return &data().VisibleConversions;
}

void CXXRecordDecl::removeConversion(const NamedDecl *ConvDecl) {
  // This operation is O(N) but extremely rare.  Sema only uses it to
  // remove UsingShadowDecls in a class that were followed by a direct
  // declaration, e.g.:
  //   class A : B {
  //     using B::operator int;
  //     operator int();
  //   };
  // This is uncommon by itself and even more uncommon in conjunction
  // with sufficiently large numbers of directly-declared conversions
  // that asymptotic behavior matters.

  UnresolvedSetImpl &Convs = *getConversionFunctions();
  for (unsigned I = 0, E = Convs.size(); I != E; ++I) {
    if (Convs[I].getDecl() == ConvDecl) {
      Convs.erase(I);
      assert(std::find(Convs.begin(), Convs.end(), ConvDecl) == Convs.end()
             && "conversion was found multiple times in unresolved set");
      return;
    }
  }

  llvm_unreachable("conversion not found in set!");
}

CXXRecordDecl *CXXRecordDecl::getInstantiatedFromMemberClass() const {
  if (MemberSpecializationInfo *MSInfo = getMemberSpecializationInfo())
    return cast<CXXRecordDecl>(MSInfo->getInstantiatedFrom());
  
  return 0;
}

MemberSpecializationInfo *CXXRecordDecl::getMemberSpecializationInfo() const {
  return TemplateOrInstantiation.dyn_cast<MemberSpecializationInfo *>();
}

void 
CXXRecordDecl::setInstantiationOfMemberClass(CXXRecordDecl *RD,
                                             TemplateSpecializationKind TSK) {
  assert(TemplateOrInstantiation.isNull() && 
         "Previous template or instantiation?");
  assert(!isa<ClassTemplateSpecializationDecl>(this));
  TemplateOrInstantiation 
    = new (getASTContext()) MemberSpecializationInfo(RD, TSK);
}

TemplateSpecializationKind CXXRecordDecl::getTemplateSpecializationKind() const{
  if (const ClassTemplateSpecializationDecl *Spec
        = dyn_cast<ClassTemplateSpecializationDecl>(this))
    return Spec->getSpecializationKind();
  
  if (MemberSpecializationInfo *MSInfo = getMemberSpecializationInfo())
    return MSInfo->getTemplateSpecializationKind();
  
  return TSK_Undeclared;
}

void 
CXXRecordDecl::setTemplateSpecializationKind(TemplateSpecializationKind TSK) {
  if (ClassTemplateSpecializationDecl *Spec
      = dyn_cast<ClassTemplateSpecializationDecl>(this)) {
    Spec->setSpecializationKind(TSK);
    return;
  }
  
  if (MemberSpecializationInfo *MSInfo = getMemberSpecializationInfo()) {
    MSInfo->setTemplateSpecializationKind(TSK);
    return;
  }
  
  llvm_unreachable("Not a class template or member class specialization");
}

CXXDestructorDecl *CXXRecordDecl::getDestructor() const {
  ASTContext &Context = getASTContext();
  QualType ClassType = Context.getTypeDeclType(this);

  DeclarationName Name
    = Context.DeclarationNames.getCXXDestructorName(
                                          Context.getCanonicalType(ClassType));

  DeclContext::lookup_const_iterator I, E;
  llvm::tie(I, E) = lookup(Name);
  if (I == E)
    return 0;

  CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(*I);
  return Dtor;
}

void CXXRecordDecl::completeDefinition() {
  completeDefinition(0);
}

void CXXRecordDecl::completeDefinition(CXXFinalOverriderMap *FinalOverriders) {
  RecordDecl::completeDefinition();
  
  if (hasObjectMember() && getASTContext().getLangOpts().ObjCAutoRefCount) {
    // Objective-C Automatic Reference Counting:
    //   If a class has a non-static data member of Objective-C pointer
    //   type (or array thereof), it is a non-POD type and its
    //   default constructor (if any), copy constructor, move constructor,
    //   copy assignment operator, move assignment operator, and destructor are
    //   non-trivial.
    struct DefinitionData &Data = data();
    Data.PlainOldData = false;
    Data.HasTrivialDefaultConstructor = false;
    Data.HasTrivialCopyConstructor = false;
    Data.HasTrivialMoveConstructor = false;
    Data.HasTrivialCopyAssignment = false;
    Data.HasTrivialMoveAssignment = false;
    Data.HasTrivialDestructor = false;
    Data.HasIrrelevantDestructor = false;
  }
  
  // If the class may be abstract (but hasn't been marked as such), check for
  // any pure final overriders.
  if (mayBeAbstract()) {
    CXXFinalOverriderMap MyFinalOverriders;
    if (!FinalOverriders) {
      getFinalOverriders(MyFinalOverriders);
      FinalOverriders = &MyFinalOverriders;
    }
    
    bool Done = false;
    for (CXXFinalOverriderMap::iterator M = FinalOverriders->begin(), 
                                     MEnd = FinalOverriders->end();
         M != MEnd && !Done; ++M) {
      for (OverridingMethods::iterator SO = M->second.begin(), 
                                    SOEnd = M->second.end();
           SO != SOEnd && !Done; ++SO) {
        assert(SO->second.size() > 0 && 
               "All virtual functions have overridding virtual functions");
        
        // C++ [class.abstract]p4:
        //   A class is abstract if it contains or inherits at least one
        //   pure virtual function for which the final overrider is pure
        //   virtual.
        if (SO->second.front().Method->isPure()) {
          data().Abstract = true;
          Done = true;
          break;
        }
      }
    }
  }
  
  // Set access bits correctly on the directly-declared conversions.
  for (UnresolvedSetIterator I = data().Conversions.begin(), 
                             E = data().Conversions.end(); 
       I != E; ++I)
    data().Conversions.setAccess(I, (*I)->getAccess());
}

bool CXXRecordDecl::mayBeAbstract() const {
  if (data().Abstract || isInvalidDecl() || !data().Polymorphic ||
      isDependentContext())
    return false;
  
  for (CXXRecordDecl::base_class_const_iterator B = bases_begin(),
                                             BEnd = bases_end();
       B != BEnd; ++B) {
    CXXRecordDecl *BaseDecl 
      = cast<CXXRecordDecl>(B->getType()->getAs<RecordType>()->getDecl());
    if (BaseDecl->isAbstract())
      return true;
  }
  
  return false;
}

void CXXMethodDecl::anchor() { }

static bool recursivelyOverrides(const CXXMethodDecl *DerivedMD,
                                 const CXXMethodDecl *BaseMD) {
  for (CXXMethodDecl::method_iterator I = DerivedMD->begin_overridden_methods(),
         E = DerivedMD->end_overridden_methods(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    if (MD->getCanonicalDecl() == BaseMD->getCanonicalDecl())
      return true;
    if (recursivelyOverrides(MD, BaseMD))
      return true;
  }
  return false;
}

CXXMethodDecl *
CXXMethodDecl::getCorrespondingMethodInClass(const CXXRecordDecl *RD,
                                             bool MayBeBase) {
  if (this->getParent()->getCanonicalDecl() == RD->getCanonicalDecl())
    return this;

  // Lookup doesn't work for destructors, so handle them separately.
  if (isa<CXXDestructorDecl>(this)) {
    CXXMethodDecl *MD = RD->getDestructor();
    if (MD) {
      if (recursivelyOverrides(MD, this))
        return MD;
      if (MayBeBase && recursivelyOverrides(this, MD))
        return MD;
    }
    return NULL;
  }

  lookup_const_result Candidates = RD->lookup(getDeclName());
  for (NamedDecl * const * I = Candidates.first; I != Candidates.second; ++I) {
    CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(*I);
    if (!MD)
      continue;
    if (recursivelyOverrides(MD, this))
      return MD;
    if (MayBeBase && recursivelyOverrides(this, MD))
      return MD;
  }

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    const RecordType *RT = I->getType()->getAs<RecordType>();
    if (!RT)
      continue;
    const CXXRecordDecl *Base = cast<CXXRecordDecl>(RT->getDecl());
    CXXMethodDecl *T = this->getCorrespondingMethodInClass(Base);
    if (T)
      return T;
  }

  return NULL;
}

CXXMethodDecl *
CXXMethodDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                      SourceLocation StartLoc,
                      const DeclarationNameInfo &NameInfo,
                      QualType T, TypeSourceInfo *TInfo,
                      bool isStatic, StorageClass SCAsWritten, bool isInline,
                      bool isConstexpr, SourceLocation EndLocation) {
  return new (C) CXXMethodDecl(CXXMethod, RD, StartLoc, NameInfo, T, TInfo,
                               isStatic, SCAsWritten, isInline, isConstexpr,
                               EndLocation);
}

CXXMethodDecl *CXXMethodDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(CXXMethodDecl));
  return new (Mem) CXXMethodDecl(CXXMethod, 0, SourceLocation(), 
                                 DeclarationNameInfo(), QualType(),
                                 0, false, SC_None, false, false,
                                 SourceLocation());
}

bool CXXMethodDecl::isUsualDeallocationFunction() const {
  if (getOverloadedOperator() != OO_Delete &&
      getOverloadedOperator() != OO_Array_Delete)
    return false;

  // C++ [basic.stc.dynamic.deallocation]p2:
  //   A template instance is never a usual deallocation function,
  //   regardless of its signature.
  if (getPrimaryTemplate())
    return false;

  // C++ [basic.stc.dynamic.deallocation]p2:
  //   If a class T has a member deallocation function named operator delete 
  //   with exactly one parameter, then that function is a usual (non-placement)
  //   deallocation function. [...]
  if (getNumParams() == 1)
    return true;
  
  // C++ [basic.stc.dynamic.deallocation]p2:
  //   [...] If class T does not declare such an operator delete but does 
  //   declare a member deallocation function named operator delete with 
  //   exactly two parameters, the second of which has type std::size_t (18.1),
  //   then this function is a usual deallocation function.
  ASTContext &Context = getASTContext();
  if (getNumParams() != 2 ||
      !Context.hasSameUnqualifiedType(getParamDecl(1)->getType(),
                                      Context.getSizeType()))
    return false;
                 
  // This function is a usual deallocation function if there are no 
  // single-parameter deallocation functions of the same kind.
  for (DeclContext::lookup_const_result R = getDeclContext()->lookup(getDeclName());
       R.first != R.second; ++R.first) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(*R.first))
      if (FD->getNumParams() == 1)
        return false;
  }
  
  return true;
}

bool CXXMethodDecl::isCopyAssignmentOperator() const {
  // C++0x [class.copy]p17:
  //  A user-declared copy assignment operator X::operator= is a non-static 
  //  non-template member function of class X with exactly one parameter of 
  //  type X, X&, const X&, volatile X& or const volatile X&.
  if (/*operator=*/getOverloadedOperator() != OO_Equal ||
      /*non-static*/ isStatic() || 
      /*non-template*/getPrimaryTemplate() || getDescribedFunctionTemplate())
    return false;
      
  QualType ParamType = getParamDecl(0)->getType();
  if (const LValueReferenceType *Ref = ParamType->getAs<LValueReferenceType>())
    ParamType = Ref->getPointeeType();
  
  ASTContext &Context = getASTContext();
  QualType ClassType
    = Context.getCanonicalType(Context.getTypeDeclType(getParent()));
  return Context.hasSameUnqualifiedType(ClassType, ParamType);
}

bool CXXMethodDecl::isMoveAssignmentOperator() const {
  // C++0x [class.copy]p19:
  //  A user-declared move assignment operator X::operator= is a non-static
  //  non-template member function of class X with exactly one parameter of type
  //  X&&, const X&&, volatile X&&, or const volatile X&&.
  if (getOverloadedOperator() != OO_Equal || isStatic() ||
      getPrimaryTemplate() || getDescribedFunctionTemplate())
    return false;

  QualType ParamType = getParamDecl(0)->getType();
  if (!isa<RValueReferenceType>(ParamType))
    return false;
  ParamType = ParamType->getPointeeType();

  ASTContext &Context = getASTContext();
  QualType ClassType
    = Context.getCanonicalType(Context.getTypeDeclType(getParent()));
  return Context.hasSameUnqualifiedType(ClassType, ParamType);
}

void CXXMethodDecl::addOverriddenMethod(const CXXMethodDecl *MD) {
  assert(MD->isCanonicalDecl() && "Method is not canonical!");
  assert(!MD->getParent()->isDependentContext() &&
         "Can't add an overridden method to a class template!");
  assert(MD->isVirtual() && "Method is not virtual!");

  getASTContext().addOverriddenMethod(this, MD);
}

CXXMethodDecl::method_iterator CXXMethodDecl::begin_overridden_methods() const {
  if (isa<CXXConstructorDecl>(this)) return 0;
  return getASTContext().overridden_methods_begin(this);
}

CXXMethodDecl::method_iterator CXXMethodDecl::end_overridden_methods() const {
  if (isa<CXXConstructorDecl>(this)) return 0;
  return getASTContext().overridden_methods_end(this);
}

unsigned CXXMethodDecl::size_overridden_methods() const {
  if (isa<CXXConstructorDecl>(this)) return 0;
  return getASTContext().overridden_methods_size(this);
}

QualType CXXMethodDecl::getThisType(ASTContext &C) const {
  // C++ 9.3.2p1: The type of this in a member function of a class X is X*.
  // If the member function is declared const, the type of this is const X*,
  // if the member function is declared volatile, the type of this is
  // volatile X*, and if the member function is declared const volatile,
  // the type of this is const volatile X*.

  assert(isInstance() && "No 'this' for static methods!");

  QualType ClassTy = C.getTypeDeclType(getParent());
  ClassTy = C.getQualifiedType(ClassTy,
                               Qualifiers::fromCVRMask(getTypeQualifiers()));
  return C.getPointerType(ClassTy);
}

bool CXXMethodDecl::hasInlineBody() const {
  // If this function is a template instantiation, look at the template from 
  // which it was instantiated.
  const FunctionDecl *CheckFn = getTemplateInstantiationPattern();
  if (!CheckFn)
    CheckFn = this;
  
  const FunctionDecl *fn;
  return CheckFn->hasBody(fn) && !fn->isOutOfLine();
}

bool CXXMethodDecl::isLambdaStaticInvoker() const {
  return getParent()->isLambda() && 
         getIdentifier() && getIdentifier()->getName() == "__invoke";
}


CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       TypeSourceInfo *TInfo, bool IsVirtual,
                                       SourceLocation L, Expr *Init,
                                       SourceLocation R,
                                       SourceLocation EllipsisLoc)
  : Initializee(TInfo), MemberOrEllipsisLocation(EllipsisLoc), Init(Init), 
    LParenLoc(L), RParenLoc(R), IsDelegating(false), IsVirtual(IsVirtual), 
    IsWritten(false), SourceOrderOrNumArrayIndices(0)
{
}

CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       FieldDecl *Member,
                                       SourceLocation MemberLoc,
                                       SourceLocation L, Expr *Init,
                                       SourceLocation R)
  : Initializee(Member), MemberOrEllipsisLocation(MemberLoc), Init(Init),
    LParenLoc(L), RParenLoc(R), IsDelegating(false), IsVirtual(false),
    IsWritten(false), SourceOrderOrNumArrayIndices(0)
{
}

CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       IndirectFieldDecl *Member,
                                       SourceLocation MemberLoc,
                                       SourceLocation L, Expr *Init,
                                       SourceLocation R)
  : Initializee(Member), MemberOrEllipsisLocation(MemberLoc), Init(Init),
    LParenLoc(L), RParenLoc(R), IsDelegating(false), IsVirtual(false),
    IsWritten(false), SourceOrderOrNumArrayIndices(0)
{
}

CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       TypeSourceInfo *TInfo,
                                       SourceLocation L, Expr *Init, 
                                       SourceLocation R)
  : Initializee(TInfo), MemberOrEllipsisLocation(), Init(Init),
    LParenLoc(L), RParenLoc(R), IsDelegating(true), IsVirtual(false),
    IsWritten(false), SourceOrderOrNumArrayIndices(0)
{
}

CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       FieldDecl *Member,
                                       SourceLocation MemberLoc,
                                       SourceLocation L, Expr *Init,
                                       SourceLocation R,
                                       VarDecl **Indices,
                                       unsigned NumIndices)
  : Initializee(Member), MemberOrEllipsisLocation(MemberLoc), Init(Init), 
    LParenLoc(L), RParenLoc(R), IsVirtual(false),
    IsWritten(false), SourceOrderOrNumArrayIndices(NumIndices)
{
  VarDecl **MyIndices = reinterpret_cast<VarDecl **> (this + 1);
  memcpy(MyIndices, Indices, NumIndices * sizeof(VarDecl *));
}

CXXCtorInitializer *CXXCtorInitializer::Create(ASTContext &Context,
                                               FieldDecl *Member, 
                                               SourceLocation MemberLoc,
                                               SourceLocation L, Expr *Init,
                                               SourceLocation R,
                                               VarDecl **Indices,
                                               unsigned NumIndices) {
  void *Mem = Context.Allocate(sizeof(CXXCtorInitializer) +
                               sizeof(VarDecl *) * NumIndices,
                               llvm::alignOf<CXXCtorInitializer>());
  return new (Mem) CXXCtorInitializer(Context, Member, MemberLoc, L, Init, R,
                                      Indices, NumIndices);
}

TypeLoc CXXCtorInitializer::getBaseClassLoc() const {
  if (isBaseInitializer())
    return Initializee.get<TypeSourceInfo*>()->getTypeLoc();
  else
    return TypeLoc();
}

const Type *CXXCtorInitializer::getBaseClass() const {
  if (isBaseInitializer())
    return Initializee.get<TypeSourceInfo*>()->getType().getTypePtr();
  else
    return 0;
}

SourceLocation CXXCtorInitializer::getSourceLocation() const {
  if (isAnyMemberInitializer())
    return getMemberLocation();

  if (isInClassMemberInitializer())
    return getAnyMember()->getLocation();
  
  if (TypeSourceInfo *TSInfo = Initializee.get<TypeSourceInfo*>())
    return TSInfo->getTypeLoc().getLocalSourceRange().getBegin();
  
  return SourceLocation();
}

SourceRange CXXCtorInitializer::getSourceRange() const {
  if (isInClassMemberInitializer()) {
    FieldDecl *D = getAnyMember();
    if (Expr *I = D->getInClassInitializer())
      return I->getSourceRange();
    return SourceRange();
  }

  return SourceRange(getSourceLocation(), getRParenLoc());
}

void CXXConstructorDecl::anchor() { }

CXXConstructorDecl *
CXXConstructorDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(CXXConstructorDecl));
  return new (Mem) CXXConstructorDecl(0, SourceLocation(),DeclarationNameInfo(),
                                      QualType(), 0, false, false, false,false);
}

CXXConstructorDecl *
CXXConstructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                           SourceLocation StartLoc,
                           const DeclarationNameInfo &NameInfo,
                           QualType T, TypeSourceInfo *TInfo,
                           bool isExplicit, bool isInline,
                           bool isImplicitlyDeclared, bool isConstexpr) {
  assert(NameInfo.getName().getNameKind()
         == DeclarationName::CXXConstructorName &&
         "Name must refer to a constructor");
  return new (C) CXXConstructorDecl(RD, StartLoc, NameInfo, T, TInfo,
                                    isExplicit, isInline, isImplicitlyDeclared,
                                    isConstexpr);
}

CXXConstructorDecl *CXXConstructorDecl::getTargetConstructor() const {
  assert(isDelegatingConstructor() && "Not a delegating constructor!");
  Expr *E = (*init_begin())->getInit()->IgnoreImplicit();
  if (CXXConstructExpr *Construct = dyn_cast<CXXConstructExpr>(E))
    return Construct->getConstructor();
  
  return 0;
}

bool CXXConstructorDecl::isDefaultConstructor() const {
  // C++ [class.ctor]p5:
  //   A default constructor for a class X is a constructor of class
  //   X that can be called without an argument.
  return (getNumParams() == 0) ||
         (getNumParams() > 0 && getParamDecl(0)->hasDefaultArg());
}

bool
CXXConstructorDecl::isCopyConstructor(unsigned &TypeQuals) const {
  return isCopyOrMoveConstructor(TypeQuals) &&
         getParamDecl(0)->getType()->isLValueReferenceType();
}

bool CXXConstructorDecl::isMoveConstructor(unsigned &TypeQuals) const {
  return isCopyOrMoveConstructor(TypeQuals) &&
    getParamDecl(0)->getType()->isRValueReferenceType();
}

/// \brief Determine whether this is a copy or move constructor.
bool CXXConstructorDecl::isCopyOrMoveConstructor(unsigned &TypeQuals) const {
  // C++ [class.copy]p2:
  //   A non-template constructor for class X is a copy constructor
  //   if its first parameter is of type X&, const X&, volatile X& or
  //   const volatile X&, and either there are no other parameters
  //   or else all other parameters have default arguments (8.3.6).
  // C++0x [class.copy]p3:
  //   A non-template constructor for class X is a move constructor if its
  //   first parameter is of type X&&, const X&&, volatile X&&, or 
  //   const volatile X&&, and either there are no other parameters or else 
  //   all other parameters have default arguments.
  if ((getNumParams() < 1) ||
      (getNumParams() > 1 && !getParamDecl(1)->hasDefaultArg()) ||
      (getPrimaryTemplate() != 0) ||
      (getDescribedFunctionTemplate() != 0))
    return false;
  
  const ParmVarDecl *Param = getParamDecl(0);
  
  // Do we have a reference type? 
  const ReferenceType *ParamRefType = Param->getType()->getAs<ReferenceType>();
  if (!ParamRefType)
    return false;
  
  // Is it a reference to our class type?
  ASTContext &Context = getASTContext();
  
  CanQualType PointeeType
    = Context.getCanonicalType(ParamRefType->getPointeeType());
  CanQualType ClassTy 
    = Context.getCanonicalType(Context.getTagDeclType(getParent()));
  if (PointeeType.getUnqualifiedType() != ClassTy)
    return false;
  
  // FIXME: other qualifiers?
  
  // We have a copy or move constructor.
  TypeQuals = PointeeType.getCVRQualifiers();
  return true;  
}

bool CXXConstructorDecl::isConvertingConstructor(bool AllowExplicit) const {
  // C++ [class.conv.ctor]p1:
  //   A constructor declared without the function-specifier explicit
  //   that can be called with a single parameter specifies a
  //   conversion from the type of its first parameter to the type of
  //   its class. Such a constructor is called a converting
  //   constructor.
  if (isExplicit() && !AllowExplicit)
    return false;

  return (getNumParams() == 0 &&
          getType()->getAs<FunctionProtoType>()->isVariadic()) ||
         (getNumParams() == 1) ||
         (getNumParams() > 1 &&
          (getParamDecl(1)->hasDefaultArg() ||
           getParamDecl(1)->isParameterPack()));
}

bool CXXConstructorDecl::isSpecializationCopyingObject() const {
  if ((getNumParams() < 1) ||
      (getNumParams() > 1 && !getParamDecl(1)->hasDefaultArg()) ||
      (getPrimaryTemplate() == 0) ||
      (getDescribedFunctionTemplate() != 0))
    return false;

  const ParmVarDecl *Param = getParamDecl(0);

  ASTContext &Context = getASTContext();
  CanQualType ParamType = Context.getCanonicalType(Param->getType());
  
  // Is it the same as our our class type?
  CanQualType ClassTy 
    = Context.getCanonicalType(Context.getTagDeclType(getParent()));
  if (ParamType.getUnqualifiedType() != ClassTy)
    return false;
  
  return true;  
}

const CXXConstructorDecl *CXXConstructorDecl::getInheritedConstructor() const {
  // Hack: we store the inherited constructor in the overridden method table
  method_iterator It = getASTContext().overridden_methods_begin(this);
  if (It == getASTContext().overridden_methods_end(this))
    return 0;

  return cast<CXXConstructorDecl>(*It);
}

void
CXXConstructorDecl::setInheritedConstructor(const CXXConstructorDecl *BaseCtor){
  // Hack: we store the inherited constructor in the overridden method table
  assert(getASTContext().overridden_methods_size(this) == 0 &&
         "Base ctor already set.");
  getASTContext().addOverriddenMethod(this, BaseCtor);
}

void CXXDestructorDecl::anchor() { }

CXXDestructorDecl *
CXXDestructorDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(CXXDestructorDecl));
  return new (Mem) CXXDestructorDecl(0, SourceLocation(), DeclarationNameInfo(),
                                   QualType(), 0, false, false);
}

CXXDestructorDecl *
CXXDestructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          SourceLocation StartLoc,
                          const DeclarationNameInfo &NameInfo,
                          QualType T, TypeSourceInfo *TInfo,
                          bool isInline, bool isImplicitlyDeclared) {
  assert(NameInfo.getName().getNameKind()
         == DeclarationName::CXXDestructorName &&
         "Name must refer to a destructor");
  return new (C) CXXDestructorDecl(RD, StartLoc, NameInfo, T, TInfo, isInline,
                                   isImplicitlyDeclared);
}

void CXXConversionDecl::anchor() { }

CXXConversionDecl *
CXXConversionDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(CXXConversionDecl));
  return new (Mem) CXXConversionDecl(0, SourceLocation(), DeclarationNameInfo(),
                                     QualType(), 0, false, false, false,
                                     SourceLocation());
}

CXXConversionDecl *
CXXConversionDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          SourceLocation StartLoc,
                          const DeclarationNameInfo &NameInfo,
                          QualType T, TypeSourceInfo *TInfo,
                          bool isInline, bool isExplicit,
                          bool isConstexpr, SourceLocation EndLocation) {
  assert(NameInfo.getName().getNameKind()
         == DeclarationName::CXXConversionFunctionName &&
         "Name must refer to a conversion function");
  return new (C) CXXConversionDecl(RD, StartLoc, NameInfo, T, TInfo,
                                   isInline, isExplicit, isConstexpr,
                                   EndLocation);
}

bool CXXConversionDecl::isLambdaToBlockPointerConversion() const {
  return isImplicit() && getParent()->isLambda() &&
         getConversionType()->isBlockPointerType();
}

void LinkageSpecDecl::anchor() { }

LinkageSpecDecl *LinkageSpecDecl::Create(ASTContext &C,
                                         DeclContext *DC,
                                         SourceLocation ExternLoc,
                                         SourceLocation LangLoc,
                                         LanguageIDs Lang,
                                         SourceLocation RBraceLoc) {
  return new (C) LinkageSpecDecl(DC, ExternLoc, LangLoc, Lang, RBraceLoc);
}

LinkageSpecDecl *LinkageSpecDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(LinkageSpecDecl));
  return new (Mem) LinkageSpecDecl(0, SourceLocation(), SourceLocation(),
                                   lang_c, SourceLocation());
}

void UsingDirectiveDecl::anchor() { }

UsingDirectiveDecl *UsingDirectiveDecl::Create(ASTContext &C, DeclContext *DC,
                                               SourceLocation L,
                                               SourceLocation NamespaceLoc,
                                           NestedNameSpecifierLoc QualifierLoc,
                                               SourceLocation IdentLoc,
                                               NamedDecl *Used,
                                               DeclContext *CommonAncestor) {
  if (NamespaceDecl *NS = dyn_cast_or_null<NamespaceDecl>(Used))
    Used = NS->getOriginalNamespace();
  return new (C) UsingDirectiveDecl(DC, L, NamespaceLoc, QualifierLoc,
                                    IdentLoc, Used, CommonAncestor);
}

UsingDirectiveDecl *
UsingDirectiveDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(UsingDirectiveDecl));
  return new (Mem) UsingDirectiveDecl(0, SourceLocation(), SourceLocation(),
                                      NestedNameSpecifierLoc(),
                                      SourceLocation(), 0, 0);
}

NamespaceDecl *UsingDirectiveDecl::getNominatedNamespace() {
  if (NamespaceAliasDecl *NA =
        dyn_cast_or_null<NamespaceAliasDecl>(NominatedNamespace))
    return NA->getNamespace();
  return cast_or_null<NamespaceDecl>(NominatedNamespace);
}

void NamespaceDecl::anchor() { }

NamespaceDecl::NamespaceDecl(DeclContext *DC, bool Inline, 
                             SourceLocation StartLoc,
                             SourceLocation IdLoc, IdentifierInfo *Id,
                             NamespaceDecl *PrevDecl)
  : NamedDecl(Namespace, DC, IdLoc, Id), DeclContext(Namespace),
    LocStart(StartLoc), RBraceLoc(), AnonOrFirstNamespaceAndInline(0, Inline) 
{
  setPreviousDeclaration(PrevDecl);
  
  if (PrevDecl)
    AnonOrFirstNamespaceAndInline.setPointer(PrevDecl->getOriginalNamespace());
}

NamespaceDecl *NamespaceDecl::Create(ASTContext &C, DeclContext *DC,
                                     bool Inline, SourceLocation StartLoc,
                                     SourceLocation IdLoc, IdentifierInfo *Id,
                                     NamespaceDecl *PrevDecl) {
  return new (C) NamespaceDecl(DC, Inline, StartLoc, IdLoc, Id, PrevDecl);
}

NamespaceDecl *NamespaceDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(NamespaceDecl));
  return new (Mem) NamespaceDecl(0, false, SourceLocation(), SourceLocation(), 
                                 0, 0);
}

void NamespaceAliasDecl::anchor() { }

NamespaceAliasDecl *NamespaceAliasDecl::Create(ASTContext &C, DeclContext *DC,
                                               SourceLocation UsingLoc,
                                               SourceLocation AliasLoc,
                                               IdentifierInfo *Alias,
                                           NestedNameSpecifierLoc QualifierLoc,
                                               SourceLocation IdentLoc,
                                               NamedDecl *Namespace) {
  if (NamespaceDecl *NS = dyn_cast_or_null<NamespaceDecl>(Namespace))
    Namespace = NS->getOriginalNamespace();
  return new (C) NamespaceAliasDecl(DC, UsingLoc, AliasLoc, Alias, 
                                    QualifierLoc, IdentLoc, Namespace);
}

NamespaceAliasDecl *
NamespaceAliasDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(NamespaceAliasDecl));
  return new (Mem) NamespaceAliasDecl(0, SourceLocation(), SourceLocation(), 0,
                                      NestedNameSpecifierLoc(), 
                                      SourceLocation(), 0);
}

void UsingShadowDecl::anchor() { }

UsingShadowDecl *
UsingShadowDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(UsingShadowDecl));
  return new (Mem) UsingShadowDecl(0, SourceLocation(), 0, 0);
}

UsingDecl *UsingShadowDecl::getUsingDecl() const {
  const UsingShadowDecl *Shadow = this;
  while (const UsingShadowDecl *NextShadow =
         dyn_cast<UsingShadowDecl>(Shadow->UsingOrNextShadow))
    Shadow = NextShadow;
  return cast<UsingDecl>(Shadow->UsingOrNextShadow);
}

void UsingDecl::anchor() { }

void UsingDecl::addShadowDecl(UsingShadowDecl *S) {
  assert(std::find(shadow_begin(), shadow_end(), S) == shadow_end() &&
         "declaration already in set");
  assert(S->getUsingDecl() == this);

  if (FirstUsingShadow.getPointer())
    S->UsingOrNextShadow = FirstUsingShadow.getPointer();
  FirstUsingShadow.setPointer(S);
}

void UsingDecl::removeShadowDecl(UsingShadowDecl *S) {
  assert(std::find(shadow_begin(), shadow_end(), S) != shadow_end() &&
         "declaration not in set");
  assert(S->getUsingDecl() == this);

  // Remove S from the shadow decl chain. This is O(n) but hopefully rare.

  if (FirstUsingShadow.getPointer() == S) {
    FirstUsingShadow.setPointer(
      dyn_cast<UsingShadowDecl>(S->UsingOrNextShadow));
    S->UsingOrNextShadow = this;
    return;
  }

  UsingShadowDecl *Prev = FirstUsingShadow.getPointer();
  while (Prev->UsingOrNextShadow != S)
    Prev = cast<UsingShadowDecl>(Prev->UsingOrNextShadow);
  Prev->UsingOrNextShadow = S->UsingOrNextShadow;
  S->UsingOrNextShadow = this;
}

UsingDecl *UsingDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation UL,
                             NestedNameSpecifierLoc QualifierLoc,
                             const DeclarationNameInfo &NameInfo,
                             bool IsTypeNameArg) {
  return new (C) UsingDecl(DC, UL, QualifierLoc, NameInfo, IsTypeNameArg);
}

UsingDecl *UsingDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(UsingDecl));
  return new (Mem) UsingDecl(0, SourceLocation(), NestedNameSpecifierLoc(),
                             DeclarationNameInfo(), false);
}

void UnresolvedUsingValueDecl::anchor() { }

UnresolvedUsingValueDecl *
UnresolvedUsingValueDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation UsingLoc,
                                 NestedNameSpecifierLoc QualifierLoc,
                                 const DeclarationNameInfo &NameInfo) {
  return new (C) UnresolvedUsingValueDecl(DC, C.DependentTy, UsingLoc,
                                          QualifierLoc, NameInfo);
}

UnresolvedUsingValueDecl *
UnresolvedUsingValueDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(UnresolvedUsingValueDecl));
  return new (Mem) UnresolvedUsingValueDecl(0, QualType(), SourceLocation(),
                                            NestedNameSpecifierLoc(),
                                            DeclarationNameInfo());
}

void UnresolvedUsingTypenameDecl::anchor() { }

UnresolvedUsingTypenameDecl *
UnresolvedUsingTypenameDecl::Create(ASTContext &C, DeclContext *DC,
                                    SourceLocation UsingLoc,
                                    SourceLocation TypenameLoc,
                                    NestedNameSpecifierLoc QualifierLoc,
                                    SourceLocation TargetNameLoc,
                                    DeclarationName TargetName) {
  return new (C) UnresolvedUsingTypenameDecl(DC, UsingLoc, TypenameLoc,
                                             QualifierLoc, TargetNameLoc,
                                             TargetName.getAsIdentifierInfo());
}

UnresolvedUsingTypenameDecl *
UnresolvedUsingTypenameDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, 
                                       sizeof(UnresolvedUsingTypenameDecl));
  return new (Mem) UnresolvedUsingTypenameDecl(0, SourceLocation(),
                                               SourceLocation(),
                                               NestedNameSpecifierLoc(),
                                               SourceLocation(),
                                               0);
}

void StaticAssertDecl::anchor() { }

StaticAssertDecl *StaticAssertDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation StaticAssertLoc,
                                           Expr *AssertExpr,
                                           StringLiteral *Message,
                                           SourceLocation RParenLoc,
                                           bool Failed) {
  return new (C) StaticAssertDecl(DC, StaticAssertLoc, AssertExpr, Message,
                                  RParenLoc, Failed);
}

StaticAssertDecl *StaticAssertDecl::CreateDeserialized(ASTContext &C, 
                                                       unsigned ID) {
  void *Mem = AllocateDeserializedDecl(C, ID, sizeof(StaticAssertDecl));
  return new (Mem) StaticAssertDecl(0, SourceLocation(), 0, 0,
                                    SourceLocation(), false);
}

static const char *getAccessName(AccessSpecifier AS) {
  switch (AS) {
    case AS_none:
      llvm_unreachable("Invalid access specifier!");
    case AS_public:
      return "public";
    case AS_private:
      return "private";
    case AS_protected:
      return "protected";
  }
  llvm_unreachable("Invalid access specifier!");
}

const DiagnosticBuilder &clang::operator<<(const DiagnosticBuilder &DB,
                                           AccessSpecifier AS) {
  return DB << getAccessName(AS);
}

const PartialDiagnostic &clang::operator<<(const PartialDiagnostic &DB,
                                           AccessSpecifier AS) {
  return DB << getAccessName(AS);
}
