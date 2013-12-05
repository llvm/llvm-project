//===--- Initialization.h - Semantic Analysis for Initializers --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides supporting data types for initialization of objects.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_INITIALIZATION_H
#define LLVM_CLANG_SEMA_INITIALIZATION_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Type.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Ownership.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace clang {
  
class CXXBaseSpecifier;
class DeclaratorDecl;
class DeclaratorInfo;
class FieldDecl;
class FunctionDecl;
class ParmVarDecl;
class Sema;
class TypeLoc;
class VarDecl;
class ObjCMethodDecl;
  
/// \brief Describes an entity that is being initialized.
class InitializedEntity {
public:
  /// \brief Specifies the kind of entity being initialized.
  enum EntityKind {
    /// \brief The entity being initialized is a variable.
    EK_Variable,
    /// \brief The entity being initialized is a function parameter.
    EK_Parameter,
    /// \brief The entity being initialized is the result of a function call.
    EK_Result,
    /// \brief The entity being initialized is an exception object that
    /// is being thrown.
    EK_Exception,
    /// \brief The entity being initialized is a non-static data member 
    /// subobject.
    EK_Member,
    /// \brief The entity being initialized is an element of an array.
    EK_ArrayElement,
    /// \brief The entity being initialized is an object (or array of
    /// objects) allocated via new.
    EK_New,
    /// \brief The entity being initialized is a temporary object.
    EK_Temporary,
    /// \brief The entity being initialized is a base member subobject.
    EK_Base,
    /// \brief The initialization is being done by a delegating constructor.
    EK_Delegating,
    /// \brief The entity being initialized is an element of a vector.
    /// or vector.
    EK_VectorElement,
    /// \brief The entity being initialized is a field of block descriptor for
    /// the copied-in c++ object.
    EK_BlockElement,
    /// \brief The entity being initialized is the real or imaginary part of a
    /// complex number.
    EK_ComplexElement,
    /// \brief The entity being initialized is the field that captures a 
    /// variable in a lambda.
    EK_LambdaCapture,
    /// \brief The entity being initialized is the initializer for a compound
    /// literal.
    EK_CompoundLiteralInit,
    /// \brief The entity being implicitly initialized back to the formal
    /// result type.
    EK_RelatedResult,
    /// \brief The entity being initialized is a function parameter; function
    /// is member of group of audited CF APIs.
    EK_Parameter_CF_Audited

    // Note: err_init_conversion_failed in DiagnosticSemaKinds.td uses this
    // enum as an index for its first %select.  When modifying this list,
    // that diagnostic text needs to be updated as well.
  };
  
private:
  /// \brief The kind of entity being initialized.
  EntityKind Kind;

  /// \brief If non-NULL, the parent entity in which this
  /// initialization occurs.
  const InitializedEntity *Parent;

  /// \brief The type of the object or reference being initialized.
  QualType Type;

  struct LN {
    /// \brief When Kind == EK_Result, EK_Exception, EK_New, the
    /// location of the 'return', 'throw', or 'new' keyword,
    /// respectively. When Kind == EK_Temporary, the location where
    /// the temporary is being created.
    unsigned Location;

    /// \brief Whether the entity being initialized may end up using the
    /// named return value optimization (NRVO).
    bool NRVO;
  };

  struct C {
    /// \brief The name of the variable being captured by an EK_LambdaCapture.
    IdentifierInfo *VarID;

    /// \brief The source location at which the capture occurs.
    unsigned Location;
  };

  union {
    /// \brief When Kind == EK_Variable, or EK_Member, the VarDecl or
    /// FieldDecl, respectively.
    DeclaratorDecl *VariableOrMember;
    
    /// \brief When Kind == EK_RelatedResult, the ObjectiveC method where
    /// result type was implicitly changed to accommodate ARC semantics.
    ObjCMethodDecl *MethodDecl;

    /// \brief When Kind == EK_Parameter, the ParmVarDecl, with the
    /// low bit indicating whether the parameter is "consumed".
    uintptr_t Parameter;
    
    /// \brief When Kind == EK_Temporary or EK_CompoundLiteralInit, the type
    /// source information for the temporary.
    TypeSourceInfo *TypeInfo;

    struct LN LocAndNRVO;
    
    /// \brief When Kind == EK_Base, the base specifier that provides the 
    /// base class. The lower bit specifies whether the base is an inherited
    /// virtual base.
    uintptr_t Base;

    /// \brief When Kind == EK_ArrayElement, EK_VectorElement, or
    /// EK_ComplexElement, the index of the array or vector element being
    /// initialized. 
    unsigned Index;

    struct C Capture;
  };

  InitializedEntity() { }

  /// \brief Create the initialization entity for a variable.
  InitializedEntity(VarDecl *Var)
    : Kind(EK_Variable), Parent(0), Type(Var->getType()),
      VariableOrMember(Var) { }
  
  /// \brief Create the initialization entity for the result of a
  /// function, throwing an object, performing an explicit cast, or
  /// initializing a parameter for which there is no declaration.
  InitializedEntity(EntityKind Kind, SourceLocation Loc, QualType Type,
                    bool NRVO = false)
    : Kind(Kind), Parent(0), Type(Type)
  {
    LocAndNRVO.Location = Loc.getRawEncoding();
    LocAndNRVO.NRVO = NRVO;
  }
  
  /// \brief Create the initialization entity for a member subobject.
  InitializedEntity(FieldDecl *Member, const InitializedEntity *Parent) 
    : Kind(EK_Member), Parent(Parent), Type(Member->getType()),
      VariableOrMember(Member) { }
  
  /// \brief Create the initialization entity for an array element.
  InitializedEntity(ASTContext &Context, unsigned Index, 
                    const InitializedEntity &Parent);

  /// \brief Create the initialization entity for a lambda capture.
  InitializedEntity(IdentifierInfo *VarID, QualType FieldType, SourceLocation Loc)
    : Kind(EK_LambdaCapture), Parent(0), Type(FieldType) 
  {
    Capture.VarID = VarID;
    Capture.Location = Loc.getRawEncoding();
  }
  
public:
  /// \brief Create the initialization entity for a variable.
  static InitializedEntity InitializeVariable(VarDecl *Var) {
    return InitializedEntity(Var);
  }

  /// \brief Create the initialization entity for a parameter.
  static InitializedEntity InitializeParameter(ASTContext &Context,
                                               ParmVarDecl *Parm) {
    return InitializeParameter(Context, Parm, Parm->getType());
  }

  /// \brief Create the initialization entity for a parameter, but use
  /// another type.
  static InitializedEntity InitializeParameter(ASTContext &Context,
                                               ParmVarDecl *Parm,
                                               QualType Type) {
    bool Consumed = (Context.getLangOpts().ObjCAutoRefCount &&
                     Parm->hasAttr<NSConsumedAttr>());

    InitializedEntity Entity;
    Entity.Kind = EK_Parameter;
    Entity.Type =
      Context.getVariableArrayDecayedType(Type.getUnqualifiedType());
    Entity.Parent = 0;
    Entity.Parameter
      = (static_cast<uintptr_t>(Consumed) | reinterpret_cast<uintptr_t>(Parm));
    return Entity;
  }

  /// \brief Create the initialization entity for a parameter that is
  /// only known by its type.
  static InitializedEntity InitializeParameter(ASTContext &Context,
                                               QualType Type,
                                               bool Consumed) {
    InitializedEntity Entity;
    Entity.Kind = EK_Parameter;
    Entity.Type = Context.getVariableArrayDecayedType(Type);
    Entity.Parent = 0;
    Entity.Parameter = (Consumed);
    return Entity;
  }

  /// \brief Create the initialization entity for the result of a function.
  static InitializedEntity InitializeResult(SourceLocation ReturnLoc,
                                            QualType Type, bool NRVO) {
    return InitializedEntity(EK_Result, ReturnLoc, Type, NRVO);
  }

  static InitializedEntity InitializeBlock(SourceLocation BlockVarLoc,
                                           QualType Type, bool NRVO) {
    return InitializedEntity(EK_BlockElement, BlockVarLoc, Type, NRVO);
  }
  
  /// \brief Create the initialization entity for an exception object.
  static InitializedEntity InitializeException(SourceLocation ThrowLoc,
                                               QualType Type, bool NRVO) {
    return InitializedEntity(EK_Exception, ThrowLoc, Type, NRVO);
  }

  /// \brief Create the initialization entity for an object allocated via new.
  static InitializedEntity InitializeNew(SourceLocation NewLoc, QualType Type) {
    return InitializedEntity(EK_New, NewLoc, Type);
  }
  
  /// \brief Create the initialization entity for a temporary.
  static InitializedEntity InitializeTemporary(QualType Type) {
    InitializedEntity Result(EK_Temporary, SourceLocation(), Type);
    Result.TypeInfo = 0;
    return Result;
  }

  /// \brief Create the initialization entity for a temporary.
  static InitializedEntity InitializeTemporary(TypeSourceInfo *TypeInfo) {
    InitializedEntity Result(EK_Temporary, SourceLocation(), 
                             TypeInfo->getType());
    Result.TypeInfo = TypeInfo;
    return Result;
  }
  
  /// \brief Create the initialization entity for a related result.
  static InitializedEntity InitializeRelatedResult(ObjCMethodDecl *MD,
                                                   QualType Type) {
    InitializedEntity Result(EK_RelatedResult, SourceLocation(), Type);
    Result.MethodDecl = MD;
    return Result;
  }


  /// \brief Create the initialization entity for a base class subobject.
  static InitializedEntity InitializeBase(ASTContext &Context,
                                          const CXXBaseSpecifier *Base,
                                          bool IsInheritedVirtualBase);

  /// \brief Create the initialization entity for a delegated constructor.
  static InitializedEntity InitializeDelegation(QualType Type) {
    return InitializedEntity(EK_Delegating, SourceLocation(), Type);
  }
  
  /// \brief Create the initialization entity for a member subobject.
  static InitializedEntity InitializeMember(FieldDecl *Member,
                                          const InitializedEntity *Parent = 0) {
    return InitializedEntity(Member, Parent);
  }
  
  /// \brief Create the initialization entity for a member subobject.
  static InitializedEntity InitializeMember(IndirectFieldDecl *Member,
                                      const InitializedEntity *Parent = 0) {
    return InitializedEntity(Member->getAnonField(), Parent);
  }

  /// \brief Create the initialization entity for an array element.
  static InitializedEntity InitializeElement(ASTContext &Context, 
                                             unsigned Index, 
                                             const InitializedEntity &Parent) {
    return InitializedEntity(Context, Index, Parent);
  }

  /// \brief Create the initialization entity for a lambda capture.
  static InitializedEntity InitializeLambdaCapture(IdentifierInfo *VarID,
                                                   QualType FieldType,
                                                   SourceLocation Loc) {
    return InitializedEntity(VarID, FieldType, Loc);
  }

  /// \brief Create the entity for a compound literal initializer.
  static InitializedEntity InitializeCompoundLiteralInit(TypeSourceInfo *TSI) {
    InitializedEntity Result(EK_CompoundLiteralInit, SourceLocation(),
                             TSI->getType());
    Result.TypeInfo = TSI;
    return Result;
  }


  /// \brief Determine the kind of initialization.
  EntityKind getKind() const { return Kind; }
  
  /// \brief Retrieve the parent of the entity being initialized, when
  /// the initialization itself is occurring within the context of a
  /// larger initialization.
  const InitializedEntity *getParent() const { return Parent; }

  /// \brief Retrieve type being initialized.
  QualType getType() const { return Type; }
  
  /// \brief Retrieve complete type-source information for the object being 
  /// constructed, if known.
  TypeSourceInfo *getTypeSourceInfo() const {
    if (Kind == EK_Temporary || Kind == EK_CompoundLiteralInit)
      return TypeInfo;
    
    return 0;
  }
  
  /// \brief Retrieve the name of the entity being initialized.
  DeclarationName getName() const;

  /// \brief Retrieve the variable, parameter, or field being
  /// initialized.
  DeclaratorDecl *getDecl() const;
  
  /// \brief Retrieve the ObjectiveC method being initialized.
  ObjCMethodDecl *getMethodDecl() const { return MethodDecl; }

  /// \brief Determine whether this initialization allows the named return 
  /// value optimization, which also applies to thrown objects.
  bool allowsNRVO() const;

  bool isParameterKind() const {
    return (getKind() == EK_Parameter  ||
            getKind() == EK_Parameter_CF_Audited);
  }
  /// \brief Determine whether this initialization consumes the
  /// parameter.
  bool isParameterConsumed() const {
    assert(isParameterKind() && "Not a parameter");
    return (Parameter & 1);
  }
                                  
  /// \brief Retrieve the base specifier.
  const CXXBaseSpecifier *getBaseSpecifier() const {
    assert(getKind() == EK_Base && "Not a base specifier");
    return reinterpret_cast<const CXXBaseSpecifier *>(Base & ~0x1);
  }

  /// \brief Return whether the base is an inherited virtual base.
  bool isInheritedVirtualBase() const {
    assert(getKind() == EK_Base && "Not a base specifier");
    return Base & 0x1;
  }

  /// \brief Determine the location of the 'return' keyword when initializing
  /// the result of a function call.
  SourceLocation getReturnLoc() const {
    assert(getKind() == EK_Result && "No 'return' location!");
    return SourceLocation::getFromRawEncoding(LocAndNRVO.Location);
  }

  /// \brief Determine the location of the 'throw' keyword when initializing
  /// an exception object.
  SourceLocation getThrowLoc() const {
    assert(getKind() == EK_Exception && "No 'throw' location!");
    return SourceLocation::getFromRawEncoding(LocAndNRVO.Location);
  }

  /// \brief If this is already the initializer for an array or vector
  /// element, sets the element index.
  void setElementIndex(unsigned Index) {
    assert(getKind() == EK_ArrayElement || getKind() == EK_VectorElement ||
           getKind() == EK_ComplexElement);
    this->Index = Index;
  }
  /// \brief For a lambda capture, return the capture's name.
  StringRef getCapturedVarName() const {
    assert(getKind() == EK_LambdaCapture && "Not a lambda capture!");
    return Capture.VarID->getName();
  }
  /// \brief Determine the location of the capture when initializing
  /// field from a captured variable in a lambda.
  SourceLocation getCaptureLoc() const {
    assert(getKind() == EK_LambdaCapture && "Not a lambda capture!");
    return SourceLocation::getFromRawEncoding(Capture.Location);
  }
  
  void setParameterCFAudited() {
    Kind = EK_Parameter_CF_Audited;
  }

  /// Dump a representation of the initialized entity to standard error,
  /// for debugging purposes.
  void dump() const;

private:
  unsigned dumpImpl(raw_ostream &OS) const;
};
  
/// \brief Describes the kind of initialization being performed, along with 
/// location information for tokens related to the initialization (equal sign,
/// parentheses).
class InitializationKind {
public:
  /// \brief The kind of initialization being performed.
  enum InitKind {
    IK_Direct,       ///< Direct initialization
    IK_DirectList,   ///< Direct list-initialization
    IK_Copy,         ///< Copy initialization
    IK_Default,      ///< Default initialization
    IK_Value         ///< Value initialization
  };
  
private:
  /// \brief The context of the initialization.
  enum InitContext {
    IC_Normal,         ///< Normal context
    IC_ExplicitConvs,  ///< Normal context, but allows explicit conversion funcs
    IC_Implicit,       ///< Implicit context (value initialization)
    IC_StaticCast,     ///< Static cast context
    IC_CStyleCast,     ///< C-style cast context
    IC_FunctionalCast  ///< Functional cast context
  };
  
  /// \brief The kind of initialization being performed.
  InitKind Kind : 8;

  /// \brief The context of the initialization.
  InitContext Context : 8;
  
  /// \brief The source locations involved in the initialization.
  SourceLocation Locations[3];
  
  InitializationKind(InitKind Kind, InitContext Context, SourceLocation Loc1, 
                     SourceLocation Loc2, SourceLocation Loc3)
    : Kind(Kind), Context(Context)
  {
    Locations[0] = Loc1;
    Locations[1] = Loc2;
    Locations[2] = Loc3;
  }
  
public:
  /// \brief Create a direct initialization.
  static InitializationKind CreateDirect(SourceLocation InitLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation RParenLoc) {
    return InitializationKind(IK_Direct, IC_Normal,
                              InitLoc, LParenLoc, RParenLoc);
  }

  static InitializationKind CreateDirectList(SourceLocation InitLoc) {
    return InitializationKind(IK_DirectList, IC_Normal,
                              InitLoc, InitLoc, InitLoc);
  }

  /// \brief Create a direct initialization due to a cast that isn't a C-style 
  /// or functional cast.
  static InitializationKind CreateCast(SourceRange TypeRange) {
    return InitializationKind(IK_Direct, IC_StaticCast, TypeRange.getBegin(),
                              TypeRange.getBegin(), TypeRange.getEnd());
  }
  
  /// \brief Create a direct initialization for a C-style cast.
  static InitializationKind CreateCStyleCast(SourceLocation StartLoc,
                                             SourceRange TypeRange,
                                             bool InitList) {
    // C++ cast syntax doesn't permit init lists, but C compound literals are
    // exactly that.
    return InitializationKind(InitList ? IK_DirectList : IK_Direct,
                              IC_CStyleCast, StartLoc, TypeRange.getBegin(),
                              TypeRange.getEnd());
  }

  /// \brief Create a direct initialization for a functional cast.
  static InitializationKind CreateFunctionalCast(SourceRange TypeRange,
                                                 bool InitList) {
    return InitializationKind(InitList ? IK_DirectList : IK_Direct,
                              IC_FunctionalCast, TypeRange.getBegin(),
                              TypeRange.getBegin(), TypeRange.getEnd());
  }

  /// \brief Create a copy initialization.
  static InitializationKind CreateCopy(SourceLocation InitLoc,
                                       SourceLocation EqualLoc,
                                       bool AllowExplicitConvs = false) {
    return InitializationKind(IK_Copy, 
                              AllowExplicitConvs? IC_ExplicitConvs : IC_Normal,
                              InitLoc, EqualLoc, EqualLoc);
  }
  
  /// \brief Create a default initialization.
  static InitializationKind CreateDefault(SourceLocation InitLoc) {
    return InitializationKind(IK_Default, IC_Normal, InitLoc, InitLoc, InitLoc);
  }
  
  /// \brief Create a value initialization.
  static InitializationKind CreateValue(SourceLocation InitLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation RParenLoc,
                                        bool isImplicit = false) {
    return InitializationKind(IK_Value, isImplicit ? IC_Implicit : IC_Normal,
                              InitLoc, LParenLoc, RParenLoc);
  }
  
  /// \brief Determine the initialization kind.
  InitKind getKind() const {
    return Kind;
  }
  
  /// \brief Determine whether this initialization is an explicit cast.
  bool isExplicitCast() const {
    return Context >= IC_StaticCast;
  }
  
  /// \brief Determine whether this initialization is a C-style cast.
  bool isCStyleOrFunctionalCast() const { 
    return Context >= IC_CStyleCast; 
  }

  /// \brief Determine whether this is a C-style cast.
  bool isCStyleCast() const {
    return Context == IC_CStyleCast;
  }

  /// \brief Determine whether this is a functional-style cast.
  bool isFunctionalCast() const {
    return Context == IC_FunctionalCast;
  }

  /// \brief Determine whether this initialization is an implicit
  /// value-initialization, e.g., as occurs during aggregate
  /// initialization.
  bool isImplicitValueInit() const { return Context == IC_Implicit; }

  /// \brief Retrieve the location at which initialization is occurring.
  SourceLocation getLocation() const { return Locations[0]; }
  
  /// \brief Retrieve the source range that covers the initialization.
  SourceRange getRange() const { 
    return SourceRange(Locations[0], Locations[2]);
  }
  
  /// \brief Retrieve the location of the equal sign for copy initialization
  /// (if present).
  SourceLocation getEqualLoc() const {
    assert(Kind == IK_Copy && "Only copy initialization has an '='");
    return Locations[1];
  }

  bool isCopyInit() const { return Kind == IK_Copy; }

  /// \brief Retrieve whether this initialization allows the use of explicit
  ///        constructors.
  bool AllowExplicit() const { return !isCopyInit(); }

  /// \brief Retrieve whether this initialization allows the use of explicit
  /// conversion functions when binding a reference. If the reference is the
  /// first parameter in a copy or move constructor, such conversions are
  /// permitted even though we are performing copy-initialization.
  bool allowExplicitConversionFunctionsInRefBinding() const {
    return !isCopyInit() || Context == IC_ExplicitConvs;
  }
  
  /// \brief Retrieve the source range containing the locations of the open
  /// and closing parentheses for value and direct initializations.
  SourceRange getParenRange() const {
    assert((Kind == IK_Direct || Kind == IK_Value) &&
           "Only direct- and value-initialization have parentheses");
    return SourceRange(Locations[1], Locations[2]);
  }
};

/// \brief Describes the sequence of initializations required to initialize
/// a given object or reference with a set of arguments.
class InitializationSequence {
public:
  /// \brief Describes the kind of initialization sequence computed.
  enum SequenceKind {
    /// \brief A failed initialization sequence. The failure kind tells what
    /// happened.
    FailedSequence = 0,

    /// \brief A dependent initialization, which could not be
    /// type-checked due to the presence of dependent types or
    /// dependently-typed expressions.
    DependentSequence,

    /// \brief A normal sequence.
    NormalSequence
  };
  
  /// \brief Describes the kind of a particular step in an initialization
  /// sequence.
  enum StepKind {
    /// \brief Resolve the address of an overloaded function to a specific
    /// function declaration.
    SK_ResolveAddressOfOverloadedFunction,
    /// \brief Perform a derived-to-base cast, producing an rvalue.
    SK_CastDerivedToBaseRValue,
    /// \brief Perform a derived-to-base cast, producing an xvalue.
    SK_CastDerivedToBaseXValue,
    /// \brief Perform a derived-to-base cast, producing an lvalue.
    SK_CastDerivedToBaseLValue,
    /// \brief Reference binding to an lvalue.
    SK_BindReference,
    /// \brief Reference binding to a temporary.
    SK_BindReferenceToTemporary,
    /// \brief An optional copy of a temporary object to another
    /// temporary object, which is permitted (but not required) by
    /// C++98/03 but not C++0x.
    SK_ExtraneousCopyToTemporary,
    /// \brief Perform a user-defined conversion, either via a conversion
    /// function or via a constructor.
    SK_UserConversion,
    /// \brief Perform a qualification conversion, producing an rvalue.
    SK_QualificationConversionRValue,
    /// \brief Perform a qualification conversion, producing an xvalue.
    SK_QualificationConversionXValue,
    /// \brief Perform a qualification conversion, producing an lvalue.
    SK_QualificationConversionLValue,
    /// \brief Perform a load from a glvalue, producing an rvalue.
    SK_LValueToRValue,
    /// \brief Perform an implicit conversion sequence.
    SK_ConversionSequence,
    /// \brief Perform an implicit conversion sequence without narrowing.
    SK_ConversionSequenceNoNarrowing,
    /// \brief Perform list-initialization without a constructor
    SK_ListInitialization,
    /// \brief Perform list-initialization with a constructor.
    SK_ListConstructorCall,
    /// \brief Unwrap the single-element initializer list for a reference.
    SK_UnwrapInitList,
    /// \brief Rewrap the single-element initializer list for a reference.
    SK_RewrapInitList,
    /// \brief Perform initialization via a constructor.
    SK_ConstructorInitialization,
    /// \brief Zero-initialize the object
    SK_ZeroInitialization,
    /// \brief C assignment
    SK_CAssignment,
    /// \brief Initialization by string
    SK_StringInit,
    /// \brief An initialization that "converts" an Objective-C object
    /// (not a point to an object) to another Objective-C object type.
    SK_ObjCObjectConversion,
    /// \brief Array initialization (from an array rvalue).
    /// This is a GNU C extension.
    SK_ArrayInit,
    /// \brief Array initialization from a parenthesized initializer list.
    /// This is a GNU C++ extension.
    SK_ParenthesizedArrayInit,
    /// \brief Pass an object by indirect copy-and-restore.
    SK_PassByIndirectCopyRestore,
    /// \brief Pass an object by indirect restore.
    SK_PassByIndirectRestore,
    /// \brief Produce an Objective-C object pointer.
    SK_ProduceObjCObject,
    /// \brief Construct a std::initializer_list from an initializer list.
    SK_StdInitializerList,
    /// \brief Initialize an OpenCL sampler from an integer.
    SK_OCLSamplerInit,
    /// \brief Passing zero to a function where OpenCL event_t is expected.
    SK_OCLZeroEvent
  };
  
  /// \brief A single step in the initialization sequence.
  class Step {
  public:
    /// \brief The kind of conversion or initialization step we are taking.
    StepKind Kind;
    
    // \brief The type that results from this initialization.
    QualType Type;

    struct F {
      bool HadMultipleCandidates;
      FunctionDecl *Function;
      DeclAccessPair FoundDecl;
    };

    union {
      /// \brief When Kind == SK_ResolvedOverloadedFunction or Kind ==
      /// SK_UserConversion, the function that the expression should be 
      /// resolved to or the conversion function to call, respectively.
      /// When Kind == SK_ConstructorInitialization or SK_ListConstruction,
      /// the constructor to be called.
      ///
      /// Always a FunctionDecl, plus a Boolean flag telling if it was
      /// selected from an overloaded set having size greater than 1.
      /// For conversion decls, the naming class is the source type.
      /// For construct decls, the naming class is the target type.
      struct F Function;

      /// \brief When Kind = SK_ConversionSequence, the implicit conversion
      /// sequence.
      ImplicitConversionSequence *ICS;

      /// \brief When Kind = SK_RewrapInitList, the syntactic form of the
      /// wrapping list.
      InitListExpr *WrappingSyntacticList;
    };

    void Destroy();
  };
  
private:
  /// \brief The kind of initialization sequence computed.
  enum SequenceKind SequenceKind;
  
  /// \brief Steps taken by this initialization.
  SmallVector<Step, 4> Steps;
  
public:
  /// \brief Describes why initialization failed.
  enum FailureKind {
    /// \brief Too many initializers provided for a reference.
    FK_TooManyInitsForReference,
    /// \brief Array must be initialized with an initializer list.
    FK_ArrayNeedsInitList,
    /// \brief Array must be initialized with an initializer list or a 
    /// string literal.
    FK_ArrayNeedsInitListOrStringLiteral,
    /// \brief Array must be initialized with an initializer list or a
    /// wide string literal.
    FK_ArrayNeedsInitListOrWideStringLiteral,
    /// \brief Initializing a wide char array with narrow string literal.
    FK_NarrowStringIntoWideCharArray,
    /// \brief Initializing char array with wide string literal.
    FK_WideStringIntoCharArray,
    /// \brief Initializing wide char array with incompatible wide string
    /// literal.
    FK_IncompatWideStringIntoWideChar,
    /// \brief Array type mismatch.
    FK_ArrayTypeMismatch,
    /// \brief Non-constant array initializer
    FK_NonConstantArrayInit,
    /// \brief Cannot resolve the address of an overloaded function.
    FK_AddressOfOverloadFailed,
    /// \brief Overloading due to reference initialization failed.
    FK_ReferenceInitOverloadFailed,
    /// \brief Non-const lvalue reference binding to a temporary.
    FK_NonConstLValueReferenceBindingToTemporary,
    /// \brief Non-const lvalue reference binding to an lvalue of unrelated
    /// type.
    FK_NonConstLValueReferenceBindingToUnrelated,
    /// \brief Rvalue reference binding to an lvalue.
    FK_RValueReferenceBindingToLValue,
    /// \brief Reference binding drops qualifiers.
    FK_ReferenceInitDropsQualifiers,
    /// \brief Reference binding failed.
    FK_ReferenceInitFailed,
    /// \brief Implicit conversion failed.
    FK_ConversionFailed,
    /// \brief Implicit conversion failed.
    FK_ConversionFromPropertyFailed,
    /// \brief Too many initializers for scalar
    FK_TooManyInitsForScalar,
    /// \brief Reference initialization from an initializer list
    FK_ReferenceBindingToInitList,
    /// \brief Initialization of some unused destination type with an
    /// initializer list.
    FK_InitListBadDestinationType,
    /// \brief Overloading for a user-defined conversion failed.
    FK_UserConversionOverloadFailed,
    /// \brief Overloading for initialization by constructor failed.
    FK_ConstructorOverloadFailed,
    /// \brief Overloading for list-initialization by constructor failed.
    FK_ListConstructorOverloadFailed,
    /// \brief Default-initialization of a 'const' object.
    FK_DefaultInitOfConst,
    /// \brief Initialization of an incomplete type.
    FK_Incomplete,
    /// \brief Variable-length array must not have an initializer.
    FK_VariableLengthArrayHasInitializer,
    /// \brief List initialization failed at some point.
    FK_ListInitializationFailed,
    /// \brief Initializer has a placeholder type which cannot be
    /// resolved by initialization.
    FK_PlaceholderType,
    /// \brief List-copy-initialization chose an explicit constructor.
    FK_ExplicitConstructor
  };
  
private:
  /// \brief The reason why initialization failed.
  FailureKind Failure;

  /// \brief The failed result of overload resolution.
  OverloadingResult FailedOverloadResult;
  
  /// \brief The candidate set created when initialization failed.
  OverloadCandidateSet FailedCandidateSet;

  /// \brief The incomplete type that caused a failure.
  QualType FailedIncompleteType;
  
  /// \brief Prints a follow-up note that highlights the location of
  /// the initialized entity, if it's remote.
  void PrintInitLocationNote(Sema &S, const InitializedEntity &Entity);

public:
  /// \brief Try to perform initialization of the given entity, creating a 
  /// record of the steps required to perform the initialization.
  ///
  /// The generated initialization sequence will either contain enough
  /// information to diagnose 
  ///
  /// \param S the semantic analysis object.
  ///
  /// \param Entity the entity being initialized.
  ///
  /// \param Kind the kind of initialization being performed.
  ///
  /// \param Args the argument(s) provided for initialization.
  ///
  /// \param InInitList true if we are initializing from an expression within
  ///        an initializer list. This disallows narrowing conversions in C++11
  ///        onwards.
  InitializationSequence(Sema &S, 
                         const InitializedEntity &Entity,
                         const InitializationKind &Kind,
                         MultiExprArg Args,
                         bool InInitList = false);
  void InitializeFrom(Sema &S, const InitializedEntity &Entity,
                      const InitializationKind &Kind, MultiExprArg Args,
                      bool InInitList);

  ~InitializationSequence();
  
  /// \brief Perform the actual initialization of the given entity based on
  /// the computed initialization sequence.
  ///
  /// \param S the semantic analysis object.
  ///
  /// \param Entity the entity being initialized.
  ///
  /// \param Kind the kind of initialization being performed.
  ///
  /// \param Args the argument(s) provided for initialization, ownership of
  /// which is transferred into the routine.
  ///
  /// \param ResultType if non-NULL, will be set to the type of the
  /// initialized object, which is the type of the declaration in most
  /// cases. However, when the initialized object is a variable of
  /// incomplete array type and the initializer is an initializer
  /// list, this type will be set to the completed array type.
  ///
  /// \returns an expression that performs the actual object initialization, if
  /// the initialization is well-formed. Otherwise, emits diagnostics
  /// and returns an invalid expression.
  ExprResult Perform(Sema &S,
                     const InitializedEntity &Entity,
                     const InitializationKind &Kind,
                     MultiExprArg Args,
                     QualType *ResultType = 0);
  
  /// \brief Diagnose an potentially-invalid initialization sequence.
  ///
  /// \returns true if the initialization sequence was ill-formed, 
  /// false otherwise.
  bool Diagnose(Sema &S, 
                const InitializedEntity &Entity,
                const InitializationKind &Kind,
                ArrayRef<Expr *> Args);
  
  /// \brief Determine the kind of initialization sequence computed.
  enum SequenceKind getKind() const { return SequenceKind; }
  
  /// \brief Set the kind of sequence computed.
  void setSequenceKind(enum SequenceKind SK) { SequenceKind = SK; }
  
  /// \brief Determine whether the initialization sequence is valid.
  LLVM_EXPLICIT operator bool() const { return !Failed(); }

  /// \brief Determine whether the initialization sequence is invalid.
  bool Failed() const { return SequenceKind == FailedSequence; }

  typedef SmallVectorImpl<Step>::const_iterator step_iterator;
  step_iterator step_begin() const { return Steps.begin(); }
  step_iterator step_end()   const { return Steps.end(); }

  /// \brief Determine whether this initialization is a direct reference 
  /// binding (C++ [dcl.init.ref]).
  bool isDirectReferenceBinding() const;
  
  /// \brief Determine whether this initialization failed due to an ambiguity.
  bool isAmbiguous() const;
  
  /// \brief Determine whether this initialization is direct call to a 
  /// constructor.
  bool isConstructorInitialization() const;

  /// \brief Returns whether the last step in this initialization sequence is a
  /// narrowing conversion, defined by C++0x [dcl.init.list]p7.
  ///
  /// If this function returns true, *isInitializerConstant will be set to
  /// describe whether *Initializer was a constant expression.  If
  /// *isInitializerConstant is set to true, *ConstantValue will be set to the
  /// evaluated value of *Initializer.
  bool endsWithNarrowing(ASTContext &Ctx, const Expr *Initializer,
                         bool *isInitializerConstant,
                         APValue *ConstantValue) const;

  /// \brief Add a new step in the initialization that resolves the address
  /// of an overloaded function to a specific function declaration.
  ///
  /// \param Function the function to which the overloaded function reference
  /// resolves.
  void AddAddressOverloadResolutionStep(FunctionDecl *Function,
                                        DeclAccessPair Found,
                                        bool HadMultipleCandidates);

  /// \brief Add a new step in the initialization that performs a derived-to-
  /// base cast.
  ///
  /// \param BaseType the base type to which we will be casting.
  ///
  /// \param Category Indicates whether the result will be treated as an
  /// rvalue, an xvalue, or an lvalue.
  void AddDerivedToBaseCastStep(QualType BaseType,
                                ExprValueKind Category);
     
  /// \brief Add a new step binding a reference to an object.
  ///
  /// \param BindingTemporary True if we are binding a reference to a temporary
  /// object (thereby extending its lifetime); false if we are binding to an
  /// lvalue or an lvalue treated as an rvalue.
  void AddReferenceBindingStep(QualType T, bool BindingTemporary);

  /// \brief Add a new step that makes an extraneous copy of the input
  /// to a temporary of the same class type.
  ///
  /// This extraneous copy only occurs during reference binding in
  /// C++98/03, where we are permitted (but not required) to introduce
  /// an extra copy. At a bare minimum, we must check that we could
  /// call the copy constructor, and produce a diagnostic if the copy
  /// constructor is inaccessible or no copy constructor matches.
  //
  /// \param T The type of the temporary being created.
  void AddExtraneousCopyToTemporary(QualType T);

  /// \brief Add a new step invoking a conversion function, which is either
  /// a constructor or a conversion function.
  void AddUserConversionStep(FunctionDecl *Function,
                             DeclAccessPair FoundDecl,
                             QualType T,
                             bool HadMultipleCandidates);

  /// \brief Add a new step that performs a qualification conversion to the
  /// given type.
  void AddQualificationConversionStep(QualType Ty,
                                     ExprValueKind Category);
  
  /// \brief Add a new step that performs a load of the given type.
  ///
  /// Although the term "LValueToRValue" is conventional, this applies to both
  /// lvalues and xvalues.
  void AddLValueToRValueStep(QualType Ty);

  /// \brief Add a new step that applies an implicit conversion sequence.
  void AddConversionSequenceStep(const ImplicitConversionSequence &ICS,
                                 QualType T, bool TopLevelOfInitList = false);

  /// \brief Add a list-initialization step.
  void AddListInitializationStep(QualType T);

  /// \brief Add a constructor-initialization step.
  ///
  /// \param FromInitList The constructor call is syntactically an initializer
  /// list.
  /// \param AsInitList The constructor is called as an init list constructor.
  void AddConstructorInitializationStep(CXXConstructorDecl *Constructor,
                                        AccessSpecifier Access,
                                        QualType T,
                                        bool HadMultipleCandidates,
                                        bool FromInitList, bool AsInitList);

  /// \brief Add a zero-initialization step.
  void AddZeroInitializationStep(QualType T);

  /// \brief Add a C assignment step.
  //
  // FIXME: It isn't clear whether this should ever be needed;
  // ideally, we would handle everything needed in C in the common
  // path. However, that isn't the case yet.
  void AddCAssignmentStep(QualType T);

  /// \brief Add a string init step.
  void AddStringInitStep(QualType T);

  /// \brief Add an Objective-C object conversion step, which is
  /// always a no-op.
  void AddObjCObjectConversionStep(QualType T);

  /// \brief Add an array initialization step.
  void AddArrayInitStep(QualType T);

  /// \brief Add a parenthesized array initialization step.
  void AddParenthesizedArrayInitStep(QualType T);

  /// \brief Add a step to pass an object by indirect copy-restore.
  void AddPassByIndirectCopyRestoreStep(QualType T, bool shouldCopy);

  /// \brief Add a step to "produce" an Objective-C object (by
  /// retaining it).
  void AddProduceObjCObjectStep(QualType T);

  /// \brief Add a step to construct a std::initializer_list object from an
  /// initializer list.
  void AddStdInitializerListConstructionStep(QualType T);

  /// \brief Add a step to initialize an OpenCL sampler from an integer
  /// constant.
  void AddOCLSamplerInitStep(QualType T);

  /// \brief Add a step to initialize an OpenCL event_t from a NULL
  /// constant.
  void AddOCLZeroEventStep(QualType T);

  /// \brief Add steps to unwrap a initializer list for a reference around a
  /// single element and rewrap it at the end.
  void RewrapReferenceInitList(QualType T, InitListExpr *Syntactic);

  /// \brief Note that this initialization sequence failed.
  void SetFailed(FailureKind Failure) {
    SequenceKind = FailedSequence;
    this->Failure = Failure;
    assert((Failure != FK_Incomplete || !FailedIncompleteType.isNull()) &&
           "Incomplete type failure requires a type!");
  }
  
  /// \brief Note that this initialization sequence failed due to failed
  /// overload resolution.
  void SetOverloadFailure(FailureKind Failure, OverloadingResult Result);
  
  /// \brief Retrieve a reference to the candidate set when overload
  /// resolution fails.
  OverloadCandidateSet &getFailedCandidateSet() {
    return FailedCandidateSet;
  }

  /// \brief Get the overloading result, for when the initialization
  /// sequence failed due to a bad overload.
  OverloadingResult getFailedOverloadResult() const {
    return FailedOverloadResult;
  }

  /// \brief Note that this initialization sequence failed due to an
  /// incomplete type.
  void setIncompleteTypeFailure(QualType IncompleteType) {
    FailedIncompleteType = IncompleteType;
    SetFailed(FK_Incomplete);
  }

  /// \brief Determine why initialization failed.
  FailureKind getFailureKind() const {
    assert(Failed() && "Not an initialization failure!");
    return Failure;
  }

  /// \brief Dump a representation of this initialization sequence to 
  /// the given stream, for debugging purposes.
  void dump(raw_ostream &OS) const;
  
  /// \brief Dump a representation of this initialization sequence to 
  /// standard error, for debugging purposes.
  void dump() const;
};
  
} // end namespace clang

#endif // LLVM_CLANG_SEMA_INITIALIZATION_H
