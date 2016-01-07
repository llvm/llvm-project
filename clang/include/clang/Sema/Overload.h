//===--- Overload.h - C++ Overloading ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures and types used in C++
// overload resolution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_OVERLOAD_H
#define LLVM_CLANG_SEMA_OVERLOAD_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/UnresolvedSet.h"
#include "clang/Sema/SemaFixItUtils.h"
#include "clang/Sema/TemplateDeduction.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"

namespace clang {
  class ASTContext;
  class CXXConstructorDecl;
  class CXXConversionDecl;
  class FunctionDecl;
  class Sema;

  /// OverloadingResult - Capture the result of performing overload
  /// resolution.
  enum OverloadingResult {
    OR_Success,             ///< Overload resolution succeeded.
    OR_No_Viable_Function,  ///< No viable function found.
    OR_Ambiguous,           ///< Ambiguous candidates found.
    OR_Deleted              ///< Succeeded, but refers to a deleted function.
  };
  
  enum OverloadCandidateDisplayKind {
    /// Requests that all candidates be shown.  Viable candidates will
    /// be printed first.
    OCD_AllCandidates,

    /// Requests that only viable candidates be shown.
    OCD_ViableCandidates
  };

  /// ImplicitConversionKind - The kind of implicit conversion used to
  /// convert an argument to a parameter's type. The enumerator values
  /// match with Table 9 of (C++ 13.3.3.1.1) and are listed such that
  /// better conversion kinds have smaller values.
  enum ImplicitConversionKind {
    ICK_Identity = 0,          ///< Identity conversion (no conversion)
    ICK_Lvalue_To_Rvalue,      ///< Lvalue-to-rvalue conversion (C++ 4.1)
    ICK_Array_To_Pointer,      ///< Array-to-pointer conversion (C++ 4.2)
    ICK_Function_To_Pointer,   ///< Function-to-pointer (C++ 4.3)
    ICK_NoReturn_Adjustment,   ///< Removal of noreturn from a type (Clang)
    ICK_Qualification,         ///< Qualification conversions (C++ 4.4)
    ICK_Integral_Promotion,    ///< Integral promotions (C++ 4.5)
    ICK_Floating_Promotion,    ///< Floating point promotions (C++ 4.6)
    ICK_Complex_Promotion,     ///< Complex promotions (Clang extension)
    ICK_Integral_Conversion,   ///< Integral conversions (C++ 4.7)
    ICK_Floating_Conversion,   ///< Floating point conversions (C++ 4.8)
    ICK_Complex_Conversion,    ///< Complex conversions (C99 6.3.1.6)
    ICK_Floating_Integral,     ///< Floating-integral conversions (C++ 4.9)
    ICK_Pointer_Conversion,    ///< Pointer conversions (C++ 4.10)
    ICK_Pointer_Member,        ///< Pointer-to-member conversions (C++ 4.11)
    ICK_Boolean_Conversion,    ///< Boolean conversions (C++ 4.12)
    ICK_Compatible_Conversion, ///< Conversions between compatible types in C99
    ICK_Derived_To_Base,       ///< Derived-to-base (C++ [over.best.ics])
    ICK_Vector_Conversion,     ///< Vector conversions
    ICK_Vector_Splat,          ///< A vector splat from an arithmetic type
    ICK_Complex_Real,          ///< Complex-real conversions (C99 6.3.1.7)
    ICK_Block_Pointer_Conversion,    ///< Block Pointer conversions 
    ICK_TransparentUnionConversion, ///< Transparent Union Conversions
    ICK_Writeback_Conversion,  ///< Objective-C ARC writeback conversion
    ICK_Zero_Event_Conversion, ///< Zero constant to event (OpenCL1.2 6.12.10)
    ICK_C_Only_Conversion,     ///< Conversions allowed in C, but not C++
    ICK_Num_Conversion_Kinds,  ///< The number of conversion kinds
  };

  /// ImplicitConversionRank - The rank of an implicit conversion
  /// kind. The enumerator values match with Table 9 of (C++
  /// 13.3.3.1.1) and are listed such that better conversion ranks
  /// have smaller values.
  enum ImplicitConversionRank {
    ICR_Exact_Match = 0,         ///< Exact Match
    ICR_Promotion,               ///< Promotion
    ICR_Conversion,              ///< Conversion
    ICR_Complex_Real_Conversion, ///< Complex <-> Real conversion
    ICR_Writeback_Conversion,    ///< ObjC ARC writeback conversion
    ICR_C_Conversion             ///< Conversion only allowed in the C standard.
                                 ///  (e.g. void* to char*)
  };

  ImplicitConversionRank GetConversionRank(ImplicitConversionKind Kind);

  /// NarrowingKind - The kind of narrowing conversion being performed by a
  /// standard conversion sequence according to C++11 [dcl.init.list]p7.
  enum NarrowingKind {
    /// Not a narrowing conversion.
    NK_Not_Narrowing,

    /// A narrowing conversion by virtue of the source and destination types.
    NK_Type_Narrowing,

    /// A narrowing conversion, because a constant expression got narrowed.
    NK_Constant_Narrowing,

    /// A narrowing conversion, because a non-constant-expression variable might
    /// have got narrowed.
    NK_Variable_Narrowing
  };

  /// StandardConversionSequence - represents a standard conversion
  /// sequence (C++ 13.3.3.1.1). A standard conversion sequence
  /// contains between zero and three conversions. If a particular
  /// conversion is not needed, it will be set to the identity conversion
  /// (ICK_Identity). Note that the three conversions are
  /// specified as separate members (rather than in an array) so that
  /// we can keep the size of a standard conversion sequence to a
  /// single word.
  class StandardConversionSequence {
  public:
    /// First -- The first conversion can be an lvalue-to-rvalue
    /// conversion, array-to-pointer conversion, or
    /// function-to-pointer conversion.
    ImplicitConversionKind First : 8;

    /// Second - The second conversion can be an integral promotion,
    /// floating point promotion, integral conversion, floating point
    /// conversion, floating-integral conversion, pointer conversion,
    /// pointer-to-member conversion, or boolean conversion.
    ImplicitConversionKind Second : 8;

    /// Third - The third conversion can be a qualification conversion.
    ImplicitConversionKind Third : 8;

    /// \brief Whether this is the deprecated conversion of a
    /// string literal to a pointer to non-const character data
    /// (C++ 4.2p2).
    unsigned DeprecatedStringLiteralToCharPtr : 1;

    /// \brief Whether the qualification conversion involves a change in the
    /// Objective-C lifetime (for automatic reference counting).
    unsigned QualificationIncludesObjCLifetime : 1;
    
    /// IncompatibleObjC - Whether this is an Objective-C conversion
    /// that we should warn about (if we actually use it).
    unsigned IncompatibleObjC : 1;

    /// ReferenceBinding - True when this is a reference binding
    /// (C++ [over.ics.ref]).
    unsigned ReferenceBinding : 1;

    /// DirectBinding - True when this is a reference binding that is a
    /// direct binding (C++ [dcl.init.ref]).
    unsigned DirectBinding : 1;

    /// \brief Whether this is an lvalue reference binding (otherwise, it's
    /// an rvalue reference binding).
    unsigned IsLvalueReference : 1;
    
    /// \brief Whether we're binding to a function lvalue.
    unsigned BindsToFunctionLvalue : 1;
    
    /// \brief Whether we're binding to an rvalue.
    unsigned BindsToRvalue : 1;
    
    /// \brief Whether this binds an implicit object argument to a 
    /// non-static member function without a ref-qualifier.
    unsigned BindsImplicitObjectArgumentWithoutRefQualifier : 1;
    
    /// \brief Whether this binds a reference to an object with a different
    /// Objective-C lifetime qualifier.
    unsigned ObjCLifetimeConversionBinding : 1;
    
    /// FromType - The type that this conversion is converting
    /// from. This is an opaque pointer that can be translated into a
    /// QualType.
    void *FromTypePtr;

    /// ToType - The types that this conversion is converting to in
    /// each step. This is an opaque pointer that can be translated
    /// into a QualType.
    void *ToTypePtrs[3];

    /// CopyConstructor - The copy constructor that is used to perform
    /// this conversion, when the conversion is actually just the
    /// initialization of an object via copy constructor. Such
    /// conversions are either identity conversions or derived-to-base
    /// conversions.
    CXXConstructorDecl *CopyConstructor;

    void setFromType(QualType T) { FromTypePtr = T.getAsOpaquePtr(); }
    void setToType(unsigned Idx, QualType T) { 
      assert(Idx < 3 && "To type index is out of range");
      ToTypePtrs[Idx] = T.getAsOpaquePtr(); 
    }
    void setAllToTypes(QualType T) {
      ToTypePtrs[0] = T.getAsOpaquePtr(); 
      ToTypePtrs[1] = ToTypePtrs[0];
      ToTypePtrs[2] = ToTypePtrs[0];
    }

    QualType getFromType() const {
      return QualType::getFromOpaquePtr(FromTypePtr);
    }
    QualType getToType(unsigned Idx) const {
      assert(Idx < 3 && "To type index is out of range");
      return QualType::getFromOpaquePtr(ToTypePtrs[Idx]);
    }

    void setAsIdentityConversion();
    
    bool isIdentityConversion() const {
      return Second == ICK_Identity && Third == ICK_Identity;
    }
    
    ImplicitConversionRank getRank() const;
    NarrowingKind getNarrowingKind(ASTContext &Context, const Expr *Converted,
                                   APValue &ConstantValue,
                                   QualType &ConstantType) const;
    bool isPointerConversionToBool() const;
    bool isPointerConversionToVoidPointer(ASTContext& Context) const;
    void dump() const;
  };

  /// UserDefinedConversionSequence - Represents a user-defined
  /// conversion sequence (C++ 13.3.3.1.2).
  struct UserDefinedConversionSequence {
    /// \brief Represents the standard conversion that occurs before
    /// the actual user-defined conversion.
    ///
    /// C++11 13.3.3.1.2p1:
    ///   If the user-defined conversion is specified by a constructor
    ///   (12.3.1), the initial standard conversion sequence converts
    ///   the source type to the type required by the argument of the
    ///   constructor. If the user-defined conversion is specified by
    ///   a conversion function (12.3.2), the initial standard
    ///   conversion sequence converts the source type to the implicit
    ///   object parameter of the conversion function.
    StandardConversionSequence Before;

    /// EllipsisConversion - When this is true, it means user-defined
    /// conversion sequence starts with a ... (ellipsis) conversion, instead of
    /// a standard conversion. In this case, 'Before' field must be ignored.
    // FIXME. I much rather put this as the first field. But there seems to be
    // a gcc code gen. bug which causes a crash in a test. Putting it here seems
    // to work around the crash.
    bool EllipsisConversion : 1;

    /// HadMultipleCandidates - When this is true, it means that the
    /// conversion function was resolved from an overloaded set having
    /// size greater than 1.
    bool HadMultipleCandidates : 1;

    /// After - Represents the standard conversion that occurs after
    /// the actual user-defined conversion.
    StandardConversionSequence After;

    /// ConversionFunction - The function that will perform the
    /// user-defined conversion. Null if the conversion is an
    /// aggregate initialization from an initializer list.
    FunctionDecl* ConversionFunction;

    /// \brief The declaration that we found via name lookup, which might be
    /// the same as \c ConversionFunction or it might be a using declaration
    /// that refers to \c ConversionFunction.
    DeclAccessPair FoundConversionFunction;

    void dump() const;
  };

  /// Represents an ambiguous user-defined conversion sequence.
  struct AmbiguousConversionSequence {
    typedef SmallVector<FunctionDecl*, 4> ConversionSet;

    void *FromTypePtr;
    void *ToTypePtr;
    char Buffer[sizeof(ConversionSet)];

    QualType getFromType() const {
      return QualType::getFromOpaquePtr(FromTypePtr);
    }
    QualType getToType() const {
      return QualType::getFromOpaquePtr(ToTypePtr);
    }
    void setFromType(QualType T) { FromTypePtr = T.getAsOpaquePtr(); }
    void setToType(QualType T) { ToTypePtr = T.getAsOpaquePtr(); }

    ConversionSet &conversions() {
      return *reinterpret_cast<ConversionSet*>(Buffer);
    }

    const ConversionSet &conversions() const {
      return *reinterpret_cast<const ConversionSet*>(Buffer);
    }

    void addConversion(FunctionDecl *D) {
      conversions().push_back(D);
    }

    typedef ConversionSet::iterator iterator;
    iterator begin() { return conversions().begin(); }
    iterator end() { return conversions().end(); }

    typedef ConversionSet::const_iterator const_iterator;
    const_iterator begin() const { return conversions().begin(); }
    const_iterator end() const { return conversions().end(); }

    void construct();
    void destruct();
    void copyFrom(const AmbiguousConversionSequence &);
  };

  /// BadConversionSequence - Records information about an invalid
  /// conversion sequence.
  struct BadConversionSequence {
    enum FailureKind {
      no_conversion,
      unrelated_class,
      bad_qualifiers,
      lvalue_ref_to_rvalue,
      rvalue_ref_to_lvalue
    };

    // This can be null, e.g. for implicit object arguments.
    Expr *FromExpr;

    FailureKind Kind;

  private:
    // The type we're converting from (an opaque QualType).
    void *FromTy;

    // The type we're converting to (an opaque QualType).
    void *ToTy;

  public:
    void init(FailureKind K, Expr *From, QualType To) {
      init(K, From->getType(), To);
      FromExpr = From;
    }
    void init(FailureKind K, QualType From, QualType To) {
      Kind = K;
      FromExpr = nullptr;
      setFromType(From);
      setToType(To);
    }

    QualType getFromType() const { return QualType::getFromOpaquePtr(FromTy); }
    QualType getToType() const { return QualType::getFromOpaquePtr(ToTy); }

    void setFromExpr(Expr *E) {
      FromExpr = E;
      setFromType(E->getType());
    }
    void setFromType(QualType T) { FromTy = T.getAsOpaquePtr(); }
    void setToType(QualType T) { ToTy = T.getAsOpaquePtr(); }
  };

  /// ImplicitConversionSequence - Represents an implicit conversion
  /// sequence, which may be a standard conversion sequence
  /// (C++ 13.3.3.1.1), user-defined conversion sequence (C++ 13.3.3.1.2),
  /// or an ellipsis conversion sequence (C++ 13.3.3.1.3).
  class ImplicitConversionSequence {
  public:
    /// Kind - The kind of implicit conversion sequence. BadConversion
    /// specifies that there is no conversion from the source type to
    /// the target type.  AmbiguousConversion represents the unique
    /// ambiguous conversion (C++0x [over.best.ics]p10).
    enum Kind {
      StandardConversion = 0,
      UserDefinedConversion,
      AmbiguousConversion,
      EllipsisConversion,
      BadConversion
    };

  private:
    enum {
      Uninitialized = BadConversion + 1
    };

    /// ConversionKind - The kind of implicit conversion sequence.
    unsigned ConversionKind : 30;

    /// \brief Whether the target is really a std::initializer_list, and the
    /// sequence only represents the worst element conversion.
    bool StdInitializerListElement : 1;

    void setKind(Kind K) {
      destruct();
      ConversionKind = K;
    }

    void destruct() {
      if (ConversionKind == AmbiguousConversion) Ambiguous.destruct();
    }

  public:
    union {
      /// When ConversionKind == StandardConversion, provides the
      /// details of the standard conversion sequence.
      StandardConversionSequence Standard;

      /// When ConversionKind == UserDefinedConversion, provides the
      /// details of the user-defined conversion sequence.
      UserDefinedConversionSequence UserDefined;

      /// When ConversionKind == AmbiguousConversion, provides the
      /// details of the ambiguous conversion.
      AmbiguousConversionSequence Ambiguous;

      /// When ConversionKind == BadConversion, provides the details
      /// of the bad conversion.
      BadConversionSequence Bad;
    };

    ImplicitConversionSequence()
      : ConversionKind(Uninitialized), StdInitializerListElement(false)
    {}
    ~ImplicitConversionSequence() {
      destruct();
    }
    ImplicitConversionSequence(const ImplicitConversionSequence &Other)
      : ConversionKind(Other.ConversionKind),
        StdInitializerListElement(Other.StdInitializerListElement)
    {
      switch (ConversionKind) {
      case Uninitialized: break;
      case StandardConversion: Standard = Other.Standard; break;
      case UserDefinedConversion: UserDefined = Other.UserDefined; break;
      case AmbiguousConversion: Ambiguous.copyFrom(Other.Ambiguous); break;
      case EllipsisConversion: break;
      case BadConversion: Bad = Other.Bad; break;
      }
    }

    ImplicitConversionSequence &
        operator=(const ImplicitConversionSequence &Other) {
      destruct();
      new (this) ImplicitConversionSequence(Other);
      return *this;
    }
    
    Kind getKind() const {
      assert(isInitialized() && "querying uninitialized conversion");
      return Kind(ConversionKind);
    }
    
    /// \brief Return a ranking of the implicit conversion sequence
    /// kind, where smaller ranks represent better conversion
    /// sequences.
    ///
    /// In particular, this routine gives user-defined conversion
    /// sequences and ambiguous conversion sequences the same rank,
    /// per C++ [over.best.ics]p10.
    unsigned getKindRank() const {
      switch (getKind()) {
      case StandardConversion: 
        return 0;

      case UserDefinedConversion:
      case AmbiguousConversion: 
        return 1;

      case EllipsisConversion:
        return 2;

      case BadConversion:
        return 3;
      }

      llvm_unreachable("Invalid ImplicitConversionSequence::Kind!");
    }

    bool isBad() const { return getKind() == BadConversion; }
    bool isStandard() const { return getKind() == StandardConversion; }
    bool isEllipsis() const { return getKind() == EllipsisConversion; }
    bool isAmbiguous() const { return getKind() == AmbiguousConversion; }
    bool isUserDefined() const { return getKind() == UserDefinedConversion; }
    bool isFailure() const { return isBad() || isAmbiguous(); }

    /// Determines whether this conversion sequence has been
    /// initialized.  Most operations should never need to query
    /// uninitialized conversions and should assert as above.
    bool isInitialized() const { return ConversionKind != Uninitialized; }

    /// Sets this sequence as a bad conversion for an explicit argument.
    void setBad(BadConversionSequence::FailureKind Failure,
                Expr *FromExpr, QualType ToType) {
      setKind(BadConversion);
      Bad.init(Failure, FromExpr, ToType);
    }

    /// Sets this sequence as a bad conversion for an implicit argument.
    void setBad(BadConversionSequence::FailureKind Failure,
                QualType FromType, QualType ToType) {
      setKind(BadConversion);
      Bad.init(Failure, FromType, ToType);
    }

    void setStandard() { setKind(StandardConversion); }
    void setEllipsis() { setKind(EllipsisConversion); }
    void setUserDefined() { setKind(UserDefinedConversion); }
    void setAmbiguous() {
      if (ConversionKind == AmbiguousConversion) return;
      ConversionKind = AmbiguousConversion;
      Ambiguous.construct();
    }

    /// \brief Whether the target is really a std::initializer_list, and the
    /// sequence only represents the worst element conversion.
    bool isStdInitializerListElement() const {
      return StdInitializerListElement;
    }

    void setStdInitializerListElement(bool V = true) {
      StdInitializerListElement = V;
    }

    // The result of a comparison between implicit conversion
    // sequences. Use Sema::CompareImplicitConversionSequences to
    // actually perform the comparison.
    enum CompareKind {
      Better = -1,
      Indistinguishable = 0,
      Worse = 1
    };

    void DiagnoseAmbiguousConversion(Sema &S,
                                     SourceLocation CaretLoc,
                                     const PartialDiagnostic &PDiag) const;

    void dump() const;
  };

  enum OverloadFailureKind {
    ovl_fail_too_many_arguments,
    ovl_fail_too_few_arguments,
    ovl_fail_bad_conversion,
    ovl_fail_bad_deduction,

    /// This conversion candidate was not considered because it
    /// duplicates the work of a trivial or derived-to-base
    /// conversion.
    ovl_fail_trivial_conversion,

    /// This conversion candidate was not considered because it is
    /// an illegal instantiation of a constructor temploid: it is
    /// callable with one argument, we only have one argument, and
    /// its first parameter type is exactly the type of the class.
    ///
    /// Defining such a constructor directly is illegal, and
    /// template-argument deduction is supposed to ignore such
    /// instantiations, but we can still get one with the right
    /// kind of implicit instantiation.
    ovl_fail_illegal_constructor,

    /// This conversion candidate is not viable because its result
    /// type is not implicitly convertible to the desired type.
    ovl_fail_bad_final_conversion,

    /// This conversion function template specialization candidate is not
    /// viable because the final conversion was not an exact match.
    ovl_fail_final_conversion_not_exact,

    /// (CUDA) This candidate was not viable because the callee
    /// was not accessible from the caller's target (i.e. host->device,
    /// global->host, device->host).
    ovl_fail_bad_target,

    /// This candidate function was not viable because an enable_if
    /// attribute disabled it.
    ovl_fail_enable_if,

    /// This candidate was not viable because its address could not be taken.
    ovl_fail_addr_not_available
  };

  /// OverloadCandidate - A single candidate in an overload set (C++ 13.3).
  struct OverloadCandidate {
    /// Function - The actual function that this candidate
    /// represents. When NULL, this is a built-in candidate
    /// (C++ [over.oper]) or a surrogate for a conversion to a
    /// function pointer or reference (C++ [over.call.object]).
    FunctionDecl *Function;

    /// FoundDecl - The original declaration that was looked up /
    /// invented / otherwise found, together with its access.
    /// Might be a UsingShadowDecl or a FunctionTemplateDecl.
    DeclAccessPair FoundDecl;

    // BuiltinTypes - Provides the return and parameter types of a
    // built-in overload candidate. Only valid when Function is NULL.
    struct {
      QualType ResultTy;
      QualType ParamTypes[3];
    } BuiltinTypes;

    /// Surrogate - The conversion function for which this candidate
    /// is a surrogate, but only if IsSurrogate is true.
    CXXConversionDecl *Surrogate;

    /// Conversions - The conversion sequences used to convert the
    /// function arguments to the function parameters, the pointer points to a
    /// fixed size array with NumConversions elements. The memory is owned by
    /// the OverloadCandidateSet.
    ImplicitConversionSequence *Conversions;

    /// The FixIt hints which can be used to fix the Bad candidate.
    ConversionFixItGenerator Fix;

    /// NumConversions - The number of elements in the Conversions array.
    unsigned NumConversions;

    /// Viable - True to indicate that this overload candidate is viable.
    bool Viable;

    /// IsSurrogate - True to indicate that this candidate is a
    /// surrogate for a conversion to a function pointer or reference
    /// (C++ [over.call.object]).
    bool IsSurrogate;

    /// IgnoreObjectArgument - True to indicate that the first
    /// argument's conversion, which for this function represents the
    /// implicit object argument, should be ignored. This will be true
    /// when the candidate is a static member function (where the
    /// implicit object argument is just a placeholder) or a
    /// non-static member function when the call doesn't have an
    /// object argument.
    bool IgnoreObjectArgument;

    /// FailureKind - The reason why this candidate is not viable.
    /// Actually an OverloadFailureKind.
    unsigned char FailureKind;

    /// \brief The number of call arguments that were explicitly provided,
    /// to be used while performing partial ordering of function templates.
    unsigned ExplicitCallArguments;

    union {
      DeductionFailureInfo DeductionFailure;
      
      /// FinalConversion - For a conversion function (where Function is
      /// a CXXConversionDecl), the standard conversion that occurs
      /// after the call to the overload candidate to convert the result
      /// of calling the conversion function to the required type.
      StandardConversionSequence FinalConversion;
    };

    /// hasAmbiguousConversion - Returns whether this overload
    /// candidate requires an ambiguous conversion or not.
    bool hasAmbiguousConversion() const {
      for (unsigned i = 0, e = NumConversions; i != e; ++i) {
        if (!Conversions[i].isInitialized()) return false;
        if (Conversions[i].isAmbiguous()) return true;
      }
      return false;
    }

    bool TryToFixBadConversion(unsigned Idx, Sema &S) {
      bool CanFix = Fix.tryToFixConversion(
                      Conversions[Idx].Bad.FromExpr,
                      Conversions[Idx].Bad.getFromType(),
                      Conversions[Idx].Bad.getToType(), S);

      // If at least one conversion fails, the candidate cannot be fixed.
      if (!CanFix)
        Fix.clear();

      return CanFix;
    }

    unsigned getNumParams() const {
      if (IsSurrogate) {
        auto STy = Surrogate->getConversionType();
        while (STy->isPointerType() || STy->isReferenceType())
          STy = STy->getPointeeType();
        return STy->getAs<FunctionProtoType>()->getNumParams();
      }
      if (Function)
        return Function->getNumParams();
      return ExplicitCallArguments;
    }
  };

  /// OverloadCandidateSet - A set of overload candidates, used in C++
  /// overload resolution (C++ 13.3).
  class OverloadCandidateSet {
  public:
    enum CandidateSetKind {
      /// Normal lookup.
      CSK_Normal,
      /// Lookup for candidates for a call using operator syntax. Candidates
      /// that have no parameters of class type will be skipped unless there
      /// is a parameter of (reference to) enum type and the corresponding
      /// argument is of the same enum type.
      CSK_Operator
    };

  private:
    SmallVector<OverloadCandidate, 16> Candidates;
    llvm::SmallPtrSet<Decl *, 16> Functions;

    // Allocator for OverloadCandidate::Conversions. We store the first few
    // elements inline to avoid allocation for small sets.
    llvm::BumpPtrAllocator ConversionSequenceAllocator;

    SourceLocation Loc;
    CandidateSetKind Kind;

    unsigned NumInlineSequences;
    llvm::AlignedCharArray<llvm::AlignOf<ImplicitConversionSequence>::Alignment,
                           16 * sizeof(ImplicitConversionSequence)> InlineSpace;

    OverloadCandidateSet(const OverloadCandidateSet &) = delete;
    void operator=(const OverloadCandidateSet &) = delete;

    void destroyCandidates();

  public:
    OverloadCandidateSet(SourceLocation Loc, CandidateSetKind CSK)
        : Loc(Loc), Kind(CSK), NumInlineSequences(0) {}
    ~OverloadCandidateSet() { destroyCandidates(); }

    SourceLocation getLocation() const { return Loc; }
    CandidateSetKind getKind() const { return Kind; }

    /// \brief Determine when this overload candidate will be new to the
    /// overload set.
    bool isNewCandidate(Decl *F) {
      return Functions.insert(F->getCanonicalDecl()).second;
    }

    /// \brief Clear out all of the candidates.
    void clear();

    typedef SmallVectorImpl<OverloadCandidate>::iterator iterator;
    iterator begin() { return Candidates.begin(); }
    iterator end() { return Candidates.end(); }

    size_t size() const { return Candidates.size(); }
    bool empty() const { return Candidates.empty(); }

    /// \brief Add a new candidate with NumConversions conversion sequence slots
    /// to the overload set.
    OverloadCandidate &addCandidate(unsigned NumConversions = 0) {
      Candidates.push_back(OverloadCandidate());
      OverloadCandidate &C = Candidates.back();

      // Assign space from the inline array if there are enough free slots
      // available.
      if (NumConversions + NumInlineSequences <= 16) {
        ImplicitConversionSequence *I =
            (ImplicitConversionSequence *)InlineSpace.buffer;
        C.Conversions = &I[NumInlineSequences];
        NumInlineSequences += NumConversions;
      } else {
        // Otherwise get memory from the allocator.
        C.Conversions = ConversionSequenceAllocator
                          .Allocate<ImplicitConversionSequence>(NumConversions);
      }

      // Construct the new objects.
      for (unsigned i = 0; i != NumConversions; ++i)
        new (&C.Conversions[i]) ImplicitConversionSequence();

      C.NumConversions = NumConversions;
      return C;
    }

    /// Find the best viable function on this overload set, if it exists.
    OverloadingResult BestViableFunction(Sema &S, SourceLocation Loc,
                                         OverloadCandidateSet::iterator& Best,
                                         bool UserDefinedConversion = false);

    void NoteCandidates(Sema &S,
                        OverloadCandidateDisplayKind OCD,
                        ArrayRef<Expr *> Args,
                        StringRef Opc = "",
                        SourceLocation Loc = SourceLocation());
  };

  bool isBetterOverloadCandidate(Sema &S,
                                 const OverloadCandidate& Cand1,
                                 const OverloadCandidate& Cand2,
                                 SourceLocation Loc,
                                 bool UserDefinedConversion = false);
} // end namespace clang

#endif // LLVM_CLANG_SEMA_OVERLOAD_H
