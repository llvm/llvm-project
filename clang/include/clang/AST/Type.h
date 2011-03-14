//===--- Type.h - C Language Family Type Representation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Type interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPE_H
#define LLVM_CLANG_AST_TYPE_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/Visibility.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateName.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/type_traits.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"

using llvm::isa;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
namespace clang {
  enum {
    TypeAlignmentInBits = 4,
    TypeAlignment = 1 << TypeAlignmentInBits
  };
  class Type;
  class ExtQuals;
  class QualType;
}

namespace llvm {
  template <typename T>
  class PointerLikeTypeTraits;
  template<>
  class PointerLikeTypeTraits< ::clang::Type*> {
  public:
    static inline void *getAsVoidPointer(::clang::Type *P) { return P; }
    static inline ::clang::Type *getFromVoidPointer(void *P) {
      return static_cast< ::clang::Type*>(P);
    }
    enum { NumLowBitsAvailable = clang::TypeAlignmentInBits };
  };
  template<>
  class PointerLikeTypeTraits< ::clang::ExtQuals*> {
  public:
    static inline void *getAsVoidPointer(::clang::ExtQuals *P) { return P; }
    static inline ::clang::ExtQuals *getFromVoidPointer(void *P) {
      return static_cast< ::clang::ExtQuals*>(P);
    }
    enum { NumLowBitsAvailable = clang::TypeAlignmentInBits };
  };

  template <>
  struct isPodLike<clang::QualType> { static const bool value = true; };
}

namespace clang {
  class ASTContext;
  class TypedefDecl;
  class TemplateDecl;
  class TemplateTypeParmDecl;
  class NonTypeTemplateParmDecl;
  class TemplateTemplateParmDecl;
  class TagDecl;
  class RecordDecl;
  class CXXRecordDecl;
  class EnumDecl;
  class FieldDecl;
  class ObjCInterfaceDecl;
  class ObjCProtocolDecl;
  class ObjCMethodDecl;
  class UnresolvedUsingTypenameDecl;
  class Expr;
  class Stmt;
  class SourceLocation;
  class StmtIteratorBase;
  class TemplateArgument;
  class TemplateArgumentLoc;
  class TemplateArgumentListInfo;
  class ElaboratedType;
  class ExtQuals;
  class ExtQualsTypeCommonBase;
  struct PrintingPolicy;

  template <typename> class CanQual;  
  typedef CanQual<Type> CanQualType;

  // Provide forward declarations for all of the *Type classes
#define TYPE(Class, Base) class Class##Type;
#include "clang/AST/TypeNodes.def"

/// Qualifiers - The collection of all-type qualifiers we support.
/// Clang supports five independent qualifiers:
/// * C99: const, volatile, and restrict
/// * Embedded C (TR18037): address spaces
/// * Objective C: the GC attributes (none, weak, or strong)
class Qualifiers {
public:
  enum TQ { // NOTE: These flags must be kept in sync with DeclSpec::TQ.
    Const    = 0x1,
    Restrict = 0x2,
    Volatile = 0x4,
    CVRMask = Const | Volatile | Restrict
  };

  enum GC {
    GCNone = 0,
    Weak,
    Strong
  };

  enum {
    /// The maximum supported address space number.
    /// 24 bits should be enough for anyone.
    MaxAddressSpace = 0xffffffu,

    /// The width of the "fast" qualifier mask.
    FastWidth = 3,

    /// The fast qualifier mask.
    FastMask = (1 << FastWidth) - 1
  };

  Qualifiers() : Mask(0) {}

  static Qualifiers fromFastMask(unsigned Mask) {
    Qualifiers Qs;
    Qs.addFastQualifiers(Mask);
    return Qs;
  }

  static Qualifiers fromCVRMask(unsigned CVR) {
    Qualifiers Qs;
    Qs.addCVRQualifiers(CVR);
    return Qs;
  }

  // Deserialize qualifiers from an opaque representation.
  static Qualifiers fromOpaqueValue(unsigned opaque) {
    Qualifiers Qs;
    Qs.Mask = opaque;
    return Qs;
  }

  // Serialize these qualifiers into an opaque representation.
  unsigned getAsOpaqueValue() const {
    return Mask;
  }

  bool hasConst() const { return Mask & Const; }
  void setConst(bool flag) {
    Mask = (Mask & ~Const) | (flag ? Const : 0);
  }
  void removeConst() { Mask &= ~Const; }
  void addConst() { Mask |= Const; }

  bool hasVolatile() const { return Mask & Volatile; }
  void setVolatile(bool flag) {
    Mask = (Mask & ~Volatile) | (flag ? Volatile : 0);
  }
  void removeVolatile() { Mask &= ~Volatile; }
  void addVolatile() { Mask |= Volatile; }

  bool hasRestrict() const { return Mask & Restrict; }
  void setRestrict(bool flag) {
    Mask = (Mask & ~Restrict) | (flag ? Restrict : 0);
  }
  void removeRestrict() { Mask &= ~Restrict; }
  void addRestrict() { Mask |= Restrict; }

  bool hasCVRQualifiers() const { return getCVRQualifiers(); }
  unsigned getCVRQualifiers() const { return Mask & CVRMask; }
  void setCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask = (Mask & ~CVRMask) | mask;
  }
  void removeCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask &= ~mask;
  }
  void removeCVRQualifiers() {
    removeCVRQualifiers(CVRMask);
  }
  void addCVRQualifiers(unsigned mask) {
    assert(!(mask & ~CVRMask) && "bitmask contains non-CVR bits");
    Mask |= mask;
  }

  bool hasObjCGCAttr() const { return Mask & GCAttrMask; }
  GC getObjCGCAttr() const { return GC((Mask & GCAttrMask) >> GCAttrShift); }
  void setObjCGCAttr(GC type) {
    Mask = (Mask & ~GCAttrMask) | (type << GCAttrShift);
  }
  void removeObjCGCAttr() { setObjCGCAttr(GCNone); }
  void addObjCGCAttr(GC type) {
    assert(type);
    setObjCGCAttr(type);
  }

  bool hasAddressSpace() const { return Mask & AddressSpaceMask; }
  unsigned getAddressSpace() const { return Mask >> AddressSpaceShift; }
  void setAddressSpace(unsigned space) {
    assert(space <= MaxAddressSpace);
    Mask = (Mask & ~AddressSpaceMask)
         | (((uint32_t) space) << AddressSpaceShift);
  }
  void removeAddressSpace() { setAddressSpace(0); }
  void addAddressSpace(unsigned space) {
    assert(space);
    setAddressSpace(space);
  }

  // Fast qualifiers are those that can be allocated directly
  // on a QualType object.
  bool hasFastQualifiers() const { return getFastQualifiers(); }
  unsigned getFastQualifiers() const { return Mask & FastMask; }
  void setFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask = (Mask & ~FastMask) | mask;
  }
  void removeFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask &= ~mask;
  }
  void removeFastQualifiers() {
    removeFastQualifiers(FastMask);
  }
  void addFastQualifiers(unsigned mask) {
    assert(!(mask & ~FastMask) && "bitmask contains non-fast qualifier bits");
    Mask |= mask;
  }

  /// hasNonFastQualifiers - Return true if the set contains any
  /// qualifiers which require an ExtQuals node to be allocated.
  bool hasNonFastQualifiers() const { return Mask & ~FastMask; }
  Qualifiers getNonFastQualifiers() const {
    Qualifiers Quals = *this;
    Quals.setFastQualifiers(0);
    return Quals;
  }

  /// hasQualifiers - Return true if the set contains any qualifiers.
  bool hasQualifiers() const { return Mask; }
  bool empty() const { return !Mask; }

  /// \brief Add the qualifiers from the given set to this set.
  void addQualifiers(Qualifiers Q) {
    // If the other set doesn't have any non-boolean qualifiers, just
    // bit-or it in.
    if (!(Q.Mask & ~CVRMask))
      Mask |= Q.Mask;
    else {
      Mask |= (Q.Mask & CVRMask);
      if (Q.hasAddressSpace())
        addAddressSpace(Q.getAddressSpace());
      if (Q.hasObjCGCAttr())
        addObjCGCAttr(Q.getObjCGCAttr());
    }
  }

  /// \brief Add the qualifiers from the given set to this set, given that
  /// they don't conflict.
  void addConsistentQualifiers(Qualifiers qs) {
    assert(getAddressSpace() == qs.getAddressSpace() ||
           !hasAddressSpace() || !qs.hasAddressSpace());
    assert(getObjCGCAttr() == qs.getObjCGCAttr() ||
           !hasObjCGCAttr() || !qs.hasObjCGCAttr());
    Mask |= qs.Mask;
  }

  /// \brief Determines if these qualifiers compatibly include another set.
  /// Generally this answers the question of whether an object with the other
  /// qualifiers can be safely used as an object with these qualifiers.
  bool compatiblyIncludes(Qualifiers other) const {
    // Non-CVR qualifiers must match exactly.  CVR qualifiers may subset.
    return ((Mask & ~CVRMask) == (other.Mask & ~CVRMask)) &&
           (((Mask & CVRMask) | (other.Mask & CVRMask)) == (Mask & CVRMask));
  }

  bool isSupersetOf(Qualifiers Other) const;

  bool operator==(Qualifiers Other) const { return Mask == Other.Mask; }
  bool operator!=(Qualifiers Other) const { return Mask != Other.Mask; }

  operator bool() const { return hasQualifiers(); }

  Qualifiers &operator+=(Qualifiers R) {
    addQualifiers(R);
    return *this;
  }

  // Union two qualifier sets.  If an enumerated qualifier appears
  // in both sets, use the one from the right.
  friend Qualifiers operator+(Qualifiers L, Qualifiers R) {
    L += R;
    return L;
  }
  
  Qualifiers &operator-=(Qualifiers R) {
    Mask = Mask & ~(R.Mask);
    return *this;
  }

  /// \brief Compute the difference between two qualifier sets.
  friend Qualifiers operator-(Qualifiers L, Qualifiers R) {
    L -= R;
    return L;
  }
  
  std::string getAsString() const;
  std::string getAsString(const PrintingPolicy &Policy) const {
    std::string Buffer;
    getAsStringInternal(Buffer, Policy);
    return Buffer;
  }
  void getAsStringInternal(std::string &S, const PrintingPolicy &Policy) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(Mask);
  }

private:

  // bits:     |0 1 2|3 .. 4|5  ..  31|
  //           |C R V|GCAttr|AddrSpace|
  uint32_t Mask;

  static const uint32_t GCAttrMask = 0x18;
  static const uint32_t GCAttrShift = 3;
  static const uint32_t AddressSpaceMask = ~(CVRMask | GCAttrMask);
  static const uint32_t AddressSpaceShift = 5;
};

/// CallingConv - Specifies the calling convention that a function uses.
enum CallingConv {
  CC_Default,
  CC_C,           // __attribute__((cdecl))
  CC_X86StdCall,  // __attribute__((stdcall))
  CC_X86FastCall, // __attribute__((fastcall))
  CC_X86ThisCall, // __attribute__((thiscall))
  CC_X86Pascal    // __attribute__((pascal))
};

typedef std::pair<const Type*, Qualifiers> SplitQualType;

/// QualType - For efficiency, we don't store CV-qualified types as nodes on
/// their own: instead each reference to a type stores the qualifiers.  This
/// greatly reduces the number of nodes we need to allocate for types (for
/// example we only need one for 'int', 'const int', 'volatile int',
/// 'const volatile int', etc).
///
/// As an added efficiency bonus, instead of making this a pair, we
/// just store the two bits we care about in the low bits of the
/// pointer.  To handle the packing/unpacking, we make QualType be a
/// simple wrapper class that acts like a smart pointer.  A third bit
/// indicates whether there are extended qualifiers present, in which
/// case the pointer points to a special structure.
class QualType {
  // Thankfully, these are efficiently composable.
  llvm::PointerIntPair<llvm::PointerUnion<const Type*,const ExtQuals*>,
                       Qualifiers::FastWidth> Value;

  const ExtQuals *getExtQualsUnsafe() const {
    return Value.getPointer().get<const ExtQuals*>();
  }

  const Type *getTypePtrUnsafe() const {
    return Value.getPointer().get<const Type*>();
  }

  const ExtQualsTypeCommonBase *getCommonPtr() const {
    assert(!isNull() && "Cannot retrieve a NULL type pointer");
    uintptr_t CommonPtrVal
      = reinterpret_cast<uintptr_t>(Value.getOpaqueValue());
    CommonPtrVal &= ~(uintptr_t)((1 << TypeAlignmentInBits) - 1);
    return reinterpret_cast<ExtQualsTypeCommonBase*>(CommonPtrVal);
  }

  friend class QualifierCollector;
public:
  QualType() {}

  QualType(const Type *Ptr, unsigned Quals)
    : Value(Ptr, Quals) {}
  QualType(const ExtQuals *Ptr, unsigned Quals)
    : Value(Ptr, Quals) {}

  unsigned getLocalFastQualifiers() const { return Value.getInt(); }
  void setLocalFastQualifiers(unsigned Quals) { Value.setInt(Quals); }

  /// Retrieves a pointer to the underlying (unqualified) type.
  /// This should really return a const Type, but it's not worth
  /// changing all the users right now.
  ///
  /// This function requires that the type not be NULL. If the type might be
  /// NULL, use the (slightly less efficient) \c getTypePtrOrNull().
  const Type *getTypePtr() const;
  
  const Type *getTypePtrOrNull() const;

  /// Divides a QualType into its unqualified type and a set of local
  /// qualifiers.
  SplitQualType split() const;

  void *getAsOpaquePtr() const { return Value.getOpaqueValue(); }
  static QualType getFromOpaquePtr(const void *Ptr) {
    QualType T;
    T.Value.setFromOpaqueValue(const_cast<void*>(Ptr));
    return T;
  }

  const Type &operator*() const {
    return *getTypePtr();
  }

  const Type *operator->() const {
    return getTypePtr();
  }

  bool isCanonical() const;
  bool isCanonicalAsParam() const;

  /// isNull - Return true if this QualType doesn't point to a type yet.
  bool isNull() const {
    return Value.getPointer().isNull();
  }

  /// \brief Determine whether this particular QualType instance has the 
  /// "const" qualifier set, without looking through typedefs that may have
  /// added "const" at a different level.
  bool isLocalConstQualified() const {
    return (getLocalFastQualifiers() & Qualifiers::Const);
  }
  
  /// \brief Determine whether this type is const-qualified.
  bool isConstQualified() const;
  
  /// \brief Determine whether this particular QualType instance has the 
  /// "restrict" qualifier set, without looking through typedefs that may have
  /// added "restrict" at a different level.
  bool isLocalRestrictQualified() const {
    return (getLocalFastQualifiers() & Qualifiers::Restrict);
  }
  
  /// \brief Determine whether this type is restrict-qualified.
  bool isRestrictQualified() const;
  
  /// \brief Determine whether this particular QualType instance has the 
  /// "volatile" qualifier set, without looking through typedefs that may have
  /// added "volatile" at a different level.
  bool isLocalVolatileQualified() const {
    return (getLocalFastQualifiers() & Qualifiers::Volatile);
  }

  /// \brief Determine whether this type is volatile-qualified.
  bool isVolatileQualified() const;
  
  /// \brief Determine whether this particular QualType instance has any
  /// qualifiers, without looking through any typedefs that might add 
  /// qualifiers at a different level.
  bool hasLocalQualifiers() const {
    return getLocalFastQualifiers() || hasLocalNonFastQualifiers();
  }

  /// \brief Determine whether this type has any qualifiers.
  bool hasQualifiers() const;
  
  /// \brief Determine whether this particular QualType instance has any
  /// "non-fast" qualifiers, e.g., those that are stored in an ExtQualType
  /// instance.
  bool hasLocalNonFastQualifiers() const {
    return Value.getPointer().is<const ExtQuals*>();
  }

  /// \brief Retrieve the set of qualifiers local to this particular QualType
  /// instance, not including any qualifiers acquired through typedefs or
  /// other sugar.
  Qualifiers getLocalQualifiers() const;

  /// \brief Retrieve the set of qualifiers applied to this type.
  Qualifiers getQualifiers() const;
  
  /// \brief Retrieve the set of CVR (const-volatile-restrict) qualifiers 
  /// local to this particular QualType instance, not including any qualifiers
  /// acquired through typedefs or other sugar.
  unsigned getLocalCVRQualifiers() const {
    return getLocalFastQualifiers();
  }

  /// \brief Retrieve the set of CVR (const-volatile-restrict) qualifiers 
  /// applied to this type.
  unsigned getCVRQualifiers() const;

  bool isConstant(ASTContext& Ctx) const {
    return QualType::isConstant(*this, Ctx);
  }

  // Don't promise in the API that anything besides 'const' can be
  // easily added.

  /// addConst - add the specified type qualifier to this QualType.  
  void addConst() {
    addFastQualifiers(Qualifiers::Const);
  }
  QualType withConst() const {
    return withFastQualifiers(Qualifiers::Const);
  }

  void addFastQualifiers(unsigned TQs) {
    assert(!(TQs & ~Qualifiers::FastMask)
           && "non-fast qualifier bits set in mask!");
    Value.setInt(Value.getInt() | TQs);
  }

  void removeLocalConst();
  void removeLocalVolatile();
  void removeLocalRestrict();
  void removeLocalCVRQualifiers(unsigned Mask);

  void removeLocalFastQualifiers() { Value.setInt(0); }
  void removeLocalFastQualifiers(unsigned Mask) {
    assert(!(Mask & ~Qualifiers::FastMask) && "mask has non-fast qualifiers");
    Value.setInt(Value.getInt() & ~Mask);
  }

  // Creates a type with the given qualifiers in addition to any
  // qualifiers already on this type.
  QualType withFastQualifiers(unsigned TQs) const {
    QualType T = *this;
    T.addFastQualifiers(TQs);
    return T;
  }

  // Creates a type with exactly the given fast qualifiers, removing
  // any existing fast qualifiers.
  QualType withExactLocalFastQualifiers(unsigned TQs) const {
    return withoutLocalFastQualifiers().withFastQualifiers(TQs);
  }

  // Removes fast qualifiers, but leaves any extended qualifiers in place.
  QualType withoutLocalFastQualifiers() const {
    QualType T = *this;
    T.removeLocalFastQualifiers();
    return T;
  }

  QualType getCanonicalType() const;

  /// \brief Return this type with all of the instance-specific qualifiers
  /// removed, but without removing any qualifiers that may have been applied
  /// through typedefs.
  QualType getLocalUnqualifiedType() const { return QualType(getTypePtr(), 0); }

  /// \brief Retrieve the unqualified variant of the given type,
  /// removing as little sugar as possible.
  ///
  /// This routine looks through various kinds of sugar to find the
  /// least-desugared type that is unqualified. For example, given:
  ///
  /// \code
  /// typedef int Integer;
  /// typedef const Integer CInteger;
  /// typedef CInteger DifferenceType;
  /// \endcode
  ///
  /// Executing \c getUnqualifiedType() on the type \c DifferenceType will
  /// desugar until we hit the type \c Integer, which has no qualifiers on it.
  ///
  /// The resulting type might still be qualified if it's an array
  /// type.  To strip qualifiers even from within an array type, use
  /// ASTContext::getUnqualifiedArrayType.
  inline QualType getUnqualifiedType() const;

  /// getSplitUnqualifiedType - Retrieve the unqualified variant of the
  /// given type, removing as little sugar as possible.
  ///
  /// Like getUnqualifiedType(), but also returns the set of
  /// qualifiers that were built up.
  ///
  /// The resulting type might still be qualified if it's an array
  /// type.  To strip qualifiers even from within an array type, use
  /// ASTContext::getUnqualifiedArrayType.
  inline SplitQualType getSplitUnqualifiedType() const;
  
  bool isMoreQualifiedThan(QualType Other) const;
  bool isAtLeastAsQualifiedAs(QualType Other) const;
  QualType getNonReferenceType() const;

  /// \brief Determine the type of a (typically non-lvalue) expression with the
  /// specified result type.
  ///                       
  /// This routine should be used for expressions for which the return type is
  /// explicitly specified (e.g., in a cast or call) and isn't necessarily
  /// an lvalue. It removes a top-level reference (since there are no 
  /// expressions of reference type) and deletes top-level cvr-qualifiers
  /// from non-class types (in C++) or all types (in C).
  QualType getNonLValueExprType(ASTContext &Context) const;
  
  /// getDesugaredType - Return the specified type with any "sugar" removed from
  /// the type.  This takes off typedefs, typeof's etc.  If the outer level of
  /// the type is already concrete, it returns it unmodified.  This is similar
  /// to getting the canonical type, but it doesn't remove *all* typedefs.  For
  /// example, it returns "T*" as "T*", (not as "int*"), because the pointer is
  /// concrete.
  ///
  /// Qualifiers are left in place.
  QualType getDesugaredType(const ASTContext &Context) const {
    return getDesugaredType(*this, Context);
  }

  SplitQualType getSplitDesugaredType() const {
    return getSplitDesugaredType(*this);
  }

  /// IgnoreParens - Returns the specified type after dropping any
  /// outer-level parentheses.
  QualType IgnoreParens() const {
    if (isa<ParenType>(*this))
      return QualType::IgnoreParens(*this);
    return *this;
  }

  /// operator==/!= - Indicate whether the specified types and qualifiers are
  /// identical.
  friend bool operator==(const QualType &LHS, const QualType &RHS) {
    return LHS.Value == RHS.Value;
  }
  friend bool operator!=(const QualType &LHS, const QualType &RHS) {
    return LHS.Value != RHS.Value;
  }
  std::string getAsString() const {
    return getAsString(split());
  }
  static std::string getAsString(SplitQualType split) {
    return getAsString(split.first, split.second);
  }
  static std::string getAsString(const Type *ty, Qualifiers qs);

  std::string getAsString(const PrintingPolicy &Policy) const {
    std::string S;
    getAsStringInternal(S, Policy);
    return S;
  }
  void getAsStringInternal(std::string &Str,
                           const PrintingPolicy &Policy) const {
    return getAsStringInternal(split(), Str, Policy);
  }
  static void getAsStringInternal(SplitQualType split, std::string &out,
                                  const PrintingPolicy &policy) {
    return getAsStringInternal(split.first, split.second, out, policy);
  }
  static void getAsStringInternal(const Type *ty, Qualifiers qs,
                                  std::string &out,
                                  const PrintingPolicy &policy);

  void dump(const char *s) const;
  void dump() const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(getAsOpaquePtr());
  }

  /// getAddressSpace - Return the address space of this type.
  inline unsigned getAddressSpace() const;

  /// GCAttrTypesAttr - Returns gc attribute of this type.
  inline Qualifiers::GC getObjCGCAttr() const;

  /// isObjCGCWeak true when Type is objc's weak.
  bool isObjCGCWeak() const {
    return getObjCGCAttr() == Qualifiers::Weak;
  }

  /// isObjCGCStrong true when Type is objc's strong.
  bool isObjCGCStrong() const {
    return getObjCGCAttr() == Qualifiers::Strong;
  }

  enum DestructionKind {
    DK_none,
    DK_cxx_destructor
  };

  /// isDestructedType - nonzero if objects of this type require
  /// non-trivial work to clean up after.  Non-zero because it's
  /// conceivable that qualifiers (objc_gc(weak)?) could make
  /// something require destruction.
  DestructionKind isDestructedType() const {
    return isDestructedTypeImpl(*this);
  }

private:
  // These methods are implemented in a separate translation unit;
  // "static"-ize them to avoid creating temporary QualTypes in the
  // caller.
  static bool isConstant(QualType T, ASTContext& Ctx);
  static QualType getDesugaredType(QualType T, const ASTContext &Context);
  static SplitQualType getSplitDesugaredType(QualType T);
  static SplitQualType getSplitUnqualifiedTypeImpl(QualType type);
  static QualType IgnoreParens(QualType T);
  static DestructionKind isDestructedTypeImpl(QualType type);
};

} // end clang.

namespace llvm {
/// Implement simplify_type for QualType, so that we can dyn_cast from QualType
/// to a specific Type class.
template<> struct simplify_type<const ::clang::QualType> {
  typedef const ::clang::Type *SimpleType;
  static SimpleType getSimplifiedValue(const ::clang::QualType &Val) {
    return Val.getTypePtr();
  }
};
template<> struct simplify_type< ::clang::QualType>
  : public simplify_type<const ::clang::QualType> {};

// Teach SmallPtrSet that QualType is "basically a pointer".
template<>
class PointerLikeTypeTraits<clang::QualType> {
public:
  static inline void *getAsVoidPointer(clang::QualType P) {
    return P.getAsOpaquePtr();
  }
  static inline clang::QualType getFromVoidPointer(void *P) {
    return clang::QualType::getFromOpaquePtr(P);
  }
  // Various qualifiers go in low bits.
  enum { NumLowBitsAvailable = 0 };
};

} // end namespace llvm

namespace clang {

/// \brief Base class that is common to both the \c ExtQuals and \c Type 
/// classes, which allows \c QualType to access the common fields between the
/// two.
///
class ExtQualsTypeCommonBase {
  ExtQualsTypeCommonBase(const Type *baseType, QualType canon)
    : BaseType(baseType), CanonicalType(canon) {}

  /// \brief The "base" type of an extended qualifiers type (\c ExtQuals) or
  /// a self-referential pointer (for \c Type).
  ///
  /// This pointer allows an efficient mapping from a QualType to its 
  /// underlying type pointer.
  const Type *const BaseType;

  /// \brief The canonical type of this type.  A QualType.
  QualType CanonicalType;

  friend class QualType;
  friend class Type;
  friend class ExtQuals;
};
  
/// ExtQuals - We can encode up to four bits in the low bits of a
/// type pointer, but there are many more type qualifiers that we want
/// to be able to apply to an arbitrary type.  Therefore we have this
/// struct, intended to be heap-allocated and used by QualType to
/// store qualifiers.
///
/// The current design tags the 'const', 'restrict', and 'volatile' qualifiers 
/// in three low bits on the QualType pointer; a fourth bit records whether
/// the pointer is an ExtQuals node. The extended qualifiers (address spaces,
/// Objective-C GC attributes) are much more rare.
class ExtQuals : public ExtQualsTypeCommonBase, public llvm::FoldingSetNode {
  // NOTE: changing the fast qualifiers should be straightforward as
  // long as you don't make 'const' non-fast.
  // 1. Qualifiers:
  //    a) Modify the bitmasks (Qualifiers::TQ and DeclSpec::TQ).
  //       Fast qualifiers must occupy the low-order bits.
  //    b) Update Qualifiers::FastWidth and FastMask.
  // 2. QualType:
  //    a) Update is{Volatile,Restrict}Qualified(), defined inline.
  //    b) Update remove{Volatile,Restrict}, defined near the end of
  //       this header.
  // 3. ASTContext:
  //    a) Update get{Volatile,Restrict}Type.

  /// Quals - the immutable set of qualifiers applied by this
  /// node;  always contains extended qualifiers.
  Qualifiers Quals;

  ExtQuals *this_() { return this; }

public:
  ExtQuals(const Type *baseType, QualType canon, Qualifiers quals) 
    : ExtQualsTypeCommonBase(baseType,
                             canon.isNull() ? QualType(this_(), 0) : canon),
      Quals(quals)
  {
    assert(Quals.hasNonFastQualifiers()
           && "ExtQuals created with no fast qualifiers");
    assert(!Quals.hasFastQualifiers()
           && "ExtQuals created with fast qualifiers");
  }

  Qualifiers getQualifiers() const { return Quals; }

  bool hasObjCGCAttr() const { return Quals.hasObjCGCAttr(); }
  Qualifiers::GC getObjCGCAttr() const { return Quals.getObjCGCAttr(); }

  bool hasAddressSpace() const { return Quals.hasAddressSpace(); }
  unsigned getAddressSpace() const { return Quals.getAddressSpace(); }

  const Type *getBaseType() const { return BaseType; }

public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, getBaseType(), Quals);
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const Type *BaseType,
                      Qualifiers Quals) {
    assert(!Quals.hasFastQualifiers() && "fast qualifiers in ExtQuals hash!");
    ID.AddPointer(BaseType);
    Quals.Profile(ID);
  }
};

/// \brief The kind of C++0x ref-qualifier associated with a function type, 
/// which determines whether a member function's "this" object can be an 
/// lvalue, rvalue, or neither.
enum RefQualifierKind {
  /// \brief No ref-qualifier was provided.
  RQ_None = 0,
  /// \brief An lvalue ref-qualifier was provided (\c &).
  RQ_LValue,
  /// \brief An rvalue ref-qualifier was provided (\c &&).
  RQ_RValue
};
  
/// Type - This is the base class of the type hierarchy.  A central concept
/// with types is that each type always has a canonical type.  A canonical type
/// is the type with any typedef names stripped out of it or the types it
/// references.  For example, consider:
///
///  typedef int  foo;
///  typedef foo* bar;
///    'int *'    'foo *'    'bar'
///
/// There will be a Type object created for 'int'.  Since int is canonical, its
/// canonicaltype pointer points to itself.  There is also a Type for 'foo' (a
/// TypedefType).  Its CanonicalType pointer points to the 'int' Type.  Next
/// there is a PointerType that represents 'int*', which, like 'int', is
/// canonical.  Finally, there is a PointerType type for 'foo*' whose canonical
/// type is 'int*', and there is a TypedefType for 'bar', whose canonical type
/// is also 'int*'.
///
/// Non-canonical types are useful for emitting diagnostics, without losing
/// information about typedefs being used.  Canonical types are useful for type
/// comparisons (they allow by-pointer equality tests) and useful for reasoning
/// about whether something has a particular form (e.g. is a function type),
/// because they implicitly, recursively, strip all typedefs out of a type.
///
/// Types, once created, are immutable.
///
class Type : public ExtQualsTypeCommonBase {
public:
  enum TypeClass {
#define TYPE(Class, Base) Class,
#define LAST_TYPE(Class) TypeLast = Class,
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
    TagFirst = Record, TagLast = Enum
  };

private:
  Type(const Type&);           // DO NOT IMPLEMENT.
  void operator=(const Type&); // DO NOT IMPLEMENT.

  /// Bitfields required by the Type class.
  class TypeBitfields {
    friend class Type;
    template <class T> friend class TypePropertyCache;

    /// TypeClass bitfield - Enum that specifies what subclass this belongs to.
    unsigned TC : 8;

    /// Dependent - Whether this type is a dependent type (C++ [temp.dep.type]).
    /// Note that this should stay at the end of the ivars for Type so that
    /// subclasses can pack their bitfields into the same word.
    unsigned Dependent : 1;
  
    /// \brief Whether this type is a variably-modified type (C99 6.7.5).
    unsigned VariablyModified : 1;

    /// \brief Whether this type contains an unexpanded parameter pack
    /// (for C++0x variadic templates).
    unsigned ContainsUnexpandedParameterPack : 1;
  
    /// \brief Nonzero if the cache (i.e. the bitfields here starting
    /// with 'Cache') is valid.  If so, then this is a
    /// LangOptions::VisibilityMode+1.
    mutable unsigned CacheValidAndVisibility : 2;
  
    /// \brief Linkage of this type.
    mutable unsigned CachedLinkage : 2;

    /// \brief Whether this type involves and local or unnamed types. 
    mutable unsigned CachedLocalOrUnnamed : 1;
  
    /// \brief FromAST - Whether this type comes from an AST file.
    mutable unsigned FromAST : 1;

    bool isCacheValid() const {
      return (CacheValidAndVisibility != 0);
    }
    Visibility getVisibility() const {
      assert(isCacheValid() && "getting linkage from invalid cache");
      return static_cast<Visibility>(CacheValidAndVisibility-1);
    }
    Linkage getLinkage() const {
      assert(isCacheValid() && "getting linkage from invalid cache");
      return static_cast<Linkage>(CachedLinkage);
    }
    bool hasLocalOrUnnamedType() const {
      assert(isCacheValid() && "getting linkage from invalid cache");
      return CachedLocalOrUnnamed;
    }
  };
  enum { NumTypeBits = 17 };

protected:
  // These classes allow subclasses to somewhat cleanly pack bitfields
  // into Type.

  class ArrayTypeBitfields {
    friend class ArrayType;

    unsigned : NumTypeBits;

    /// IndexTypeQuals - CVR qualifiers from declarations like
    /// 'int X[static restrict 4]'. For function parameters only.
    unsigned IndexTypeQuals : 3;

    /// SizeModifier - storage class qualifiers from declarations like
    /// 'int X[static restrict 4]'. For function parameters only.
    /// Actually an ArrayType::ArraySizeModifier.
    unsigned SizeModifier : 3;
  };

  class BuiltinTypeBitfields {
    friend class BuiltinType;

    unsigned : NumTypeBits;

    /// The kind (BuiltinType::Kind) of builtin type this is.
    unsigned Kind : 8;
  };

  class FunctionTypeBitfields {
    friend class FunctionType;

    unsigned : NumTypeBits;

    /// Extra information which affects how the function is called, like
    /// regparm and the calling convention.
    unsigned ExtInfo : 8;

    /// Whether the function is variadic.  Only used by FunctionProtoType.
    unsigned Variadic : 1;

    /// TypeQuals - Used only by FunctionProtoType, put here to pack with the
    /// other bitfields.
    /// The qualifiers are part of FunctionProtoType because...
    ///
    /// C++ 8.3.5p4: The return type, the parameter type list and the
    /// cv-qualifier-seq, [...], are part of the function type.
    unsigned TypeQuals : 3;
    
    /// \brief The ref-qualifier associated with a \c FunctionProtoType.
    ///
    /// This is a value of type \c RefQualifierKind.
    unsigned RefQualifier : 2;
  };

  class ObjCObjectTypeBitfields {
    friend class ObjCObjectType;

    unsigned : NumTypeBits;

    /// NumProtocols - The number of protocols stored directly on this
    /// object type.
    unsigned NumProtocols : 32 - NumTypeBits;
  };

  class ReferenceTypeBitfields {
    friend class ReferenceType;

    unsigned : NumTypeBits;

    /// True if the type was originally spelled with an lvalue sigil.
    /// This is never true of rvalue references but can also be false
    /// on lvalue references because of C++0x [dcl.typedef]p9,
    /// as follows:
    ///
    ///   typedef int &ref;    // lvalue, spelled lvalue
    ///   typedef int &&rvref; // rvalue
    ///   ref &a;              // lvalue, inner ref, spelled lvalue
    ///   ref &&a;             // lvalue, inner ref
    ///   rvref &a;            // lvalue, inner ref, spelled lvalue
    ///   rvref &&a;           // rvalue, inner ref
    unsigned SpelledAsLValue : 1;

    /// True if the inner type is a reference type.  This only happens
    /// in non-canonical forms.
    unsigned InnerRef : 1;
  };

  class TypeWithKeywordBitfields {
    friend class TypeWithKeyword;

    unsigned : NumTypeBits;

    /// An ElaboratedTypeKeyword.  8 bits for efficient access.
    unsigned Keyword : 8;
  };

  class VectorTypeBitfields {
    friend class VectorType;

    unsigned : NumTypeBits;

    /// VecKind - The kind of vector, either a generic vector type or some
    /// target-specific vector type such as for AltiVec or Neon.
    unsigned VecKind : 3;

    /// NumElements - The number of elements in the vector.
    unsigned NumElements : 29 - NumTypeBits;
  };

  class AttributedTypeBitfields {
    friend class AttributedType;

    unsigned : NumTypeBits;

    /// AttrKind - an AttributedType::Kind
    unsigned AttrKind : 32 - NumTypeBits;
  };

  union {
    TypeBitfields TypeBits;
    ArrayTypeBitfields ArrayTypeBits;
    AttributedTypeBitfields AttributedTypeBits;
    BuiltinTypeBitfields BuiltinTypeBits;
    FunctionTypeBitfields FunctionTypeBits;
    ObjCObjectTypeBitfields ObjCObjectTypeBits;
    ReferenceTypeBitfields ReferenceTypeBits;
    TypeWithKeywordBitfields TypeWithKeywordBits;
    VectorTypeBitfields VectorTypeBits;
  };

private:
  /// \brief Set whether this type comes from an AST file.
  void setFromAST(bool V = true) const { 
    TypeBits.FromAST = V;
  }

  template <class T> friend class TypePropertyCache;

protected:
  // silence VC++ warning C4355: 'this' : used in base member initializer list
  Type *this_() { return this; }
  Type(TypeClass tc, QualType canon, bool Dependent, bool VariablyModified,
       bool ContainsUnexpandedParameterPack)
    : ExtQualsTypeCommonBase(this,
                             canon.isNull() ? QualType(this_(), 0) : canon) {
    TypeBits.TC = tc;
    TypeBits.Dependent = Dependent;
    TypeBits.VariablyModified = VariablyModified;
    TypeBits.ContainsUnexpandedParameterPack = ContainsUnexpandedParameterPack;
    TypeBits.CacheValidAndVisibility = 0;
    TypeBits.CachedLocalOrUnnamed = false;
    TypeBits.CachedLinkage = NoLinkage;
    TypeBits.FromAST = false;
  }
  friend class ASTContext;

  void setDependent(bool D = true) { TypeBits.Dependent = D; }
  void setVariablyModified(bool VM = true) { TypeBits.VariablyModified = VM; }
  void setContainsUnexpandedParameterPack(bool PP = true) {
    TypeBits.ContainsUnexpandedParameterPack = PP;
  }

public:
  TypeClass getTypeClass() const { return static_cast<TypeClass>(TypeBits.TC); }

  /// \brief Whether this type comes from an AST file.
  bool isFromAST() const { return TypeBits.FromAST; }

  /// \brief Whether this type is or contains an unexpanded parameter
  /// pack, used to support C++0x variadic templates.
  ///
  /// A type that contains a parameter pack shall be expanded by the
  /// ellipsis operator at some point. For example, the typedef in the
  /// following example contains an unexpanded parameter pack 'T':
  ///
  /// \code
  /// template<typename ...T>
  /// struct X {
  ///   typedef T* pointer_types; // ill-formed; T is a parameter pack.
  /// };
  /// \endcode
  ///
  /// Note that this routine does not specify which 
  bool containsUnexpandedParameterPack() const { 
    return TypeBits.ContainsUnexpandedParameterPack;
  }

  /// Determines if this type would be canonical if it had no further
  /// qualification.
  bool isCanonicalUnqualified() const {
    return CanonicalType == QualType(this, 0);
  }

  /// Types are partitioned into 3 broad categories (C99 6.2.5p1):
  /// object types, function types, and incomplete types.

  /// isIncompleteType - Return true if this is an incomplete type.
  /// A type that can describe objects, but which lacks information needed to
  /// determine its size (e.g. void, or a fwd declared struct). Clients of this
  /// routine will need to determine if the size is actually required.
  bool isIncompleteType() const;

  /// isIncompleteOrObjectType - Return true if this is an incomplete or object
  /// type, in other words, not a function type.
  bool isIncompleteOrObjectType() const {
    return !isFunctionType();
  }
  
  /// \brief Determine whether this type is an object type.
  bool isObjectType() const {
    // C++ [basic.types]p8:
    //   An object type is a (possibly cv-qualified) type that is not a 
    //   function type, not a reference type, and not a void type.
    return !isReferenceType() && !isFunctionType() && !isVoidType();
  }

  /// isPODType - Return true if this is a plain-old-data type (C++ 3.9p10).
  bool isPODType() const;

  /// isLiteralType - Return true if this is a literal type
  /// (C++0x [basic.types]p10)
  bool isLiteralType() const;

  /// Helper methods to distinguish type categories. All type predicates
  /// operate on the canonical type, ignoring typedefs and qualifiers.

  /// isBuiltinType - returns true if the type is a builtin type.
  bool isBuiltinType() const;

  /// isSpecificBuiltinType - Test for a particular builtin type.
  bool isSpecificBuiltinType(unsigned K) const;

  /// isPlaceholderType - Test for a type which does not represent an
  /// actual type-system type but is instead used as a placeholder for
  /// various convenient purposes within Clang.  All such types are
  /// BuiltinTypes.
  bool isPlaceholderType() const;

  /// isIntegerType() does *not* include complex integers (a GCC extension).
  /// isComplexIntegerType() can be used to test for complex integers.
  bool isIntegerType() const;     // C99 6.2.5p17 (int, char, bool, enum)
  bool isEnumeralType() const;
  bool isBooleanType() const;
  bool isCharType() const;
  bool isWideCharType() const;
  bool isAnyCharacterType() const;
  bool isIntegralType(ASTContext &Ctx) const;
  
  /// \brief Determine whether this type is an integral or enumeration type.
  bool isIntegralOrEnumerationType() const;
  /// \brief Determine whether this type is an integral or unscoped enumeration
  /// type.
  bool isIntegralOrUnscopedEnumerationType() const;
                                   
  /// Floating point categories.
  bool isRealFloatingType() const; // C99 6.2.5p10 (float, double, long double)
  /// isComplexType() does *not* include complex integers (a GCC extension).
  /// isComplexIntegerType() can be used to test for complex integers.
  bool isComplexType() const;      // C99 6.2.5p11 (complex)
  bool isAnyComplexType() const;   // C99 6.2.5p11 (complex) + Complex Int.
  bool isFloatingType() const;     // C99 6.2.5p11 (real floating + complex)
  bool isRealType() const;         // C99 6.2.5p17 (real floating + integer)
  bool isArithmeticType() const;   // C99 6.2.5p18 (integer + floating)
  bool isVoidType() const;         // C99 6.2.5p19
  bool isDerivedType() const;      // C99 6.2.5p20
  bool isScalarType() const;       // C99 6.2.5p21 (arithmetic + pointers)
  bool isAggregateType() const;

  // Type Predicates: Check to see if this type is structurally the specified
  // type, ignoring typedefs and qualifiers.
  bool isFunctionType() const;
  bool isFunctionNoProtoType() const { return getAs<FunctionNoProtoType>(); }
  bool isFunctionProtoType() const { return getAs<FunctionProtoType>(); }
  bool isPointerType() const;
  bool isAnyPointerType() const;   // Any C pointer or ObjC object pointer
  bool isBlockPointerType() const;
  bool isVoidPointerType() const;
  bool isReferenceType() const;
  bool isLValueReferenceType() const;
  bool isRValueReferenceType() const;
  bool isFunctionPointerType() const;
  bool isMemberPointerType() const;
  bool isMemberFunctionPointerType() const;
  bool isMemberDataPointerType() const;
  bool isArrayType() const;
  bool isConstantArrayType() const;
  bool isIncompleteArrayType() const;
  bool isVariableArrayType() const;
  bool isDependentSizedArrayType() const;
  bool isRecordType() const;
  bool isClassType() const;
  bool isStructureType() const;
  bool isStructureOrClassType() const;
  bool isUnionType() const;
  bool isComplexIntegerType() const;            // GCC _Complex integer type.
  bool isVectorType() const;                    // GCC vector type.
  bool isExtVectorType() const;                 // Extended vector type.
  bool isObjCObjectPointerType() const;         // Pointer to *any* ObjC object.
  // FIXME: change this to 'raw' interface type, so we can used 'interface' type
  // for the common case.
  bool isObjCObjectType() const;                // NSString or typeof(*(id)0)
  bool isObjCQualifiedInterfaceType() const;    // NSString<foo>
  bool isObjCQualifiedIdType() const;           // id<foo>
  bool isObjCQualifiedClassType() const;        // Class<foo>
  bool isObjCObjectOrInterfaceType() const;
  bool isObjCIdType() const;                    // id
  bool isObjCClassType() const;                 // Class
  bool isObjCSelType() const;                 // Class
  bool isObjCBuiltinType() const;               // 'id' or 'Class'
  bool isTemplateTypeParmType() const;          // C++ template type parameter
  bool isNullPtrType() const;                   // C++0x nullptr_t

  enum ScalarTypeKind {
    STK_Pointer,
    STK_MemberPointer,
    STK_Bool,
    STK_Integral,
    STK_Floating,
    STK_IntegralComplex,
    STK_FloatingComplex
  };
  /// getScalarTypeKind - Given that this is a scalar type, classify it.
  ScalarTypeKind getScalarTypeKind() const;

  /// isDependentType - Whether this type is a dependent type, meaning
  /// that its definition somehow depends on a template parameter
  /// (C++ [temp.dep.type]).
  bool isDependentType() const { return TypeBits.Dependent; }
  
  /// \brief Whether this type is a variably-modified type (C99 6.7.5).
  bool isVariablyModifiedType() const { return TypeBits.VariablyModified; }

  /// \brief Whether this type involves a variable-length array type
  /// with a definite size.
  bool hasSizedVLAType() const;
  
  /// \brief Whether this type is or contains a local or unnamed type.
  bool hasUnnamedOrLocalType() const;
  
  bool isOverloadableType() const;

  /// \brief Determine wither this type is a C++ elaborated-type-specifier.
  bool isElaboratedTypeSpecifier() const;
  
  /// hasPointerRepresentation - Whether this type is represented
  /// natively as a pointer; this includes pointers, references, block
  /// pointers, and Objective-C interface, qualified id, and qualified
  /// interface types, as well as nullptr_t.
  bool hasPointerRepresentation() const;

  /// hasObjCPointerRepresentation - Whether this type can represent
  /// an objective pointer type for the purpose of GC'ability
  bool hasObjCPointerRepresentation() const;

  /// \brief Determine whether this type has an integer representation
  /// of some sort, e.g., it is an integer type or a vector.
  bool hasIntegerRepresentation() const;

  /// \brief Determine whether this type has an signed integer representation
  /// of some sort, e.g., it is an signed integer type or a vector.
  bool hasSignedIntegerRepresentation() const;

  /// \brief Determine whether this type has an unsigned integer representation
  /// of some sort, e.g., it is an unsigned integer type or a vector.
  bool hasUnsignedIntegerRepresentation() const;

  /// \brief Determine whether this type has a floating-point representation
  /// of some sort, e.g., it is a floating-point type or a vector thereof.
  bool hasFloatingRepresentation() const;

  // Type Checking Functions: Check to see if this type is structurally the
  // specified type, ignoring typedefs and qualifiers, and return a pointer to
  // the best type we can.
  const RecordType *getAsStructureType() const;
  /// NOTE: getAs*ArrayType are methods on ASTContext.
  const RecordType *getAsUnionType() const;
  const ComplexType *getAsComplexIntegerType() const; // GCC complex int type.
  // The following is a convenience method that returns an ObjCObjectPointerType
  // for object declared using an interface.
  const ObjCObjectPointerType *getAsObjCInterfacePointerType() const;
  const ObjCObjectPointerType *getAsObjCQualifiedIdType() const;
  const ObjCObjectType *getAsObjCQualifiedInterfaceType() const;
  const CXXRecordDecl *getCXXRecordDeclForPointerType() const;

  /// \brief Retrieves the CXXRecordDecl that this type refers to, either
  /// because the type is a RecordType or because it is the injected-class-name 
  /// type of a class template or class template partial specialization.
  CXXRecordDecl *getAsCXXRecordDecl() const;

  /// \brief Get the AutoType whose type will be deduced for a variable with
  /// an initializer of this type. This looks through declarators like pointer
  /// types, but not through decltype or typedefs.
  AutoType *getContainedAutoType() const;
  
  /// Member-template getAs<specific type>'.  Look through sugar for
  /// an instance of <specific type>.   This scheme will eventually
  /// replace the specific getAsXXXX methods above.
  ///
  /// There are some specializations of this member template listed
  /// immediately following this class.
  template <typename T> const T *getAs() const;

  /// A variant of getAs<> for array types which silently discards
  /// qualifiers from the outermost type.
  const ArrayType *getAsArrayTypeUnsafe() const;

  /// Member-template castAs<specific type>.  Look through sugar for
  /// the underlying instance of <specific type>.
  ///
  /// This method has the same relationship to getAs<T> as cast<T> has
  /// to dyn_cast<T>; which is to say, the underlying type *must*
  /// have the intended type, and this method will never return null.
  template <typename T> const T *castAs() const;

  /// A variant of castAs<> for array type which silently discards
  /// qualifiers from the outermost type.
  const ArrayType *castAsArrayTypeUnsafe() const;

  /// getBaseElementTypeUnsafe - Get the base element type of this
  /// type, potentially discarding type qualifiers.  This method
  /// should never be used when type qualifiers are meaningful.
  const Type *getBaseElementTypeUnsafe() const;

  /// getArrayElementTypeNoTypeQual - If this is an array type, return the
  /// element type of the array, potentially with type qualifiers missing.
  /// This method should never be used when type qualifiers are meaningful.
  const Type *getArrayElementTypeNoTypeQual() const;

  /// getPointeeType - If this is a pointer, ObjC object pointer, or block
  /// pointer, this returns the respective pointee.
  QualType getPointeeType() const;

  /// getUnqualifiedDesugaredType() - Return the specified type with
  /// any "sugar" removed from the type, removing any typedefs,
  /// typeofs, etc., as well as any qualifiers.
  const Type *getUnqualifiedDesugaredType() const;

  /// More type predicates useful for type checking/promotion
  bool isPromotableIntegerType() const; // C99 6.3.1.1p2

  /// isSignedIntegerType - Return true if this is an integer type that is
  /// signed, according to C99 6.2.5p4 [char, signed char, short, int, long..],
  /// an enum decl which has a signed representation, or a vector of signed
  /// integer element type.
  bool isSignedIntegerType() const;

  /// isUnsignedIntegerType - Return true if this is an integer type that is
  /// unsigned, according to C99 6.2.5p6 [which returns true for _Bool], an enum
  /// decl which has an unsigned representation, or a vector of unsigned integer
  /// element type.
  bool isUnsignedIntegerType() const;

  /// isConstantSizeType - Return true if this is not a variable sized type,
  /// according to the rules of C99 6.7.5p3.  It is not legal to call this on
  /// incomplete types.
  bool isConstantSizeType() const;

  /// isSpecifierType - Returns true if this type can be represented by some
  /// set of type specifiers.
  bool isSpecifierType() const;

  /// \brief Determine the linkage of this type.
  Linkage getLinkage() const;

  /// \brief Determine the visibility of this type.
  Visibility getVisibility() const;

  /// \brief Determine the linkage and visibility of this type.
  std::pair<Linkage,Visibility> getLinkageAndVisibility() const;
  
  /// \brief Note that the linkage is no longer known.
  void ClearLinkageCache();
  
  const char *getTypeClassName() const;

  QualType getCanonicalTypeInternal() const {
    return CanonicalType;
  }
  CanQualType getCanonicalTypeUnqualified() const; // in CanonicalType.h
  void dump() const;
  static bool classof(const Type *) { return true; }

  friend class ASTReader;
  friend class ASTWriter;
};

template <> inline const TypedefType *Type::getAs() const {
  return dyn_cast<TypedefType>(this);
}

// We can do canonical leaf types faster, because we don't have to
// worry about preserving child type decoration.
#define TYPE(Class, Base)
#define LEAF_TYPE(Class) \
template <> inline const Class##Type *Type::getAs() const { \
  return dyn_cast<Class##Type>(CanonicalType); \
} \
template <> inline const Class##Type *Type::castAs() const { \
  return cast<Class##Type>(CanonicalType); \
}
#include "clang/AST/TypeNodes.def"


/// BuiltinType - This class is used for builtin types like 'int'.  Builtin
/// types are always canonical and have a literal name field.
class BuiltinType : public Type {
public:
  enum Kind {
    Void,

    Bool,     // This is bool and/or _Bool.
    Char_U,   // This is 'char' for targets where char is unsigned.
    UChar,    // This is explicitly qualified unsigned char.
    WChar_U,  // This is 'wchar_t' for C++, when unsigned.
    Char16,   // This is 'char16_t' for C++.
    Char32,   // This is 'char32_t' for C++.
    UShort,
    UInt,
    ULong,
    ULongLong,
    UInt128,  // __uint128_t

    Char_S,   // This is 'char' for targets where char is signed.
    SChar,    // This is explicitly qualified signed char.
    WChar_S,  // This is 'wchar_t' for C++, when signed.
    Short,
    Int,
    Long,
    LongLong,
    Int128,   // __int128_t

    Float, Double, LongDouble,

    NullPtr,  // This is the type of C++0x 'nullptr'.

    /// This represents the type of an expression whose type is
    /// totally unknown, e.g. 'T::foo'.  It is permitted for this to
    /// appear in situations where the structure of the type is
    /// theoretically deducible.
    Dependent,

    Overload,  // This represents the type of an overloaded function declaration.

    /// The primitive Objective C 'id' type.  The type pointed to by the
    /// user-visible 'id' type.  Only ever shows up in an AST as the base
    /// type of an ObjCObjectType.
    ObjCId,

    /// The primitive Objective C 'Class' type.  The type pointed to by the
    /// user-visible 'Class' type.  Only ever shows up in an AST as the
    /// base type of an ObjCObjectType.
    ObjCClass,

    ObjCSel    // This represents the ObjC 'SEL' type.
  };

public:
  BuiltinType(Kind K)
    : Type(Builtin, QualType(), /*Dependent=*/(K == Dependent),
           /*VariablyModified=*/false,
           /*Unexpanded paramter pack=*/false) {
    BuiltinTypeBits.Kind = K;
  }

  Kind getKind() const { return static_cast<Kind>(BuiltinTypeBits.Kind); }
  const char *getName(const LangOptions &LO) const;

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  bool isInteger() const {
    return getKind() >= Bool && getKind() <= Int128;
  }

  bool isSignedInteger() const {
    return getKind() >= Char_S && getKind() <= Int128;
  }

  bool isUnsignedInteger() const {
    return getKind() >= Bool && getKind() <= UInt128;
  }

  bool isFloatingPoint() const {
    return getKind() >= Float && getKind() <= LongDouble;
  }

  /// Determines whether this type is a "forbidden" placeholder type,
  /// i.e. a type which cannot appear in arbitrary positions in a
  /// fully-formed expression.
  bool isPlaceholderType() const {
    return getKind() == Overload;
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Builtin; }
  static bool classof(const BuiltinType *) { return true; }
};

/// ComplexType - C99 6.2.5p11 - Complex values.  This supports the C99 complex
/// types (_Complex float etc) as well as the GCC integer complex extensions.
///
class ComplexType : public Type, public llvm::FoldingSetNode {
  QualType ElementType;
  ComplexType(QualType Element, QualType CanonicalPtr) :
    Type(Complex, CanonicalPtr, Element->isDependentType(),
         Element->isVariablyModifiedType(),
         Element->containsUnexpandedParameterPack()),
    ElementType(Element) {
  }
  friend class ASTContext;  // ASTContext creates these.

public:
  QualType getElementType() const { return ElementType; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Element) {
    ID.AddPointer(Element.getAsOpaquePtr());
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Complex; }
  static bool classof(const ComplexType *) { return true; }
};

/// ParenType - Sugar for parentheses used when specifying types.
///
class ParenType : public Type, public llvm::FoldingSetNode {
  QualType Inner;

  ParenType(QualType InnerType, QualType CanonType) :
    Type(Paren, CanonType, InnerType->isDependentType(),
         InnerType->isVariablyModifiedType(),
         InnerType->containsUnexpandedParameterPack()),
    Inner(InnerType) {
  }
  friend class ASTContext;  // ASTContext creates these.

public:

  QualType getInnerType() const { return Inner; }

  bool isSugared() const { return true; }
  QualType desugar() const { return getInnerType(); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getInnerType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Inner) {
    Inner.Profile(ID);
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Paren; }
  static bool classof(const ParenType *) { return true; }
};

/// PointerType - C99 6.7.5.1 - Pointer Declarators.
///
class PointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;

  PointerType(QualType Pointee, QualType CanonicalPtr) :
    Type(Pointer, CanonicalPtr, Pointee->isDependentType(),
         Pointee->isVariablyModifiedType(),
         Pointee->containsUnexpandedParameterPack()), 
    PointeeType(Pointee) {
  }
  friend class ASTContext;  // ASTContext creates these.

public:

  QualType getPointeeType() const { return PointeeType; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pointee) {
    ID.AddPointer(Pointee.getAsOpaquePtr());
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Pointer; }
  static bool classof(const PointerType *) { return true; }
};

/// BlockPointerType - pointer to a block type.
/// This type is to represent types syntactically represented as
/// "void (^)(int)", etc. Pointee is required to always be a function type.
///
class BlockPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;  // Block is some kind of pointer type
  BlockPointerType(QualType Pointee, QualType CanonicalCls) :
    Type(BlockPointer, CanonicalCls, Pointee->isDependentType(),
         Pointee->isVariablyModifiedType(),
         Pointee->containsUnexpandedParameterPack()),
    PointeeType(Pointee) {
  }
  friend class ASTContext;  // ASTContext creates these.
  
public:

  // Get the pointee type. Pointee is required to always be a function type.
  QualType getPointeeType() const { return PointeeType; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
      Profile(ID, getPointeeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pointee) {
      ID.AddPointer(Pointee.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == BlockPointer;
  }
  static bool classof(const BlockPointerType *) { return true; }
};

/// ReferenceType - Base for LValueReferenceType and RValueReferenceType
///
class ReferenceType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;

protected:
  ReferenceType(TypeClass tc, QualType Referencee, QualType CanonicalRef,
                bool SpelledAsLValue) :
    Type(tc, CanonicalRef, Referencee->isDependentType(),
         Referencee->isVariablyModifiedType(),
         Referencee->containsUnexpandedParameterPack()), 
    PointeeType(Referencee) 
  {
    ReferenceTypeBits.SpelledAsLValue = SpelledAsLValue;
    ReferenceTypeBits.InnerRef = Referencee->isReferenceType();
  }
  
public:
  bool isSpelledAsLValue() const { return ReferenceTypeBits.SpelledAsLValue; }
  bool isInnerRef() const { return ReferenceTypeBits.InnerRef; }
  
  QualType getPointeeTypeAsWritten() const { return PointeeType; }
  QualType getPointeeType() const {
    // FIXME: this might strip inner qualifiers; okay?
    const ReferenceType *T = this;
    while (T->isInnerRef())
      T = T->PointeeType->castAs<ReferenceType>();
    return T->PointeeType;
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, PointeeType, isSpelledAsLValue());
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      QualType Referencee,
                      bool SpelledAsLValue) {
    ID.AddPointer(Referencee.getAsOpaquePtr());
    ID.AddBoolean(SpelledAsLValue);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == LValueReference ||
           T->getTypeClass() == RValueReference;
  }
  static bool classof(const ReferenceType *) { return true; }
};

/// LValueReferenceType - C++ [dcl.ref] - Lvalue reference
///
class LValueReferenceType : public ReferenceType {
  LValueReferenceType(QualType Referencee, QualType CanonicalRef,
                      bool SpelledAsLValue) :
    ReferenceType(LValueReference, Referencee, CanonicalRef, SpelledAsLValue)
  {}
  friend class ASTContext; // ASTContext creates these
public:
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == LValueReference;
  }
  static bool classof(const LValueReferenceType *) { return true; }
};

/// RValueReferenceType - C++0x [dcl.ref] - Rvalue reference
///
class RValueReferenceType : public ReferenceType {
  RValueReferenceType(QualType Referencee, QualType CanonicalRef) :
    ReferenceType(RValueReference, Referencee, CanonicalRef, false) {
  }
  friend class ASTContext; // ASTContext creates these
public:
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == RValueReference;
  }
  static bool classof(const RValueReferenceType *) { return true; }
};

/// MemberPointerType - C++ 8.3.3 - Pointers to members
///
class MemberPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;
  /// The class of which the pointee is a member. Must ultimately be a
  /// RecordType, but could be a typedef or a template parameter too.
  const Type *Class;

  MemberPointerType(QualType Pointee, const Type *Cls, QualType CanonicalPtr) :
    Type(MemberPointer, CanonicalPtr,
         Cls->isDependentType() || Pointee->isDependentType(),
         Pointee->isVariablyModifiedType(),
         (Cls->containsUnexpandedParameterPack() || 
          Pointee->containsUnexpandedParameterPack())),
    PointeeType(Pointee), Class(Cls) {
  }
  friend class ASTContext; // ASTContext creates these.
  
public:
  QualType getPointeeType() const { return PointeeType; }

  /// Returns true if the member type (i.e. the pointee type) is a
  /// function type rather than a data-member type.
  bool isMemberFunctionPointer() const {
    return PointeeType->isFunctionProtoType();
  }

  /// Returns true if the member type (i.e. the pointee type) is a
  /// data type rather than a function type.
  bool isMemberDataPointer() const {
    return !PointeeType->isFunctionProtoType();
  }

  const Type *getClass() const { return Class; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType(), getClass());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pointee,
                      const Type *Class) {
    ID.AddPointer(Pointee.getAsOpaquePtr());
    ID.AddPointer(Class);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == MemberPointer;
  }
  static bool classof(const MemberPointerType *) { return true; }
};

/// ArrayType - C99 6.7.5.2 - Array Declarators.
///
class ArrayType : public Type, public llvm::FoldingSetNode {
public:
  /// ArraySizeModifier - Capture whether this is a normal array (e.g. int X[4])
  /// an array with a static size (e.g. int X[static 4]), or an array
  /// with a star size (e.g. int X[*]).
  /// 'static' is only allowed on function parameters.
  enum ArraySizeModifier {
    Normal, Static, Star
  };
private:
  /// ElementType - The element type of the array.
  QualType ElementType;

protected:
  // C++ [temp.dep.type]p1:
  //   A type is dependent if it is...
  //     - an array type constructed from any dependent type or whose
  //       size is specified by a constant expression that is
  //       value-dependent,
  ArrayType(TypeClass tc, QualType et, QualType can,
            ArraySizeModifier sm, unsigned tq,
            bool ContainsUnexpandedParameterPack)
    : Type(tc, can, et->isDependentType() || tc == DependentSizedArray,
           (tc == VariableArray || et->isVariablyModifiedType()),
           ContainsUnexpandedParameterPack),
      ElementType(et) {
    ArrayTypeBits.IndexTypeQuals = tq;
    ArrayTypeBits.SizeModifier = sm;
  }

  friend class ASTContext;  // ASTContext creates these.

public:
  QualType getElementType() const { return ElementType; }
  ArraySizeModifier getSizeModifier() const {
    return ArraySizeModifier(ArrayTypeBits.SizeModifier);
  }
  Qualifiers getIndexTypeQualifiers() const {
    return Qualifiers::fromCVRMask(getIndexTypeCVRQualifiers());
  }
  unsigned getIndexTypeCVRQualifiers() const {
    return ArrayTypeBits.IndexTypeQuals;
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == ConstantArray ||
           T->getTypeClass() == VariableArray ||
           T->getTypeClass() == IncompleteArray ||
           T->getTypeClass() == DependentSizedArray;
  }
  static bool classof(const ArrayType *) { return true; }
};

/// ConstantArrayType - This class represents the canonical version of
/// C arrays with a specified constant size.  For example, the canonical
/// type for 'int A[4 + 4*100]' is a ConstantArrayType where the element
/// type is 'int' and the size is 404.
class ConstantArrayType : public ArrayType {
  llvm::APInt Size; // Allows us to unique the type.

  ConstantArrayType(QualType et, QualType can, const llvm::APInt &size,
                    ArraySizeModifier sm, unsigned tq)
    : ArrayType(ConstantArray, et, can, sm, tq,
                et->containsUnexpandedParameterPack()),
      Size(size) {}
protected:
  ConstantArrayType(TypeClass tc, QualType et, QualType can,
                    const llvm::APInt &size, ArraySizeModifier sm, unsigned tq)
    : ArrayType(tc, et, can, sm, tq, et->containsUnexpandedParameterPack()), 
      Size(size) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  const llvm::APInt &getSize() const { return Size; }
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  
  /// \brief Determine the number of bits required to address a member of
  // an array with the given element type and number of elements.
  static unsigned getNumAddressingBits(ASTContext &Context,
                                       QualType ElementType,
                                       const llvm::APInt &NumElements);
  
  /// \brief Determine the maximum number of active bits that an array's size
  /// can require, which limits the maximum size of the array.
  static unsigned getMaxSizeBits(ASTContext &Context);
  
  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType(), getSize(),
            getSizeModifier(), getIndexTypeCVRQualifiers());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ET,
                      const llvm::APInt &ArraySize, ArraySizeModifier SizeMod,
                      unsigned TypeQuals) {
    ID.AddPointer(ET.getAsOpaquePtr());
    ID.AddInteger(ArraySize.getZExtValue());
    ID.AddInteger(SizeMod);
    ID.AddInteger(TypeQuals);
  }
  static bool classof(const Type *T) {
    return T->getTypeClass() == ConstantArray;
  }
  static bool classof(const ConstantArrayType *) { return true; }
};

/// IncompleteArrayType - This class represents C arrays with an unspecified
/// size.  For example 'int A[]' has an IncompleteArrayType where the element
/// type is 'int' and the size is unspecified.
class IncompleteArrayType : public ArrayType {

  IncompleteArrayType(QualType et, QualType can,
                      ArraySizeModifier sm, unsigned tq)
    : ArrayType(IncompleteArray, et, can, sm, tq, 
                et->containsUnexpandedParameterPack()) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == IncompleteArray;
  }
  static bool classof(const IncompleteArrayType *) { return true; }

  friend class StmtIteratorBase;

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType(), getSizeModifier(),
            getIndexTypeCVRQualifiers());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, QualType ET,
                      ArraySizeModifier SizeMod, unsigned TypeQuals) {
    ID.AddPointer(ET.getAsOpaquePtr());
    ID.AddInteger(SizeMod);
    ID.AddInteger(TypeQuals);
  }
};

/// VariableArrayType - This class represents C arrays with a specified size
/// which is not an integer-constant-expression.  For example, 'int s[x+foo()]'.
/// Since the size expression is an arbitrary expression, we store it as such.
///
/// Note: VariableArrayType's aren't uniqued (since the expressions aren't) and
/// should not be: two lexically equivalent variable array types could mean
/// different things, for example, these variables do not have the same type
/// dynamically:
///
/// void foo(int x) {
///   int Y[x];
///   ++x;
///   int Z[x];
/// }
///
class VariableArrayType : public ArrayType {
  /// SizeExpr - An assignment expression. VLA's are only permitted within
  /// a function block.
  Stmt *SizeExpr;
  /// Brackets - The left and right array brackets.
  SourceRange Brackets;

  VariableArrayType(QualType et, QualType can, Expr *e,
                    ArraySizeModifier sm, unsigned tq,
                    SourceRange brackets)
    : ArrayType(VariableArray, et, can, sm, tq, 
                et->containsUnexpandedParameterPack()),
      SizeExpr((Stmt*) e), Brackets(brackets) {}
  friend class ASTContext;  // ASTContext creates these.

public:
  Expr *getSizeExpr() const {
    // We use C-style casts instead of cast<> here because we do not wish
    // to have a dependency of Type.h on Stmt.h/Expr.h.
    return (Expr*) SizeExpr;
  }
  SourceRange getBracketsRange() const { return Brackets; }
  SourceLocation getLBracketLoc() const { return Brackets.getBegin(); }
  SourceLocation getRBracketLoc() const { return Brackets.getEnd(); }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == VariableArray;
  }
  static bool classof(const VariableArrayType *) { return true; }

  friend class StmtIteratorBase;

  void Profile(llvm::FoldingSetNodeID &ID) {
    assert(0 && "Cannnot unique VariableArrayTypes.");
  }
};

/// DependentSizedArrayType - This type represents an array type in
/// C++ whose size is a value-dependent expression. For example:
///
/// \code
/// template<typename T, int Size>
/// class array {
///   T data[Size];
/// };
/// \endcode
///
/// For these types, we won't actually know what the array bound is
/// until template instantiation occurs, at which point this will
/// become either a ConstantArrayType or a VariableArrayType.
class DependentSizedArrayType : public ArrayType {
  const ASTContext &Context;

  /// \brief An assignment expression that will instantiate to the
  /// size of the array.
  ///
  /// The expression itself might be NULL, in which case the array
  /// type will have its size deduced from an initializer.
  Stmt *SizeExpr;

  /// Brackets - The left and right array brackets.
  SourceRange Brackets;

  DependentSizedArrayType(const ASTContext &Context, QualType et, QualType can,
                          Expr *e, ArraySizeModifier sm, unsigned tq,
                          SourceRange brackets);

  friend class ASTContext;  // ASTContext creates these.

public:
  Expr *getSizeExpr() const {
    // We use C-style casts instead of cast<> here because we do not wish
    // to have a dependency of Type.h on Stmt.h/Expr.h.
    return (Expr*) SizeExpr;
  }
  SourceRange getBracketsRange() const { return Brackets; }
  SourceLocation getLBracketLoc() const { return Brackets.getBegin(); }
  SourceLocation getRBracketLoc() const { return Brackets.getEnd(); }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == DependentSizedArray;
  }
  static bool classof(const DependentSizedArrayType *) { return true; }

  friend class StmtIteratorBase;


  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getElementType(),
            getSizeModifier(), getIndexTypeCVRQualifiers(), getSizeExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      QualType ET, ArraySizeModifier SizeMod,
                      unsigned TypeQuals, Expr *E);
};

/// DependentSizedExtVectorType - This type represent an extended vector type
/// where either the type or size is dependent. For example:
/// @code
/// template<typename T, int Size>
/// class vector {
///   typedef T __attribute__((ext_vector_type(Size))) type;
/// }
/// @endcode
class DependentSizedExtVectorType : public Type, public llvm::FoldingSetNode {
  const ASTContext &Context;
  Expr *SizeExpr;
  /// ElementType - The element type of the array.
  QualType ElementType;
  SourceLocation loc;

  DependentSizedExtVectorType(const ASTContext &Context, QualType ElementType,
                              QualType can, Expr *SizeExpr, SourceLocation loc);

  friend class ASTContext;

public:
  Expr *getSizeExpr() const { return SizeExpr; }
  QualType getElementType() const { return ElementType; }
  SourceLocation getAttributeLoc() const { return loc; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == DependentSizedExtVector;
  }
  static bool classof(const DependentSizedExtVectorType *) { return true; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getElementType(), getSizeExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      QualType ElementType, Expr *SizeExpr);
};


/// VectorType - GCC generic vector type. This type is created using
/// __attribute__((vector_size(n)), where "n" specifies the vector size in
/// bytes; or from an Altivec __vector or vector declaration.
/// Since the constructor takes the number of vector elements, the
/// client is responsible for converting the size into the number of elements.
class VectorType : public Type, public llvm::FoldingSetNode {
public:
  enum VectorKind {
    GenericVector,  // not a target-specific vector type
    AltiVecVector,  // is AltiVec vector
    AltiVecPixel,   // is AltiVec 'vector Pixel'
    AltiVecBool,    // is AltiVec 'vector bool ...'
    NeonVector,     // is ARM Neon vector
    NeonPolyVector  // is ARM Neon polynomial vector
  };
protected:
  /// ElementType - The element type of the vector.
  QualType ElementType;

  VectorType(QualType vecType, unsigned nElements, QualType canonType,
             VectorKind vecKind);
  
  VectorType(TypeClass tc, QualType vecType, unsigned nElements,
             QualType canonType, VectorKind vecKind);

  friend class ASTContext;  // ASTContext creates these.
  
public:

  QualType getElementType() const { return ElementType; }
  unsigned getNumElements() const { return VectorTypeBits.NumElements; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  VectorKind getVectorKind() const {
    return VectorKind(VectorTypeBits.VecKind);
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getElementType(), getNumElements(),
            getTypeClass(), getVectorKind());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ElementType,
                      unsigned NumElements, TypeClass TypeClass,
                      VectorKind VecKind) {
    ID.AddPointer(ElementType.getAsOpaquePtr());
    ID.AddInteger(NumElements);
    ID.AddInteger(TypeClass);
    ID.AddInteger(VecKind);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == Vector || T->getTypeClass() == ExtVector;
  }
  static bool classof(const VectorType *) { return true; }
};

/// ExtVectorType - Extended vector type. This type is created using
/// __attribute__((ext_vector_type(n)), where "n" is the number of elements.
/// Unlike vector_size, ext_vector_type is only allowed on typedef's. This
/// class enables syntactic extensions, like Vector Components for accessing
/// points, colors, and textures (modeled after OpenGL Shading Language).
class ExtVectorType : public VectorType {
  ExtVectorType(QualType vecType, unsigned nElements, QualType canonType) :
    VectorType(ExtVector, vecType, nElements, canonType, GenericVector) {}
  friend class ASTContext;  // ASTContext creates these.
public:
  static int getPointAccessorIdx(char c) {
    switch (c) {
    default: return -1;
    case 'x': return 0;
    case 'y': return 1;
    case 'z': return 2;
    case 'w': return 3;
    }
  }
  static int getNumericAccessorIdx(char c) {
    switch (c) {
      default: return -1;
      case '0': return 0;
      case '1': return 1;
      case '2': return 2;
      case '3': return 3;
      case '4': return 4;
      case '5': return 5;
      case '6': return 6;
      case '7': return 7;
      case '8': return 8;
      case '9': return 9;
      case 'A':
      case 'a': return 10;
      case 'B':
      case 'b': return 11;
      case 'C':
      case 'c': return 12;
      case 'D':
      case 'd': return 13;
      case 'E':
      case 'e': return 14;
      case 'F':
      case 'f': return 15;
    }
  }

  static int getAccessorIdx(char c) {
    if (int idx = getPointAccessorIdx(c)+1) return idx-1;
    return getNumericAccessorIdx(c);
  }

  bool isAccessorWithinNumElements(char c) const {
    if (int idx = getAccessorIdx(c)+1)
      return unsigned(idx-1) < getNumElements();
    return false;
  }
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == ExtVector;
  }
  static bool classof(const ExtVectorType *) { return true; }
};

/// FunctionType - C99 6.7.5.3 - Function Declarators.  This is the common base
/// class of FunctionNoProtoType and FunctionProtoType.
///
class FunctionType : public Type {
  // The type returned by the function.
  QualType ResultType;

 public:
  /// ExtInfo - A class which abstracts out some details necessary for
  /// making a call.
  ///
  /// It is not actually used directly for storing this information in
  /// a FunctionType, although FunctionType does currently use the
  /// same bit-pattern.
  ///
  // If you add a field (say Foo), other than the obvious places (both,
  // constructors, compile failures), what you need to update is
  // * Operator==
  // * getFoo
  // * withFoo
  // * functionType. Add Foo, getFoo.
  // * ASTContext::getFooType
  // * ASTContext::mergeFunctionTypes
  // * FunctionNoProtoType::Profile
  // * FunctionProtoType::Profile
  // * TypePrinter::PrintFunctionProto
  // * AST read and write
  // * Codegen
  class ExtInfo {
    // Feel free to rearrange or add bits, but if you go over 8,
    // you'll need to adjust both the Bits field below and
    // Type::FunctionTypeBitfields.

    //   |  CC  |noreturn|regparm
    //   |0 .. 2|   3    |4 ..  6
    enum { CallConvMask = 0x7 };
    enum { NoReturnMask = 0x8 };
    enum { RegParmMask = ~(CallConvMask | NoReturnMask),
           RegParmOffset = 4 };

    unsigned char Bits;

    ExtInfo(unsigned Bits) : Bits(static_cast<unsigned char>(Bits)) {}

    friend class FunctionType;

   public:
    // Constructor with no defaults. Use this when you know that you
    // have all the elements (when reading an AST file for example).
    ExtInfo(bool noReturn, unsigned regParm, CallingConv cc) {
      Bits = ((unsigned) cc) |
             (noReturn ? NoReturnMask : 0) |
             (regParm << RegParmOffset);
    }

    // Constructor with all defaults. Use when for example creating a
    // function know to use defaults.
    ExtInfo() : Bits(0) {}

    bool getNoReturn() const { return Bits & NoReturnMask; }
    unsigned getRegParm() const { return Bits >> RegParmOffset; }
    CallingConv getCC() const { return CallingConv(Bits & CallConvMask); }

    bool operator==(ExtInfo Other) const {
      return Bits == Other.Bits;
    }
    bool operator!=(ExtInfo Other) const {
      return Bits != Other.Bits;
    }

    // Note that we don't have setters. That is by design, use
    // the following with methods instead of mutating these objects.

    ExtInfo withNoReturn(bool noReturn) const {
      if (noReturn)
        return ExtInfo(Bits | NoReturnMask);
      else
        return ExtInfo(Bits & ~NoReturnMask);
    }

    ExtInfo withRegParm(unsigned RegParm) const {
      return ExtInfo((Bits & ~RegParmMask) | (RegParm << RegParmOffset));
    }

    ExtInfo withCallingConv(CallingConv cc) const {
      return ExtInfo((Bits & ~CallConvMask) | (unsigned) cc);
    }

    void Profile(llvm::FoldingSetNodeID &ID) const {
      ID.AddInteger(Bits);
    }
  };

protected:
  FunctionType(TypeClass tc, QualType res, bool variadic,
               unsigned typeQuals, RefQualifierKind RefQualifier,
               QualType Canonical, bool Dependent,
               bool VariablyModified, bool ContainsUnexpandedParameterPack, 
               ExtInfo Info)
    : Type(tc, Canonical, Dependent, VariablyModified, 
           ContainsUnexpandedParameterPack), 
      ResultType(res) {
    FunctionTypeBits.ExtInfo = Info.Bits;
    FunctionTypeBits.Variadic = variadic;
    FunctionTypeBits.TypeQuals = typeQuals;
    FunctionTypeBits.RefQualifier = static_cast<unsigned>(RefQualifier);
  }
  bool isVariadic() const { return FunctionTypeBits.Variadic; }
  unsigned getTypeQuals() const { return FunctionTypeBits.TypeQuals; }
  
  RefQualifierKind getRefQualifier() const {
    return static_cast<RefQualifierKind>(FunctionTypeBits.RefQualifier);
  }

public:

  QualType getResultType() const { return ResultType; }
  
  unsigned getRegParmType() const { return getExtInfo().getRegParm(); }
  bool getNoReturnAttr() const { return getExtInfo().getNoReturn(); }
  CallingConv getCallConv() const { return getExtInfo().getCC(); }
  ExtInfo getExtInfo() const { return ExtInfo(FunctionTypeBits.ExtInfo); }

  /// \brief Determine the type of an expression that calls a function of
  /// this type.
  QualType getCallResultType(ASTContext &Context) const { 
    return getResultType().getNonLValueExprType(Context);
  }

  static llvm::StringRef getNameForCallConv(CallingConv CC);

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto ||
           T->getTypeClass() == FunctionProto;
  }
  static bool classof(const FunctionType *) { return true; }
};

/// FunctionNoProtoType - Represents a K&R-style 'int foo()' function, which has
/// no information available about its arguments.
class FunctionNoProtoType : public FunctionType, public llvm::FoldingSetNode {
  FunctionNoProtoType(QualType Result, QualType Canonical, ExtInfo Info)
    : FunctionType(FunctionNoProto, Result, false, 0, RQ_None, Canonical,
                   /*Dependent=*/false, Result->isVariablyModifiedType(), 
                   /*ContainsUnexpandedParameterPack=*/false, Info) {}

  friend class ASTContext;  // ASTContext creates these.
  
public:
  // No additional state past what FunctionType provides.

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getResultType(), getExtInfo());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ResultType,
                      ExtInfo Info) {
    Info.Profile(ID);
    ID.AddPointer(ResultType.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto;
  }
  static bool classof(const FunctionNoProtoType *) { return true; }
};

/// FunctionProtoType - Represents a prototype with argument type info, e.g.
/// 'int foo(int)' or 'int foo(void)'.  'void' is represented as having no
/// arguments, not as having a single void argument. Such a type can have an
/// exception specification, but this specification is not part of the canonical
/// type.
class FunctionProtoType : public FunctionType, public llvm::FoldingSetNode {
public:
  /// ExtProtoInfo - Extra information about a function prototype.
  struct ExtProtoInfo {
    ExtProtoInfo() :
      Variadic(false), HasExceptionSpec(false), HasAnyExceptionSpec(false),
      TypeQuals(0), RefQualifier(RQ_None), NumExceptions(0), Exceptions(0) {}

    FunctionType::ExtInfo ExtInfo;
    bool Variadic;
    bool HasExceptionSpec;
    bool HasAnyExceptionSpec;
    unsigned char TypeQuals;
    RefQualifierKind RefQualifier;
    unsigned NumExceptions;
    const QualType *Exceptions;
  };

private:
  /// \brief Determine whether there are any argument types that
  /// contain an unexpanded parameter pack.
  static bool containsAnyUnexpandedParameterPack(const QualType *ArgArray, 
                                                 unsigned numArgs) {
    for (unsigned Idx = 0; Idx < numArgs; ++Idx)
      if (ArgArray[Idx]->containsUnexpandedParameterPack())
        return true;

    return false;
  }

  FunctionProtoType(QualType result, const QualType *args, unsigned numArgs,
                    QualType canonical, const ExtProtoInfo &epi);

  /// NumArgs - The number of arguments this function has, not counting '...'.
  unsigned NumArgs : 20;

  /// NumExceptions - The number of types in the exception spec, if any.
  unsigned NumExceptions : 10;

  /// HasExceptionSpec - Whether this function has an exception spec at all.
  unsigned HasExceptionSpec : 1;

  /// HasAnyExceptionSpec - Whether this function has a throw(...) spec.
  unsigned HasAnyExceptionSpec : 1;

  /// ArgInfo - There is an variable size array after the class in memory that
  /// holds the argument types.

  /// Exceptions - There is another variable size array after ArgInfo that
  /// holds the exception types.

  friend class ASTContext;  // ASTContext creates these.

public:
  unsigned getNumArgs() const { return NumArgs; }
  QualType getArgType(unsigned i) const {
    assert(i < NumArgs && "Invalid argument number!");
    return arg_type_begin()[i];
  }

  ExtProtoInfo getExtProtoInfo() const {
    ExtProtoInfo EPI;
    EPI.ExtInfo = getExtInfo();
    EPI.Variadic = isVariadic();
    EPI.HasExceptionSpec = hasExceptionSpec();
    EPI.HasAnyExceptionSpec = hasAnyExceptionSpec();
    EPI.TypeQuals = static_cast<unsigned char>(getTypeQuals());
    EPI.RefQualifier = getRefQualifier();
    EPI.NumExceptions = NumExceptions;
    EPI.Exceptions = exception_begin();
    return EPI;
  }

  bool hasExceptionSpec() const { return HasExceptionSpec; }
  bool hasAnyExceptionSpec() const { return HasAnyExceptionSpec; }
  unsigned getNumExceptions() const { return NumExceptions; }
  QualType getExceptionType(unsigned i) const {
    assert(i < NumExceptions && "Invalid exception number!");
    return exception_begin()[i];
  }
  bool hasEmptyExceptionSpec() const {
    return hasExceptionSpec() && !hasAnyExceptionSpec() &&
      getNumExceptions() == 0;
  }

  /// \brief Returns true if this function type has a non-throwing
  /// exception-specification.
  bool hasNonThrowingExceptionSpec() const {
    // FIXME: this needs to be updated for C++0x.
    return hasEmptyExceptionSpec();
  }

  using FunctionType::isVariadic;
  
  /// \brief Determines whether this function prototype contains a
  /// parameter pack at the end.
  ///
  /// A function template whose last parameter is a parameter pack can be
  /// called with an arbitrary number of arguments, much like a variadic
  /// function. However, 
  bool isTemplateVariadic() const;
  
  unsigned getTypeQuals() const { return FunctionType::getTypeQuals(); }

  
  /// \brief Retrieve the ref-qualifier associated with this function type.
  RefQualifierKind getRefQualifier() const {
    return FunctionType::getRefQualifier();
  }
  
  typedef const QualType *arg_type_iterator;
  arg_type_iterator arg_type_begin() const {
    return reinterpret_cast<const QualType *>(this+1);
  }
  arg_type_iterator arg_type_end() const { return arg_type_begin()+NumArgs; }

  typedef const QualType *exception_iterator;
  exception_iterator exception_begin() const {
    // exceptions begin where arguments end
    return arg_type_end();
  }
  exception_iterator exception_end() const {
    return exception_begin() + NumExceptions;
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionProto;
  }
  static bool classof(const FunctionProtoType *) { return true; }

  void Profile(llvm::FoldingSetNodeID &ID);
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Result,
                      arg_type_iterator ArgTys, unsigned NumArgs,
                      const ExtProtoInfo &EPI);
};


/// \brief Represents the dependent type named by a dependently-scoped
/// typename using declaration, e.g.
///   using typename Base<T>::foo;
/// Template instantiation turns these into the underlying type.
class UnresolvedUsingType : public Type {
  UnresolvedUsingTypenameDecl *Decl;

  UnresolvedUsingType(const UnresolvedUsingTypenameDecl *D)
    : Type(UnresolvedUsing, QualType(), true, false, 
           /*ContainsUnexpandedParameterPack=*/false),
      Decl(const_cast<UnresolvedUsingTypenameDecl*>(D)) {}
  friend class ASTContext; // ASTContext creates these.
public:

  UnresolvedUsingTypenameDecl *getDecl() const { return Decl; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == UnresolvedUsing;
  }
  static bool classof(const UnresolvedUsingType *) { return true; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    return Profile(ID, Decl);
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      UnresolvedUsingTypenameDecl *D) {
    ID.AddPointer(D);
  }
};


class TypedefType : public Type {
  TypedefDecl *Decl;
protected:
  TypedefType(TypeClass tc, const TypedefDecl *D, QualType can)
    : Type(tc, can, can->isDependentType(), can->isVariablyModifiedType(), 
           /*ContainsUnexpandedParameterPack=*/false),
      Decl(const_cast<TypedefDecl*>(D)) {
    assert(!isa<TypedefType>(can) && "Invalid canonical type");
  }
  friend class ASTContext;  // ASTContext creates these.
public:

  TypedefDecl *getDecl() const { return Decl; }

  bool isSugared() const { return true; }
  QualType desugar() const;

  static bool classof(const Type *T) { return T->getTypeClass() == Typedef; }
  static bool classof(const TypedefType *) { return true; }
};

/// TypeOfExprType (GCC extension).
class TypeOfExprType : public Type {
  Expr *TOExpr;

protected:
  TypeOfExprType(Expr *E, QualType can = QualType());
  friend class ASTContext;  // ASTContext creates these.
public:
  Expr *getUnderlyingExpr() const { return TOExpr; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const;

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return true; }

  static bool classof(const Type *T) { return T->getTypeClass() == TypeOfExpr; }
  static bool classof(const TypeOfExprType *) { return true; }
};

/// \brief Internal representation of canonical, dependent
/// typeof(expr) types.
///
/// This class is used internally by the ASTContext to manage
/// canonical, dependent types, only. Clients will only see instances
/// of this class via TypeOfExprType nodes.
class DependentTypeOfExprType
  : public TypeOfExprType, public llvm::FoldingSetNode {
  const ASTContext &Context;

public:
  DependentTypeOfExprType(const ASTContext &Context, Expr *E)
    : TypeOfExprType(E), Context(Context) { }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getUnderlyingExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      Expr *E);
};

/// TypeOfType (GCC extension).
class TypeOfType : public Type {
  QualType TOType;
  TypeOfType(QualType T, QualType can)
    : Type(TypeOf, can, T->isDependentType(), T->isVariablyModifiedType(), 
           T->containsUnexpandedParameterPack()), 
      TOType(T) {
    assert(!isa<TypedefType>(can) && "Invalid canonical type");
  }
  friend class ASTContext;  // ASTContext creates these.
public:
  QualType getUnderlyingType() const { return TOType; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const { return getUnderlyingType(); }

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return true; }

  static bool classof(const Type *T) { return T->getTypeClass() == TypeOf; }
  static bool classof(const TypeOfType *) { return true; }
};

/// DecltypeType (C++0x)
class DecltypeType : public Type {
  Expr *E;

  // FIXME: We could get rid of UnderlyingType if we wanted to: We would have to
  // Move getDesugaredType to ASTContext so that it can call getDecltypeForExpr
  // from it.
  QualType UnderlyingType;

protected:
  DecltypeType(Expr *E, QualType underlyingType, QualType can = QualType());
  friend class ASTContext;  // ASTContext creates these.
public:
  Expr *getUnderlyingExpr() const { return E; }
  QualType getUnderlyingType() const { return UnderlyingType; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const { return getUnderlyingType(); }

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return !isDependentType(); }

  static bool classof(const Type *T) { return T->getTypeClass() == Decltype; }
  static bool classof(const DecltypeType *) { return true; }
};

/// \brief Internal representation of canonical, dependent
/// decltype(expr) types.
///
/// This class is used internally by the ASTContext to manage
/// canonical, dependent types, only. Clients will only see instances
/// of this class via DecltypeType nodes.
class DependentDecltypeType : public DecltypeType, public llvm::FoldingSetNode {
  const ASTContext &Context;

public:
  DependentDecltypeType(const ASTContext &Context, Expr *E);

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getUnderlyingExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      Expr *E);
};

class TagType : public Type {
  /// Stores the TagDecl associated with this type. The decl may point to any
  /// TagDecl that declares the entity.
  TagDecl * decl;

protected:
  TagType(TypeClass TC, const TagDecl *D, QualType can);

public:
  TagDecl *getDecl() const;

  /// @brief Determines whether this type is in the process of being
  /// defined.
  bool isBeingDefined() const;

  static bool classof(const Type *T) {
    return T->getTypeClass() >= TagFirst && T->getTypeClass() <= TagLast;
  }
  static bool classof(const TagType *) { return true; }
  static bool classof(const RecordType *) { return true; }
  static bool classof(const EnumType *) { return true; }
};

/// RecordType - This is a helper class that allows the use of isa/cast/dyncast
/// to detect TagType objects of structs/unions/classes.
class RecordType : public TagType {
protected:
  explicit RecordType(const RecordDecl *D)
    : TagType(Record, reinterpret_cast<const TagDecl*>(D), QualType()) { }
  explicit RecordType(TypeClass TC, RecordDecl *D)
    : TagType(TC, reinterpret_cast<const TagDecl*>(D), QualType()) { }
  friend class ASTContext;   // ASTContext creates these.
public:

  RecordDecl *getDecl() const {
    return reinterpret_cast<RecordDecl*>(TagType::getDecl());
  }

  // FIXME: This predicate is a helper to QualType/Type. It needs to
  // recursively check all fields for const-ness. If any field is declared
  // const, it needs to return false.
  bool hasConstFields() const { return false; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const TagType *T);
  static bool classof(const Type *T) {
    return isa<TagType>(T) && classof(cast<TagType>(T));
  }
  static bool classof(const RecordType *) { return true; }
};

/// EnumType - This is a helper class that allows the use of isa/cast/dyncast
/// to detect TagType objects of enums.
class EnumType : public TagType {
  explicit EnumType(const EnumDecl *D)
    : TagType(Enum, reinterpret_cast<const TagDecl*>(D), QualType()) { }
  friend class ASTContext;   // ASTContext creates these.
public:

  EnumDecl *getDecl() const {
    return reinterpret_cast<EnumDecl*>(TagType::getDecl());
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const TagType *T);
  static bool classof(const Type *T) {
    return isa<TagType>(T) && classof(cast<TagType>(T));
  }
  static bool classof(const EnumType *) { return true; }
};

/// AttributedType - An attributed type is a type to which a type
/// attribute has been applied.  The "modified type" is the
/// fully-sugared type to which the attributed type was applied;
/// generally it is not canonically equivalent to the attributed type.
/// The "equivalent type" is the minimally-desugared type which the
/// type is canonically equivalent to.
///
/// For example, in the following attributed type:
///     int32_t __attribute__((vector_size(16)))
///   - the modified type is the TypedefType for int32_t
///   - the equivalent type is VectorType(16, int32_t)
///   - the canonical type is VectorType(16, int)
class AttributedType : public Type, public llvm::FoldingSetNode {
public:
  // It is really silly to have yet another attribute-kind enum, but
  // clang::attr::Kind doesn't currently cover the pure type attrs.
  enum Kind {
    // Expression operand.
    attr_address_space,
    attr_regparm,
    attr_vector_size,
    attr_neon_vector_type,
    attr_neon_polyvector_type,

    FirstExprOperandKind = attr_address_space,
    LastExprOperandKind = attr_neon_polyvector_type,

    // Enumerated operand (string or keyword).
    attr_objc_gc,

    FirstEnumOperandKind = attr_objc_gc,
    LastEnumOperandKind = attr_objc_gc,

    // No operand.
    attr_noreturn,
    attr_cdecl,
    attr_fastcall,
    attr_stdcall,
    attr_thiscall,
    attr_pascal
  };

private:
  QualType ModifiedType;
  QualType EquivalentType;

  friend class ASTContext; // creates these

  AttributedType(QualType canon, Kind attrKind,
                 QualType modified, QualType equivalent)
    : Type(Attributed, canon, canon->isDependentType(),
           canon->isVariablyModifiedType(),
           canon->containsUnexpandedParameterPack()),
      ModifiedType(modified), EquivalentType(equivalent) {
    AttributedTypeBits.AttrKind = attrKind;
  }

public:
  Kind getAttrKind() const {
    return static_cast<Kind>(AttributedTypeBits.AttrKind);
  }

  QualType getModifiedType() const { return ModifiedType; }
  QualType getEquivalentType() const { return EquivalentType; }

  bool isSugared() const { return true; }
  QualType desugar() const { return getEquivalentType(); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getAttrKind(), ModifiedType, EquivalentType);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, Kind attrKind,
                      QualType modified, QualType equivalent) {
    ID.AddInteger(attrKind);
    ID.AddPointer(modified.getAsOpaquePtr());
    ID.AddPointer(equivalent.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == Attributed;
  }
  static bool classof(const AttributedType *T) { return true; }
};

class TemplateTypeParmType : public Type, public llvm::FoldingSetNode {
  unsigned Depth : 15;
  unsigned ParameterPack : 1;
  unsigned Index : 16;
  IdentifierInfo *Name;

  TemplateTypeParmType(unsigned D, unsigned I, bool PP, IdentifierInfo *N,
                       QualType Canon)
    : Type(TemplateTypeParm, Canon, /*Dependent=*/true,
           /*VariablyModified=*/false, PP),
      Depth(D), ParameterPack(PP), Index(I), Name(N) { }

  TemplateTypeParmType(unsigned D, unsigned I, bool PP)
    : Type(TemplateTypeParm, QualType(this, 0), /*Dependent=*/true,
           /*VariablyModified=*/false, PP),
      Depth(D), ParameterPack(PP), Index(I), Name(0) { }

  friend class ASTContext;  // ASTContext creates these

public:
  unsigned getDepth() const { return Depth; }
  unsigned getIndex() const { return Index; }
  bool isParameterPack() const { return ParameterPack; }
  IdentifierInfo *getName() const { return Name; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Depth, Index, ParameterPack, Name);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, unsigned Depth,
                      unsigned Index, bool ParameterPack,
                      IdentifierInfo *Name) {
    ID.AddInteger(Depth);
    ID.AddInteger(Index);
    ID.AddBoolean(ParameterPack);
    ID.AddPointer(Name);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == TemplateTypeParm;
  }
  static bool classof(const TemplateTypeParmType *T) { return true; }
};

/// \brief Represents the result of substituting a type for a template
/// type parameter.
///
/// Within an instantiated template, all template type parameters have
/// been replaced with these.  They are used solely to record that a
/// type was originally written as a template type parameter;
/// therefore they are never canonical.
class SubstTemplateTypeParmType : public Type, public llvm::FoldingSetNode {
  // The original type parameter.
  const TemplateTypeParmType *Replaced;

  SubstTemplateTypeParmType(const TemplateTypeParmType *Param, QualType Canon)
    : Type(SubstTemplateTypeParm, Canon, Canon->isDependentType(),
           Canon->isVariablyModifiedType(),
           Canon->containsUnexpandedParameterPack()),
      Replaced(Param) { }

  friend class ASTContext;

public:
  IdentifierInfo *getName() const { return Replaced->getName(); }

  /// Gets the template parameter that was substituted for.
  const TemplateTypeParmType *getReplacedParameter() const {
    return Replaced;
  }

  /// Gets the type that was substituted for the template
  /// parameter.
  QualType getReplacementType() const {
    return getCanonicalTypeInternal();
  }

  bool isSugared() const { return true; }
  QualType desugar() const { return getReplacementType(); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getReplacedParameter(), getReplacementType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const TemplateTypeParmType *Replaced,
                      QualType Replacement) {
    ID.AddPointer(Replaced);
    ID.AddPointer(Replacement.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == SubstTemplateTypeParm;
  }
  static bool classof(const SubstTemplateTypeParmType *T) { return true; }
};

/// \brief Represents the result of substituting a set of types for a template
/// type parameter pack.
///
/// When a pack expansion in the source code contains multiple parameter packs
/// and those parameter packs correspond to different levels of template
/// parameter lists, this type node is used to represent a template type 
/// parameter pack from an outer level, which has already had its argument pack
/// substituted but that still lives within a pack expansion that itself
/// could not be instantiated. When actually performing a substitution into
/// that pack expansion (e.g., when all template parameters have corresponding
/// arguments), this type will be replaced with the \c SubstTemplateTypeParmType
/// at the current pack substitution index.
class SubstTemplateTypeParmPackType : public Type, public llvm::FoldingSetNode {
  /// \brief The original type parameter.
  const TemplateTypeParmType *Replaced;
  
  /// \brief A pointer to the set of template arguments that this
  /// parameter pack is instantiated with.
  const TemplateArgument *Arguments;
  
  /// \brief The number of template arguments in \c Arguments.
  unsigned NumArguments;
  
  SubstTemplateTypeParmPackType(const TemplateTypeParmType *Param, 
                                QualType Canon,
                                const TemplateArgument &ArgPack);
  
  friend class ASTContext;
  
public:
  IdentifierInfo *getName() const { return Replaced->getName(); }
  
  /// Gets the template parameter that was substituted for.
  const TemplateTypeParmType *getReplacedParameter() const {
    return Replaced;
  }
  
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }
  
  TemplateArgument getArgumentPack() const;
  
  void Profile(llvm::FoldingSetNodeID &ID);
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const TemplateTypeParmType *Replaced,
                      const TemplateArgument &ArgPack);
  
  static bool classof(const Type *T) {
    return T->getTypeClass() == SubstTemplateTypeParmPack;
  }
  static bool classof(const SubstTemplateTypeParmPackType *T) { return true; }
};

/// \brief Represents a C++0x auto type.
///
/// These types are usually a placeholder for a deduced type. However, within
/// templates and before the initializer is attached, there is no deduced type
/// and an auto type is type-dependent and canonical.
class AutoType : public Type, public llvm::FoldingSetNode {
  AutoType(QualType DeducedType)
    : Type(Auto, DeducedType.isNull() ? QualType(this, 0) : DeducedType,
           /*Dependent=*/DeducedType.isNull(),
           /*VariablyModified=*/false, /*ContainsParameterPack=*/false) {
    assert((DeducedType.isNull() || !DeducedType->isDependentType()) &&
           "deduced a dependent type for auto");
  }

  friend class ASTContext;  // ASTContext creates these

public:
  bool isSugared() const { return isDeduced(); }
  QualType desugar() const { return getCanonicalTypeInternal(); }

  QualType getDeducedType() const {
    return isDeduced() ? getCanonicalTypeInternal() : QualType();
  }
  bool isDeduced() const {
    return !isDependentType();
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getDeducedType());
  }

  static void Profile(llvm::FoldingSetNodeID &ID,
                      QualType Deduced) {
    ID.AddPointer(Deduced.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == Auto;
  }
  static bool classof(const AutoType *T) { return true; }
};

/// \brief Represents the type of a template specialization as written
/// in the source code.
///
/// Template specialization types represent the syntactic form of a
/// template-id that refers to a type, e.g., @c vector<int>. Some
/// template specialization types are syntactic sugar, whose canonical
/// type will point to some other type node that represents the
/// instantiation or class template specialization. For example, a
/// class template specialization type of @c vector<int> will refer to
/// a tag type for the instantiation
/// @c std::vector<int, std::allocator<int>>.
///
/// Other template specialization types, for which the template name
/// is dependent, may be canonical types. These types are always
/// dependent.
class TemplateSpecializationType
  : public Type, public llvm::FoldingSetNode {
  /// \brief The name of the template being specialized.
  TemplateName Template;

  /// \brief - The number of template arguments named in this class
  /// template specialization.
  unsigned NumArgs;

  TemplateSpecializationType(TemplateName T,
                             const TemplateArgument *Args,
                             unsigned NumArgs, QualType Canon);

  friend class ASTContext;  // ASTContext creates these

public:
  /// \brief Determine whether any of the given template arguments are
  /// dependent.
  static bool anyDependentTemplateArguments(const TemplateArgument *Args,
                                            unsigned NumArgs);

  static bool anyDependentTemplateArguments(const TemplateArgumentLoc *Args,
                                            unsigned NumArgs);

  static bool anyDependentTemplateArguments(const TemplateArgumentListInfo &);

  /// \brief Print a template argument list, including the '<' and '>'
  /// enclosing the template arguments.
  static std::string PrintTemplateArgumentList(const TemplateArgument *Args,
                                               unsigned NumArgs,
                                               const PrintingPolicy &Policy,
                                               bool SkipBrackets = false);

  static std::string PrintTemplateArgumentList(const TemplateArgumentLoc *Args,
                                               unsigned NumArgs,
                                               const PrintingPolicy &Policy);

  static std::string PrintTemplateArgumentList(const TemplateArgumentListInfo &,
                                               const PrintingPolicy &Policy);

  /// True if this template specialization type matches a current
  /// instantiation in the context in which it is found.
  bool isCurrentInstantiation() const {
    return isa<InjectedClassNameType>(getCanonicalTypeInternal());
  }

  typedef const TemplateArgument * iterator;

  iterator begin() const { return getArgs(); }
  iterator end() const; // defined inline in TemplateBase.h

  /// \brief Retrieve the name of the template that we are specializing.
  TemplateName getTemplateName() const { return Template; }

  /// \brief Retrieve the template arguments.
  const TemplateArgument *getArgs() const {
    return reinterpret_cast<const TemplateArgument *>(this + 1);
  }

  /// \brief Retrieve the number of template arguments.
  unsigned getNumArgs() const { return NumArgs; }

  /// \brief Retrieve a specific template argument as a type.
  /// \precondition @c isArgType(Arg)
  const TemplateArgument &getArg(unsigned Idx) const; // in TemplateBase.h

  bool isSugared() const {
    return !isDependentType() || isCurrentInstantiation();
  }
  QualType desugar() const { return getCanonicalTypeInternal(); }

  void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Ctx) {
    Profile(ID, Template, getArgs(), NumArgs, Ctx);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, TemplateName T,
                      const TemplateArgument *Args,
                      unsigned NumArgs,
                      const ASTContext &Context);

  static bool classof(const Type *T) {
    return T->getTypeClass() == TemplateSpecialization;
  }
  static bool classof(const TemplateSpecializationType *T) { return true; }
};

/// \brief The injected class name of a C++ class template or class
/// template partial specialization.  Used to record that a type was
/// spelled with a bare identifier rather than as a template-id; the
/// equivalent for non-templated classes is just RecordType.
///
/// Injected class name types are always dependent.  Template
/// instantiation turns these into RecordTypes.
///
/// Injected class name types are always canonical.  This works
/// because it is impossible to compare an injected class name type
/// with the corresponding non-injected template type, for the same
/// reason that it is impossible to directly compare template
/// parameters from different dependent contexts: injected class name
/// types can only occur within the scope of a particular templated
/// declaration, and within that scope every template specialization
/// will canonicalize to the injected class name (when appropriate
/// according to the rules of the language).
class InjectedClassNameType : public Type {
  CXXRecordDecl *Decl;

  /// The template specialization which this type represents.
  /// For example, in
  ///   template <class T> class A { ... };
  /// this is A<T>, whereas in
  ///   template <class X, class Y> class A<B<X,Y> > { ... };
  /// this is A<B<X,Y> >.
  ///
  /// It is always unqualified, always a template specialization type,
  /// and always dependent.
  QualType InjectedType;

  friend class ASTContext; // ASTContext creates these.
  friend class ASTReader; // FIXME: ASTContext::getInjectedClassNameType is not
                          // currently suitable for AST reading, too much
                          // interdependencies.
  InjectedClassNameType(CXXRecordDecl *D, QualType TST)
    : Type(InjectedClassName, QualType(), /*Dependent=*/true,
           /*VariablyModified=*/false, 
           /*ContainsUnexpandedParameterPack=*/false),
      Decl(D), InjectedType(TST) {
    assert(isa<TemplateSpecializationType>(TST));
    assert(!TST.hasQualifiers());
    assert(TST->isDependentType());
  }

public:
  QualType getInjectedSpecializationType() const { return InjectedType; }
  const TemplateSpecializationType *getInjectedTST() const {
    return cast<TemplateSpecializationType>(InjectedType.getTypePtr());
  }

  CXXRecordDecl *getDecl() const;

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == InjectedClassName;
  }
  static bool classof(const InjectedClassNameType *T) { return true; }
};

/// \brief The kind of a tag type.
enum TagTypeKind {
  /// \brief The "struct" keyword.
  TTK_Struct,
  /// \brief The "union" keyword.
  TTK_Union,
  /// \brief The "class" keyword.
  TTK_Class,
  /// \brief The "enum" keyword.
  TTK_Enum
};

/// \brief The elaboration keyword that precedes a qualified type name or
/// introduces an elaborated-type-specifier.
enum ElaboratedTypeKeyword {
  /// \brief The "struct" keyword introduces the elaborated-type-specifier.
  ETK_Struct,
  /// \brief The "union" keyword introduces the elaborated-type-specifier.
  ETK_Union,
  /// \brief The "class" keyword introduces the elaborated-type-specifier.
  ETK_Class,
  /// \brief The "enum" keyword introduces the elaborated-type-specifier.
  ETK_Enum,
  /// \brief The "typename" keyword precedes the qualified type name, e.g.,
  /// \c typename T::type.
  ETK_Typename,
  /// \brief No keyword precedes the qualified type name.
  ETK_None
};

/// A helper class for Type nodes having an ElaboratedTypeKeyword.
/// The keyword in stored in the free bits of the base class.
/// Also provides a few static helpers for converting and printing
/// elaborated type keyword and tag type kind enumerations.
class TypeWithKeyword : public Type {
protected:
  TypeWithKeyword(ElaboratedTypeKeyword Keyword, TypeClass tc,
                  QualType Canonical, bool Dependent, bool VariablyModified, 
                  bool ContainsUnexpandedParameterPack)
  : Type(tc, Canonical, Dependent, VariablyModified, 
         ContainsUnexpandedParameterPack) {
    TypeWithKeywordBits.Keyword = Keyword;
  }

public:
  ElaboratedTypeKeyword getKeyword() const {
    return static_cast<ElaboratedTypeKeyword>(TypeWithKeywordBits.Keyword);
  }

  /// getKeywordForTypeSpec - Converts a type specifier (DeclSpec::TST)
  /// into an elaborated type keyword.
  static ElaboratedTypeKeyword getKeywordForTypeSpec(unsigned TypeSpec);

  /// getTagTypeKindForTypeSpec - Converts a type specifier (DeclSpec::TST)
  /// into a tag type kind.  It is an error to provide a type specifier
  /// which *isn't* a tag kind here.
  static TagTypeKind getTagTypeKindForTypeSpec(unsigned TypeSpec);

  /// getKeywordForTagDeclKind - Converts a TagTypeKind into an
  /// elaborated type keyword.
  static ElaboratedTypeKeyword getKeywordForTagTypeKind(TagTypeKind Tag);

  /// getTagTypeKindForKeyword - Converts an elaborated type keyword into
  // a TagTypeKind. It is an error to provide an elaborated type keyword
  /// which *isn't* a tag kind here.
  static TagTypeKind getTagTypeKindForKeyword(ElaboratedTypeKeyword Keyword);

  static bool KeywordIsTagTypeKind(ElaboratedTypeKeyword Keyword);

  static const char *getKeywordName(ElaboratedTypeKeyword Keyword);

  static const char *getTagTypeKindName(TagTypeKind Kind) {
    return getKeywordName(getKeywordForTagTypeKind(Kind));
  }

  class CannotCastToThisType {};
  static CannotCastToThisType classof(const Type *);
};

/// \brief Represents a type that was referred to using an elaborated type
/// keyword, e.g., struct S, or via a qualified name, e.g., N::M::type,
/// or both.
///
/// This type is used to keep track of a type name as written in the
/// source code, including tag keywords and any nested-name-specifiers.
/// The type itself is always "sugar", used to express what was written
/// in the source code but containing no additional semantic information.
class ElaboratedType : public TypeWithKeyword, public llvm::FoldingSetNode {

  /// \brief The nested name specifier containing the qualifier.
  NestedNameSpecifier *NNS;

  /// \brief The type that this qualified name refers to.
  QualType NamedType;

  ElaboratedType(ElaboratedTypeKeyword Keyword, NestedNameSpecifier *NNS,
                 QualType NamedType, QualType CanonType)
    : TypeWithKeyword(Keyword, Elaborated, CanonType,
                      NamedType->isDependentType(),
                      NamedType->isVariablyModifiedType(),
                      NamedType->containsUnexpandedParameterPack()),
      NNS(NNS), NamedType(NamedType) {
    assert(!(Keyword == ETK_None && NNS == 0) &&
           "ElaboratedType cannot have elaborated type keyword "
           "and name qualifier both null.");
  }

  friend class ASTContext;  // ASTContext creates these

public:
  ~ElaboratedType();

  /// \brief Retrieve the qualification on this type.
  NestedNameSpecifier *getQualifier() const { return NNS; }

  /// \brief Retrieve the type named by the qualified-id.
  QualType getNamedType() const { return NamedType; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const { return getNamedType(); }

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const { return true; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getKeyword(), NNS, NamedType);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, ElaboratedTypeKeyword Keyword,
                      NestedNameSpecifier *NNS, QualType NamedType) {
    ID.AddInteger(Keyword);
    ID.AddPointer(NNS);
    NamedType.Profile(ID);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == Elaborated;
  }
  static bool classof(const ElaboratedType *T) { return true; }
};

/// \brief Represents a qualified type name for which the type name is
/// dependent. 
///
/// DependentNameType represents a class of dependent types that involve a 
/// dependent nested-name-specifier (e.g., "T::") followed by a (dependent) 
/// name of a type. The DependentNameType may start with a "typename" (for a
/// typename-specifier), "class", "struct", "union", or "enum" (for a 
/// dependent elaborated-type-specifier), or nothing (in contexts where we
/// know that we must be referring to a type, e.g., in a base class specifier).
class DependentNameType : public TypeWithKeyword, public llvm::FoldingSetNode {

  /// \brief The nested name specifier containing the qualifier.
  NestedNameSpecifier *NNS;

  /// \brief The type that this typename specifier refers to.
  const IdentifierInfo *Name;

  DependentNameType(ElaboratedTypeKeyword Keyword, NestedNameSpecifier *NNS, 
                    const IdentifierInfo *Name, QualType CanonType)
    : TypeWithKeyword(Keyword, DependentName, CanonType, /*Dependent=*/true,
                      /*VariablyModified=*/false,
                      NNS->containsUnexpandedParameterPack()),
      NNS(NNS), Name(Name) {
    assert(NNS->isDependent() &&
           "DependentNameType requires a dependent nested-name-specifier");
  }

  friend class ASTContext;  // ASTContext creates these

public:
  /// \brief Retrieve the qualification on this type.
  NestedNameSpecifier *getQualifier() const { return NNS; }

  /// \brief Retrieve the type named by the typename specifier as an
  /// identifier.
  ///
  /// This routine will return a non-NULL identifier pointer when the
  /// form of the original typename was terminated by an identifier,
  /// e.g., "typename T::type".
  const IdentifierInfo *getIdentifier() const {
    return Name;
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getKeyword(), NNS, Name);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, ElaboratedTypeKeyword Keyword,
                      NestedNameSpecifier *NNS, const IdentifierInfo *Name) {
    ID.AddInteger(Keyword);
    ID.AddPointer(NNS);
    ID.AddPointer(Name);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == DependentName;
  }
  static bool classof(const DependentNameType *T) { return true; }
};

/// DependentTemplateSpecializationType - Represents a template
/// specialization type whose template cannot be resolved, e.g.
///   A<T>::template B<T>
class DependentTemplateSpecializationType :
  public TypeWithKeyword, public llvm::FoldingSetNode {

  /// \brief The nested name specifier containing the qualifier.
  NestedNameSpecifier *NNS;

  /// \brief The identifier of the template.
  const IdentifierInfo *Name;

  /// \brief - The number of template arguments named in this class
  /// template specialization.
  unsigned NumArgs;

  const TemplateArgument *getArgBuffer() const {
    return reinterpret_cast<const TemplateArgument*>(this+1);
  }
  TemplateArgument *getArgBuffer() {
    return reinterpret_cast<TemplateArgument*>(this+1);
  }

  DependentTemplateSpecializationType(ElaboratedTypeKeyword Keyword,
                                      NestedNameSpecifier *NNS,
                                      const IdentifierInfo *Name,
                                      unsigned NumArgs,
                                      const TemplateArgument *Args,
                                      QualType Canon);

  friend class ASTContext;  // ASTContext creates these

public:
  NestedNameSpecifier *getQualifier() const { return NNS; }
  const IdentifierInfo *getIdentifier() const { return Name; }

  /// \brief Retrieve the template arguments.
  const TemplateArgument *getArgs() const {
    return getArgBuffer();
  }

  /// \brief Retrieve the number of template arguments.
  unsigned getNumArgs() const { return NumArgs; }

  const TemplateArgument &getArg(unsigned Idx) const; // in TemplateBase.h

  typedef const TemplateArgument * iterator;
  iterator begin() const { return getArgs(); }
  iterator end() const; // inline in TemplateBase.h

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context) {
    Profile(ID, Context, getKeyword(), NNS, Name, NumArgs, getArgs());
  }

  static void Profile(llvm::FoldingSetNodeID &ID,
                      const ASTContext &Context,
                      ElaboratedTypeKeyword Keyword,
                      NestedNameSpecifier *Qualifier,
                      const IdentifierInfo *Name,
                      unsigned NumArgs,
                      const TemplateArgument *Args);

  static bool classof(const Type *T) {
    return T->getTypeClass() == DependentTemplateSpecialization;
  }
  static bool classof(const DependentTemplateSpecializationType *T) {
    return true;
  }  
};

/// \brief Represents a pack expansion of types.
///
/// Pack expansions are part of C++0x variadic templates. A pack
/// expansion contains a pattern, which itself contains one or more
/// "unexpanded" parameter packs. When instantiated, a pack expansion
/// produces a series of types, each instantiated from the pattern of
/// the expansion, where the Ith instantiation of the pattern uses the
/// Ith arguments bound to each of the unexpanded parameter packs. The
/// pack expansion is considered to "expand" these unexpanded
/// parameter packs.
///
/// \code
/// template<typename ...Types> struct tuple;
///
/// template<typename ...Types> 
/// struct tuple_of_references {
///   typedef tuple<Types&...> type;
/// };
/// \endcode
///
/// Here, the pack expansion \c Types&... is represented via a
/// PackExpansionType whose pattern is Types&.
class PackExpansionType : public Type, public llvm::FoldingSetNode {
  /// \brief The pattern of the pack expansion.
  QualType Pattern;

  /// \brief The number of expansions that this pack expansion will
  /// generate when substituted (+1), or indicates that 
  ///
  /// This field will only have a non-zero value when some of the parameter 
  /// packs that occur within the pattern have been substituted but others have 
  /// not.
  unsigned NumExpansions;
  
  PackExpansionType(QualType Pattern, QualType Canon,
                    llvm::Optional<unsigned> NumExpansions)
    : Type(PackExpansion, Canon, /*Dependent=*/true,
           /*VariableModified=*/Pattern->isVariablyModifiedType(),
           /*ContainsUnexpandedParameterPack=*/false),
      Pattern(Pattern), 
      NumExpansions(NumExpansions? *NumExpansions + 1: 0) { }

  friend class ASTContext;  // ASTContext creates these
  
public:
  /// \brief Retrieve the pattern of this pack expansion, which is the
  /// type that will be repeatedly instantiated when instantiating the
  /// pack expansion itself.
  QualType getPattern() const { return Pattern; }

  /// \brief Retrieve the number of expansions that this pack expansion will
  /// generate, if known.
  llvm::Optional<unsigned> getNumExpansions() const {
    if (NumExpansions)
      return NumExpansions - 1;
    
    return llvm::Optional<unsigned>();
  }
  
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPattern(), getNumExpansions());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pattern,
                      llvm::Optional<unsigned> NumExpansions) {
    ID.AddPointer(Pattern.getAsOpaquePtr());
    ID.AddBoolean(NumExpansions);
    if (NumExpansions)
      ID.AddInteger(*NumExpansions);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == PackExpansion;
  }
  static bool classof(const PackExpansionType *T) {
    return true;
  }  
};

/// ObjCObjectType - Represents a class type in Objective C.
/// Every Objective C type is a combination of a base type and a
/// list of protocols.
///
/// Given the following declarations:
///   @class C;
///   @protocol P;
///
/// 'C' is an ObjCInterfaceType C.  It is sugar for an ObjCObjectType
/// with base C and no protocols.
///
/// 'C<P>' is an ObjCObjectType with base C and protocol list [P].
///
/// 'id' is a TypedefType which is sugar for an ObjCPointerType whose
/// pointee is an ObjCObjectType with base BuiltinType::ObjCIdType
/// and no protocols.
///
/// 'id<P>' is an ObjCPointerType whose pointee is an ObjCObjecType
/// with base BuiltinType::ObjCIdType and protocol list [P].  Eventually
/// this should get its own sugar class to better represent the source.
class ObjCObjectType : public Type {
  // ObjCObjectType.NumProtocols - the number of protocols stored
  // after the ObjCObjectPointerType node.
  //
  // These protocols are those written directly on the type.  If
  // protocol qualifiers ever become additive, the iterators will need
  // to get kindof complicated.
  //
  // In the canonical object type, these are sorted alphabetically
  // and uniqued.

  /// Either a BuiltinType or an InterfaceType or sugar for either.
  QualType BaseType;

  ObjCProtocolDecl * const *getProtocolStorage() const {
    return const_cast<ObjCObjectType*>(this)->getProtocolStorage();
  }

  ObjCProtocolDecl **getProtocolStorage();

protected:
  ObjCObjectType(QualType Canonical, QualType Base, 
                 ObjCProtocolDecl * const *Protocols, unsigned NumProtocols);

  enum Nonce_ObjCInterface { Nonce_ObjCInterface };
  ObjCObjectType(enum Nonce_ObjCInterface)
        : Type(ObjCInterface, QualType(), false, false, false),
      BaseType(QualType(this_(), 0)) {
    ObjCObjectTypeBits.NumProtocols = 0;
  }

public:
  /// getBaseType - Gets the base type of this object type.  This is
  /// always (possibly sugar for) one of:
  ///  - the 'id' builtin type (as opposed to the 'id' type visible to the
  ///    user, which is a typedef for an ObjCPointerType)
  ///  - the 'Class' builtin type (same caveat)
  ///  - an ObjCObjectType (currently always an ObjCInterfaceType)
  QualType getBaseType() const { return BaseType; }

  bool isObjCId() const {
    return getBaseType()->isSpecificBuiltinType(BuiltinType::ObjCId);
  }
  bool isObjCClass() const {
    return getBaseType()->isSpecificBuiltinType(BuiltinType::ObjCClass);
  }
  bool isObjCUnqualifiedId() const { return qual_empty() && isObjCId(); }
  bool isObjCUnqualifiedClass() const { return qual_empty() && isObjCClass(); }
  bool isObjCUnqualifiedIdOrClass() const {
    if (!qual_empty()) return false;
    if (const BuiltinType *T = getBaseType()->getAs<BuiltinType>())
      return T->getKind() == BuiltinType::ObjCId ||
             T->getKind() == BuiltinType::ObjCClass;
    return false;
  }
  bool isObjCQualifiedId() const { return !qual_empty() && isObjCId(); }
  bool isObjCQualifiedClass() const { return !qual_empty() && isObjCClass(); }

  /// Gets the interface declaration for this object type, if the base type
  /// really is an interface.
  ObjCInterfaceDecl *getInterface() const;

  typedef ObjCProtocolDecl * const *qual_iterator;

  qual_iterator qual_begin() const { return getProtocolStorage(); }
  qual_iterator qual_end() const { return qual_begin() + getNumProtocols(); }

  bool qual_empty() const { return getNumProtocols() == 0; }

  /// getNumProtocols - Return the number of qualifying protocols in this
  /// interface type, or 0 if there are none.
  unsigned getNumProtocols() const { return ObjCObjectTypeBits.NumProtocols; }

  /// \brief Fetch a protocol by index.
  ObjCProtocolDecl *getProtocol(unsigned I) const {
    assert(I < getNumProtocols() && "Out-of-range protocol access");
    return qual_begin()[I];
  }
  
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == ObjCObject ||
           T->getTypeClass() == ObjCInterface;
  }
  static bool classof(const ObjCObjectType *) { return true; }
};

/// ObjCObjectTypeImpl - A class providing a concrete implementation
/// of ObjCObjectType, so as to not increase the footprint of
/// ObjCInterfaceType.  Code outside of ASTContext and the core type
/// system should not reference this type.
class ObjCObjectTypeImpl : public ObjCObjectType, public llvm::FoldingSetNode {
  friend class ASTContext;

  // If anyone adds fields here, ObjCObjectType::getProtocolStorage()
  // will need to be modified.

  ObjCObjectTypeImpl(QualType Canonical, QualType Base, 
                     ObjCProtocolDecl * const *Protocols,
                     unsigned NumProtocols)
    : ObjCObjectType(Canonical, Base, Protocols, NumProtocols) {}

public:
  void Profile(llvm::FoldingSetNodeID &ID);
  static void Profile(llvm::FoldingSetNodeID &ID,
                      QualType Base,
                      ObjCProtocolDecl *const *protocols, 
                      unsigned NumProtocols);  
};

inline ObjCProtocolDecl **ObjCObjectType::getProtocolStorage() {
  return reinterpret_cast<ObjCProtocolDecl**>(
            static_cast<ObjCObjectTypeImpl*>(this) + 1);
}

/// ObjCInterfaceType - Interfaces are the core concept in Objective-C for
/// object oriented design.  They basically correspond to C++ classes.  There
/// are two kinds of interface types, normal interfaces like "NSString" and
/// qualified interfaces, which are qualified with a protocol list like
/// "NSString<NSCopyable, NSAmazing>".
///
/// ObjCInterfaceType guarantees the following properties when considered
/// as a subtype of its superclass, ObjCObjectType:
///   - There are no protocol qualifiers.  To reinforce this, code which
///     tries to invoke the protocol methods via an ObjCInterfaceType will
///     fail to compile.
///   - It is its own base type.  That is, if T is an ObjCInterfaceType*,
///     T->getBaseType() == QualType(T, 0).
class ObjCInterfaceType : public ObjCObjectType {
  ObjCInterfaceDecl *Decl;

  ObjCInterfaceType(const ObjCInterfaceDecl *D)
    : ObjCObjectType(Nonce_ObjCInterface),
      Decl(const_cast<ObjCInterfaceDecl*>(D)) {}
  friend class ASTContext;  // ASTContext creates these.

public:
  /// getDecl - Get the declaration of this interface.
  ObjCInterfaceDecl *getDecl() const { return Decl; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == ObjCInterface;
  }
  static bool classof(const ObjCInterfaceType *) { return true; }

  // Nonsense to "hide" certain members of ObjCObjectType within this
  // class.  People asking for protocols on an ObjCInterfaceType are
  // not going to get what they want: ObjCInterfaceTypes are
  // guaranteed to have no protocols.
  enum {
    qual_iterator,
    qual_begin,
    qual_end,
    getNumProtocols,
    getProtocol
  };
};

inline ObjCInterfaceDecl *ObjCObjectType::getInterface() const {
  if (const ObjCInterfaceType *T =
        getBaseType()->getAs<ObjCInterfaceType>())
    return T->getDecl();
  return 0;
}

/// ObjCObjectPointerType - Used to represent a pointer to an
/// Objective C object.  These are constructed from pointer
/// declarators when the pointee type is an ObjCObjectType (or sugar
/// for one).  In addition, the 'id' and 'Class' types are typedefs
/// for these, and the protocol-qualified types 'id<P>' and 'Class<P>'
/// are translated into these.
///
/// Pointers to pointers to Objective C objects are still PointerTypes;
/// only the first level of pointer gets it own type implementation.
class ObjCObjectPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;

  ObjCObjectPointerType(QualType Canonical, QualType Pointee)
    : Type(ObjCObjectPointer, Canonical, false, false, false),
      PointeeType(Pointee) {}
  friend class ASTContext;  // ASTContext creates these.

public:
  /// getPointeeType - Gets the type pointed to by this ObjC pointer.
  /// The result will always be an ObjCObjectType or sugar thereof.
  QualType getPointeeType() const { return PointeeType; }

  /// getObjCObjectType - Gets the type pointed to by this ObjC
  /// pointer.  This method always returns non-null.
  ///
  /// This method is equivalent to getPointeeType() except that
  /// it discards any typedefs (or other sugar) between this
  /// type and the "outermost" object type.  So for:
  ///   @class A; @protocol P; @protocol Q;
  ///   typedef A<P> AP;
  ///   typedef A A1;
  ///   typedef A1<P> A1P;
  ///   typedef A1P<Q> A1PQ;
  /// For 'A*', getObjectType() will return 'A'.
  /// For 'A<P>*', getObjectType() will return 'A<P>'.
  /// For 'AP*', getObjectType() will return 'A<P>'.
  /// For 'A1*', getObjectType() will return 'A'.
  /// For 'A1<P>*', getObjectType() will return 'A1<P>'.
  /// For 'A1P*', getObjectType() will return 'A1<P>'.
  /// For 'A1PQ*', getObjectType() will return 'A1<Q>', because
  ///   adding protocols to a protocol-qualified base discards the
  ///   old qualifiers (for now).  But if it didn't, getObjectType()
  ///   would return 'A1P<Q>' (and we'd have to make iterating over
  ///   qualifiers more complicated).
  const ObjCObjectType *getObjectType() const {
    return PointeeType->castAs<ObjCObjectType>();
  }

  /// getInterfaceType - If this pointer points to an Objective C
  /// @interface type, gets the type for that interface.  Any protocol
  /// qualifiers on the interface are ignored.
  ///
  /// \return null if the base type for this pointer is 'id' or 'Class'
  const ObjCInterfaceType *getInterfaceType() const {
    return getObjectType()->getBaseType()->getAs<ObjCInterfaceType>();
  }

  /// getInterfaceDecl - If this pointer points to an Objective @interface
  /// type, gets the declaration for that interface.
  ///
  /// \return null if the base type for this pointer is 'id' or 'Class'
  ObjCInterfaceDecl *getInterfaceDecl() const {
    return getObjectType()->getInterface();
  }

  /// isObjCIdType - True if this is equivalent to the 'id' type, i.e. if
  /// its object type is the primitive 'id' type with no protocols.
  bool isObjCIdType() const {
    return getObjectType()->isObjCUnqualifiedId();
  }

  /// isObjCClassType - True if this is equivalent to the 'Class' type,
  /// i.e. if its object tive is the primitive 'Class' type with no protocols.
  bool isObjCClassType() const {
    return getObjectType()->isObjCUnqualifiedClass();
  }
  
  /// isObjCQualifiedIdType - True if this is equivalent to 'id<P>' for some
  /// non-empty set of protocols.
  bool isObjCQualifiedIdType() const {
    return getObjectType()->isObjCQualifiedId();
  }

  /// isObjCQualifiedClassType - True if this is equivalent to 'Class<P>' for
  /// some non-empty set of protocols.
  bool isObjCQualifiedClassType() const {
    return getObjectType()->isObjCQualifiedClass();
  }

  /// An iterator over the qualifiers on the object type.  Provided
  /// for convenience.  This will always iterate over the full set of
  /// protocols on a type, not just those provided directly.
  typedef ObjCObjectType::qual_iterator qual_iterator;

  qual_iterator qual_begin() const {
    return getObjectType()->qual_begin();
  }
  qual_iterator qual_end() const {
    return getObjectType()->qual_end();
  }
  bool qual_empty() const { return getObjectType()->qual_empty(); }

  /// getNumProtocols - Return the number of qualifying protocols on
  /// the object type.
  unsigned getNumProtocols() const {
    return getObjectType()->getNumProtocols();
  }

  /// \brief Retrieve a qualifying protocol by index on the object
  /// type.
  ObjCProtocolDecl *getProtocol(unsigned I) const {
    return getObjectType()->getProtocol(I);
  }
  
  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType T) {
    ID.AddPointer(T.getAsOpaquePtr());
  }
  static bool classof(const Type *T) {
    return T->getTypeClass() == ObjCObjectPointer;
  }
  static bool classof(const ObjCObjectPointerType *) { return true; }
};

/// A qualifier set is used to build a set of qualifiers.
class QualifierCollector : public Qualifiers {
public:
  QualifierCollector(Qualifiers Qs = Qualifiers()) : Qualifiers(Qs) {}

  /// Collect any qualifiers on the given type and return an
  /// unqualified type.  The qualifiers are assumed to be consistent
  /// with those already in the type.
  const Type *strip(QualType type) {
    addFastQualifiers(type.getLocalFastQualifiers());
    if (!type.hasLocalNonFastQualifiers())
      return type.getTypePtrUnsafe();
      
    const ExtQuals *extQuals = type.getExtQualsUnsafe();
    addConsistentQualifiers(extQuals->getQualifiers());
    return extQuals->getBaseType();
  }

  /// Apply the collected qualifiers to the given type.
  QualType apply(const ASTContext &Context, QualType QT) const;

  /// Apply the collected qualifiers to the given type.
  QualType apply(const ASTContext &Context, const Type* T) const;
};


// Inline function definitions.

inline const Type *QualType::getTypePtr() const {
  return getCommonPtr()->BaseType;
}

inline const Type *QualType::getTypePtrOrNull() const {
  return (isNull() ? 0 : getCommonPtr()->BaseType);
}

inline SplitQualType QualType::split() const {
  if (!hasLocalNonFastQualifiers())
    return SplitQualType(getTypePtrUnsafe(),
                         Qualifiers::fromFastMask(getLocalFastQualifiers()));

  const ExtQuals *eq = getExtQualsUnsafe();
  Qualifiers qs = eq->getQualifiers();
  qs.addFastQualifiers(getLocalFastQualifiers());
  return SplitQualType(eq->getBaseType(), qs);
}

inline Qualifiers QualType::getLocalQualifiers() const {
  Qualifiers Quals;
  if (hasLocalNonFastQualifiers())
    Quals = getExtQualsUnsafe()->getQualifiers();
  Quals.addFastQualifiers(getLocalFastQualifiers());
  return Quals;
}

inline Qualifiers QualType::getQualifiers() const {
  Qualifiers quals = getCommonPtr()->CanonicalType.getLocalQualifiers();
  quals.addFastQualifiers(getLocalFastQualifiers());
  return quals;
}

inline unsigned QualType::getCVRQualifiers() const {
  unsigned cvr = getCommonPtr()->CanonicalType.getLocalCVRQualifiers();
  cvr |= getLocalCVRQualifiers();
  return cvr;
}

inline QualType QualType::getCanonicalType() const {
  QualType canon = getCommonPtr()->CanonicalType;
  return canon.withFastQualifiers(getLocalFastQualifiers());
}

inline bool QualType::isCanonical() const {
  return getTypePtr()->isCanonicalUnqualified();
}

inline bool QualType::isCanonicalAsParam() const {
  if (!isCanonical()) return false;
  if (hasLocalQualifiers()) return false;
  
  const Type *T = getTypePtr();
  if (T->isVariablyModifiedType() && T->hasSizedVLAType())
    return false;

  return !isa<FunctionType>(T) && !isa<ArrayType>(T);
}

inline bool QualType::isConstQualified() const {
  return isLocalConstQualified() || 
         getCommonPtr()->CanonicalType.isLocalConstQualified();
}

inline bool QualType::isRestrictQualified() const {
  return isLocalRestrictQualified() || 
         getCommonPtr()->CanonicalType.isLocalRestrictQualified();
}


inline bool QualType::isVolatileQualified() const {
  return isLocalVolatileQualified() || 
         getCommonPtr()->CanonicalType.isLocalVolatileQualified();
}
  
inline bool QualType::hasQualifiers() const {
  return hasLocalQualifiers() ||
         getCommonPtr()->CanonicalType.hasLocalQualifiers();
}

inline QualType QualType::getUnqualifiedType() const {
  if (!getTypePtr()->getCanonicalTypeInternal().hasLocalQualifiers())
    return QualType(getTypePtr(), 0);

  return QualType(getSplitUnqualifiedTypeImpl(*this).first, 0);
}

inline SplitQualType QualType::getSplitUnqualifiedType() const {
  if (!getTypePtr()->getCanonicalTypeInternal().hasLocalQualifiers())
    return split();

  return getSplitUnqualifiedTypeImpl(*this);
}
  
inline void QualType::removeLocalConst() {
  removeLocalFastQualifiers(Qualifiers::Const);
}

inline void QualType::removeLocalRestrict() {
  removeLocalFastQualifiers(Qualifiers::Restrict);
}

inline void QualType::removeLocalVolatile() {
  removeLocalFastQualifiers(Qualifiers::Volatile);
}

inline void QualType::removeLocalCVRQualifiers(unsigned Mask) {
  assert(!(Mask & ~Qualifiers::CVRMask) && "mask has non-CVR bits");
  assert((int)Qualifiers::CVRMask == (int)Qualifiers::FastMask);

  // Fast path: we don't need to touch the slow qualifiers.
  removeLocalFastQualifiers(Mask);
}

/// getAddressSpace - Return the address space of this type.
inline unsigned QualType::getAddressSpace() const {
  return getQualifiers().getAddressSpace();
}

/// getObjCGCAttr - Return the gc attribute of this type.
inline Qualifiers::GC QualType::getObjCGCAttr() const {
  return getQualifiers().getObjCGCAttr();
}

inline FunctionType::ExtInfo getFunctionExtInfo(const Type &t) {
  if (const PointerType *PT = t.getAs<PointerType>()) {
    if (const FunctionType *FT = PT->getPointeeType()->getAs<FunctionType>())
      return FT->getExtInfo();
  } else if (const FunctionType *FT = t.getAs<FunctionType>())
    return FT->getExtInfo();

  return FunctionType::ExtInfo();
}

inline FunctionType::ExtInfo getFunctionExtInfo(QualType t) {
  return getFunctionExtInfo(*t);
}

/// \brief Determine whether this set of qualifiers is a superset of the given 
/// set of qualifiers.
inline bool Qualifiers::isSupersetOf(Qualifiers Other) const {
  return Mask != Other.Mask && (Mask | Other.Mask) == Mask;
}

/// isMoreQualifiedThan - Determine whether this type is more
/// qualified than the Other type. For example, "const volatile int"
/// is more qualified than "const int", "volatile int", and
/// "int". However, it is not more qualified than "const volatile
/// int".
inline bool QualType::isMoreQualifiedThan(QualType other) const {
  Qualifiers myQuals = getQualifiers();
  Qualifiers otherQuals = other.getQualifiers();
  return (myQuals != otherQuals && myQuals.compatiblyIncludes(otherQuals));
}

/// isAtLeastAsQualifiedAs - Determine whether this type is at last
/// as qualified as the Other type. For example, "const volatile
/// int" is at least as qualified as "const int", "volatile int",
/// "int", and "const volatile int".
inline bool QualType::isAtLeastAsQualifiedAs(QualType other) const {
  return getQualifiers().compatiblyIncludes(other.getQualifiers());
}

/// getNonReferenceType - If Type is a reference type (e.g., const
/// int&), returns the type that the reference refers to ("const
/// int"). Otherwise, returns the type itself. This routine is used
/// throughout Sema to implement C++ 5p6:
///
///   If an expression initially has the type "reference to T" (8.3.2,
///   8.5.3), the type is adjusted to "T" prior to any further
///   analysis, the expression designates the object or function
///   denoted by the reference, and the expression is an lvalue.
inline QualType QualType::getNonReferenceType() const {
  if (const ReferenceType *RefType = (*this)->getAs<ReferenceType>())
    return RefType->getPointeeType();
  else
    return *this;
}

inline bool Type::isFunctionType() const {
  return isa<FunctionType>(CanonicalType);
}
inline bool Type::isPointerType() const {
  return isa<PointerType>(CanonicalType);
}
inline bool Type::isAnyPointerType() const {
  return isPointerType() || isObjCObjectPointerType();
}
inline bool Type::isBlockPointerType() const {
  return isa<BlockPointerType>(CanonicalType);
}
inline bool Type::isReferenceType() const {
  return isa<ReferenceType>(CanonicalType);
}
inline bool Type::isLValueReferenceType() const {
  return isa<LValueReferenceType>(CanonicalType);
}
inline bool Type::isRValueReferenceType() const {
  return isa<RValueReferenceType>(CanonicalType);
}
inline bool Type::isFunctionPointerType() const {
  if (const PointerType *T = getAs<PointerType>())
    return T->getPointeeType()->isFunctionType();
  else
    return false;
}
inline bool Type::isMemberPointerType() const {
  return isa<MemberPointerType>(CanonicalType);
}
inline bool Type::isMemberFunctionPointerType() const {
  if (const MemberPointerType* T = getAs<MemberPointerType>())
    return T->isMemberFunctionPointer();
  else
    return false;
}
inline bool Type::isMemberDataPointerType() const {
  if (const MemberPointerType* T = getAs<MemberPointerType>())
    return T->isMemberDataPointer();
  else
    return false;
}
inline bool Type::isArrayType() const {
  return isa<ArrayType>(CanonicalType);
}
inline bool Type::isConstantArrayType() const {
  return isa<ConstantArrayType>(CanonicalType);
}
inline bool Type::isIncompleteArrayType() const {
  return isa<IncompleteArrayType>(CanonicalType);
}
inline bool Type::isVariableArrayType() const {
  return isa<VariableArrayType>(CanonicalType);
}
inline bool Type::isDependentSizedArrayType() const {
  return isa<DependentSizedArrayType>(CanonicalType);
}
inline bool Type::isBuiltinType() const {
  return isa<BuiltinType>(CanonicalType);
}
inline bool Type::isRecordType() const {
  return isa<RecordType>(CanonicalType);
}
inline bool Type::isEnumeralType() const {
  return isa<EnumType>(CanonicalType);
}
inline bool Type::isAnyComplexType() const {
  return isa<ComplexType>(CanonicalType);
}
inline bool Type::isVectorType() const {
  return isa<VectorType>(CanonicalType);
}
inline bool Type::isExtVectorType() const {
  return isa<ExtVectorType>(CanonicalType);
}
inline bool Type::isObjCObjectPointerType() const {
  return isa<ObjCObjectPointerType>(CanonicalType);
}
inline bool Type::isObjCObjectType() const {
  return isa<ObjCObjectType>(CanonicalType);
}
inline bool Type::isObjCObjectOrInterfaceType() const {
  return isa<ObjCInterfaceType>(CanonicalType) || 
    isa<ObjCObjectType>(CanonicalType);
}

inline bool Type::isObjCQualifiedIdType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->isObjCQualifiedIdType();
  return false;
}
inline bool Type::isObjCQualifiedClassType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->isObjCQualifiedClassType();
  return false;
}
inline bool Type::isObjCIdType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->isObjCIdType();
  return false;
}
inline bool Type::isObjCClassType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->isObjCClassType();
  return false;
}
inline bool Type::isObjCSelType() const {
  if (const PointerType *OPT = getAs<PointerType>())
    return OPT->getPointeeType()->isSpecificBuiltinType(BuiltinType::ObjCSel);
  return false;
}
inline bool Type::isObjCBuiltinType() const {
  return isObjCIdType() || isObjCClassType() || isObjCSelType();
}
inline bool Type::isTemplateTypeParmType() const {
  return isa<TemplateTypeParmType>(CanonicalType);
}

inline bool Type::isSpecificBuiltinType(unsigned K) const {
  if (const BuiltinType *BT = getAs<BuiltinType>())
    if (BT->getKind() == (BuiltinType::Kind) K)
      return true;
  return false;
}

inline bool Type::isPlaceholderType() const {
  if (const BuiltinType *BT = getAs<BuiltinType>())
    return BT->isPlaceholderType();
  return false;
}

/// \brief Determines whether this is a type for which one can define
/// an overloaded operator.
inline bool Type::isOverloadableType() const {
  return isDependentType() || isRecordType() || isEnumeralType();
}

inline bool Type::hasPointerRepresentation() const {
  return (isPointerType() || isReferenceType() || isBlockPointerType() ||
          isObjCObjectPointerType() || isNullPtrType());
}

inline bool Type::hasObjCPointerRepresentation() const {
  return isObjCObjectPointerType();
}

inline const Type *Type::getBaseElementTypeUnsafe() const {
  const Type *type = this;
  while (const ArrayType *arrayType = type->getAsArrayTypeUnsafe())
    type = arrayType->getElementType().getTypePtr();
  return type;
}

/// Insertion operator for diagnostics.  This allows sending QualType's into a
/// diagnostic with <<.
inline const DiagnosticBuilder &operator<<(const DiagnosticBuilder &DB,
                                           QualType T) {
  DB.AddTaggedVal(reinterpret_cast<intptr_t>(T.getAsOpaquePtr()),
                  Diagnostic::ak_qualtype);
  return DB;
}

/// Insertion operator for partial diagnostics.  This allows sending QualType's
/// into a diagnostic with <<.
inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                           QualType T) {
  PD.AddTaggedVal(reinterpret_cast<intptr_t>(T.getAsOpaquePtr()),
                  Diagnostic::ak_qualtype);
  return PD;
}

// Helper class template that is used by Type::getAs to ensure that one does
// not try to look through a qualified type to get to an array type.
template<typename T,
         bool isArrayType = (llvm::is_same<T, ArrayType>::value ||
                             llvm::is_base_of<ArrayType, T>::value)>
struct ArrayType_cannot_be_used_with_getAs { };
  
template<typename T>
struct ArrayType_cannot_be_used_with_getAs<T, true>;
  
/// Member-template getAs<specific type>'.
template <typename T> const T *Type::getAs() const {
  ArrayType_cannot_be_used_with_getAs<T> at;
  (void)at;
  
  // If this is directly a T type, return it.
  if (const T *Ty = dyn_cast<T>(this))
    return Ty;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<T>(CanonicalType))
    return 0;

  // If this is a typedef for the type, strip the typedef off without
  // losing all typedef information.
  return cast<T>(getUnqualifiedDesugaredType());
}

inline const ArrayType *Type::getAsArrayTypeUnsafe() const {
  // If this is directly an array type, return it.
  if (const ArrayType *arr = dyn_cast<ArrayType>(this))
    return arr;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ArrayType>(CanonicalType))
    return 0;

  // If this is a typedef for the type, strip the typedef off without
  // losing all typedef information.
  return cast<ArrayType>(getUnqualifiedDesugaredType());
}

template <typename T> const T *Type::castAs() const {
  ArrayType_cannot_be_used_with_getAs<T> at;
  (void) at;

  assert(isa<T>(CanonicalType));
  if (const T *ty = dyn_cast<T>(this)) return ty;
  return cast<T>(getUnqualifiedDesugaredType());
}

inline const ArrayType *Type::castAsArrayTypeUnsafe() const {
  assert(isa<ArrayType>(CanonicalType));
  if (const ArrayType *arr = dyn_cast<ArrayType>(this)) return arr;
  return cast<ArrayType>(getUnqualifiedDesugaredType());
}

}  // end namespace clang

#endif
