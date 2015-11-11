//===--- Type.h - C Language Family Type Representation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief C Language Family Type Representation
///
/// This file defines the clang::Type interface and subclasses, used to
/// represent types for languages in the C family.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TYPE_H
#define LLVM_CLANG_AST_TYPE_H

#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateName.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/ExceptionSpecificationType.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/Visibility.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"

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
  class TypedefNameDecl;
  class TemplateDecl;
  class TemplateTypeParmDecl;
  class NonTypeTemplateParmDecl;
  class TemplateTemplateParmDecl;
  class TagDecl;
  class RecordDecl;
  class CXXRecordDecl;
  class EnumDecl;
  class FieldDecl;
  class FunctionDecl;
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

/// The collection of all-type qualifiers we support.
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

  enum ObjCLifetime {
    /// There is no lifetime qualification on this type.
    OCL_None,

    /// This object can be modified without requiring retains or
    /// releases.
    OCL_ExplicitNone,

    /// Assigning into this object requires the old value to be
    /// released and the new value to be retained.  The timing of the
    /// release of the old value is inexact: it may be moved to
    /// immediately after the last known point where the value is
    /// live.
    OCL_Strong,

    /// Reading or writing from this object requires a barrier call.
    OCL_Weak,

    /// Assigning into this object requires a lifetime extension.
    OCL_Autoreleasing
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

  /// Returns the common set of qualifiers while removing them from
  /// the given sets.
  static Qualifiers removeCommonQualifiers(Qualifiers &L, Qualifiers &R) {
    // If both are only CVR-qualified, bit operations are sufficient.
    if (!(L.Mask & ~CVRMask) && !(R.Mask & ~CVRMask)) {
      Qualifiers Q;
      Q.Mask = L.Mask & R.Mask;
      L.Mask &= ~Q.Mask;
      R.Mask &= ~Q.Mask;
      return Q;
    }

    Qualifiers Q;
    unsigned CommonCRV = L.getCVRQualifiers() & R.getCVRQualifiers();
    Q.addCVRQualifiers(CommonCRV);
    L.removeCVRQualifiers(CommonCRV);
    R.removeCVRQualifiers(CommonCRV);

    if (L.getObjCGCAttr() == R.getObjCGCAttr()) {
      Q.setObjCGCAttr(L.getObjCGCAttr());
      L.removeObjCGCAttr();
      R.removeObjCGCAttr();
    }

    if (L.getObjCLifetime() == R.getObjCLifetime()) {
      Q.setObjCLifetime(L.getObjCLifetime());
      L.removeObjCLifetime();
      R.removeObjCLifetime();
    }

    if (L.getAddressSpace() == R.getAddressSpace()) {
      Q.setAddressSpace(L.getAddressSpace());
      L.removeAddressSpace();
      R.removeAddressSpace();
    }
    return Q;
  }

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
  Qualifiers withoutObjCGCAttr() const {
    Qualifiers qs = *this;
    qs.removeObjCGCAttr();
    return qs;
  }
  Qualifiers withoutObjCLifetime() const {
    Qualifiers qs = *this;
    qs.removeObjCLifetime();
    return qs;
  }

  bool hasObjCLifetime() const { return Mask & LifetimeMask; }
  ObjCLifetime getObjCLifetime() const {
    return ObjCLifetime((Mask & LifetimeMask) >> LifetimeShift);
  }
  void setObjCLifetime(ObjCLifetime type) {
    Mask = (Mask & ~LifetimeMask) | (type << LifetimeShift);
  }
  void removeObjCLifetime() { setObjCLifetime(OCL_None); }
  void addObjCLifetime(ObjCLifetime type) {
    assert(type);
    assert(!hasObjCLifetime());
    Mask |= (type << LifetimeShift);
  }

  /// True if the lifetime is neither None or ExplicitNone.
  bool hasNonTrivialObjCLifetime() const {
    ObjCLifetime lifetime = getObjCLifetime();
    return (lifetime > OCL_ExplicitNone);
  }

  /// True if the lifetime is either strong or weak.
  bool hasStrongOrWeakObjCLifetime() const {
    ObjCLifetime lifetime = getObjCLifetime();
    return (lifetime == OCL_Strong || lifetime == OCL_Weak);
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

  /// Return true if the set contains any qualifiers which require an ExtQuals
  /// node to be allocated.
  bool hasNonFastQualifiers() const { return Mask & ~FastMask; }
  Qualifiers getNonFastQualifiers() const {
    Qualifiers Quals = *this;
    Quals.setFastQualifiers(0);
    return Quals;
  }

  /// Return true if the set contains any qualifiers.
  bool hasQualifiers() const { return Mask; }
  bool empty() const { return !Mask; }

  /// Add the qualifiers from the given set to this set.
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
      if (Q.hasObjCLifetime())
        addObjCLifetime(Q.getObjCLifetime());
    }
  }

  /// \brief Remove the qualifiers from the given set from this set.
  void removeQualifiers(Qualifiers Q) {
    // If the other set doesn't have any non-boolean qualifiers, just
    // bit-and the inverse in.
    if (!(Q.Mask & ~CVRMask))
      Mask &= ~Q.Mask;
    else {
      Mask &= ~(Q.Mask & CVRMask);
      if (getObjCGCAttr() == Q.getObjCGCAttr())
        removeObjCGCAttr();
      if (getObjCLifetime() == Q.getObjCLifetime())
        removeObjCLifetime();
      if (getAddressSpace() == Q.getAddressSpace())
        removeAddressSpace();
    }
  }

  /// Add the qualifiers from the given set to this set, given that
  /// they don't conflict.
  void addConsistentQualifiers(Qualifiers qs) {
    assert(getAddressSpace() == qs.getAddressSpace() ||
           !hasAddressSpace() || !qs.hasAddressSpace());
    assert(getObjCGCAttr() == qs.getObjCGCAttr() ||
           !hasObjCGCAttr() || !qs.hasObjCGCAttr());
    assert(getObjCLifetime() == qs.getObjCLifetime() ||
           !hasObjCLifetime() || !qs.hasObjCLifetime());
    Mask |= qs.Mask;
  }

  /// Returns true if this address space is a superset of the other one.
  /// OpenCL v2.0 defines conversion rules (OpenCLC v2.0 s6.5.5) and notion of
  /// overlapping address spaces.
  /// CL1.1 or CL1.2:
  ///   every address space is a superset of itself.
  /// CL2.0 adds:
  ///   __generic is a superset of any address space except for __constant.
  bool isAddressSpaceSupersetOf(Qualifiers other) const {
    return
        // Address spaces must match exactly.
        getAddressSpace() == other.getAddressSpace() ||
        // Otherwise in OpenCLC v2.0 s6.5.5: every address space except
        // for __constant can be used as __generic.
        (getAddressSpace() == LangAS::opencl_generic &&
         other.getAddressSpace() != LangAS::opencl_constant);
  }

  /// Determines if these qualifiers compatibly include another set.
  /// Generally this answers the question of whether an object with the other
  /// qualifiers can be safely used as an object with these qualifiers.
  bool compatiblyIncludes(Qualifiers other) const {
    return isAddressSpaceSupersetOf(other) &&
           // ObjC GC qualifiers can match, be added, or be removed, but can't
           // be changed.
           (getObjCGCAttr() == other.getObjCGCAttr() || !hasObjCGCAttr() ||
            !other.hasObjCGCAttr()) &&
           // ObjC lifetime qualifiers must match exactly.
           getObjCLifetime() == other.getObjCLifetime() &&
           // CVR qualifiers may subset.
           (((Mask & CVRMask) | (other.Mask & CVRMask)) == (Mask & CVRMask));
  }

  /// \brief Determines if these qualifiers compatibly include another set of
  /// qualifiers from the narrow perspective of Objective-C ARC lifetime.
  ///
  /// One set of Objective-C lifetime qualifiers compatibly includes the other
  /// if the lifetime qualifiers match, or if both are non-__weak and the
  /// including set also contains the 'const' qualifier, or both are non-__weak
  /// and one is None (which can only happen in non-ARC modes).
  bool compatiblyIncludesObjCLifetime(Qualifiers other) const {
    if (getObjCLifetime() == other.getObjCLifetime())
      return true;

    if (getObjCLifetime() == OCL_Weak || other.getObjCLifetime() == OCL_Weak)
      return false;

    if (getObjCLifetime() == OCL_None || other.getObjCLifetime() == OCL_None)
      return true;

    return hasConst();
  }

  /// \brief Determine whether this set of qualifiers is a strict superset of
  /// another set of qualifiers, not considering qualifier compatibility.
  bool isStrictSupersetOf(Qualifiers Other) const;

  bool operator==(Qualifiers Other) const { return Mask == Other.Mask; }
  bool operator!=(Qualifiers Other) const { return Mask != Other.Mask; }

  explicit operator bool() const { return hasQualifiers(); }

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
    removeQualifiers(R);
    return *this;
  }

  /// \brief Compute the difference between two qualifier sets.
  friend Qualifiers operator-(Qualifiers L, Qualifiers R) {
    L -= R;
    return L;
  }

  std::string getAsString() const;
  std::string getAsString(const PrintingPolicy &Policy) const;

  bool isEmptyWhenPrinted(const PrintingPolicy &Policy) const;
  void print(raw_ostream &OS, const PrintingPolicy &Policy,
             bool appendSpaceIfNonEmpty = false) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(Mask);
  }

private:

  // bits:     |0 1 2|3 .. 4|5  ..  7|8   ...   31|
  //           |C R V|GCAttr|Lifetime|AddressSpace|
  uint32_t Mask;

  static const uint32_t GCAttrMask = 0x18;
  static const uint32_t GCAttrShift = 3;
  static const uint32_t LifetimeMask = 0xE0;
  static const uint32_t LifetimeShift = 5;
  static const uint32_t AddressSpaceMask = ~(CVRMask|GCAttrMask|LifetimeMask);
  static const uint32_t AddressSpaceShift = 8;
};

/// A std::pair-like structure for storing a qualified type split
/// into its local qualifiers and its locally-unqualified type.
struct SplitQualType {
  /// The locally-unqualified type.
  const Type *Ty;

  /// The local qualifiers.
  Qualifiers Quals;

  SplitQualType() : Ty(nullptr), Quals() {}
  SplitQualType(const Type *ty, Qualifiers qs) : Ty(ty), Quals(qs) {}

  SplitQualType getSingleStepDesugaredType() const; // end of this file

  // Make std::tie work.
  std::pair<const Type *,Qualifiers> asPair() const {
    return std::pair<const Type *, Qualifiers>(Ty, Quals);
  }

  friend bool operator==(SplitQualType a, SplitQualType b) {
    return a.Ty == b.Ty && a.Quals == b.Quals;
  }
  friend bool operator!=(SplitQualType a, SplitQualType b) {
    return a.Ty != b.Ty || a.Quals != b.Quals;
  }
};

/// The kind of type we are substituting Objective-C type arguments into.
///
/// The kind of substitution affects the replacement of type parameters when
/// no concrete type information is provided, e.g., when dealing with an
/// unspecialized type.
enum class ObjCSubstitutionContext {
  /// An ordinary type.
  Ordinary,
  /// The result type of a method or function.
  Result,
  /// The parameter type of a method or function.
  Parameter,
  /// The type of a property.
  Property,
  /// The superclass of a type.
  Superclass,
};

/// A (possibly-)qualified type.
///
/// For efficiency, we don't store CV-qualified types as nodes on their
/// own: instead each reference to a type stores the qualifiers.  This
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
  ///
  /// This function requires that the type not be NULL. If the type might be
  /// NULL, use the (slightly less efficient) \c getTypePtrOrNull().
  const Type *getTypePtr() const;

  const Type *getTypePtrOrNull() const;

  /// Retrieves a pointer to the name of the base type.
  const IdentifierInfo *getBaseTypeIdentifier() const;

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

  /// Return true if this QualType doesn't point to a type yet.
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

  /// \brief Determine whether this is a Plain Old Data (POD) type (C++ 3.9p10).
  bool isPODType(ASTContext &Context) const;

  /// Return true if this is a POD type according to the rules of the C++98
  /// standard, regardless of the current compilation's language.
  bool isCXX98PODType(ASTContext &Context) const;

  /// Return true if this is a POD type according to the more relaxed rules
  /// of the C++11 standard, regardless of the current compilation's language.
  /// (C++0x [basic.types]p9)
  bool isCXX11PODType(ASTContext &Context) const;

  /// Return true if this is a trivial type per (C++0x [basic.types]p9)
  bool isTrivialType(ASTContext &Context) const;

  /// Return true if this is a trivially copyable type (C++0x [basic.types]p9)
  bool isTriviallyCopyableType(ASTContext &Context) const;

  // Don't promise in the API that anything besides 'const' can be
  // easily added.

  /// Add the `const` type qualifier to this QualType.
  void addConst() {
    addFastQualifiers(Qualifiers::Const);
  }
  QualType withConst() const {
    return withFastQualifiers(Qualifiers::Const);
  }

  /// Add the `volatile` type qualifier to this QualType.
  void addVolatile() {
    addFastQualifiers(Qualifiers::Volatile);
  }
  QualType withVolatile() const {
    return withFastQualifiers(Qualifiers::Volatile);
  }

  /// Add the `restrict` qualifier to this QualType.
  void addRestrict() {
    addFastQualifiers(Qualifiers::Restrict);
  }
  QualType withRestrict() const {
    return withFastQualifiers(Qualifiers::Restrict);
  }

  QualType withCVRQualifiers(unsigned CVR) const {
    return withFastQualifiers(CVR);
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
  /// The resulting type might still be qualified if it's sugar for an array
  /// type.  To strip qualifiers even from within a sugared array type, use
  /// ASTContext::getUnqualifiedArrayType.
  inline QualType getUnqualifiedType() const;

  /// Retrieve the unqualified variant of the given type, removing as little
  /// sugar as possible.
  ///
  /// Like getUnqualifiedType(), but also returns the set of
  /// qualifiers that were built up.
  ///
  /// The resulting type might still be qualified if it's sugar for an array
  /// type.  To strip qualifiers even from within a sugared array type, use
  /// ASTContext::getUnqualifiedArrayType.
  inline SplitQualType getSplitUnqualifiedType() const;

  /// \brief Determine whether this type is more qualified than the other
  /// given type, requiring exact equality for non-CVR qualifiers.
  bool isMoreQualifiedThan(QualType Other) const;

  /// \brief Determine whether this type is at least as qualified as the other
  /// given type, requiring exact equality for non-CVR qualifiers.
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
  QualType getNonLValueExprType(const ASTContext &Context) const;

  /// Return the specified type with any "sugar" removed from
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

  /// \brief Return the specified type with one level of "sugar" removed from
  /// the type.
  ///
  /// This routine takes off the first typedef, typeof, etc. If the outer level
  /// of the type is already concrete, it returns it unmodified.
  QualType getSingleStepDesugaredType(const ASTContext &Context) const {
    return getSingleStepDesugaredTypeImpl(*this, Context);
  }

  /// Returns the specified type after dropping any
  /// outer-level parentheses.
  QualType IgnoreParens() const {
    if (isa<ParenType>(*this))
      return QualType::IgnoreParens(*this);
    return *this;
  }

  /// Indicate whether the specified types and qualifiers are identical.
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
    return getAsString(split.Ty, split.Quals);
  }
  static std::string getAsString(const Type *ty, Qualifiers qs);

  std::string getAsString(const PrintingPolicy &Policy) const;

  void print(raw_ostream &OS, const PrintingPolicy &Policy,
             const Twine &PlaceHolder = Twine()) const {
    print(split(), OS, Policy, PlaceHolder);
  }
  static void print(SplitQualType split, raw_ostream &OS,
                    const PrintingPolicy &policy, const Twine &PlaceHolder) {
    return print(split.Ty, split.Quals, OS, policy, PlaceHolder);
  }
  static void print(const Type *ty, Qualifiers qs,
                    raw_ostream &OS, const PrintingPolicy &policy,
                    const Twine &PlaceHolder);

  void getAsStringInternal(std::string &Str,
                           const PrintingPolicy &Policy) const {
    return getAsStringInternal(split(), Str, Policy);
  }
  static void getAsStringInternal(SplitQualType split, std::string &out,
                                  const PrintingPolicy &policy) {
    return getAsStringInternal(split.Ty, split.Quals, out, policy);
  }
  static void getAsStringInternal(const Type *ty, Qualifiers qs,
                                  std::string &out,
                                  const PrintingPolicy &policy);

  class StreamedQualTypeHelper {
    const QualType &T;
    const PrintingPolicy &Policy;
    const Twine &PlaceHolder;
  public:
    StreamedQualTypeHelper(const QualType &T, const PrintingPolicy &Policy,
                           const Twine &PlaceHolder)
      : T(T), Policy(Policy), PlaceHolder(PlaceHolder) { }

    friend raw_ostream &operator<<(raw_ostream &OS,
                                   const StreamedQualTypeHelper &SQT) {
      SQT.T.print(OS, SQT.Policy, SQT.PlaceHolder);
      return OS;
    }
  };

  StreamedQualTypeHelper stream(const PrintingPolicy &Policy,
                                const Twine &PlaceHolder = Twine()) const {
    return StreamedQualTypeHelper(*this, Policy, PlaceHolder);
  }

  void dump(const char *s) const;
  void dump() const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(getAsOpaquePtr());
  }

  /// Return the address space of this type.
  inline unsigned getAddressSpace() const;

  /// Returns gc attribute of this type.
  inline Qualifiers::GC getObjCGCAttr() const;

  /// true when Type is objc's weak.
  bool isObjCGCWeak() const {
    return getObjCGCAttr() == Qualifiers::Weak;
  }

  /// true when Type is objc's strong.
  bool isObjCGCStrong() const {
    return getObjCGCAttr() == Qualifiers::Strong;
  }

  /// Returns lifetime attribute of this type.
  Qualifiers::ObjCLifetime getObjCLifetime() const {
    return getQualifiers().getObjCLifetime();
  }

  bool hasNonTrivialObjCLifetime() const {
    return getQualifiers().hasNonTrivialObjCLifetime();
  }

  bool hasStrongOrWeakObjCLifetime() const {
    return getQualifiers().hasStrongOrWeakObjCLifetime();
  }

  enum DestructionKind {
    DK_none,
    DK_cxx_destructor,
    DK_objc_strong_lifetime,
    DK_objc_weak_lifetime
  };

  /// Returns a nonzero value if objects of this type require
  /// non-trivial work to clean up after.  Non-zero because it's
  /// conceivable that qualifiers (objc_gc(weak)?) could make
  /// something require destruction.
  DestructionKind isDestructedType() const {
    return isDestructedTypeImpl(*this);
  }

  /// Determine whether expressions of the given type are forbidden
  /// from being lvalues in C.
  ///
  /// The expression types that are forbidden to be lvalues are:
  ///   - 'void', but not qualified void
  ///   - function types
  ///
  /// The exact rule here is C99 6.3.2.1:
  ///   An lvalue is an expression with an object type or an incomplete
  ///   type other than void.
  bool isCForbiddenLValueType() const;

  /// Substitute type arguments for the Objective-C type parameters used in the
  /// subject type.
  ///
  /// \param ctx ASTContext in which the type exists.
  ///
  /// \param typeArgs The type arguments that will be substituted for the
  /// Objective-C type parameters in the subject type, which are generally
  /// computed via \c Type::getObjCSubstitutions. If empty, the type
  /// parameters will be replaced with their bounds or id/Class, as appropriate
  /// for the context.
  ///
  /// \param context The context in which the subject type was written.
  ///
  /// \returns the resulting type.
  QualType substObjCTypeArgs(ASTContext &ctx,
                             ArrayRef<QualType> typeArgs,
                             ObjCSubstitutionContext context) const;

  /// Substitute type arguments from an object type for the Objective-C type
  /// parameters used in the subject type.
  ///
  /// This operation combines the computation of type arguments for
  /// substitution (\c Type::getObjCSubstitutions) with the actual process of
  /// substitution (\c QualType::substObjCTypeArgs) for the convenience of
  /// callers that need to perform a single substitution in isolation.
  ///
  /// \param objectType The type of the object whose member type we're
  /// substituting into. For example, this might be the receiver of a message
  /// or the base of a property access.
  ///
  /// \param dc The declaration context from which the subject type was
  /// retrieved, which indicates (for example) which type parameters should
  /// be substituted.
  ///
  /// \param context The context in which the subject type was written.
  ///
  /// \returns the subject type after replacing all of the Objective-C type
  /// parameters with their corresponding arguments.
  QualType substObjCMemberType(QualType objectType,
                               const DeclContext *dc,
                               ObjCSubstitutionContext context) const;

  /// Strip Objective-C "__kindof" types from the given type.
  QualType stripObjCKindOfType(const ASTContext &ctx) const;

private:
  // These methods are implemented in a separate translation unit;
  // "static"-ize them to avoid creating temporary QualTypes in the
  // caller.
  static bool isConstant(QualType T, ASTContext& Ctx);
  static QualType getDesugaredType(QualType T, const ASTContext &Context);
  static SplitQualType getSplitDesugaredType(QualType T);
  static SplitQualType getSplitUnqualifiedTypeImpl(QualType type);
  static QualType getSingleStepDesugaredTypeImpl(QualType type,
                                                 const ASTContext &C);
  static QualType IgnoreParens(QualType T);
  static DestructionKind isDestructedTypeImpl(QualType type);
};

} // end clang.

namespace llvm {
/// Implement simplify_type for QualType, so that we can dyn_cast from QualType
/// to a specific Type class.
template<> struct simplify_type< ::clang::QualType> {
  typedef const ::clang::Type *SimpleType;
  static SimpleType getSimplifiedValue(::clang::QualType Val) {
    return Val.getTypePtr();
  }
};

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

/// We can encode up to four bits in the low bits of a
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

  /// The immutable set of qualifiers applied by this node. Always contains
  /// extended qualifiers.
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

  bool hasObjCLifetime() const { return Quals.hasObjCLifetime(); }
  Qualifiers::ObjCLifetime getObjCLifetime() const {
    return Quals.getObjCLifetime();
  }

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

/// The kind of C++11 ref-qualifier associated with a function type.
/// This determines whether a member function's "this" object can be an
/// lvalue, rvalue, or neither.
enum RefQualifierKind {
  /// \brief No ref-qualifier was provided.
  RQ_None = 0,
  /// \brief An lvalue ref-qualifier was provided (\c &).
  RQ_LValue,
  /// \brief An rvalue ref-qualifier was provided (\c &&).
  RQ_RValue
};

/// Which keyword(s) were used to create an AutoType.
enum class AutoTypeKeyword {
  /// \brief auto
  Auto,
  /// \brief decltype(auto)
  DecltypeAuto,
  /// \brief __auto_type (GNU extension)
  GNUAutoType
};

/// The base class of the type hierarchy.
///
/// A central concept with types is that each type always has a canonical
/// type.  A canonical type is the type with any typedef names stripped out
/// of it or the types it references.  For example, consider:
///
///  typedef int  foo;
///  typedef foo* bar;
///    'int *'    'foo *'    'bar'
///
/// There will be a Type object created for 'int'.  Since int is canonical, its
/// CanonicalType pointer points to itself.  There is also a Type for 'foo' (a
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
  Type(const Type &) = delete;
  void operator=(const Type &) = delete;

  /// Bitfields required by the Type class.
  class TypeBitfields {
    friend class Type;
    template <class T> friend class TypePropertyCache;

    /// TypeClass bitfield - Enum that specifies what subclass this belongs to.
    unsigned TC : 8;

    /// Whether this type is a dependent type (C++ [temp.dep.type]).
    unsigned Dependent : 1;

    /// Whether this type somehow involves a template parameter, even
    /// if the resolution of the type does not depend on a template parameter.
    unsigned InstantiationDependent : 1;

    /// Whether this type is a variably-modified type (C99 6.7.5).
    unsigned VariablyModified : 1;

    /// \brief Whether this type contains an unexpanded parameter pack
    /// (for C++11 variadic templates).
    unsigned ContainsUnexpandedParameterPack : 1;

    /// \brief True if the cache (i.e. the bitfields here starting with
    /// 'Cache') is valid.
    mutable unsigned CacheValid : 1;

    /// \brief Linkage of this type.
    mutable unsigned CachedLinkage : 3;

    /// \brief Whether this type involves and local or unnamed types.
    mutable unsigned CachedLocalOrUnnamed : 1;

    /// \brief Whether this type comes from an AST file.
    mutable unsigned FromAST : 1;

    bool isCacheValid() const {
      return CacheValid;
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
  enum { NumTypeBits = 18 };

protected:
  // These classes allow subclasses to somewhat cleanly pack bitfields
  // into Type.

  class ArrayTypeBitfields {
    friend class ArrayType;

    unsigned : NumTypeBits;

    /// CVR qualifiers from declarations like
    /// 'int X[static restrict 4]'. For function parameters only.
    unsigned IndexTypeQuals : 3;

    /// Storage class qualifiers from declarations like
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
    friend class FunctionProtoType;

    unsigned : NumTypeBits;

    /// Extra information which affects how the function is called, like
    /// regparm and the calling convention.
    unsigned ExtInfo : 9;

    /// Used only by FunctionProtoType, put here to pack with the
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

    /// The number of type arguments stored directly on this object type.
    unsigned NumTypeArgs : 7;

    /// The number of protocols stored directly on this object type.
    unsigned NumProtocols : 6;

    /// Whether this is a "kindof" type.
    unsigned IsKindOf : 1;
  };
  static_assert(NumTypeBits + 7 + 6 + 1 <= 32, "Does not fit in an unsigned");

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

    /// The kind of vector, either a generic vector type or some
    /// target-specific vector type such as for AltiVec or Neon.
    unsigned VecKind : 3;

    /// The number of elements in the vector.
    unsigned NumElements : 29 - NumTypeBits;

    enum { MaxNumElements = (1 << (29 - NumTypeBits)) - 1 };
  };

  class AttributedTypeBitfields {
    friend class AttributedType;

    unsigned : NumTypeBits;

    /// An AttributedType::Kind
    unsigned AttrKind : 32 - NumTypeBits;
  };

  class AutoTypeBitfields {
    friend class AutoType;

    unsigned : NumTypeBits;

    /// Was this placeholder type spelled as 'auto', 'decltype(auto)',
    /// or '__auto_type'?  AutoTypeKeyword value.
    unsigned Keyword : 2;
  };

  union {
    TypeBitfields TypeBits;
    ArrayTypeBitfields ArrayTypeBits;
    AttributedTypeBitfields AttributedTypeBits;
    AutoTypeBitfields AutoTypeBits;
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
  Type(TypeClass tc, QualType canon, bool Dependent,
       bool InstantiationDependent, bool VariablyModified,
       bool ContainsUnexpandedParameterPack)
    : ExtQualsTypeCommonBase(this,
                             canon.isNull() ? QualType(this_(), 0) : canon) {
    TypeBits.TC = tc;
    TypeBits.Dependent = Dependent;
    TypeBits.InstantiationDependent = Dependent || InstantiationDependent;
    TypeBits.VariablyModified = VariablyModified;
    TypeBits.ContainsUnexpandedParameterPack = ContainsUnexpandedParameterPack;
    TypeBits.CacheValid = false;
    TypeBits.CachedLocalOrUnnamed = false;
    TypeBits.CachedLinkage = NoLinkage;
    TypeBits.FromAST = false;
  }
  friend class ASTContext;

  void setDependent(bool D = true) {
    TypeBits.Dependent = D;
    if (D)
      TypeBits.InstantiationDependent = true;
  }
  void setInstantiationDependent(bool D = true) {
    TypeBits.InstantiationDependent = D; }
  void setVariablyModified(bool VM = true) { TypeBits.VariablyModified = VM;
  }
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

  /// Pull a single level of sugar off of this locally-unqualified type.
  /// Users should generally prefer SplitQualType::getSingleStepDesugaredType()
  /// or QualType::getSingleStepDesugaredType(const ASTContext&).
  QualType getLocallyUnqualifiedSingleStepDesugaredType() const;

  /// Types are partitioned into 3 broad categories (C99 6.2.5p1):
  /// object types, function types, and incomplete types.

  /// Return true if this is an incomplete type.
  /// A type that can describe objects, but which lacks information needed to
  /// determine its size (e.g. void, or a fwd declared struct). Clients of this
  /// routine will need to determine if the size is actually required.
  ///
  /// \brief Def If non-null, and the type refers to some kind of declaration
  /// that can be completed (such as a C struct, C++ class, or Objective-C
  /// class), will be set to the declaration.
  bool isIncompleteType(NamedDecl **Def = nullptr) const;

  /// Return true if this is an incomplete or object
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

  /// Return true if this is a literal type
  /// (C++11 [basic.types]p10)
  bool isLiteralType(const ASTContext &Ctx) const;

  /// Test if this type is a standard-layout type.
  /// (C++0x [basic.type]p9)
  bool isStandardLayoutType() const;

  /// Helper methods to distinguish type categories. All type predicates
  /// operate on the canonical type, ignoring typedefs and qualifiers.

  /// Returns true if the type is a builtin type.
  bool isBuiltinType() const;

  /// Test for a particular builtin type.
  bool isSpecificBuiltinType(unsigned K) const;

  /// Test for a type which does not represent an actual type-system type but
  /// is instead used as a placeholder for various convenient purposes within
  /// Clang.  All such types are BuiltinTypes.
  bool isPlaceholderType() const;
  const BuiltinType *getAsPlaceholderType() const;

  /// Test for a specific placeholder type.
  bool isSpecificPlaceholderType(unsigned K) const;

  /// Test for a placeholder type other than Overload; see
  /// BuiltinType::isNonOverloadPlaceholderType.
  bool isNonOverloadPlaceholderType() const;

  /// isIntegerType() does *not* include complex integers (a GCC extension).
  /// isComplexIntegerType() can be used to test for complex integers.
  bool isIntegerType() const;     // C99 6.2.5p17 (int, char, bool, enum)
  bool isEnumeralType() const;
  bool isBooleanType() const;
  bool isCharType() const;
  bool isWideCharType() const;
  bool isChar16Type() const;
  bool isChar32Type() const;
  bool isAnyCharacterType() const;
  bool isIntegralType(ASTContext &Ctx) const;

  /// Determine whether this type is an integral or enumeration type.
  bool isIntegralOrEnumerationType() const;
  /// Determine whether this type is an integral or unscoped enumeration type.
  bool isIntegralOrUnscopedEnumerationType() const;

  /// Floating point categories.
  bool isRealFloatingType() const; // C99 6.2.5p10 (float, double, long double)
  /// isComplexType() does *not* include complex integers (a GCC extension).
  /// isComplexIntegerType() can be used to test for complex integers.
  bool isComplexType() const;      // C99 6.2.5p11 (complex)
  bool isAnyComplexType() const;   // C99 6.2.5p11 (complex) + Complex Int.
  bool isFloatingType() const;     // C99 6.2.5p11 (real floating + complex)
  bool isHalfType() const;         // OpenCL 6.1.1.1, NEON (IEEE 754-2008 half)
  bool isRealType() const;         // C99 6.2.5p17 (real floating + integer)
  bool isArithmeticType() const;   // C99 6.2.5p18 (integer + floating)
  bool isVoidType() const;         // C99 6.2.5p19
  bool isScalarType() const;       // C99 6.2.5p21 (arithmetic + pointers)
  bool isAggregateType() const;
  bool isFundamentalType() const;
  bool isCompoundType() const;

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
  bool isObjCBoxableRecordType() const;
  bool isInterfaceType() const;
  bool isStructureOrClassType() const;
  bool isUnionType() const;
  bool isComplexIntegerType() const;            // GCC _Complex integer type.
  bool isVectorType() const;                    // GCC vector type.
  bool isExtVectorType() const;                 // Extended vector type.
  bool isObjCObjectPointerType() const;         // pointer to ObjC object
  bool isObjCRetainableType() const;            // ObjC object or block pointer
  bool isObjCLifetimeType() const;              // (array of)* retainable type
  bool isObjCIndirectLifetimeType() const;      // (pointer to)* lifetime type
  bool isObjCNSObjectType() const;              // __attribute__((NSObject))
  bool isObjCIndependentClassType() const;      // __attribute__((objc_independent_class))
  // FIXME: change this to 'raw' interface type, so we can used 'interface' type
  // for the common case.
  bool isObjCObjectType() const;                // NSString or typeof(*(id)0)
  bool isObjCQualifiedInterfaceType() const;    // NSString<foo>
  bool isObjCQualifiedIdType() const;           // id<foo>
  bool isObjCQualifiedClassType() const;        // Class<foo>
  bool isObjCObjectOrInterfaceType() const;
  bool isObjCIdType() const;                    // id

  /// Whether the type is Objective-C 'id' or a __kindof type of an
  /// object type, e.g., __kindof NSView * or __kindof id
  /// <NSCopying>.
  ///
  /// \param bound Will be set to the bound on non-id subtype types,
  /// which will be (possibly specialized) Objective-C class type, or
  /// null for 'id.
  bool isObjCIdOrObjectKindOfType(const ASTContext &ctx,
                                  const ObjCObjectType *&bound) const;

  bool isObjCClassType() const;                 // Class

  /// Whether the type is Objective-C 'Class' or a __kindof type of an
  /// Class type, e.g., __kindof Class <NSCopying>.
  ///
  /// Unlike \c isObjCIdOrObjectKindOfType, there is no relevant bound
  /// here because Objective-C's type system cannot express "a class
  /// object for a subclass of NSFoo".
  bool isObjCClassOrClassKindOfType() const;

  bool isBlockCompatibleObjCPointerType(ASTContext &ctx) const;
  bool isObjCSelType() const;                 // Class
  bool isObjCBuiltinType() const;               // 'id' or 'Class'
  bool isObjCARCBridgableType() const;
  bool isCARCBridgableType() const;
  bool isTemplateTypeParmType() const;          // C++ template type parameter
  bool isNullPtrType() const;                   // C++0x nullptr_t
  bool isAtomicType() const;                    // C11 _Atomic()

  bool isImage1dT() const;               // OpenCL image1d_t
  bool isImage1dArrayT() const;          // OpenCL image1d_array_t
  bool isImage1dBufferT() const;         // OpenCL image1d_buffer_t
  bool isImage2dT() const;               // OpenCL image2d_t
  bool isImage2dArrayT() const;          // OpenCL image2d_array_t
  bool isImage2dDepthT() const;          // OpenCL image_2d_depth_t
  bool isImage2dArrayDepthT() const;     // OpenCL image_2d_array_depth_t
  bool isImage2dMSAAT() const;           // OpenCL image_2d_msaa_t
  bool isImage2dArrayMSAAT() const;      // OpenCL image_2d_array_msaa_t
  bool isImage2dMSAATDepth() const;      // OpenCL image_2d_msaa_depth_t
  bool isImage2dArrayMSAATDepth() const; // OpenCL image_2d_array_msaa_depth_t
  bool isImage3dT() const;               // OpenCL image3d_t

  bool isImageType() const;                     // Any OpenCL image type

  bool isSamplerT() const;                      // OpenCL sampler_t
  bool isEventT() const;                        // OpenCL event_t
  bool isClkEventT() const;                     // OpenCL clk_event_t
  bool isQueueT() const;                        // OpenCL queue_t
  bool isNDRangeT() const;                      // OpenCL ndrange_t
  bool isReserveIDT() const;                    // OpenCL reserve_id_t

  bool isOpenCLSpecificType() const;            // Any OpenCL specific type

  /// Determines if this type, which must satisfy
  /// isObjCLifetimeType(), is implicitly __unsafe_unretained rather
  /// than implicitly __strong.
  bool isObjCARCImplicitlyUnretainedType() const;

  /// Return the implicit lifetime for this type, which must not be dependent.
  Qualifiers::ObjCLifetime getObjCARCImplicitLifetime() const;

  enum ScalarTypeKind {
    STK_CPointer,
    STK_BlockPointer,
    STK_ObjCObjectPointer,
    STK_MemberPointer,
    STK_Bool,
    STK_Integral,
    STK_Floating,
    STK_IntegralComplex,
    STK_FloatingComplex
  };
  /// Given that this is a scalar type, classify it.
  ScalarTypeKind getScalarTypeKind() const;

  /// Whether this type is a dependent type, meaning that its definition
  /// somehow depends on a template parameter (C++ [temp.dep.type]).
  bool isDependentType() const { return TypeBits.Dependent; }

  /// \brief Determine whether this type is an instantiation-dependent type,
  /// meaning that the type involves a template parameter (even if the
  /// definition does not actually depend on the type substituted for that
  /// template parameter).
  bool isInstantiationDependentType() const {
    return TypeBits.InstantiationDependent;
  }

  /// \brief Determine whether this type is an undeduced type, meaning that
  /// it somehow involves a C++11 'auto' type which has not yet been deduced.
  bool isUndeducedType() const;

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

  bool canDecayToPointerType() const;

  /// Whether this type is represented natively as a pointer.  This includes
  /// pointers, references, block pointers, and Objective-C interface,
  /// qualified id, and qualified interface types, as well as nullptr_t.
  bool hasPointerRepresentation() const;

  /// Whether this type can represent an objective pointer type for the
  /// purpose of GC'ability
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
  const ObjCObjectType *getAsObjCInterfaceType() const;
  // The following is a convenience method that returns an ObjCObjectPointerType
  // for object declared using an interface.
  const ObjCObjectPointerType *getAsObjCInterfacePointerType() const;
  const ObjCObjectPointerType *getAsObjCQualifiedIdType() const;
  const ObjCObjectPointerType *getAsObjCQualifiedClassType() const;
  const ObjCObjectType *getAsObjCQualifiedInterfaceType() const;

  /// \brief Retrieves the CXXRecordDecl that this type refers to, either
  /// because the type is a RecordType or because it is the injected-class-name
  /// type of a class template or class template partial specialization.
  CXXRecordDecl *getAsCXXRecordDecl() const;

  /// \brief Retrieves the TagDecl that this type refers to, either
  /// because the type is a TagType or because it is the injected-class-name
  /// type of a class template or class template partial specialization.
  TagDecl *getAsTagDecl() const;

  /// If this is a pointer or reference to a RecordType, return the
  /// CXXRecordDecl that that type refers to.
  ///
  /// If this is not a pointer or reference, or the type being pointed to does
  /// not refer to a CXXRecordDecl, returns NULL.
  const CXXRecordDecl *getPointeeCXXRecordDecl() const;

  /// Get the AutoType whose type will be deduced for a variable with
  /// an initializer of this type. This looks through declarators like pointer
  /// types, but not through decltype or typedefs.
  AutoType *getContainedAutoType() const;

  /// Member-template getAs<specific type>'.  Look through sugar for
  /// an instance of \<specific type>.   This scheme will eventually
  /// replace the specific getAsXXXX methods above.
  ///
  /// There are some specializations of this member template listed
  /// immediately following this class.
  template <typename T> const T *getAs() const;

  /// A variant of getAs<> for array types which silently discards
  /// qualifiers from the outermost type.
  const ArrayType *getAsArrayTypeUnsafe() const;

  /// Member-template castAs<specific type>.  Look through sugar for
  /// the underlying instance of \<specific type>.
  ///
  /// This method has the same relationship to getAs<T> as cast<T> has
  /// to dyn_cast<T>; which is to say, the underlying type *must*
  /// have the intended type, and this method will never return null.
  template <typename T> const T *castAs() const;

  /// A variant of castAs<> for array type which silently discards
  /// qualifiers from the outermost type.
  const ArrayType *castAsArrayTypeUnsafe() const;

  /// Get the base element type of this type, potentially discarding type
  /// qualifiers.  This should never be used when type qualifiers
  /// are meaningful.
  const Type *getBaseElementTypeUnsafe() const;

  /// If this is an array type, return the element type of the array,
  /// potentially with type qualifiers missing.
  /// This should never be used when type qualifiers are meaningful.
  const Type *getArrayElementTypeNoTypeQual() const;

  /// If this is a pointer, ObjC object pointer, or block
  /// pointer, this returns the respective pointee.
  QualType getPointeeType() const;

  /// Return the specified type with any "sugar" removed from the type,
  /// removing any typedefs, typeofs, etc., as well as any qualifiers.
  const Type *getUnqualifiedDesugaredType() const;

  /// More type predicates useful for type checking/promotion
  bool isPromotableIntegerType() const; // C99 6.3.1.1p2

  /// Return true if this is an integer type that is
  /// signed, according to C99 6.2.5p4 [char, signed char, short, int, long..],
  /// or an enum decl which has a signed representation.
  bool isSignedIntegerType() const;

  /// Return true if this is an integer type that is
  /// unsigned, according to C99 6.2.5p6 [which returns true for _Bool],
  /// or an enum decl which has an unsigned representation.
  bool isUnsignedIntegerType() const;

  /// Determines whether this is an integer type that is signed or an
  /// enumeration types whose underlying type is a signed integer type.
  bool isSignedIntegerOrEnumerationType() const;

  /// Determines whether this is an integer type that is unsigned or an
  /// enumeration types whose underlying type is a unsigned integer type.
  bool isUnsignedIntegerOrEnumerationType() const;

  /// Return true if this is not a variable sized type,
  /// according to the rules of C99 6.7.5p3.  It is not legal to call this on
  /// incomplete types.
  bool isConstantSizeType() const;

  /// Returns true if this type can be represented by some
  /// set of type specifiers.
  bool isSpecifierType() const;

  /// Determine the linkage of this type.
  Linkage getLinkage() const;

  /// Determine the visibility of this type.
  Visibility getVisibility() const {
    return getLinkageAndVisibility().getVisibility();
  }

  /// Return true if the visibility was explicitly set is the code.
  bool isVisibilityExplicit() const {
    return getLinkageAndVisibility().isVisibilityExplicit();
  }

  /// Determine the linkage and visibility of this type.
  LinkageInfo getLinkageAndVisibility() const;

  /// True if the computed linkage is valid. Used for consistency
  /// checking. Should always return true.
  bool isLinkageValid() const;

  /// Determine the nullability of the given type.
  ///
  /// Note that nullability is only captured as sugar within the type
  /// system, not as part of the canonical type, so nullability will
  /// be lost by canonicalization and desugaring.
  Optional<NullabilityKind> getNullability(const ASTContext &context) const;

  /// Determine whether the given type can have a nullability
  /// specifier applied to it, i.e., if it is any kind of pointer type
  /// or a dependent type that could instantiate to any kind of
  /// pointer type.
  bool canHaveNullability() const;

  /// Retrieve the set of substitutions required when accessing a member
  /// of the Objective-C receiver type that is declared in the given context.
  ///
  /// \c *this is the type of the object we're operating on, e.g., the
  /// receiver for a message send or the base of a property access, and is
  /// expected to be of some object or object pointer type.
  ///
  /// \param dc The declaration context for which we are building up a
  /// substitution mapping, which should be an Objective-C class, extension,
  /// category, or method within.
  ///
  /// \returns an array of type arguments that can be substituted for
  /// the type parameters of the given declaration context in any type described
  /// within that context, or an empty optional to indicate that no
  /// substitution is required.
  Optional<ArrayRef<QualType>>
  getObjCSubstitutions(const DeclContext *dc) const;

  /// Determines if this is an ObjC interface type that may accept type
  /// parameters.
  bool acceptsObjCTypeParams() const;

  const char *getTypeClassName() const;

  QualType getCanonicalTypeInternal() const {
    return CanonicalType;
  }
  CanQualType getCanonicalTypeUnqualified() const; // in CanonicalType.h
  void dump() const;

  friend class ASTReader;
  friend class ASTWriter;
};

/// \brief This will check for a TypedefType by removing any existing sugar
/// until it reaches a TypedefType or a non-sugared type.
template <> const TypedefType *Type::getAs() const;

/// \brief This will check for a TemplateSpecializationType by removing any
/// existing sugar until it reaches a TemplateSpecializationType or a
/// non-sugared type.
template <> const TemplateSpecializationType *Type::getAs() const;

/// \brief This will check for an AttributedType by removing any existing sugar
/// until it reaches an AttributedType or a non-sugared type.
template <> const AttributedType *Type::getAs() const;

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


/// This class is used for builtin types like 'int'.  Builtin
/// types are always canonical and have a literal name field.
class BuiltinType : public Type {
public:
  enum Kind {
#define BUILTIN_TYPE(Id, SingletonId) Id,
#define LAST_BUILTIN_TYPE(Id) LastKind = Id
#include "clang/AST/BuiltinTypes.def"
  };

public:
  BuiltinType(Kind K)
    : Type(Builtin, QualType(), /*Dependent=*/(K == Dependent),
           /*InstantiationDependent=*/(K == Dependent),
           /*VariablyModified=*/false,
           /*Unexpanded paramter pack=*/false) {
    BuiltinTypeBits.Kind = K;
  }

  Kind getKind() const { return static_cast<Kind>(BuiltinTypeBits.Kind); }
  StringRef getName(const PrintingPolicy &Policy) const;
  const char *getNameAsCString(const PrintingPolicy &Policy) const {
    // The StringRef is null-terminated.
    StringRef str = getName(Policy);
    assert(!str.empty() && str.data()[str.size()] == '\0');
    return str.data();
  }

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
    return getKind() >= Half && getKind() <= LongDouble;
  }

  /// Determines whether the given kind corresponds to a placeholder type.
  static bool isPlaceholderTypeKind(Kind K) {
    return K >= Overload;
  }

  /// Determines whether this type is a placeholder type, i.e. a type
  /// which cannot appear in arbitrary positions in a fully-formed
  /// expression.
  bool isPlaceholderType() const {
    return isPlaceholderTypeKind(getKind());
  }

  /// Determines whether this type is a placeholder type other than
  /// Overload.  Most placeholder types require only syntactic
  /// information about their context in order to be resolved (e.g.
  /// whether it is a call expression), which means they can (and
  /// should) be resolved in an earlier "phase" of analysis.
  /// Overload expressions sometimes pick up further information
  /// from their context, like whether the context expects a
  /// specific function-pointer type, and so frequently need
  /// special treatment.
  bool isNonOverloadPlaceholderType() const {
    return getKind() > Overload;
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Builtin; }
};

/// Complex values, per C99 6.2.5p11.  This supports the C99 complex
/// types (_Complex float etc) as well as the GCC integer complex extensions.
///
class ComplexType : public Type, public llvm::FoldingSetNode {
  QualType ElementType;
  ComplexType(QualType Element, QualType CanonicalPtr) :
    Type(Complex, CanonicalPtr, Element->isDependentType(),
         Element->isInstantiationDependentType(),
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
};

/// Sugar for parentheses used when specifying types.
///
class ParenType : public Type, public llvm::FoldingSetNode {
  QualType Inner;

  ParenType(QualType InnerType, QualType CanonType) :
    Type(Paren, CanonType, InnerType->isDependentType(),
         InnerType->isInstantiationDependentType(),
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
};

/// PointerType - C99 6.7.5.1 - Pointer Declarators.
///
class PointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;

  PointerType(QualType Pointee, QualType CanonicalPtr) :
    Type(Pointer, CanonicalPtr, Pointee->isDependentType(),
         Pointee->isInstantiationDependentType(),
         Pointee->isVariablyModifiedType(),
         Pointee->containsUnexpandedParameterPack()),
    PointeeType(Pointee) {
  }
  friend class ASTContext;  // ASTContext creates these.

public:

  QualType getPointeeType() const { return PointeeType; }

  /// Returns true if address spaces of pointers overlap.
  /// OpenCL v2.0 defines conversion rules for pointers to different
  /// address spaces (OpenCLC v2.0 s6.5.5) and notion of overlapping
  /// address spaces.
  /// CL1.1 or CL1.2:
  ///   address spaces overlap iff they are they same.
  /// CL2.0 adds:
  ///   __generic overlaps with any address space except for __constant.
  bool isAddressSpaceOverlapping(const PointerType &other) const {
    Qualifiers thisQuals = PointeeType.getQualifiers();
    Qualifiers otherQuals = other.getPointeeType().getQualifiers();
    // Address spaces overlap if at least one of them is a superset of another
    return thisQuals.isAddressSpaceSupersetOf(otherQuals) ||
           otherQuals.isAddressSpaceSupersetOf(thisQuals);
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pointee) {
    ID.AddPointer(Pointee.getAsOpaquePtr());
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Pointer; }
};

/// Represents a type which was implicitly adjusted by the semantic
/// engine for arbitrary reasons.  For example, array and function types can
/// decay, and function types can have their calling conventions adjusted.
class AdjustedType : public Type, public llvm::FoldingSetNode {
  QualType OriginalTy;
  QualType AdjustedTy;

protected:
  AdjustedType(TypeClass TC, QualType OriginalTy, QualType AdjustedTy,
               QualType CanonicalPtr)
      : Type(TC, CanonicalPtr, OriginalTy->isDependentType(),
             OriginalTy->isInstantiationDependentType(),
             OriginalTy->isVariablyModifiedType(),
             OriginalTy->containsUnexpandedParameterPack()),
        OriginalTy(OriginalTy), AdjustedTy(AdjustedTy) {}

  friend class ASTContext;  // ASTContext creates these.

public:
  QualType getOriginalType() const { return OriginalTy; }
  QualType getAdjustedType() const { return AdjustedTy; }

  bool isSugared() const { return true; }
  QualType desugar() const { return AdjustedTy; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, OriginalTy, AdjustedTy);
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Orig, QualType New) {
    ID.AddPointer(Orig.getAsOpaquePtr());
    ID.AddPointer(New.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == Adjusted || T->getTypeClass() == Decayed;
  }
};

/// Represents a pointer type decayed from an array or function type.
class DecayedType : public AdjustedType {

  DecayedType(QualType OriginalType, QualType DecayedPtr, QualType CanonicalPtr)
      : AdjustedType(Decayed, OriginalType, DecayedPtr, CanonicalPtr) {
    assert(isa<PointerType>(getAdjustedType()));
  }

  friend class ASTContext;  // ASTContext creates these.

public:
  QualType getDecayedType() const { return getAdjustedType(); }

  QualType getPointeeType() const {
    return cast<PointerType>(getDecayedType())->getPointeeType();
  }

  static bool classof(const Type *T) { return T->getTypeClass() == Decayed; }
};

/// Pointer to a block type.
/// This type is to represent types syntactically represented as
/// "void (^)(int)", etc. Pointee is required to always be a function type.
///
class BlockPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;  // Block is some kind of pointer type
  BlockPointerType(QualType Pointee, QualType CanonicalCls) :
    Type(BlockPointer, CanonicalCls, Pointee->isDependentType(),
         Pointee->isInstantiationDependentType(),
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
};

/// Base for LValueReferenceType and RValueReferenceType
///
class ReferenceType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;

protected:
  ReferenceType(TypeClass tc, QualType Referencee, QualType CanonicalRef,
                bool SpelledAsLValue) :
    Type(tc, CanonicalRef, Referencee->isDependentType(),
         Referencee->isInstantiationDependentType(),
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
};

/// An lvalue reference type, per C++11 [dcl.ref].
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
};

/// An rvalue reference type, per C++11 [dcl.ref].
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
};

/// A pointer to member type per C++ 8.3.3 - Pointers to members.
///
/// This includes both pointers to data members and pointer to member functions.
///
class MemberPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;
  /// The class of which the pointee is a member. Must ultimately be a
  /// RecordType, but could be a typedef or a template parameter too.
  const Type *Class;

  MemberPointerType(QualType Pointee, const Type *Cls, QualType CanonicalPtr) :
    Type(MemberPointer, CanonicalPtr,
         Cls->isDependentType() || Pointee->isDependentType(),
         (Cls->isInstantiationDependentType() ||
          Pointee->isInstantiationDependentType()),
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
  CXXRecordDecl *getMostRecentCXXRecordDecl() const;

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
};

/// Represents an array type, per C99 6.7.5.2 - Array Declarators.
///
class ArrayType : public Type, public llvm::FoldingSetNode {
public:
  /// Capture whether this is a normal array (e.g. int X[4])
  /// an array with a static size (e.g. int X[static 4]), or an array
  /// with a star size (e.g. int X[*]).
  /// 'static' is only allowed on function parameters.
  enum ArraySizeModifier {
    Normal, Static, Star
  };
private:
  /// The element type of the array.
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
           et->isInstantiationDependentType() || tc == DependentSizedArray,
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
};

/// Represents the canonical version of C arrays with a specified constant size.
/// For example, the canonical type for 'int A[4 + 4*100]' is a
/// ConstantArrayType where the element type is 'int' and the size is 404.
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
};

/// Represents a C array with an unspecified size.  For example 'int A[]' has
/// an IncompleteArrayType where the element type is 'int' and the size is
/// unspecified.
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

/// Represents a C array with a specified size that is not an
/// integer-constant-expression.  For example, 'int s[x+foo()]'.
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
  /// An assignment-expression. VLA's are only permitted within
  /// a function block.
  Stmt *SizeExpr;
  /// The range spanned by the left and right array brackets.
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

  friend class StmtIteratorBase;

  void Profile(llvm::FoldingSetNodeID &ID) {
    llvm_unreachable("Cannot unique VariableArrayTypes.");
  }
};

/// Represents an array type in C++ whose size is a value-dependent expression.
///
/// For example:
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
  /// The expression itself might be null, in which case the array
  /// type will have its size deduced from an initializer.
  Stmt *SizeExpr;

  /// The range spanned by the left and right array brackets.
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

  friend class StmtIteratorBase;


  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getElementType(),
            getSizeModifier(), getIndexTypeCVRQualifiers(), getSizeExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      QualType ET, ArraySizeModifier SizeMod,
                      unsigned TypeQuals, Expr *E);
};

/// Represents an extended vector type where either the type or size is
/// dependent.
///
/// For example:
/// \code
/// template<typename T, int Size>
/// class vector {
///   typedef T __attribute__((ext_vector_type(Size))) type;
/// }
/// \endcode
class DependentSizedExtVectorType : public Type, public llvm::FoldingSetNode {
  const ASTContext &Context;
  Expr *SizeExpr;
  /// The element type of the array.
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

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getElementType(), getSizeExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      QualType ElementType, Expr *SizeExpr);
};


/// Represents a GCC generic vector type. This type is created using
/// __attribute__((vector_size(n)), where "n" specifies the vector size in
/// bytes; or from an Altivec __vector or vector declaration.
/// Since the constructor takes the number of vector elements, the
/// client is responsible for converting the size into the number of elements.
class VectorType : public Type, public llvm::FoldingSetNode {
public:
  enum VectorKind {
    GenericVector,  ///< not a target-specific vector type
    AltiVecVector,  ///< is AltiVec vector
    AltiVecPixel,   ///< is AltiVec 'vector Pixel'
    AltiVecBool,    ///< is AltiVec 'vector bool ...'
    NeonVector,     ///< is ARM Neon vector
    NeonPolyVector  ///< is ARM Neon polynomial vector
  };
protected:
  /// The element type of the vector.
  QualType ElementType;

  VectorType(QualType vecType, unsigned nElements, QualType canonType,
             VectorKind vecKind);

  VectorType(TypeClass tc, QualType vecType, unsigned nElements,
             QualType canonType, VectorKind vecKind);

  friend class ASTContext;  // ASTContext creates these.

public:

  QualType getElementType() const { return ElementType; }
  unsigned getNumElements() const { return VectorTypeBits.NumElements; }
  static bool isVectorSizeTooLarge(unsigned NumElements) {
    return NumElements > VectorTypeBitfields::MaxNumElements;
  }

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
};

/// FunctionType - C99 6.7.5.3 - Function Declarators.  This is the common base
/// class of FunctionNoProtoType and FunctionProtoType.
///
class FunctionType : public Type {
  // The type returned by the function.
  QualType ResultType;

 public:
  /// A class which abstracts out some details necessary for
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
    // Feel free to rearrange or add bits, but if you go over 9,
    // you'll need to adjust both the Bits field below and
    // Type::FunctionTypeBitfields.

    //   |  CC  |noreturn|produces|regparm|
    //   |0 .. 3|   4    |    5   | 6 .. 8|
    //
    // regparm is either 0 (no regparm attribute) or the regparm value+1.
    enum { CallConvMask = 0xF };
    enum { NoReturnMask = 0x10 };
    enum { ProducesResultMask = 0x20 };
    enum { RegParmMask = ~(CallConvMask | NoReturnMask | ProducesResultMask),
           RegParmOffset = 6 }; // Assumed to be the last field

    uint16_t Bits;

    ExtInfo(unsigned Bits) : Bits(static_cast<uint16_t>(Bits)) {}

    friend class FunctionType;

   public:
    // Constructor with no defaults. Use this when you know that you
    // have all the elements (when reading an AST file for example).
    ExtInfo(bool noReturn, bool hasRegParm, unsigned regParm, CallingConv cc,
            bool producesResult) {
      assert((!hasRegParm || regParm < 7) && "Invalid regparm value");
      Bits = ((unsigned) cc) |
             (noReturn ? NoReturnMask : 0) |
             (producesResult ? ProducesResultMask : 0) |
             (hasRegParm ? ((regParm + 1) << RegParmOffset) : 0);
    }

    // Constructor with all defaults. Use when for example creating a
    // function known to use defaults.
    ExtInfo() : Bits(CC_C) { }

    // Constructor with just the calling convention, which is an important part
    // of the canonical type.
    ExtInfo(CallingConv CC) : Bits(CC) { }

    bool getNoReturn() const { return Bits & NoReturnMask; }
    bool getProducesResult() const { return Bits & ProducesResultMask; }
    bool getHasRegParm() const { return (Bits >> RegParmOffset) != 0; }
    unsigned getRegParm() const {
      unsigned RegParm = Bits >> RegParmOffset;
      if (RegParm > 0)
        --RegParm;
      return RegParm;
    }
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

    ExtInfo withProducesResult(bool producesResult) const {
      if (producesResult)
        return ExtInfo(Bits | ProducesResultMask);
      else
        return ExtInfo(Bits & ~ProducesResultMask);
    }

    ExtInfo withRegParm(unsigned RegParm) const {
      assert(RegParm < 7 && "Invalid regparm value");
      return ExtInfo((Bits & ~RegParmMask) |
                     ((RegParm + 1) << RegParmOffset));
    }

    ExtInfo withCallingConv(CallingConv cc) const {
      return ExtInfo((Bits & ~CallConvMask) | (unsigned) cc);
    }

    void Profile(llvm::FoldingSetNodeID &ID) const {
      ID.AddInteger(Bits);
    }
  };

protected:
  FunctionType(TypeClass tc, QualType res,
               QualType Canonical, bool Dependent,
               bool InstantiationDependent,
               bool VariablyModified, bool ContainsUnexpandedParameterPack,
               ExtInfo Info)
    : Type(tc, Canonical, Dependent, InstantiationDependent, VariablyModified,
           ContainsUnexpandedParameterPack),
      ResultType(res) {
    FunctionTypeBits.ExtInfo = Info.Bits;
  }
  unsigned getTypeQuals() const { return FunctionTypeBits.TypeQuals; }

public:
  QualType getReturnType() const { return ResultType; }

  bool getHasRegParm() const { return getExtInfo().getHasRegParm(); }
  unsigned getRegParmType() const { return getExtInfo().getRegParm(); }
  /// Determine whether this function type includes the GNU noreturn
  /// attribute. The C++11 [[noreturn]] attribute does not affect the function
  /// type.
  bool getNoReturnAttr() const { return getExtInfo().getNoReturn(); }
  CallingConv getCallConv() const { return getExtInfo().getCC(); }
  ExtInfo getExtInfo() const { return ExtInfo(FunctionTypeBits.ExtInfo); }
  bool isConst() const { return getTypeQuals() & Qualifiers::Const; }
  bool isVolatile() const { return getTypeQuals() & Qualifiers::Volatile; }
  bool isRestrict() const { return getTypeQuals() & Qualifiers::Restrict; }

  /// \brief Determine the type of an expression that calls a function of
  /// this type.
  QualType getCallResultType(ASTContext &Context) const {
    return getReturnType().getNonLValueExprType(Context);
  }

  static StringRef getNameForCallConv(CallingConv CC);

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto ||
           T->getTypeClass() == FunctionProto;
  }
};

/// Represents a K&R-style 'int foo()' function, which has
/// no information available about its arguments.
class FunctionNoProtoType : public FunctionType, public llvm::FoldingSetNode {
  FunctionNoProtoType(QualType Result, QualType Canonical, ExtInfo Info)
    : FunctionType(FunctionNoProto, Result, Canonical,
                   /*Dependent=*/false, /*InstantiationDependent=*/false,
                   Result->isVariablyModifiedType(),
                   /*ContainsUnexpandedParameterPack=*/false, Info) {}

  friend class ASTContext;  // ASTContext creates these.

public:
  // No additional state past what FunctionType provides.

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getReturnType(), getExtInfo());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType ResultType,
                      ExtInfo Info) {
    Info.Profile(ID);
    ID.AddPointer(ResultType.getAsOpaquePtr());
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionNoProto;
  }
};

/// Represents a prototype with parameter type info, e.g.
/// 'int foo(int)' or 'int foo(void)'.  'void' is represented as having no
/// parameters, not as having a single void parameter. Such a type can have an
/// exception specification, but this specification is not part of the canonical
/// type.
class FunctionProtoType : public FunctionType, public llvm::FoldingSetNode {
public:
  struct ExceptionSpecInfo {
    ExceptionSpecInfo()
        : Type(EST_None), NoexceptExpr(nullptr),
          SourceDecl(nullptr), SourceTemplate(nullptr) {}

    ExceptionSpecInfo(ExceptionSpecificationType EST)
        : Type(EST), NoexceptExpr(nullptr), SourceDecl(nullptr),
          SourceTemplate(nullptr) {}

    /// The kind of exception specification this is.
    ExceptionSpecificationType Type;
    /// Explicitly-specified list of exception types.
    ArrayRef<QualType> Exceptions;
    /// Noexcept expression, if this is EST_ComputedNoexcept.
    Expr *NoexceptExpr;
    /// The function whose exception specification this is, for
    /// EST_Unevaluated and EST_Uninstantiated.
    FunctionDecl *SourceDecl;
    /// The function template whose exception specification this is instantiated
    /// from, for EST_Uninstantiated.
    FunctionDecl *SourceTemplate;
  };

  /// Extra information about a function prototype.
  struct ExtProtoInfo {
    ExtProtoInfo()
        : Variadic(false), HasTrailingReturn(false), TypeQuals(0),
          RefQualifier(RQ_None), ConsumedParameters(nullptr) {}

    ExtProtoInfo(CallingConv CC)
        : ExtInfo(CC), Variadic(false), HasTrailingReturn(false), TypeQuals(0),
          RefQualifier(RQ_None), ConsumedParameters(nullptr) {}

    ExtProtoInfo withExceptionSpec(const ExceptionSpecInfo &O) {
      ExtProtoInfo Result(*this);
      Result.ExceptionSpec = O;
      return Result;
    }

    FunctionType::ExtInfo ExtInfo;
    bool Variadic : 1;
    bool HasTrailingReturn : 1;
    unsigned char TypeQuals;
    RefQualifierKind RefQualifier;
    ExceptionSpecInfo ExceptionSpec;
    const bool *ConsumedParameters;
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

  FunctionProtoType(QualType result, ArrayRef<QualType> params,
                    QualType canonical, const ExtProtoInfo &epi);

  /// The number of parameters this function has, not counting '...'.
  unsigned NumParams : 15;

  /// The number of types in the exception spec, if any.
  unsigned NumExceptions : 9;

  /// The type of exception specification this function has.
  unsigned ExceptionSpecType : 4;

  /// Whether this function has any consumed parameters.
  unsigned HasAnyConsumedParams : 1;

  /// Whether the function is variadic.
  unsigned Variadic : 1;

  /// Whether this function has a trailing return type.
  unsigned HasTrailingReturn : 1;

  // ParamInfo - There is an variable size array after the class in memory that
  // holds the parameter types.

  // Exceptions - There is another variable size array after ArgInfo that
  // holds the exception types.

  // NoexceptExpr - Instead of Exceptions, there may be a single Expr* pointing
  // to the expression in the noexcept() specifier.

  // ExceptionSpecDecl, ExceptionSpecTemplate - Instead of Exceptions, there may
  // be a pair of FunctionDecl* pointing to the function which should be used to
  // instantiate this function type's exception specification, and the function
  // from which it should be instantiated.

  // ConsumedParameters - A variable size array, following Exceptions
  // and of length NumParams, holding flags indicating which parameters
  // are consumed.  This only appears if HasAnyConsumedParams is true.

  friend class ASTContext;  // ASTContext creates these.

  const bool *getConsumedParamsBuffer() const {
    assert(hasAnyConsumedParams());

    // Find the end of the exceptions.
    Expr *const *eh_end = reinterpret_cast<Expr *const *>(exception_end());
    if (getExceptionSpecType() == EST_ComputedNoexcept)
      eh_end += 1; // NoexceptExpr
    // The memory layout of these types isn't handled here, so
    // hopefully this is never called for them?
    assert(getExceptionSpecType() != EST_Uninstantiated &&
           getExceptionSpecType() != EST_Unevaluated);

    return reinterpret_cast<const bool*>(eh_end);
  }

public:
  unsigned getNumParams() const { return NumParams; }
  QualType getParamType(unsigned i) const {
    assert(i < NumParams && "invalid parameter index");
    return param_type_begin()[i];
  }
  ArrayRef<QualType> getParamTypes() const {
    return llvm::makeArrayRef(param_type_begin(), param_type_end());
  }

  ExtProtoInfo getExtProtoInfo() const {
    ExtProtoInfo EPI;
    EPI.ExtInfo = getExtInfo();
    EPI.Variadic = isVariadic();
    EPI.HasTrailingReturn = hasTrailingReturn();
    EPI.ExceptionSpec.Type = getExceptionSpecType();
    EPI.TypeQuals = static_cast<unsigned char>(getTypeQuals());
    EPI.RefQualifier = getRefQualifier();
    if (EPI.ExceptionSpec.Type == EST_Dynamic) {
      EPI.ExceptionSpec.Exceptions = exceptions();
    } else if (EPI.ExceptionSpec.Type == EST_ComputedNoexcept) {
      EPI.ExceptionSpec.NoexceptExpr = getNoexceptExpr();
    } else if (EPI.ExceptionSpec.Type == EST_Uninstantiated) {
      EPI.ExceptionSpec.SourceDecl = getExceptionSpecDecl();
      EPI.ExceptionSpec.SourceTemplate = getExceptionSpecTemplate();
    } else if (EPI.ExceptionSpec.Type == EST_Unevaluated) {
      EPI.ExceptionSpec.SourceDecl = getExceptionSpecDecl();
    }
    if (hasAnyConsumedParams())
      EPI.ConsumedParameters = getConsumedParamsBuffer();
    return EPI;
  }

  /// Get the kind of exception specification on this function.
  ExceptionSpecificationType getExceptionSpecType() const {
    return static_cast<ExceptionSpecificationType>(ExceptionSpecType);
  }
  /// Return whether this function has any kind of exception spec.
  bool hasExceptionSpec() const {
    return getExceptionSpecType() != EST_None;
  }
  /// Return whether this function has a dynamic (throw) exception spec.
  bool hasDynamicExceptionSpec() const {
    return isDynamicExceptionSpec(getExceptionSpecType());
  }
  /// Return whether this function has a noexcept exception spec.
  bool hasNoexceptExceptionSpec() const {
    return isNoexceptExceptionSpec(getExceptionSpecType());
  }
  /// Return whether this function has a dependent exception spec.
  bool hasDependentExceptionSpec() const;
  /// Result type of getNoexceptSpec().
  enum NoexceptResult {
    NR_NoNoexcept,  ///< There is no noexcept specifier.
    NR_BadNoexcept, ///< The noexcept specifier has a bad expression.
    NR_Dependent,   ///< The noexcept specifier is dependent.
    NR_Throw,       ///< The noexcept specifier evaluates to false.
    NR_Nothrow      ///< The noexcept specifier evaluates to true.
  };
  /// Get the meaning of the noexcept spec on this function, if any.
  NoexceptResult getNoexceptSpec(const ASTContext &Ctx) const;
  unsigned getNumExceptions() const { return NumExceptions; }
  QualType getExceptionType(unsigned i) const {
    assert(i < NumExceptions && "Invalid exception number!");
    return exception_begin()[i];
  }
  Expr *getNoexceptExpr() const {
    if (getExceptionSpecType() != EST_ComputedNoexcept)
      return nullptr;
    // NoexceptExpr sits where the arguments end.
    return *reinterpret_cast<Expr *const *>(param_type_end());
  }
  /// \brief If this function type has an exception specification which hasn't
  /// been determined yet (either because it has not been evaluated or because
  /// it has not been instantiated), this is the function whose exception
  /// specification is represented by this type.
  FunctionDecl *getExceptionSpecDecl() const {
    if (getExceptionSpecType() != EST_Uninstantiated &&
        getExceptionSpecType() != EST_Unevaluated)
      return nullptr;
    return reinterpret_cast<FunctionDecl *const *>(param_type_end())[0];
  }
  /// \brief If this function type has an uninstantiated exception
  /// specification, this is the function whose exception specification
  /// should be instantiated to find the exception specification for
  /// this type.
  FunctionDecl *getExceptionSpecTemplate() const {
    if (getExceptionSpecType() != EST_Uninstantiated)
      return nullptr;
    return reinterpret_cast<FunctionDecl *const *>(param_type_end())[1];
  }
  /// Determine whether this function type has a non-throwing exception
  /// specification. If this depends on template arguments, returns
  /// \c ResultIfDependent.
  bool isNothrow(const ASTContext &Ctx, bool ResultIfDependent = false) const;

  bool isVariadic() const { return Variadic; }

  /// Determines whether this function prototype contains a
  /// parameter pack at the end.
  ///
  /// A function template whose last parameter is a parameter pack can be
  /// called with an arbitrary number of arguments, much like a variadic
  /// function.
  bool isTemplateVariadic() const;

  bool hasTrailingReturn() const { return HasTrailingReturn; }

  unsigned getTypeQuals() const { return FunctionType::getTypeQuals(); }


  /// Retrieve the ref-qualifier associated with this function type.
  RefQualifierKind getRefQualifier() const {
    return static_cast<RefQualifierKind>(FunctionTypeBits.RefQualifier);
  }

  typedef const QualType *param_type_iterator;
  typedef llvm::iterator_range<param_type_iterator> param_type_range;

  param_type_range param_types() const {
    return param_type_range(param_type_begin(), param_type_end());
  }
  param_type_iterator param_type_begin() const {
    return reinterpret_cast<const QualType *>(this+1);
  }
  param_type_iterator param_type_end() const {
    return param_type_begin() + NumParams;
  }

  typedef const QualType *exception_iterator;

  ArrayRef<QualType> exceptions() const {
    return llvm::makeArrayRef(exception_begin(), exception_end());
  }
  exception_iterator exception_begin() const {
    // exceptions begin where arguments end
    return param_type_end();
  }
  exception_iterator exception_end() const {
    if (getExceptionSpecType() != EST_Dynamic)
      return exception_begin();
    return exception_begin() + NumExceptions;
  }

  bool hasAnyConsumedParams() const { return HasAnyConsumedParams; }
  bool isParamConsumed(unsigned I) const {
    assert(I < getNumParams() && "parameter index out of range");
    if (hasAnyConsumedParams())
      return getConsumedParamsBuffer()[I];
    return false;
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void printExceptionSpecification(raw_ostream &OS,
                                   const PrintingPolicy &Policy) const;

  static bool classof(const Type *T) {
    return T->getTypeClass() == FunctionProto;
  }

  void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Ctx);
  static void Profile(llvm::FoldingSetNodeID &ID, QualType Result,
                      param_type_iterator ArgTys, unsigned NumArgs,
                      const ExtProtoInfo &EPI, const ASTContext &Context);
};

/// \brief Represents the dependent type named by a dependently-scoped
/// typename using declaration, e.g.
///   using typename Base<T>::foo;
///
/// Template instantiation turns these into the underlying type.
class UnresolvedUsingType : public Type {
  UnresolvedUsingTypenameDecl *Decl;

  UnresolvedUsingType(const UnresolvedUsingTypenameDecl *D)
    : Type(UnresolvedUsing, QualType(), true, true, false,
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

  void Profile(llvm::FoldingSetNodeID &ID) {
    return Profile(ID, Decl);
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      UnresolvedUsingTypenameDecl *D) {
    ID.AddPointer(D);
  }
};


class TypedefType : public Type {
  TypedefNameDecl *Decl;
protected:
  TypedefType(TypeClass tc, const TypedefNameDecl *D, QualType can)
    : Type(tc, can, can->isDependentType(),
           can->isInstantiationDependentType(),
           can->isVariablyModifiedType(),
           /*ContainsUnexpandedParameterPack=*/false),
      Decl(const_cast<TypedefNameDecl*>(D)) {
    assert(!isa<TypedefType>(can) && "Invalid canonical type");
  }
  friend class ASTContext;  // ASTContext creates these.
public:

  TypedefNameDecl *getDecl() const { return Decl; }

  bool isSugared() const { return true; }
  QualType desugar() const;

  static bool classof(const Type *T) { return T->getTypeClass() == Typedef; }
};

/// Represents a `typeof` (or __typeof__) expression (a GCC extension).
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
  bool isSugared() const;

  static bool classof(const Type *T) { return T->getTypeClass() == TypeOfExpr; }
};

/// \brief Internal representation of canonical, dependent
/// `typeof(expr)` types.
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

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getUnderlyingExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      Expr *E);
};

/// Represents `typeof(type)`, a GCC extension.
class TypeOfType : public Type {
  QualType TOType;
  TypeOfType(QualType T, QualType can)
    : Type(TypeOf, can, T->isDependentType(),
           T->isInstantiationDependentType(),
           T->isVariablyModifiedType(),
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
};

/// Represents the type `decltype(expr)` (C++11).
class DecltypeType : public Type {
  Expr *E;
  QualType UnderlyingType;

protected:
  DecltypeType(Expr *E, QualType underlyingType, QualType can = QualType());
  friend class ASTContext;  // ASTContext creates these.
public:
  Expr *getUnderlyingExpr() const { return E; }
  QualType getUnderlyingType() const { return UnderlyingType; }

  /// \brief Remove a single level of sugar.
  QualType desugar() const;

  /// \brief Returns whether this type directly provides sugar.
  bool isSugared() const;

  static bool classof(const Type *T) { return T->getTypeClass() == Decltype; }
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

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, Context, getUnderlyingExpr());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Context,
                      Expr *E);
};

/// A unary type transform, which is a type constructed from another.
class UnaryTransformType : public Type {
public:
  enum UTTKind {
    EnumUnderlyingType
  };

private:
  /// The untransformed type.
  QualType BaseType;
  /// The transformed type if not dependent, otherwise the same as BaseType.
  QualType UnderlyingType;

  UTTKind UKind;
protected:
  UnaryTransformType(QualType BaseTy, QualType UnderlyingTy, UTTKind UKind,
                     QualType CanonicalTy);
  friend class ASTContext;
public:
  bool isSugared() const { return !isDependentType(); }
  QualType desugar() const { return UnderlyingType; }

  QualType getUnderlyingType() const { return UnderlyingType; }
  QualType getBaseType() const { return BaseType; }

  UTTKind getUTTKind() const { return UKind; }

  static bool classof(const Type *T) {
    return T->getTypeClass() == UnaryTransform;
  }
};

class TagType : public Type {
  /// Stores the TagDecl associated with this type. The decl may point to any
  /// TagDecl that declares the entity.
  TagDecl * decl;

  friend class ASTReader;

protected:
  TagType(TypeClass TC, const TagDecl *D, QualType can);

public:
  TagDecl *getDecl() const;

  /// Determines whether this type is in the process of being defined.
  bool isBeingDefined() const;

  static bool classof(const Type *T) {
    return T->getTypeClass() >= TagFirst && T->getTypeClass() <= TagLast;
  }
};

/// A helper class that allows the use of isa/cast/dyncast
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

  static bool classof(const Type *T) { return T->getTypeClass() == Record; }
};

/// A helper class that allows the use of isa/cast/dyncast
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

  static bool classof(const Type *T) { return T->getTypeClass() == Enum; }
};

/// An attributed type is a type to which a type attribute has been applied.
///
/// The "modified type" is the fully-sugared type to which the attributed
/// type was applied; generally it is not canonically equivalent to the
/// attributed type. The "equivalent type" is the minimally-desugared type
/// which the type is canonically equivalent to.
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
    attr_objc_ownership,
    attr_pcs,
    attr_pcs_vfp,

    FirstEnumOperandKind = attr_objc_gc,
    LastEnumOperandKind = attr_pcs_vfp,

    // No operand.
    attr_noreturn,
    attr_cdecl,
    attr_fastcall,
    attr_stdcall,
    attr_thiscall,
    attr_pascal,
    attr_vectorcall,
    attr_inteloclbicc,
    attr_ms_abi,
    attr_sysv_abi,
    attr_ptr32,
    attr_ptr64,
    attr_sptr,
    attr_uptr,
    attr_nonnull,
    attr_nullable,
    attr_null_unspecified,
    attr_objc_kindof,
  };

private:
  QualType ModifiedType;
  QualType EquivalentType;

  friend class ASTContext; // creates these

  AttributedType(QualType canon, Kind attrKind,
                 QualType modified, QualType equivalent)
    : Type(Attributed, canon, canon->isDependentType(),
           canon->isInstantiationDependentType(),
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

  /// Does this attribute behave like a type qualifier?
  ///
  /// A type qualifier adjusts a type to provide specialized rules for
  /// a specific object, like the standard const and volatile qualifiers.
  /// This includes attributes controlling things like nullability,
  /// address spaces, and ARC ownership.  The value of the object is still
  /// largely described by the modified type.
  ///
  /// In contrast, many type attributes "rewrite" their modified type to
  /// produce a fundamentally different type, not necessarily related in any
  /// formalizable way to the original type.  For example, calling convention
  /// and vector attributes are not simple type qualifiers.
  ///
  /// Type qualifiers are often, but not always, reflected in the canonical
  /// type.
  bool isQualifier() const;

  bool isMSTypeSpec() const;

  bool isCallingConv() const;

  llvm::Optional<NullabilityKind> getImmediateNullability() const;

  /// Retrieve the attribute kind corresponding to the given
  /// nullability kind.
  static Kind getNullabilityAttrKind(NullabilityKind kind) {
    switch (kind) {
    case NullabilityKind::NonNull:
      return attr_nonnull;

    case NullabilityKind::Nullable:
      return attr_nullable;

    case NullabilityKind::Unspecified:
      return attr_null_unspecified;
    }
    llvm_unreachable("Unknown nullability kind.");
  }

  /// Strip off the top-level nullability annotation on the given
  /// type, if it's there.
  ///
  /// \param T The type to strip. If the type is exactly an
  /// AttributedType specifying nullability (without looking through
  /// type sugar), the nullability is returned and this type changed
  /// to the underlying modified type.
  ///
  /// \returns the top-level nullability, if present.
  static Optional<NullabilityKind> stripOuterNullability(QualType &T);

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
};

class TemplateTypeParmType : public Type, public llvm::FoldingSetNode {
  // Helper data collector for canonical types.
  struct CanonicalTTPTInfo {
    unsigned Depth : 15;
    unsigned ParameterPack : 1;
    unsigned Index : 16;
  };

  union {
    // Info for the canonical type.
    CanonicalTTPTInfo CanTTPTInfo;
    // Info for the non-canonical type.
    TemplateTypeParmDecl *TTPDecl;
  };

  /// Build a non-canonical type.
  TemplateTypeParmType(TemplateTypeParmDecl *TTPDecl, QualType Canon)
    : Type(TemplateTypeParm, Canon, /*Dependent=*/true,
           /*InstantiationDependent=*/true,
           /*VariablyModified=*/false,
           Canon->containsUnexpandedParameterPack()),
      TTPDecl(TTPDecl) { }

  /// Build the canonical type.
  TemplateTypeParmType(unsigned D, unsigned I, bool PP)
    : Type(TemplateTypeParm, QualType(this, 0),
           /*Dependent=*/true,
           /*InstantiationDependent=*/true,
           /*VariablyModified=*/false, PP) {
    CanTTPTInfo.Depth = D;
    CanTTPTInfo.Index = I;
    CanTTPTInfo.ParameterPack = PP;
  }

  friend class ASTContext;  // ASTContext creates these

  const CanonicalTTPTInfo& getCanTTPTInfo() const {
    QualType Can = getCanonicalTypeInternal();
    return Can->castAs<TemplateTypeParmType>()->CanTTPTInfo;
  }

public:
  unsigned getDepth() const { return getCanTTPTInfo().Depth; }
  unsigned getIndex() const { return getCanTTPTInfo().Index; }
  bool isParameterPack() const { return getCanTTPTInfo().ParameterPack; }

  TemplateTypeParmDecl *getDecl() const {
    return isCanonicalUnqualified() ? nullptr : TTPDecl;
  }

  IdentifierInfo *getIdentifier() const;

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getDepth(), getIndex(), isParameterPack(), getDecl());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, unsigned Depth,
                      unsigned Index, bool ParameterPack,
                      TemplateTypeParmDecl *TTPDecl) {
    ID.AddInteger(Depth);
    ID.AddInteger(Index);
    ID.AddBoolean(ParameterPack);
    ID.AddPointer(TTPDecl);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == TemplateTypeParm;
  }
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
           Canon->isInstantiationDependentType(),
           Canon->isVariablyModifiedType(),
           Canon->containsUnexpandedParameterPack()),
      Replaced(Param) { }

  friend class ASTContext;

public:
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
  IdentifierInfo *getIdentifier() const { return Replaced->getIdentifier(); }

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
};

/// \brief Represents a C++11 auto or C++14 decltype(auto) type.
///
/// These types are usually a placeholder for a deduced type. However, before
/// the initializer is attached, or if the initializer is type-dependent, there
/// is no deduced type and an auto type is canonical. In the latter case, it is
/// also a dependent type.
class AutoType : public Type, public llvm::FoldingSetNode {
  AutoType(QualType DeducedType, AutoTypeKeyword Keyword, bool IsDependent)
    : Type(Auto, DeducedType.isNull() ? QualType(this, 0) : DeducedType,
           /*Dependent=*/IsDependent, /*InstantiationDependent=*/IsDependent,
           /*VariablyModified=*/false,
           /*ContainsParameterPack=*/DeducedType.isNull()
               ? false : DeducedType->containsUnexpandedParameterPack()) {
    assert((DeducedType.isNull() || !IsDependent) &&
           "auto deduced to dependent type");
    AutoTypeBits.Keyword = (unsigned)Keyword;
  }

  friend class ASTContext;  // ASTContext creates these

public:
  bool isDecltypeAuto() const {
    return getKeyword() == AutoTypeKeyword::DecltypeAuto;
  }
  AutoTypeKeyword getKeyword() const {
    return (AutoTypeKeyword)AutoTypeBits.Keyword;
  }

  bool isSugared() const { return !isCanonicalUnqualified(); }
  QualType desugar() const { return getCanonicalTypeInternal(); }

  /// \brief Get the type deduced for this auto type, or null if it's either
  /// not been deduced or was deduced to a dependent type.
  QualType getDeducedType() const {
    return !isCanonicalUnqualified() ? getCanonicalTypeInternal() : QualType();
  }
  bool isDeduced() const {
    return !isCanonicalUnqualified() || isDependentType();
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getDeducedType(), getKeyword(), isDependentType());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, QualType Deduced,
                      AutoTypeKeyword Keyword, bool IsDependent) {
    ID.AddPointer(Deduced.getAsOpaquePtr());
    ID.AddInteger((unsigned)Keyword);
    ID.AddBoolean(IsDependent);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == Auto;
  }
};

/// \brief Represents a type template specialization; the template
/// must be a class template, a type alias template, or a template
/// template parameter.  A template which cannot be resolved to one of
/// these, e.g. because it is written with a dependent scope
/// specifier, is instead represented as a
/// @c DependentTemplateSpecializationType.
///
/// A non-dependent template specialization type is always "sugar",
/// typically for a \c RecordType.  For example, a class template
/// specialization type of \c vector<int> will refer to a tag type for
/// the instantiation \c std::vector<int, std::allocator<int>>
///
/// Template specializations are dependent if either the template or
/// any of the template arguments are dependent, in which case the
/// type may also be canonical.
///
/// Instances of this type are allocated with a trailing array of
/// TemplateArguments, followed by a QualType representing the
/// non-canonical aliased type when the template is a type alias
/// template.
class LLVM_ALIGNAS(/*alignof(uint64_t)*/ 8) TemplateSpecializationType
    : public Type,
      public llvm::FoldingSetNode {
  /// The name of the template being specialized.  This is
  /// either a TemplateName::Template (in which case it is a
  /// ClassTemplateDecl*, a TemplateTemplateParmDecl*, or a
  /// TypeAliasTemplateDecl*), a
  /// TemplateName::SubstTemplateTemplateParmPack, or a
  /// TemplateName::SubstTemplateTemplateParm (in which case the
  /// replacement must, recursively, be one of these).
  TemplateName Template;

  /// The number of template arguments named in this class template
  /// specialization.
  unsigned NumArgs : 31;

  /// Whether this template specialization type is a substituted type alias.
  bool TypeAlias : 1;

  TemplateSpecializationType(TemplateName T,
                             const TemplateArgument *Args,
                             unsigned NumArgs, QualType Canon,
                             QualType Aliased);

  friend class ASTContext;  // ASTContext creates these

public:
  /// Determine whether any of the given template arguments are dependent.
  static bool anyDependentTemplateArguments(const TemplateArgumentLoc *Args,
                                            unsigned NumArgs,
                                            bool &InstantiationDependent);

  static bool anyDependentTemplateArguments(const TemplateArgumentListInfo &,
                                            bool &InstantiationDependent);

  /// \brief Print a template argument list, including the '<' and '>'
  /// enclosing the template arguments.
  static void PrintTemplateArgumentList(raw_ostream &OS,
                                        const TemplateArgument *Args,
                                        unsigned NumArgs,
                                        const PrintingPolicy &Policy,
                                        bool SkipBrackets = false);

  static void PrintTemplateArgumentList(raw_ostream &OS,
                                        const TemplateArgumentLoc *Args,
                                        unsigned NumArgs,
                                        const PrintingPolicy &Policy);

  static void PrintTemplateArgumentList(raw_ostream &OS,
                                        const TemplateArgumentListInfo &,
                                        const PrintingPolicy &Policy);

  /// True if this template specialization type matches a current
  /// instantiation in the context in which it is found.
  bool isCurrentInstantiation() const {
    return isa<InjectedClassNameType>(getCanonicalTypeInternal());
  }

  /// \brief Determine if this template specialization type is for a type alias
  /// template that has been substituted.
  ///
  /// Nearly every template specialization type whose template is an alias
  /// template will be substituted. However, this is not the case when
  /// the specialization contains a pack expansion but the template alias
  /// does not have a corresponding parameter pack, e.g.,
  ///
  /// \code
  /// template<typename T, typename U, typename V> struct S;
  /// template<typename T, typename U> using A = S<T, int, U>;
  /// template<typename... Ts> struct X {
  ///   typedef A<Ts...> type; // not a type alias
  /// };
  /// \endcode
  bool isTypeAlias() const { return TypeAlias; }

  /// Get the aliased type, if this is a specialization of a type alias
  /// template.
  QualType getAliasedType() const {
    assert(isTypeAlias() && "not a type alias template specialization");
    return *reinterpret_cast<const QualType*>(end());
  }

  typedef const TemplateArgument * iterator;

  iterator begin() const { return getArgs(); }
  iterator end() const; // defined inline in TemplateBase.h

  /// Retrieve the name of the template that we are specializing.
  TemplateName getTemplateName() const { return Template; }

  /// Retrieve the template arguments.
  const TemplateArgument *getArgs() const {
    return reinterpret_cast<const TemplateArgument *>(this + 1);
  }

  /// Retrieve the number of template arguments.
  unsigned getNumArgs() const { return NumArgs; }

  /// Retrieve a specific template argument as a type.
  /// \pre \c isArgType(Arg)
  const TemplateArgument &getArg(unsigned Idx) const; // in TemplateBase.h

  bool isSugared() const {
    return !isDependentType() || isCurrentInstantiation() || isTypeAlias();
  }
  QualType desugar() const { return getCanonicalTypeInternal(); }

  void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &Ctx) {
    Profile(ID, Template, getArgs(), NumArgs, Ctx);
    if (isTypeAlias())
      getAliasedType().Profile(ID);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, TemplateName T,
                      const TemplateArgument *Args,
                      unsigned NumArgs,
                      const ASTContext &Context);

  static bool classof(const Type *T) {
    return T->getTypeClass() == TemplateSpecialization;
  }
};

/// The injected class name of a C++ class template or class
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
           /*InstantiationDependent=*/true,
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
};

/// \brief The kind of a tag type.
enum TagTypeKind {
  /// \brief The "struct" keyword.
  TTK_Struct,
  /// \brief The "__interface" keyword.
  TTK_Interface,
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
  /// \brief The "__interface" keyword introduces the elaborated-type-specifier.
  ETK_Interface,
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
                  QualType Canonical, bool Dependent,
                  bool InstantiationDependent, bool VariablyModified,
                  bool ContainsUnexpandedParameterPack)
  : Type(tc, Canonical, Dependent, InstantiationDependent, VariablyModified,
         ContainsUnexpandedParameterPack) {
    TypeWithKeywordBits.Keyword = Keyword;
  }

public:
  ElaboratedTypeKeyword getKeyword() const {
    return static_cast<ElaboratedTypeKeyword>(TypeWithKeywordBits.Keyword);
  }

  /// Converts a type specifier (DeclSpec::TST) into an elaborated type keyword.
  static ElaboratedTypeKeyword getKeywordForTypeSpec(unsigned TypeSpec);

  /// Converts a type specifier (DeclSpec::TST) into a tag type kind.
  /// It is an error to provide a type specifier which *isn't* a tag kind here.
  static TagTypeKind getTagTypeKindForTypeSpec(unsigned TypeSpec);

  /// Converts a TagTypeKind into an elaborated type keyword.
  static ElaboratedTypeKeyword getKeywordForTagTypeKind(TagTypeKind Tag);

  /// Converts an elaborated type keyword into a TagTypeKind.
  /// It is an error to provide an elaborated type keyword
  /// which *isn't* a tag kind here.
  static TagTypeKind getTagTypeKindForKeyword(ElaboratedTypeKeyword Keyword);

  static bool KeywordIsTagTypeKind(ElaboratedTypeKeyword Keyword);

  static StringRef getKeywordName(ElaboratedTypeKeyword Keyword);

  static StringRef getTagTypeKindName(TagTypeKind Kind) {
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

  /// The nested name specifier containing the qualifier.
  NestedNameSpecifier *NNS;

  /// The type that this qualified name refers to.
  QualType NamedType;

  ElaboratedType(ElaboratedTypeKeyword Keyword, NestedNameSpecifier *NNS,
                 QualType NamedType, QualType CanonType)
    : TypeWithKeyword(Keyword, Elaborated, CanonType,
                      NamedType->isDependentType(),
                      NamedType->isInstantiationDependentType(),
                      NamedType->isVariablyModifiedType(),
                      NamedType->containsUnexpandedParameterPack()),
      NNS(NNS), NamedType(NamedType) {
    assert(!(Keyword == ETK_None && NNS == nullptr) &&
           "ElaboratedType cannot have elaborated type keyword "
           "and name qualifier both null.");
  }

  friend class ASTContext;  // ASTContext creates these

public:
  ~ElaboratedType();

  /// Retrieve the qualification on this type.
  NestedNameSpecifier *getQualifier() const { return NNS; }

  /// Retrieve the type named by the qualified-id.
  QualType getNamedType() const { return NamedType; }

  /// Remove a single level of sugar.
  QualType desugar() const { return getNamedType(); }

  /// Returns whether this type directly provides sugar.
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
};

/// \brief Represents a qualified type name for which the type name is
/// dependent.
///
/// DependentNameType represents a class of dependent types that involve a
/// possibly dependent nested-name-specifier (e.g., "T::") followed by a
/// name of a type. The DependentNameType may start with a "typename" (for a
/// typename-specifier), "class", "struct", "union", or "enum" (for a
/// dependent elaborated-type-specifier), or nothing (in contexts where we
/// know that we must be referring to a type, e.g., in a base class specifier).
/// Typically the nested-name-specifier is dependent, but in MSVC compatibility
/// mode, this type is used with non-dependent names to delay name lookup until
/// instantiation.
class DependentNameType : public TypeWithKeyword, public llvm::FoldingSetNode {

  /// \brief The nested name specifier containing the qualifier.
  NestedNameSpecifier *NNS;

  /// \brief The type that this typename specifier refers to.
  const IdentifierInfo *Name;

  DependentNameType(ElaboratedTypeKeyword Keyword, NestedNameSpecifier *NNS,
                    const IdentifierInfo *Name, QualType CanonType)
    : TypeWithKeyword(Keyword, DependentName, CanonType, /*Dependent=*/true,
                      /*InstantiationDependent=*/true,
                      /*VariablyModified=*/false,
                      NNS->containsUnexpandedParameterPack()),
      NNS(NNS), Name(Name) {}

  friend class ASTContext;  // ASTContext creates these

public:
  /// Retrieve the qualification on this type.
  NestedNameSpecifier *getQualifier() const { return NNS; }

  /// Retrieve the type named by the typename specifier as an identifier.
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
};

/// Represents a template specialization type whose template cannot be
/// resolved, e.g.
///   A<T>::template B<T>
class LLVM_ALIGNAS(/*alignof(uint64_t)*/ 8) DependentTemplateSpecializationType
    : public TypeWithKeyword,
      public llvm::FoldingSetNode {

  /// The nested name specifier containing the qualifier.
  NestedNameSpecifier *NNS;

  /// The identifier of the template.
  const IdentifierInfo *Name;

  /// \brief The number of template arguments named in this class template
  /// specialization.
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
};

/// \brief Represents a pack expansion of types.
///
/// Pack expansions are part of C++11 variadic templates. A pack
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
                    Optional<unsigned> NumExpansions)
    : Type(PackExpansion, Canon, /*Dependent=*/Pattern->isDependentType(),
           /*InstantiationDependent=*/true,
           /*VariablyModified=*/Pattern->isVariablyModifiedType(),
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
  Optional<unsigned> getNumExpansions() const {
    if (NumExpansions)
      return NumExpansions - 1;

    return None;
  }

  bool isSugared() const { return !Pattern->isDependentType(); }
  QualType desugar() const { return isSugared() ? Pattern : QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPattern(), getNumExpansions());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, QualType Pattern,
                      Optional<unsigned> NumExpansions) {
    ID.AddPointer(Pattern.getAsOpaquePtr());
    ID.AddBoolean(NumExpansions.hasValue());
    if (NumExpansions)
      ID.AddInteger(*NumExpansions);
  }

  static bool classof(const Type *T) {
    return T->getTypeClass() == PackExpansion;
  }
};

/// Represents a class type in Objective C.
///
/// Every Objective C type is a combination of a base type, a set of
/// type arguments (optional, for parameterized classes) and a list of
/// protocols.
///
/// Given the following declarations:
/// \code
///   \@class C<T>;
///   \@protocol P;
/// \endcode
///
/// 'C' is an ObjCInterfaceType C.  It is sugar for an ObjCObjectType
/// with base C and no protocols.
///
/// 'C<P>' is an unspecialized ObjCObjectType with base C and protocol list [P].
/// 'C<C*>' is a specialized ObjCObjectType with type arguments 'C*' and no 
/// protocol list.
/// 'C<C*><P>' is a specialized ObjCObjectType with base C, type arguments 'C*',
/// and protocol list [P].
///
/// 'id' is a TypedefType which is sugar for an ObjCObjectPointerType whose
/// pointee is an ObjCObjectType with base BuiltinType::ObjCIdType
/// and no protocols.
///
/// 'id<P>' is an ObjCObjectPointerType whose pointee is an ObjCObjectType
/// with base BuiltinType::ObjCIdType and protocol list [P].  Eventually
/// this should get its own sugar class to better represent the source.
class ObjCObjectType : public Type {
  // ObjCObjectType.NumTypeArgs - the number of type arguments stored
  // after the ObjCObjectPointerType node.
  // ObjCObjectType.NumProtocols - the number of protocols stored
  // after the type arguments of ObjCObjectPointerType node.
  //
  // These protocols are those written directly on the type.  If
  // protocol qualifiers ever become additive, the iterators will need
  // to get kindof complicated.
  //
  // In the canonical object type, these are sorted alphabetically
  // and uniqued.

  /// Either a BuiltinType or an InterfaceType or sugar for either.
  QualType BaseType;

  /// Cached superclass type.
  mutable llvm::PointerIntPair<const ObjCObjectType *, 1, bool>
    CachedSuperClassType;

  ObjCProtocolDecl * const *getProtocolStorage() const {
    return const_cast<ObjCObjectType*>(this)->getProtocolStorage();
  }

  QualType *getTypeArgStorage();
  const QualType *getTypeArgStorage() const {
    return const_cast<ObjCObjectType *>(this)->getTypeArgStorage();
  }

  ObjCProtocolDecl **getProtocolStorage();

protected:
  ObjCObjectType(QualType Canonical, QualType Base,
                 ArrayRef<QualType> typeArgs,
                 ArrayRef<ObjCProtocolDecl *> protocols,
                 bool isKindOf);

  enum Nonce_ObjCInterface { Nonce_ObjCInterface };
  ObjCObjectType(enum Nonce_ObjCInterface)
        : Type(ObjCInterface, QualType(), false, false, false, false),
      BaseType(QualType(this_(), 0)) {
    ObjCObjectTypeBits.NumProtocols = 0;
    ObjCObjectTypeBits.NumTypeArgs = 0;
    ObjCObjectTypeBits.IsKindOf = 0;
  }

  void computeSuperClassTypeSlow() const;

public:
  /// Gets the base type of this object type.  This is always (possibly
  /// sugar for) one of:
  ///  - the 'id' builtin type (as opposed to the 'id' type visible to the
  ///    user, which is a typedef for an ObjCObjectPointerType)
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

  /// Determine whether this object type is "specialized", meaning
  /// that it has type arguments.
  bool isSpecialized() const;

  /// Determine whether this object type was written with type arguments.
  bool isSpecializedAsWritten() const {
    return ObjCObjectTypeBits.NumTypeArgs > 0;
  }

  /// Determine whether this object type is "unspecialized", meaning
  /// that it has no type arguments.
  bool isUnspecialized() const { return !isSpecialized(); }

  /// Determine whether this object type is "unspecialized" as
  /// written, meaning that it has no type arguments.
  bool isUnspecializedAsWritten() const { return !isSpecializedAsWritten(); }

  /// Retrieve the type arguments of this object type (semantically).
  ArrayRef<QualType> getTypeArgs() const;

  /// Retrieve the type arguments of this object type as they were
  /// written.
  ArrayRef<QualType> getTypeArgsAsWritten() const {
    return llvm::makeArrayRef(getTypeArgStorage(),
                              ObjCObjectTypeBits.NumTypeArgs);
  }

  typedef ObjCProtocolDecl * const *qual_iterator;
  typedef llvm::iterator_range<qual_iterator> qual_range;

  qual_range quals() const { return qual_range(qual_begin(), qual_end()); }
  qual_iterator qual_begin() const { return getProtocolStorage(); }
  qual_iterator qual_end() const { return qual_begin() + getNumProtocols(); }

  bool qual_empty() const { return getNumProtocols() == 0; }

  /// Return the number of qualifying protocols in this interface type,
  /// or 0 if there are none.
  unsigned getNumProtocols() const { return ObjCObjectTypeBits.NumProtocols; }

  /// Fetch a protocol by index.
  ObjCProtocolDecl *getProtocol(unsigned I) const {
    assert(I < getNumProtocols() && "Out-of-range protocol access");
    return qual_begin()[I];
  }

  /// Retrieve all of the protocol qualifiers.
  ArrayRef<ObjCProtocolDecl *> getProtocols() const {
    return ArrayRef<ObjCProtocolDecl *>(qual_begin(), getNumProtocols());
  }

  /// Whether this is a "__kindof" type as written.
  bool isKindOfTypeAsWritten() const { return ObjCObjectTypeBits.IsKindOf; }

  /// Whether this ia a "__kindof" type (semantically).
  bool isKindOfType() const;

  /// Retrieve the type of the superclass of this object type.
  ///
  /// This operation substitutes any type arguments into the
  /// superclass of the current class type, potentially producing a
  /// specialization of the superclass type. Produces a null type if
  /// there is no superclass.
  QualType getSuperClassType() const {
    if (!CachedSuperClassType.getInt())
      computeSuperClassTypeSlow();

    assert(CachedSuperClassType.getInt() && "Superclass not set?");
    return QualType(CachedSuperClassType.getPointer(), 0);
  }

  /// Strip off the Objective-C "kindof" type and (with it) any
  /// protocol qualifiers.
  QualType stripObjCKindOfTypeAndQuals(const ASTContext &ctx) const;

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == ObjCObject ||
           T->getTypeClass() == ObjCInterface;
  }
};

/// A class providing a concrete implementation
/// of ObjCObjectType, so as to not increase the footprint of
/// ObjCInterfaceType.  Code outside of ASTContext and the core type
/// system should not reference this type.
class ObjCObjectTypeImpl : public ObjCObjectType, public llvm::FoldingSetNode {
  friend class ASTContext;

  // If anyone adds fields here, ObjCObjectType::getProtocolStorage()
  // will need to be modified.

  ObjCObjectTypeImpl(QualType Canonical, QualType Base,
                     ArrayRef<QualType> typeArgs,
                     ArrayRef<ObjCProtocolDecl *> protocols,
                     bool isKindOf)
    : ObjCObjectType(Canonical, Base, typeArgs, protocols, isKindOf) {}

public:
  void Profile(llvm::FoldingSetNodeID &ID);
  static void Profile(llvm::FoldingSetNodeID &ID,
                      QualType Base,
                      ArrayRef<QualType> typeArgs,
                      ArrayRef<ObjCProtocolDecl *> protocols,
                      bool isKindOf);
};

inline QualType *ObjCObjectType::getTypeArgStorage() {
  return reinterpret_cast<QualType *>(static_cast<ObjCObjectTypeImpl*>(this)+1);
}

inline ObjCProtocolDecl **ObjCObjectType::getProtocolStorage() {
    return reinterpret_cast<ObjCProtocolDecl**>(
             getTypeArgStorage() + ObjCObjectTypeBits.NumTypeArgs);
}

/// Interfaces are the core concept in Objective-C for object oriented design.
/// They basically correspond to C++ classes.  There are two kinds of interface
/// types: normal interfaces like `NSString`, and qualified interfaces, which
/// are qualified with a protocol list like `NSString<NSCopyable, NSAmazing>`.
///
/// ObjCInterfaceType guarantees the following properties when considered
/// as a subtype of its superclass, ObjCObjectType:
///   - There are no protocol qualifiers.  To reinforce this, code which
///     tries to invoke the protocol methods via an ObjCInterfaceType will
///     fail to compile.
///   - It is its own base type.  That is, if T is an ObjCInterfaceType*,
///     T->getBaseType() == QualType(T, 0).
class ObjCInterfaceType : public ObjCObjectType {
  mutable ObjCInterfaceDecl *Decl;

  ObjCInterfaceType(const ObjCInterfaceDecl *D)
    : ObjCObjectType(Nonce_ObjCInterface),
      Decl(const_cast<ObjCInterfaceDecl*>(D)) {}
  friend class ASTContext;  // ASTContext creates these.
  friend class ASTReader;
  friend class ObjCInterfaceDecl;

public:
  /// Get the declaration of this interface.
  ObjCInterfaceDecl *getDecl() const { return Decl; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  static bool classof(const Type *T) {
    return T->getTypeClass() == ObjCInterface;
  }

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
  QualType baseType = getBaseType();
  while (const ObjCObjectType *ObjT = baseType->getAs<ObjCObjectType>()) {
    if (const ObjCInterfaceType *T = dyn_cast<ObjCInterfaceType>(ObjT))
      return T->getDecl();

    baseType = ObjT->getBaseType();
  }

  return nullptr;
}

/// Represents a pointer to an Objective C object.
///
/// These are constructed from pointer declarators when the pointee type is
/// an ObjCObjectType (or sugar for one).  In addition, the 'id' and 'Class'
/// types are typedefs for these, and the protocol-qualified types 'id<P>'
/// and 'Class<P>' are translated into these.
///
/// Pointers to pointers to Objective C objects are still PointerTypes;
/// only the first level of pointer gets it own type implementation.
class ObjCObjectPointerType : public Type, public llvm::FoldingSetNode {
  QualType PointeeType;

  ObjCObjectPointerType(QualType Canonical, QualType Pointee)
    : Type(ObjCObjectPointer, Canonical,
           Pointee->isDependentType(),
           Pointee->isInstantiationDependentType(),
           Pointee->isVariablyModifiedType(),
           Pointee->containsUnexpandedParameterPack()),
      PointeeType(Pointee) {}
  friend class ASTContext;  // ASTContext creates these.

public:
  /// Gets the type pointed to by this ObjC pointer.
  /// The result will always be an ObjCObjectType or sugar thereof.
  QualType getPointeeType() const { return PointeeType; }

  /// Gets the type pointed to by this ObjC pointer.  Always returns non-null.
  ///
  /// This method is equivalent to getPointeeType() except that
  /// it discards any typedefs (or other sugar) between this
  /// type and the "outermost" object type.  So for:
  /// \code
  ///   \@class A; \@protocol P; \@protocol Q;
  ///   typedef A<P> AP;
  ///   typedef A A1;
  ///   typedef A1<P> A1P;
  ///   typedef A1P<Q> A1PQ;
  /// \endcode
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

  /// If this pointer points to an Objective C
  /// \@interface type, gets the type for that interface.  Any protocol
  /// qualifiers on the interface are ignored.
  ///
  /// \return null if the base type for this pointer is 'id' or 'Class'
  const ObjCInterfaceType *getInterfaceType() const;

  /// If this pointer points to an Objective \@interface
  /// type, gets the declaration for that interface.
  ///
  /// \return null if the base type for this pointer is 'id' or 'Class'
  ObjCInterfaceDecl *getInterfaceDecl() const {
    return getObjectType()->getInterface();
  }

  /// True if this is equivalent to the 'id' type, i.e. if
  /// its object type is the primitive 'id' type with no protocols.
  bool isObjCIdType() const {
    return getObjectType()->isObjCUnqualifiedId();
  }

  /// True if this is equivalent to the 'Class' type,
  /// i.e. if its object tive is the primitive 'Class' type with no protocols.
  bool isObjCClassType() const {
    return getObjectType()->isObjCUnqualifiedClass();
  }

  /// True if this is equivalent to the 'id' or 'Class' type,
  bool isObjCIdOrClassType() const {
    return getObjectType()->isObjCUnqualifiedIdOrClass();
  }

  /// True if this is equivalent to 'id<P>' for some non-empty set of
  /// protocols.
  bool isObjCQualifiedIdType() const {
    return getObjectType()->isObjCQualifiedId();
  }

  /// True if this is equivalent to 'Class<P>' for some non-empty set of
  /// protocols.
  bool isObjCQualifiedClassType() const {
    return getObjectType()->isObjCQualifiedClass();
  }

  /// Whether this is a "__kindof" type.
  bool isKindOfType() const { return getObjectType()->isKindOfType(); }

  /// Whether this type is specialized, meaning that it has type arguments.
  bool isSpecialized() const { return getObjectType()->isSpecialized(); }

  /// Whether this type is specialized, meaning that it has type arguments.
  bool isSpecializedAsWritten() const {
    return getObjectType()->isSpecializedAsWritten();
  }

  /// Whether this type is unspecialized, meaning that is has no type arguments.
  bool isUnspecialized() const { return getObjectType()->isUnspecialized(); }

  /// Determine whether this object type is "unspecialized" as
  /// written, meaning that it has no type arguments.
  bool isUnspecializedAsWritten() const { return !isSpecializedAsWritten(); }

  /// Retrieve the type arguments for this type.
  ArrayRef<QualType> getTypeArgs() const {
    return getObjectType()->getTypeArgs();
  }

  /// Retrieve the type arguments for this type.
  ArrayRef<QualType> getTypeArgsAsWritten() const {
    return getObjectType()->getTypeArgsAsWritten();
  }

  /// An iterator over the qualifiers on the object type.  Provided
  /// for convenience.  This will always iterate over the full set of
  /// protocols on a type, not just those provided directly.
  typedef ObjCObjectType::qual_iterator qual_iterator;
  typedef llvm::iterator_range<qual_iterator> qual_range;

  qual_range quals() const { return qual_range(qual_begin(), qual_end()); }
  qual_iterator qual_begin() const {
    return getObjectType()->qual_begin();
  }
  qual_iterator qual_end() const {
    return getObjectType()->qual_end();
  }
  bool qual_empty() const { return getObjectType()->qual_empty(); }

  /// Return the number of qualifying protocols on the object type.
  unsigned getNumProtocols() const {
    return getObjectType()->getNumProtocols();
  }

  /// Retrieve a qualifying protocol by index on the object type.
  ObjCProtocolDecl *getProtocol(unsigned I) const {
    return getObjectType()->getProtocol(I);
  }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  /// Retrieve the type of the superclass of this object pointer type.
  ///
  /// This operation substitutes any type arguments into the
  /// superclass of the current class type, potentially producing a
  /// pointer to a specialization of the superclass type. Produces a
  /// null type if there is no superclass.
  QualType getSuperClassType() const;

  /// Strip off the Objective-C "kindof" type and (with it) any
  /// protocol qualifiers.
  const ObjCObjectPointerType *stripObjCKindOfTypeAndQuals(
                                 const ASTContext &ctx) const;

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getPointeeType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType T) {
    ID.AddPointer(T.getAsOpaquePtr());
  }
  static bool classof(const Type *T) {
    return T->getTypeClass() == ObjCObjectPointer;
  }
};

class AtomicType : public Type, public llvm::FoldingSetNode {
  QualType ValueType;

  AtomicType(QualType ValTy, QualType Canonical)
    : Type(Atomic, Canonical, ValTy->isDependentType(),
           ValTy->isInstantiationDependentType(),
           ValTy->isVariablyModifiedType(),
           ValTy->containsUnexpandedParameterPack()),
      ValueType(ValTy) {}
  friend class ASTContext;  // ASTContext creates these.

  public:
  /// Gets the type contained by this atomic type, i.e.
  /// the type returned by performing an atomic load of this atomic type.
  QualType getValueType() const { return ValueType; }

  bool isSugared() const { return false; }
  QualType desugar() const { return QualType(this, 0); }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getValueType());
  }
  static void Profile(llvm::FoldingSetNodeID &ID, QualType T) {
    ID.AddPointer(T.getAsOpaquePtr());
  }
  static bool classof(const Type *T) {
    return T->getTypeClass() == Atomic;
  }
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

inline SplitQualType SplitQualType::getSingleStepDesugaredType() const {
  SplitQualType desugar =
    Ty->getLocallyUnqualifiedSingleStepDesugaredType().split();
  desugar.Quals.addConsistentQualifiers(Quals);
  return desugar;
}

inline const Type *QualType::getTypePtr() const {
  return getCommonPtr()->BaseType;
}

inline const Type *QualType::getTypePtrOrNull() const {
  return (isNull() ? nullptr : getCommonPtr()->BaseType);
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

  return QualType(getSplitUnqualifiedTypeImpl(*this).Ty, 0);
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

/// Return the address space of this type.
inline unsigned QualType::getAddressSpace() const {
  return getQualifiers().getAddressSpace();
}

/// Return the gc attribute of this type.
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

/// Determine whether this type is more
/// qualified than the Other type. For example, "const volatile int"
/// is more qualified than "const int", "volatile int", and
/// "int". However, it is not more qualified than "const volatile
/// int".
inline bool QualType::isMoreQualifiedThan(QualType other) const {
  Qualifiers myQuals = getQualifiers();
  Qualifiers otherQuals = other.getQualifiers();
  return (myQuals != otherQuals && myQuals.compatiblyIncludes(otherQuals));
}

/// Determine whether this type is at last
/// as qualified as the Other type. For example, "const volatile
/// int" is at least as qualified as "const int", "volatile int",
/// "int", and "const volatile int".
inline bool QualType::isAtLeastAsQualifiedAs(QualType other) const {
  return getQualifiers().compatiblyIncludes(other.getQualifiers());
}

/// If Type is a reference type (e.g., const
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

inline bool QualType::isCForbiddenLValueType() const {
  return ((getTypePtr()->isVoidType() && !hasQualifiers()) ||
          getTypePtr()->isFunctionType());
}

/// Tests whether the type is categorized as a fundamental type.
///
/// \returns True for types specified in C++0x [basic.fundamental].
inline bool Type::isFundamentalType() const {
  return isVoidType() ||
         // FIXME: It's really annoying that we don't have an
         // 'isArithmeticType()' which agrees with the standard definition.
         (isArithmeticType() && !isEnumeralType());
}

/// Tests whether the type is categorized as a compound type.
///
/// \returns True for types specified in C++0x [basic.compound].
inline bool Type::isCompoundType() const {
  // C++0x [basic.compound]p1:
  //   Compound types can be constructed in the following ways:
  //    -- arrays of objects of a given type [...];
  return isArrayType() ||
  //    -- functions, which have parameters of given types [...];
         isFunctionType() ||
  //    -- pointers to void or objects or functions [...];
         isPointerType() ||
  //    -- references to objects or functions of a given type. [...]
         isReferenceType() ||
  //    -- classes containing a sequence of objects of various types, [...];
         isRecordType() ||
  //    -- unions, which are classes capable of containing objects of different
  //               types at different times;
         isUnionType() ||
  //    -- enumerations, which comprise a set of named constant values. [...];
         isEnumeralType() ||
  //    -- pointers to non-static class members, [...].
         isMemberPointerType();
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
inline bool Type::isAtomicType() const {
  return isa<AtomicType>(CanonicalType);
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

inline bool Type::isImage1dT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage1d);
}

inline bool Type::isImage1dArrayT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage1dArray);
}

inline bool Type::isImage1dBufferT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage1dBuffer);
}

inline bool Type::isImage2dT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage2d);
}

inline bool Type::isImage2dArrayT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage2dArray);
}

inline bool Type::isImage2dDepthT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage2dDepth);
}

inline bool Type::isImage2dArrayDepthT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage2dArrayDepth);
}

inline bool Type::isImage2dMSAAT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage2dMSAA);
}

inline bool Type::isImage2dArrayMSAAT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage2dArrayMSAA);
}

inline bool Type::isImage2dMSAATDepth() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage2dMSAADepth);
}

inline bool Type::isImage2dArrayMSAATDepth() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage2dArrayMSAADepth);
}

inline bool Type::isImage3dT() const {
  return isSpecificBuiltinType(BuiltinType::OCLImage3d);
}

inline bool Type::isSamplerT() const {
  return isSpecificBuiltinType(BuiltinType::OCLSampler);
}

inline bool Type::isEventT() const {
  return isSpecificBuiltinType(BuiltinType::OCLEvent);
}

inline bool Type::isClkEventT() const {
  return isSpecificBuiltinType(BuiltinType::OCLClkEvent);
}

inline bool Type::isQueueT() const {
  return isSpecificBuiltinType(BuiltinType::OCLQueue);
}

inline bool Type::isNDRangeT() const {
  return isSpecificBuiltinType(BuiltinType::OCLNDRange);
}

inline bool Type::isReserveIDT() const {
  return isSpecificBuiltinType(BuiltinType::OCLReserveID);
}

inline bool Type::isImageType() const {
  return isImage3dT() || isImage2dT() || isImage2dArrayT() ||
         isImage2dDepthT() || isImage2dArrayDepthT() || isImage2dMSAAT() ||
         isImage2dArrayMSAAT() || isImage2dMSAATDepth() ||
         isImage2dArrayMSAATDepth() || isImage1dT() || isImage1dArrayT() ||
         isImage1dBufferT();
}

inline bool Type::isOpenCLSpecificType() const {
  return isSamplerT() || isEventT() || isImageType() || isClkEventT() ||
         isQueueT() || isNDRangeT() || isReserveIDT();
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
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(this))
    return BT->isPlaceholderType();
  return false;
}

inline const BuiltinType *Type::getAsPlaceholderType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(this))
    if (BT->isPlaceholderType())
      return BT;
  return nullptr;
}

inline bool Type::isSpecificPlaceholderType(unsigned K) const {
  assert(BuiltinType::isPlaceholderTypeKind((BuiltinType::Kind) K));
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(this))
    return (BT->getKind() == (BuiltinType::Kind) K);
  return false;
}

inline bool Type::isNonOverloadPlaceholderType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(this))
    return BT->isNonOverloadPlaceholderType();
  return false;
}

inline bool Type::isVoidType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Void;
  return false;
}

inline bool Type::isHalfType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Half;
  // FIXME: Should we allow complex __fp16? Probably not.
  return false;
}

inline bool Type::isNullPtrType() const {
  if (const BuiltinType *BT = getAs<BuiltinType>())
    return BT->getKind() == BuiltinType::NullPtr;
  return false;
}

extern bool IsEnumDeclComplete(EnumDecl *);
extern bool IsEnumDeclScoped(EnumDecl *);

inline bool Type::isIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::Int128;
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType)) {
    // Incomplete enum types are not treated as integer types.
    // FIXME: In C++, enum types are never integer types.
    return IsEnumDeclComplete(ET->getDecl()) &&
      !IsEnumDeclScoped(ET->getDecl());
  }
  return false;
}

inline bool Type::isScalarType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() > BuiltinType::Void &&
           BT->getKind() <= BuiltinType::NullPtr;
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    // Enums are scalar types, but only if they are defined.  Incomplete enums
    // are not treated as scalar types.
    return IsEnumDeclComplete(ET->getDecl());
  return isa<PointerType>(CanonicalType) ||
         isa<BlockPointerType>(CanonicalType) ||
         isa<MemberPointerType>(CanonicalType) ||
         isa<ComplexType>(CanonicalType) ||
         isa<ObjCObjectPointerType>(CanonicalType);
}

inline bool Type::isIntegralOrEnumerationType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::Int128;

  // Check for a complete enum type; incomplete enum types are not properly an
  // enumeration type in the sense required here.
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    return IsEnumDeclComplete(ET->getDecl());

  return false;  
}

inline bool Type::isBooleanType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Bool;
  return false;
}

inline bool Type::isUndeducedType() const {
  const AutoType *AT = getContainedAutoType();
  return AT && !AT->isDeduced();
}

/// \brief Determines whether this is a type for which one can define
/// an overloaded operator.
inline bool Type::isOverloadableType() const {
  return isDependentType() || isRecordType() || isEnumeralType();
}

/// \brief Determines whether this type can decay to a pointer type.
inline bool Type::canDecayToPointerType() const {
  return isFunctionType() || isArrayType();
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
                  DiagnosticsEngine::ak_qualtype);
  return DB;
}

/// Insertion operator for partial diagnostics.  This allows sending QualType's
/// into a diagnostic with <<.
inline const PartialDiagnostic &operator<<(const PartialDiagnostic &PD,
                                           QualType T) {
  PD.AddTaggedVal(reinterpret_cast<intptr_t>(T.getAsOpaquePtr()),
                  DiagnosticsEngine::ak_qualtype);
  return PD;
}

// Helper class template that is used by Type::getAs to ensure that one does
// not try to look through a qualified type to get to an array type.
template <typename T, bool isArrayType = (std::is_same<T, ArrayType>::value ||
                                          std::is_base_of<ArrayType, T>::value)>
struct ArrayType_cannot_be_used_with_getAs {};

template<typename T>
struct ArrayType_cannot_be_used_with_getAs<T, true>;

// Member-template getAs<specific type>'.
template <typename T> const T *Type::getAs() const {
  ArrayType_cannot_be_used_with_getAs<T> at;
  (void)at;

  // If this is directly a T type, return it.
  if (const T *Ty = dyn_cast<T>(this))
    return Ty;

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<T>(CanonicalType))
    return nullptr;

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
    return nullptr;

  // If this is a typedef for the type, strip the typedef off without
  // losing all typedef information.
  return cast<ArrayType>(getUnqualifiedDesugaredType());
}

template <typename T> const T *Type::castAs() const {
  ArrayType_cannot_be_used_with_getAs<T> at;
  (void) at;

  if (const T *ty = dyn_cast<T>(this)) return ty;
  assert(isa<T>(CanonicalType));
  return cast<T>(getUnqualifiedDesugaredType());
}

inline const ArrayType *Type::castAsArrayTypeUnsafe() const {
  assert(isa<ArrayType>(CanonicalType));
  if (const ArrayType *arr = dyn_cast<ArrayType>(this)) return arr;
  return cast<ArrayType>(getUnqualifiedDesugaredType());
}

}  // end namespace clang

#endif
