#ifndef TYPE_H
#define TYPE_H


namespace ABI{
class Type {
  public:
  enum TypeClass {
Builtin
  };
  explicit Type(TypeClass TC) : Tc(TC) {} 

    TypeClass getTypeClass() const { return Tc; }
    virtual ~Type() = default;

  private:
  //save the type class
    TypeClass Tc;
  }


class ABIBuiltinType : public Type {
  public:
  enum Kind {
    Void,
  Bool,
  Integer,
  SignedInteger,
  UnsignedInteger,
  Int128,
  FloatingPoint
  };

  private:
  Kind kind; 

  public:
  ABIBuiltinType(Kind K)
      : Type(Builtin), kind(K){}
  // ABIType(Kind K = Integer);
  Kind getKind() const { return kind;}

  bool isInteger() const {
    return getKind() == Kind::Integer || getKind() == Kind::SignedInteger || getKind() == Kind::UnsignedInteger || getKind() == Kind::Int128;
  }

  bool isSignedInteger() const {
    return getKind() == Kind::SignedInteger;
  }

  bool isUnsignedInteger() const {
    return getKind() == Kind::UnsignedInteger;
  }

  bool isFloatingPoint() const {
    return getKind() == Kind::FloatingPoint;
  }

}

}

#endif










/// Base class that is common to both the \c ExtQuals and \c Type
/// classes, which allows \c QualType to access the common fields between the
/// two.
class ExtQualsTypeCommonBase {
  friend class ExtQuals;
  friend class QualType;
  friend class Type;
  friend class ASTReader;

  /// The "base" type of an extended qualifiers type (\c ExtQuals) or
  /// a self-referential pointer (for \c Type).
  ///
  /// This pointer allows an efficient mapping from a QualType to its
  /// underlying type pointer.
  const Type *const BaseType;

  /// The canonical type of this type.  A QualType.
  QualType CanonicalType;

  ExtQualsTypeCommonBase(const Type *baseType, QualType canon)
      : BaseType(baseType), CanonicalType(canon) {}
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
class alignas(TypeAlignment) ExtQuals : public ExtQualsTypeCommonBase,
                                        public llvm::FoldingSetNode {
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
        Quals(quals) {
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
  LangAS getAddressSpace() const { return Quals.getAddressSpace(); }

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
class alignas(TypeAlignment) Type : public ExtQualsTypeCommonBase {
public:
  enum TypeClass {
#define TYPE(Class, Base) Class,
#define LAST_TYPE(Class) TypeLast = Class
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.inc"
  };

private:
  /// Bitfields required by the Type class.
  class TypeBitfields {
    friend class Type;
    template <class T> friend class TypePropertyCache;

    /// TypeClass bitfield - Enum that specifies what subclass this belongs to.
    LLVM_PREFERRED_TYPE(TypeClass)
    unsigned TC : 8;

    /// Store information on the type dependency.
    LLVM_PREFERRED_TYPE(TypeDependence)
    unsigned Dependence : llvm::BitWidth<TypeDependence>;

    /// True if the cache (i.e. the bitfields here starting with
    /// 'Cache') is valid.
    LLVM_PREFERRED_TYPE(bool)
    mutable unsigned CacheValid : 1;

    /// Linkage of this type.
    LLVM_PREFERRED_TYPE(Linkage)
    mutable unsigned CachedLinkage : 3;

    /// Whether this type involves and local or unnamed types.
    LLVM_PREFERRED_TYPE(bool)
    mutable unsigned CachedLocalOrUnnamed : 1;

    /// Whether this type comes from an AST file.
    LLVM_PREFERRED_TYPE(bool)
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
  enum { NumTypeBits = 8 + llvm::BitWidth<TypeDependence> + 6 };

protected:
  // These classes allow subclasses to somewhat cleanly pack bitfields
  // into Type.

  class ArrayTypeBitfields {
    friend class ArrayType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// CVR qualifiers from declarations like
    /// 'int X[static restrict 4]'. For function parameters only.
    LLVM_PREFERRED_TYPE(Qualifiers)
    unsigned IndexTypeQuals : 3;

    /// Storage class qualifiers from declarations like
    /// 'int X[static restrict 4]'. For function parameters only.
    LLVM_PREFERRED_TYPE(ArraySizeModifier)
    unsigned SizeModifier : 3;
  };
  enum { NumArrayTypeBits = NumTypeBits + 6 };

  class ConstantArrayTypeBitfields {
    friend class ConstantArrayType;

    LLVM_PREFERRED_TYPE(ArrayTypeBitfields)
    unsigned : NumArrayTypeBits;

    /// Whether we have a stored size expression.
    LLVM_PREFERRED_TYPE(bool)
    unsigned HasExternalSize : 1;

    LLVM_PREFERRED_TYPE(unsigned)
    unsigned SizeWidth : 5;
  };

  class BuiltinTypeBitfields {
    friend class BuiltinType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// The kind (BuiltinType::Kind) of builtin type this is.
    static constexpr unsigned NumOfBuiltinTypeBits = 9;
    unsigned Kind : NumOfBuiltinTypeBits;
  };

public:
  static constexpr int FunctionTypeNumParamsWidth = 16;
  static constexpr int FunctionTypeNumParamsLimit = (1 << 16) - 1;

protected:
  /// FunctionTypeBitfields store various bits belonging to FunctionProtoType.
  /// Only common bits are stored here. Additional uncommon bits are stored
  /// in a trailing object after FunctionProtoType.
  class FunctionTypeBitfields {
    friend class FunctionProtoType;
    friend class FunctionType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// The ref-qualifier associated with a \c FunctionProtoType.
    ///
    /// This is a value of type \c RefQualifierKind.
    LLVM_PREFERRED_TYPE(RefQualifierKind)
    unsigned RefQualifier : 2;

    /// Used only by FunctionProtoType, put here to pack with the
    /// other bitfields.
    /// The qualifiers are part of FunctionProtoType because...
    ///
    /// C++ 8.3.5p4: The return type, the parameter type list and the
    /// cv-qualifier-seq, [...], are part of the function type.
    LLVM_PREFERRED_TYPE(Qualifiers)
    unsigned FastTypeQuals : Qualifiers::FastWidth;
    /// Whether this function has extended Qualifiers.
    LLVM_PREFERRED_TYPE(bool)
    unsigned HasExtQuals : 1;

    /// The type of exception specification this function has.
    LLVM_PREFERRED_TYPE(ExceptionSpecificationType)
    unsigned ExceptionSpecType : 4;

    /// Whether this function has extended parameter information.
    LLVM_PREFERRED_TYPE(bool)
    unsigned HasExtParameterInfos : 1;

    /// Whether this function has extra bitfields for the prototype.
    LLVM_PREFERRED_TYPE(bool)
    unsigned HasExtraBitfields : 1;

    /// Whether the function is variadic.
    LLVM_PREFERRED_TYPE(bool)
    unsigned Variadic : 1;

    /// Whether this function has a trailing return type.
    LLVM_PREFERRED_TYPE(bool)
    unsigned HasTrailingReturn : 1;

    /// Extra information which affects how the function is called, like
    /// regparm and the calling convention.
    LLVM_PREFERRED_TYPE(CallingConv)
    unsigned ExtInfo : 14;

    /// The number of parameters this function has, not counting '...'.
    /// According to [implimits] 8 bits should be enough here but this is
    /// somewhat easy to exceed with metaprogramming and so we would like to
    /// keep NumParams as wide as reasonably possible.
    unsigned NumParams : FunctionTypeNumParamsWidth;
  };

  class ObjCObjectTypeBitfields {
    friend class ObjCObjectType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// The number of type arguments stored directly on this object type.
    unsigned NumTypeArgs : 7;

    /// The number of protocols stored directly on this object type.
    unsigned NumProtocols : 6;

    /// Whether this is a "kindof" type.
    LLVM_PREFERRED_TYPE(bool)
    unsigned IsKindOf : 1;
  };

  class ReferenceTypeBitfields {
    friend class ReferenceType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
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
    LLVM_PREFERRED_TYPE(bool)
    unsigned SpelledAsLValue : 1;

    /// True if the inner type is a reference type.  This only happens
    /// in non-canonical forms.
    LLVM_PREFERRED_TYPE(bool)
    unsigned InnerRef : 1;
  };

  class TypeWithKeywordBitfields {
    friend class TypeWithKeyword;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// An ElaboratedTypeKeyword.  8 bits for efficient access.
    LLVM_PREFERRED_TYPE(ElaboratedTypeKeyword)
    unsigned Keyword : 8;
  };

  enum { NumTypeWithKeywordBits = NumTypeBits + 8 };

  class ElaboratedTypeBitfields {
    friend class ElaboratedType;

    LLVM_PREFERRED_TYPE(TypeWithKeywordBitfields)
    unsigned : NumTypeWithKeywordBits;

    /// Whether the ElaboratedType has a trailing OwnedTagDecl.
    LLVM_PREFERRED_TYPE(bool)
    unsigned HasOwnedTagDecl : 1;
  };

  class VectorTypeBitfields {
    friend class VectorType;
    friend class DependentVectorType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// The kind of vector, either a generic vector type or some
    /// target-specific vector type such as for AltiVec or Neon.
    LLVM_PREFERRED_TYPE(VectorKind)
    unsigned VecKind : 4;
    /// The number of elements in the vector.
    uint32_t NumElements;
  };

  class AttributedTypeBitfields {
    friend class AttributedType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    LLVM_PREFERRED_TYPE(attr::Kind)
    unsigned AttrKind : 32 - NumTypeBits;
  };

  class AutoTypeBitfields {
    friend class AutoType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// Was this placeholder type spelled as 'auto', 'decltype(auto)',
    /// or '__auto_type'?  AutoTypeKeyword value.
    LLVM_PREFERRED_TYPE(AutoTypeKeyword)
    unsigned Keyword : 2;

    /// The number of template arguments in the type-constraints, which is
    /// expected to be able to hold at least 1024 according to [implimits].
    /// However as this limit is somewhat easy to hit with template
    /// metaprogramming we'd prefer to keep it as large as possible.
    /// At the moment it has been left as a non-bitfield since this type
    /// safely fits in 64 bits as an unsigned, so there is no reason to
    /// introduce the performance impact of a bitfield.
    unsigned NumArgs;
  };

  class TypeOfBitfields {
    friend class TypeOfType;
    friend class TypeOfExprType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;
    LLVM_PREFERRED_TYPE(TypeOfKind)
    unsigned Kind : 1;
  };

  class UsingBitfields {
    friend class UsingType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// True if the underlying type is different from the declared one.
    LLVM_PREFERRED_TYPE(bool)
    unsigned hasTypeDifferentFromDecl : 1;
  };

  class TypedefBitfields {
    friend class TypedefType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// True if the underlying type is different from the declared one.
    LLVM_PREFERRED_TYPE(bool)
    unsigned hasTypeDifferentFromDecl : 1;
  };

  class TemplateTypeParmTypeBitfields {
    friend class TemplateTypeParmType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// The depth of the template parameter.
    unsigned Depth : 15;

    /// Whether this is a template parameter pack.
    LLVM_PREFERRED_TYPE(bool)
    unsigned ParameterPack : 1;

    /// The index of the template parameter.
    unsigned Index : 16;
  };

  class SubstTemplateTypeParmTypeBitfields {
    friend class SubstTemplateTypeParmType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    LLVM_PREFERRED_TYPE(bool)
    unsigned HasNonCanonicalUnderlyingType : 1;

    LLVM_PREFERRED_TYPE(SubstTemplateTypeParmTypeFlag)
    unsigned SubstitutionFlag : 1;

    // The index of the template parameter this substitution represents.
    unsigned Index : 15;

    /// Represents the index within a pack if this represents a substitution
    /// from a pack expansion. This index starts at the end of the pack and
    /// increments towards the beginning.
    /// Positive non-zero number represents the index + 1.
    /// Zero means this is not substituted from an expansion.
    unsigned PackIndex : 16;
  };

  class SubstTemplateTypeParmPackTypeBitfields {
    friend class SubstTemplateTypeParmPackType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    // The index of the template parameter this substitution represents.
    unsigned Index : 16;

    /// The number of template arguments in \c Arguments, which is
    /// expected to be able to hold at least 1024 according to [implimits].
    /// However as this limit is somewhat easy to hit with template
    /// metaprogramming we'd prefer to keep it as large as possible.
    unsigned NumArgs : 16;
  };

  class TemplateSpecializationTypeBitfields {
    friend class TemplateSpecializationType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// Whether this template specialization type is a substituted type alias.
    LLVM_PREFERRED_TYPE(bool)
    unsigned TypeAlias : 1;

    /// The number of template arguments named in this class template
    /// specialization, which is expected to be able to hold at least 1024
    /// according to [implimits]. However, as this limit is somewhat easy to
    /// hit with template metaprogramming we'd prefer to keep it as large
    /// as possible. At the moment it has been left as a non-bitfield since
    /// this type safely fits in 64 bits as an unsigned, so there is no reason
    /// to introduce the performance impact of a bitfield.
    unsigned NumArgs;
  };

  class DependentTemplateSpecializationTypeBitfields {
    friend class DependentTemplateSpecializationType;

    LLVM_PREFERRED_TYPE(TypeWithKeywordBitfields)
    unsigned : NumTypeWithKeywordBits;

    /// The number of template arguments named in this class template
    /// specialization, which is expected to be able to hold at least 1024
    /// according to [implimits]. However, as this limit is somewhat easy to
    /// hit with template metaprogramming we'd prefer to keep it as large
    /// as possible. At the moment it has been left as a non-bitfield since
    /// this type safely fits in 64 bits as an unsigned, so there is no reason
    /// to introduce the performance impact of a bitfield.
    unsigned NumArgs;
  };

  class PackExpansionTypeBitfields {
    friend class PackExpansionType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    /// The number of expansions that this pack expansion will
    /// generate when substituted (+1), which is expected to be able to
    /// hold at least 1024 according to [implimits]. However, as this limit
    /// is somewhat easy to hit with template metaprogramming we'd prefer to
    /// keep it as large as possible. At the moment it has been left as a
    /// non-bitfield since this type safely fits in 64 bits as an unsigned, so
    /// there is no reason to introduce the performance impact of a bitfield.
    ///
    /// This field will only have a non-zero value when some of the parameter
    /// packs that occur within the pattern have been substituted but others
    /// have not.
    unsigned NumExpansions;
  };

  class CountAttributedTypeBitfields {
    friend class CountAttributedType;

    LLVM_PREFERRED_TYPE(TypeBitfields)
    unsigned : NumTypeBits;

    static constexpr unsigned NumCoupledDeclsBits = 4;
    unsigned NumCoupledDecls : NumCoupledDeclsBits;
    LLVM_PREFERRED_TYPE(bool)
    unsigned CountInBytes : 1;
    LLVM_PREFERRED_TYPE(bool)
    unsigned OrNull : 1;
  };
  static_assert(sizeof(CountAttributedTypeBitfields) <= sizeof(unsigned));

  union {
    TypeBitfields TypeBits;
    ArrayTypeBitfields ArrayTypeBits;
    ConstantArrayTypeBitfields ConstantArrayTypeBits;
    AttributedTypeBitfields AttributedTypeBits;
    AutoTypeBitfields AutoTypeBits;
    TypeOfBitfields TypeOfBits;
    TypedefBitfields TypedefBits;
    UsingBitfields UsingBits;
    BuiltinTypeBitfields BuiltinTypeBits;
    FunctionTypeBitfields FunctionTypeBits;
    ObjCObjectTypeBitfields ObjCObjectTypeBits;
    ReferenceTypeBitfields ReferenceTypeBits;
    TypeWithKeywordBitfields TypeWithKeywordBits;
    ElaboratedTypeBitfields ElaboratedTypeBits;
    VectorTypeBitfields VectorTypeBits;
    TemplateTypeParmTypeBitfields TemplateTypeParmTypeBits;
    SubstTemplateTypeParmTypeBitfields SubstTemplateTypeParmTypeBits;
    SubstTemplateTypeParmPackTypeBitfields SubstTemplateTypeParmPackTypeBits;
    TemplateSpecializationTypeBitfields TemplateSpecializationTypeBits;
    DependentTemplateSpecializationTypeBitfields
      DependentTemplateSpecializationTypeBits;
    PackExpansionTypeBitfields PackExpansionTypeBits;
    CountAttributedTypeBitfields CountAttributedTypeBits;
  };

private:
  template <class T> friend class TypePropertyCache;

  /// Set whether this type comes from an AST file.
  void setFromAST(bool V = true) const {
    TypeBits.FromAST = V;
  }

protected:
  friend class ASTContext;

  Type(TypeClass tc, QualType canon, TypeDependence Dependence)
      : ExtQualsTypeCommonBase(this,
                               canon.isNull() ? QualType(this_(), 0) : canon) {
    static_assert(sizeof(*this) <=
                      alignof(decltype(*this)) + sizeof(ExtQualsTypeCommonBase),
                  "changing bitfields changed sizeof(Type)!");
    static_assert(alignof(decltype(*this)) % TypeAlignment == 0,
                  "Insufficient alignment!");
    TypeBits.TC = tc;
    TypeBits.Dependence = static_cast<unsigned>(Dependence);
    TypeBits.CacheValid = false;
    TypeBits.CachedLocalOrUnnamed = false;
    TypeBits.CachedLinkage = llvm::to_underlying(Linkage::Invalid);
    TypeBits.FromAST = false;
  }

  // silence VC++ warning C4355: 'this' : used in base member initializer list
  Type *this_() { return this; }

  void setDependence(TypeDependence D) {
    TypeBits.Dependence = static_cast<unsigned>(D);
  }

  void addDependence(TypeDependence D) { setDependence(getDependence() | D); }

public:
  friend class ASTReader;
  friend class ASTWriter;
  template <class T> friend class serialization::AbstractTypeReader;
  template <class T> friend class serialization::AbstractTypeWriter;

  Type(const Type &) = delete;
  Type(Type &&) = delete;
  Type &operator=(const Type &) = delete;
  Type &operator=(Type &&) = delete;

  TypeClass getTypeClass() const { return static_cast<TypeClass>(TypeBits.TC); }

  /// Whether this type comes from an AST file.
  bool isFromAST() const { return TypeBits.FromAST; }

  /// Whether this type is or contains an unexpanded parameter
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
    return getDependence() & TypeDependence::UnexpandedPack;
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

  /// As an extension, we classify types as one of "sized" or "sizeless";
  /// every type is one or the other.  Standard types are all sized;
  /// sizeless types are purely an extension.
  ///
  /// Sizeless types contain data with no specified size, alignment,
  /// or layout.
  bool isSizelessType() const;
  bool isSizelessBuiltinType() const;

  /// Returns true for all scalable vector types.
  bool isSizelessVectorType() const;

  /// Returns true for SVE scalable vector types.
  bool isSVESizelessBuiltinType() const;

  /// Returns true for RVV scalable vector types.
  bool isRVVSizelessBuiltinType() const;

  /// Check if this is a WebAssembly Externref Type.
  bool isWebAssemblyExternrefType() const;

  /// Returns true if this is a WebAssembly table type: either an array of
  /// reference types, or a pointer to a reference type (which can only be
  /// created by array to pointer decay).
  bool isWebAssemblyTableType() const;

  /// Determines if this is a sizeless type supported by the
  /// 'arm_sve_vector_bits' type attribute, which can be applied to a single
  /// SVE vector or predicate, excluding tuple types such as svint32x4_t.
  bool isSveVLSBuiltinType() const;

  /// Returns the representative type for the element of an SVE builtin type.
  /// This is used to represent fixed-length SVE vectors created with the
  /// 'arm_sve_vector_bits' type attribute as VectorType.
  QualType getSveEltType(const ASTContext &Ctx) const;

  /// Determines if this is a sizeless type supported by the
  /// 'riscv_rvv_vector_bits' type attribute, which can be applied to a single
  /// RVV vector or mask.
  bool isRVVVLSBuiltinType() const;

  /// Returns the representative type for the element of an RVV builtin type.
  /// This is used to represent fixed-length RVV vectors created with the
  /// 'riscv_rvv_vector_bits' type attribute as VectorType.
  QualType getRVVEltType(const ASTContext &Ctx) const;

  /// Returns the representative type for the element of a sizeless vector
  /// builtin type.
  QualType getSizelessVectorEltType(const ASTContext &Ctx) const;

  /// Types are partitioned into 3 broad categories (C99 6.2.5p1):
  /// object types, function types, and incomplete types.

  /// Return true if this is an incomplete type.
  /// A type that can describe objects, but which lacks information needed to
  /// determine its size (e.g. void, or a fwd declared struct). Clients of this
  /// routine will need to determine if the size is actually required.
  ///
  /// Def If non-null, and the type refers to some kind of declaration
  /// that can be completed (such as a C struct, C++ class, or Objective-C
  /// class), will be set to the declaration.
  bool isIncompleteType(NamedDecl **Def = nullptr) const;

  /// Return true if this is an incomplete or object
  /// type, in other words, not a function type.
  bool isIncompleteOrObjectType() const {
    return !isFunctionType();
  }

  /// Determine whether this type is an object type.
  bool isObjectType() const {
    // C++ [basic.types]p8:
    //   An object type is a (possibly cv-qualified) type that is not a
    //   function type, not a reference type, and not a void type.
    return !isReferenceType() && !isFunctionType() && !isVoidType();
  }

  /// Return true if this is a literal type
  /// (C++11 [basic.types]p10)
  bool isLiteralType(const ASTContext &Ctx) const;

  /// Determine if this type is a structural type, per C++20 [temp.param]p7.
  bool isStructuralType() const;

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

  /// Determine whether this type is a scoped enumeration type.
  bool isScopedEnumeralType() const;
  bool isBooleanType() const;
  bool isCharType() const;
  bool isWideCharType() const;
  bool isChar8Type() const;
  bool isChar16Type() const;
  bool isChar32Type() const;
  bool isAnyCharacterType() const;
  bool isIntegralType(const ASTContext &Ctx) const;

  /// Determine whether this type is an integral or enumeration type.
  bool isIntegralOrEnumerationType() const;

  /// Determine whether this type is an integral or unscoped enumeration type.
  bool isIntegralOrUnscopedEnumerationType() const;
  bool isUnscopedEnumerationType() const;

  /// Floating point categories.
  bool isRealFloatingType() const; // C99 6.2.5p10 (float, double, long double)
  /// isComplexType() does *not* include complex integers (a GCC extension).
  /// isComplexIntegerType() can be used to test for complex integers.
  bool isComplexType() const;      // C99 6.2.5p11 (complex)
  bool isAnyComplexType() const;   // C99 6.2.5p11 (complex) + Complex Int.
  bool isFloatingType() const;     // C99 6.2.5p11 (real floating + complex)
  bool isHalfType() const;         // OpenCL 6.1.1.1, NEON (IEEE 754-2008 half)
  bool isFloat16Type() const;      // C11 extension ISO/IEC TS 18661
  bool isFloat32Type() const;
  bool isDoubleType() const;
  bool isBFloat16Type() const;
  bool isMFloat8Type() const;
  bool isFloat128Type() const;
  bool isIbm128Type() const;
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
  bool isPointerOrReferenceType() const;
  bool isSignableType() const;
  bool isAnyPointerType() const;   // Any C pointer or ObjC object pointer
  bool isCountAttributedType() const;
  bool isBlockPointerType() const;
  bool isVoidPointerType() const;
  bool isReferenceType() const;
  bool isLValueReferenceType() const;
  bool isRValueReferenceType() const;
  bool isObjectPointerType() const;
  bool isFunctionPointerType() const;
  bool isFunctionReferenceType() const;
  bool isMemberPointerType() const;
  bool isMemberFunctionPointerType() const;
  bool isMemberDataPointerType() const;
  bool isArrayType() const;
  bool isConstantArrayType() const;
  bool isIncompleteArrayType() const;
  bool isVariableArrayType() const;
  bool isArrayParameterType() const;
  bool isDependentSizedArrayType() const;
  bool isRecordType() const;
  bool isClassType() const;
  bool isStructureType() const;
  bool isStructureTypeWithFlexibleArrayMember() const;
  bool isObjCBoxableRecordType() const;
  bool isInterfaceType() const;
  bool isStructureOrClassType() const;
  bool isUnionType() const;
  bool isComplexIntegerType() const;            // GCC _Complex integer type.
  bool isVectorType() const;                    // GCC vector type.
  bool isExtVectorType() const;                 // Extended vector type.
  bool isExtVectorBoolType() const;             // Extended vector type with bool element.
  // Extended vector type with bool element that is packed. HLSL doesn't pack
  // its bool vectors.
  bool isPackedVectorBoolType(const ASTContext &ctx) const;
  bool isSubscriptableVectorType() const;
  bool isMatrixType() const;                    // Matrix type.
  bool isConstantMatrixType() const;            // Constant matrix type.
  bool isDependentAddressSpaceType() const;     // value-dependent address space qualifier
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
  bool isDecltypeType() const;
  /// Was this type written with the special inert-in-ARC __unsafe_unretained
  /// qualifier?
  ///
  /// This approximates the answer to the following question: if this
  /// translation unit were compiled in ARC, would this type be qualified
  /// with __unsafe_unretained?
  bool isObjCInertUnsafeUnretainedType() const {
    return hasAttr(attr::ObjCInertUnsafeUnretained);
  }

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
  bool isNullPtrType() const;                   // C++11 std::nullptr_t or
                                                // C23   nullptr_t
  bool isNothrowT() const;                      // C++   std::nothrow_t
  bool isAlignValT() const;                     // C++17 std::align_val_t
  bool isStdByteType() const;                   // C++17 std::byte
  bool isAtomicType() const;                    // C11 _Atomic()
  bool isUndeducedAutoType() const;             // C++11 auto or
                                                // C++14 decltype(auto)
  bool isTypedefNameType() const;               // typedef or alias template

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix) \
  bool is##Id##Type() const;
#include "clang/Basic/OpenCLImageTypes.def"

  bool isImageType() const;                     // Any OpenCL image type

  bool isSamplerT() const;                      // OpenCL sampler_t
  bool isEventT() const;                        // OpenCL event_t
  bool isClkEventT() const;                     // OpenCL clk_event_t
  bool isQueueT() const;                        // OpenCL queue_t
  bool isReserveIDT() const;                    // OpenCL reserve_id_t

#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) \
  bool is##Id##Type() const;
#include "clang/Basic/OpenCLExtensionTypes.def"
  // Type defined in cl_intel_device_side_avc_motion_estimation OpenCL extension
  bool isOCLIntelSubgroupAVCType() const;
  bool isOCLExtOpaqueType() const;              // Any OpenCL extension type

  bool isPipeType() const;                      // OpenCL pipe type
  bool isBitIntType() const;                    // Bit-precise integer type
  bool isOpenCLSpecificType() const;            // Any OpenCL specific type

#define HLSL_INTANGIBLE_TYPE(Name, Id, SingletonId) bool is##Id##Type() const;
#include "clang/Basic/HLSLIntangibleTypes.def"
  bool isHLSLSpecificType() const; // Any HLSL specific type
  bool isHLSLBuiltinIntangibleType() const; // Any HLSL builtin intangible type
  bool isHLSLAttributedResourceType() const;
  bool isHLSLResourceRecord() const;
  bool isHLSLIntangibleType()
      const; // Any HLSL intangible type (builtin, array, class)

  /// Determines if this type, which must satisfy
  /// isObjCLifetimeType(), is implicitly __unsafe_unretained rather
  /// than implicitly __strong.
  bool isObjCARCImplicitlyUnretainedType() const;

  /// Check if the type is the CUDA device builtin surface type.
  bool isCUDADeviceBuiltinSurfaceType() const;
  /// Check if the type is the CUDA device builtin texture type.
  bool isCUDADeviceBuiltinTextureType() const;

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
    STK_FloatingComplex,
    STK_FixedPoint
  };

  /// Given that this is a scalar type, classify it.
  ScalarTypeKind getScalarTypeKind() const;

  TypeDependence getDependence() const {
    return static_cast<TypeDependence>(TypeBits.Dependence);
  }

  /// Whether this type is an error type.
  bool containsErrors() const {
    return getDependence() & TypeDependence::Error;
  }

  /// Whether this type is a dependent type, meaning that its definition
  /// somehow depends on a template parameter (C++ [temp.dep.type]).
  bool isDependentType() const {
    return getDependence() & TypeDependence::Dependent;
  }

  /// Determine whether this type is an instantiation-dependent type,
  /// meaning that the type involves a template parameter (even if the
  /// definition does not actually depend on the type substituted for that
  /// template parameter).
  bool isInstantiationDependentType() const {
    return getDependence() & TypeDependence::Instantiation;
  }

  /// Determine whether this type is an undeduced type, meaning that
  /// it somehow involves a C++11 'auto' type or similar which has not yet been
  /// deduced.
  bool isUndeducedType() const;

  /// Whether this type is a variably-modified type (C99 6.7.5).
  bool isVariablyModifiedType() const {
    return getDependence() & TypeDependence::VariablyModified;
  }

  /// Whether this type involves a variable-length array type
  /// with a definite size.
  bool hasSizedVLAType() const;

  /// Whether this type is or contains a local or unnamed type.
  bool hasUnnamedOrLocalType() const;

  bool isOverloadableType() const;

  /// Determine wither this type is a C++ elaborated-type-specifier.
  bool isElaboratedTypeSpecifier() const;

  bool canDecayToPointerType() const;

  /// Whether this type is represented natively as a pointer.  This includes
  /// pointers, references, block pointers, and Objective-C interface,
  /// qualified id, and qualified interface types, as well as nullptr_t.
  bool hasPointerRepresentation() const;

  /// Whether this type can represent an objective pointer type for the
  /// purpose of GC'ability
  bool hasObjCPointerRepresentation() const;

  /// Determine whether this type has an integer representation
  /// of some sort, e.g., it is an integer type or a vector.
  bool hasIntegerRepresentation() const;

  /// Determine whether this type has an signed integer representation
  /// of some sort, e.g., it is an signed integer type or a vector.
  bool hasSignedIntegerRepresentation() const;

  /// Determine whether this type has an unsigned integer representation
  /// of some sort, e.g., it is an unsigned integer type or a vector.
  bool hasUnsignedIntegerRepresentation() const;

  /// Determine whether this type has a floating-point representation
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

  /// Retrieves the CXXRecordDecl that this type refers to, either
  /// because the type is a RecordType or because it is the injected-class-name
  /// type of a class template or class template partial specialization.
  CXXRecordDecl *getAsCXXRecordDecl() const;

  /// Retrieves the RecordDecl this type refers to.
  RecordDecl *getAsRecordDecl() const;

  /// Retrieves the TagDecl that this type refers to, either
  /// because the type is a TagType or because it is the injected-class-name
  /// type of a class template or class template partial specialization.
  TagDecl *getAsTagDecl() const;

  /// If this is a pointer or reference to a RecordType, return the
  /// CXXRecordDecl that the type refers to.
  ///
  /// If this is not a pointer or reference, or the type being pointed to does
  /// not refer to a CXXRecordDecl, returns NULL.
  const CXXRecordDecl *getPointeeCXXRecordDecl() const;

  /// Get the DeducedType whose type will be deduced for a variable with
  /// an initializer of this type. This looks through declarators like pointer
  /// types, but not through decltype or typedefs.
  DeducedType *getContainedDeducedType() const;

  /// Get the AutoType whose type will be deduced for a variable with
  /// an initializer of this type. This looks through declarators like pointer
  /// types, but not through decltype or typedefs.
  AutoType *getContainedAutoType() const {
    return dyn_cast_or_null<AutoType>(getContainedDeducedType());
  }

  /// Determine whether this type was written with a leading 'auto'
  /// corresponding to a trailing return type (possibly for a nested
  /// function type within a pointer to function type or similar).
  bool hasAutoForTrailingReturnType() const;

  /// Member-template getAs<specific type>'.  Look through sugar for
  /// an instance of \<specific type>.   This scheme will eventually
  /// replace the specific getAsXXXX methods above.
  ///
  /// There are some specializations of this member template listed
  /// immediately following this class.
  template <typename T> const T *getAs() const;

  /// Member-template getAsAdjusted<specific type>. Look through specific kinds
  /// of sugar (parens, attributes, etc) for an instance of \<specific type>.
  /// This is used when you need to walk over sugar nodes that represent some
  /// kind of type adjustment from a type that was written as a \<specific type>
  /// to another type that is still canonically a \<specific type>.
  template <typename T> const T *getAsAdjusted() const;

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

  /// Determine whether this type had the specified attribute applied to it
  /// (looking through top-level type sugar).
  bool hasAttr(attr::Kind AK) const;

  /// Get the base element type of this type, potentially discarding type
  /// qualifiers.  This should never be used when type qualifiers
  /// are meaningful.
  const Type *getBaseElementTypeUnsafe() const;

  /// If this is an array type, return the element type of the array,
  /// potentially with type qualifiers missing.
  /// This should never be used when type qualifiers are meaningful.
  const Type *getArrayElementTypeNoTypeQual() const;

  /// If this is a pointer type, return the pointee type.
  /// If this is an array type, return the array element type.
  /// This should never be used when type qualifiers are meaningful.
  const Type *getPointeeOrArrayElementType() const;

  /// If this is a pointer, ObjC object pointer, or block
  /// pointer, this returns the respective pointee.
  QualType getPointeeType() const;

  /// Return the specified type with any "sugar" removed from the type,
  /// removing any typedefs, typeofs, etc., as well as any qualifiers.
  const Type *getUnqualifiedDesugaredType() const;

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

  /// Return true if this is a fixed point type according to
  /// ISO/IEC JTC1 SC22 WG14 N1169.
  bool isFixedPointType() const;

  /// Return true if this is a fixed point or integer type.
  bool isFixedPointOrIntegerType() const;

  /// Return true if this can be converted to (or from) a fixed point type.
  bool isConvertibleToFixedPointType() const;

  /// Return true if this is a saturated fixed point type according to
  /// ISO/IEC JTC1 SC22 WG14 N1169. This type can be signed or unsigned.
  bool isSaturatedFixedPointType() const;

  /// Return true if this is a saturated fixed point type according to
  /// ISO/IEC JTC1 SC22 WG14 N1169. This type can be signed or unsigned.
  bool isUnsaturatedFixedPointType() const;

  /// Return true if this is a fixed point type that is signed according
  /// to ISO/IEC JTC1 SC22 WG14 N1169. This type can also be saturated.
  bool isSignedFixedPointType() const;

  /// Return true if this is a fixed point type that is unsigned according
  /// to ISO/IEC JTC1 SC22 WG14 N1169. This type can also be saturated.
  bool isUnsignedFixedPointType() const;

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
  std::optional<NullabilityKind> getNullability() const;

  /// Determine whether the given type can have a nullability
  /// specifier applied to it, i.e., if it is any kind of pointer type.
  ///
  /// \param ResultIfUnknown The value to return if we don't yet know whether
  ///        this type can have nullability because it is dependent.
  bool canHaveNullability(bool ResultIfUnknown = true) const;

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
  std::optional<ArrayRef<QualType>>
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
  void dump(llvm::raw_ostream &OS, const ASTContext &Context) const;
};