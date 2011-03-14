//===--- Type.cpp - Type representation and manipulation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements type-related functionality.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace clang;

bool QualType::isConstant(QualType T, ASTContext &Ctx) {
  if (T.isConstQualified())
    return true;

  if (const ArrayType *AT = Ctx.getAsArrayType(T))
    return AT->getElementType().isConstant(Ctx);

  return false;
}

unsigned ConstantArrayType::getNumAddressingBits(ASTContext &Context,
                                                 QualType ElementType,
                                               const llvm::APInt &NumElements) {
  llvm::APSInt SizeExtended(NumElements, true);
  unsigned SizeTypeBits = Context.getTypeSize(Context.getSizeType());
  SizeExtended = SizeExtended.extend(std::max(SizeTypeBits,
                                              SizeExtended.getBitWidth()) * 2);

  uint64_t ElementSize
    = Context.getTypeSizeInChars(ElementType).getQuantity();
  llvm::APSInt TotalSize(llvm::APInt(SizeExtended.getBitWidth(), ElementSize));
  TotalSize *= SizeExtended;  
  
  return TotalSize.getActiveBits();
}

unsigned ConstantArrayType::getMaxSizeBits(ASTContext &Context) {
  unsigned Bits = Context.getTypeSize(Context.getSizeType());
  
  // GCC appears to only allow 63 bits worth of address space when compiling
  // for 64-bit, so we do the same.
  if (Bits == 64)
    --Bits;
  
  return Bits;
}

DependentSizedArrayType::DependentSizedArrayType(const ASTContext &Context, 
                                                 QualType et, QualType can,
                                                 Expr *e, ArraySizeModifier sm,
                                                 unsigned tq,
                                                 SourceRange brackets)
    : ArrayType(DependentSizedArray, et, can, sm, tq, 
                (et->containsUnexpandedParameterPack() ||
                 (e && e->containsUnexpandedParameterPack()))),
      Context(Context), SizeExpr((Stmt*) e), Brackets(brackets) 
{
}

void DependentSizedArrayType::Profile(llvm::FoldingSetNodeID &ID,
                                      const ASTContext &Context,
                                      QualType ET,
                                      ArraySizeModifier SizeMod,
                                      unsigned TypeQuals,
                                      Expr *E) {
  ID.AddPointer(ET.getAsOpaquePtr());
  ID.AddInteger(SizeMod);
  ID.AddInteger(TypeQuals);
  E->Profile(ID, Context, true);
}

DependentSizedExtVectorType::DependentSizedExtVectorType(const
                                                         ASTContext &Context,
                                                         QualType ElementType,
                                                         QualType can, 
                                                         Expr *SizeExpr, 
                                                         SourceLocation loc)
    : Type(DependentSizedExtVector, can, /*Dependent=*/true,
           ElementType->isVariablyModifiedType(), 
           (ElementType->containsUnexpandedParameterPack() ||
            (SizeExpr && SizeExpr->containsUnexpandedParameterPack()))),
      Context(Context), SizeExpr(SizeExpr), ElementType(ElementType),
      loc(loc) 
{
}

void
DependentSizedExtVectorType::Profile(llvm::FoldingSetNodeID &ID,
                                     const ASTContext &Context,
                                     QualType ElementType, Expr *SizeExpr) {
  ID.AddPointer(ElementType.getAsOpaquePtr());
  SizeExpr->Profile(ID, Context, true);
}

VectorType::VectorType(QualType vecType, unsigned nElements, QualType canonType,
                       VectorKind vecKind)
  : Type(Vector, canonType, vecType->isDependentType(),
         vecType->isVariablyModifiedType(),
         vecType->containsUnexpandedParameterPack()),
    ElementType(vecType) 
{
  VectorTypeBits.VecKind = vecKind;
  VectorTypeBits.NumElements = nElements;
}

VectorType::VectorType(TypeClass tc, QualType vecType, unsigned nElements,
                       QualType canonType, VectorKind vecKind)
  : Type(tc, canonType, vecType->isDependentType(),
         vecType->isVariablyModifiedType(),
         vecType->containsUnexpandedParameterPack()), 
    ElementType(vecType) 
{
  VectorTypeBits.VecKind = vecKind;
  VectorTypeBits.NumElements = nElements;
}

/// getArrayElementTypeNoTypeQual - If this is an array type, return the
/// element type of the array, potentially with type qualifiers missing.
/// This method should never be used when type qualifiers are meaningful.
const Type *Type::getArrayElementTypeNoTypeQual() const {
  // If this is directly an array type, return it.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(this))
    return ATy->getElementType().getTypePtr();

  // If the canonical form of this type isn't the right kind, reject it.
  if (!isa<ArrayType>(CanonicalType))
    return 0;

  // If this is a typedef for an array type, strip the typedef off without
  // losing all typedef information.
  return cast<ArrayType>(getUnqualifiedDesugaredType())
    ->getElementType().getTypePtr();
}

/// getDesugaredType - Return the specified type with any "sugar" removed from
/// the type.  This takes off typedefs, typeof's etc.  If the outer level of
/// the type is already concrete, it returns it unmodified.  This is similar
/// to getting the canonical type, but it doesn't remove *all* typedefs.  For
/// example, it returns "T*" as "T*", (not as "int*"), because the pointer is
/// concrete.
QualType QualType::getDesugaredType(QualType T, const ASTContext &Context) {
  SplitQualType split = getSplitDesugaredType(T);
  return Context.getQualifiedType(split.first, split.second);
}

SplitQualType QualType::getSplitDesugaredType(QualType T) {
  QualifierCollector Qs;

  QualType Cur = T;
  while (true) {
    const Type *CurTy = Qs.strip(Cur);
    switch (CurTy->getTypeClass()) {
#define ABSTRACT_TYPE(Class, Parent)
#define TYPE(Class, Parent) \
    case Type::Class: { \
      const Class##Type *Ty = cast<Class##Type>(CurTy); \
      if (!Ty->isSugared()) \
        return SplitQualType(Ty, Qs); \
      Cur = Ty->desugar(); \
      break; \
    }
#include "clang/AST/TypeNodes.def"
    }
  }
}

SplitQualType QualType::getSplitUnqualifiedTypeImpl(QualType type) {
  SplitQualType split = type.split();

  // All the qualifiers we've seen so far.
  Qualifiers quals = split.second;

  // The last type node we saw with any nodes inside it.
  const Type *lastTypeWithQuals = split.first;

  while (true) {
    QualType next;

    // Do a single-step desugar, aborting the loop if the type isn't
    // sugared.
    switch (split.first->getTypeClass()) {
#define ABSTRACT_TYPE(Class, Parent)
#define TYPE(Class, Parent) \
    case Type::Class: { \
      const Class##Type *ty = cast<Class##Type>(split.first); \
      if (!ty->isSugared()) goto done; \
      next = ty->desugar(); \
      break; \
    }
#include "clang/AST/TypeNodes.def"
    }

    // Otherwise, split the underlying type.  If that yields qualifiers,
    // update the information.
    split = next.split();
    if (!split.second.empty()) {
      lastTypeWithQuals = split.first;
      quals.addConsistentQualifiers(split.second);
    }
  }

 done:
  return SplitQualType(lastTypeWithQuals, quals);
}

QualType QualType::IgnoreParens(QualType T) {
  // FIXME: this seems inherently un-qualifiers-safe.
  while (const ParenType *PT = T->getAs<ParenType>())
    T = PT->getInnerType();
  return T;
}

/// getUnqualifiedDesugaredType - Pull any qualifiers and syntactic
/// sugar off the given type.  This should produce an object of the
/// same dynamic type as the canonical type.
const Type *Type::getUnqualifiedDesugaredType() const {
  const Type *Cur = this;

  while (true) {
    switch (Cur->getTypeClass()) {
#define ABSTRACT_TYPE(Class, Parent)
#define TYPE(Class, Parent) \
    case Class: { \
      const Class##Type *Ty = cast<Class##Type>(Cur); \
      if (!Ty->isSugared()) return Cur; \
      Cur = Ty->desugar().getTypePtr(); \
      break; \
    }
#include "clang/AST/TypeNodes.def"
    }
  }
}

/// isVoidType - Helper method to determine if this is the 'void' type.
bool Type::isVoidType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Void;
  return false;
}

bool Type::isDerivedType() const {
  switch (CanonicalType->getTypeClass()) {
  case Pointer:
  case VariableArray:
  case ConstantArray:
  case IncompleteArray:
  case FunctionProto:
  case FunctionNoProto:
  case LValueReference:
  case RValueReference:
  case Record:
    return true;
  default:
    return false;
  }
}

bool Type::isClassType() const {
  if (const RecordType *RT = getAs<RecordType>())
    return RT->getDecl()->isClass();
  return false;
}
bool Type::isStructureType() const {
  if (const RecordType *RT = getAs<RecordType>())
    return RT->getDecl()->isStruct();
  return false;
}
bool Type::isStructureOrClassType() const {
  if (const RecordType *RT = getAs<RecordType>())
    return RT->getDecl()->isStruct() || RT->getDecl()->isClass();
  return false;
}
bool Type::isVoidPointerType() const {
  if (const PointerType *PT = getAs<PointerType>())
    return PT->getPointeeType()->isVoidType();
  return false;
}

bool Type::isUnionType() const {
  if (const RecordType *RT = getAs<RecordType>())
    return RT->getDecl()->isUnion();
  return false;
}

bool Type::isComplexType() const {
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->getElementType()->isFloatingType();
  return false;
}

bool Type::isComplexIntegerType() const {
  // Check for GCC complex integer extension.
  return getAsComplexIntegerType();
}

const ComplexType *Type::getAsComplexIntegerType() const {
  if (const ComplexType *Complex = getAs<ComplexType>())
    if (Complex->getElementType()->isIntegerType())
      return Complex;
  return 0;
}

QualType Type::getPointeeType() const {
  if (const PointerType *PT = getAs<PointerType>())
    return PT->getPointeeType();
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>())
    return OPT->getPointeeType();
  if (const BlockPointerType *BPT = getAs<BlockPointerType>())
    return BPT->getPointeeType();
  if (const ReferenceType *RT = getAs<ReferenceType>())
    return RT->getPointeeType();
  return QualType();
}

const RecordType *Type::getAsStructureType() const {
  // If this is directly a structure type, return it.
  if (const RecordType *RT = dyn_cast<RecordType>(this)) {
    if (RT->getDecl()->isStruct())
      return RT;
  }

  // If the canonical form of this type isn't the right kind, reject it.
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType)) {
    if (!RT->getDecl()->isStruct())
      return 0;

    // If this is a typedef for a structure type, strip the typedef off without
    // losing all typedef information.
    return cast<RecordType>(getUnqualifiedDesugaredType());
  }
  return 0;
}

const RecordType *Type::getAsUnionType() const {
  // If this is directly a union type, return it.
  if (const RecordType *RT = dyn_cast<RecordType>(this)) {
    if (RT->getDecl()->isUnion())
      return RT;
  }

  // If the canonical form of this type isn't the right kind, reject it.
  if (const RecordType *RT = dyn_cast<RecordType>(CanonicalType)) {
    if (!RT->getDecl()->isUnion())
      return 0;

    // If this is a typedef for a union type, strip the typedef off without
    // losing all typedef information.
    return cast<RecordType>(getUnqualifiedDesugaredType());
  }

  return 0;
}

ObjCObjectType::ObjCObjectType(QualType Canonical, QualType Base,
                               ObjCProtocolDecl * const *Protocols,
                               unsigned NumProtocols)
  : Type(ObjCObject, Canonical, false, false, false),
    BaseType(Base) 
{
  ObjCObjectTypeBits.NumProtocols = NumProtocols;
  assert(getNumProtocols() == NumProtocols &&
         "bitfield overflow in protocol count");
  if (NumProtocols)
    memcpy(getProtocolStorage(), Protocols,
           NumProtocols * sizeof(ObjCProtocolDecl*));
}

const ObjCObjectType *Type::getAsObjCQualifiedInterfaceType() const {
  // There is no sugar for ObjCObjectType's, just return the canonical
  // type pointer if it is the right class.  There is no typedef information to
  // return and these cannot be Address-space qualified.
  if (const ObjCObjectType *T = getAs<ObjCObjectType>())
    if (T->getNumProtocols() && T->getInterface())
      return T;
  return 0;
}

bool Type::isObjCQualifiedInterfaceType() const {
  return getAsObjCQualifiedInterfaceType() != 0;
}

const ObjCObjectPointerType *Type::getAsObjCQualifiedIdType() const {
  // There is no sugar for ObjCQualifiedIdType's, just return the canonical
  // type pointer if it is the right class.
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>()) {
    if (OPT->isObjCQualifiedIdType())
      return OPT;
  }
  return 0;
}

const ObjCObjectPointerType *Type::getAsObjCInterfacePointerType() const {
  if (const ObjCObjectPointerType *OPT = getAs<ObjCObjectPointerType>()) {
    if (OPT->getInterfaceType())
      return OPT;
  }
  return 0;
}

const CXXRecordDecl *Type::getCXXRecordDeclForPointerType() const {
  if (const PointerType *PT = getAs<PointerType>())
    if (const RecordType *RT = PT->getPointeeType()->getAs<RecordType>())
      return dyn_cast<CXXRecordDecl>(RT->getDecl());
  return 0;
}

CXXRecordDecl *Type::getAsCXXRecordDecl() const {
  if (const RecordType *RT = getAs<RecordType>())
    return dyn_cast<CXXRecordDecl>(RT->getDecl());
  else if (const InjectedClassNameType *Injected
                                  = getAs<InjectedClassNameType>())
    return Injected->getDecl();
  
  return 0;
}

namespace {
  class GetContainedAutoVisitor :
    public TypeVisitor<GetContainedAutoVisitor, AutoType*> {
  public:
    using TypeVisitor<GetContainedAutoVisitor, AutoType*>::Visit;
    AutoType *Visit(QualType T) {
      if (T.isNull())
        return 0;
      return Visit(T.getTypePtr());
    }

    // The 'auto' type itself.
    AutoType *VisitAutoType(const AutoType *AT) {
      return const_cast<AutoType*>(AT);
    }

    // Only these types can contain the desired 'auto' type.
    AutoType *VisitPointerType(const PointerType *T) {
      return Visit(T->getPointeeType());
    }
    AutoType *VisitBlockPointerType(const BlockPointerType *T) {
      return Visit(T->getPointeeType());
    }
    AutoType *VisitReferenceType(const ReferenceType *T) {
      return Visit(T->getPointeeTypeAsWritten());
    }
    AutoType *VisitMemberPointerType(const MemberPointerType *T) {
      return Visit(T->getPointeeType());
    }
    AutoType *VisitArrayType(const ArrayType *T) {
      return Visit(T->getElementType());
    }
    AutoType *VisitDependentSizedExtVectorType(
      const DependentSizedExtVectorType *T) {
      return Visit(T->getElementType());
    }
    AutoType *VisitVectorType(const VectorType *T) {
      return Visit(T->getElementType());
    }
    AutoType *VisitFunctionType(const FunctionType *T) {
      return Visit(T->getResultType());
    }
    AutoType *VisitParenType(const ParenType *T) {
      return Visit(T->getInnerType());
    }
    AutoType *VisitAttributedType(const AttributedType *T) {
      return Visit(T->getModifiedType());
    }
  };
}

AutoType *Type::getContainedAutoType() const {
  return GetContainedAutoVisitor().Visit(this);
}

bool Type::isIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::Int128;
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    // Incomplete enum types are not treated as integer types.
    // FIXME: In C++, enum types are never integer types.
    return ET->getDecl()->isComplete();
  return false;
}

bool Type::hasIntegerRepresentation() const {
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isIntegerType();
  else
    return isIntegerType();
}

/// \brief Determine whether this type is an integral type.
///
/// This routine determines whether the given type is an integral type per 
/// C++ [basic.fundamental]p7. Although the C standard does not define the
/// term "integral type", it has a similar term "integer type", and in C++
/// the two terms are equivalent. However, C's "integer type" includes 
/// enumeration types, while C++'s "integer type" does not. The \c ASTContext
/// parameter is used to determine whether we should be following the C or
/// C++ rules when determining whether this type is an integral/integer type.
///
/// For cases where C permits "an integer type" and C++ permits "an integral
/// type", use this routine.
///
/// For cases where C permits "an integer type" and C++ permits "an integral
/// or enumeration type", use \c isIntegralOrEnumerationType() instead. 
///
/// \param Ctx The context in which this type occurs.
///
/// \returns true if the type is considered an integral type, false otherwise.
bool Type::isIntegralType(ASTContext &Ctx) const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
    BT->getKind() <= BuiltinType::Int128;
  
  if (!Ctx.getLangOptions().CPlusPlus)
    if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
      return ET->getDecl()->isComplete(); // Complete enum types are integral in C.
  
  return false;
}

bool Type::isIntegralOrEnumerationType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::Int128;

  // Check for a complete enum type; incomplete enum types are not properly an
  // enumeration type in the sense required here.
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    return ET->getDecl()->isComplete();

  return false;  
}

bool Type::isIntegralOrUnscopedEnumerationType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::Int128;

  // Check for a complete enum type; incomplete enum types are not properly an
  // enumeration type in the sense required here.
  // C++0x: However, if the underlying type of the enum is fixed, it is
  // considered complete.
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    return ET->getDecl()->isComplete() && !ET->getDecl()->isScoped();

  return false;
}


bool Type::isBooleanType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Bool;
  return false;
}

bool Type::isCharType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Char_U ||
           BT->getKind() == BuiltinType::UChar ||
           BT->getKind() == BuiltinType::Char_S ||
           BT->getKind() == BuiltinType::SChar;
  return false;
}

bool Type::isWideCharType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::WChar_S ||
           BT->getKind() == BuiltinType::WChar_U;
  return false;
}

/// \brief Determine whether this type is any of the built-in character
/// types.
bool Type::isAnyCharacterType() const {
  const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType);
  if (BT == 0) return false;
  switch (BT->getKind()) {
  default: return false;
  case BuiltinType::Char_U:
  case BuiltinType::UChar:
  case BuiltinType::WChar_U:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::Char_S:
  case BuiltinType::SChar:
  case BuiltinType::WChar_S:
    return true;
  }
}

/// isSignedIntegerType - Return true if this is an integer type that is
/// signed, according to C99 6.2.5p4 [char, signed char, short, int, long..],
/// an enum decl which has a signed representation
bool Type::isSignedIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType)) {
    return BT->getKind() >= BuiltinType::Char_S &&
           BT->getKind() <= BuiltinType::Int128;
  }

  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType)) {
    // Incomplete enum types are not treated as integer types.
    // FIXME: In C++, enum types are never integer types.
    if (ET->getDecl()->isComplete())
      return ET->getDecl()->getIntegerType()->isSignedIntegerType();
  }

  return false;
}

bool Type::hasSignedIntegerRepresentation() const {
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isSignedIntegerType();
  else
    return isSignedIntegerType();
}

/// isUnsignedIntegerType - Return true if this is an integer type that is
/// unsigned, according to C99 6.2.5p6 [which returns true for _Bool], an enum
/// decl which has an unsigned representation
bool Type::isUnsignedIntegerType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType)) {
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::UInt128;
  }

  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType)) {
    // Incomplete enum types are not treated as integer types.
    // FIXME: In C++, enum types are never integer types.
    if (ET->getDecl()->isComplete())
      return ET->getDecl()->getIntegerType()->isUnsignedIntegerType();
  }

  return false;
}

bool Type::hasUnsignedIntegerRepresentation() const {
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isUnsignedIntegerType();
  else
    return isUnsignedIntegerType();
}

bool Type::isFloatingType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Float &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const ComplexType *CT = dyn_cast<ComplexType>(CanonicalType))
    return CT->getElementType()->isFloatingType();
  return false;
}

bool Type::hasFloatingRepresentation() const {
  if (const VectorType *VT = dyn_cast<VectorType>(CanonicalType))
    return VT->getElementType()->isFloatingType();
  else
    return isFloatingType();
}

bool Type::isRealFloatingType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->isFloatingPoint();
  return false;
}

bool Type::isRealType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
      return ET->getDecl()->isComplete() && !ET->getDecl()->isScoped();
  return false;
}

bool Type::isArithmeticType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongDouble;
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    // GCC allows forward declaration of enum types (forbid by C99 6.7.2.3p2).
    // If a body isn't seen by the time we get here, return false.
    //
    // C++0x: Enumerations are not arithmetic types. For now, just return
    // false for scoped enumerations since that will disable any
    // unwanted implicit conversions.
    return !ET->getDecl()->isScoped() && ET->getDecl()->isComplete();
  return isa<ComplexType>(CanonicalType);
}

bool Type::isScalarType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() > BuiltinType::Void &&
           BT->getKind() <= BuiltinType::NullPtr;
  if (const EnumType *ET = dyn_cast<EnumType>(CanonicalType))
    // Enums are scalar types, but only if they are defined.  Incomplete enums
    // are not treated as scalar types.
    return ET->getDecl()->isComplete();
  return isa<PointerType>(CanonicalType) ||
         isa<BlockPointerType>(CanonicalType) ||
         isa<MemberPointerType>(CanonicalType) ||
         isa<ComplexType>(CanonicalType) ||
         isa<ObjCObjectPointerType>(CanonicalType);
}

Type::ScalarTypeKind Type::getScalarTypeKind() const {
  assert(isScalarType());

  const Type *T = CanonicalType.getTypePtr();
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(T)) {
    if (BT->getKind() == BuiltinType::Bool) return STK_Bool;
    if (BT->getKind() == BuiltinType::NullPtr) return STK_Pointer;
    if (BT->isInteger()) return STK_Integral;
    if (BT->isFloatingPoint()) return STK_Floating;
    llvm_unreachable("unknown scalar builtin type");
  } else if (isa<PointerType>(T) ||
             isa<BlockPointerType>(T) ||
             isa<ObjCObjectPointerType>(T)) {
    return STK_Pointer;
  } else if (isa<MemberPointerType>(T)) {
    return STK_MemberPointer;
  } else if (isa<EnumType>(T)) {
    assert(cast<EnumType>(T)->getDecl()->isComplete());
    return STK_Integral;
  } else if (const ComplexType *CT = dyn_cast<ComplexType>(T)) {
    if (CT->getElementType()->isRealFloatingType())
      return STK_FloatingComplex;
    return STK_IntegralComplex;
  }

  llvm_unreachable("unknown scalar type");
  return STK_Pointer;
}

/// \brief Determines whether the type is a C++ aggregate type or C
/// aggregate or union type.
///
/// An aggregate type is an array or a class type (struct, union, or
/// class) that has no user-declared constructors, no private or
/// protected non-static data members, no base classes, and no virtual
/// functions (C++ [dcl.init.aggr]p1). The notion of an aggregate type
/// subsumes the notion of C aggregates (C99 6.2.5p21) because it also
/// includes union types.
bool Type::isAggregateType() const {
  if (const RecordType *Record = dyn_cast<RecordType>(CanonicalType)) {
    if (CXXRecordDecl *ClassDecl = dyn_cast<CXXRecordDecl>(Record->getDecl()))
      return ClassDecl->isAggregate();

    return true;
  }

  return isa<ArrayType>(CanonicalType);
}

/// isConstantSizeType - Return true if this is not a variable sized type,
/// according to the rules of C99 6.7.5p3.  It is not legal to call this on
/// incomplete types or dependent types.
bool Type::isConstantSizeType() const {
  assert(!isIncompleteType() && "This doesn't make sense for incomplete types");
  assert(!isDependentType() && "This doesn't make sense for dependent types");
  // The VAT must have a size, as it is known to be complete.
  return !isa<VariableArrayType>(CanonicalType);
}

/// isIncompleteType - Return true if this is an incomplete type (C99 6.2.5p1)
/// - a type that can describe objects, but which lacks information needed to
/// determine its size.
bool Type::isIncompleteType() const {
  switch (CanonicalType->getTypeClass()) {
  default: return false;
  case Builtin:
    // Void is the only incomplete builtin type.  Per C99 6.2.5p19, it can never
    // be completed.
    return isVoidType();
  case Enum:
    // An enumeration with fixed underlying type is complete (C++0x 7.2p3).
    if (cast<EnumType>(CanonicalType)->getDecl()->isFixed())
        return false;
    // Fall through.
  case Record:
    // A tagged type (struct/union/enum/class) is incomplete if the decl is a
    // forward declaration, but not a full definition (C99 6.2.5p22).
    return !cast<TagType>(CanonicalType)->getDecl()->isDefinition();
  case ConstantArray:
    // An array is incomplete if its element type is incomplete
    // (C++ [dcl.array]p1).
    // We don't handle variable arrays (they're not allowed in C++) or
    // dependent-sized arrays (dependent types are never treated as incomplete).
    return cast<ArrayType>(CanonicalType)->getElementType()->isIncompleteType();
  case IncompleteArray:
    // An array of unknown size is an incomplete type (C99 6.2.5p22).
    return true;
  case ObjCObject:
    return cast<ObjCObjectType>(CanonicalType)->getBaseType()
                                                         ->isIncompleteType();
  case ObjCInterface:
    // ObjC interfaces are incomplete if they are @class, not @interface.
    return cast<ObjCInterfaceType>(CanonicalType)->getDecl()->isForwardDecl();
  }
}

/// isPODType - Return true if this is a plain-old-data type (C++ 3.9p10)
bool Type::isPODType() const {
  // The compiler shouldn't query this for incomplete types, but the user might.
  // We return false for that case. Except for incomplete arrays of PODs, which
  // are PODs according to the standard.
  if (isIncompleteArrayType() &&
      cast<ArrayType>(CanonicalType)->getElementType()->isPODType())
    return true;
  if (isIncompleteType())
    return false;

  switch (CanonicalType->getTypeClass()) {
    // Everything not explicitly mentioned is not POD.
  default: return false;
  case VariableArray:
  case ConstantArray:
    // IncompleteArray is handled above.
    return cast<ArrayType>(CanonicalType)->getElementType()->isPODType();

  case Builtin:
  case Complex:
  case Pointer:
  case MemberPointer:
  case Vector:
  case ExtVector:
  case ObjCObjectPointer:
  case BlockPointer:
    return true;

  case Enum:
    return true;

  case Record:
    if (CXXRecordDecl *ClassDecl
          = dyn_cast<CXXRecordDecl>(cast<RecordType>(CanonicalType)->getDecl()))
      return ClassDecl->isPOD();

    // C struct/union is POD.
    return true;
  }
}

bool Type::isLiteralType() const {
  if (isIncompleteType())
    return false;

  // C++0x [basic.types]p10:
  //   A type is a literal type if it is:
  switch (CanonicalType->getTypeClass()) {
    // We're whitelisting
  default: return false;

    //   -- a scalar type
  case Builtin:
  case Complex:
  case Pointer:
  case MemberPointer:
  case Vector:
  case ExtVector:
  case ObjCObjectPointer:
  case Enum:
    return true;

    //   -- a class type with ...
  case Record:
    // FIXME: Do the tests
    return false;

    //   -- an array of literal type
    // Extension: variable arrays cannot be literal types, since they're
    // runtime-sized.
  case ConstantArray:
    return cast<ArrayType>(CanonicalType)->getElementType()->isLiteralType();
  }
}

bool Type::isPromotableIntegerType() const {
  if (const BuiltinType *BT = getAs<BuiltinType>())
    switch (BT->getKind()) {
    case BuiltinType::Bool:
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
      return true;
    default:
      return false;
    }

  // Enumerated types are promotable to their compatible integer types
  // (C99 6.3.1.1) a.k.a. its underlying type (C++ [conv.prom]p2).
  if (const EnumType *ET = getAs<EnumType>()){
    if (this->isDependentType() || ET->getDecl()->getPromotionType().isNull()
        || ET->getDecl()->isScoped())
      return false;
    
    const BuiltinType *BT
      = ET->getDecl()->getPromotionType()->getAs<BuiltinType>();
    return BT->getKind() == BuiltinType::Int
           || BT->getKind() == BuiltinType::UInt;
  }
  
  return false;
}

bool Type::isNullPtrType() const {
  if (const BuiltinType *BT = getAs<BuiltinType>())
    return BT->getKind() == BuiltinType::NullPtr;
  return false;
}

bool Type::isSpecifierType() const {
  // Note that this intentionally does not use the canonical type.
  switch (getTypeClass()) {
  case Builtin:
  case Record:
  case Enum:
  case Typedef:
  case Complex:
  case TypeOfExpr:
  case TypeOf:
  case TemplateTypeParm:
  case SubstTemplateTypeParm:
  case TemplateSpecialization:
  case Elaborated:
  case DependentName:
  case DependentTemplateSpecialization:
  case ObjCInterface:
  case ObjCObject:
  case ObjCObjectPointer: // FIXME: object pointers aren't really specifiers
    return true;
  default:
    return false;
  }
}

ElaboratedTypeKeyword
TypeWithKeyword::getKeywordForTypeSpec(unsigned TypeSpec) {
  switch (TypeSpec) {
  default: return ETK_None;
  case TST_typename: return ETK_Typename;
  case TST_class: return ETK_Class;
  case TST_struct: return ETK_Struct;
  case TST_union: return ETK_Union;
  case TST_enum: return ETK_Enum;
  }
}

TagTypeKind
TypeWithKeyword::getTagTypeKindForTypeSpec(unsigned TypeSpec) {
  switch(TypeSpec) {
  case TST_class: return TTK_Class;
  case TST_struct: return TTK_Struct;
  case TST_union: return TTK_Union;
  case TST_enum: return TTK_Enum;
  }
  
  llvm_unreachable("Type specifier is not a tag type kind.");
  return TTK_Union;
}

ElaboratedTypeKeyword
TypeWithKeyword::getKeywordForTagTypeKind(TagTypeKind Kind) {
  switch (Kind) {
  case TTK_Class: return ETK_Class;
  case TTK_Struct: return ETK_Struct;
  case TTK_Union: return ETK_Union;
  case TTK_Enum: return ETK_Enum;
  }
  llvm_unreachable("Unknown tag type kind.");
}

TagTypeKind
TypeWithKeyword::getTagTypeKindForKeyword(ElaboratedTypeKeyword Keyword) {
  switch (Keyword) {
  case ETK_Class: return TTK_Class;
  case ETK_Struct: return TTK_Struct;
  case ETK_Union: return TTK_Union;
  case ETK_Enum: return TTK_Enum;
  case ETK_None: // Fall through.
  case ETK_Typename:
    llvm_unreachable("Elaborated type keyword is not a tag type kind.");
  }
  llvm_unreachable("Unknown elaborated type keyword.");
}

bool
TypeWithKeyword::KeywordIsTagTypeKind(ElaboratedTypeKeyword Keyword) {
  switch (Keyword) {
  case ETK_None:
  case ETK_Typename:
    return false;
  case ETK_Class:
  case ETK_Struct:
  case ETK_Union:
  case ETK_Enum:
    return true;
  }
  llvm_unreachable("Unknown elaborated type keyword.");
}

const char*
TypeWithKeyword::getKeywordName(ElaboratedTypeKeyword Keyword) {
  switch (Keyword) {
  case ETK_None: return "";
  case ETK_Typename: return "typename";
  case ETK_Class:  return "class";
  case ETK_Struct: return "struct";
  case ETK_Union:  return "union";
  case ETK_Enum:   return "enum";
  }

  llvm_unreachable("Unknown elaborated type keyword.");
  return "";
}

DependentTemplateSpecializationType::DependentTemplateSpecializationType(
                         ElaboratedTypeKeyword Keyword,
                         NestedNameSpecifier *NNS, const IdentifierInfo *Name,
                         unsigned NumArgs, const TemplateArgument *Args,
                         QualType Canon)
  : TypeWithKeyword(Keyword, DependentTemplateSpecialization, Canon, true,
                    /*VariablyModified=*/false,
                    NNS && NNS->containsUnexpandedParameterPack()),
    NNS(NNS), Name(Name), NumArgs(NumArgs) {
  assert((!NNS || NNS->isDependent()) &&
         "DependentTemplateSpecializatonType requires dependent qualifier");
  for (unsigned I = 0; I != NumArgs; ++I) {
    if (Args[I].containsUnexpandedParameterPack())
      setContainsUnexpandedParameterPack();

    new (&getArgBuffer()[I]) TemplateArgument(Args[I]);
  }
}

void
DependentTemplateSpecializationType::Profile(llvm::FoldingSetNodeID &ID,
                                             const ASTContext &Context,
                                             ElaboratedTypeKeyword Keyword,
                                             NestedNameSpecifier *Qualifier,
                                             const IdentifierInfo *Name,
                                             unsigned NumArgs,
                                             const TemplateArgument *Args) {
  ID.AddInteger(Keyword);
  ID.AddPointer(Qualifier);
  ID.AddPointer(Name);
  for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
    Args[Idx].Profile(ID, Context);
}

bool Type::isElaboratedTypeSpecifier() const {
  ElaboratedTypeKeyword Keyword;
  if (const ElaboratedType *Elab = dyn_cast<ElaboratedType>(this))
    Keyword = Elab->getKeyword();
  else if (const DependentNameType *DepName = dyn_cast<DependentNameType>(this))
    Keyword = DepName->getKeyword();
  else if (const DependentTemplateSpecializationType *DepTST =
             dyn_cast<DependentTemplateSpecializationType>(this))
    Keyword = DepTST->getKeyword();
  else
    return false;

  return TypeWithKeyword::KeywordIsTagTypeKind(Keyword);
}

const char *Type::getTypeClassName() const {
  switch (TypeBits.TC) {
#define ABSTRACT_TYPE(Derived, Base)
#define TYPE(Derived, Base) case Derived: return #Derived;
#include "clang/AST/TypeNodes.def"
  }
  
  llvm_unreachable("Invalid type class.");
  return 0;
}

const char *BuiltinType::getName(const LangOptions &LO) const {
  switch (getKind()) {
  case Void:              return "void";
  case Bool:              return LO.Bool ? "bool" : "_Bool";
  case Char_S:            return "char";
  case Char_U:            return "char";
  case SChar:             return "signed char";
  case Short:             return "short";
  case Int:               return "int";
  case Long:              return "long";
  case LongLong:          return "long long";
  case Int128:            return "__int128_t";
  case UChar:             return "unsigned char";
  case UShort:            return "unsigned short";
  case UInt:              return "unsigned int";
  case ULong:             return "unsigned long";
  case ULongLong:         return "unsigned long long";
  case UInt128:           return "__uint128_t";
  case Float:             return "float";
  case Double:            return "double";
  case LongDouble:        return "long double";
  case WChar_S:
  case WChar_U:           return "wchar_t";
  case Char16:            return "char16_t";
  case Char32:            return "char32_t";
  case NullPtr:           return "nullptr_t";
  case Overload:          return "<overloaded function type>";
  case Dependent:         return "<dependent type>";
  case ObjCId:            return "id";
  case ObjCClass:         return "Class";
  case ObjCSel:           return "SEL";
  }
  
  llvm_unreachable("Invalid builtin type.");
  return 0;
}

QualType QualType::getNonLValueExprType(ASTContext &Context) const {
  if (const ReferenceType *RefType = getTypePtr()->getAs<ReferenceType>())
    return RefType->getPointeeType();
  
  // C++0x [basic.lval]:
  //   Class prvalues can have cv-qualified types; non-class prvalues always 
  //   have cv-unqualified types.
  //
  // See also C99 6.3.2.1p2.
  if (!Context.getLangOptions().CPlusPlus ||
      (!getTypePtr()->isDependentType() && !getTypePtr()->isRecordType()))
    return getUnqualifiedType();
  
  return *this;
}

llvm::StringRef FunctionType::getNameForCallConv(CallingConv CC) {
  switch (CC) {
  case CC_Default: 
    llvm_unreachable("no name for default cc");
    return "";

  case CC_C: return "cdecl";
  case CC_X86StdCall: return "stdcall";
  case CC_X86FastCall: return "fastcall";
  case CC_X86ThisCall: return "thiscall";
  case CC_X86Pascal: return "pascal";
  }

  llvm_unreachable("Invalid calling convention.");
  return "";
}

FunctionProtoType::FunctionProtoType(QualType result, const QualType *args,
                                     unsigned numArgs, QualType canonical,
                                     const ExtProtoInfo &epi)
  : FunctionType(FunctionProto, result, epi.Variadic, epi.TypeQuals, 
                 epi.RefQualifier, canonical,
                 result->isDependentType(),
                 result->isVariablyModifiedType(),
                 result->containsUnexpandedParameterPack(),
                 epi.ExtInfo),
    NumArgs(numArgs), NumExceptions(epi.NumExceptions),
    HasExceptionSpec(epi.HasExceptionSpec),
    HasAnyExceptionSpec(epi.HasAnyExceptionSpec)
{
  // Fill in the trailing argument array.
  QualType *argSlot = reinterpret_cast<QualType*>(this+1);
  for (unsigned i = 0; i != numArgs; ++i) {
    if (args[i]->isDependentType())
      setDependent();

    if (args[i]->containsUnexpandedParameterPack())
      setContainsUnexpandedParameterPack();

    argSlot[i] = args[i];
  }
  
  // Fill in the exception array.
  QualType *exnSlot = argSlot + numArgs;
  for (unsigned i = 0, e = epi.NumExceptions; i != e; ++i) {
    if (epi.Exceptions[i]->isDependentType())
      setDependent();

    if (epi.Exceptions[i]->containsUnexpandedParameterPack())
      setContainsUnexpandedParameterPack();

    exnSlot[i] = epi.Exceptions[i];
  }
}

bool FunctionProtoType::isTemplateVariadic() const {
  for (unsigned ArgIdx = getNumArgs(); ArgIdx; --ArgIdx)
    if (isa<PackExpansionType>(getArgType(ArgIdx - 1)))
      return true;
  
  return false;
}

void FunctionProtoType::Profile(llvm::FoldingSetNodeID &ID, QualType Result,
                                const QualType *ArgTys, unsigned NumArgs,
                                const ExtProtoInfo &epi) {
  ID.AddPointer(Result.getAsOpaquePtr());
  for (unsigned i = 0; i != NumArgs; ++i)
    ID.AddPointer(ArgTys[i].getAsOpaquePtr());
  ID.AddBoolean(epi.Variadic);
  ID.AddInteger(epi.TypeQuals);
  ID.AddInteger(epi.RefQualifier);
  if (epi.HasExceptionSpec) {
    ID.AddBoolean(epi.HasAnyExceptionSpec);
    for (unsigned i = 0; i != epi.NumExceptions; ++i)
      ID.AddPointer(epi.Exceptions[i].getAsOpaquePtr());
  }
  epi.ExtInfo.Profile(ID);
}

void FunctionProtoType::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getResultType(), arg_type_begin(), NumArgs, getExtProtoInfo());
}

QualType TypedefType::desugar() const {
  return getDecl()->getUnderlyingType();
}

TypeOfExprType::TypeOfExprType(Expr *E, QualType can)
  : Type(TypeOfExpr, can, E->isTypeDependent(), 
         E->getType()->isVariablyModifiedType(),
         E->containsUnexpandedParameterPack()), 
    TOExpr(E) {
}

QualType TypeOfExprType::desugar() const {
  return getUnderlyingExpr()->getType();
}

void DependentTypeOfExprType::Profile(llvm::FoldingSetNodeID &ID,
                                      const ASTContext &Context, Expr *E) {
  E->Profile(ID, Context, true);
}

DecltypeType::DecltypeType(Expr *E, QualType underlyingType, QualType can)
  : Type(Decltype, can, E->isTypeDependent(), 
         E->getType()->isVariablyModifiedType(), 
         E->containsUnexpandedParameterPack()), 
    E(E),
  UnderlyingType(underlyingType) {
}

DependentDecltypeType::DependentDecltypeType(const ASTContext &Context, Expr *E)
  : DecltypeType(E, Context.DependentTy), Context(Context) { }

void DependentDecltypeType::Profile(llvm::FoldingSetNodeID &ID,
                                    const ASTContext &Context, Expr *E) {
  E->Profile(ID, Context, true);
}

TagType::TagType(TypeClass TC, const TagDecl *D, QualType can)
  : Type(TC, can, D->isDependentType(), /*VariablyModified=*/false, 
         /*ContainsUnexpandedParameterPack=*/false),
    decl(const_cast<TagDecl*>(D)) {}

static TagDecl *getInterestingTagDecl(TagDecl *decl) {
  for (TagDecl::redecl_iterator I = decl->redecls_begin(),
                                E = decl->redecls_end();
       I != E; ++I) {
    if (I->isDefinition() || I->isBeingDefined())
      return *I;
  }
  // If there's no definition (not even in progress), return what we have.
  return decl;
}

TagDecl *TagType::getDecl() const {
  return getInterestingTagDecl(decl);
}

bool TagType::isBeingDefined() const {
  return getDecl()->isBeingDefined();
}

CXXRecordDecl *InjectedClassNameType::getDecl() const {
  return cast<CXXRecordDecl>(getInterestingTagDecl(Decl));
}

bool RecordType::classof(const TagType *TT) {
  return isa<RecordDecl>(TT->getDecl());
}

bool EnumType::classof(const TagType *TT) {
  return isa<EnumDecl>(TT->getDecl());
}

SubstTemplateTypeParmPackType::
SubstTemplateTypeParmPackType(const TemplateTypeParmType *Param, 
                              QualType Canon,
                              const TemplateArgument &ArgPack)
  : Type(SubstTemplateTypeParmPack, Canon, true, false, true), Replaced(Param), 
    Arguments(ArgPack.pack_begin()), NumArguments(ArgPack.pack_size()) 
{ 
}

TemplateArgument SubstTemplateTypeParmPackType::getArgumentPack() const {
  return TemplateArgument(Arguments, NumArguments);
}

void SubstTemplateTypeParmPackType::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getReplacedParameter(), getArgumentPack());
}

void SubstTemplateTypeParmPackType::Profile(llvm::FoldingSetNodeID &ID,
                                           const TemplateTypeParmType *Replaced,
                                            const TemplateArgument &ArgPack) {
  ID.AddPointer(Replaced);
  ID.AddInteger(ArgPack.pack_size());
  for (TemplateArgument::pack_iterator P = ArgPack.pack_begin(), 
                                    PEnd = ArgPack.pack_end();
       P != PEnd; ++P)
    ID.AddPointer(P->getAsType().getAsOpaquePtr());
}

bool TemplateSpecializationType::
anyDependentTemplateArguments(const TemplateArgumentListInfo &Args) {
  return anyDependentTemplateArguments(Args.getArgumentArray(), Args.size());
}

bool TemplateSpecializationType::
anyDependentTemplateArguments(const TemplateArgumentLoc *Args, unsigned N) {
  for (unsigned i = 0; i != N; ++i)
    if (Args[i].getArgument().isDependent())
      return true;
  return false;
}

bool TemplateSpecializationType::
anyDependentTemplateArguments(const TemplateArgument *Args, unsigned N) {
  for (unsigned i = 0; i != N; ++i)
    if (Args[i].isDependent())
      return true;
  return false;
}

TemplateSpecializationType::
TemplateSpecializationType(TemplateName T,
                           const TemplateArgument *Args,
                           unsigned NumArgs, QualType Canon)
  : Type(TemplateSpecialization,
         Canon.isNull()? QualType(this, 0) : Canon,
         T.isDependent(), false, T.containsUnexpandedParameterPack()),
    Template(T), NumArgs(NumArgs) 
{
  assert(!T.getAsDependentTemplateName() && 
         "Use DependentTemplateSpecializationType for dependent template-name");
  assert((!Canon.isNull() ||
          T.isDependent() || anyDependentTemplateArguments(Args, NumArgs)) &&
         "No canonical type for non-dependent class template specialization");

  TemplateArgument *TemplateArgs
    = reinterpret_cast<TemplateArgument *>(this + 1);
  for (unsigned Arg = 0; Arg < NumArgs; ++Arg) {
    // Update dependent and variably-modified bits.
    if (Args[Arg].isDependent())
      setDependent();
    if (Args[Arg].getKind() == TemplateArgument::Type &&
        Args[Arg].getAsType()->isVariablyModifiedType())
      setVariablyModified();
    if (Args[Arg].containsUnexpandedParameterPack())
      setContainsUnexpandedParameterPack();

    new (&TemplateArgs[Arg]) TemplateArgument(Args[Arg]);
  }
}

void
TemplateSpecializationType::Profile(llvm::FoldingSetNodeID &ID,
                                    TemplateName T,
                                    const TemplateArgument *Args,
                                    unsigned NumArgs,
                                    const ASTContext &Context) {
  T.Profile(ID);
  for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
    Args[Idx].Profile(ID, Context);
}

QualType
QualifierCollector::apply(const ASTContext &Context, QualType QT) const {
  if (!hasNonFastQualifiers())
    return QT.withFastQualifiers(getFastQualifiers());

  return Context.getQualifiedType(QT, *this);
}

QualType
QualifierCollector::apply(const ASTContext &Context, const Type *T) const {
  if (!hasNonFastQualifiers())
    return QualType(T, getFastQualifiers());

  return Context.getQualifiedType(T, *this);
}

void ObjCObjectTypeImpl::Profile(llvm::FoldingSetNodeID &ID,
                                 QualType BaseType,
                                 ObjCProtocolDecl * const *Protocols,
                                 unsigned NumProtocols) {
  ID.AddPointer(BaseType.getAsOpaquePtr());
  for (unsigned i = 0; i != NumProtocols; i++)
    ID.AddPointer(Protocols[i]);
}

void ObjCObjectTypeImpl::Profile(llvm::FoldingSetNodeID &ID) {
  Profile(ID, getBaseType(), qual_begin(), getNumProtocols());
}

namespace {

/// \brief The cached properties of a type.
class CachedProperties {
  char linkage;
  char visibility;
  bool local;
  
public:
  CachedProperties(Linkage linkage, Visibility visibility, bool local)
    : linkage(linkage), visibility(visibility), local(local) {}
  
  Linkage getLinkage() const { return (Linkage) linkage; }
  Visibility getVisibility() const { return (Visibility) visibility; }
  bool hasLocalOrUnnamedType() const { return local; }
  
  friend CachedProperties merge(CachedProperties L, CachedProperties R) {
    return CachedProperties(minLinkage(L.getLinkage(), R.getLinkage()),
                            minVisibility(L.getVisibility(), R.getVisibility()),
                         L.hasLocalOrUnnamedType() | R.hasLocalOrUnnamedType());
  }
};
}

static CachedProperties computeCachedProperties(const Type *T);

namespace clang {
/// The type-property cache.  This is templated so as to be
/// instantiated at an internal type to prevent unnecessary symbol
/// leakage.
template <class Private> class TypePropertyCache {
public:
  static CachedProperties get(QualType T) {
    return get(T.getTypePtr());
  }

  static CachedProperties get(const Type *T) {
    ensure(T);
    return CachedProperties(T->TypeBits.getLinkage(),
                            T->TypeBits.getVisibility(),
                            T->TypeBits.hasLocalOrUnnamedType());
  }

  static void ensure(const Type *T) {
    // If the cache is valid, we're okay.
    if (T->TypeBits.isCacheValid()) return;

    // If this type is non-canonical, ask its canonical type for the
    // relevant information.
    if (!T->isCanonicalUnqualified()) {
      const Type *CT = T->getCanonicalTypeInternal().getTypePtr();
      ensure(CT);
      T->TypeBits.CacheValidAndVisibility =
        CT->TypeBits.CacheValidAndVisibility;
      T->TypeBits.CachedLinkage = CT->TypeBits.CachedLinkage;
      T->TypeBits.CachedLocalOrUnnamed = CT->TypeBits.CachedLocalOrUnnamed;
      return;
    }

    // Compute the cached properties and then set the cache.
    CachedProperties Result = computeCachedProperties(T);
    T->TypeBits.CacheValidAndVisibility = Result.getVisibility() + 1U;
    assert(T->TypeBits.isCacheValid() &&
           T->TypeBits.getVisibility() == Result.getVisibility());
    T->TypeBits.CachedLinkage = Result.getLinkage();
    T->TypeBits.CachedLocalOrUnnamed = Result.hasLocalOrUnnamedType();
  }
};
}

// Instantiate the friend template at a private class.  In a
// reasonable implementation, these symbols will be internal.
// It is terrible that this is the best way to accomplish this.
namespace { class Private {}; }
typedef TypePropertyCache<Private> Cache;

static CachedProperties computeCachedProperties(const Type *T) {
  switch (T->getTypeClass()) {
#define TYPE(Class,Base)
#define NON_CANONICAL_TYPE(Class,Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    llvm_unreachable("didn't expect a non-canonical type here");

#define TYPE(Class,Base)
#define DEPENDENT_TYPE(Class,Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class,Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    // Treat dependent types as external.
    assert(T->isDependentType());
    return CachedProperties(ExternalLinkage, DefaultVisibility, false);

  case Type::Builtin:
    // C++ [basic.link]p8:
    //   A type is said to have linkage if and only if:
    //     - it is a fundamental type (3.9.1); or
    return CachedProperties(ExternalLinkage, DefaultVisibility, false);

  case Type::Record:
  case Type::Enum: {
    const TagDecl *Tag = cast<TagType>(T)->getDecl();

    // C++ [basic.link]p8:
    //     - it is a class or enumeration type that is named (or has a name
    //       for linkage purposes (7.1.3)) and the name has linkage; or
    //     -  it is a specialization of a class template (14); or
    NamedDecl::LinkageInfo LV = Tag->getLinkageAndVisibility();
    bool IsLocalOrUnnamed =
      Tag->getDeclContext()->isFunctionOrMethod() ||
      (!Tag->getIdentifier() && !Tag->getTypedefForAnonDecl());
    return CachedProperties(LV.linkage(), LV.visibility(), IsLocalOrUnnamed);
  }

    // C++ [basic.link]p8:
    //   - it is a compound type (3.9.2) other than a class or enumeration, 
    //     compounded exclusively from types that have linkage; or
  case Type::Complex:
    return Cache::get(cast<ComplexType>(T)->getElementType());
  case Type::Pointer:
    return Cache::get(cast<PointerType>(T)->getPointeeType());
  case Type::BlockPointer:
    return Cache::get(cast<BlockPointerType>(T)->getPointeeType());
  case Type::LValueReference:
  case Type::RValueReference:
    return Cache::get(cast<ReferenceType>(T)->getPointeeType());
  case Type::MemberPointer: {
    const MemberPointerType *MPT = cast<MemberPointerType>(T);
    return merge(Cache::get(MPT->getClass()),
                 Cache::get(MPT->getPointeeType()));
  }
  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    return Cache::get(cast<ArrayType>(T)->getElementType());
  case Type::Vector:
  case Type::ExtVector:
    return Cache::get(cast<VectorType>(T)->getElementType());
  case Type::FunctionNoProto:
    return Cache::get(cast<FunctionType>(T)->getResultType());
  case Type::FunctionProto: {
    const FunctionProtoType *FPT = cast<FunctionProtoType>(T);
    CachedProperties result = Cache::get(FPT->getResultType());
    for (FunctionProtoType::arg_type_iterator ai = FPT->arg_type_begin(),
           ae = FPT->arg_type_end(); ai != ae; ++ai)
      result = merge(result, Cache::get(*ai));
    return result;
  }
  case Type::ObjCInterface: {
    NamedDecl::LinkageInfo LV =
      cast<ObjCInterfaceType>(T)->getDecl()->getLinkageAndVisibility();
    return CachedProperties(LV.linkage(), LV.visibility(), false);
  }
  case Type::ObjCObject:
    return Cache::get(cast<ObjCObjectType>(T)->getBaseType());
  case Type::ObjCObjectPointer:
    return Cache::get(cast<ObjCObjectPointerType>(T)->getPointeeType());
  }

  llvm_unreachable("unhandled type class");

  // C++ [basic.link]p8:
  //   Names not covered by these rules have no linkage.
  return CachedProperties(NoLinkage, DefaultVisibility, false);
}

/// \brief Determine the linkage of this type.
Linkage Type::getLinkage() const {
  Cache::ensure(this);
  return TypeBits.getLinkage();
}

/// \brief Determine the linkage of this type.
Visibility Type::getVisibility() const {
  Cache::ensure(this);
  return TypeBits.getVisibility();
}

bool Type::hasUnnamedOrLocalType() const {
  Cache::ensure(this);
  return TypeBits.hasLocalOrUnnamedType();
}

std::pair<Linkage,Visibility> Type::getLinkageAndVisibility() const {
  Cache::ensure(this);
  return std::make_pair(TypeBits.getLinkage(), TypeBits.getVisibility());
}

void Type::ClearLinkageCache() {
  TypeBits.CacheValidAndVisibility = 0;
  if (QualType(this, 0) != CanonicalType)
    CanonicalType->TypeBits.CacheValidAndVisibility = 0;
}

bool Type::hasSizedVLAType() const {
  if (!isVariablyModifiedType()) return false;

  if (const PointerType *ptr = getAs<PointerType>())
    return ptr->getPointeeType()->hasSizedVLAType();
  if (const ReferenceType *ref = getAs<ReferenceType>())
    return ref->getPointeeType()->hasSizedVLAType();
  if (const ArrayType *arr = getAsArrayTypeUnsafe()) {
    if (isa<VariableArrayType>(arr) && 
        cast<VariableArrayType>(arr)->getSizeExpr())
      return true;

    return arr->getElementType()->hasSizedVLAType();
  }

  return false;
}

QualType::DestructionKind QualType::isDestructedTypeImpl(QualType type) {
  /// Currently, the only destruction kind we recognize is C++ objects
  /// with non-trivial destructors.
  const CXXRecordDecl *record =
    type->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
  if (record && !record->hasTrivialDestructor())
    return DK_cxx_destructor;

  return DK_none;
}
