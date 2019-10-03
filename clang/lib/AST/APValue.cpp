//===--- APValue.cpp - Union class for APFloat/APSInt/Complex -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the APValue class.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

/// The identity of a type_info object depends on the canonical unqualified
/// type only.
TypeInfoLValue::TypeInfoLValue(const Type *T)
    : T(T->getCanonicalTypeUnqualified().getTypePtr()) {}

void TypeInfoLValue::print(llvm::raw_ostream &Out,
                           const PrintingPolicy &Policy) const {
  Out << "typeid(";
  QualType(getType(), 0).print(Out, Policy);
  Out << ")";
}

static_assert(
    1 << llvm::PointerLikeTypeTraits<TypeInfoLValue>::NumLowBitsAvailable <=
        alignof(Type),
    "Type is insufficiently aligned");

APValue::LValueBase::LValueBase(const ValueDecl *P, unsigned I, unsigned V)
    : Ptr(P), Local{I, V} {}
APValue::LValueBase::LValueBase(const Expr *P, unsigned I, unsigned V)
    : Ptr(P), Local{I, V} {}

APValue::LValueBase APValue::LValueBase::getDynamicAlloc(DynamicAllocLValue LV,
                                                         QualType Type) {
  LValueBase Base;
  Base.Ptr = LV;
  Base.DynamicAllocType = Type.getAsOpaquePtr();
  return Base;
}

APValue::LValueBase APValue::LValueBase::getTypeInfo(TypeInfoLValue LV,
                                                     QualType TypeInfo) {
  LValueBase Base;
  Base.Ptr = LV;
  Base.TypeInfoType = TypeInfo.getAsOpaquePtr();
  return Base;
}

unsigned APValue::LValueBase::getCallIndex() const {
  return (is<TypeInfoLValue>() || is<DynamicAllocLValue>()) ? 0
                                                            : Local.CallIndex;
}

unsigned APValue::LValueBase::getVersion() const {
  return (is<TypeInfoLValue>() || is<DynamicAllocLValue>()) ? 0 : Local.Version;
}

QualType APValue::LValueBase::getTypeInfoType() const {
  assert(is<TypeInfoLValue>() && "not a type_info lvalue");
  return QualType::getFromOpaquePtr(TypeInfoType);
}

QualType APValue::LValueBase::getDynamicAllocType() const {
  assert(is<DynamicAllocLValue>() && "not a dynamic allocation lvalue");
  return QualType::getFromOpaquePtr(DynamicAllocType);
}

namespace clang {
bool operator==(const APValue::LValueBase &LHS,
                const APValue::LValueBase &RHS) {
  if (LHS.Ptr != RHS.Ptr)
    return false;
  if (LHS.is<TypeInfoLValue>())
    return true;
  return LHS.Local.CallIndex == RHS.Local.CallIndex &&
         LHS.Local.Version == RHS.Local.Version;
}
}

namespace {
  struct LVBase {
    APValue::LValueBase Base;
    CharUnits Offset;
    unsigned PathLength;
    bool IsNullPtr : 1;
    bool IsOnePastTheEnd : 1;
  };
}

void *APValue::LValueBase::getOpaqueValue() const {
  return Ptr.getOpaqueValue();
}

bool APValue::LValueBase::isNull() const {
  return Ptr.isNull();
}

APValue::LValueBase::operator bool () const {
  return static_cast<bool>(Ptr);
}

clang::APValue::LValueBase
llvm::DenseMapInfo<clang::APValue::LValueBase>::getEmptyKey() {
  return clang::APValue::LValueBase(
      DenseMapInfo<const ValueDecl*>::getEmptyKey());
}

clang::APValue::LValueBase
llvm::DenseMapInfo<clang::APValue::LValueBase>::getTombstoneKey() {
  return clang::APValue::LValueBase(
      DenseMapInfo<const ValueDecl*>::getTombstoneKey());
}

namespace clang {
llvm::hash_code hash_value(const APValue::LValueBase &Base) {
  if (Base.is<TypeInfoLValue>() || Base.is<DynamicAllocLValue>())
    return llvm::hash_value(Base.getOpaqueValue());
  return llvm::hash_combine(Base.getOpaqueValue(), Base.getCallIndex(),
                            Base.getVersion());
}
}

unsigned llvm::DenseMapInfo<clang::APValue::LValueBase>::getHashValue(
    const clang::APValue::LValueBase &Base) {
  return hash_value(Base);
}

bool llvm::DenseMapInfo<clang::APValue::LValueBase>::isEqual(
    const clang::APValue::LValueBase &LHS,
    const clang::APValue::LValueBase &RHS) {
  return LHS == RHS;
}

struct APValue::LV : LVBase {
  static const unsigned InlinePathSpace =
      (DataSize - sizeof(LVBase)) / sizeof(LValuePathEntry);

  /// Path - The sequence of base classes, fields and array indices to follow to
  /// walk from Base to the subobject. When performing GCC-style folding, there
  /// may not be such a path.
  union {
    LValuePathEntry Path[InlinePathSpace];
    LValuePathEntry *PathPtr;
  };

  LV() { PathLength = (unsigned)-1; }
  ~LV() { resizePath(0); }

  void resizePath(unsigned Length) {
    if (Length == PathLength)
      return;
    if (hasPathPtr())
      delete [] PathPtr;
    PathLength = Length;
    if (hasPathPtr())
      PathPtr = new LValuePathEntry[Length];
  }

  bool hasPath() const { return PathLength != (unsigned)-1; }
  bool hasPathPtr() const { return hasPath() && PathLength > InlinePathSpace; }

  LValuePathEntry *getPath() { return hasPathPtr() ? PathPtr : Path; }
  const LValuePathEntry *getPath() const {
    return hasPathPtr() ? PathPtr : Path;
  }
};

namespace {
  struct MemberPointerBase {
    llvm::PointerIntPair<const ValueDecl*, 1, bool> MemberAndIsDerivedMember;
    unsigned PathLength;
  };
}

struct APValue::MemberPointerData : MemberPointerBase {
  static const unsigned InlinePathSpace =
      (DataSize - sizeof(MemberPointerBase)) / sizeof(const CXXRecordDecl*);
  typedef const CXXRecordDecl *PathElem;
  union {
    PathElem Path[InlinePathSpace];
    PathElem *PathPtr;
  };

  MemberPointerData() { PathLength = 0; }
  ~MemberPointerData() { resizePath(0); }

  void resizePath(unsigned Length) {
    if (Length == PathLength)
      return;
    if (hasPathPtr())
      delete [] PathPtr;
    PathLength = Length;
    if (hasPathPtr())
      PathPtr = new PathElem[Length];
  }

  bool hasPathPtr() const { return PathLength > InlinePathSpace; }

  PathElem *getPath() { return hasPathPtr() ? PathPtr : Path; }
  const PathElem *getPath() const {
    return hasPathPtr() ? PathPtr : Path;
  }
};

// FIXME: Reduce the malloc traffic here.

APValue::Arr::Arr(unsigned NumElts, unsigned Size) :
  Elts(new APValue[NumElts + (NumElts != Size ? 1 : 0)]),
  NumElts(NumElts), ArrSize(Size) {}
APValue::Arr::~Arr() { delete [] Elts; }

APValue::StructData::StructData(unsigned NumBases, unsigned NumFields) :
  Elts(new APValue[NumBases+NumFields]),
  NumBases(NumBases), NumFields(NumFields) {}
APValue::StructData::~StructData() {
  delete [] Elts;
}

APValue::UnionData::UnionData() : Field(nullptr), Value(new APValue) {}
APValue::UnionData::~UnionData () {
  delete Value;
}

APValue::APValue(const APValue &RHS) : Kind(None) {
  switch (RHS.getKind()) {
  case None:
  case Indeterminate:
    Kind = RHS.getKind();
    break;
  case Int:
    MakeInt();
    setInt(RHS.getInt());
    break;
  case Float:
    MakeFloat();
    setFloat(RHS.getFloat());
    break;
  case FixedPoint: {
    APFixedPoint FXCopy = RHS.getFixedPoint();
    MakeFixedPoint(std::move(FXCopy));
    break;
  }
  case Vector:
    MakeVector();
    setVector(((const Vec *)(const char *)RHS.Data.buffer)->Elts,
              RHS.getVectorLength());
    break;
  case ComplexInt:
    MakeComplexInt();
    setComplexInt(RHS.getComplexIntReal(), RHS.getComplexIntImag());
    break;
  case ComplexFloat:
    MakeComplexFloat();
    setComplexFloat(RHS.getComplexFloatReal(), RHS.getComplexFloatImag());
    break;
  case LValue:
    MakeLValue();
    if (RHS.hasLValuePath())
      setLValue(RHS.getLValueBase(), RHS.getLValueOffset(), RHS.getLValuePath(),
                RHS.isLValueOnePastTheEnd(), RHS.isNullPointer());
    else
      setLValue(RHS.getLValueBase(), RHS.getLValueOffset(), NoLValuePath(),
                RHS.isNullPointer());
    break;
  case Array:
    MakeArray(RHS.getArrayInitializedElts(), RHS.getArraySize());
    for (unsigned I = 0, N = RHS.getArrayInitializedElts(); I != N; ++I)
      getArrayInitializedElt(I) = RHS.getArrayInitializedElt(I);
    if (RHS.hasArrayFiller())
      getArrayFiller() = RHS.getArrayFiller();
    break;
  case Struct:
    MakeStruct(RHS.getStructNumBases(), RHS.getStructNumFields());
    for (unsigned I = 0, N = RHS.getStructNumBases(); I != N; ++I)
      getStructBase(I) = RHS.getStructBase(I);
    for (unsigned I = 0, N = RHS.getStructNumFields(); I != N; ++I)
      getStructField(I) = RHS.getStructField(I);
    break;
  case Union:
    MakeUnion();
    setUnion(RHS.getUnionField(), RHS.getUnionValue());
    break;
  case MemberPointer:
    MakeMemberPointer(RHS.getMemberPointerDecl(),
                      RHS.isMemberPointerToDerivedMember(),
                      RHS.getMemberPointerPath());
    break;
  case AddrLabelDiff:
    MakeAddrLabelDiff();
    setAddrLabelDiff(RHS.getAddrLabelDiffLHS(), RHS.getAddrLabelDiffRHS());
    break;
  }
}

void APValue::DestroyDataAndMakeUninit() {
  if (Kind == Int)
    ((APSInt*)(char*)Data.buffer)->~APSInt();
  else if (Kind == Float)
    ((APFloat*)(char*)Data.buffer)->~APFloat();
  else if (Kind == FixedPoint)
    ((APFixedPoint *)(char *)Data.buffer)->~APFixedPoint();
  else if (Kind == Vector)
    ((Vec*)(char*)Data.buffer)->~Vec();
  else if (Kind == ComplexInt)
    ((ComplexAPSInt*)(char*)Data.buffer)->~ComplexAPSInt();
  else if (Kind == ComplexFloat)
    ((ComplexAPFloat*)(char*)Data.buffer)->~ComplexAPFloat();
  else if (Kind == LValue)
    ((LV*)(char*)Data.buffer)->~LV();
  else if (Kind == Array)
    ((Arr*)(char*)Data.buffer)->~Arr();
  else if (Kind == Struct)
    ((StructData*)(char*)Data.buffer)->~StructData();
  else if (Kind == Union)
    ((UnionData*)(char*)Data.buffer)->~UnionData();
  else if (Kind == MemberPointer)
    ((MemberPointerData*)(char*)Data.buffer)->~MemberPointerData();
  else if (Kind == AddrLabelDiff)
    ((AddrLabelDiffData*)(char*)Data.buffer)->~AddrLabelDiffData();
  Kind = None;
}

bool APValue::needsCleanup() const {
  switch (getKind()) {
  case None:
  case Indeterminate:
  case AddrLabelDiff:
    return false;
  case Struct:
  case Union:
  case Array:
  case Vector:
    return true;
  case Int:
    return getInt().needsCleanup();
  case Float:
    return getFloat().needsCleanup();
  case FixedPoint:
    return getFixedPoint().getValue().needsCleanup();
  case ComplexFloat:
    assert(getComplexFloatImag().needsCleanup() ==
               getComplexFloatReal().needsCleanup() &&
           "In _Complex float types, real and imaginary values always have the "
           "same size.");
    return getComplexFloatReal().needsCleanup();
  case ComplexInt:
    assert(getComplexIntImag().needsCleanup() ==
               getComplexIntReal().needsCleanup() &&
           "In _Complex int types, real and imaginary values must have the "
           "same size.");
    return getComplexIntReal().needsCleanup();
  case LValue:
    return reinterpret_cast<const LV *>(Data.buffer)->hasPathPtr();
  case MemberPointer:
    return reinterpret_cast<const MemberPointerData *>(Data.buffer)
        ->hasPathPtr();
  }
  llvm_unreachable("Unknown APValue kind!");
}

void APValue::swap(APValue &RHS) {
  std::swap(Kind, RHS.Kind);
  char TmpData[DataSize];
  memcpy(TmpData, Data.buffer, DataSize);
  memcpy(Data.buffer, RHS.Data.buffer, DataSize);
  memcpy(RHS.Data.buffer, TmpData, DataSize);
}

LLVM_DUMP_METHOD void APValue::dump() const {
  dump(llvm::errs());
  llvm::errs() << '\n';
}

static double GetApproxValue(const llvm::APFloat &F) {
  llvm::APFloat V = F;
  bool ignored;
  V.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
            &ignored);
  return V.convertToDouble();
}

void APValue::dump(raw_ostream &OS) const {
  switch (getKind()) {
  case None:
    OS << "None";
    return;
  case Indeterminate:
    OS << "Indeterminate";
    return;
  case Int:
    OS << "Int: " << getInt();
    return;
  case Float:
    OS << "Float: " << GetApproxValue(getFloat());
    return;
  case FixedPoint:
    OS << "FixedPoint : " << getFixedPoint();
    return;
  case Vector:
    OS << "Vector: ";
    getVectorElt(0).dump(OS);
    for (unsigned i = 1; i != getVectorLength(); ++i) {
      OS << ", ";
      getVectorElt(i).dump(OS);
    }
    return;
  case ComplexInt:
    OS << "ComplexInt: " << getComplexIntReal() << ", " << getComplexIntImag();
    return;
  case ComplexFloat:
    OS << "ComplexFloat: " << GetApproxValue(getComplexFloatReal())
       << ", " << GetApproxValue(getComplexFloatImag());
    return;
  case LValue:
    OS << "LValue: <todo>";
    return;
  case Array:
    OS << "Array: ";
    for (unsigned I = 0, N = getArrayInitializedElts(); I != N; ++I) {
      getArrayInitializedElt(I).dump(OS);
      if (I != getArraySize() - 1) OS << ", ";
    }
    if (hasArrayFiller()) {
      OS << getArraySize() - getArrayInitializedElts() << " x ";
      getArrayFiller().dump(OS);
    }
    return;
  case Struct:
    OS << "Struct ";
    if (unsigned N = getStructNumBases()) {
      OS << " bases: ";
      getStructBase(0).dump(OS);
      for (unsigned I = 1; I != N; ++I) {
        OS << ", ";
        getStructBase(I).dump(OS);
      }
    }
    if (unsigned N = getStructNumFields()) {
      OS << " fields: ";
      getStructField(0).dump(OS);
      for (unsigned I = 1; I != N; ++I) {
        OS << ", ";
        getStructField(I).dump(OS);
      }
    }
    return;
  case Union:
    OS << "Union: ";
    getUnionValue().dump(OS);
    return;
  case MemberPointer:
    OS << "MemberPointer: <todo>";
    return;
  case AddrLabelDiff:
    OS << "AddrLabelDiff: <todo>";
    return;
  }
  llvm_unreachable("Unknown APValue kind!");
}

void APValue::printPretty(raw_ostream &Out, const ASTContext &Ctx,
                          QualType Ty) const {
  switch (getKind()) {
  case APValue::None:
    Out << "<out of lifetime>";
    return;
  case APValue::Indeterminate:
    Out << "<uninitialized>";
    return;
  case APValue::Int:
    if (Ty->isBooleanType())
      Out << (getInt().getBoolValue() ? "true" : "false");
    else
      Out << getInt();
    return;
  case APValue::Float:
    Out << GetApproxValue(getFloat());
    return;
  case APValue::FixedPoint:
    Out << getFixedPoint();
    return;
  case APValue::Vector: {
    Out << '{';
    QualType ElemTy = Ty->castAs<VectorType>()->getElementType();
    getVectorElt(0).printPretty(Out, Ctx, ElemTy);
    for (unsigned i = 1; i != getVectorLength(); ++i) {
      Out << ", ";
      getVectorElt(i).printPretty(Out, Ctx, ElemTy);
    }
    Out << '}';
    return;
  }
  case APValue::ComplexInt:
    Out << getComplexIntReal() << "+" << getComplexIntImag() << "i";
    return;
  case APValue::ComplexFloat:
    Out << GetApproxValue(getComplexFloatReal()) << "+"
        << GetApproxValue(getComplexFloatImag()) << "i";
    return;
  case APValue::LValue: {
    bool IsReference = Ty->isReferenceType();
    QualType InnerTy
      = IsReference ? Ty.getNonReferenceType() : Ty->getPointeeType();
    if (InnerTy.isNull())
      InnerTy = Ty;

    LValueBase Base = getLValueBase();
    if (!Base) {
      if (isNullPointer()) {
        Out << (Ctx.getLangOpts().CPlusPlus11 ? "nullptr" : "0");
      } else if (IsReference) {
        Out << "*(" << InnerTy.stream(Ctx.getPrintingPolicy()) << "*)"
            << getLValueOffset().getQuantity();
      } else {
        Out << "(" << Ty.stream(Ctx.getPrintingPolicy()) << ")"
            << getLValueOffset().getQuantity();
      }
      return;
    }

    if (!hasLValuePath()) {
      // No lvalue path: just print the offset.
      CharUnits O = getLValueOffset();
      CharUnits S = Ctx.getTypeSizeInChars(InnerTy);
      if (!O.isZero()) {
        if (IsReference)
          Out << "*(";
        if (O % S) {
          Out << "(char*)";
          S = CharUnits::One();
        }
        Out << '&';
      } else if (!IsReference) {
        Out << '&';
      }

      if (const ValueDecl *VD = Base.dyn_cast<const ValueDecl*>())
        Out << *VD;
      else if (TypeInfoLValue TI = Base.dyn_cast<TypeInfoLValue>()) {
        TI.print(Out, Ctx.getPrintingPolicy());
      } else if (DynamicAllocLValue DA = Base.dyn_cast<DynamicAllocLValue>()) {
        Out << "{*new "
            << Base.getDynamicAllocType().stream(Ctx.getPrintingPolicy()) << "#"
            << DA.getIndex() << "}";
      } else {
        assert(Base.get<const Expr *>() != nullptr &&
               "Expecting non-null Expr");
        Base.get<const Expr*>()->printPretty(Out, nullptr,
                                             Ctx.getPrintingPolicy());
      }

      if (!O.isZero()) {
        Out << " + " << (O / S);
        if (IsReference)
          Out << ')';
      }
      return;
    }

    // We have an lvalue path. Print it out nicely.
    if (!IsReference)
      Out << '&';
    else if (isLValueOnePastTheEnd())
      Out << "*(&";

    QualType ElemTy;
    if (const ValueDecl *VD = Base.dyn_cast<const ValueDecl*>()) {
      Out << *VD;
      ElemTy = VD->getType();
    } else if (TypeInfoLValue TI = Base.dyn_cast<TypeInfoLValue>()) {
      TI.print(Out, Ctx.getPrintingPolicy());
      ElemTy = Base.getTypeInfoType();
    } else if (DynamicAllocLValue DA = Base.dyn_cast<DynamicAllocLValue>()) {
      Out << "{*new "
          << Base.getDynamicAllocType().stream(Ctx.getPrintingPolicy()) << "#"
          << DA.getIndex() << "}";
      ElemTy = Base.getDynamicAllocType();
    } else {
      const Expr *E = Base.get<const Expr*>();
      assert(E != nullptr && "Expecting non-null Expr");
      E->printPretty(Out, nullptr, Ctx.getPrintingPolicy());
      // FIXME: This is wrong if E is a MaterializeTemporaryExpr with an lvalue
      // adjustment.
      ElemTy = E->getType();
    }

    ArrayRef<LValuePathEntry> Path = getLValuePath();
    const CXXRecordDecl *CastToBase = nullptr;
    for (unsigned I = 0, N = Path.size(); I != N; ++I) {
      if (ElemTy->getAs<RecordType>()) {
        // The lvalue refers to a class type, so the next path entry is a base
        // or member.
        const Decl *BaseOrMember = Path[I].getAsBaseOrMember().getPointer();
        if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(BaseOrMember)) {
          CastToBase = RD;
          ElemTy = Ctx.getRecordType(RD);
        } else {
          const ValueDecl *VD = cast<ValueDecl>(BaseOrMember);
          Out << ".";
          if (CastToBase)
            Out << *CastToBase << "::";
          Out << *VD;
          ElemTy = VD->getType();
        }
      } else {
        // The lvalue must refer to an array.
        Out << '[' << Path[I].getAsArrayIndex() << ']';
        ElemTy = Ctx.getAsArrayType(ElemTy)->getElementType();
      }
    }

    // Handle formatting of one-past-the-end lvalues.
    if (isLValueOnePastTheEnd()) {
      // FIXME: If CastToBase is non-0, we should prefix the output with
      // "(CastToBase*)".
      Out << " + 1";
      if (IsReference)
        Out << ')';
    }
    return;
  }
  case APValue::Array: {
    const ArrayType *AT = Ctx.getAsArrayType(Ty);
    QualType ElemTy = AT->getElementType();
    Out << '{';
    if (unsigned N = getArrayInitializedElts()) {
      getArrayInitializedElt(0).printPretty(Out, Ctx, ElemTy);
      for (unsigned I = 1; I != N; ++I) {
        Out << ", ";
        if (I == 10) {
          // Avoid printing out the entire contents of large arrays.
          Out << "...";
          break;
        }
        getArrayInitializedElt(I).printPretty(Out, Ctx, ElemTy);
      }
    }
    Out << '}';
    return;
  }
  case APValue::Struct: {
    Out << '{';
    const RecordDecl *RD = Ty->castAs<RecordType>()->getDecl();
    bool First = true;
    if (unsigned N = getStructNumBases()) {
      const CXXRecordDecl *CD = cast<CXXRecordDecl>(RD);
      CXXRecordDecl::base_class_const_iterator BI = CD->bases_begin();
      for (unsigned I = 0; I != N; ++I, ++BI) {
        assert(BI != CD->bases_end());
        if (!First)
          Out << ", ";
        getStructBase(I).printPretty(Out, Ctx, BI->getType());
        First = false;
      }
    }
    for (const auto *FI : RD->fields()) {
      if (!First)
        Out << ", ";
      if (FI->isUnnamedBitfield()) continue;
      getStructField(FI->getFieldIndex()).
        printPretty(Out, Ctx, FI->getType());
      First = false;
    }
    Out << '}';
    return;
  }
  case APValue::Union:
    Out << '{';
    if (const FieldDecl *FD = getUnionField()) {
      Out << "." << *FD << " = ";
      getUnionValue().printPretty(Out, Ctx, FD->getType());
    }
    Out << '}';
    return;
  case APValue::MemberPointer:
    // FIXME: This is not enough to unambiguously identify the member in a
    // multiple-inheritance scenario.
    if (const ValueDecl *VD = getMemberPointerDecl()) {
      Out << '&' << *cast<CXXRecordDecl>(VD->getDeclContext()) << "::" << *VD;
      return;
    }
    Out << "0";
    return;
  case APValue::AddrLabelDiff:
    Out << "&&" << getAddrLabelDiffLHS()->getLabel()->getName();
    Out << " - ";
    Out << "&&" << getAddrLabelDiffRHS()->getLabel()->getName();
    return;
  }
  llvm_unreachable("Unknown APValue kind!");
}

std::string APValue::getAsString(const ASTContext &Ctx, QualType Ty) const {
  std::string Result;
  llvm::raw_string_ostream Out(Result);
  printPretty(Out, Ctx, Ty);
  Out.flush();
  return Result;
}

bool APValue::toIntegralConstant(APSInt &Result, QualType SrcTy,
                                 const ASTContext &Ctx) const {
  if (isInt()) {
    Result = getInt();
    return true;
  }

  if (isLValue() && isNullPointer()) {
    Result = Ctx.MakeIntValue(Ctx.getTargetNullPointerValue(SrcTy), SrcTy);
    return true;
  }

  if (isLValue() && !getLValueBase()) {
    Result = Ctx.MakeIntValue(getLValueOffset().getQuantity(), SrcTy);
    return true;
  }

  return false;
}

const APValue::LValueBase APValue::getLValueBase() const {
  assert(isLValue() && "Invalid accessor");
  return ((const LV*)(const void*)Data.buffer)->Base;
}

bool APValue::isLValueOnePastTheEnd() const {
  assert(isLValue() && "Invalid accessor");
  return ((const LV*)(const void*)Data.buffer)->IsOnePastTheEnd;
}

CharUnits &APValue::getLValueOffset() {
  assert(isLValue() && "Invalid accessor");
  return ((LV*)(void*)Data.buffer)->Offset;
}

bool APValue::hasLValuePath() const {
  assert(isLValue() && "Invalid accessor");
  return ((const LV*)(const char*)Data.buffer)->hasPath();
}

ArrayRef<APValue::LValuePathEntry> APValue::getLValuePath() const {
  assert(isLValue() && hasLValuePath() && "Invalid accessor");
  const LV &LVal = *((const LV*)(const char*)Data.buffer);
  return llvm::makeArrayRef(LVal.getPath(), LVal.PathLength);
}

unsigned APValue::getLValueCallIndex() const {
  assert(isLValue() && "Invalid accessor");
  return ((const LV*)(const char*)Data.buffer)->Base.getCallIndex();
}

unsigned APValue::getLValueVersion() const {
  assert(isLValue() && "Invalid accessor");
  return ((const LV*)(const char*)Data.buffer)->Base.getVersion();
}

bool APValue::isNullPointer() const {
  assert(isLValue() && "Invalid usage");
  return ((const LV*)(const char*)Data.buffer)->IsNullPtr;
}

void APValue::setLValue(LValueBase B, const CharUnits &O, NoLValuePath,
                        bool IsNullPtr) {
  assert(isLValue() && "Invalid accessor");
  LV &LVal = *((LV*)(char*)Data.buffer);
  LVal.Base = B;
  LVal.IsOnePastTheEnd = false;
  LVal.Offset = O;
  LVal.resizePath((unsigned)-1);
  LVal.IsNullPtr = IsNullPtr;
}

void APValue::setLValue(LValueBase B, const CharUnits &O,
                        ArrayRef<LValuePathEntry> Path, bool IsOnePastTheEnd,
                        bool IsNullPtr) {
  assert(isLValue() && "Invalid accessor");
  LV &LVal = *((LV*)(char*)Data.buffer);
  LVal.Base = B;
  LVal.IsOnePastTheEnd = IsOnePastTheEnd;
  LVal.Offset = O;
  LVal.resizePath(Path.size());
  memcpy(LVal.getPath(), Path.data(), Path.size() * sizeof(LValuePathEntry));
  LVal.IsNullPtr = IsNullPtr;
}

const ValueDecl *APValue::getMemberPointerDecl() const {
  assert(isMemberPointer() && "Invalid accessor");
  const MemberPointerData &MPD =
      *((const MemberPointerData *)(const char *)Data.buffer);
  return MPD.MemberAndIsDerivedMember.getPointer();
}

bool APValue::isMemberPointerToDerivedMember() const {
  assert(isMemberPointer() && "Invalid accessor");
  const MemberPointerData &MPD =
      *((const MemberPointerData *)(const char *)Data.buffer);
  return MPD.MemberAndIsDerivedMember.getInt();
}

ArrayRef<const CXXRecordDecl*> APValue::getMemberPointerPath() const {
  assert(isMemberPointer() && "Invalid accessor");
  const MemberPointerData &MPD =
      *((const MemberPointerData *)(const char *)Data.buffer);
  return llvm::makeArrayRef(MPD.getPath(), MPD.PathLength);
}

void APValue::MakeLValue() {
  assert(isAbsent() && "Bad state change");
  static_assert(sizeof(LV) <= DataSize, "LV too big");
  new ((void*)(char*)Data.buffer) LV();
  Kind = LValue;
}

void APValue::MakeArray(unsigned InitElts, unsigned Size) {
  assert(isAbsent() && "Bad state change");
  new ((void*)(char*)Data.buffer) Arr(InitElts, Size);
  Kind = Array;
}

void APValue::MakeMemberPointer(const ValueDecl *Member, bool IsDerivedMember,
                                ArrayRef<const CXXRecordDecl*> Path) {
  assert(isAbsent() && "Bad state change");
  MemberPointerData *MPD = new ((void*)(char*)Data.buffer) MemberPointerData;
  Kind = MemberPointer;
  MPD->MemberAndIsDerivedMember.setPointer(Member);
  MPD->MemberAndIsDerivedMember.setInt(IsDerivedMember);
  MPD->resizePath(Path.size());
  memcpy(MPD->getPath(), Path.data(), Path.size()*sizeof(const CXXRecordDecl*));
}
