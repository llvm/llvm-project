//===-- CodeGen.cpp -- bridge to lower to LLVM ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "DescriptorModel.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

/// The Tilikum bridge performs the conversion of operations from both the FIR
/// and standard dialects to the LLVM-IR dialect.
///
/// Some FIR operations may be lowered to other dialects, such as standard, but
/// some FIR operations will pass through to the Tilikum bridge.  This may be
/// necessary to preserve the semantics of the Fortran program.

#undef TODO
#define TODO() llvm_unreachable("not yet implemented")

using namespace llvm;

using OperandTy = ArrayRef<mlir::Value>;

static cl::opt<bool>
    disableFirToLLVMIR("disable-fir2llvmir",
                       cl::desc("disable FIR to LLVM-IR dialect pass"),
                       cl::init(false), cl::Hidden);

static cl::opt<bool> disableLLVM("disable-llvm", cl::desc("disable LLVM pass"),
                                 cl::init(false), cl::Hidden);

namespace fir {
/// return true if all `Value`s in `operands` are `ConstantOp`s
bool allConstants(OperandTy operands) {
  for (auto opnd : operands) {
    if (auto defop = opnd.getDefiningOp())
      if (isa<mlir::LLVM::ConstantOp>(defop) || isa<mlir::ConstantOp>(defop))
        continue;
    return false;
  }
  return true;
}
} // namespace fir

using SmallVecResult = SmallVector<mlir::Value, 4>;
using AttributeTy = ArrayRef<mlir::NamedAttribute>;

static constexpr unsigned defaultAlign = 8;

namespace {

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class FIRToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  FIRToLLVMTypeConverter(mlir::MLIRContext *context, fir::NameUniquer &uniquer)
      : LLVMTypeConverter(context), kindMapping(context), uniquer(uniquer) {
    addConversion([&](fir::BoxType box) { return convertBoxType(box); });
    addConversion(
        [&](fir::BoxCharType boxchar) { return convertBoxCharType(boxchar); });
    addConversion(
        [&](fir::BoxProcType boxproc) { return convertBoxProcType(boxproc); });
    addConversion(
        [&](fir::CharacterType charTy) { return convertCharType(charTy); });
    addConversion([&](fir::CplxType cplx) {
      return convertComplexType(cplx.getFKind());
    });
    addConversion(
        [&](fir::RecordType derived) { return convertRecordType(derived); });
    addConversion([&](fir::DimsType dims) {
      return mlir::LLVM::LLVMType::getArrayTy(dimsType(), dims.getRank());
    });
    addConversion([&](fir::FieldType field) {
      return mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
    });
    addConversion([&](fir::HeapType heap) { return convertPointerLike(heap); });
    addConversion([&](fir::IntType intr) { return convertIntegerType(intr); });
    addConversion([&](fir::LenType field) {
      return mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
    });
    addConversion(
        [&](fir::LogicalType logical) { return convertLogicalType(logical); });
    addConversion(
        [&](fir::PointerType pointer) { return convertPointerLike(pointer); });
    addConversion(
        [&](fir::RealType real) { return convertRealType(real.getFKind()); });
    addConversion(
        [&](fir::ReferenceType ref) { return convertPointerLike(ref); });
    addConversion([&](fir::SequenceType sequence) {
      return convertSequenceType(sequence);
    });
    addConversion([&](fir::TypeDescType tdesc) {
      return convertTypeDescType(tdesc.getContext());
    });
    addConversion(
        [&](mlir::TupleType tuple) { return convertTupleType(tuple); });
    addConversion(
        [&](mlir::ComplexType cmplx) { return convertComplexType(cmplx); });
    addConversion([&](mlir::NoneType none) {
      return mlir::LLVM::LLVMType::getStructTy(llvmDialect, {});
    });
  }

  // This returns the type of a single column. Rows are added by the caller.
  // fir.dims<r>  -->  llvm<"[r x [3 x i64]]">
  mlir::LLVM::LLVMType dimsType() {
    auto i64Ty{mlir::LLVM::LLVMType::getInt64Ty(llvmDialect)};
    return mlir::LLVM::LLVMType::getArrayTy(i64Ty, 3);
  }

  // i32 is used here because LLVM wants i32 constants when indexing into struct
  // types. Indexing into other aggregate types is more flexible.
  mlir::LLVM::LLVMType offsetType() {
    return mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  }

  // i64 can be used to index into aggregates like arrays
  mlir::LLVM::LLVMType indexType() {
    return mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
  }

  // TODO
  bool requiresExtendedDesc() { return false; }

  // This corresponds to the descriptor as defined ISO_Fortran_binding.h and the
  // addendum defined in descriptor.h.
  mlir::LLVM::LLVMType convertBoxType(fir::BoxType box, int rank = -1) {
    // (buffer*, ele-size, rank, type-descriptor, attribute, [dims])
    SmallVector<mlir::LLVM::LLVMType, 6> parts;
    mlir::Type ele = box.getEleTy();
    auto eleTy = unwrap(convertType(ele));
    // buffer*
    if (ele.isa<fir::SequenceType>() && eleTy.isPointerTy())
      parts.push_back(eleTy);
    else
      parts.push_back(eleTy.getPointerTo());
    parts.push_back(fir::getDescFieldTypeModel<1>()(llvmDialect));
    parts.push_back(fir::getDescFieldTypeModel<2>()(llvmDialect));
    parts.push_back(fir::getDescFieldTypeModel<3>()(llvmDialect));
    parts.push_back(fir::getDescFieldTypeModel<4>()(llvmDialect));
    parts.push_back(fir::getDescFieldTypeModel<5>()(llvmDialect));
    parts.push_back(fir::getDescFieldTypeModel<6>()(llvmDialect));
    if (rank > 0) {
      auto rowTy = fir::getDescFieldTypeModel<7>()(llvmDialect);
      parts.push_back(mlir::LLVM::LLVMType::getArrayTy(rowTy, rank));
    }
    // opt-type-ptr: i8* (see fir.tdesc)
    if (requiresExtendedDesc()) {
      parts.push_back(fir::getExtendedDescFieldTypeModel<8>()(llvmDialect));
      parts.push_back(fir::getExtendedDescFieldTypeModel<9>()(llvmDialect));
      auto rowTy = fir::getExtendedDescFieldTypeModel<10>()(llvmDialect);
      unsigned numLenParams = 0; // FIXME
      parts.push_back(mlir::LLVM::LLVMType::getArrayTy(rowTy, numLenParams));
    }
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, parts).getPointerTo();
  }

  // fir.boxchar<n>  -->  llvm<"{ ix*, i64 }">   where ix is kind mapping
  mlir::LLVM::LLVMType convertBoxCharType(fir::BoxCharType boxchar) {
    auto ptrTy = convertCharType(boxchar.getEleTy()).getPointerTo();
    auto i64Ty = mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{ptrTy, i64Ty};
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  // fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
  mlir::LLVM::LLVMType convertBoxProcType(fir::BoxProcType boxproc) {
    auto funcTy = convertType(boxproc.getEleTy());
    auto ptrTy = unwrap(funcTy).getPointerTo();
    auto i8Ty = mlir::LLVM::LLVMType::getInt8Ty(llvmDialect);
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{ptrTy, i8Ty};
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  unsigned characterBitsize(fir::CharacterType charTy) {
    return kindMapping.getCharacterBitsize(charTy.getFKind());
  }

  // fir.char<n>  -->  llvm<"ix*">   where ix is scaled by kind mapping
  mlir::LLVM::LLVMType convertCharType(fir::CharacterType charTy) {
    return mlir::LLVM::LLVMType::getIntNTy(llvmDialect,
                                           characterBitsize(charTy));
  }

  mlir::LLVM::LLVMType convertComplexPartType(fir::KindTy kind) {
    auto realID = kindMapping.getComplexTypeID(kind);
    return fromRealTypeID(realID, kind);
  }

  // fir.complex<n>  -->  llvm<"{ anyfloat, anyfloat }">
  mlir::LLVM::LLVMType convertComplexType(fir::KindTy kind) {
    auto realTy = convertComplexPartType(kind);
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{realTy, realTy};
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  mlir::LLVM::LLVMType getDefaultInt() {
    // FIXME: this should be tied to the front-end default
    return mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
  }

  // fir.int<n>  -->  llvm.ix   where ix is a kind mapping
  mlir::LLVM::LLVMType convertIntegerType(fir::IntType intTy) {
    return mlir::LLVM::LLVMType::getIntNTy(
        llvmDialect, kindMapping.getIntegerBitsize(intTy.getFKind()));
  }

  // fir.logical<n>  -->  llvm.ix  where ix is a kind mapping
  mlir::LLVM::LLVMType convertLogicalType(fir::LogicalType boolTy) {
    return mlir::LLVM::LLVMType::getIntNTy(
        llvmDialect, kindMapping.getLogicalBitsize(boolTy.getFKind()));
  }

  template <typename A>
  mlir::LLVM::LLVMType convertPointerLike(A &ty) {
    mlir::Type eleTy = ty.getEleTy();
    // A sequence type is a special case. A sequence of runtime size on its
    // interior dimensions lowers to a memory reference. In that case, we
    // degenerate the array and do not want a the type to become `T**` but
    // merely `T*`.
    if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (!seqTy.hasConstantShape()) {
        if (seqTy.hasConstantInterior())
          return unwrap(convertType(seqTy));
        eleTy = seqTy.getEleTy();
      }
    }
    return unwrap(convertType(eleTy)).getPointerTo();
  }

  // convert a front-end kind value to either a std or LLVM IR dialect type
  // fir.real<n>  -->  llvm.anyfloat  where anyfloat is a kind mapping
  mlir::LLVM::LLVMType convertRealType(fir::KindTy kind) {
    return fromRealTypeID(kindMapping.getRealTypeID(kind), kind);
  }

  // fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
  mlir::LLVM::LLVMType convertRecordType(fir::RecordType derived) {
    auto name{derived.getName()};
    // The cache is needed to keep a unique mapping from name -> StructType
    auto iter{identStructCache.find(name)};
    if (iter != identStructCache.end())
      return iter->second;
    auto st{mlir::LLVM::LLVMType::createStructTy(llvmDialect, name)};
    identStructCache[name] = st;
    SmallVector<mlir::LLVM::LLVMType, 8> members;
    for (auto mem : derived.getTypeList())
      members.push_back(convertType(mem.second).cast<mlir::LLVM::LLVMType>());
    mlir::LLVM::LLVMType::setStructTyBody(st, members);
    return st;
  }

  // fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
  mlir::LLVM::LLVMType convertSequenceType(fir::SequenceType seq) {
    assert(seq.hasConstantInterior() && "cannot lower type to LLVM IR");
    auto baseTy = unwrap(convertType(seq.getEleTy()));
    auto shape = seq.getShape();
    auto constRows = seq.getConstantRows();
    if (constRows) {
      decltype(constRows) i = constRows;
      for (auto e : shape) {
        baseTy = mlir::LLVM::LLVMType::getArrayTy(baseTy, e);
        if (--i == 0)
          break;
      }
      if (seq.hasConstantShape())
        return baseTy;
    }
    return baseTy.getPointerTo();
  }

  // tuple<TS...>  -->  llvm<"{ ts... }">
  mlir::LLVM::LLVMType convertTupleType(mlir::TupleType tuple) {
    SmallVector<mlir::Type, 8> inMembers;
    tuple.getFlattenedTypes(inMembers);
    SmallVector<mlir::LLVM::LLVMType, 8> members;
    for (auto mem : inMembers)
      members.push_back(convertType(mem).cast<mlir::LLVM::LLVMType>());
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, members);
  }

  // complex<T>  --> llvm<"{t,t}">
  mlir::LLVM::LLVMType convertComplexType(mlir::ComplexType complex) {
    auto eleTy = unwrap(convertType(complex.getElementType()));
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{eleTy, eleTy};
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  // fir.tdesc<any>  -->  llvm<"i8*">
  // FIXME: for now use a void*, however pointer identity is not sufficient for
  // the f18 object v. class distinction
  mlir::LLVM::LLVMType convertTypeDescType(mlir::MLIRContext *ctx) {
    return mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  }

  /// Convert llvm::Type::TypeID to mlir::LLVM::LLVMType
  mlir::LLVM::LLVMType fromRealTypeID(llvm::Type::TypeID typeID,
                                      fir::KindTy kind) {
    switch (typeID) {
    case llvm::Type::TypeID::HalfTyID:
      return mlir::LLVM::LLVMType::getHalfTy(llvmDialect);
    case llvm::Type::TypeID::FloatTyID:
      return mlir::LLVM::LLVMType::getFloatTy(llvmDialect);
    case llvm::Type::TypeID::DoubleTyID:
      return mlir::LLVM::LLVMType::getDoubleTy(llvmDialect);
    case llvm::Type::TypeID::X86_FP80TyID:
      return mlir::LLVM::LLVMType::getX86_FP80Ty(llvmDialect);
    case llvm::Type::TypeID::FP128TyID:
      return mlir::LLVM::LLVMType::getFP128Ty(llvmDialect);
    default:
      emitError(UnknownLoc::get(llvmDialect->getContext()))
          << "unsupported type: !fir.real<" << kind << ">";
      return {};
    }
  }

  /// HACK: cloned from LLVMTypeConverter since this is private there
  mlir::LLVM::LLVMType unwrap(mlir::Type type) {
    if (!type)
      return nullptr;
    auto *mlirContext = type.getContext();
    auto wrappedLLVMType = type.dyn_cast<mlir::LLVM::LLVMType>();
    if (!wrappedLLVMType)
      emitError(UnknownLoc::get(mlirContext),
                "conversion resulted in a non-LLVM type");
    return wrappedLLVMType;
  }

  /// Returns false iff the sequence type has a shape and the shape is constant.
  static bool unknownShape(fir::SequenceType::Shape shape) {
    // does the shape even exist?
    auto size = shape.size();
    if (size == 0)
      return true;
    // if it exists, are any dimensions deferred?
    for (decltype(size) i = 0, sz = size; i < sz; ++i)
      if (shape[i] == fir::SequenceType::getUnknownExtent())
        return true;
    return false;
  }

  /// Does this record type have dynamically inlined subobjects? Note: this
  /// should not look through references as they are not inlined.
  static bool dynamicallySized(fir::RecordType seqTy) {
    for (auto field : seqTy.getTypeList()) {
      if (auto arr = field.second.dyn_cast<fir::SequenceType>()) {
        if (unknownShape(arr.getShape()))
          return true;
      } else if (auto rec = field.second.dyn_cast<fir::RecordType>()) {
        if (dynamicallySized(rec))
          return true;
      }
    }
    return false;
  }

  static bool dynamicallySized(mlir::Type ty) {
    if (auto arr = ty.dyn_cast<fir::SequenceType>())
      ty = arr.getEleTy();
    if (auto rec = ty.dyn_cast<fir::RecordType>())
      return dynamicallySized(rec);
    return false;
  }

  fir::NameUniquer &getUniquer() { return uniquer; }

  fir::KindMapping &getKindMap() { return kindMapping; }

private:
  fir::KindMapping kindMapping;
  fir::NameUniquer &uniquer;
  static StringMap<mlir::LLVM::LLVMType> identStructCache;
};

// instantiate static data member
StringMap<mlir::LLVM::LLVMType> FIRToLLVMTypeConverter::identStructCache;
} // namespace

/// remove `omitNames` (by name) from the attribute dictionary
static SmallVector<mlir::NamedAttribute, 4>
pruneNamedAttrDict(AttributeTy attrs, ArrayRef<StringRef> omitNames) {
  SmallVector<mlir::NamedAttribute, 4> result;
  for (auto x : attrs) {
    bool omit = false;
    for (auto o : omitNames)
      if (x.first.strref() == o) {
        omit = true;
        break;
      }
    if (!omit)
      result.push_back(x);
  }
  return result;
}

inline mlir::LLVM::LLVMType getVoidPtrType(mlir::LLVM::LLVMDialect *dialect) {
  return mlir::LLVM::LLVMType::getInt8PtrTy(dialect);
}

namespace {
/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public mlir::OpConversionPattern<FromOp> {
public:
  explicit FIROpConversion(mlir::MLIRContext *ctx,
                           FIRToLLVMTypeConverter &lowering)
      : mlir::OpConversionPattern<FromOp>(lowering, ctx, 1) {}

protected:
  LLVMContext &getLLVMContext() const { return lowerTy().getLLVMContext(); }
  mlir::LLVM::LLVMDialect *getDialect() const { return lowerTy().getDialect(); }
  mlir::Type convertType(mlir::Type ty) const {
    return lowerTy().convertType(ty);
  }
  mlir::LLVM::LLVMType unwrap(mlir::Type ty) const {
    return lowerTy().unwrap(ty);
  }
  mlir::LLVM::LLVMType voidPtrTy() const {
    return getVoidPtrType(getDialect());
  }

  mlir::LLVM::ConstantOp
  genConstantOffset(mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    int offset) const {
    auto ity = lowerTy().offsetType();
    auto cattr = rewriter.getI32IntegerAttr(offset);
    return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
  }

  /// Method to construct code sequence to get the rank from a box.
  mlir::Value getRankFromBox(mlir::Location loc, mlir::Value box,
                             mlir::Type resultTy,
                             mlir::ConversionPatternRewriter &rewriter) const {
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c3 = genConstantOffset(loc, rewriter, 3);
    llvm::SmallVector<mlir::Value, 3> args = {box, c0, c3};
    auto pty = unwrap(resultTy).getPointerTo();
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, pty, args);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, resultTy, p);
  }

  /// Method to construct code sequence to get the triple for dimension `dim`
  /// from a box.
  llvm::SmallVector<mlir::Value, 3>
  getDimsFromBox(mlir::Location loc, llvm::ArrayRef<mlir::Type> retTys,
                 mlir::Value box, mlir::Value dim,
                 mlir::ConversionPatternRewriter &rewriter) const {
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c7 = genConstantOffset(loc, rewriter, 7);
    auto l0 = loadFromOffset(loc, box, c0, c7, dim, 0, retTys[0], rewriter);
    auto l1 = loadFromOffset(loc, box, c0, c7, dim, 1, retTys[1], rewriter);
    auto l2 = loadFromOffset(loc, box, c0, c7, dim, 2, retTys[2], rewriter);
    return {l0.getResult(), l1.getResult(), l2.getResult()};
  }

  mlir::LLVM::LoadOp
  loadFromOffset(mlir::Location loc, mlir::Value a, mlir::LLVM::ConstantOp c0,
                 mlir::LLVM::ConstantOp c7, mlir::Value dim, int off,
                 mlir::Type ty,
                 mlir::ConversionPatternRewriter &rewriter) const {
    auto pty = unwrap(ty).getPointerTo();
    auto c = genConstantOffset(loc, rewriter, off);
    auto p = genGEP(loc, pty, rewriter, a, c0, c7, dim, c);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  }

  template <typename... ARGS>
  mlir::LLVM::GEPOp genGEP(mlir::Location loc, mlir::LLVM::LLVMType ty,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Value base, ARGS... args) const {
    SmallVector<mlir::Value, 8> cv{args...};
    return rewriter.create<mlir::LLVM::GEPOp>(loc, ty, base, cv);
  }

  FIRToLLVMTypeConverter &lowerTy() const {
    return *static_cast<FIRToLLVMTypeConverter *>(this->getTypeConverter());
  }
};

/// FIR conversion pattern template
template <typename FromOp>
class FIROpAndTypeConversion : public FIROpConversion<FromOp> {
public:
  using FIROpConversion<FromOp>::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(FromOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type ty = this->convertType(op.getType());
    return doRewrite(op, ty, operands, rewriter);
  }

  virtual mlir::LogicalResult
  doRewrite(FromOp addr, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("derived class must override");
  }
};
} // namespace

static Block *createBlock(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Block *insertBefore) {
  assert(insertBefore && "expected valid insertion block");
  return rewriter.createBlock(insertBefore->getParent(),
                              mlir::Region::iterator(insertBefore));
}

/// Create an LLVM dialect global
static void createGlobal(mlir::Location loc, mlir::ModuleOp mod, StringRef name,
                         mlir::LLVM::LLVMType type,
                         mlir::ConversionPatternRewriter &rewriter) {
  if (mod.lookupSymbol<mlir::LLVM::GlobalOp>(name))
    return;
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  modBuilder.create<mlir::LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                          mlir::LLVM::Linkage::Weak, name,
                                          mlir::Attribute{});
}

namespace {
struct AddrOfOpConversion : public FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AddrOfOp addr, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = unwrap(convertType(addr.getType()));
    auto attrs = pruneNamedAttrDict(addr.getAttrs(), {"symbol"});
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(
        addr, ty, addr.symbol().getRootReference(), attrs);
    return success();
  }
};
} // namespace

static mlir::LLVM::ConstantOp
genConstantIndex(mlir::Location loc, mlir::LLVM::LLVMType ity,
                 mlir::ConversionPatternRewriter &rewriter, int offset) {
  auto cattr = rewriter.getI64IntegerAttr(offset);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
}

namespace {
/// convert to LLVM IR dialect `alloca`
struct AllocaOpConversion : public FIROpConversion<fir::AllocaOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AllocaOp alloc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = alloc.getLoc();
    auto ity = lowerTy().indexType();
    auto c1 = genConstantIndex(loc, ity, rewriter, 1);
    auto size = c1.getResult();
    for (auto opnd : operands)
      size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, opnd);
    auto ty = convertType(alloc.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(alloc, ty, size,
                                                      alloc.getAttrs());
    return success();
  }
};
} // namespace

static mlir::LLVM::LLVMFuncOp
getMalloc(fir::AllocMemOp op, mlir::ConversionPatternRewriter &rewriter,
          mlir::LLVM::LLVMDialect *dialect) {
  auto module = op.getParentOfType<mlir::ModuleOp>();
  if (auto mallocFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("malloc"))
    return mallocFunc;
  mlir::OpBuilder moduleBuilder(
      op.getParentOfType<mlir::ModuleOp>().getBodyRegion());
  auto indexType = mlir::LLVM::LLVMType::getInt64Ty(dialect);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "malloc",
      mlir::LLVM::LLVMType::getFunctionTy(getVoidPtrType(dialect), indexType,
                                          /*isVarArg=*/false));
}

namespace {
/// convert to `call` to the runtime to `malloc` memory
struct AllocMemOpConversion : public FIROpConversion<fir::AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AllocMemOp heap, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(heap.getType());
    auto dialect = getDialect();
    auto mallocFunc = getMalloc(heap, rewriter, dialect);
    auto loc = heap.getLoc();
    auto ity = lowerTy().indexType();
    auto c1 = genConstantIndex(loc, ity, rewriter, 1);
    auto size = c1.getResult();
    for (auto opnd : operands)
      size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, opnd);
    heap.setAttr("callee", rewriter.getSymbolRefAttr(mallocFunc));
    SmallVector<mlir::Value, 1> args{size};
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(heap, ty, args,
                                                    heap.getAttrs());
    return success();
  }
};
} // namespace

/// obtain the free() function
static mlir::LLVM::LLVMFuncOp getFree(fir::FreeMemOp op,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      mlir::LLVM::LLVMDialect *dialect) {
  auto module = op.getParentOfType<mlir::ModuleOp>();
  if (auto freeFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("free"))
    return freeFunc;
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto voidType = mlir::LLVM::LLVMType::getVoidTy(dialect);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "free",
      mlir::LLVM::LLVMType::getFunctionTy(voidType, getVoidPtrType(dialect),
                                          /*isVarArg=*/false));
}

namespace {
/// lower a freemem instruction into a call to free()
struct FreeMemOpConversion : public FIROpConversion<fir::FreeMemOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::FreeMemOp freemem, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dialect = getDialect();
    auto freeFunc = getFree(freemem, rewriter, dialect);
    auto bitcast = rewriter.create<mlir::LLVM::BitcastOp>(
        freemem.getLoc(), voidPtrTy(), operands[0]);
    freemem.setAttr("callee", rewriter.getSymbolRefAttr(freeFunc));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        freemem, mlir::LLVM::LLVMType::getVoidTy(dialect),
        SmallVector<mlir::Value, 1>{bitcast}, freemem.getAttrs());
    return success();
  }
};

/// convert to returning the first element of the box (any flavor)
struct BoxAddrOpConversion : public FIROpConversion<fir::BoxAddrOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxAddrOp boxaddr, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxaddr.getLoc();
    auto ty = convertType(boxaddr.getType());
    if (auto argty = boxaddr.val().getType().dyn_cast<fir::BoxType>()) {
      auto c0 = genConstantOffset(loc, rewriter, 0);
      auto pty = unwrap(ty).getPointerTo();
      auto p = genGEP(loc, unwrap(pty), rewriter, a, c0, c0);
      // load the pointer from the buffer
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(boxaddr, ty, p);
    } else {
      auto c0attr = rewriter.getI32IntegerAttr(0);
      auto c0 = mlir::ArrayAttr::get(c0attr, boxaddr.getContext());
      rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxaddr, ty, a,
                                                              c0);
    }
    return success();
  }
};

/// convert to an extractvalue for the 2nd part of the boxchar
struct BoxCharLenOpConversion : public FIROpConversion<fir::BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxCharLenOp boxchar, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto ty = convertType(boxchar.getType());
    auto ctx = boxchar.getContext();
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxchar, ty, a, c1);
    return success();
  }
};

/// convert to a triple set of GEPs and loads
struct BoxDimsOpConversion : public FIROpConversion<fir::BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxDimsOp boxdims, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 3> resultTypes = {
        convertType(boxdims.getResult(0).getType()),
        convertType(boxdims.getResult(1).getType()),
        convertType(boxdims.getResult(2).getType()),
    };
    auto results = getDimsFromBox(boxdims.getLoc(), resultTypes, operands[0],
                                  operands[1], rewriter);
    rewriter.replaceOp(boxdims, results);
    return success();
  }
};

struct BoxEleSizeOpConversion : public FIROpConversion<fir::BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxEleSizeOp boxelesz, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxelesz.getLoc();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c1 = genConstantOffset(loc, rewriter, 1);
    auto ty = convertType(boxelesz.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c1);
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(boxelesz, ty, p);
    return success();
  }
};

struct BoxIsAllocOpConversion : public FIROpConversion<fir::BoxIsAllocOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsAllocOp boxisalloc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxisalloc.getLoc();
    auto ity = lowerTy().offsetType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c5 = genConstantOffset(loc, rewriter, 5);
    auto ty = convertType(boxisalloc.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c5);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto ab = genConstantOffset(loc, rewriter, 2);
    auto bit = rewriter.create<mlir::LLVM::AndOp>(loc, ity, ld, ab);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisalloc, mlir::LLVM::ICmpPredicate::ne, bit, c0);
    return success();
  }
};

struct BoxIsArrayOpConversion : public FIROpConversion<fir::BoxIsArrayOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsArrayOp boxisarray, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxisarray.getLoc();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c3 = genConstantOffset(loc, rewriter, 3);
    auto ty = convertType(boxisarray.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c3);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisarray, mlir::LLVM::ICmpPredicate::ne, ld, c0);
    return success();
  }
};

struct BoxIsPtrOpConversion : public FIROpConversion<fir::BoxIsPtrOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsPtrOp boxisptr, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxisptr.getLoc();
    auto ty = convertType(boxisptr.getType());
    auto ity = lowerTy().offsetType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c5 = genConstantOffset(loc, rewriter, 5);
    SmallVector<mlir::Value, 4> args{a, c0, c5};
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, ty, args);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto ab = genConstantOffset(loc, rewriter, 1);
    auto bit = rewriter.create<mlir::LLVM::AndOp>(loc, ity, ld, ab);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisptr, mlir::LLVM::ICmpPredicate::ne, bit, c0);
    return success();
  }
};

struct BoxProcHostOpConversion : public FIROpConversion<fir::BoxProcHostOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxProcHostOp boxprochost, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto ty = convertType(boxprochost.getType());
    auto ctx = boxprochost.getContext();
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxprochost, ty, a,
                                                            c1);
    return success();
  }
};

struct BoxRankOpConversion : public FIROpConversion<fir::BoxRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxRankOp boxrank, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxrank.getLoc();
    auto ty = convertType(boxrank.getType());
    auto result = getRankFromBox(loc, a, ty, rewriter);
    rewriter.replaceOp(boxrank, result);
    return success();
  }
};

struct BoxTypeDescOpConversion : public FIROpConversion<fir::BoxTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxTypeDescOp boxtypedesc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto loc = boxtypedesc.getLoc();
    auto ty = convertType(boxtypedesc.getType());
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c4 = genConstantOffset(loc, rewriter, 4);
    SmallVector<mlir::Value, 4> args{a, c0, c4};
    auto pty = unwrap(ty).getPointerTo();
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, pty, args);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto i8ptr = mlir::LLVM::LLVMType::getInt8PtrTy(getDialect());
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(boxtypedesc, i8ptr, ld);
    return success();
  }
};

struct StringLitOpConversion : public FIROpConversion<fir::StringLitOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::StringLitOp constop, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(constop.getType());
    auto attr = constop.getValue();
    if (attr.isa<mlir::StringAttr>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(constop, ty, attr);
    } else {
      // convert the array attr to a dense elements attr
      // LLVMIR dialect knows how to lower the latter to LLVM IR
      auto arr = attr.cast<mlir::ArrayAttr>();
      auto size = constop.getSize().cast<mlir::IntegerAttr>().getInt();
      auto eleTy = constop.getType().cast<fir::SequenceType>().getEleTy();
      auto bits = lowerTy().characterBitsize(eleTy.cast<fir::CharacterType>());
      auto charTy = rewriter.getIntegerType(bits);
      auto det = mlir::VectorType::get({size}, charTy);
      // convert each character to a precise bitsize
      llvm::SmallVector<mlir::Attribute, 64> vec;
      for (auto a : arr.getValue())
        vec.push_back(mlir::IntegerAttr::get(
            charTy, a.cast<mlir::IntegerAttr>().getValue().sextOrTrunc(bits)));
      auto dea = mlir::DenseElementsAttr::get(det, vec);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(constop, ty, dea);
    }
    return success();
  }
};

/// direct call LLVM function
struct CallOpConversion : public FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::CallOp call, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type, 4> resultTys;
    for (auto r : call.getResults())
      resultTys.push_back(convertType(r.getType()));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(call, resultTys, operands,
                                                    call.getAttrs());
    return success();
  }
};

/// Compare complex values
///
/// Per 10.1, the only comparisons available are .EQ. (oeq) and .NE. (une).
///
/// For completeness, all other comparison are done on the real component only.
struct CmpcOpConversion : public FIROpConversion<fir::CmpcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::CmpcOp cmp, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ctxt = cmp.getContext();
    auto kind = cmp.lhs().getType().cast<fir::CplxType>().getFKind();
    auto ty = convertType(fir::RealType::get(ctxt, kind));
    auto loc = cmp.getLoc();
    auto pos0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctxt);
    SmallVector<mlir::Value, 2> rp{
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[0], pos0),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[1],
                                                    pos0)};
    auto rcp = rewriter.create<mlir::LLVM::FCmpOp>(loc, ty, rp, cmp.getAttrs());
    auto pos1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctxt);
    SmallVector<mlir::Value, 2> ip{
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[0], pos1),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[1],
                                                    pos1)};
    auto icp = rewriter.create<mlir::LLVM::FCmpOp>(loc, ty, ip, cmp.getAttrs());
    SmallVector<mlir::Value, 2> cp{rcp, icp};
    switch (cmp.getPredicate()) {
    case mlir::CmpFPredicate::OEQ: // .EQ.
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(cmp, ty, cp);
      break;
    case mlir::CmpFPredicate::UNE: // .NE.
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(cmp, ty, cp);
      break;
    default:
      rewriter.replaceOp(cmp, rcp.getResult());
      break;
    }
    return success();
  }
};

struct CmpfOpConversion : public FIROpConversion<fir::CmpfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::CmpfOp cmp, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = convertType(cmp.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(cmp, type, operands,
                                                    cmp.getAttrs());
    return success();
  }
};

struct ConstcOpConversion : public FIROpConversion<fir::ConstcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ConstcOp conc, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = conc.getLoc();
    auto ctx = conc.getContext();
    auto ty = convertType(conc.getType());
    auto ct = conc.getType().cast<fir::CplxType>();
    auto ety = lowerTy().convertComplexPartType(ct.getFKind());
    auto ri = mlir::FloatAttr::get(ety, getValue(conc.getReal()));
    auto rp = rewriter.create<mlir::LLVM::ConstantOp>(loc, ety, ri);
    auto ii = mlir::FloatAttr::get(ety, getValue(conc.getImaginary()));
    auto ip = rewriter.create<mlir::LLVM::ConstantOp>(loc, ety, ii);
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto r = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto rr = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r, rp, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(conc, ty, rr, ip,
                                                           c1);
    return success();
  }

  inline llvm::APFloat getValue(mlir::Attribute attr) const {
    return attr.cast<fir::RealAttr>().getValue();
  }
};

struct ConstfOpConversion : public FIROpConversion<fir::ConstfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ConstfOp conf, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(conf.getType());
    auto val = conf.constantAttr();
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(conf, ty, val);
    return success();
  }
};

/// convert value of from-type to value of to-type
struct ConvertOpConversion : public FIROpConversion<fir::ConvertOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ConvertOp convert, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto fromTy_ = convertType(convert.value().getType());
    auto fromTy = unwrap(fromTy_);
    auto toTy_ = convertType(convert.res().getType());
    auto toTy = unwrap(toTy_);
    auto *fromLLVMTy = fromTy.getUnderlyingType();
    auto *toLLVMTy = toTy.getUnderlyingType();
    auto &op0 = operands[0];
    if (fromLLVMTy == toLLVMTy) {
      rewriter.replaceOp(convert, op0);
      return success();
    }
    auto loc = convert.getLoc();
    auto convertFpToFp = [&](mlir::Value val, unsigned fromBits,
                             unsigned toBits, mlir::Type toTy) -> mlir::Value {
      // FIXME: what if different reps (F16, BF16) are the same size?
      assert(fromBits != toBits);
      if (fromBits > toBits)
        return rewriter.create<mlir::LLVM::FPTruncOp>(loc, toTy, val);
      return rewriter.create<mlir::LLVM::FPExtOp>(loc, toTy, val);
    };
    if (fir::isa_complex(convert.value().getType()) &&
        fir::isa_complex(convert.res().getType())) {
      // Special case: handle the conversion of a complex such that both the
      // real and imaginary parts are converted together.
      auto zero = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0),
                                       convert.getContext());
      auto one = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1),
                                      convert.getContext());
      auto rp =
          rewriter.create<mlir::LLVM::ExtractValueOp>(loc, fromTy_, op0, zero);
      auto ip =
          rewriter.create<mlir::LLVM::ExtractValueOp>(loc, fromTy_, op0, one);
      auto ty = convertType(getComplexEleTy(convert.value().getType()));
      auto nt = convertType(getComplexEleTy(convert.res().getType()));
      auto fromBits = unwrap(ty).getUnderlyingType()->getPrimitiveSizeInBits();
      auto toBits = unwrap(nt).getUnderlyingType()->getPrimitiveSizeInBits();
      auto rc = convertFpToFp(rp, fromBits, toBits, nt);
      auto ic = convertFpToFp(ip, fromBits, toBits, nt);
      auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, toTy_);
      auto i1 =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, toTy_, un, rc, zero);
      rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(convert, toTy_, i1,
                                                             ic, one);
      return mlir::success();
    }
    if (fromLLVMTy->isFloatingPointTy()) {
      if (toLLVMTy->isFloatingPointTy()) {
        auto fromBits = fromLLVMTy->getPrimitiveSizeInBits();
        auto toBits = toLLVMTy->getPrimitiveSizeInBits();
        auto v = convertFpToFp(op0, fromBits, toBits, toTy);
        rewriter.replaceOp(convert, v);
        return mlir::success();
      }
      if (toLLVMTy->isIntegerTy()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(convert, toTy, op0);
        return mlir::success();
      }
    } else if (fromLLVMTy->isIntegerTy()) {
      if (toLLVMTy->isIntegerTy()) {
        std::size_t fromBits{fromLLVMTy->getIntegerBitWidth()};
        std::size_t toBits{toLLVMTy->getIntegerBitWidth()};
        assert(fromBits != toBits);
        if (fromBits > toBits) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(convert, toTy, op0);
          return mlir::success();
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(convert, toTy, op0);
        return mlir::success();
      }
      if (toLLVMTy->isFloatingPointTy()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(convert, toTy, op0);
        return mlir::success();
      }
      if (toLLVMTy->isPointerTy()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(convert, toTy, op0);
        return mlir::success();
      }
    } else if (fromLLVMTy->isPointerTy()) {
      if (toLLVMTy->isIntegerTy()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(convert, toTy, op0);
        return mlir::success();
      }
      if (toLLVMTy->isPointerTy()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(convert, toTy, op0);
        return mlir::success();
      }
    }
    return emitError(loc) << "cannot convert " << fromTy_ << " to " << toTy_;
  }

  static mlir::Type getComplexEleTy(mlir::Type complex) {
    if (auto cc = complex.dyn_cast<mlir::ComplexType>())
      return cc.getElementType();
    return complex.cast<fir::CplxType>().getElementType();
  }
};

/// virtual call to a method in a dispatch table
struct DispatchOpConversion : public FIROpConversion<fir::DispatchOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DispatchOp dispatch, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(dispatch.getFunctionType());
    // get the table, lookup the method, fetch the func-ptr
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(dispatch, ty, operands,
                                                    None);
    TODO();
    return success();
  }
};

/// dispatch table for a Fortran derived type
struct DispatchTableOpConversion
    : public FIROpConversion<fir::DispatchTableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DispatchTableOp dispTab, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO();
    return success();
  }
};

/// entry in a dispatch table; binds a method-name to a function
struct DTEntryOpConversion : public FIROpConversion<fir::DTEntryOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DTEntryOp dtEnt, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO();
    return success();
  }
};

/// create a CHARACTER box
struct EmboxCharOpConversion : public FIROpConversion<fir::EmboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxCharOp emboxchar, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto a = operands[0];
    auto b = operands[1];
    auto loc = emboxchar.getLoc();
    auto ctx = emboxchar.getContext();
    auto ty = convertType(emboxchar.getType());
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, un, a, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(emboxchar, ty, r, b,
                                                           c1);
    return success();
  }
};

// Common base class for lowering of embox to descriptor creation.
template <typename OP>
struct EmboxCommonConversion : public FIROpConversion<OP> {
  using FIROpConversion<OP>::FIROpConversion;

  /// Generate an alloca of size `size` and cast it to type `toTy`
  mlir::LLVM::AllocaOp
  genAllocaWithType(mlir::Location loc, mlir::LLVM::LLVMType toTy,
                    unsigned alignment, mlir::LLVM::LLVMDialect *dialect,
                    mlir::ConversionPatternRewriter &rewriter) const {
    auto thisPt = rewriter.saveInsertionPoint();
    auto *thisBlock = rewriter.getInsertionBlock();
    auto func = mlir::cast<mlir::LLVM::LLVMFuncOp>(thisBlock->getParentOp());
    rewriter.setInsertionPointToStart(&func.front());
    auto sz = this->genConstantOffset(loc, rewriter, 1);
    auto al = rewriter.create<mlir::LLVM::AllocaOp>(loc, toTy, sz, alignment);
    rewriter.restoreInsertionPoint(thisPt);
    return al;
  }

  template <typename... FLDS>
  mlir::LLVM::GEPOp genGEPToField(mlir::Location loc, mlir::LLVM::LLVMType ty,
                                  mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Value base, mlir::Value zero,
                                  FLDS... fields) const {
    return this->genGEP(loc, ty.getPointerTo(), rewriter, base, zero,
                        this->genConstantOffset(loc, rewriter, fields)...);
  }

  static mlir::LLVM::LLVMType getBoxEleTy(mlir::LLVM::LLVMType boxPtrTy,
                                          unsigned i) {
    return boxPtrTy.getPointerElementTy().getStructElementType(i);
  }

  // Perform an extension or truncation as needed on an integer value
  mlir::Value integerCast(mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter,
                          mlir::LLVM::LLVMType ty, mlir::Value val) const {
    auto toSize = ty.getUnderlyingType()->getPrimitiveSizeInBits();
    auto fromSize = val.getType()
                        .cast<mlir::LLVM::LLVMType>()
                        .getUnderlyingType()
                        ->getPrimitiveSizeInBits();
    if (toSize < fromSize)
      return rewriter.create<mlir::LLVM::TruncOp>(loc, ty, val);
    if (toSize > fromSize)
      return rewriter.create<mlir::LLVM::SExtOp>(loc, ty, val);
    return val;
  }

  // Get the element size and CFI type code of the boxed value.
  std::tuple<mlir::Value, mlir::Value>
  getSizeAndTypeCode(mlir::Location loc,
                     mlir::ConversionPatternRewriter &rewriter,
                     mlir::Type boxEleTy) const {
    auto doInteger =
        [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode;
      switch (width) {
      case 8:
        typeCode = CFI_type_int8_t;
        break;
      case 16:
        typeCode = CFI_type_int16_t;
        break;
      case 32:
        typeCode = CFI_type_int32_t;
        break;
      case 64:
        typeCode = CFI_type_int64_t;
        break;
      case 128:
        typeCode = CFI_type_int128_t;
        break;
      default:
        llvm_unreachable("unsupported integer size");
      }
      return {this->genConstantOffset(loc, rewriter, width / 8),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doFloat = [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode;
      switch (width) {
      case 32:
        typeCode = CFI_type_float;
        break;
      case 64:
        typeCode = CFI_type_double;
        break;
      case 80:
      case 128:
        typeCode = CFI_type_long_double;
        break;
      default:
        llvm_unreachable("unsupported real size");
      }
      return {this->genConstantOffset(loc, rewriter, width / 8),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doComplex =
        [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode;
      switch (width) {
      case 32:
        typeCode = CFI_type_float_Complex;
        break;
      case 64:
        typeCode = CFI_type_double_Complex;
        break;
      case 80:
      case 128:
        typeCode = CFI_type_long_double_Complex;
        break;
      default:
        llvm_unreachable("unsupported complex size");
      }
      return {this->genConstantOffset(loc, rewriter, width / 4),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto getKindMap = [&]() -> fir::KindMapping & {
      return this->lowerTy().getKindMap();
    };

    if (fir::isa_integer(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::IntegerType>())
        return doInteger(ty.getWidth());
      auto ty = boxEleTy.cast<fir::IntType>();
      return doInteger(getKindMap().getIntegerBitsize(ty.getFKind()));
    }
    if (fir::isa_real(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::FloatType>())
        return doFloat(ty.getWidth());
      auto ty = boxEleTy.cast<fir::RealType>();
      return doFloat(getKindMap().getRealBitsize(ty.getFKind()));
    }
    if (fir::isa_complex(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::ComplexType>())
        return doComplex(
            ty.getElementType().cast<mlir::FloatType>().getWidth());
      auto ty = boxEleTy.cast<fir::CplxType>();
      return doComplex(getKindMap().getRealBitsize(ty.getFKind()));
    }
    if (auto ty = boxEleTy.dyn_cast<fir::CharacterType>()) {
      TODO();
    }
    if (auto ty = boxEleTy.dyn_cast<fir::LogicalType>())
      return doInteger(getKindMap().getLogicalBitsize(ty.getFKind()));
    if (auto seqTy = boxEleTy.dyn_cast<fir::SequenceType>())
      return getSizeAndTypeCode(loc, rewriter, seqTy.getEleTy());
    if (boxEleTy.isa<fir::RecordType>()) {
      TODO();
    }
    if (fir::isa_ref_type(boxEleTy)) {
      // FIXME: use the target pointer size rather than sizeof(void*)
      return {this->genConstantOffset(loc, rewriter, sizeof(void *)),
              this->genConstantOffset(loc, rewriter, CFI_type_cptr)};
    }
    // fail: unhandled case
    TODO();
  }
};

/// Create a generic box on a memory reference. This conversions lowers the
/// abstract box to the appropriate, initialized descriptor.
struct EmboxOpConversion : public EmboxCommonConversion<fir::EmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxOp embox, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // There should be no dims on this embox op
    assert(!embox.getDims());

    auto loc = embox.getLoc();
    auto *dialect = getDialect();
    auto boxTy = embox.getType().dyn_cast<fir::BoxType>();
    assert(boxTy);
    auto ty = unwrap(lowerTy().convertBoxType(boxTy, 0));
    auto alloca = genAllocaWithType(loc, ty, defaultAlign, dialect, rewriter);
    auto c0 = genConstantOffset(loc, rewriter, 0);

    // Basic pattern to write a field in the descriptor
    auto storeField = [&](unsigned fldIndex, mlir::Value value,
                          const std::function<mlir::Value(
                              mlir::LLVM::LLVMType, mlir::Value)> &applyCast) {
      auto fldTy = getBoxEleTy(ty, fldIndex);
      auto fldPtr = genGEPToField(loc, fldTy, rewriter, alloca, c0, fldIndex);
      auto fld = applyCast(fldTy, value);
      rewriter.create<mlir::LLVM::StoreOp>(loc, fld, fldPtr);
    };

    // Write each of the fields with the appropriate values
    storeField(0, operands[0], [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, ty, val).getResult();
    });
    auto [eleSize, cfiTy] = getSizeAndTypeCode(loc, rewriter, boxTy.getEleTy());
    storeField(1, eleSize, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    auto version = genConstantOffset(loc, rewriter, CFI_VERSION);
    storeField(2, version, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    storeField(3, /*rank*/ c0, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    storeField(4, cfiTy, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    auto attr = genConstantOffset(loc, rewriter, CFI_attribute_other);
    storeField(5, attr, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    storeField(6, /*addend*/ c0, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });

    rewriter.replaceOp(embox, alloca.getResult());
    return success();
  }
};

/// create a generic box on a memory reference
struct XEmboxOpConversion : public EmboxCommonConversion<fir::XEmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::XEmboxOp xbox, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = xbox.getLoc();
    auto *dialect = getDialect();
    auto rank = xbox.getRank();
    auto boxTy = xbox.getType().dyn_cast<fir::BoxType>();
    assert(boxTy);
    auto ty = unwrap(lowerTy().convertBoxType(boxTy, rank));
    auto alloca = genAllocaWithType(loc, ty, defaultAlign, dialect, rewriter);
    auto c0 = genConstantOffset(loc, rewriter, 0);

    // Basic pattern to write a field in the descriptor
    auto storeField = [&](unsigned fldIndex, mlir::Value value,
                          const std::function<mlir::Value(
                              mlir::LLVM::LLVMType, mlir::Value)> &applyCast) {
      auto fldTy = getBoxEleTy(ty, fldIndex);
      auto fldPtr = genGEPToField(loc, fldTy, rewriter, alloca, c0, fldIndex);
      auto fld = applyCast(fldTy, value);
      rewriter.create<mlir::LLVM::StoreOp>(loc, fld, fldPtr);
    };

    // Write each of the fields with the appropriate values
    storeField(0, operands[0], [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, ty, val).getResult();
    });
    auto [eleSize, cfiTy] = getSizeAndTypeCode(loc, rewriter, boxTy.getEleTy());
    storeField(1, eleSize, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    auto version = genConstantOffset(loc, rewriter, CFI_VERSION);
    storeField(2, version, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    auto rankVal = genConstantOffset(loc, rewriter, rank);
    storeField(3, rankVal, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    storeField(4, cfiTy, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    auto attr = genConstantOffset(loc, rewriter, CFI_attribute_other);
    storeField(5, attr, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });
    storeField(6, /*addend*/ c0, [&](mlir::LLVM::LLVMType ty, mlir::Value val) {
      return integerCast(loc, rewriter, ty, val);
    });

    unsigned dimsOff = 1;
    auto i64Ty = mlir::LLVM::LLVMType::getInt64Ty(dialect);
    auto i64PtrTy = i64Ty.getPointerTo();
    for (unsigned d = 0; d < rank; ++d) {
      // store lower bound (normally 0)
      auto f70p = genGEPToField(loc, i64PtrTy, rewriter, alloca, c0, 7, d, 0);
      rewriter.create<mlir::LLVM::StoreOp>(loc, operands[dimsOff++], f70p);
      // store extent
      auto f71p = genGEPToField(loc, i64PtrTy, rewriter, alloca, c0, 7, d, 1);
      rewriter.create<mlir::LLVM::StoreOp>(loc, operands[dimsOff++], f71p);
      // store step (scaled by extent to save a multiplication)
      auto f72p = genGEPToField(loc, i64PtrTy, rewriter, alloca, c0, 7, d, 2);
      rewriter.create<mlir::LLVM::StoreOp>(loc, operands[dimsOff++], f72p);
    }
    auto desc = rewriter.create<mlir::LLVM::BitcastOp>(
        loc, lowerTy().convertType(boxTy), alloca);
    rewriter.replaceOp(xbox, desc.getResult());
    return success();
  }
};

/// create a procedure pointer box
struct EmboxProcOpConversion : public FIROpConversion<fir::EmboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxProcOp emboxproc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = emboxproc.getLoc();
    auto ctx = emboxproc.getContext();
    auto ty = convertType(emboxproc.getType());
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, un,
                                                        operands[0], c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(emboxproc, ty, r,
                                                           operands[1], c1);
    return success();
  }
};

// Code shared between insert_value and extract_value Ops.
struct ValueOpCommon {
  static mlir::Attribute getValue(mlir::Value value) {
    auto defOp = value.getDefiningOp();
    if (auto v = dyn_cast<mlir::LLVM::ConstantOp>(defOp))
      return v.value();
    if (auto v = dyn_cast<mlir::ConstantOp>(defOp))
      return v.value();
    llvm_unreachable("must be a constant op");
    return {};
  }

  // Translate the arguments pertaining to any multidimensional array to
  // row-major order for LLVM-IR.
  static void toRowMajor(llvm::SmallVectorImpl<mlir::Attribute> &attrs,
                         mlir::LLVM::LLVMType ty) {
    assert(ty && "type is null");
    auto *llTy = ty.getUnderlyingType();
    const auto end = attrs.size();
    for (std::remove_const_t<decltype(end)> i = 0; i < end; ++i) {
      if (auto *seq = dyn_cast<llvm::ArrayType>(llTy)) {
        const auto dim = getDimension(seq);
        if (dim > 1) {
          std::reverse(attrs.begin() + i, attrs.begin() + i + dim);
          i += dim - 1;
        }
        llTy = getArrayElementType(seq);
      } else if (auto *st = dyn_cast<llvm::StructType>(llTy)) {
        llTy = st->getElementType(attrs[i].cast<mlir::IntegerAttr>().getInt());
      } else {
        llvm_unreachable("index into invalid type");
      }
    }
  }

private:
  static unsigned getDimension(llvm::ArrayType *ty) {
    unsigned result = 1;
    for (auto *eleTy = dyn_cast<llvm::ArrayType>(ty->getElementType()); eleTy;
         eleTy = dyn_cast<llvm::ArrayType>(eleTy->getElementType()))
      ++result;
    return result;
  }

  static llvm::Type *getArrayElementType(llvm::ArrayType *ty) {
    auto *eleTy = ty->getElementType();
    while (auto *arrTy = dyn_cast<llvm::ArrayType>(eleTy))
      eleTy = arrTy->getElementType();
    return eleTy;
  }
};

/// Extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion
    : public FIROpAndTypeConversion<fir::ExtractValueOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::ExtractValueOp extractVal, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    if (!fir::allConstants(operands.drop_front(1)))
      llvm_unreachable("fir.extract_value incorrectly formed");
    // since all indices are constants use LLVM's extractvalue instruction
    SmallVector<mlir::Attribute, 8> attrs;
    for (std::size_t i = 1, end{operands.size()}; i < end; ++i)
      attrs.push_back(getValue(operands[i]));
    toRowMajor(attrs, lowerTy().unwrap(operands[0].getType()));
    auto position = mlir::ArrayAttr::get(attrs, extractVal.getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        extractVal, ty, operands[0], position);
    return success();
  }
};

/// InsertValue is the generalized instruction for the composition of new
/// aggregate type values.
struct InsertValueOpConversion
    : public FIROpAndTypeConversion<fir::InsertValueOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::InsertValueOp insertVal, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    assert(fir::allConstants(operands.drop_front(2)));
    // since all indices must be constants use LLVM's insertvalue instruction
    SmallVector<mlir::Attribute, 8> attrs;
    for (std::size_t i = 2, end{operands.size()}; i < end; ++i)
      attrs.push_back(getValue(operands[i]));
    toRowMajor(attrs, lowerTy().unwrap(operands[0].getType()));
    auto position = mlir::ArrayAttr::get(attrs, insertVal.getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        insertVal, ty, operands[0], operands[1], position);
    return success();
  }
};

/// InsertOnRange inserts a value into a sequence over a range of offsets.
struct InsertOnRangeOpConversion
    : public FIROpAndTypeConversion<fir::InsertOnRangeOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::InsertOnRangeOp range, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    TODO();
    return success();
  }
};

/// XArrayCoor is the address arithmetic on a dynamically shaped, etc. array.
/// (See the static restriction on coordinate_of.) array_coor determines the
/// coordinate (location) of a specific element.
struct XArrayCoorOpConversion
    : public FIROpAndTypeConversion<fir::XArrayCoorOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::XArrayCoorOp coor, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = coor.getLoc();
    auto rank = coor.getRank();
    assert(coor.indexOperands().size() == rank);
    assert(coor.dimsOperands().size() == 3 * rank);
    unsigned dimsOff = 1;
    unsigned indexOff = 1 + coor.dimsOperands().size();
    // Cast the base address to a pointer to T
    auto base = rewriter.create<mlir::LLVM::BitcastOp>(loc, ty, operands[0]);
    auto idxTy = lowerTy().indexType();
    mlir::Value prevExt = genConstantIndex(loc, idxTy, rewriter, 1);
    mlir::Value off = genConstantIndex(loc, idxTy, rewriter, 0);
    for (unsigned i = 0; i < rank; ++i) {
      auto index = asType(loc, rewriter, idxTy, operands[indexOff++]);
      auto lb = asType(loc, rewriter, idxTy, operands[dimsOff++]);
      auto nextExt = asType(loc, rewriter, idxTy, operands[dimsOff++]);
      auto step = asType(loc, rewriter, idxTy, operands[dimsOff++]);
      // For each dimension, i, add to the running pointer offset the value of
      // (index_i - lb_i) * step_i * extent_{i-1}.
      // Note: LLVM will do constant folding, etc.
      auto diff = rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, index, lb);
      auto sc0 = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, step);
      auto sc1 = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, sc0, prevExt);
      off = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc1, off);
      prevExt =
          rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, prevExt, nextExt);
    }
    llvm::SmallVector<mlir::Value, 4> args{base, off};
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(coor, ty, args);
    return success();
  }

  mlir::Value asType(mlir::Location loc,
                     mlir::ConversionPatternRewriter &rewriter,
                     mlir::LLVM::LLVMType toTy, mlir::Value val) const {
    auto *fromLLVMTy = unwrap(convertType(val.getType())).getUnderlyingType();
    auto *toLLVMTy = toTy.getUnderlyingType();
    assert(fromLLVMTy->isIntegerTy() && toLLVMTy->isIntegerTy());
    if (fromLLVMTy->getIntegerBitWidth() < toLLVMTy->getIntegerBitWidth())
      return rewriter.create<mlir::LLVM::SExtOp>(loc, toTy, val);
    if (fromLLVMTy->getIntegerBitWidth() > toLLVMTy->getIntegerBitWidth())
      return rewriter.create<mlir::LLVM::TruncOp>(loc, toTy, val);
    return val;
  }
};

/// Convert to (memory) reference to a reference to a subobject.
/// The coordinate_of op is a Swiss army knife operation that can be used on
/// (memory) references to records, arrays, complex, etc. as well as boxes.
/// With unboxed arrays, there is the restriction that the array have a static
/// shape in all but the last column.
struct CoordinateOpConversion
    : public FIROpAndTypeConversion<fir::CoordinateOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::CoordinateOp coor, mlir::Type ty, OperandTy operands,
            mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = coor.getLoc();
    auto c0 = genConstantIndex(loc, lowerTy().indexType(), rewriter, 0);
    mlir::Value base = operands[0];
    auto firTy = coor.getBaseType();
    mlir::Type cpnTy = getReferenceEleTy(firTy);
    bool columnIsDeferred = false;
    bool hasSubdimension = hasSubDimensions(cpnTy);

    // if argument 0 is boxed, get the base pointer from the box
    if (auto boxTy = firTy.dyn_cast<fir::BoxType>()) {

      // Special case:
      //   %lenp = len_param_index foo, type<T(foo:i32)...>
      //   %addr = coordinate_of %box, %lenp
      if (coor.getNumOperands() == 2) {
        auto coorPtr = *coor.coor().begin();
        auto s = coorPtr.getDefiningOp();
        if (s && isa<fir::LenParamIndexOp>(s)) {
          mlir::Value lenParam = operands[1]; // byte offset
          auto bc =
              rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy(), base);
          auto gep = genGEP(loc, unwrap(ty), rewriter, bc, lenParam);
          rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(coor, unwrap(ty),
                                                             gep);
          return success();
        }
      }

      auto c0_ = genConstantOffset(loc, rewriter, 0);
      auto pty = unwrap(convertType(boxTy.getEleTy())).getPointerTo();
      // Extract the boxed reference
      auto p = genGEP(loc, pty, rewriter, base, c0, c0_);
      // base = box->data : ptr
      base = rewriter.create<mlir::LLVM::LoadOp>(loc, pty, p);

      // If the base has dynamic shape, it has to be boxed as the dimension
      // information is saved in the box.
      if (FIRToLLVMTypeConverter::dynamicallySized(cpnTy)) {
        TODO();
        return success();
      }
    } else {
      if (FIRToLLVMTypeConverter::dynamicallySized(cpnTy))
        return mlir::emitError(loc, "bare reference to unknown shape");
    }
    if (!hasSubdimension)
      columnIsDeferred = true;

    if (!validCoordinate(cpnTy, operands.drop_front(1)))
      return mlir::emitError(loc, "coordinate has incorrect dimension");

    // if arrays has known shape
    const bool hasKnownShape =
        arraysHaveKnownShape(cpnTy, operands.drop_front(1));

    // If only the column is `?`, then we can simply place the column value in
    // the 0-th GEP position.
    if (auto arrTy = cpnTy.dyn_cast<fir::SequenceType>()) {
      if (!hasKnownShape) {
        const auto sz = arrTy.getDimension();
        if (arraysHaveKnownShape(arrTy.getEleTy(),
                                 operands.drop_front(1 + sz))) {
          auto shape = arrTy.getShape();
          bool allConst = true;
          for (std::remove_const_t<decltype(sz)> i = 0; i < sz - 1; ++i)
            if (shape[i] < 0) {
              allConst = false;
              break;
            }
          if (allConst)
            columnIsDeferred = true;
        }
      }
    }

    if (hasKnownShape || columnIsDeferred) {
      SmallVector<mlir::Value, 8> offs;
      if (hasKnownShape && hasSubdimension)
        offs.push_back(c0);
      const auto sz = operands.size();
      llvm::Optional<int> dims;
      SmallVector<mlir::Value, 8> arrIdx;
      for (std::remove_const_t<decltype(sz)> i = 1; i < sz; ++i) {
        auto nxtOpnd = operands[i];

        if (!cpnTy)
          return mlir::emitError(loc, "invalid coordinate/check failed");

        // check if the i-th coordinate relates to an array
        if (dims.hasValue()) {
          arrIdx.push_back(nxtOpnd);
          int dimsLeft = *dims;
          if (dimsLeft > 1) {
            dims = dimsLeft - 1;
            continue;
          }
          cpnTy = cpnTy.cast<fir::SequenceType>().getEleTy();
          // append array range in reverse (FIR arrays are column-major)
          offs.append(arrIdx.rbegin(), arrIdx.rend());
          arrIdx.clear();
          dims.reset();
          continue;
        } else if (auto arrTy = cpnTy.dyn_cast<fir::SequenceType>()) {
          int d = arrTy.getDimension() - 1;
          if (d > 0) {
            dims = d;
            arrIdx.push_back(nxtOpnd);
            continue;
          }
          cpnTy = cpnTy.cast<fir::SequenceType>().getEleTy();
          offs.push_back(nxtOpnd);
          continue;
        }

        // check if the i-th coordinate relates to a field
        if (auto strTy = cpnTy.dyn_cast<fir::RecordType>()) {
          cpnTy = strTy.getType(getIntValue(nxtOpnd));
        } else if (auto strTy = cpnTy.dyn_cast<mlir::TupleType>()) {
          cpnTy = strTy.getType(getIntValue(nxtOpnd));
        } else {
          cpnTy = nullptr;
        }
        offs.push_back(nxtOpnd);
      }
      if (dims.hasValue())
        offs.append(arrIdx.rbegin(), arrIdx.rend());
      mlir::Value retval = genGEP(loc, unwrap(ty), rewriter, base, offs);
      rewriter.replaceOp(coor, retval);
      return success();
    }

    // Taking a coordinate of an array with deferred shape. In this case, the
    // array must be boxed. We need to retrieve the array triples from the box.
    //
    // Given:
    //
    //   %box ... : box<array<? x ? x ? x i32>>
    //   %addr = coordinate_of %box, %0, %1, %2
    //
    // We want to lower this into an llvm GEP as:
    //
    //   %i1 = (%0 - %box.dims(0).lo) * %box.dims(0).str
    //   %i2 = (%1 - %box.dims(1).lo) * %box.dims(1).str * %box.dims(0).ext
    //   %scale_by = %box.dims(1).ext * %box.dims(0).ext
    //   %i3 = (%2 - %box.dims(2).lo) * %box.dims(2).str * %scale_by
    //   %offset = %i3 + %i2 + %i1
    //   %addr = getelementptr i32, i32* %box.ref, i64 %offset
    //
    // Section 18.5.3 para 3 specifies when and how to interpret the `lo`
    // value(s) of the triple. The implication is that they must always be
    // zero for `coordinate_of`. This is because we do not use `coordinate_of`
    // to compute the offset into a `box<ptr>` or `box<heap>`. The coordinate
    // is pointer arithmetic. Pointers along a path must be explicitly
    // dereferenced with a `load`.

    if (!firTy.isa<fir::BoxType>())
      return mlir::emitError(loc, "base must have box type");
    if (!cpnTy.isa<fir::SequenceType>())
      return mlir::emitError(loc, "base element must be reference to array");
    auto baseTy = cpnTy.cast<fir::SequenceType>();
    const auto baseDim = baseTy.getDimension();
    if (!arraysHaveKnownShape(baseTy.getEleTy(),
                              operands.drop_front(1 + baseDim)))
      return mlir::emitError(loc, "base element has deferred shapes");

    // Generate offset computation.
    TODO();

    return failure();
  }

  bool hasSubDimensions(mlir::Type type) const {
    return type.isa<fir::SequenceType>() || type.isa<fir::RecordType>() ||
           type.isa<mlir::TupleType>();
  }

  /// Walk the abstract memory layout and determine if the path traverses any
  /// array types with unknown shape. Return true iff all the array types have a
  /// constant shape along the path.
  bool arraysHaveKnownShape(mlir::Type type, OperandTy coors) const {
    const auto sz = coors.size();
    std::remove_const_t<decltype(sz)> i = 0;
    for (; i < sz; ++i) {
      auto nxtOpnd = coors[i];
      if (auto arrTy = type.dyn_cast<fir::SequenceType>()) {
        if (FIRToLLVMTypeConverter::unknownShape(arrTy.getShape()))
          return false;
        i += arrTy.getDimension() - 1;
        type = arrTy.getEleTy();
      } else if (auto strTy = type.dyn_cast<fir::RecordType>()) {
        type = strTy.getType(getIntValue(nxtOpnd));
      } else if (auto strTy = type.dyn_cast<mlir::TupleType>()) {
        type = strTy.getType(getIntValue(nxtOpnd));
      } else {
        return true;
      }
    }
    return true;
  }

  bool validCoordinate(mlir::Type type, OperandTy coors) const {
    const auto sz = coors.size();
    std::remove_const_t<decltype(sz)> i = 0;
    bool subEle = false;
    bool ptrEle = false;
    for (; i < sz; ++i) {
      auto nxtOpnd = coors[i];
      if (auto arrTy = type.dyn_cast<fir::SequenceType>()) {
        subEle = true;
        i += arrTy.getDimension() - 1;
        type = arrTy.getEleTy();
      } else if (auto strTy = type.dyn_cast<fir::RecordType>()) {
        subEle = true;
        type = strTy.getType(getIntValue(nxtOpnd));
      } else if (auto strTy = type.dyn_cast<mlir::TupleType>()) {
        subEle = true;
        type = strTy.getType(getIntValue(nxtOpnd));
      } else {
        ptrEle = true;
      }
    }
    if (ptrEle)
      return (!subEle) && (sz == 1);
    return subEle && (i >= sz);
  }

  /// Returns the element type of the reference `refTy`.
  static mlir::Type getReferenceEleTy(mlir::Type refTy) {
    if (auto boxTy = refTy.dyn_cast<fir::BoxType>())
      return boxTy.getEleTy();
    if (auto ptrTy = refTy.dyn_cast<fir::ReferenceType>())
      return ptrTy.getEleTy();
    if (auto ptrTy = refTy.dyn_cast<fir::PointerType>())
      return ptrTy.getEleTy();
    if (auto ptrTy = refTy.dyn_cast<fir::HeapType>())
      return ptrTy.getEleTy();
    llvm_unreachable("not a reference type");
  }

  /// return true if all `Value`s in `operands` are not `FieldIndexOp`s
  static bool noFieldIndexOps(mlir::Operation::operand_range operands) {
    for (auto opnd : operands) {
      if (auto defop = opnd.getDefiningOp())
        if (dyn_cast<fir::FieldIndexOp>(defop))
          return false;
    }
    return true;
  }

  SmallVector<mlir::Value, 8> arguments(OperandTy vec, unsigned s,
                                        unsigned e) const {
    return {vec.begin() + s, vec.begin() + e};
  }

  int64_t getIntValue(mlir::Value val) const {
    if (val)
      if (auto defop = val.getDefiningOp())
        if (auto constOp = dyn_cast<mlir::ConstantIntOp>(defop))
          return constOp.getValue();
    llvm_unreachable("must be a constant");
  }
};

/// convert a field index to a runtime function that computes the byte offset
/// of the dynamic field
struct FieldIndexOpConversion : public FIROpConversion<fir::FieldIndexOp> {
  using FIROpConversion::FIROpConversion;

  // NB: most field references should be resolved by this point
  mlir::LogicalResult
  matchAndRewrite(fir::FieldIndexOp field, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // call the compiler generated function to determine the byte offset of
    // the field at runtime
    auto symAttr =
        mlir::SymbolRefAttr::get(methodName(field), field.getContext());
    SmallVector<mlir::NamedAttribute, 1> attrs{
        rewriter.getNamedAttr("callee", symAttr)};
    auto ty = lowerTy().offsetType();
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(field, ty, operands, attrs);
    return success();
  }

  // constructing the name of the method
  inline static std::string methodName(fir::FieldIndexOp field) {
    auto fldName = field.field_id();
    auto type = field.on_type().cast<fir::RecordType>();
    // note: using std::string to dodge a bug in g++ 7.4.0
    std::string tyName = type.getName().str();
    llvm::Twine methodName = "_QQOFFSETOF_" + tyName + "_" + fldName;
    return methodName.str();
  }
};

struct LenParamIndexOpConversion
    : public FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  // FIXME: this should be specialized by the runtime target
  mlir::LogicalResult
  matchAndRewrite(fir::LenParamIndexOp lenp, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ity = lowerTy().indexType();
    auto onty = lenp.getOnType();
    // size of portable descriptor
    const unsigned boxsize = 24; // FIXME
    unsigned offset = boxsize;
    // add the size of the rows of triples
    if (auto arr = onty.dyn_cast<fir::SequenceType>()) {
      offset += 3 * arr.getDimension();
    }
    // advance over some addendum fields
    const unsigned addendumOffset{sizeof(void *) + sizeof(uint64_t)};
    offset += addendumOffset;
    // add the offset into the LENs
    offset += 0; // FIXME
    auto attr = rewriter.getI64IntegerAttr(offset);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(lenp, ity, attr);
    return success();
  }
};

/// lower the fir.end operation to a null (erasing it)
struct FirEndOpConversion : public FIROpConversion<fir::FirEndOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::FirEndOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {});
    return success();
  }
};

/// lower a type descriptor to a global constant
struct GenTypeDescOpConversion : public FIROpConversion<fir::GenTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GenTypeDescOp gentypedesc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = gentypedesc.getLoc();
    auto inTy = gentypedesc.getInType();
    auto name = consName(rewriter, inTy);
    auto gty = unwrap(convertType(inTy));
    auto pty = gty.getPointerTo();
    auto module = gentypedesc.getParentOfType<mlir::ModuleOp>();
    createGlobal(loc, module, name, gty, rewriter);
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(gentypedesc, pty,
                                                         name);
    return success();
  }

  std::string consName(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Type type) const {
    if (auto d = type.dyn_cast<fir::RecordType>()) {
      auto name = d.getName();
      auto pair = fir::NameUniquer::deconstruct(name);
      return lowerTy().getUniquer().doTypeDescriptor(
          pair.second.modules, pair.second.host, pair.second.name,
          pair.second.kinds);
    }
    llvm_unreachable("no name found");
  }
};

struct GlobalLenOpConversion : public FIROpConversion<fir::GlobalLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalLenOp globalLen, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO();
    return success();
  }
};

struct HasValueOpConversion : public FIROpConversion<fir::HasValueOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::HasValueOp op, OperandTy operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }
};

struct GlobalOpConversion : public FIROpConversion<fir::GlobalOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalOp global, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tyAttr = unwrap(convertType(global.getType()));
    auto loc = global.getLoc();
    mlir::Attribute initAttr{};
    if (global.initVal())
      initAttr = global.initVal().getValue();
    auto linkage = convertLinkage(global.linkName());
    auto isConst = global.constant().hasValue();
    auto g = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, tyAttr, isConst, linkage, global.sym_name(), initAttr);
    auto &gr = g.getInitializerRegion();
    rewriter.inlineRegionBefore(global.region(), gr, gr.end());
    rewriter.eraseOp(global);
    return success();
  }

  mlir::LLVM::Linkage
  convertLinkage(llvm::Optional<llvm::StringRef> optLinkage) const {
    if (optLinkage.hasValue()) {
      auto name = optLinkage.getValue();
      if (name == "internal")
        return mlir::LLVM::Linkage::Internal;
      if (name == "common")
        return mlir::LLVM::Linkage::Common;
      if (name == "weak")
        return mlir::LLVM::Linkage::Weak;
    }
    return mlir::LLVM::Linkage::External;
  }
};

// convert to LLVM IR dialect `load`
struct LoadOpConversion : public FIROpConversion<fir::LoadOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::LoadOp load, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(load.getType());
    auto at = load.getAttrs();
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(load, ty, operands, at);
    return success();
  }
};

// FIXME: how do we want to enforce this in LLVM-IR? Can we manipulate the fast
// math flags?
struct NoReassocOpConversion : public FIROpConversion<fir::NoReassocOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::NoReassocOp noreassoc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    noreassoc.replaceAllUsesWith(operands[0]);
    rewriter.eraseOp(noreassoc);
    return success();
  }
};

void genCondBrOp(mlir::Location loc, mlir::Value cmp, mlir::Block *dest,
                 llvm::Optional<OperandTy> destOps,
                 mlir::ConversionPatternRewriter &rewriter,
                 mlir::Block *newBlock) {
  if (destOps.hasValue())
    rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, dest, destOps.getValue(),
                                          newBlock, mlir::ValueRange());
  else
    rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, dest, newBlock);
}

template <typename A, typename B>
void genBrOp(A caseOp, mlir::Block *dest, llvm::Optional<B> destOps,
             mlir::ConversionPatternRewriter &rewriter) {
  if (destOps.hasValue())
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(caseOp, destOps.getValue(),
                                                  dest);
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(caseOp, llvm::None, dest);
}

void genCaseLadderStep(mlir::Location loc, mlir::Value cmp, mlir::Block *dest,
                       llvm::Optional<OperandTy> destOps,
                       mlir::ConversionPatternRewriter &rewriter) {
  auto *thisBlock = rewriter.getInsertionBlock();
  auto *newBlock = createBlock(rewriter, dest);
  rewriter.setInsertionPointToEnd(thisBlock);
  genCondBrOp(loc, cmp, dest, destOps, rewriter, newBlock);
  rewriter.setInsertionPointToEnd(newBlock);
}

/// Conversion of `fir.select_case`
///
/// TODO: lowering of CHARACTER type cases
struct SelectCaseOpConversion : public FIROpConversion<fir::SelectCaseOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectCaseOp caseOp, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const auto conds = caseOp.getNumConditions();
    auto attrName = fir::SelectCaseOp::getCasesAttr();
    auto cases = caseOp.getAttrOfType<mlir::ArrayAttr>(attrName).getValue();
    // Type can be CHARACTER, INTEGER, or LOGICAL (C1145)
    [[maybe_unused]] auto ty = caseOp.getSelector().getType();
    auto selector = caseOp.getSelector(operands);
    auto loc = caseOp.getLoc();
    assert(conds > 0 && "fir.selectcase must have cases");
    for (std::remove_const_t<decltype(conds)> t = 0; t != conds; ++t) {
      mlir::Block *dest = caseOp.getSuccessor(t);
      auto destOps = caseOp.getSuccessorOperands(operands, t);
      auto cmpOps = *caseOp.getCompareOperands(operands, t);
      auto caseArg = *cmpOps.begin();
      auto &attr = cases[t];
      if (attr.isa<fir::PointIntervalAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::eq, selector, caseArg);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::LowerBoundAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, caseArg, selector);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::UpperBoundAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, caseArg);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::ClosedIntervalAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, caseArg, selector);
        auto *thisBlock = rewriter.getInsertionBlock();
        auto *newBlock1 = createBlock(rewriter, dest);
        auto *newBlock2 = createBlock(rewriter, dest);
        rewriter.setInsertionPointToEnd(thisBlock);
        rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, newBlock1, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock1);
        auto caseArg_ = *(cmpOps.begin() + 1);
        auto cmp_ = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, caseArg_);
        genCondBrOp(loc, cmp_, dest, destOps, rewriter, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock2);
        continue;
      }
      assert(attr.isa<mlir::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      genBrOp(caseOp, dest, destOps, rewriter);
    }
    return success();
  }
};

template <typename OP>
void selectMatchAndRewrite(FIRToLLVMTypeConverter &lowering, OP select,
                           OperandTy operands,
                           mlir::ConversionPatternRewriter &rewriter) {
  // We could target the LLVM switch instruction, but it isn't part of the
  // LLVM IR dialect.  Create an if-then-else ladder instead.
  auto conds = select.getNumConditions();
  auto attrName = OP::getCasesAttr();
  auto caseAttr = select.template getAttrOfType<mlir::ArrayAttr>(attrName);
  auto cases = caseAttr.getValue();
  auto ty = select.getSelector().getType();
  auto ity = lowering.convertType(ty);
  auto selector = select.getSelector(operands);
  auto loc = select.getLoc();
  assert(conds > 0 && "select must have cases");
  for (decltype(conds) t = 0; t != conds; ++t) {
    mlir::Block *dest = select.getSuccessor(t);
    auto destOps = select.getSuccessorOperands(operands, t);
    auto &attr = cases[t];
    if (auto intAttr = attr.template dyn_cast<mlir::IntegerAttr>()) {
      auto ci = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, ity, rewriter.getIntegerAttr(ty, intAttr.getInt()));
      auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, selector, ci);
      genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
      continue;
    }
    assert(attr.template dyn_cast_or_null<mlir::UnitAttr>());
    assert((t + 1 == conds) && "unit must be last");
    genBrOp(select, dest, destOps, rewriter);
  }
}

/// conversion of fir::SelectOp to an if-then-else ladder
struct SelectOpConversion : public FIROpConversion<fir::SelectOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectOp>(lowerTy(), op, operands, rewriter);
    return success();
  }
};

/// conversion of fir::SelectRankOp to an if-then-else ladder
struct SelectRankOpConversion : public FIROpConversion<fir::SelectRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectRankOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectRankOp>(lowerTy(), op, operands, rewriter);
    return success();
  }
};

struct SelectTypeOpConversion : public FIROpConversion<fir::SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectTypeOp select, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::emitError(select.getLoc(),
                    "fir.select_type should have already been converted");
    return failure();
  }
};

// convert to LLVM IR dialect `store`
struct StoreOpConversion : public FIROpConversion<fir::StoreOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::StoreOp store, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(store, operands[0],
                                                     operands[1]);
    return success();
  }
};

// cons an extractvalue on a tuple value, returning value at element `x`
mlir::LLVM::ExtractValueOp genExtractValueWithIndex(
    mlir::Location loc, mlir::Value tuple, mlir::LLVM::LLVMType ty,
    mlir::ConversionPatternRewriter &rewriter, mlir::MLIRContext *ctx, int x) {
  auto cx = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(x), ctx);
  auto xty = ty.getStructElementType(x);
  return rewriter.create<mlir::LLVM::ExtractValueOp>(loc, xty, tuple, cx);
}

// unbox a CHARACTER box value, yielding its components
struct UnboxCharOpConversion : public FIROpConversion<fir::UnboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnboxCharOp unboxchar, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *ctx = unboxchar.getContext();
    auto loc = unboxchar.getLoc();
    auto tuple = operands[0];
    auto ty = unwrap(tuple.getType());
    mlir::Value ptr =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 0);
    mlir::Value len =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 1);
    std::vector<mlir::Value> repls = {ptr, len};
    unboxchar.replaceAllUsesWith(repls);
    rewriter.eraseOp(unboxchar);
    return success();
  }
};

// unbox a generic box reference, yielding its components
struct UnboxOpConversion : public FIROpConversion<fir::UnboxOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnboxOp unbox, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = unbox.getLoc();
    auto tuple = operands[0];
    auto ty = unwrap(tuple.getType());
    auto oty = lowerTy().offsetType();
    auto c0 = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, oty, rewriter.getI32IntegerAttr(0));
    mlir::Value ptr = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 0);
    mlir::Value len = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 1);
    mlir::Value ver = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 2);
    mlir::Value rank = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 3);
    mlir::Value type = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 4);
    mlir::Value attr = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 5);
    mlir::Value xtra = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 6);
    // FIXME: add dims, etc.
    std::vector<mlir::Value> repls{ptr, len, ver, rank, type, attr, xtra};
    unbox.replaceAllUsesWith(repls);
    rewriter.eraseOp(unbox);
    return success();
  }

  // generate a GEP into a structure and load the element at position `x`
  mlir::LLVM::LoadOp genLoadWithIndex(mlir::Location loc, mlir::Value tuple,
                                      mlir::LLVM::LLVMType ty,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      mlir::LLVM::LLVMType oty,
                                      mlir::LLVM::ConstantOp c0, int x) const {
    auto ax = rewriter.getI32IntegerAttr(x);
    auto cx = rewriter.create<mlir::LLVM::ConstantOp>(loc, oty, ax);
    auto xty = ty.getStructElementType(x);
    auto gep = genGEP(loc, xty.getPointerTo(), rewriter, tuple, c0, cx);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, xty, gep);
  }
};

// unbox a procedure box value, yielding its components
struct UnboxProcOpConversion : public FIROpConversion<fir::UnboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnboxProcOp unboxproc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *ctx = unboxproc.getContext();
    auto loc = unboxproc.getLoc();
    auto tuple = operands[0];
    auto ty = unwrap(tuple.getType());
    mlir::Value ptr =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 0);
    mlir::Value host =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 1);
    std::vector<mlir::Value> repls{ptr, host};
    unboxproc.replaceAllUsesWith(repls);
    rewriter.eraseOp(unboxproc);
    return success();
  }
};

// convert to LLVM IR dialect `undef`
struct UndefOpConversion : public FIROpConversion<fir::UndefOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UndefOp undef, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(
        undef, convertType(undef.getType()));
    return success();
  }
};

// convert to LLVM IR dialect `unreachable`
struct UnreachableOpConversion : public FIROpConversion<fir::UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnreachableOp unreach, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(unreach);
    return success();
  }
};

//
// Primitive operations on Real (floating-point) types
//

/// Convert a floating-point primitive
template <typename LLVMOP, typename BINOP>
void lowerRealBinaryOp(BINOP binop, OperandTy operands,
                       mlir::ConversionPatternRewriter &rewriter,
                       FIRToLLVMTypeConverter &lowering) {
  auto ty = lowering.convertType(binop.getType());
  rewriter.replaceOpWithNewOp<LLVMOP>(binop, ty, operands);
}

struct AddfOpConversion : public FIROpConversion<fir::AddfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AddfOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<mlir::LLVM::FAddOp>(op, operands, rewriter, lowerTy());
    return success();
  }
};
struct SubfOpConversion : public FIROpConversion<fir::SubfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SubfOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<mlir::LLVM::FSubOp>(op, operands, rewriter, lowerTy());
    return success();
  }
};
struct MulfOpConversion : public FIROpConversion<fir::MulfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::MulfOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<mlir::LLVM::FMulOp>(op, operands, rewriter, lowerTy());
    return success();
  }
};
struct DivfOpConversion : public FIROpConversion<fir::DivfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DivfOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<mlir::LLVM::FDivOp>(op, operands, rewriter, lowerTy());
    return success();
  }
};
struct ModfOpConversion : public FIROpConversion<fir::ModfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ModfOp op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<mlir::LLVM::FRemOp>(op, operands, rewriter, lowerTy());
    return success();
  }
};

struct NegfOpConversion : public FIROpConversion<fir::NegfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::NegfOp neg, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(neg.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::FNegOp>(neg, ty, operands);
    return success();
  }
};

//
// Primitive operations on Complex types
//

/// Generate inline code for complex addition/subtraction
template <typename LLVMOP, typename OPTY>
mlir::LLVM::InsertValueOp complexSum(OPTY sumop, OperandTy opnds,
                                     mlir::ConversionPatternRewriter &rewriter,
                                     FIRToLLVMTypeConverter &lowering) {
  auto a = opnds[0];
  auto b = opnds[1];
  auto loc = sumop.getLoc();
  auto ctx = sumop.getContext();
  auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
  auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
  auto ty = lowering.convertType(sumop.getType());
  auto x = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c0);
  auto x_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c0);
  auto rx = rewriter.create<LLVMOP>(loc, ty, x, x_);
  auto y = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c1);
  auto y_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c1);
  auto ry = rewriter.create<LLVMOP>(loc, ty, y, y_);
  auto r = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
  auto r_ = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r, rx, c0);
  return rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r_, ry, c1);
}

struct AddcOpConversion : public FIROpConversion<fir::AddcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AddcOp addc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) * (x' + iy')
    // result: (x + x') + i(y + y')
    auto r =
        complexSum<mlir::LLVM::FAddOp>(addc, operands, rewriter, lowerTy());
    addc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(addc, r.getResult());
    return success();
  }
};

struct SubcOpConversion : public FIROpConversion<fir::SubcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SubcOp subc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) * (x' + iy')
    // result: (x - x') + i(y - y')
    auto r =
        complexSum<mlir::LLVM::FSubOp>(subc, operands, rewriter, lowerTy());
    subc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(subc, r.getResult());
    return success();
  }
};

/// Inlined complex multiply
struct MulcOpConversion : public FIROpConversion<fir::MulcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::MulcOp mulc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO: should this just call __muldc3 ?
    // given: (x + iy) * (x' + iy')
    // result: (xx'-yy')+i(xy'+yx')
    auto a = operands[0];
    auto b = operands[1];
    auto loc = mulc.getLoc();
    auto ctx = mulc.getContext();
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto ty = convertType(mulc.getType());
    auto x = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c0);
    auto x_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c0);
    auto xx_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x, x_);
    auto y = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c1);
    auto yx_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y, x_);
    auto y_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c1);
    auto xy_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x, y_);
    auto ri = rewriter.create<mlir::LLVM::FAddOp>(loc, ty, xy_, yx_);
    auto yy_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y, y_);
    auto rr = rewriter.create<mlir::LLVM::FSubOp>(loc, ty, xx_, yy_);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r_ = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r_, ri, c1);
    mulc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(mulc, r.getResult());
    return success();
  }
};

/// Inlined complex division
struct DivcOpConversion : public FIROpConversion<fir::DivcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DivcOp divc, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO: should this just call __divdc3 ?
    // given: (x + iy) / (x' + iy')
    // result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'
    auto a = operands[0];
    auto b = operands[1];
    auto loc = divc.getLoc();
    auto ctx = divc.getContext();
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto ty = convertType(divc.getType());
    auto x = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c0);
    auto x_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c0);
    auto xx_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x, x_);
    auto x_x_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x_, x_);
    auto y = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c1);
    auto yx_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y, x_);
    auto y_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c1);
    auto xy_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x, y_);
    auto yy_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y, y_);
    auto y_y_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y_, y_);
    auto d = rewriter.create<mlir::LLVM::FAddOp>(loc, ty, x_x_, y_y_);
    auto rrn = rewriter.create<mlir::LLVM::FAddOp>(loc, ty, xx_, yy_);
    auto rin = rewriter.create<mlir::LLVM::FSubOp>(loc, ty, yx_, xy_);
    auto rr = rewriter.create<mlir::LLVM::FDivOp>(loc, ty, rrn, d);
    auto ri = rewriter.create<mlir::LLVM::FDivOp>(loc, ty, rin, d);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r_ = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r_, ri, c1);
    divc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(divc, r.getResult());
    return success();
  }
};

/// Inlined complex negation
struct NegcOpConversion : public FIROpConversion<fir::NegcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::NegcOp neg, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: -(x + iy)
    // result: -x - iy
    auto ctxt = neg.getContext();
    auto ty = convertType(neg.getType());
    auto loc = neg.getLoc();
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctxt);
    auto &o0 = operands[0];
    auto rp = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, o0, c0);
    auto nrp = rewriter.create<mlir::LLVM::FNegOp>(loc, ty, rp);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctxt);
    auto ip = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, o0, c1);
    auto nip = rewriter.create<mlir::LLVM::FNegOp>(loc, ty, ip);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, o0, nrp, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(neg, ty, r, nip, c1);
    return success();
  }
};

// Lower a SELECT operation into a cascade of conditional branches. The last
// case must be the `true` condition.
/// Convert FIR dialect to LLVM dialect
///
/// This pass lowers all FIR dialect operations to LLVM IR dialect.  An
/// MLIR pass is used to lower residual Std dialect to LLVM IR dialect.
struct FIRToLLVMLoweringPass
    : public mlir::PassWrapper<FIRToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  FIRToLLVMLoweringPass(fir::NameUniquer &uniquer) : uniquer{uniquer} {}

  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    if (disableFirToLLVMIR)
      return;

    auto *context{&getContext()};
    FIRToLLVMTypeConverter typeConverter{context, uniquer};
    auto loc = mlir::UnknownLoc::get(context);
    mlir::OwningRewritePatternList pattern;
    pattern.insert<
        AddcOpConversion, AddfOpConversion, AddrOfOpConversion,
        AllocaOpConversion, AllocMemOpConversion, BoxAddrOpConversion,
        BoxCharLenOpConversion, BoxDimsOpConversion, BoxEleSizeOpConversion,
        BoxIsAllocOpConversion, BoxIsArrayOpConversion, BoxIsPtrOpConversion,
        BoxProcHostOpConversion, BoxRankOpConversion, BoxTypeDescOpConversion,
        CallOpConversion, CmpcOpConversion, CmpfOpConversion,
        ConstcOpConversion, ConstfOpConversion, ConvertOpConversion,
        CoordinateOpConversion, DispatchOpConversion, DispatchTableOpConversion,
        DivcOpConversion, DivfOpConversion, DTEntryOpConversion,
        EmboxOpConversion, EmboxCharOpConversion, EmboxProcOpConversion,
        FieldIndexOpConversion, FirEndOpConversion, ExtractValueOpConversion,
        FreeMemOpConversion, GenTypeDescOpConversion, GlobalLenOpConversion,
        GlobalOpConversion, HasValueOpConversion, InsertOnRangeOpConversion,
        InsertValueOpConversion, LenParamIndexOpConversion, LoadOpConversion,
        ModfOpConversion, MulcOpConversion, MulfOpConversion, NegcOpConversion,
        NegfOpConversion, NoReassocOpConversion, SelectCaseOpConversion,
        SelectOpConversion, SelectRankOpConversion, SelectTypeOpConversion,
        StoreOpConversion, StringLitOpConversion, SubcOpConversion,
        SubfOpConversion, UnboxCharOpConversion, UnboxOpConversion,
        UnboxProcOpConversion, UndefOpConversion, UnreachableOpConversion,
        XArrayCoorOpConversion, XEmboxOpConversion>(context, typeConverter);
    // Workaround D80285: beware, optional LowerToLLVMOptions argument of
    // populateStdToLLVMConversionPatterns is broken. It ends up creating a
    // reference over a temp that has the lifetime of the call. Do not use
    // it.
    mlir::LowerToLLVMOptions options;
    mlir::populateStdToLLVMConversionPatterns(typeConverter, pattern, options);
    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    // required NOPs for applying a full conversion
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

    // apply the patterns
    if (mlir::failed(mlir::applyFullConversion(getModule(), target,
                                               std::move(pattern)))) {
      mlir::emitError(loc, "error in converting to LLVM-IR dialect\n");
      signalPassFailure();
    }
  }

private:
  fir::NameUniquer &uniquer;
};

/// Lower from LLVM IR dialect to proper LLVM-IR and dump the module
struct LLVMIRLoweringPass
    : public mlir::PassWrapper<LLVMIRLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  LLVMIRLoweringPass(raw_ostream &output) : output{output} {}

  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    if (disableLLVM)
      return;

    if (auto llvmModule = mlir::translateModuleToLLVMIR(getModule())) {
      llvmModule->print(output, nullptr);
      return;
    }

    auto *ctxt = getModule().getContext();
    mlir::emitError(mlir::UnknownLoc::get(ctxt), "could not emit LLVM-IR\n");
    signalPassFailure();
  }

private:
  llvm::raw_ostream &output;
};

} // namespace

std::unique_ptr<mlir::Pass>
fir::createFIRToLLVMPass(fir::NameUniquer &nameUniquer) {
  return std::make_unique<FIRToLLVMLoweringPass>(nameUniquer);
}

std::unique_ptr<mlir::Pass>
fir::createLLVMDialectToLLVMPass(llvm::raw_ostream &output) {
  return std::make_unique<LLVMIRLoweringPass>(output);
}

// Register the FIR to LLVM-IR pass
static mlir::PassRegistration<FIRToLLVMLoweringPass>
    passLowFIR("fir-to-llvmir",
               "Conversion of the FIR dialect to the LLVM-IR dialect", [] {
                 fir::NameUniquer dummy;
                 return std::make_unique<FIRToLLVMLoweringPass>(dummy);
               });
