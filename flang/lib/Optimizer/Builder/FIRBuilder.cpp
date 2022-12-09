//===-- FIRBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

static llvm::cl::opt<std::size_t>
    nameLengthHashSize("length-to-hash-string-literal",
                       llvm::cl::desc("string literals that exceed this length"
                                      " will use a hash value as their symbol "
                                      "name"),
                       llvm::cl::init(32));

mlir::func::FuncOp fir::FirOpBuilder::createFunction(mlir::Location loc,
                                                     mlir::ModuleOp module,
                                                     llvm::StringRef name,
                                                     mlir::FunctionType ty) {
  return fir::createFuncOp(loc, module, name, ty);
}

mlir::func::FuncOp fir::FirOpBuilder::getNamedFunction(mlir::ModuleOp modOp,
                                                       llvm::StringRef name) {
  return modOp.lookupSymbol<mlir::func::FuncOp>(name);
}

mlir::func::FuncOp
fir::FirOpBuilder::getNamedFunction(mlir::ModuleOp modOp,
                                    mlir::SymbolRefAttr symbol) {
  return modOp.lookupSymbol<mlir::func::FuncOp>(symbol);
}

fir::GlobalOp fir::FirOpBuilder::getNamedGlobal(mlir::ModuleOp modOp,
                                                llvm::StringRef name) {
  return modOp.lookupSymbol<fir::GlobalOp>(name);
}

mlir::Type fir::FirOpBuilder::getRefType(mlir::Type eleTy) {
  assert(!eleTy.isa<fir::ReferenceType>() && "cannot be a reference type");
  return fir::ReferenceType::get(eleTy);
}

mlir::Type fir::FirOpBuilder::getVarLenSeqTy(mlir::Type eleTy, unsigned rank) {
  fir::SequenceType::Shape shape(rank, fir::SequenceType::getUnknownExtent());
  return fir::SequenceType::get(shape, eleTy);
}

mlir::Type fir::FirOpBuilder::getRealType(int kind) {
  switch (kindMap.getRealTypeID(kind)) {
  case llvm::Type::TypeID::HalfTyID:
    return mlir::FloatType::getF16(getContext());
  case llvm::Type::TypeID::FloatTyID:
    return mlir::FloatType::getF32(getContext());
  case llvm::Type::TypeID::DoubleTyID:
    return mlir::FloatType::getF64(getContext());
  case llvm::Type::TypeID::X86_FP80TyID:
    return mlir::FloatType::getF80(getContext());
  case llvm::Type::TypeID::FP128TyID:
    return mlir::FloatType::getF128(getContext());
  default:
    fir::emitFatalError(mlir::UnknownLoc::get(getContext()),
                        "unsupported type !fir.real<kind>");
  }
}

mlir::Value fir::FirOpBuilder::createNullConstant(mlir::Location loc,
                                                  mlir::Type ptrType) {
  auto ty = ptrType ? ptrType : getRefType(getNoneType());
  return create<fir::ZeroOp>(loc, ty);
}

mlir::Value fir::FirOpBuilder::createIntegerConstant(mlir::Location loc,
                                                     mlir::Type ty,
                                                     std::int64_t cst) {
  return create<mlir::arith::ConstantOp>(loc, ty, getIntegerAttr(ty, cst));
}

mlir::Value
fir::FirOpBuilder::createRealConstant(mlir::Location loc, mlir::Type fltTy,
                                      llvm::APFloat::integerPart val) {
  auto apf = [&]() -> llvm::APFloat {
    if (auto ty = fltTy.dyn_cast<fir::RealType>())
      return llvm::APFloat(kindMap.getFloatSemantics(ty.getFKind()), val);
    if (fltTy.isF16())
      return llvm::APFloat(llvm::APFloat::IEEEhalf(), val);
    if (fltTy.isBF16())
      return llvm::APFloat(llvm::APFloat::BFloat(), val);
    if (fltTy.isF32())
      return llvm::APFloat(llvm::APFloat::IEEEsingle(), val);
    if (fltTy.isF64())
      return llvm::APFloat(llvm::APFloat::IEEEdouble(), val);
    if (fltTy.isF80())
      return llvm::APFloat(llvm::APFloat::x87DoubleExtended(), val);
    if (fltTy.isF128())
      return llvm::APFloat(llvm::APFloat::IEEEquad(), val);
    llvm_unreachable("unhandled MLIR floating-point type");
  };
  return createRealConstant(loc, fltTy, apf());
}

mlir::Value fir::FirOpBuilder::createRealConstant(mlir::Location loc,
                                                  mlir::Type fltTy,
                                                  const llvm::APFloat &value) {
  if (fltTy.isa<mlir::FloatType>()) {
    auto attr = getFloatAttr(fltTy, value);
    return create<mlir::arith::ConstantOp>(loc, fltTy, attr);
  }
  llvm_unreachable("should use builtin floating-point type");
}

static llvm::SmallVector<mlir::Value>
elideExtentsAlreadyInType(mlir::Type type, mlir::ValueRange shape) {
  auto arrTy = type.dyn_cast<fir::SequenceType>();
  if (shape.empty() || !arrTy)
    return {};
  // elide the constant dimensions before construction
  assert(shape.size() == arrTy.getDimension());
  llvm::SmallVector<mlir::Value> dynamicShape;
  auto typeShape = arrTy.getShape();
  for (unsigned i = 0, end = arrTy.getDimension(); i < end; ++i)
    if (typeShape[i] == fir::SequenceType::getUnknownExtent())
      dynamicShape.push_back(shape[i]);
  return dynamicShape;
}

static llvm::SmallVector<mlir::Value>
elideLengthsAlreadyInType(mlir::Type type, mlir::ValueRange lenParams) {
  if (lenParams.empty())
    return {};
  if (auto arrTy = type.dyn_cast<fir::SequenceType>())
    type = arrTy.getEleTy();
  if (fir::hasDynamicSize(type))
    return lenParams;
  return {};
}

/// Allocate a local variable.
/// A local variable ought to have a name in the source code.
mlir::Value fir::FirOpBuilder::allocateLocal(
    mlir::Location loc, mlir::Type ty, llvm::StringRef uniqName,
    llvm::StringRef name, bool pinned, llvm::ArrayRef<mlir::Value> shape,
    llvm::ArrayRef<mlir::Value> lenParams, bool asTarget) {
  // Convert the shape extents to `index`, as needed.
  llvm::SmallVector<mlir::Value> indices;
  llvm::SmallVector<mlir::Value> elidedShape =
      elideExtentsAlreadyInType(ty, shape);
  llvm::SmallVector<mlir::Value> elidedLenParams =
      elideLengthsAlreadyInType(ty, lenParams);
  auto idxTy = getIndexType();
  for (mlir::Value sh : elidedShape)
    indices.push_back(createConvert(loc, idxTy, sh));
  // Add a target attribute, if needed.
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  if (asTarget)
    attrs.emplace_back(
        mlir::StringAttr::get(getContext(), fir::getTargetAttrName()),
        getUnitAttr());
  // Create the local variable.
  if (name.empty()) {
    if (uniqName.empty())
      return create<fir::AllocaOp>(loc, ty, pinned, elidedLenParams, indices,
                                   attrs);
    return create<fir::AllocaOp>(loc, ty, uniqName, pinned, elidedLenParams,
                                 indices, attrs);
  }
  return create<fir::AllocaOp>(loc, ty, uniqName, name, pinned, elidedLenParams,
                               indices, attrs);
}

mlir::Value fir::FirOpBuilder::allocateLocal(
    mlir::Location loc, mlir::Type ty, llvm::StringRef uniqName,
    llvm::StringRef name, llvm::ArrayRef<mlir::Value> shape,
    llvm::ArrayRef<mlir::Value> lenParams, bool asTarget) {
  return allocateLocal(loc, ty, uniqName, name, /*pinned=*/false, shape,
                       lenParams, asTarget);
}

/// Get the block for adding Allocas.
mlir::Block *fir::FirOpBuilder::getAllocaBlock() {
  auto iface =
      getRegion().getParentOfType<mlir::omp::OutlineableOpenMPOpInterface>();
  return iface ? iface.getAllocaBlock() : getEntryBlock();
}

/// Create a temporary variable on the stack. Anonymous temporaries have no
/// `name` value. Temporaries do not require a uniqued name.
mlir::Value
fir::FirOpBuilder::createTemporary(mlir::Location loc, mlir::Type type,
                                   llvm::StringRef name, mlir::ValueRange shape,
                                   mlir::ValueRange lenParams,
                                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  llvm::SmallVector<mlir::Value> dynamicShape =
      elideExtentsAlreadyInType(type, shape);
  llvm::SmallVector<mlir::Value> dynamicLength =
      elideLengthsAlreadyInType(type, lenParams);
  InsertPoint insPt;
  const bool hoistAlloc = dynamicShape.empty() && dynamicLength.empty();
  if (hoistAlloc) {
    insPt = saveInsertionPoint();
    setInsertionPointToStart(getAllocaBlock());
  }

  // If the alloca is inside an OpenMP Op which will be outlined then pin the
  // alloca here.
  const bool pinned =
      getRegion().getParentOfType<mlir::omp::OutlineableOpenMPOpInterface>();
  assert(!type.isa<fir::ReferenceType>() && "cannot be a reference");
  auto ae =
      create<fir::AllocaOp>(loc, type, /*unique_name=*/llvm::StringRef{}, name,
                            pinned, dynamicLength, dynamicShape, attrs);
  if (hoistAlloc)
    restoreInsertionPoint(insPt);
  return ae;
}

/// Create a global variable in the (read-only) data section. A global variable
/// must have a unique name to identify and reference it.
fir::GlobalOp fir::FirOpBuilder::createGlobal(mlir::Location loc,
                                              mlir::Type type,
                                              llvm::StringRef name,
                                              mlir::StringAttr linkage,
                                              mlir::Attribute value,
                                              bool isConst, bool isTarget) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody(), module.getBody()->end());
  auto glob =
      create<fir::GlobalOp>(loc, name, isConst, isTarget, type, value, linkage);
  restoreInsertionPoint(insertPt);
  return glob;
}

fir::GlobalOp fir::FirOpBuilder::createGlobal(
    mlir::Location loc, mlir::Type type, llvm::StringRef name, bool isConst,
    bool isTarget, std::function<void(FirOpBuilder &)> bodyBuilder,
    mlir::StringAttr linkage) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody(), module.getBody()->end());
  auto glob = create<fir::GlobalOp>(loc, name, isConst, isTarget, type,
                                    mlir::Attribute{}, linkage);
  auto &region = glob.getRegion();
  region.push_back(new mlir::Block);
  auto &block = glob.getRegion().back();
  setInsertionPointToStart(&block);
  bodyBuilder(*this);
  restoreInsertionPoint(insertPt);
  return glob;
}

fir::DispatchTableOp fir::FirOpBuilder::createDispatchTableOp(
    mlir::Location loc, llvm::StringRef name, llvm::StringRef parentName) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto dt = module.lookupSymbol<fir::DispatchTableOp>(name))
    return dt;
  setInsertionPoint(module.getBody(), module.getBody()->end());
  auto dt = create<fir::DispatchTableOp>(loc, name, mlir::Type{}, parentName);
  restoreInsertionPoint(insertPt);
  return dt;
}

mlir::Value
fir::FirOpBuilder::convertWithSemantics(mlir::Location loc, mlir::Type toTy,
                                        mlir::Value val,
                                        bool allowCharacterConversion) {
  assert(toTy && "store location must be typed");
  auto fromTy = val.getType();
  if (fromTy == toTy)
    return val;
  fir::factory::Complex helper{*this, loc};
  if ((fir::isa_real(fromTy) || fir::isa_integer(fromTy)) &&
      fir::isa_complex(toTy)) {
    // imaginary part is zero
    auto eleTy = helper.getComplexPartType(toTy);
    auto cast = createConvert(loc, eleTy, val);
    llvm::APFloat zero{
        kindMap.getFloatSemantics(toTy.cast<fir::ComplexType>().getFKind()), 0};
    auto imag = createRealConstant(loc, eleTy, zero);
    return helper.createComplex(toTy, cast, imag);
  }
  if (fir::isa_complex(fromTy) &&
      (fir::isa_integer(toTy) || fir::isa_real(toTy))) {
    // drop the imaginary part
    auto rp = helper.extractComplexPart(val, /*isImagPart=*/false);
    return createConvert(loc, toTy, rp);
  }
  if (allowCharacterConversion) {
    if (fromTy.isa<fir::BoxCharType>()) {
      // Extract the address of the character string and pass it
      fir::factory::CharacterExprHelper charHelper{*this, loc};
      std::pair<mlir::Value, mlir::Value> unboxchar =
          charHelper.createUnboxChar(val);
      return createConvert(loc, toTy, unboxchar.first);
    }
    if (auto boxType = toTy.dyn_cast<fir::BoxCharType>()) {
      // Extract the address of the actual argument and create a boxed
      // character value with an undefined length
      // TODO: We should really calculate the total size of the actual
      // argument in characters and use it as the length of the string
      auto refType = getRefType(boxType.getEleTy());
      mlir::Value charBase = createConvert(loc, refType, val);
      mlir::Value unknownLen = create<fir::UndefOp>(loc, getIndexType());
      fir::factory::CharacterExprHelper charHelper{*this, loc};
      return charHelper.createEmboxChar(charBase, unknownLen);
    }
  }
  if (fir::isa_ref_type(toTy) && fir::isa_box_type(fromTy)) {
    // Call is expecting a raw data pointer, not a box. Get the data pointer out
    // of the box and pass that.
    assert((fir::unwrapRefType(toTy) ==
                fir::unwrapRefType(fir::unwrapPassByRefType(fromTy)) &&
            "element types expected to match"));
    return create<fir::BoxAddrOp>(loc, toTy, val);
  }

  if (fir::isPolymorphicType(fromTy) &&
      (fir::isAllocatableType(fromTy) || fir::isPointerType(fromTy)) &&
      fir::isPolymorphicType(toTy)) {
    return create<fir::ReboxOp>(loc, toTy, val, mlir::Value{},
                                /*slice=*/mlir::Value{});
  }

  return createConvert(loc, toTy, val);
}

mlir::Value fir::FirOpBuilder::createConvert(mlir::Location loc,
                                             mlir::Type toTy, mlir::Value val) {
  if (val.getType() != toTy) {
    assert(!fir::isa_derived(toTy));
    return create<fir::ConvertOp>(loc, toTy, val);
  }
  return val;
}

void fir::FirOpBuilder::createStoreWithConvert(mlir::Location loc,
                                               mlir::Value val,
                                               mlir::Value addr) {
  mlir::Value cast =
      createConvert(loc, fir::unwrapRefType(addr.getType()), val);
  create<fir::StoreOp>(loc, cast, addr);
}

fir::StringLitOp fir::FirOpBuilder::createStringLitOp(mlir::Location loc,
                                                      llvm::StringRef data) {
  auto type = fir::CharacterType::get(getContext(), 1, data.size());
  auto strAttr = mlir::StringAttr::get(getContext(), data);
  auto valTag = mlir::StringAttr::get(getContext(), fir::StringLitOp::value());
  mlir::NamedAttribute dataAttr(valTag, strAttr);
  auto sizeTag = mlir::StringAttr::get(getContext(), fir::StringLitOp::size());
  mlir::NamedAttribute sizeAttr(sizeTag, getI64IntegerAttr(data.size()));
  llvm::SmallVector<mlir::NamedAttribute> attrs{dataAttr, sizeAttr};
  return create<fir::StringLitOp>(loc, llvm::ArrayRef<mlir::Type>{type},
                                  llvm::None, attrs);
}

mlir::Value fir::FirOpBuilder::genShape(mlir::Location loc,
                                        llvm::ArrayRef<mlir::Value> exts) {
  auto shapeType = fir::ShapeType::get(getContext(), exts.size());
  return create<fir::ShapeOp>(loc, shapeType, exts);
}

mlir::Value fir::FirOpBuilder::genShape(mlir::Location loc,
                                        llvm::ArrayRef<mlir::Value> shift,
                                        llvm::ArrayRef<mlir::Value> exts) {
  auto shapeType = fir::ShapeShiftType::get(getContext(), exts.size());
  llvm::SmallVector<mlir::Value> shapeArgs;
  auto idxTy = getIndexType();
  for (auto [lbnd, ext] : llvm::zip(shift, exts)) {
    auto lb = createConvert(loc, idxTy, lbnd);
    shapeArgs.push_back(lb);
    shapeArgs.push_back(ext);
  }
  return create<fir::ShapeShiftOp>(loc, shapeType, shapeArgs);
}

mlir::Value fir::FirOpBuilder::genShape(mlir::Location loc,
                                        const fir::AbstractArrayBox &arr) {
  if (arr.lboundsAllOne())
    return genShape(loc, arr.getExtents());
  return genShape(loc, arr.getLBounds(), arr.getExtents());
}

mlir::Value fir::FirOpBuilder::genShift(mlir::Location loc,
                                        llvm::ArrayRef<mlir::Value> shift) {
  auto shiftType = fir::ShiftType::get(getContext(), shift.size());
  return create<fir::ShiftOp>(loc, shiftType, shift);
}

mlir::Value fir::FirOpBuilder::createShape(mlir::Location loc,
                                           const fir::ExtendedValue &exv) {
  return exv.match(
      [&](const fir::ArrayBoxValue &box) { return genShape(loc, box); },
      [&](const fir::CharArrayBoxValue &box) { return genShape(loc, box); },
      [&](const fir::BoxValue &box) -> mlir::Value {
        if (!box.getLBounds().empty()) {
          auto shiftType =
              fir::ShiftType::get(getContext(), box.getLBounds().size());
          return create<fir::ShiftOp>(loc, shiftType, box.getLBounds());
        }
        return {};
      },
      [&](const fir::MutableBoxValue &) -> mlir::Value {
        // MutableBoxValue must be read into another category to work with them
        // outside of allocation/assignment contexts.
        fir::emitFatalError(loc, "createShape on MutableBoxValue");
      },
      [&](auto) -> mlir::Value { fir::emitFatalError(loc, "not an array"); });
}

mlir::Value fir::FirOpBuilder::createSlice(mlir::Location loc,
                                           const fir::ExtendedValue &exv,
                                           mlir::ValueRange triples,
                                           mlir::ValueRange path) {
  if (triples.empty()) {
    // If there is no slicing by triple notation, then take the whole array.
    auto fullShape = [&](const llvm::ArrayRef<mlir::Value> lbounds,
                         llvm::ArrayRef<mlir::Value> extents) -> mlir::Value {
      llvm::SmallVector<mlir::Value> trips;
      auto idxTy = getIndexType();
      auto one = createIntegerConstant(loc, idxTy, 1);
      if (lbounds.empty()) {
        for (auto v : extents) {
          trips.push_back(one);
          trips.push_back(v);
          trips.push_back(one);
        }
        return create<fir::SliceOp>(loc, trips, path);
      }
      for (auto [lbnd, extent] : llvm::zip(lbounds, extents)) {
        auto lb = createConvert(loc, idxTy, lbnd);
        auto ext = createConvert(loc, idxTy, extent);
        auto shift = create<mlir::arith::SubIOp>(loc, lb, one);
        auto ub = create<mlir::arith::AddIOp>(loc, ext, shift);
        trips.push_back(lb);
        trips.push_back(ub);
        trips.push_back(one);
      }
      return create<fir::SliceOp>(loc, trips, path);
    };
    return exv.match(
        [&](const fir::ArrayBoxValue &box) {
          return fullShape(box.getLBounds(), box.getExtents());
        },
        [&](const fir::CharArrayBoxValue &box) {
          return fullShape(box.getLBounds(), box.getExtents());
        },
        [&](const fir::BoxValue &box) {
          auto extents = fir::factory::readExtents(*this, loc, box);
          return fullShape(box.getLBounds(), extents);
        },
        [&](const fir::MutableBoxValue &) -> mlir::Value {
          // MutableBoxValue must be read into another category to work with
          // them outside of allocation/assignment contexts.
          fir::emitFatalError(loc, "createSlice on MutableBoxValue");
        },
        [&](auto) -> mlir::Value { fir::emitFatalError(loc, "not an array"); });
  }
  return create<fir::SliceOp>(loc, triples, path);
}

mlir::Value fir::FirOpBuilder::createBox(mlir::Location loc,
                                         const fir::ExtendedValue &exv,
                                         bool isPolymorphic) {
  mlir::Value itemAddr = fir::getBase(exv);
  if (itemAddr.getType().isa<fir::BaseBoxType>())
    return itemAddr;
  auto elementType = fir::dyn_cast_ptrEleTy(itemAddr.getType());
  if (!elementType) {
    mlir::emitError(loc, "internal: expected a memory reference type ")
        << itemAddr.getType();
    llvm_unreachable("not a memory reference type");
  }
  mlir::Type boxTy = fir::BoxType::get(elementType);
  mlir::Value tdesc;
  if (isPolymorphic)
    boxTy = fir::ClassType::get(elementType);
  return exv.match(
      [&](const fir::ArrayBoxValue &box) -> mlir::Value {
        mlir::Value empty;
        mlir::ValueRange emptyRange;
        mlir::Value s = createShape(loc, exv);
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, s, /*slice=*/empty,
                                    /*typeparams=*/emptyRange,
                                    isPolymorphic ? box.getTdesc() : tdesc);
      },
      [&](const fir::CharArrayBoxValue &box) -> mlir::Value {
        mlir::Value s = createShape(loc, exv);
        if (fir::factory::CharacterExprHelper::hasConstantLengthInType(exv))
          return create<fir::EmboxOp>(loc, boxTy, itemAddr, s);

        mlir::Value emptySlice;
        llvm::SmallVector<mlir::Value> lenParams{box.getLen()};
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, s, emptySlice,
                                    lenParams);
      },
      [&](const fir::CharBoxValue &box) -> mlir::Value {
        if (fir::factory::CharacterExprHelper::hasConstantLengthInType(exv))
          return create<fir::EmboxOp>(loc, boxTy, itemAddr);
        mlir::Value emptyShape, emptySlice;
        llvm::SmallVector<mlir::Value> lenParams{box.getLen()};
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, emptyShape,
                                    emptySlice, lenParams);
      },
      [&](const fir::MutableBoxValue &x) -> mlir::Value {
        return create<fir::LoadOp>(
            loc, fir::factory::getMutableIRBox(*this, loc, x));
      },
      [&](const fir::PolymorphicValue &p) -> mlir::Value {
        mlir::Value empty;
        mlir::ValueRange emptyRange;
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, empty, empty,
                                    emptyRange,
                                    isPolymorphic ? p.getTdesc() : tdesc);
      },
      [&](const auto &) -> mlir::Value {
        mlir::Value empty;
        mlir::ValueRange emptyRange;
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, empty, empty,
                                    emptyRange, tdesc);
      });
}

void fir::FirOpBuilder::dumpFunc() { getFunction().dump(); }

static mlir::Value
genNullPointerComparison(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value addr,
                         mlir::arith::CmpIPredicate condition) {
  auto intPtrTy = builder.getIntPtrType();
  auto ptrToInt = builder.createConvert(loc, intPtrTy, addr);
  auto c0 = builder.createIntegerConstant(loc, intPtrTy, 0);
  return builder.create<mlir::arith::CmpIOp>(loc, condition, ptrToInt, c0);
}

mlir::Value fir::FirOpBuilder::genIsNotNullAddr(mlir::Location loc,
                                                mlir::Value addr) {
  return genNullPointerComparison(*this, loc, addr,
                                  mlir::arith::CmpIPredicate::ne);
}

mlir::Value fir::FirOpBuilder::genIsNullAddr(mlir::Location loc,
                                             mlir::Value addr) {
  return genNullPointerComparison(*this, loc, addr,
                                  mlir::arith::CmpIPredicate::eq);
}

mlir::Value fir::FirOpBuilder::genExtentFromTriplet(mlir::Location loc,
                                                    mlir::Value lb,
                                                    mlir::Value ub,
                                                    mlir::Value step,
                                                    mlir::Type type) {
  auto zero = createIntegerConstant(loc, type, 0);
  lb = createConvert(loc, type, lb);
  ub = createConvert(loc, type, ub);
  step = createConvert(loc, type, step);
  auto diff = create<mlir::arith::SubIOp>(loc, ub, lb);
  auto add = create<mlir::arith::AddIOp>(loc, diff, step);
  auto div = create<mlir::arith::DivSIOp>(loc, add, step);
  auto cmp = create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt,
                                         div, zero);
  return create<mlir::arith::SelectOp>(loc, cmp, div, zero);
}

void fir::FirOpBuilder::setCommonAttributes(mlir::Operation *op) const {
  auto fmi = mlir::dyn_cast<mlir::arith::ArithFastMathInterface>(*op);
  if (!fmi)
    return;
  // TODO: use fmi.setFastMathFlagsAttr() after D137114 is merged.
  //       For now set the attribute by the name.
  llvm::StringRef arithFMFAttrName = fmi.getFastMathAttrName();
  if (fastMathFlags != mlir::arith::FastMathFlags::none)
    op->setAttr(arithFMFAttrName, mlir::arith::FastMathFlagsAttr::get(
                                      op->getContext(), fastMathFlags));
}

void fir::FirOpBuilder::setFastMathFlags(
    Fortran::common::MathOptionsBase options) {
  mlir::arith::FastMathFlags arithFMF{};
  if (options.getFPContractEnabled()) {
    arithFMF = arithFMF | mlir::arith::FastMathFlags::contract;
  }
  if (options.getNoHonorInfs()) {
    arithFMF = arithFMF | mlir::arith::FastMathFlags::ninf;
  }
  if (options.getNoHonorNaNs()) {
    arithFMF = arithFMF | mlir::arith::FastMathFlags::nnan;
  }
  if (options.getApproxFunc()) {
    arithFMF = arithFMF | mlir::arith::FastMathFlags::afn;
  }
  if (options.getNoSignedZeros()) {
    arithFMF = arithFMF | mlir::arith::FastMathFlags::nsz;
  }
  if (options.getAssociativeMath()) {
    arithFMF = arithFMF | mlir::arith::FastMathFlags::reassoc;
  }
  if (options.getReciprocalMath()) {
    arithFMF = arithFMF | mlir::arith::FastMathFlags::arcp;
  }
  setFastMathFlags(arithFMF);
}

//===--------------------------------------------------------------------===//
// ExtendedValue inquiry helper implementation
//===--------------------------------------------------------------------===//

mlir::Value fir::factory::readCharLen(fir::FirOpBuilder &builder,
                                      mlir::Location loc,
                                      const fir::ExtendedValue &box) {
  return box.match(
      [&](const fir::CharBoxValue &x) -> mlir::Value { return x.getLen(); },
      [&](const fir::CharArrayBoxValue &x) -> mlir::Value {
        return x.getLen();
      },
      [&](const fir::BoxValue &x) -> mlir::Value {
        assert(x.isCharacter());
        if (!x.getExplicitParameters().empty())
          return x.getExplicitParameters()[0];
        return fir::factory::CharacterExprHelper{builder, loc}
            .readLengthFromBox(x.getAddr());
      },
      [&](const fir::MutableBoxValue &x) -> mlir::Value {
        return readCharLen(builder, loc,
                           fir::factory::genMutableBoxRead(builder, loc, x));
      },
      [&](const auto &) -> mlir::Value {
        fir::emitFatalError(
            loc, "Character length inquiry on a non-character entity");
      });
}

mlir::Value fir::factory::readExtent(fir::FirOpBuilder &builder,
                                     mlir::Location loc,
                                     const fir::ExtendedValue &box,
                                     unsigned dim) {
  assert(box.rank() > dim);
  return box.match(
      [&](const fir::ArrayBoxValue &x) -> mlir::Value {
        return x.getExtents()[dim];
      },
      [&](const fir::CharArrayBoxValue &x) -> mlir::Value {
        return x.getExtents()[dim];
      },
      [&](const fir::BoxValue &x) -> mlir::Value {
        if (!x.getExplicitExtents().empty())
          return x.getExplicitExtents()[dim];
        auto idxTy = builder.getIndexType();
        auto dimVal = builder.createIntegerConstant(loc, idxTy, dim);
        return builder
            .create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, x.getAddr(),
                                    dimVal)
            .getResult(1);
      },
      [&](const fir::MutableBoxValue &x) -> mlir::Value {
        return readExtent(builder, loc,
                          fir::factory::genMutableBoxRead(builder, loc, x),
                          dim);
      },
      [&](const auto &) -> mlir::Value {
        fir::emitFatalError(loc, "extent inquiry on scalar");
      });
}

mlir::Value fir::factory::readLowerBound(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         const fir::ExtendedValue &box,
                                         unsigned dim,
                                         mlir::Value defaultValue) {
  assert(box.rank() > dim);
  auto lb = box.match(
      [&](const fir::ArrayBoxValue &x) -> mlir::Value {
        return x.getLBounds().empty() ? mlir::Value{} : x.getLBounds()[dim];
      },
      [&](const fir::CharArrayBoxValue &x) -> mlir::Value {
        return x.getLBounds().empty() ? mlir::Value{} : x.getLBounds()[dim];
      },
      [&](const fir::BoxValue &x) -> mlir::Value {
        return x.getLBounds().empty() ? mlir::Value{} : x.getLBounds()[dim];
      },
      [&](const fir::MutableBoxValue &x) -> mlir::Value {
        return readLowerBound(builder, loc,
                              fir::factory::genMutableBoxRead(builder, loc, x),
                              dim, defaultValue);
      },
      [&](const auto &) -> mlir::Value {
        fir::emitFatalError(loc, "lower bound inquiry on scalar");
      });
  if (lb)
    return lb;
  return defaultValue;
}

llvm::SmallVector<mlir::Value>
fir::factory::readExtents(fir::FirOpBuilder &builder, mlir::Location loc,
                          const fir::BoxValue &box) {
  llvm::SmallVector<mlir::Value> result;
  auto explicitExtents = box.getExplicitExtents();
  if (!explicitExtents.empty()) {
    result.append(explicitExtents.begin(), explicitExtents.end());
    return result;
  }
  auto rank = box.rank();
  auto idxTy = builder.getIndexType();
  for (decltype(rank) dim = 0; dim < rank; ++dim) {
    auto dimVal = builder.createIntegerConstant(loc, idxTy, dim);
    auto dimInfo = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                  box.getAddr(), dimVal);
    result.emplace_back(dimInfo.getResult(1));
  }
  return result;
}

llvm::SmallVector<mlir::Value>
fir::factory::getExtents(mlir::Location loc, fir::FirOpBuilder &builder,
                         const fir::ExtendedValue &box) {
  return box.match(
      [&](const fir::ArrayBoxValue &x) -> llvm::SmallVector<mlir::Value> {
        return {x.getExtents().begin(), x.getExtents().end()};
      },
      [&](const fir::CharArrayBoxValue &x) -> llvm::SmallVector<mlir::Value> {
        return {x.getExtents().begin(), x.getExtents().end()};
      },
      [&](const fir::BoxValue &x) -> llvm::SmallVector<mlir::Value> {
        return fir::factory::readExtents(builder, loc, x);
      },
      [&](const fir::MutableBoxValue &x) -> llvm::SmallVector<mlir::Value> {
        auto load = fir::factory::genMutableBoxRead(builder, loc, x);
        return fir::factory::getExtents(loc, builder, load);
      },
      [&](const auto &) -> llvm::SmallVector<mlir::Value> { return {}; });
}

fir::ExtendedValue fir::factory::readBoxValue(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              const fir::BoxValue &box) {
  assert(!box.isUnlimitedPolymorphic() && !box.hasAssumedRank() &&
         "cannot read unlimited polymorphic or assumed rank fir.box");
  auto addr =
      builder.create<fir::BoxAddrOp>(loc, box.getMemTy(), box.getAddr());
  if (box.isCharacter()) {
    auto len = fir::factory::readCharLen(builder, loc, box);
    if (box.rank() == 0)
      return fir::CharBoxValue(addr, len);
    return fir::CharArrayBoxValue(addr, len,
                                  fir::factory::readExtents(builder, loc, box),
                                  box.getLBounds());
  }
  if (box.isDerivedWithLenParameters())
    TODO(loc, "read fir.box with length parameters");
  if (box.rank() == 0)
    return addr;
  return fir::ArrayBoxValue(addr, fir::factory::readExtents(builder, loc, box),
                            box.getLBounds());
}

llvm::SmallVector<mlir::Value>
fir::factory::getNonDefaultLowerBounds(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       const fir::ExtendedValue &exv) {
  return exv.match(
      [&](const fir::ArrayBoxValue &array) -> llvm::SmallVector<mlir::Value> {
        return {array.getLBounds().begin(), array.getLBounds().end()};
      },
      [&](const fir::CharArrayBoxValue &array)
          -> llvm::SmallVector<mlir::Value> {
        return {array.getLBounds().begin(), array.getLBounds().end()};
      },
      [&](const fir::BoxValue &box) -> llvm::SmallVector<mlir::Value> {
        return {box.getLBounds().begin(), box.getLBounds().end()};
      },
      [&](const fir::MutableBoxValue &box) -> llvm::SmallVector<mlir::Value> {
        auto load = fir::factory::genMutableBoxRead(builder, loc, box);
        return fir::factory::getNonDefaultLowerBounds(builder, loc, load);
      },
      [&](const auto &) -> llvm::SmallVector<mlir::Value> { return {}; });
}

llvm::SmallVector<mlir::Value>
fir::factory::getNonDeferredLenParams(const fir::ExtendedValue &exv) {
  return exv.match(
      [&](const fir::CharArrayBoxValue &character)
          -> llvm::SmallVector<mlir::Value> { return {character.getLen()}; },
      [&](const fir::CharBoxValue &character)
          -> llvm::SmallVector<mlir::Value> { return {character.getLen()}; },
      [&](const fir::MutableBoxValue &box) -> llvm::SmallVector<mlir::Value> {
        return {box.nonDeferredLenParams().begin(),
                box.nonDeferredLenParams().end()};
      },
      [&](const fir::BoxValue &box) -> llvm::SmallVector<mlir::Value> {
        return {box.getExplicitParameters().begin(),
                box.getExplicitParameters().end()};
      },
      [&](const auto &) -> llvm::SmallVector<mlir::Value> { return {}; });
}

// If valTy is a box type, then we need to extract the type parameters from
// the box value.
static llvm::SmallVector<mlir::Value> getFromBox(mlir::Location loc,
                                                 fir::FirOpBuilder &builder,
                                                 mlir::Type valTy,
                                                 mlir::Value boxVal) {
  if (auto boxTy = valTy.dyn_cast<fir::BaseBoxType>()) {
    auto eleTy = fir::unwrapAllRefAndSeqType(boxTy.getEleTy());
    if (auto recTy = eleTy.dyn_cast<fir::RecordType>()) {
      if (recTy.getNumLenParams() > 0) {
        // Walk each type parameter in the record and get the value.
        TODO(loc, "generate code to get LEN type parameters");
      }
    } else if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
      if (charTy.hasDynamicLen()) {
        auto idxTy = builder.getIndexType();
        auto eleSz = builder.create<fir::BoxEleSizeOp>(loc, idxTy, boxVal);
        auto kindBytes =
            builder.getKindMap().getCharacterBitsize(charTy.getFKind()) / 8;
        mlir::Value charSz =
            builder.createIntegerConstant(loc, idxTy, kindBytes);
        mlir::Value len =
            builder.create<mlir::arith::DivSIOp>(loc, eleSz, charSz);
        return {len};
      }
    }
  }
  return {};
}

// fir::getTypeParams() will get the type parameters from the extended value.
// When the extended value is a BoxValue or MutableBoxValue, it may be necessary
// to generate code, so this factory function handles those cases.
// TODO: fix the inverted type tests, etc.
llvm::SmallVector<mlir::Value>
fir::factory::getTypeParams(mlir::Location loc, fir::FirOpBuilder &builder,
                            const fir::ExtendedValue &exv) {
  auto handleBoxed = [&](const auto &box) -> llvm::SmallVector<mlir::Value> {
    if (box.isCharacter())
      return {fir::factory::readCharLen(builder, loc, exv)};
    if (box.isDerivedWithLenParameters()) {
      // This should generate code to read the type parameters from the box.
      // This requires some consideration however as MutableBoxValues need to be
      // in a sane state to be provide the correct values.
      TODO(loc, "derived type with type parameters");
    }
    return {};
  };
  // Intentionally reuse the original code path to get type parameters for the
  // cases that were supported rather than introduce a new path.
  return exv.match(
      [&](const fir::BoxValue &box) { return handleBoxed(box); },
      [&](const fir::MutableBoxValue &box) { return handleBoxed(box); },
      [&](const auto &) { return fir::getTypeParams(exv); });
}

llvm::SmallVector<mlir::Value>
fir::factory::getTypeParams(mlir::Location loc, fir::FirOpBuilder &builder,
                            fir::ArrayLoadOp load) {
  mlir::Type memTy = load.getMemref().getType();
  if (auto boxTy = memTy.dyn_cast<fir::BaseBoxType>())
    return getFromBox(loc, builder, boxTy, load.getMemref());
  return load.getTypeparams();
}

std::string fir::factory::uniqueCGIdent(llvm::StringRef prefix,
                                        llvm::StringRef name) {
  // For "long" identifiers use a hash value
  if (name.size() > nameLengthHashSize) {
    llvm::MD5 hash;
    hash.update(name);
    llvm::MD5::MD5Result result;
    hash.final(result);
    llvm::SmallString<32> str;
    llvm::MD5::stringifyResult(result, str);
    std::string hashName = prefix.str();
    hashName.append(".").append(str.c_str());
    return fir::NameUniquer::doGenerated(hashName);
  }
  // "Short" identifiers use a reversible hex string
  std::string nm = prefix.str();
  return fir::NameUniquer::doGenerated(
      nm.append(".").append(llvm::toHex(name)));
}

mlir::Value fir::factory::locationToFilename(fir::FirOpBuilder &builder,
                                             mlir::Location loc) {
  if (auto flc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    // must be encoded as asciiz, C string
    auto fn = flc.getFilename().str() + '\0';
    return fir::getBase(createStringLiteral(builder, loc, fn));
  }
  return builder.createNullConstant(loc);
}

mlir::Value fir::factory::locationToLineNo(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Type type) {
  if (auto flc = loc.dyn_cast<mlir::FileLineColLoc>())
    return builder.createIntegerConstant(loc, type, flc.getLine());
  return builder.createIntegerConstant(loc, type, 0);
}

fir::ExtendedValue fir::factory::createStringLiteral(fir::FirOpBuilder &builder,
                                                     mlir::Location loc,
                                                     llvm::StringRef str) {
  std::string globalName = fir::factory::uniqueCGIdent("cl", str);
  auto type = fir::CharacterType::get(builder.getContext(), 1, str.size());
  auto global = builder.getNamedGlobal(globalName);
  if (!global)
    global = builder.createGlobalConstant(
        loc, type, globalName,
        [&](fir::FirOpBuilder &builder) {
          auto stringLitOp = builder.createStringLitOp(loc, str);
          builder.create<fir::HasValueOp>(loc, stringLitOp);
        },
        builder.createLinkOnceLinkage());
  auto addr = builder.create<fir::AddrOfOp>(loc, global.resultType(),
                                            global.getSymbol());
  auto len = builder.createIntegerConstant(
      loc, builder.getCharacterLengthType(), str.size());
  return fir::CharBoxValue{addr, len};
}

llvm::SmallVector<mlir::Value>
fir::factory::createExtents(fir::FirOpBuilder &builder, mlir::Location loc,
                            fir::SequenceType seqTy) {
  llvm::SmallVector<mlir::Value> extents;
  auto idxTy = builder.getIndexType();
  for (auto ext : seqTy.getShape())
    extents.emplace_back(
        ext == fir::SequenceType::getUnknownExtent()
            ? builder.create<fir::UndefOp>(loc, idxTy).getResult()
            : builder.createIntegerConstant(loc, idxTy, ext));
  return extents;
}

// FIXME: This needs some work. To correctly determine the extended value of a
// component, one needs the base object, its type, and its type parameters. (An
// alternative would be to provide an already computed address of the final
// component rather than the base object's address, the point being the result
// will require the address of the final component to create the extended
// value.) One further needs the full path of components being applied. One
// needs to apply type-based expressions to type parameters along this said
// path. (See applyPathToType for a type-only derivation.) Finally, one needs to
// compose the extended value of the terminal component, including all of its
// parameters: array lower bounds expressions, extents, type parameters, etc.
// Any of these properties may be deferred until runtime in Fortran. This
// operation may therefore generate a sizeable block of IR, including calls to
// type-based helper functions, so caching the result of this operation in the
// client would be advised as well.
fir::ExtendedValue fir::factory::componentToExtendedValue(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value component) {
  auto fieldTy = component.getType();
  if (auto ty = fir::dyn_cast_ptrEleTy(fieldTy))
    fieldTy = ty;
  if (fieldTy.isa<fir::BaseBoxType>()) {
    llvm::SmallVector<mlir::Value> nonDeferredTypeParams;
    auto eleTy = fir::unwrapSequenceType(fir::dyn_cast_ptrOrBoxEleTy(fieldTy));
    if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
      auto lenTy = builder.getCharacterLengthType();
      if (charTy.hasConstantLen())
        nonDeferredTypeParams.emplace_back(
            builder.createIntegerConstant(loc, lenTy, charTy.getLen()));
      // TODO: Starting, F2003, the dynamic character length might be dependent
      // on a PDT length parameter. There is no way to make a difference with
      // deferred length here yet.
    }
    if (auto recTy = eleTy.dyn_cast<fir::RecordType>())
      if (recTy.getNumLenParams() > 0)
        TODO(loc, "allocatable and pointer components non deferred length "
                  "parameters");

    return fir::MutableBoxValue(component, nonDeferredTypeParams,
                                /*mutableProperties=*/{});
  }
  llvm::SmallVector<mlir::Value> extents;
  if (auto seqTy = fieldTy.dyn_cast<fir::SequenceType>()) {
    fieldTy = seqTy.getEleTy();
    auto idxTy = builder.getIndexType();
    for (auto extent : seqTy.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        TODO(loc, "array component shape depending on length parameters");
      extents.emplace_back(builder.createIntegerConstant(loc, idxTy, extent));
    }
  }
  if (auto charTy = fieldTy.dyn_cast<fir::CharacterType>()) {
    auto cstLen = charTy.getLen();
    if (cstLen == fir::CharacterType::unknownLen())
      TODO(loc, "get character component length from length type parameters");
    auto len = builder.createIntegerConstant(
        loc, builder.getCharacterLengthType(), cstLen);
    if (!extents.empty())
      return fir::CharArrayBoxValue{component, len, extents};
    return fir::CharBoxValue{component, len};
  }
  if (auto recordTy = fieldTy.dyn_cast<fir::RecordType>())
    if (recordTy.getNumLenParams() != 0)
      TODO(loc,
           "lower component ref that is a derived type with length parameter");
  if (!extents.empty())
    return fir::ArrayBoxValue{component, extents};
  return component;
}

fir::ExtendedValue fir::factory::arrayElementToExtendedValue(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const fir::ExtendedValue &array, mlir::Value element) {
  return array.match(
      [&](const fir::CharBoxValue &cb) -> fir::ExtendedValue {
        return cb.clone(element);
      },
      [&](const fir::CharArrayBoxValue &bv) -> fir::ExtendedValue {
        return bv.cloneElement(element);
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        if (box.isCharacter()) {
          auto len = fir::factory::readCharLen(builder, loc, box);
          return fir::CharBoxValue{element, len};
        }
        if (box.isDerivedWithLenParameters())
          TODO(loc, "get length parameters from derived type BoxValue");
        if (box.isPolymorphic()) {
          mlir::Type tdescType =
              fir::TypeDescType::get(mlir::NoneType::get(builder.getContext()));
          mlir::Value tdesc = builder.create<fir::BoxTypeDescOp>(
              loc, tdescType, fir::getBase(box));
          return fir::PolymorphicValue(element, tdesc);
        }
        return element;
      },
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
        if (box.getTdesc())
          return fir::PolymorphicValue(element, box.getTdesc());
        return element;
      },
      [&](const auto &) -> fir::ExtendedValue { return element; });
}

fir::ExtendedValue fir::factory::arraySectionElementToExtendedValue(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const fir::ExtendedValue &array, mlir::Value element, mlir::Value slice) {
  if (!slice)
    return arrayElementToExtendedValue(builder, loc, array, element);
  auto sliceOp = mlir::dyn_cast_or_null<fir::SliceOp>(slice.getDefiningOp());
  assert(sliceOp && "slice must be a sliceOp");
  if (sliceOp.getFields().empty())
    return arrayElementToExtendedValue(builder, loc, array, element);
  // For F95, using componentToExtendedValue will work, but when PDTs are
  // lowered. It will be required to go down the slice to propagate the length
  // parameters.
  return fir::factory::componentToExtendedValue(builder, loc, element);
}

void fir::factory::genScalarAssignment(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       const fir::ExtendedValue &lhs,
                                       const fir::ExtendedValue &rhs) {
  assert(lhs.rank() == 0 && rhs.rank() == 0 && "must be scalars");
  auto type = fir::unwrapSequenceType(
      fir::unwrapPassByRefType(fir::getBase(lhs).getType()));
  if (type.isa<fir::CharacterType>()) {
    const fir::CharBoxValue *toChar = lhs.getCharBox();
    const fir::CharBoxValue *fromChar = rhs.getCharBox();
    assert(toChar && fromChar);
    fir::factory::CharacterExprHelper helper{builder, loc};
    helper.createAssign(fir::ExtendedValue{*toChar},
                        fir::ExtendedValue{*fromChar});
  } else if (type.isa<fir::RecordType>()) {
    fir::factory::genRecordAssignment(builder, loc, lhs, rhs);
  } else {
    assert(!fir::hasDynamicSize(type));
    auto rhsVal = fir::getBase(rhs);
    if (fir::isa_ref_type(rhsVal.getType()))
      rhsVal = builder.create<fir::LoadOp>(loc, rhsVal);
    mlir::Value lhsAddr = fir::getBase(lhs);
    rhsVal = builder.createConvert(loc, fir::unwrapRefType(lhsAddr.getType()),
                                   rhsVal);
    builder.create<fir::StoreOp>(loc, rhsVal, lhsAddr);
  }
}

static void genComponentByComponentAssignment(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              const fir::ExtendedValue &lhs,
                                              const fir::ExtendedValue &rhs) {
  auto lbaseType = fir::unwrapPassByRefType(fir::getBase(lhs).getType());
  auto lhsType = lbaseType.dyn_cast<fir::RecordType>();
  assert(lhsType && "lhs must be a scalar record type");
  auto rbaseType = fir::unwrapPassByRefType(fir::getBase(rhs).getType());
  auto rhsType = rbaseType.dyn_cast<fir::RecordType>();
  assert(rhsType && "rhs must be a scalar record type");
  auto fieldIndexType = fir::FieldType::get(lhsType.getContext());
  for (auto [lhsPair, rhsPair] :
       llvm::zip(lhsType.getTypeList(), rhsType.getTypeList())) {
    auto &[lFieldName, lFieldTy] = lhsPair;
    auto &[rFieldName, rFieldTy] = rhsPair;
    assert(!fir::hasDynamicSize(lFieldTy) && !fir::hasDynamicSize(rFieldTy));
    mlir::Value rField = builder.create<fir::FieldIndexOp>(
        loc, fieldIndexType, rFieldName, rhsType, fir::getTypeParams(rhs));
    auto rFieldRefType = builder.getRefType(rFieldTy);
    mlir::Value fromCoor = builder.create<fir::CoordinateOp>(
        loc, rFieldRefType, fir::getBase(rhs), rField);
    mlir::Value field = builder.create<fir::FieldIndexOp>(
        loc, fieldIndexType, lFieldName, lhsType, fir::getTypeParams(lhs));
    auto fieldRefType = builder.getRefType(lFieldTy);
    mlir::Value toCoor = builder.create<fir::CoordinateOp>(
        loc, fieldRefType, fir::getBase(lhs), field);
    llvm::Optional<fir::DoLoopOp> outerLoop;
    if (auto sequenceType = lFieldTy.dyn_cast<fir::SequenceType>()) {
      // Create loops to assign array components elements by elements.
      // Note that, since these are components, they either do not overlap,
      // or are the same and exactly overlap. They also have compile time
      // constant shapes.
      mlir::Type idxTy = builder.getIndexType();
      llvm::SmallVector<mlir::Value> indices;
      mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
      mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
      for (auto extent : llvm::reverse(sequenceType.getShape())) {
        // TODO: add zero size test !
        mlir::Value ub = builder.createIntegerConstant(loc, idxTy, extent - 1);
        auto loop = builder.create<fir::DoLoopOp>(loc, zero, ub, one);
        if (!outerLoop)
          outerLoop = loop;
        indices.push_back(loop.getInductionVar());
        builder.setInsertionPointToStart(loop.getBody());
      }
      // Set indices in column-major order.
      std::reverse(indices.begin(), indices.end());
      auto elementRefType = builder.getRefType(sequenceType.getEleTy());
      toCoor = builder.create<fir::CoordinateOp>(loc, elementRefType, toCoor,
                                                 indices);
      fromCoor = builder.create<fir::CoordinateOp>(loc, elementRefType,
                                                   fromCoor, indices);
    }
    if (auto fieldEleTy = fir::unwrapSequenceType(lFieldTy);
        fieldEleTy.isa<fir::BoxType>()) {
      assert(
          fieldEleTy.cast<fir::BoxType>().getEleTy().isa<fir::PointerType>() &&
          "allocatable members require deep copy");
      auto fromPointerValue = builder.create<fir::LoadOp>(loc, fromCoor);
      auto castTo = builder.createConvert(loc, fieldEleTy, fromPointerValue);
      builder.create<fir::StoreOp>(loc, castTo, toCoor);
    } else {
      auto from =
          fir::factory::componentToExtendedValue(builder, loc, fromCoor);
      auto to = fir::factory::componentToExtendedValue(builder, loc, toCoor);
      fir::factory::genScalarAssignment(builder, loc, to, from);
    }
    if (outerLoop)
      builder.setInsertionPointAfter(*outerLoop);
  }
}

/// Can the assignment of this record type be implement with a simple memory
/// copy (it requires no deep copy or user defined assignment of components )?
static bool recordTypeCanBeMemCopied(fir::RecordType recordType) {
  if (fir::hasDynamicSize(recordType))
    return false;
  for (auto [_, fieldType] : recordType.getTypeList()) {
    // Derived type component may have user assignment (so far, we cannot tell
    // in FIR, so assume it is always the case, TODO: get the actual info).
    if (fir::unwrapSequenceType(fieldType).isa<fir::RecordType>())
      return false;
    // Allocatable components need deep copy.
    if (auto boxType = fieldType.dyn_cast<fir::BoxType>())
      if (boxType.getEleTy().isa<fir::HeapType>())
        return false;
  }
  // Constant size components without user defined assignment and pointers can
  // be memcopied.
  return true;
}

void fir::factory::genRecordAssignment(fir::FirOpBuilder &builder,
                                       mlir::Location loc,
                                       const fir::ExtendedValue &lhs,
                                       const fir::ExtendedValue &rhs) {
  assert(lhs.rank() == 0 && rhs.rank() == 0 && "assume scalar assignment");
  auto baseTy = fir::dyn_cast_ptrOrBoxEleTy(fir::getBase(lhs).getType());
  assert(baseTy && "must be a memory type");
  // Box operands may be polymorphic, it is not entirely clear from 10.2.1.3
  // if the assignment is performed on the dynamic of declared type. Use the
  // runtime assuming it is performed on the dynamic type.
  bool hasBoxOperands = fir::getBase(lhs).getType().isa<fir::BoxType>() ||
                        fir::getBase(rhs).getType().isa<fir::BoxType>();
  auto recTy = baseTy.dyn_cast<fir::RecordType>();
  assert(recTy && "must be a record type");
  if (hasBoxOperands || !recordTypeCanBeMemCopied(recTy)) {
    auto to = fir::getBase(builder.createBox(loc, lhs));
    auto from = fir::getBase(builder.createBox(loc, rhs));
    // The runtime entry point may modify the LHS descriptor if it is
    // an allocatable. Allocatable assignment is handle elsewhere in lowering,
    // so just create a fir.ref<fir.box<>> from the fir.box to comply with the
    // runtime interface, but assume the fir.box is unchanged.
    // TODO: does this holds true with polymorphic entities ?
    auto toMutableBox = builder.createTemporary(loc, to.getType());
    builder.create<fir::StoreOp>(loc, to, toMutableBox);
    fir::runtime::genAssign(builder, loc, toMutableBox, from);
    return;
  }
  // Otherwise, the derived type has compile time constant size and for which
  // the component by component assignment can be replaced by a memory copy.
  // Since we do not know the size of the derived type in lowering, do a
  // component by component assignment. Note that a single fir.load/fir.store
  // could be used on "small" record types, but as the type size grows, this
  // leads to issues in LLVM (long compile times, long IR files, and even
  // asserts at some point). Since there is no good size boundary, just always
  // use component by component assignment here.
  genComponentByComponentAssignment(builder, loc, lhs, rhs);
}

mlir::TupleType
fir::factory::getRaggedArrayHeaderType(fir::FirOpBuilder &builder) {
  mlir::IntegerType i64Ty = builder.getIntegerType(64);
  auto arrTy = fir::SequenceType::get(builder.getIntegerType(8), 1);
  auto buffTy = fir::HeapType::get(arrTy);
  auto extTy = fir::SequenceType::get(i64Ty, 1);
  auto shTy = fir::HeapType::get(extTy);
  return mlir::TupleType::get(builder.getContext(), {i64Ty, buffTy, shTy});
}

mlir::Value fir::factory::genLenOfCharacter(
    fir::FirOpBuilder &builder, mlir::Location loc, fir::ArrayLoadOp arrLoad,
    llvm::ArrayRef<mlir::Value> path, llvm::ArrayRef<mlir::Value> substring) {
  llvm::SmallVector<mlir::Value> typeParams(arrLoad.getTypeparams());
  return genLenOfCharacter(builder, loc,
                           arrLoad.getType().cast<fir::SequenceType>(),
                           arrLoad.getMemref(), typeParams, path, substring);
}

mlir::Value fir::factory::genLenOfCharacter(
    fir::FirOpBuilder &builder, mlir::Location loc, fir::SequenceType seqTy,
    mlir::Value memref, llvm::ArrayRef<mlir::Value> typeParams,
    llvm::ArrayRef<mlir::Value> path, llvm::ArrayRef<mlir::Value> substring) {
  auto idxTy = builder.getIndexType();
  auto zero = builder.createIntegerConstant(loc, idxTy, 0);
  auto saturatedDiff = [&](mlir::Value lower, mlir::Value upper) {
    auto diff = builder.create<mlir::arith::SubIOp>(loc, upper, lower);
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    auto size = builder.create<mlir::arith::AddIOp>(loc, diff, one);
    auto cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sgt, size, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, size, zero);
  };
  if (substring.size() == 2) {
    auto upper = builder.createConvert(loc, idxTy, substring.back());
    auto lower = builder.createConvert(loc, idxTy, substring.front());
    return saturatedDiff(lower, upper);
  }
  auto lower = zero;
  if (substring.size() == 1)
    lower = builder.createConvert(loc, idxTy, substring.front());
  auto eleTy = fir::applyPathToType(seqTy, path);
  if (!fir::hasDynamicSize(eleTy)) {
    if (auto charTy = eleTy.dyn_cast<fir::CharacterType>()) {
      // Use LEN from the type.
      return builder.createIntegerConstant(loc, idxTy, charTy.getLen());
    }
    // Do we need to support !fir.array<!fir.char<k,n>>?
    fir::emitFatalError(loc,
                        "application of path did not result in a !fir.char");
  }
  if (fir::isa_box_type(memref.getType())) {
    if (memref.getType().isa<fir::BoxCharType>())
      return builder.create<fir::BoxCharLenOp>(loc, idxTy, memref);
    if (memref.getType().isa<fir::BoxType>())
      return CharacterExprHelper(builder, loc).readLengthFromBox(memref);
    fir::emitFatalError(loc, "memref has wrong type");
  }
  if (typeParams.empty()) {
    fir::emitFatalError(loc, "array_load must have typeparams");
  }
  if (fir::isa_char(seqTy.getEleTy())) {
    assert(typeParams.size() == 1 && "too many typeparams");
    return typeParams.front();
  }
  TODO(loc, "LEN of character must be computed at runtime");
}

mlir::Value fir::factory::createZeroValue(fir::FirOpBuilder &builder,
                                          mlir::Location loc, mlir::Type type) {
  mlir::Type i1 = builder.getIntegerType(1);
  if (type.isa<fir::LogicalType>() || type == i1)
    return builder.createConvert(loc, type, builder.createBool(loc, false));
  if (fir::isa_integer(type))
    return builder.createIntegerConstant(loc, type, 0);
  if (fir::isa_real(type))
    return builder.createRealZeroConstant(loc, type);
  if (fir::isa_complex(type)) {
    fir::factory::Complex complexHelper(builder, loc);
    mlir::Type partType = complexHelper.getComplexPartType(type);
    mlir::Value zeroPart = builder.createRealZeroConstant(loc, partType);
    return complexHelper.createComplex(type, zeroPart, zeroPart);
  }
  fir::emitFatalError(loc, "internal: trying to generate zero value of non "
                           "numeric or logical type");
}

llvm::Optional<std::int64_t> fir::factory::getIntIfConstant(mlir::Value value) {
  if (auto *definingOp = value.getDefiningOp())
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(definingOp))
      if (auto intAttr = cst.getValue().dyn_cast<mlir::IntegerAttr>())
        return intAttr.getInt();
  return {};
}

llvm::Optional<std::int64_t>
fir::factory::getExtentFromTriplet(mlir::Value lb, mlir::Value ub,
                                   mlir::Value stride) {
  std::function<llvm::Optional<std::int64_t>(mlir::Value)> getConstantValue =
      [&](mlir::Value value) -> llvm::Optional<std::int64_t> {
    if (auto valInt = fir::factory::getIntIfConstant(value))
      return valInt;
    auto *definingOp = value.getDefiningOp();
    if (mlir::isa_and_nonnull<fir::ConvertOp>(definingOp)) {
      auto valOp = mlir::dyn_cast<fir::ConvertOp>(definingOp);
      return getConstantValue(valOp.getValue());
    }
    return {};
  };
  if (auto lbInt = getConstantValue(lb)) {
    if (auto ubInt = getConstantValue(ub)) {
      if (auto strideInt = getConstantValue(stride)) {
        if (strideInt.value() != 0) {
          std::int64_t extent =
              1 + (ubInt.value() - lbInt.value()) / strideInt.value();
          if (extent > 0)
            return extent;
        }
      }
    }
  }
  return {};
}

mlir::Value fir::factory::genMaxWithZero(fir::FirOpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value value) {
  mlir::Value zero = builder.createIntegerConstant(loc, value.getType(), 0);
  if (mlir::Operation *definingOp = value.getDefiningOp())
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(definingOp))
      if (auto intAttr = cst.getValue().dyn_cast<mlir::IntegerAttr>())
        return intAttr.getInt() > 0 ? value : zero;
  mlir::Value valueIsGreater = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sgt, value, zero);
  return builder.create<mlir::arith::SelectOp>(loc, valueIsGreater, value,
                                               zero);
}

mlir::Value fir::factory::genCPtrOrCFunptrAddr(fir::FirOpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::Value cPtr,
                                               mlir::Type ty) {
  assert(ty.isa<fir::RecordType>());
  auto recTy = ty.dyn_cast<fir::RecordType>();
  assert(recTy.getTypeList().size() == 1);
  auto fieldName = recTy.getTypeList()[0].first;
  mlir::Type fieldTy = recTy.getTypeList()[0].second;
  auto fieldIndexType = fir::FieldType::get(ty.getContext());
  mlir::Value field =
      builder.create<fir::FieldIndexOp>(loc, fieldIndexType, fieldName, recTy,
                                        /*typeParams=*/mlir::ValueRange{});
  return builder.create<fir::CoordinateOp>(loc, builder.getRefType(fieldTy),
                                           cPtr, field);
}

fir::BoxValue fir::factory::createBoxValue(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           const fir::ExtendedValue &exv) {
  if (auto *boxValue = exv.getBoxOf<fir::BoxValue>())
    return *boxValue;
  mlir::Value box = builder.createBox(loc, exv);
  llvm::SmallVector<mlir::Value> lbounds;
  llvm::SmallVector<mlir::Value> explicitTypeParams;
  exv.match(
      [&](const fir::ArrayBoxValue &box) {
        lbounds.append(box.getLBounds().begin(), box.getLBounds().end());
      },
      [&](const fir::CharArrayBoxValue &box) {
        lbounds.append(box.getLBounds().begin(), box.getLBounds().end());
        explicitTypeParams.emplace_back(box.getLen());
      },
      [&](const fir::CharBoxValue &box) {
        explicitTypeParams.emplace_back(box.getLen());
      },
      [&](const fir::MutableBoxValue &x) {
        explicitTypeParams.append(x.nonDeferredLenParams().begin(),
                                  x.nonDeferredLenParams().end());
      },
      [](const auto &) {});
  return fir::BoxValue(box, lbounds, explicitTypeParams);
}

mlir::Value fir::factory::genCPtrOrCFunptrValue(fir::FirOpBuilder &builder,
                                                mlir::Location loc,
                                                mlir::Value cPtr) {
  mlir::Type cPtrTy = fir::unwrapRefType(cPtr.getType());
  mlir::Value cPtrAddr =
      fir::factory::genCPtrOrCFunptrAddr(builder, loc, cPtr, cPtrTy);
  return builder.create<fir::LoadOp>(loc, cPtrAddr);
}
