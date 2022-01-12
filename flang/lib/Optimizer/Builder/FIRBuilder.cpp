//===-- FIRBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
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

static constexpr std::size_t nameLengthHashSize = 32;

mlir::FuncOp fir::FirOpBuilder::createFunction(mlir::Location loc,
                                               mlir::ModuleOp module,
                                               llvm::StringRef name,
                                               mlir::FunctionType ty) {
  return fir::createFuncOp(loc, module, name, ty);
}

mlir::FuncOp fir::FirOpBuilder::getNamedFunction(mlir::ModuleOp modOp,
                                                 llvm::StringRef name) {
  return modOp.lookupSymbol<mlir::FuncOp>(name);
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
    fir::emitFatalError(UnknownLoc::get(getContext()),
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
  return create<mlir::ConstantOp>(loc, ty, getIntegerAttr(ty, cst));
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
  llvm::for_each(elidedShape, [&](mlir::Value sh) {
    indices.push_back(createConvert(loc, idxTy, sh));
  });
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
  // auto iface =
  //     getRegion().getParentOfType<mlir::omp::OutlineableOpenMPOpInterface>();
  // return iface ? iface.getAllocaBlock() : getEntryBlock();
  return getEntryBlock();
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
fir::GlobalOp
fir::FirOpBuilder::createGlobal(mlir::Location loc, mlir::Type type,
                                llvm::StringRef name, mlir::StringAttr linkage,
                                mlir::Attribute value, bool isConst) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody(), module.getBody()->end());
  auto glob = create<fir::GlobalOp>(loc, name, isConst, type, value, linkage);
  restoreInsertionPoint(insertPt);
  return glob;
}

fir::GlobalOp fir::FirOpBuilder::createGlobal(
    mlir::Location loc, mlir::Type type, llvm::StringRef name, bool isConst,
    std::function<void(FirOpBuilder &)> bodyBuilder, mlir::StringAttr linkage) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody(), module.getBody()->end());
  auto glob = create<fir::GlobalOp>(loc, name, isConst, type, mlir::Attribute{},
                                    linkage);
  auto &region = glob.getRegion();
  region.push_back(new mlir::Block);
  auto &block = glob.getRegion().back();
  setInsertionPointToStart(&block);
  bodyBuilder(*this);
  restoreInsertionPoint(insertPt);
  return glob;
}

mlir::Value fir::FirOpBuilder::convertWithSemantics(mlir::Location loc,
                                                    mlir::Type toTy,
                                                    mlir::Value val) {
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

mlir::Value fir::FirOpBuilder::createBox(mlir::Location loc,
                                         const fir::ExtendedValue &exv) {
  mlir::Value itemAddr = fir::getBase(exv);
  if (itemAddr.getType().isa<fir::BoxType>())
    return itemAddr;
  auto elementType = fir::dyn_cast_ptrEleTy(itemAddr.getType());
  if (!elementType) {
    mlir::emitError(loc, "internal: expected a memory reference type ")
        << itemAddr.getType();
    llvm_unreachable("not a memory reference type");
  }
  mlir::Type boxTy = fir::BoxType::get(elementType);
  return exv.match(
      [&](const fir::ArrayBoxValue &box) -> mlir::Value {
        mlir::Value s = createShape(loc, exv);
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, s);
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
      // UnboxedValue, ProcBoxValue or BoxValue.
      [&](const auto &) -> mlir::Value {
        return create<fir::EmboxOp>(loc, boxTy, itemAddr);
      });
}

static mlir::Value
genNullPointerComparison(fir::FirOpBuilder &builder, mlir::Location loc,
                         mlir::Value addr,
                         mlir::arith::CmpIPredicate condition) {
  auto intPtrTy = builder.getIntPtrType();
  auto ptrToInt = builder.createConvert(loc, intPtrTy, addr);
  auto c0 = builder.createIntegerConstant(loc, intPtrTy, 0);
  return builder.create<mlir::arith::CmpIOp>(loc, condition, ptrToInt, c0);
}

mlir::Value fir::FirOpBuilder::genIsNotNull(mlir::Location loc,
                                            mlir::Value addr) {
  return genNullPointerComparison(*this, loc, addr,
                                  mlir::arith::CmpIPredicate::ne);
}

mlir::Value fir::FirOpBuilder::genIsNull(mlir::Location loc, mlir::Value addr) {
  return genNullPointerComparison(*this, loc, addr,
                                  mlir::arith::CmpIPredicate::eq);
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
      [&](const fir::MutableBoxValue &) -> mlir::Value {
        // MutableBoxValue must be read into another category to work with them
        // outside of allocation/assignment contexts.
        fir::emitFatalError(loc, "readCharLen on MutableBoxValue");
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
        // MutableBoxValue must be read into another category to work with them
        // outside of allocation/assignment contexts.
        fir::emitFatalError(loc, "readExtents on MutableBoxValue");
      },
      [&](const auto &) -> mlir::Value {
        fir::emitFatalError(loc, "extent inquiry on scalar");
      });
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
fir::factory::getExtents(fir::FirOpBuilder &builder, mlir::Location loc,
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
        return fir::factory::getExtents(builder, loc, load);
      },
      [&](const auto &) -> llvm::SmallVector<mlir::Value> { return {}; });
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

mlir::TupleType
fir::factory::getRaggedArrayHeaderType(fir::FirOpBuilder &builder) {
  mlir::IntegerType i64Ty = builder.getIntegerType(64);
  auto arrTy = fir::SequenceType::get(builder.getIntegerType(8), 1);
  auto buffTy = fir::HeapType::get(arrTy);
  auto extTy = fir::SequenceType::get(i64Ty, 1);
  auto shTy = fir::HeapType::get(extTy);
  return mlir::TupleType::get(builder.getContext(), {i64Ty, buffTy, shTy});
}
