//===-- FIRBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/FIRBuilder.h"
#include "SymbolMap.h"
#include "flang/Lower/Allocatable.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/symbol.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"

static llvm::cl::opt<std::size_t>
    nameLengthHashSize("length-to-hash-string-literal",
                       llvm::cl::desc("string literals that exceed this length"
                                      " will use a hash value as their symbol "
                                      "name"),
                       llvm::cl::init(32));

mlir::FuncOp Fortran::lower::FirOpBuilder::createFunction(
    mlir::Location loc, mlir::ModuleOp module, llvm::StringRef name,
    mlir::FunctionType ty) {
  return fir::createFuncOp(loc, module, name, ty);
}

mlir::FuncOp
Fortran::lower::FirOpBuilder::getNamedFunction(mlir::ModuleOp modOp,
                                               llvm::StringRef name) {
  return modOp.lookupSymbol<mlir::FuncOp>(name);
}

fir::GlobalOp
Fortran::lower::FirOpBuilder::getNamedGlobal(mlir::ModuleOp modOp,
                                             llvm::StringRef name) {
  return modOp.lookupSymbol<fir::GlobalOp>(name);
}

mlir::Type Fortran::lower::FirOpBuilder::getRefType(mlir::Type eleTy) {
  assert(!eleTy.isa<fir::ReferenceType>() && "cannot be a reference type");
  return fir::ReferenceType::get(eleTy);
}

mlir::Type Fortran::lower::FirOpBuilder::getVarLenSeqTy(mlir::Type eleTy,
                                                        unsigned rank) {
  fir::SequenceType::Shape shape(rank, fir::SequenceType::getUnknownExtent());
  return fir::SequenceType::get(shape, eleTy);
}

mlir::Value
Fortran::lower::FirOpBuilder::createNullConstant(mlir::Location loc,
                                                 mlir::Type ptrType) {
  auto ty = ptrType ? ptrType : getRefType(getNoneType());
  return create<fir::ZeroOp>(loc, ty);
}

mlir::Value Fortran::lower::FirOpBuilder::createIntegerConstant(
    mlir::Location loc, mlir::Type ty, std::int64_t cst) {
  return create<mlir::ConstantOp>(loc, ty, getIntegerAttr(ty, cst));
}

mlir::Value Fortran::lower::FirOpBuilder::createRealConstant(
    mlir::Location loc, mlir::Type fltTy, llvm::APFloat::integerPart val) {
  auto apf = [&]() -> llvm::APFloat {
    if (auto ty = fltTy.dyn_cast<fir::RealType>()) {
      fltTy = Fortran::lower::convertReal(ty.getContext(), ty.getFKind());
      return llvm::APFloat(kindMap.getFloatSemantics(ty.getFKind()), val);
    } else {
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
    }
  };
  return createRealConstant(loc, fltTy, apf());
}

mlir::Value Fortran::lower::FirOpBuilder::createRealConstant(
    mlir::Location loc, mlir::Type fltTy, const llvm::APFloat &value) {
  if (fltTy.isa<mlir::FloatType>()) {
    auto attr = getFloatAttr(fltTy, value);
    return create<mlir::ConstantOp>(loc, fltTy, attr);
  }
  llvm_unreachable("should use builtin floating-point type");
}

/// Allocate a local variable.
/// A local variable ought to have a name in the source code.
mlir::Value Fortran::lower::FirOpBuilder::allocateLocal(
    mlir::Location loc, mlir::Type ty, llvm::StringRef uniqName,
    llvm::StringRef name, llvm::ArrayRef<mlir::Value> shape,
    llvm::ArrayRef<mlir::Value> lenParams, bool asTarget) {
  // Convert the shape extents to `index`, as needed.
  llvm::SmallVector<mlir::Value> indices;
  auto idxTy = getIndexType();
  llvm::for_each(shape, [&](mlir::Value sh) {
    indices.push_back(createConvert(loc, idxTy, sh));
  });
  // Add a target attribute, if needed.
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  if (asTarget)
    attrs.emplace_back(
        mlir::Identifier::get(fir::getTargetAttrName(), getContext()),
        getUnitAttr());
  // Create the local variable.
  if (name.empty()) {
    if (uniqName.empty())
      return create<fir::AllocaOp>(loc, ty, lenParams, indices, attrs);
    return create<fir::AllocaOp>(loc, ty, uniqName, lenParams, indices, attrs);
  }
  return create<fir::AllocaOp>(loc, ty, uniqName, name, lenParams, indices,
                               attrs);
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

/// Create a temporary variable on the stack. Anonymous temporaries have no
/// `name` value.
mlir::Value Fortran::lower::FirOpBuilder::createTemporary(
    mlir::Location loc, mlir::Type type, llvm::StringRef name,
    mlir::ValueRange shape, mlir::ValueRange lenParams,
    llvm::ArrayRef<mlir::NamedAttribute> attrs, bool isPinned) {
  llvm::SmallVector<mlir::Value> dynamicShape =
      elideExtentsAlreadyInType(type, shape);
  llvm::SmallVector<mlir::Value> dynamicLength =
      elideLengthsAlreadyInType(type, lenParams);
  InsertPoint insPt;
  const bool hoistAlloc =
      !isPinned && dynamicShape.empty() && dynamicLength.empty();
  if (hoistAlloc) {
    insPt = saveInsertionPoint();
    setInsertionPointToStart(getEntryBlock());
  }
  assert(!type.isa<fir::ReferenceType>() && "cannot be a reference");
  auto ae = create<fir::AllocaOp>(loc, type, name, dynamicLength, dynamicShape,
                                  attrs);
  if (hoistAlloc)
    restoreInsertionPoint(insPt);
  return ae;
}

/// Create a global variable in the (read-only) data section. A global variable
/// must have a unique name to identify and reference it.
fir::GlobalOp Fortran::lower::FirOpBuilder::createGlobal(
    mlir::Location loc, mlir::Type type, llvm::StringRef name,
    mlir::StringAttr linkage, mlir::Attribute value, bool isConst) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody()->getTerminator());
  auto glob = create<fir::GlobalOp>(loc, name, isConst, type, value, linkage);
  restoreInsertionPoint(insertPt);
  return glob;
}

fir::GlobalOp Fortran::lower::FirOpBuilder::createGlobal(
    mlir::Location loc, mlir::Type type, llvm::StringRef name, bool isConst,
    std::function<void(FirOpBuilder &)> bodyBuilder, mlir::StringAttr linkage) {
  auto module = getModule();
  auto insertPt = saveInsertionPoint();
  if (auto glob = module.lookupSymbol<fir::GlobalOp>(name))
    return glob;
  setInsertionPoint(module.getBody()->getTerminator());
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

mlir::Value Fortran::lower::FirOpBuilder::convertWithSemantics(
    mlir::Location loc, mlir::Type toTy, mlir::Value val) {
  assert(toTy && "store location must be typed");
  auto fromTy = val.getType();
  if (fromTy == toTy)
    return val;
  ComplexExprHelper helper{*this, loc};
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

mlir::Value Fortran::lower::FirOpBuilder::createConvert(mlir::Location loc,
                                                        mlir::Type toTy,
                                                        mlir::Value val) {
  if (val.getType() != toTy)
    return create<fir::ConvertOp>(loc, toTy, val);
  return val;
}

fir::StringLitOp
Fortran::lower::FirOpBuilder::createStringLitOp(mlir::Location loc,
                                                llvm::StringRef data) {
  auto type = fir::CharacterType::get(getContext(), 1, data.size());
  auto strAttr = mlir::StringAttr::get(getContext(), data);
  auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), getContext());
  mlir::NamedAttribute dataAttr(valTag, strAttr);
  auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), getContext());
  mlir::NamedAttribute sizeAttr(sizeTag, getI64IntegerAttr(data.size()));
  llvm::SmallVector<mlir::NamedAttribute> attrs{dataAttr, sizeAttr};
  return create<fir::StringLitOp>(loc, llvm::ArrayRef<mlir::Type>{type},
                                  llvm::None, attrs);
}

mlir::Value
Fortran::lower::FirOpBuilder::consShape(mlir::Location loc,
                                        const fir::AbstractArrayBox &arr) {
  if (arr.lboundsAllOne()) {
    auto shapeType = fir::ShapeType::get(getContext(), arr.getExtents().size());
    return create<fir::ShapeOp>(loc, shapeType, arr.getExtents());
  }
  auto shapeType =
      fir::ShapeShiftType::get(getContext(), arr.getExtents().size());
  SmallVector<mlir::Value> shapeArgs;
  auto idxTy = getIndexType();
  for (auto [lbnd, ext] : llvm::zip(arr.getLBounds(), arr.getExtents())) {
    auto lb = createConvert(loc, idxTy, lbnd);
    shapeArgs.push_back(lb);
    shapeArgs.push_back(ext);
  }
  return create<fir::ShapeShiftOp>(loc, shapeType, shapeArgs);
}

mlir::Value
Fortran::lower::FirOpBuilder::createShape(mlir::Location loc,
                                          const fir::ExtendedValue &exv) {
  return exv.match(
      [&](const fir::ArrayBoxValue &box) { return consShape(loc, box); },
      [&](const fir::CharArrayBoxValue &box) { return consShape(loc, box); },
      [&](const fir::BoxValue &box) -> mlir::Value {
        if (!box.getLBounds().empty()) {
          auto shiftType =
              fir::ShiftType::get(getContext(), box.getLBounds().size());
          return create<fir::ShiftOp>(loc, shiftType, box.getLBounds());
        }
        return mlir::Value{};
      },
      [&](const fir::MutableBoxValue &) -> mlir::Value {
        // MutableBoxValue must be read into another category to work with them
        // outside of allocation/assignment contexts.
        fir::emitFatalError(loc, "createShape on MutableBoxValue");
      },
      [&](auto) -> mlir::Value { fir::emitFatalError(loc, "not an array"); });
}

mlir::Value Fortran::lower::FirOpBuilder::createSlice(
    mlir::Location loc, const fir::ExtendedValue &exv, mlir::ValueRange triples,
    mlir::ValueRange path) {
  if (triples.empty()) {
    // If there is no slicing by triple notation, then take the whole array.
    auto fullShape = [&](const llvm::ArrayRef<mlir::Value> lbounds,
                         llvm::ArrayRef<mlir::Value> extents) -> mlir::Value {
      llvm::SmallVector<mlir::Value> trips;
      auto idxTy = getIndexType();
      auto one = createIntegerConstant(loc, idxTy, 1);
      auto sliceTy = fir::SliceType::get(getContext(), extents.size());
      if (lbounds.empty()) {
        for (auto v : extents) {
          trips.push_back(one);
          trips.push_back(v);
          trips.push_back(one);
        }
        return create<fir::SliceOp>(loc, sliceTy, trips, path);
      }
      for (auto [lbnd, ext] : llvm::zip(lbounds, extents)) {
        auto lb = createConvert(loc, idxTy, lbnd);
        trips.push_back(lb);
        trips.push_back(ext);
        trips.push_back(one);
      }
      return create<fir::SliceOp>(loc, sliceTy, trips, path);
    };
    return exv.match(
        [&](const fir::ArrayBoxValue &box) {
          return fullShape(box.getLBounds(), box.getExtents());
        },
        [&](const fir::CharArrayBoxValue &box) {
          return fullShape(box.getLBounds(), box.getExtents());
        },
        [&](const fir::BoxValue &box) {
          auto extents = Fortran::lower::readExtents(*this, loc, box);
          return fullShape(box.getLBounds(), extents);
        },
        [&](const fir::MutableBoxValue &) -> mlir::Value {
          // MutableBoxValue must be read into another category to work with
          // them outside of allocation/assignment contexts.
          fir::emitFatalError(loc, "createSlice on MutableBoxValue");
        },
        [&](auto) -> mlir::Value { fir::emitFatalError(loc, "not an array"); });
  }
  auto rank = exv.rank();
  auto sliceTy = fir::SliceType::get(getContext(), rank);
  return create<fir::SliceOp>(loc, sliceTy, triples, path);
}

mlir::Value
Fortran::lower::FirOpBuilder::createBox(mlir::Location loc,
                                        const fir::ExtendedValue &exv) {
  auto itemAddr = fir::getBase(exv);
  if (itemAddr.getType().isa<fir::BoxType>())
    return itemAddr;
  auto elementType = fir::dyn_cast_ptrEleTy(itemAddr.getType());
  if (!elementType)
    mlir::emitError(loc, "internal: expected a memory reference type ")
        << itemAddr.getType();
  auto boxTy = fir::BoxType::get(elementType);
  return exv.match(
      [&](const fir::ArrayBoxValue &box) -> mlir::Value {
        auto s = createShape(loc, exv);
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, s);
      },
      [&](const fir::CharArrayBoxValue &box) -> mlir::Value {
        auto s = createShape(loc, exv);
        if (Fortran::lower::CharacterExprHelper::hasConstantLengthInType(exv))
          return create<fir::EmboxOp>(loc, boxTy, itemAddr, s);

        mlir::Value emptySlice;
        llvm::SmallVector<mlir::Value> lenParams{box.getLen()};
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, s, emptySlice,
                                    lenParams);
      },
      [&](const fir::CharBoxValue &box) -> mlir::Value {
        if (Fortran::lower::CharacterExprHelper::hasConstantLengthInType(exv))
          return create<fir::EmboxOp>(loc, boxTy, itemAddr);
        mlir::Value emptyShape, emptySlice;
        llvm::SmallVector<mlir::Value> lenParams{box.getLen()};
        return create<fir::EmboxOp>(loc, boxTy, itemAddr, emptyShape,
                                    emptySlice, lenParams);
      },
      [&](const fir::MutableBoxValue &x) -> mlir::Value {
        return create<fir::LoadOp>(
            loc, Fortran::lower::getMutableIRBox(*this, loc, x));
      },
      [&](const auto &) -> mlir::Value {
        return create<fir::EmboxOp>(loc, boxTy, itemAddr);
      });
}

//===--------------------------------------------------------------------===//
// ExtendedValue inquiry helper implementation
//===--------------------------------------------------------------------===//

mlir::Value Fortran::lower::readCharLen(Fortran::lower::FirOpBuilder &builder,
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
        return Fortran::lower::CharacterExprHelper{builder, loc}
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

mlir::Value Fortran::lower::readExtent(Fortran::lower::FirOpBuilder &builder,
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

mlir::Value Fortran::lower::readLowerBound(Fortran::lower::FirOpBuilder &,
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
      [&](const fir::MutableBoxValue &) -> mlir::Value {
        // MutableBoxValue must be read into another category to work with them
        // outside of allocation/assignment contexts.
        fir::emitFatalError(loc, "readLowerBound on MutableBoxValue");
      },
      [&](const auto &) -> mlir::Value {
        fir::emitFatalError(loc, "lower bound inquiry on scalar");
      });
  if (lb)
    return lb;
  return defaultValue;
}

llvm::SmallVector<mlir::Value>
Fortran::lower::readExtents(Fortran::lower::FirOpBuilder &builder,
                            mlir::Location loc, const fir::BoxValue &box) {
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

std::string Fortran::lower::uniqueCGIdent(llvm::StringRef prefix,
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

std::string Fortran::lower::LiteralNameHelper::getName(
    Fortran::lower::FirOpBuilder &builder) {
  auto name = fir::NameUniquer::doGenerated("ro."s.append(typeId).append("."));
  if (!size)
    return name += "null";
  llvm::MD5 hashValue{};
  hashValue.update(ArrayRef<uint8_t>{address, size});
  llvm::MD5::MD5Result hashResult;
  hashValue.final(hashResult);
  llvm::SmallString<32> hashString;
  llvm::MD5::stringifyResult(hashResult, hashString);
  return name += hashString.c_str();
}

mlir::Value
Fortran::lower::locationToFilename(Fortran::lower::FirOpBuilder &builder,
                                   mlir::Location loc) {
  if (auto flc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    // must be encoded as asciiz, C string
    auto fn = flc.getFilename().str() + '\0';
    return fir::getBase(createStringLiteral(builder, loc, fn));
  }
  return builder.createNullConstant(loc);
}

mlir::Value
Fortran::lower::locationToLineNo(Fortran::lower::FirOpBuilder &builder,
                                 mlir::Location loc, mlir::Type type) {
  if (auto flc = loc.dyn_cast<mlir::FileLineColLoc>())
    return builder.createIntegerConstant(loc, type, flc.getLine());
  return builder.createIntegerConstant(loc, type, 0);
}

fir::ExtendedValue
Fortran::lower::createStringLiteral(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc, llvm::StringRef str) {
  std::string globalName = Fortran::lower::uniqueCGIdent("cl", str);
  auto type = fir::CharacterType::get(builder.getContext(), 1, str.size());
  auto global = builder.getNamedGlobal(globalName);
  if (!global)
    global = builder.createGlobalConstant(
        loc, type, globalName,
        [&](Fortran::lower::FirOpBuilder &builder) {
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
