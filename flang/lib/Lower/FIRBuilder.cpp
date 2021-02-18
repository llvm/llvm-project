//===-- FIRBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/FIRBuilder.h"
#include "SymbolMap.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Semantics/symbol.h"
#include "llvm/Support/ErrorHandling.h"

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
  assert(!eleTy.isa<fir::ReferenceType>());
  return fir::ReferenceType::get(eleTy);
}

mlir::Value
Fortran::lower::FirOpBuilder::createNullConstant(mlir::Location loc) {
  auto indexType = getIndexType();
  auto zero = createIntegerConstant(loc, indexType, 0);
  auto noneRefType = getRefType(getNoneType());
  return createConvert(loc, noneRefType, zero);
}

mlir::Value Fortran::lower::FirOpBuilder::createIntegerConstant(
    mlir::Location loc, mlir::Type ty, std::int64_t cst) {
  return create<mlir::ConstantOp>(loc, ty, getIntegerAttr(ty, cst));
}

mlir::Value Fortran::lower::FirOpBuilder::createRealConstant(
    mlir::Location loc, mlir::Type realType, const llvm::APFloat &val) {
  return create<mlir::ConstantOp>(loc, realType, getFloatAttr(realType, val));
}

mlir::Value
Fortran::lower::FirOpBuilder::createRealZeroConstant(mlir::Location loc,
                                                     mlir::Type realType) {
  mlir::Attribute attr;
  if (auto firType = realType.dyn_cast<fir::RealType>()) {
    attr = getFloatAttr(
        realType,
        llvm::APFloat(kindMap.getFloatSemantics(firType.getFKind()), 0));
  } else { // mlir::FloatType.
    attr = getZeroAttr(realType);
  }
  return create<mlir::ConstantOp>(loc, realType, attr);
}

mlir::Value Fortran::lower::FirOpBuilder::allocateLocal(
    mlir::Location loc, mlir::Type ty, llvm::StringRef nm,
    llvm::ArrayRef<mlir::Value> shape, bool asTarget) {
  llvm::SmallVector<mlir::Value, 8> indices;
  auto idxTy = getIndexType();
  llvm::for_each(shape, [&](mlir::Value sh) {
    indices.push_back(createConvert(loc, idxTy, sh));
  });
  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  if (asTarget)
    attrs.emplace_back(mlir::Identifier::get("target", getContext()),
                       getUnitAttr());
  return create<fir::AllocaOp>(loc, ty, nm, llvm::None, indices, attrs);
}

/// Create a temporary variable on the stack. Anonymous temporaries have no
/// `name` value.
mlir::Value Fortran::lower::FirOpBuilder::createTemporary(
    mlir::Location loc, mlir::Type type, llvm::StringRef name,
    llvm::ArrayRef<mlir::Value> shape) {
  auto insPt = saveInsertionPoint();
  if (shape.empty())
    setInsertionPointToStart(getEntryBlock());
  else
    setInsertionPointAfter(shape.back().getDefiningOp());
  assert(!type.isa<fir::ReferenceType>() && "cannot be a reference");
  auto ae = create<fir::AllocaOp>(loc, type, name, llvm::None, shape);
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
  // FIXME: add a fir::is_integer() test
  ComplexExprHelper helper{*this, loc};
  if ((fir::isa_real(fromTy) || fromTy.isSignlessInteger()) &&
      fir::isa_complex(toTy)) {
    // imaginary part is zero
    auto eleTy = helper.getComplexPartType(toTy);
    auto cast = createConvert(loc, eleTy, val);
    llvm::APFloat zero{
        kindMap.getFloatSemantics(toTy.cast<fir::ComplexType>().getFKind()), 0};
    auto imag = createRealConstant(loc, eleTy, zero);
    return helper.createComplex(toTy, cast, imag);
  }
  // FIXME: add a fir::is_integer() test
  if (fir::isa_complex(fromTy) &&
      (toTy.isSignlessInteger() || fir::isa_real(toTy))) {
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

fir::StringLitOp Fortran::lower::FirOpBuilder::createStringLit(
    mlir::Location loc, mlir::Type eleTy, llvm::StringRef data) {
  auto strAttr = mlir::StringAttr::get(getContext(), data);
  auto valTag = mlir::Identifier::get(fir::StringLitOp::value(), getContext());
  mlir::NamedAttribute dataAttr(valTag, strAttr);
  auto sizeTag = mlir::Identifier::get(fir::StringLitOp::size(), getContext());
  mlir::NamedAttribute sizeAttr(sizeTag, getI64IntegerAttr(data.size()));
  llvm::SmallVector<mlir::NamedAttribute, 2> attrs{dataAttr, sizeAttr};
  auto arrTy =
      fir::SequenceType::get(fir::SequenceType::Shape(1, data.size()), eleTy);
  return create<fir::StringLitOp>(loc, llvm::ArrayRef<mlir::Type>{arrTy},
                                  llvm::None, attrs);
}
