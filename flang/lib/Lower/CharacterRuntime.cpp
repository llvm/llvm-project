//===-- CharacterRuntime.cpp -- runtime for CHARACTER type entities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CharacterRuntime.h"
#include "../../runtime/character.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/FIRBuilder.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace Fortran::runtime;

#define NAMIFY_HELPER(X) #X
#define NAMIFY(X) NAMIFY_HELPER(IONAME(X))
#define mkRTKey(X) mkKey(RTNAME(X))

namespace Fortran::lower {
/// Static table of CHARACTER runtime calls
///
/// This logical map contains the name and type builder function for each
/// runtime function listed in the tuple. This table is fully constructed at
/// compile-time. Use the `mkRTKey` macro to access the table.
static constexpr std::tuple<
    mkRTKey(CharacterCompareScalar), mkRTKey(CharacterCompareScalar1),
    mkRTKey(CharacterCompareScalar2), mkRTKey(CharacterCompareScalar4),
    mkRTKey(CharacterCompare)>
    newCharRTTable;
} // namespace Fortran::lower

using namespace Fortran::lower;

/// Helper function to retrieve the name of the IO function given the key `A`
template <typename A>
static constexpr const char *getName() {
  return std::get<A>(newCharRTTable).name;
}

/// Helper function to retrieve the type model signature builder of the IO
/// function as defined by the key `A`
template <typename A>
static constexpr FuncTypeBuilderFunc getTypeModel() {
  return std::get<A>(newCharRTTable).getTypeModel();
}

inline int64_t getLength(mlir::Type argTy) {
  return argTy.cast<fir::SequenceType>().getShape()[0];
}

/// Get (or generate) the MLIR FuncOp for a given runtime function.
template <typename E>
static mlir::FuncOp getRuntimeFunc(mlir::Location loc,
                                   Fortran::lower::FirOpBuilder &builder) {
  auto name = getName<E>();
  auto func = builder.getNamedFunction(name);
  if (func)
    return func;
  auto funTy = getTypeModel<E>()(builder.getContext());
  func = builder.createFunction(loc, name, funTy);
  func->setAttr("fir.runtime", builder.getUnitAttr());
  return func;
}

/// Helper function to recover the KIND from the FIR type.
static int discoverKind(mlir::Type ty) {
  if (auto charTy = ty.dyn_cast<fir::CharacterType>())
    return charTy.getFKind();
  if (auto eleTy = fir::dyn_cast_ptrEleTy(ty))
    return discoverKind(eleTy);
  if (auto arrTy = ty.dyn_cast<fir::SequenceType>())
    return discoverKind(arrTy.getEleTy());
  if (auto boxTy = ty.dyn_cast<fir::BoxCharType>())
    return discoverKind(boxTy.getEleTy());
  if (auto boxTy = ty.dyn_cast<fir::BoxType>())
    return discoverKind(boxTy.getEleTy());
  llvm_unreachable("unexpected character type");
}

//===----------------------------------------------------------------------===//
// Lower character operations
//===----------------------------------------------------------------------===//

mlir::Value
Fortran::lower::genRawCharCompare(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc, mlir::CmpIPredicate cmp,
                                  mlir::Value lhsBuff, mlir::Value lhsLen,
                                  mlir::Value rhsBuff, mlir::Value rhsLen) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp beginFunc;
  switch (discoverKind(lhsBuff.getType())) {
  case 1:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar1)>(loc, builder);
    break;
  case 2:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar2)>(loc, builder);
    break;
  case 4:
    beginFunc = getRuntimeFunc<mkRTKey(CharacterCompareScalar4)>(loc, builder);
    break;
  default:
    llvm_unreachable("runtime does not support CHARACTER KIND");
  }
  auto fTy = beginFunc.getType();
  auto lptr = builder.createConvert(loc, fTy.getInput(0), lhsBuff);
  auto llen = builder.createConvert(loc, fTy.getInput(2), lhsLen);
  auto rptr = builder.createConvert(loc, fTy.getInput(1), rhsBuff);
  auto rlen = builder.createConvert(loc, fTy.getInput(3), rhsLen);
  llvm::SmallVector<mlir::Value, 4> args = {lptr, rptr, llen, rlen};
  auto tri = builder.create<mlir::CallOp>(loc, beginFunc, args).getResult(0);
  auto zero = builder.createIntegerConstant(loc, tri.getType(), 0);
  return builder.create<mlir::CmpIOp>(loc, cmp, tri, zero);
}

mlir::Value
Fortran::lower::genBoxCharCompare(Fortran::lower::AbstractConverter &converter,
                                  mlir::Location loc, mlir::CmpIPredicate cmp,
                                  mlir::Value lhs, mlir::Value rhs) {
  auto &builder = converter.getFirOpBuilder();
  Fortran::lower::CharacterExprHelper helper{builder, loc};
  auto lhsPair = helper.materializeCharacter(lhs);
  auto rhsPair = helper.materializeCharacter(rhs);
  return genRawCharCompare(converter, loc, cmp, lhsPair.first, lhsPair.second,
                           rhsPair.first, rhsPair.second);
}
