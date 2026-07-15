//===-- EnvironmentDefaults.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/EnvironmentDefaults.h"
#include "flang/Lower/EnvironmentDefault.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "llvm/ADT/ArrayRef.h"

mlir::Value fir::runtime::genEnvironmentDefaults(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const std::vector<Fortran::lower::EnvironmentDefault> &envDefaults) {
  std::string envDefaultListPtrName =
      fir::NameUniquer::doGenerated("EnvironmentDefaults");

  mlir::MLIRContext *context = builder.getContext();
  mlir::StringAttr linkOnce = builder.createLinkOnceLinkage();
  mlir::IntegerType intTy = builder.getIntegerType(8 * sizeof(int));
  fir::ReferenceType charRefTy =
      fir::ReferenceType::get(builder.getIntegerType(8));
  fir::SequenceType itemListTy = fir::SequenceType::get(
      envDefaults.size(),
      mlir::TupleType::get(context, {charRefTy, charRefTy}));
  mlir::TupleType envDefaultListTy = mlir::TupleType::get(
      context, {intTy, fir::ReferenceType::get(itemListTy)});
  fir::ReferenceType envDefaultListRefTy =
      fir::ReferenceType::get(envDefaultListTy);

  // If no defaults were specified, initialize with a null pointer.
  if (envDefaults.empty()) {
    mlir::Value nullVal = builder.createNullConstant(loc, envDefaultListRefTy);
    return nullVal;
  }

  // Create the Item list.
  mlir::IndexType idxTy = builder.getIndexType();
  mlir::IntegerAttr zero = builder.getIntegerAttr(idxTy, 0);
  mlir::IntegerAttr one = builder.getIntegerAttr(idxTy, 1);
  std::string itemListName = envDefaultListPtrName + ".items";
  auto listBuilder = [&](fir::FirOpBuilder &builder) {
    mlir::Value list = fir::UndefOp::create(builder, loc, itemListTy);
    llvm::SmallVector<mlir::Attribute, 2> idx = {mlir::Attribute{},
                                                 mlir::Attribute{}};
    auto insertStringField = [&](const std::string &s,
                                 llvm::ArrayRef<mlir::Attribute> idx) {
      mlir::Value stringAddress = fir::getBase(
          fir::factory::createStringLiteral(builder, loc, s + '\0'));
      mlir::Value addr = builder.createConvert(loc, charRefTy, stringAddress);
      return fir::InsertValueOp::create(builder, loc, itemListTy, list, addr,
                                        builder.getArrayAttr(idx));
    };

    size_t n = 0;
    for (const Fortran::lower::EnvironmentDefault &def : envDefaults) {
      idx[0] = builder.getIntegerAttr(idxTy, n);
      idx[1] = zero;
      list = insertStringField(def.varName, idx);
      idx[1] = one;
      list = insertStringField(def.defaultValue, idx);
      ++n;
    }
    fir::HasValueOp::create(builder, loc, list);
  };
  builder.createGlobalConstant(loc, itemListTy, itemListName, listBuilder,
                               linkOnce);

  // Define the EnviornmentDefaultList object.
  auto envDefaultListBuilder = [&](fir::FirOpBuilder &builder) {
    mlir::Value envDefaultList =
        fir::UndefOp::create(builder, loc, envDefaultListTy);
    mlir::Value numItems =
        builder.createIntegerConstant(loc, intTy, envDefaults.size());
    envDefaultList = fir::InsertValueOp::create(builder, loc, envDefaultListTy,
                                                envDefaultList, numItems,
                                                builder.getArrayAttr(zero));
    fir::GlobalOp itemList = builder.getNamedGlobal(itemListName);
    assert(itemList && "missing environment default list");
    mlir::Value listAddr = fir::AddrOfOp::create(
        builder, loc, itemList.resultType(), itemList.getSymbol());
    envDefaultList = fir::InsertValueOp::create(builder, loc, envDefaultListTy,
                                                envDefaultList, listAddr,
                                                builder.getArrayAttr(one));
    fir::HasValueOp::create(builder, loc, envDefaultList);
  };
  fir::GlobalOp envDefaultList = builder.createGlobalConstant(
      loc, envDefaultListTy, envDefaultListPtrName + ".list",
      envDefaultListBuilder, linkOnce);

  // Define the pointer to the list used by the runtime.
  mlir::Value addr = fir::AddrOfOp::create(
      builder, loc, envDefaultList.resultType(), envDefaultList.getSymbol());
  return addr;
}
