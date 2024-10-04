//===-- DebugTypeGenerator.cpp -- type conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "flang-debug-type-generator"

#include "DebugTypeGenerator.h"
#include "flang/Optimizer/CodeGen/DescriptorModel.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"

namespace fir {

/// Calculate offset of any field in the descriptor.
template <int DescriptorField>
std::uint64_t getComponentOffset(const mlir::DataLayout &dl,
                                 mlir::MLIRContext *context,
                                 mlir::Type llvmFieldType) {
  static_assert(DescriptorField > 0 && DescriptorField < 10);
  mlir::Type previousFieldType =
      getDescFieldTypeModel<DescriptorField - 1>()(context);
  std::uint64_t previousOffset =
      getComponentOffset<DescriptorField - 1>(dl, context, previousFieldType);
  std::uint64_t offset = previousOffset + dl.getTypeSize(previousFieldType);
  std::uint64_t fieldAlignment = dl.getTypeABIAlignment(llvmFieldType);
  return llvm::alignTo(offset, fieldAlignment);
}
template <>
std::uint64_t getComponentOffset<0>(const mlir::DataLayout &dl,
                                    mlir::MLIRContext *context,
                                    mlir::Type llvmFieldType) {
  return 0;
}

DebugTypeGenerator::DebugTypeGenerator(mlir::ModuleOp m,
                                       mlir::SymbolTable *symbolTable_,
                                       const mlir::DataLayout &dl)
    : module(m), symbolTable(symbolTable_), dataLayout{&dl},
      kindMapping(getKindMapping(m)), llvmTypeConverter(m, false, false, dl) {
  LLVM_DEBUG(llvm::dbgs() << "DITypeAttr generator\n");

  mlir::MLIRContext *context = module.getContext();

  // The debug information requires the offset of certain fields in the
  // descriptors like lower_bound and extent for each dimension.
  mlir::Type llvmDimsType = getDescFieldTypeModel<kDimsPosInBox>()(context);
  mlir::Type llvmPtrType = getDescFieldTypeModel<kAddrPosInBox>()(context);
  mlir::Type llvmLenType = getDescFieldTypeModel<kElemLenPosInBox>()(context);
  dimsOffset =
      getComponentOffset<kDimsPosInBox>(*dataLayout, context, llvmDimsType);
  dimsSize = dataLayout->getTypeSize(llvmDimsType);
  ptrSize = dataLayout->getTypeSize(llvmPtrType);
  lenOffset =
      getComponentOffset<kElemLenPosInBox>(*dataLayout, context, llvmLenType);
}

static mlir::LLVM::DITypeAttr genBasicType(mlir::MLIRContext *context,
                                           mlir::StringAttr name,
                                           unsigned bitSize,
                                           unsigned decoding) {
  return mlir::LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type, name, bitSize, decoding);
}

static mlir::LLVM::DITypeAttr genPlaceholderType(mlir::MLIRContext *context) {
  return genBasicType(context, mlir::StringAttr::get(context, "integer"),
                      /*bitSize=*/32, llvm::dwarf::DW_ATE_signed);
}

// Helper function to create DILocalVariableAttr and DbgValueOp when information
// about the size or dimension of a variable etc lives in an mlir::Value.
mlir::LLVM::DILocalVariableAttr DebugTypeGenerator::generateArtificialVariable(
    mlir::MLIRContext *context, mlir::Value val,
    mlir::LLVM::DIFileAttr fileAttr, mlir::LLVM::DIScopeAttr scope,
    fir::cg::XDeclareOp declOp) {
  // There can be multiple artificial variable for a single declOp. To help
  // distinguish them, we pad the name with a counter. The counter is the
  // position of 'val' in the operands of declOp.
  auto varID = std::distance(
      declOp.getOperands().begin(),
      std::find(declOp.getOperands().begin(), declOp.getOperands().end(), val));
  mlir::OpBuilder builder(context);
  auto name = mlir::StringAttr::get(context, "." + declOp.getUniqName().str() +
                                                 std::to_string(varID));
  builder.setInsertionPoint(declOp);
  mlir::Type type = val.getType();
  if (!mlir::isa<mlir::IntegerType>(type) || !type.isSignlessInteger()) {
    type = builder.getIntegerType(64);
    val = builder.create<fir::ConvertOp>(declOp.getLoc(), type, val);
  }
  mlir::LLVM::DITypeAttr Ty = convertType(type, fileAttr, scope, declOp);
  auto lvAttr = mlir::LLVM::DILocalVariableAttr::get(
      context, scope, name, fileAttr, /*line=*/0, /*argNo=*/0,
      /*alignInBits=*/0, Ty, mlir::LLVM::DIFlags::Artificial);
  builder.create<mlir::LLVM::DbgValueOp>(declOp.getLoc(), val, lvAttr, nullptr);
  return lvAttr;
}

mlir::LLVM::DITypeAttr DebugTypeGenerator::convertBoxedSequenceType(
    fir::SequenceType seqTy, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp,
    bool genAllocated, bool genAssociated) {

  mlir::MLIRContext *context = module.getContext();
  // FIXME: Assumed rank arrays not supported yet
  if (seqTy.hasUnknownShape())
    return genPlaceholderType(context);

  llvm::SmallVector<mlir::LLVM::DIExpressionElemAttr> ops;
  auto addOp = [&](unsigned opc, llvm::ArrayRef<uint64_t> vals) {
    ops.push_back(mlir::LLVM::DIExpressionElemAttr::get(context, opc, vals));
  };

  addOp(llvm::dwarf::DW_OP_push_object_address, {});
  addOp(llvm::dwarf::DW_OP_deref, {});

  // dataLocation = *base_addr
  mlir::LLVM::DIExpressionAttr dataLocation =
      mlir::LLVM::DIExpressionAttr::get(context, ops);
  addOp(llvm::dwarf::DW_OP_lit0, {});
  addOp(llvm::dwarf::DW_OP_ne, {});

  // allocated = associated = (*base_addr != 0)
  mlir::LLVM::DIExpressionAttr valid =
      mlir::LLVM::DIExpressionAttr::get(context, ops);
  mlir::LLVM::DIExpressionAttr allocated = genAllocated ? valid : nullptr;
  mlir::LLVM::DIExpressionAttr associated = genAssociated ? valid : nullptr;
  ops.clear();

  llvm::SmallVector<mlir::LLVM::DINodeAttr> elements;
  mlir::LLVM::DITypeAttr elemTy =
      convertType(seqTy.getEleTy(), fileAttr, scope, declOp);
  unsigned offset = dimsOffset;
  unsigned index = 0;
  mlir::IntegerType intTy = mlir::IntegerType::get(context, 64);
  const unsigned indexSize = dimsSize / 3;
  for ([[maybe_unused]] auto _ : seqTy.getShape()) {
    // For each dimension, find the offset of count, lower bound and stride in
    // the descriptor and generate the dwarf expression to extract it.
    mlir::Attribute lowerAttr = nullptr;
    // If declaration has a lower bound, use it.
    if (declOp && declOp.getShift().size() > index) {
      if (std::optional<std::int64_t> optint =
              getIntIfConstant(declOp.getShift()[index]))
        lowerAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, *optint));
      else
        lowerAttr = generateArtificialVariable(
            context, declOp.getShift()[index], fileAttr, scope, declOp);
    }
    // FIXME: If `indexSize` happens to be bigger than address size on the
    // system then we may have to change 'DW_OP_deref' here.
    addOp(llvm::dwarf::DW_OP_push_object_address, {});
    addOp(llvm::dwarf::DW_OP_plus_uconst,
          {offset + (indexSize * kDimExtentPos)});
    addOp(llvm::dwarf::DW_OP_deref, {});
    // count[i] = *(base_addr + offset + (indexSize * kDimExtentPos))
    // where 'offset' is dimsOffset + (i * dimsSize)
    mlir::LLVM::DIExpressionAttr countAttr =
        mlir::LLVM::DIExpressionAttr::get(context, ops);
    ops.clear();

    // If a lower bound was not found in the declOp, then we will get them from
    // descriptor only for pointer and allocatable case. DWARF assumes lower
    // bound of 1 when this attribute is missing.
    if (!lowerAttr && (genAllocated || genAssociated)) {
      addOp(llvm::dwarf::DW_OP_push_object_address, {});
      addOp(llvm::dwarf::DW_OP_plus_uconst,
            {offset + (indexSize * kDimLowerBoundPos)});
      addOp(llvm::dwarf::DW_OP_deref, {});
      // lower_bound[i] = *(base_addr + offset + (indexSize *
      // kDimLowerBoundPos))
      lowerAttr = mlir::LLVM::DIExpressionAttr::get(context, ops);
      ops.clear();
    }

    addOp(llvm::dwarf::DW_OP_push_object_address, {});
    addOp(llvm::dwarf::DW_OP_plus_uconst,
          {offset + (indexSize * kDimStridePos)});
    addOp(llvm::dwarf::DW_OP_deref, {});
    // stride[i] = *(base_addr + offset + (indexSize * kDimStridePos))
    mlir::LLVM::DIExpressionAttr strideAttr =
        mlir::LLVM::DIExpressionAttr::get(context, ops);
    ops.clear();

    offset += dimsSize;
    mlir::LLVM::DISubrangeAttr subrangeTy = mlir::LLVM::DISubrangeAttr::get(
        context, countAttr, lowerAttr, /*upperBound=*/nullptr, strideAttr);
    elements.push_back(subrangeTy);
    ++index;
  }
  return mlir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_array_type, /*name=*/nullptr,
      /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, elemTy,
      mlir::LLVM::DIFlags::Zero, /*sizeInBits=*/0, /*alignInBits=*/0, elements,
      dataLocation, /*rank=*/nullptr, allocated, associated);
}

// If the type is a pointer or array type then gets its underlying type.
static mlir::LLVM::DITypeAttr getUnderlyingType(mlir::LLVM::DITypeAttr Ty) {
  if (auto ptrTy =
          mlir::dyn_cast_if_present<mlir::LLVM::DIDerivedTypeAttr>(Ty)) {
    if (ptrTy.getTag() == llvm::dwarf::DW_TAG_pointer_type)
      Ty = getUnderlyingType(ptrTy.getBaseType());
  }
  if (auto comTy =
          mlir::dyn_cast_if_present<mlir::LLVM::DICompositeTypeAttr>(Ty)) {
    if (comTy.getTag() == llvm::dwarf::DW_TAG_array_type)
      Ty = getUnderlyingType(comTy.getBaseType());
  }
  return Ty;
}

// Currently, the handling of recursive debug type in mlir has some limitations.
// Those limitations were discussed at the end of the thread for following PR.
// https://github.com/llvm/llvm-project/pull/106571
//
// Problem could be explained with the following example code:
//  type t2
//   type(t1), pointer :: p1
// end type
// type t1
//   type(t2), pointer :: p2
// end type
// In the description below, type_self means a temporary type that is generated
// as a place holder while the members of that type are being processed.
//
// If we process t1 first then we will have the following structure after it has
// been processed.
// t1 -> t2 -> t1_self
// This is because when we started processing t2, we did not have the complete
// t1 but its place holder t1_self.
// Now if some entity requires t2, we will already have that in cache and will
// return it. But this t2 refers to t1_self and not to t1. In mlir handling,
// only those types are allowed to have _self reference which are wrapped by
// entity whose reference it is. So t1 -> t2 -> t1_self is ok because the
// t1_self reference can be resolved by the outer t1. But standalone t2 is not
// because there will be no way to resolve it. Until this is fixed in mlir, we
// avoid caching such types. Please see DebugTranslation::translateRecursive for
// details on how mlir handles recursive types.
static bool canCacheThisType(mlir::LLVM::DICompositeTypeAttr comTy) {
  for (auto el : comTy.getElements()) {
    if (auto mem =
            mlir::dyn_cast_if_present<mlir::LLVM::DIDerivedTypeAttr>(el)) {
      mlir::LLVM::DITypeAttr memTy = getUnderlyingType(mem.getBaseType());
      if (auto baseTy =
              mlir::dyn_cast_if_present<mlir::LLVM::DICompositeTypeAttr>(
                  memTy)) {
        // We will not cache a type if one of its member meets the following
        // conditions:
        // 1. It is a structure type
        // 2. It is a place holder type (getIsRecSelf() is true)
        // 3. It is not a self reference. It is ok to have t1_self in t1.
        if (baseTy.getTag() == llvm::dwarf::DW_TAG_structure_type &&
            baseTy.getIsRecSelf() && (comTy.getRecId() != baseTy.getRecId()))
          return false;
      }
    }
  }
  return true;
}

mlir::LLVM::DITypeAttr DebugTypeGenerator::convertRecordType(
    fir::RecordType Ty, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp) {
  // Check if this type has already been converted.
  auto iter = typeCache.find(Ty);
  if (iter != typeCache.end())
    return iter->second;

  llvm::SmallVector<mlir::LLVM::DINodeAttr> elements;
  mlir::MLIRContext *context = module.getContext();
  auto recId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
  // Generate a place holder TypeAttr which will be used if a member
  // references the parent type.
  auto comAttr = mlir::LLVM::DICompositeTypeAttr::get(
      context, recId, /*isRecSelf=*/true, llvm::dwarf::DW_TAG_structure_type,
      mlir::StringAttr::get(context, ""), fileAttr, /*line=*/0, scope,
      /*baseType=*/nullptr, mlir::LLVM::DIFlags::Zero, /*sizeInBits=*/0,
      /*alignInBits=*/0, elements, /*dataLocation=*/nullptr, /*rank=*/nullptr,
      /*allocated=*/nullptr, /*associated=*/nullptr);
  typeCache[Ty] = comAttr;

  auto result = fir::NameUniquer::deconstruct(Ty.getName());
  if (result.first != fir::NameUniquer::NameKind::DERIVED_TYPE)
    return genPlaceholderType(context);

  fir::TypeInfoOp tiOp = symbolTable->lookup<fir::TypeInfoOp>(Ty.getName());
  unsigned line = (tiOp) ? getLineFromLoc(tiOp.getLoc()) : 1;

  std::uint64_t offset = 0;
  for (auto [fieldName, fieldTy] : Ty.getTypeList()) {
    mlir::Type llvmTy;
    if (auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(fieldTy))
      llvmTy =
          llvmTypeConverter.convertBoxTypeAsStruct(boxTy, getBoxRank(boxTy));
    else
      llvmTy = llvmTypeConverter.convertType(fieldTy);

    // FIXME: Handle non defaults array bound in derived types
    uint64_t byteSize = dataLayout->getTypeSize(llvmTy);
    unsigned short byteAlign = dataLayout->getTypeABIAlignment(llvmTy);
    mlir::LLVM::DITypeAttr elemTy =
        convertType(fieldTy, fileAttr, scope, /*declOp=*/nullptr);
    offset = llvm::alignTo(offset, byteAlign);
    mlir::LLVM::DIDerivedTypeAttr tyAttr = mlir::LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_member,
        mlir::StringAttr::get(context, fieldName), elemTy, byteSize * 8,
        byteAlign * 8, offset * 8, /*optional<address space>=*/std::nullopt,
        /*extra data=*/nullptr);
    elements.push_back(tyAttr);
    offset += llvm::alignTo(byteSize, byteAlign);
  }

  auto finalAttr = mlir::LLVM::DICompositeTypeAttr::get(
      context, recId, /*isRecSelf=*/false, llvm::dwarf::DW_TAG_structure_type,
      mlir::StringAttr::get(context, result.second.name), fileAttr, line, scope,
      /*baseType=*/nullptr, mlir::LLVM::DIFlags::Zero, offset * 8,
      /*alignInBits=*/0, elements, /*dataLocation=*/nullptr, /*rank=*/nullptr,
      /*allocated=*/nullptr, /*associated=*/nullptr);
  if (canCacheThisType(finalAttr)) {
    typeCache[Ty] = finalAttr;
  } else {
    auto iter = typeCache.find(Ty);
    if (iter != typeCache.end())
      typeCache.erase(iter);
  }
  return finalAttr;
}

mlir::LLVM::DITypeAttr DebugTypeGenerator::convertSequenceType(
    fir::SequenceType seqTy, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp) {
  mlir::MLIRContext *context = module.getContext();

  llvm::SmallVector<mlir::LLVM::DINodeAttr> elements;
  mlir::LLVM::DITypeAttr elemTy =
      convertType(seqTy.getEleTy(), fileAttr, scope, declOp);

  unsigned index = 0;
  auto intTy = mlir::IntegerType::get(context, 64);
  for (fir::SequenceType::Extent dim : seqTy.getShape()) {
    mlir::Attribute lowerAttr = nullptr;
    mlir::Attribute countAttr = nullptr;
    // If declOp is present, we use the shift in it to get the lower bound of
    // the array. If it is constant, that is used. If it is not constant, we
    // create a variable that represents its location and use that as lower
    // bound. As an optimization, we don't create a lower bound when shift is a
    // constant 1 as that is the default.
    if (declOp && declOp.getShift().size() > index) {
      if (std::optional<std::int64_t> optint =
              getIntIfConstant(declOp.getShift()[index])) {
        if (*optint != 1)
          lowerAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, *optint));
      } else
        lowerAttr = generateArtificialVariable(
            context, declOp.getShift()[index], fileAttr, scope, declOp);
    }

    if (dim == seqTy.getUnknownExtent()) {
      // This path is taken for both assumed size array or when the size of the
      // array is variable. In the case of variable size, we create a variable
      // to use as countAttr. Note that fir has a constant size of -1 for
      // assumed size array. So !optint check makes sure we don't generate
      // variable in that case.
      if (declOp && declOp.getShape().size() > index) {
        std::optional<std::int64_t> optint =
            getIntIfConstant(declOp.getShape()[index]);
        if (!optint)
          countAttr = generateArtificialVariable(
              context, declOp.getShape()[index], fileAttr, scope, declOp);
      }
    } else
      countAttr = mlir::IntegerAttr::get(intTy, llvm::APInt(64, dim));

    auto subrangeTy = mlir::LLVM::DISubrangeAttr::get(
        context, countAttr, lowerAttr, /*upperBound=*/nullptr,
        /*stride=*/nullptr);
    elements.push_back(subrangeTy);
    ++index;
  }
  // Apart from arrays, the `DICompositeTypeAttr` is used for other things like
  // structure types. Many of its fields which are not applicable to arrays
  // have been set to some valid default values.

  return mlir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_array_type, /*name=*/nullptr,
      /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, elemTy,
      mlir::LLVM::DIFlags::Zero, /*sizeInBits=*/0, /*alignInBits=*/0, elements,
      /*dataLocation=*/nullptr, /*rank=*/nullptr, /*allocated=*/nullptr,
      /*associated=*/nullptr);
}

mlir::LLVM::DITypeAttr DebugTypeGenerator::convertCharacterType(
    fir::CharacterType charTy, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp,
    bool hasDescriptor) {
  mlir::MLIRContext *context = module.getContext();

  // DWARF 5 says the following about the character encoding in 5.1.1.2.
  // "DW_ATE_ASCII and DW_ATE_UCS specify encodings for the Fortran 2003
  // string kinds ASCII (ISO/IEC 646:1991) and ISO_10646 (UCS-4 in ISO/IEC
  // 10646:2000)."
  unsigned encoding = llvm::dwarf::DW_ATE_ASCII;
  if (charTy.getFKind() != 1)
    encoding = llvm::dwarf::DW_ATE_UCS;

  uint64_t sizeInBits = 0;
  mlir::LLVM::DIExpressionAttr lenExpr = nullptr;
  mlir::LLVM::DIExpressionAttr locExpr = nullptr;
  mlir::LLVM::DIVariableAttr varAttr = nullptr;

  if (hasDescriptor) {
    llvm::SmallVector<mlir::LLVM::DIExpressionElemAttr> ops;
    auto addOp = [&](unsigned opc, llvm::ArrayRef<uint64_t> vals) {
      ops.push_back(mlir::LLVM::DIExpressionElemAttr::get(context, opc, vals));
    };
    addOp(llvm::dwarf::DW_OP_push_object_address, {});
    addOp(llvm::dwarf::DW_OP_plus_uconst, {lenOffset});
    lenExpr = mlir::LLVM::DIExpressionAttr::get(context, ops);
    ops.clear();

    addOp(llvm::dwarf::DW_OP_push_object_address, {});
    addOp(llvm::dwarf::DW_OP_deref, {});
    locExpr = mlir::LLVM::DIExpressionAttr::get(context, ops);
  } else if (charTy.hasConstantLen()) {
    sizeInBits =
        charTy.getLen() * kindMapping.getCharacterBitsize(charTy.getFKind());
  } else {
    // In assumed length string, the len of the character is not part of the
    // type but can be found at the runtime. Here we create an artificial
    // variable that will contain that length. This variable is used as
    // 'stringLength' in DIStringTypeAttr.
    if (declOp && !declOp.getTypeparams().empty()) {
      mlir::LLVM::DILocalVariableAttr lvAttr = generateArtificialVariable(
          context, declOp.getTypeparams()[0], fileAttr, scope, declOp);
      varAttr = mlir::cast<mlir::LLVM::DIVariableAttr>(lvAttr);
    }
  }

  // FIXME: Currently the DIStringType in llvm does not have the option to set
  // type of the underlying character. This restricts out ability to represent
  // string with non-default characters. Please see issue #95440 for more
  // details.
  return mlir::LLVM::DIStringTypeAttr::get(
      context, llvm::dwarf::DW_TAG_string_type,
      mlir::StringAttr::get(context, ""), sizeInBits, /*alignInBits=*/0,
      /*stringLength=*/varAttr, lenExpr, locExpr, encoding);
}

mlir::LLVM::DITypeAttr DebugTypeGenerator::convertPointerLikeType(
    mlir::Type elTy, mlir::LLVM::DIFileAttr fileAttr,
    mlir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp,
    bool genAllocated, bool genAssociated) {
  mlir::MLIRContext *context = module.getContext();

  // Arrays and character need different treatment because DWARF have special
  // constructs for them to get the location from the descriptor. Rest of
  // types are handled like pointer to underlying type.
  if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(elTy))
    return convertBoxedSequenceType(seqTy, fileAttr, scope, declOp,
                                    genAllocated, genAssociated);
  if (auto charTy = mlir::dyn_cast_or_null<fir::CharacterType>(elTy))
    return convertCharacterType(charTy, fileAttr, scope, declOp,
                                /*hasDescriptor=*/true);

  mlir::LLVM::DITypeAttr elTyAttr = convertType(elTy, fileAttr, scope, declOp);

  return mlir::LLVM::DIDerivedTypeAttr::get(
      context, llvm::dwarf::DW_TAG_pointer_type,
      mlir::StringAttr::get(context, ""), elTyAttr, ptrSize,
      /*alignInBits=*/0, /*offset=*/0,
      /*optional<address space>=*/std::nullopt, /*extra data=*/nullptr);
}

mlir::LLVM::DITypeAttr
DebugTypeGenerator::convertType(mlir::Type Ty, mlir::LLVM::DIFileAttr fileAttr,
                                mlir::LLVM::DIScopeAttr scope,
                                fir::cg::XDeclareOp declOp) {
  mlir::MLIRContext *context = module.getContext();
  if (Ty.isInteger()) {
    return genBasicType(context, mlir::StringAttr::get(context, "integer"),
                        Ty.getIntOrFloatBitWidth(), llvm::dwarf::DW_ATE_signed);
  } else if (mlir::isa<mlir::FloatType>(Ty)) {
    return genBasicType(context, mlir::StringAttr::get(context, "real"),
                        Ty.getIntOrFloatBitWidth(), llvm::dwarf::DW_ATE_float);
  } else if (auto logTy = mlir::dyn_cast_or_null<fir::LogicalType>(Ty)) {
    return genBasicType(context,
                        mlir::StringAttr::get(context, logTy.getMnemonic()),
                        kindMapping.getLogicalBitsize(logTy.getFKind()),
                        llvm::dwarf::DW_ATE_boolean);
  } else if (auto cplxTy = mlir::dyn_cast_or_null<mlir::ComplexType>(Ty)) {
    auto floatTy = mlir::cast<mlir::FloatType>(cplxTy.getElementType());
    unsigned bitWidth = floatTy.getWidth();
    return genBasicType(context, mlir::StringAttr::get(context, "complex"),
                        bitWidth * 2, llvm::dwarf::DW_ATE_complex_float);
  } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(Ty)) {
    return convertSequenceType(seqTy, fileAttr, scope, declOp);
  } else if (auto charTy = mlir::dyn_cast_or_null<fir::CharacterType>(Ty)) {
    return convertCharacterType(charTy, fileAttr, scope, declOp,
                                /*hasDescriptor=*/false);
  } else if (auto recTy = mlir::dyn_cast_or_null<fir::RecordType>(Ty)) {
    return convertRecordType(recTy, fileAttr, scope, declOp);
  } else if (auto boxTy = mlir::dyn_cast_or_null<fir::BoxType>(Ty)) {
    auto elTy = boxTy.getElementType();
    if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(elTy))
      return convertBoxedSequenceType(seqTy, fileAttr, scope, declOp, false,
                                      false);
    if (auto heapTy = mlir::dyn_cast_or_null<fir::HeapType>(elTy))
      return convertPointerLikeType(heapTy.getElementType(), fileAttr, scope,
                                    declOp, /*genAllocated=*/true,
                                    /*genAssociated=*/false);
    if (auto ptrTy = mlir::dyn_cast_or_null<fir::PointerType>(elTy))
      return convertPointerLikeType(ptrTy.getElementType(), fileAttr, scope,
                                    declOp, /*genAllocated=*/false,
                                    /*genAssociated=*/true);
    return genPlaceholderType(context);
  } else {
    // FIXME: These types are currently unhandled. We are generating a
    // placeholder type to allow us to test supported bits.
    return genPlaceholderType(context);
  }
}

} // namespace fir
