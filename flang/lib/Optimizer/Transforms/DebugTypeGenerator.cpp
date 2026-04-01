//===-- DebugTypeGenerator.cpp -- type conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "flang-debug-type-generator"

#include "DebugTypeGenerator.h"
#include "flang/Optimizer/CodeGen/DescriptorModel.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "aiir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"

namespace fir {

/// Calculate offset of any field in the descriptor.
template <int DescriptorField>
std::uint64_t getComponentOffset(const aiir::DataLayout &dl,
                                 aiir::AIIRContext *context,
                                 aiir::Type llvmFieldType) {
  static_assert(DescriptorField > 0 && DescriptorField < 10);
  aiir::Type previousFieldType =
      getDescFieldTypeModel<DescriptorField - 1>()(context);
  std::uint64_t previousOffset =
      getComponentOffset<DescriptorField - 1>(dl, context, previousFieldType);
  std::uint64_t offset = previousOffset + dl.getTypeSize(previousFieldType);
  std::uint64_t fieldAlignment = dl.getTypeABIAlignment(llvmFieldType);
  return llvm::alignTo(offset, fieldAlignment);
}
template <>
std::uint64_t getComponentOffset<0>(const aiir::DataLayout &dl,
                                    aiir::AIIRContext *context,
                                    aiir::Type llvmFieldType) {
  return 0;
}

DebugTypeGenerator::DebugTypeGenerator(aiir::ModuleOp m,
                                       aiir::SymbolTable *symbolTable_,
                                       const aiir::DataLayout &dl)
    : module(m), symbolTable(symbolTable_), dataLayout{&dl},
      kindMapping(getKindMapping(m)), llvmTypeConverter(m, false, false, dl) {
  LLVM_DEBUG(llvm::dbgs() << "DITypeAttr generator\n");

  aiir::AIIRContext *context = module.getContext();

  // The debug information requires the offset of certain fields in the
  // descriptors like lower_bound and extent for each dimension.
  aiir::Type llvmDimsType = getDescFieldTypeModel<kDimsPosInBox>()(context);
  aiir::Type llvmPtrType = getDescFieldTypeModel<kAddrPosInBox>()(context);
  aiir::Type llvmLenType = getDescFieldTypeModel<kElemLenPosInBox>()(context);
  aiir::Type llvmRankType = getDescFieldTypeModel<kRankPosInBox>()(context);

  dimsOffset =
      getComponentOffset<kDimsPosInBox>(*dataLayout, context, llvmDimsType);
  dimsSize = dataLayout->getTypeSize(llvmDimsType);
  ptrSize = dataLayout->getTypeSize(llvmPtrType);
  rankSize = dataLayout->getTypeSize(llvmRankType);
  lenOffset =
      getComponentOffset<kElemLenPosInBox>(*dataLayout, context, llvmLenType);
  rankOffset =
      getComponentOffset<kRankPosInBox>(*dataLayout, context, llvmRankType);
}

static aiir::LLVM::DITypeAttr genBasicType(aiir::AIIRContext *context,
                                           aiir::StringAttr name,
                                           unsigned bitSize,
                                           unsigned decoding) {
  return aiir::LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type, name, bitSize, decoding);
}

static aiir::StringAttr getBasicTypeName(aiir::AIIRContext *context,
                                         llvm::StringRef baseName,
                                         unsigned bitSize) {
  std::ostringstream oss;
  oss << baseName.str() << "(kind=" << (bitSize / 8) << ")";
  return aiir::StringAttr::get(context, oss.str());
}

static aiir::LLVM::DITypeAttr genPlaceholderType(aiir::AIIRContext *context) {
  return genBasicType(context, getBasicTypeName(context, "integer", 32),
                      /*bitSize=*/32, llvm::dwarf::DW_ATE_signed);
}

// Helper function to create DILocalVariableAttr and DbgValueOp when information
// about the size or dimension of a variable etc lives in an aiir::Value.
aiir::LLVM::DILocalVariableAttr DebugTypeGenerator::generateArtificialVariable(
    aiir::AIIRContext *context, aiir::Value val,
    aiir::LLVM::DIFileAttr fileAttr, aiir::LLVM::DIScopeAttr scope,
    fir::cg::XDeclareOp declOp) {
  // There can be multiple artificial variable for a single declOp. To help
  // distinguish them, we pad the name with a counter. The counter is the
  // position of 'val' in the operands of declOp.
  auto varID = std::distance(
      declOp.getOperands().begin(),
      std::find(declOp.getOperands().begin(), declOp.getOperands().end(), val));
  aiir::OpBuilder builder(context);
  auto name = aiir::StringAttr::get(context, "." + declOp.getUniqName().str() +
                                                 std::to_string(varID));
  builder.setInsertionPoint(declOp);
  aiir::Type type = val.getType();
  if (!aiir::isa<aiir::IntegerType>(type) || !type.isSignlessInteger()) {
    type = builder.getIntegerType(64);
    val = fir::ConvertOp::create(builder, declOp.getLoc(), type, val);
  }
  aiir::LLVM::DITypeAttr Ty = convertType(type, fileAttr, scope, declOp);
  auto lvAttr = aiir::LLVM::DILocalVariableAttr::get(
      context, scope, name, fileAttr, /*line=*/0, /*argNo=*/0,
      /*alignInBits=*/0, Ty, aiir::LLVM::DIFlags::Artificial);
  aiir::LLVM::DbgValueOp::create(builder, declOp.getLoc(), val, lvAttr,
                                 nullptr);
  return lvAttr;
}

aiir::LLVM::DITypeAttr DebugTypeGenerator::convertBoxedSequenceType(
    fir::SequenceType seqTy, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp,
    bool genAllocated, bool genAssociated) {

  aiir::AIIRContext *context = module.getContext();
  llvm::SmallVector<aiir::LLVM::DINodeAttr> elements;
  llvm::SmallVector<aiir::LLVM::DIExpressionElemAttr> ops;
  auto addOp = [&](unsigned opc, llvm::ArrayRef<uint64_t> vals) {
    ops.push_back(aiir::LLVM::DIExpressionElemAttr::get(context, opc, vals));
  };

  addOp(llvm::dwarf::DW_OP_push_object_address, {});
  addOp(llvm::dwarf::DW_OP_deref, {});

  // dataLocation = *base_addr
  aiir::LLVM::DIExpressionAttr dataLocation =
      aiir::LLVM::DIExpressionAttr::get(context, ops);
  ops.clear();

  aiir::LLVM::DITypeAttr elemTy =
      convertType(seqTy.getEleTy(), fileAttr, scope, declOp);

  // Assumed-rank arrays
  if (seqTy.hasUnknownShape()) {
    addOp(llvm::dwarf::DW_OP_push_object_address, {});
    addOp(llvm::dwarf::DW_OP_plus_uconst, {rankOffset});
    addOp(llvm::dwarf::DW_OP_deref_size, {rankSize});
    aiir::LLVM::DIExpressionAttr rank =
        aiir::LLVM::DIExpressionAttr::get(context, ops);
    ops.clear();

    auto genSubrangeOp = [&](unsigned field) -> aiir::LLVM::DIExpressionAttr {
      // The dwarf expression for generic subrange assumes that dimension for
      // which it is being generated is already pushed on the stack. Here is the
      // formula we will use to calculate count for example.
      // *(base_addr + offset_count_0 + (dimsSize x dimension_number)).
      // where offset_count_0 is offset of the count field for the 0th dimension
      addOp(llvm::dwarf::DW_OP_push_object_address, {});
      addOp(llvm::dwarf::DW_OP_over, {});
      addOp(llvm::dwarf::DW_OP_constu, {dimsSize});
      addOp(llvm::dwarf::DW_OP_mul, {});
      addOp(llvm::dwarf::DW_OP_plus_uconst,
            {dimsOffset + ((dimsSize / 3) * field)});
      addOp(llvm::dwarf::DW_OP_plus, {});
      addOp(llvm::dwarf::DW_OP_deref, {});
      aiir::LLVM::DIExpressionAttr attr =
          aiir::LLVM::DIExpressionAttr::get(context, ops);
      ops.clear();
      return attr;
    };

    aiir::LLVM::DIExpressionAttr lowerAttr = genSubrangeOp(kDimLowerBoundPos);
    aiir::LLVM::DIExpressionAttr countAttr = genSubrangeOp(kDimExtentPos);
    aiir::LLVM::DIExpressionAttr strideAttr = genSubrangeOp(kDimStridePos);

    auto subrangeTy = aiir::LLVM::DIGenericSubrangeAttr::get(
        context, countAttr, lowerAttr, /*upperBound=*/nullptr, strideAttr);
    elements.push_back(subrangeTy);

    return aiir::LLVM::DICompositeTypeAttr::get(
        context, llvm::dwarf::DW_TAG_array_type, /*name=*/nullptr,
        /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, elemTy,
        aiir::LLVM::DIFlags::Zero, /*sizeInBits=*/0, /*alignInBits=*/0,
        dataLocation, rank, /*allocated=*/nullptr,
        /*associated=*/nullptr, elements);
  }

  addOp(llvm::dwarf::DW_OP_push_object_address, {});
  addOp(llvm::dwarf::DW_OP_deref, {});
  addOp(llvm::dwarf::DW_OP_lit0, {});
  addOp(llvm::dwarf::DW_OP_ne, {});

  // allocated = associated = (*base_addr != 0)
  aiir::LLVM::DIExpressionAttr valid =
      aiir::LLVM::DIExpressionAttr::get(context, ops);
  aiir::LLVM::DIExpressionAttr allocated = genAllocated ? valid : nullptr;
  aiir::LLVM::DIExpressionAttr associated = genAssociated ? valid : nullptr;
  ops.clear();

  unsigned offset = dimsOffset;
  unsigned index = 0;
  aiir::IntegerType intTy = aiir::IntegerType::get(context, 64);
  const unsigned indexSize = dimsSize / 3;
  for ([[maybe_unused]] auto _ : seqTy.getShape()) {
    // For each dimension, find the offset of count, lower bound and stride in
    // the descriptor and generate the dwarf expression to extract it.
    aiir::Attribute lowerAttr = nullptr;
    // If declaration has a lower bound, use it.
    if (declOp && declOp.getShift().size() > index) {
      if (std::optional<std::int64_t> optint =
              getIntIfConstant(declOp.getShift()[index]))
        lowerAttr = aiir::IntegerAttr::get(intTy, llvm::APInt(64, *optint));
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
    aiir::LLVM::DIExpressionAttr countAttr =
        aiir::LLVM::DIExpressionAttr::get(context, ops);
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
      lowerAttr = aiir::LLVM::DIExpressionAttr::get(context, ops);
      ops.clear();
    }

    addOp(llvm::dwarf::DW_OP_push_object_address, {});
    addOp(llvm::dwarf::DW_OP_plus_uconst,
          {offset + (indexSize * kDimStridePos)});
    addOp(llvm::dwarf::DW_OP_deref, {});
    // stride[i] = *(base_addr + offset + (indexSize * kDimStridePos))
    aiir::LLVM::DIExpressionAttr strideAttr =
        aiir::LLVM::DIExpressionAttr::get(context, ops);
    ops.clear();

    offset += dimsSize;
    aiir::LLVM::DISubrangeAttr subrangeTy = aiir::LLVM::DISubrangeAttr::get(
        context, countAttr, lowerAttr, /*upperBound=*/nullptr, strideAttr);
    elements.push_back(subrangeTy);
    ++index;
  }
  return aiir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_array_type, /*name=*/nullptr,
      /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, elemTy,
      aiir::LLVM::DIFlags::Zero, /*sizeInBits=*/0, /*alignInBits=*/0,
      dataLocation, /*rank=*/nullptr, allocated, associated, elements);
}

std::pair<std::uint64_t, unsigned short>
DebugTypeGenerator::getFieldSizeAndAlign(aiir::Type fieldTy) {
  aiir::Type llvmTy;
  if (auto boxTy = aiir::dyn_cast_if_present<fir::BaseBoxType>(fieldTy))
    llvmTy = llvmTypeConverter.convertBoxTypeAsStruct(boxTy, getBoxRank(boxTy));
  else
    llvmTy = llvmTypeConverter.convertType(fieldTy);

  uint64_t byteSize = dataLayout->getTypeSize(llvmTy);
  unsigned short byteAlign = dataLayout->getTypeABIAlignment(llvmTy);
  return std::pair{byteSize, byteAlign};
}

aiir::LLVM::DITypeAttr DerivedTypeCache::lookup(aiir::Type type) {
  auto iter = typeCache.find(type);
  if (iter != typeCache.end()) {
    if (iter->second.first) {
      componentActiveRecursionLevels = iter->second.second;
    }
    return iter->second.first;
  }
  return nullptr;
}

DerivedTypeCache::ActiveLevels
DerivedTypeCache::startTranslating(aiir::Type type,
                                   aiir::LLVM::DITypeAttr placeHolder) {
  derivedTypeDepth++;
  if (!placeHolder)
    return {};
  typeCache[type] = std::pair<aiir::LLVM::DITypeAttr, ActiveLevels>(
      placeHolder, {derivedTypeDepth});
  return {};
}

void DerivedTypeCache::preComponentVisitUpdate() {
  componentActiveRecursionLevels.clear();
}

void DerivedTypeCache::postComponentVisitUpdate(
    ActiveLevels &activeRecursionLevels) {
  if (componentActiveRecursionLevels.empty())
    return;
  ActiveLevels oldLevels;
  oldLevels.swap(activeRecursionLevels);
  std::set_union(componentActiveRecursionLevels.begin(),
                 componentActiveRecursionLevels.end(), oldLevels.begin(),
                 oldLevels.end(), std::back_inserter(activeRecursionLevels));
}

void DerivedTypeCache::finalize(aiir::Type ty, aiir::LLVM::DITypeAttr attr,
                                ActiveLevels &&activeRecursionLevels) {
  // If there is no nested recursion or if this type does not point to any type
  // nodes above it, it is safe to cache it indefinitely (it can be used in any
  // contexts).
  if (activeRecursionLevels.empty() ||
      (activeRecursionLevels[0] == derivedTypeDepth)) {
    typeCache[ty] = std::pair<aiir::LLVM::DITypeAttr, ActiveLevels>(attr, {});
    componentActiveRecursionLevels.clear();
    cleanUpCache(derivedTypeDepth);
    --derivedTypeDepth;
    return;
  }
  // Trim any recursion below the current type.
  if (activeRecursionLevels.back() >= derivedTypeDepth) {
    auto last = llvm::find_if(activeRecursionLevels, [&](std::int32_t depth) {
      return depth >= derivedTypeDepth;
    });
    if (last != activeRecursionLevels.end()) {
      activeRecursionLevels.erase(last, activeRecursionLevels.end());
    }
  }
  componentActiveRecursionLevels = std::move(activeRecursionLevels);
  typeCache[ty] = std::pair<aiir::LLVM::DITypeAttr, ActiveLevels>(
      attr, componentActiveRecursionLevels);
  cleanUpCache(derivedTypeDepth);
  if (!componentActiveRecursionLevels.empty())
    insertCacheCleanUp(ty, componentActiveRecursionLevels.back());
  --derivedTypeDepth;
}

void DerivedTypeCache::insertCacheCleanUp(aiir::Type type, int32_t depth) {
  auto iter = llvm::find_if(cacheCleanupList,
                            [&](const auto &x) { return x.second >= depth; });
  if (iter == cacheCleanupList.end()) {
    cacheCleanupList.emplace_back(
        std::pair<llvm::SmallVector<aiir::Type>, int32_t>({type}, depth));
    return;
  }
  if (iter->second == depth) {
    iter->first.push_back(type);
    return;
  }
  cacheCleanupList.insert(
      iter, std::pair<llvm::SmallVector<aiir::Type>, int32_t>({type}, depth));
}

void DerivedTypeCache::cleanUpCache(int32_t depth) {
  if (cacheCleanupList.empty())
    return;
  // cleanups are done in the post actions when visiting a derived type
  // tree. So if there is a clean-up for the current depth, it has to be
  // the last one (deeper ones must have been done already).
  if (cacheCleanupList.back().second == depth) {
    for (aiir::Type type : cacheCleanupList.back().first)
      typeCache[type].first = nullptr;
    cacheCleanupList.pop_back_n(1);
  }
}

aiir::LLVM::DITypeAttr DebugTypeGenerator::convertRecordType(
    fir::RecordType Ty, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp) {

  if (aiir::LLVM::DITypeAttr attr = derivedTypeCache.lookup(Ty))
    return attr;

  aiir::AIIRContext *context = module.getContext();
  auto [nameKind, sourceName] = fir::NameUniquer::deconstruct(Ty.getName());
  if (nameKind != fir::NameUniquer::NameKind::DERIVED_TYPE)
    return genPlaceholderType(context);

  llvm::SmallVector<aiir::LLVM::DINodeAttr> elements;
  // Generate a place holder TypeAttr which will be used if a member
  // references the parent type.
  auto recId = aiir::DistinctAttr::create(aiir::UnitAttr::get(context));
  auto placeHolder = aiir::LLVM::DICompositeTypeAttr::get(
      context, recId, /*isRecSelf=*/true, llvm::dwarf::DW_TAG_structure_type,
      aiir::StringAttr::get(context, ""), fileAttr, /*line=*/0, scope,
      /*baseType=*/nullptr, aiir::LLVM::DIFlags::Zero, /*sizeInBits=*/0,
      /*alignInBits=*/0, /*dataLocation=*/nullptr, /*rank=*/nullptr,
      /*allocated=*/nullptr, /*associated=*/nullptr, elements);
  DerivedTypeCache::ActiveLevels nestedRecursions =
      derivedTypeCache.startTranslating(Ty, placeHolder);

  fir::TypeInfoOp tiOp = symbolTable->lookup<fir::TypeInfoOp>(Ty.getName());
  unsigned line = (tiOp) ? getLineFromLoc(tiOp.getLoc()) : 1;

  aiir::OpBuilder builder(context);
  aiir::IntegerType intTy = aiir::IntegerType::get(context, 64);
  std::uint64_t offset = 0;
  for (auto [fieldName, fieldTy] : Ty.getTypeList()) {
    derivedTypeCache.preComponentVisitUpdate();
    auto [byteSize, byteAlign] = getFieldSizeAndAlign(fieldTy);
    std::optional<llvm::ArrayRef<int64_t>> lowerBounds =
        fir::getComponentLowerBoundsIfNonDefault(Ty, fieldName, module,
                                                 symbolTable);
    auto seqTy = aiir::dyn_cast_if_present<fir::SequenceType>(fieldTy);

    // For members of the derived types, the information about the shift in
    // lower bounds is not part of the declOp but has to be extracted from the
    // TypeInfoOp (using getComponentLowerBoundsIfNonDefault).
    aiir::LLVM::DITypeAttr elemTy;
    if (lowerBounds && seqTy &&
        lowerBounds->size() == seqTy.getShape().size()) {
      llvm::SmallVector<aiir::LLVM::DINodeAttr> arrayElements;
      for (auto [bound, dim] :
           llvm::zip_equal(*lowerBounds, seqTy.getShape())) {
        auto countAttr = aiir::IntegerAttr::get(intTy, llvm::APInt(64, dim));
        auto lowerAttr = aiir::IntegerAttr::get(intTy, llvm::APInt(64, bound));
        auto subrangeTy = aiir::LLVM::DISubrangeAttr::get(
            context, countAttr, lowerAttr, /*upperBound=*/nullptr,
            /*stride=*/nullptr);
        arrayElements.push_back(subrangeTy);
      }
      elemTy = aiir::LLVM::DICompositeTypeAttr::get(
          context, llvm::dwarf::DW_TAG_array_type, /*name=*/nullptr,
          /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr,
          convertType(seqTy.getEleTy(), fileAttr, scope, declOp),
          aiir::LLVM::DIFlags::Zero, /*sizeInBits=*/0, /*alignInBits=*/0,
          /*dataLocation=*/nullptr, /*rank=*/nullptr,
          /*allocated=*/nullptr, /*associated=*/nullptr, arrayElements);
    } else
      elemTy = convertType(fieldTy, fileAttr, scope, /*declOp=*/nullptr);
    offset = llvm::alignTo(offset, byteAlign);
    aiir::LLVM::DIDerivedTypeAttr tyAttr = aiir::LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_member,
        aiir::StringAttr::get(context, fieldName), /*file=*/nullptr, /*line=*/0,
        /*scope=*/nullptr, elemTy, byteSize * 8, byteAlign * 8, offset * 8,
        /*optional<address space>=*/std::nullopt,
        /*flags=*/aiir::LLVM::DIFlags::Zero,
        /*extra data=*/nullptr);
    elements.push_back(tyAttr);
    offset += llvm::alignTo(byteSize, byteAlign);
    derivedTypeCache.postComponentVisitUpdate(nestedRecursions);
  }

  auto finalAttr = aiir::LLVM::DICompositeTypeAttr::get(
      context, recId, /*isRecSelf=*/false, llvm::dwarf::DW_TAG_structure_type,
      aiir::StringAttr::get(context, sourceName.name), fileAttr, line, scope,
      /*baseType=*/nullptr, aiir::LLVM::DIFlags::Zero, offset * 8,
      /*alignInBits=*/0, /*dataLocation=*/nullptr, /*rank=*/nullptr,
      /*allocated=*/nullptr, /*associated=*/nullptr, elements);

  derivedTypeCache.finalize(Ty, finalAttr, std::move(nestedRecursions));

  return finalAttr;
}

aiir::LLVM::DITypeAttr DebugTypeGenerator::convertTupleType(
    aiir::TupleType Ty, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp) {
  // Check if this type has already been converted.
  if (aiir::LLVM::DITypeAttr attr = derivedTypeCache.lookup(Ty))
    return attr;

  DerivedTypeCache::ActiveLevels nestedRecursions =
      derivedTypeCache.startTranslating(Ty);

  llvm::SmallVector<aiir::LLVM::DINodeAttr> elements;
  aiir::AIIRContext *context = module.getContext();

  std::uint64_t offset = 0;
  for (auto fieldTy : Ty.getTypes()) {
    derivedTypeCache.preComponentVisitUpdate();
    auto [byteSize, byteAlign] = getFieldSizeAndAlign(fieldTy);
    aiir::LLVM::DITypeAttr elemTy =
        convertType(fieldTy, fileAttr, scope, /*declOp=*/nullptr);
    offset = llvm::alignTo(offset, byteAlign);
    aiir::LLVM::DIDerivedTypeAttr tyAttr = aiir::LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_member, aiir::StringAttr::get(context, ""),
        /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, elemTy, byteSize * 8,
        byteAlign * 8, offset * 8,
        /*optional<address space>=*/std::nullopt,
        /*flags=*/aiir::LLVM::DIFlags::Zero,
        /*extra data=*/nullptr);
    elements.push_back(tyAttr);
    offset += llvm::alignTo(byteSize, byteAlign);
    derivedTypeCache.postComponentVisitUpdate(nestedRecursions);
  }

  auto typeAttr = aiir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_structure_type,
      aiir::StringAttr::get(context, ""), fileAttr, /*line=*/0, scope,
      /*baseType=*/nullptr, aiir::LLVM::DIFlags::Zero, offset * 8,
      /*alignInBits=*/0, /*dataLocation=*/nullptr, /*rank=*/nullptr,
      /*allocated=*/nullptr, /*associated=*/nullptr, elements);
  derivedTypeCache.finalize(Ty, typeAttr, std::move(nestedRecursions));
  return typeAttr;
}

aiir::LLVM::DITypeAttr DebugTypeGenerator::convertSequenceType(
    fir::SequenceType seqTy, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp) {
  aiir::AIIRContext *context = module.getContext();

  llvm::SmallVector<aiir::LLVM::DINodeAttr> elements;
  aiir::LLVM::DITypeAttr elemTy =
      convertType(seqTy.getEleTy(), fileAttr, scope, declOp);

  unsigned index = 0;
  auto intTy = aiir::IntegerType::get(context, 64);
  for (fir::SequenceType::Extent dim : seqTy.getShape()) {
    aiir::Attribute lowerAttr = nullptr;
    aiir::Attribute countAttr = nullptr;
    // If declOp is present, we use the shift in it to get the lower bound of
    // the array. If it is constant, that is used. If it is not constant, we
    // create a variable that represents its location and use that as lower
    // bound. As an optimization, we don't create a lower bound when shift is a
    // constant 1 as that is the default.
    if (declOp && declOp.getShift().size() > index) {
      if (std::optional<std::int64_t> optint =
              getIntIfConstant(declOp.getShift()[index])) {
        if (*optint != 1)
          lowerAttr = aiir::IntegerAttr::get(intTy, llvm::APInt(64, *optint));
      } else
        lowerAttr = generateArtificialVariable(
            context, declOp.getShift()[index], fileAttr, scope, declOp);
    }

    if (dim == seqTy.getUnknownExtent()) {
      // This path is taken for both assumed size array or when the size of the
      // array is variable. In the case of variable size, we create a variable
      // to use as countAttr.
      if (declOp && declOp.getShape().size() > index) {
        if (!llvm::isa_and_nonnull<fir::AssumedSizeExtentOp>(
                declOp.getShape()[index].getDefiningOp()))
          countAttr = generateArtificialVariable(
              context, declOp.getShape()[index], fileAttr, scope, declOp);
      }
    } else
      countAttr = aiir::IntegerAttr::get(intTy, llvm::APInt(64, dim));

    auto subrangeTy = aiir::LLVM::DISubrangeAttr::get(
        context, countAttr, lowerAttr, /*upperBound=*/nullptr,
        /*stride=*/nullptr);
    elements.push_back(subrangeTy);
    ++index;
  }
  // Apart from arrays, the `DICompositeTypeAttr` is used for other things like
  // structure types. Many of its fields which are not applicable to arrays
  // have been set to some valid default values.

  return aiir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_array_type, /*name=*/nullptr,
      /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, elemTy,
      aiir::LLVM::DIFlags::Zero, /*sizeInBits=*/0, /*alignInBits=*/0,
      /*dataLocation=*/nullptr, /*rank=*/nullptr, /*allocated=*/nullptr,
      /*associated=*/nullptr, elements);
}

aiir::LLVM::DITypeAttr DebugTypeGenerator::convertVectorType(
    fir::VectorType vecTy, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp) {
  aiir::AIIRContext *context = module.getContext();

  llvm::SmallVector<aiir::LLVM::DINodeAttr> elements;
  aiir::LLVM::DITypeAttr elemTy =
      convertType(vecTy.getEleTy(), fileAttr, scope, declOp);
  auto intTy = aiir::IntegerType::get(context, 64);
  auto countAttr =
      aiir::IntegerAttr::get(intTy, llvm::APInt(64, vecTy.getLen()));
  auto subrangeTy = aiir::LLVM::DISubrangeAttr::get(
      context, countAttr, /*lowerBound=*/nullptr, /*upperBound=*/nullptr,
      /*stride=*/nullptr);
  elements.push_back(subrangeTy);
  aiir::Type llvmTy = llvmTypeConverter.convertType(vecTy.getEleTy());
  uint64_t sizeInBits = dataLayout->getTypeSize(llvmTy) * vecTy.getLen() * 8;
  std::string name("vector");
  // The element type of the vector must be integer or real so it will be a
  // DIBasicTypeAttr.
  if (auto ty = aiir::dyn_cast_if_present<aiir::LLVM::DIBasicTypeAttr>(elemTy))
    name += " " + ty.getName().str();

  name += " (" + std::to_string(vecTy.getLen()) + ")";
  return aiir::LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_array_type,
      aiir::StringAttr::get(context, name),
      /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, elemTy,
      aiir::LLVM::DIFlags::Vector, sizeInBits, /*alignInBits=*/0,
      /*dataLocation=*/nullptr, /*rank=*/nullptr, /*allocated=*/nullptr,
      /*associated=*/nullptr, elements);
}

aiir::LLVM::DITypeAttr DebugTypeGenerator::convertCharacterType(
    fir::CharacterType charTy, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp,
    bool hasDescriptor) {
  aiir::AIIRContext *context = module.getContext();

  // DWARF 5 says the following about the character encoding in 5.1.1.2.
  // "DW_ATE_ASCII and DW_ATE_UCS specify encodings for the Fortran 2003
  // string kinds ASCII (ISO/IEC 646:1991) and ISO_10646 (UCS-4 in ISO/IEC
  // 10646:2000)."
  unsigned encoding = llvm::dwarf::DW_ATE_ASCII;
  if (charTy.getFKind() != 1)
    encoding = llvm::dwarf::DW_ATE_UCS;

  uint64_t sizeInBits = 0;
  aiir::LLVM::DIExpressionAttr lenExpr = nullptr;
  aiir::LLVM::DIExpressionAttr locExpr = nullptr;
  aiir::LLVM::DIVariableAttr varAttr = nullptr;

  if (hasDescriptor) {
    llvm::SmallVector<aiir::LLVM::DIExpressionElemAttr> ops;
    auto addOp = [&](unsigned opc, llvm::ArrayRef<uint64_t> vals) {
      ops.push_back(aiir::LLVM::DIExpressionElemAttr::get(context, opc, vals));
    };
    addOp(llvm::dwarf::DW_OP_push_object_address, {});
    addOp(llvm::dwarf::DW_OP_plus_uconst, {lenOffset});
    lenExpr = aiir::LLVM::DIExpressionAttr::get(context, ops);
    ops.clear();

    addOp(llvm::dwarf::DW_OP_push_object_address, {});
    addOp(llvm::dwarf::DW_OP_deref, {});
    locExpr = aiir::LLVM::DIExpressionAttr::get(context, ops);
  } else if (charTy.hasConstantLen()) {
    sizeInBits =
        charTy.getLen() * kindMapping.getCharacterBitsize(charTy.getFKind());
  } else {
    // In assumed length string, the len of the character is not part of the
    // type but can be found at the runtime. Here we create an artificial
    // variable that will contain that length. This variable is used as
    // 'stringLength' in DIStringTypeAttr.
    if (declOp && !declOp.getTypeparams().empty()) {
      aiir::LLVM::DILocalVariableAttr lvAttr = generateArtificialVariable(
          context, declOp.getTypeparams()[0], fileAttr, scope, declOp);
      varAttr = aiir::cast<aiir::LLVM::DIVariableAttr>(lvAttr);
    }
  }

  // FIXME: Currently the DIStringType in llvm does not have the option to set
  // type of the underlying character. This restricts out ability to represent
  // string with non-default characters. Please see issue #95440 for more
  // details.
  return aiir::LLVM::DIStringTypeAttr::get(
      context, llvm::dwarf::DW_TAG_string_type,
      aiir::StringAttr::get(context, ""), sizeInBits, /*alignInBits=*/0,
      /*stringLength=*/varAttr, lenExpr, locExpr, encoding);
}

aiir::LLVM::DITypeAttr DebugTypeGenerator::convertPointerLikeType(
    aiir::Type elTy, aiir::LLVM::DIFileAttr fileAttr,
    aiir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp,
    bool genAllocated, bool genAssociated) {
  aiir::AIIRContext *context = module.getContext();

  // Arrays and character need different treatment because DWARF have special
  // constructs for them to get the location from the descriptor. Rest of
  // types are handled like pointer to underlying type.
  if (auto seqTy = aiir::dyn_cast_if_present<fir::SequenceType>(elTy))
    return convertBoxedSequenceType(seqTy, fileAttr, scope, declOp,
                                    genAllocated, genAssociated);
  if (auto charTy = aiir::dyn_cast_if_present<fir::CharacterType>(elTy))
    return convertCharacterType(charTy, fileAttr, scope, declOp,
                                /*hasDescriptor=*/true);

  // If elTy is null or none then generate a void*
  aiir::LLVM::DITypeAttr elTyAttr;
  if (!elTy || aiir::isa<aiir::NoneType>(elTy))
    elTyAttr = aiir::LLVM::DINullTypeAttr::get(context);
  else
    elTyAttr = convertType(elTy, fileAttr, scope, declOp);

  return aiir::LLVM::DIDerivedTypeAttr::get(
      context, llvm::dwarf::DW_TAG_pointer_type,
      aiir::StringAttr::get(context, ""), /*file=*/nullptr, /*line=*/0,
      /*scope=*/nullptr, elTyAttr, /*sizeInBits=*/ptrSize * 8,
      /*alignInBits=*/0, /*offset=*/0,
      /*optional<address space>=*/std::nullopt,
      /*flags=*/aiir::LLVM::DIFlags::Zero, /*extra data=*/nullptr);
}

aiir::LLVM::DITypeAttr
DebugTypeGenerator::convertType(aiir::Type Ty, aiir::LLVM::DIFileAttr fileAttr,
                                aiir::LLVM::DIScopeAttr scope,
                                fir::cg::XDeclareOp declOp) {
  aiir::AIIRContext *context = module.getContext();
  if (Ty.isInteger()) {
    unsigned bitWidth = Ty.getIntOrFloatBitWidth();
    return genBasicType(context, getBasicTypeName(context, "integer", bitWidth),
                        bitWidth, llvm::dwarf::DW_ATE_signed);
  } else if (aiir::isa<aiir::FloatType>(Ty)) {
    unsigned bitWidth = Ty.getIntOrFloatBitWidth();
    return genBasicType(context, getBasicTypeName(context, "real", bitWidth),
                        bitWidth, llvm::dwarf::DW_ATE_float);
  } else if (auto logTy = aiir::dyn_cast_if_present<fir::LogicalType>(Ty)) {
    unsigned bitWidth = kindMapping.getLogicalBitsize(logTy.getFKind());
    return genBasicType(
        context, getBasicTypeName(context, logTy.getMnemonic(), bitWidth),
        bitWidth, llvm::dwarf::DW_ATE_boolean);
  } else if (auto cplxTy = aiir::dyn_cast_if_present<aiir::ComplexType>(Ty)) {
    auto floatTy = aiir::cast<aiir::FloatType>(cplxTy.getElementType());
    unsigned bitWidth = floatTy.getWidth();
    return genBasicType(context, getBasicTypeName(context, "complex", bitWidth),
                        bitWidth * 2, llvm::dwarf::DW_ATE_complex_float);
  } else if (auto seqTy = aiir::dyn_cast_if_present<fir::SequenceType>(Ty)) {
    return convertSequenceType(seqTy, fileAttr, scope, declOp);
  } else if (auto charTy = aiir::dyn_cast_if_present<fir::CharacterType>(Ty)) {
    return convertCharacterType(charTy, fileAttr, scope, declOp,
                                /*hasDescriptor=*/false);
  } else if (auto recTy = aiir::dyn_cast_if_present<fir::RecordType>(Ty)) {
    return convertRecordType(recTy, fileAttr, scope, declOp);
  } else if (auto tupleTy = aiir::dyn_cast_if_present<aiir::TupleType>(Ty)) {
    return convertTupleType(tupleTy, fileAttr, scope, declOp);
  } else if (aiir::isa<aiir::FunctionType>(Ty)) {
    // Handle function types - these represent procedure pointers after the
    // BoxedProcedure pass has run and unwrapped the fir.boxproc type, as well
    // as dummy procedures (which are represented as function types in FIR)
    llvm::SmallVector<aiir::LLVM::DITypeAttr> types;

    auto funcTy = aiir::cast<aiir::FunctionType>(Ty);
    // Add return type (or void if no return type)
    if (funcTy.getNumResults() == 0)
      types.push_back(aiir::LLVM::DINullTypeAttr::get(context));
    else
      types.push_back(
          convertType(funcTy.getResult(0), fileAttr, scope, declOp));

    for (aiir::Type paramTy : funcTy.getInputs())
      types.push_back(convertType(paramTy, fileAttr, scope, declOp));

    auto subroutineTy = aiir::LLVM::DISubroutineTypeAttr::get(
        context, /*callingConvention=*/0, types);

    return aiir::LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_pointer_type,
        aiir::StringAttr::get(context, ""), /*file=*/nullptr, /*line=*/0,
        /*scope=*/nullptr, subroutineTy,
        /*sizeInBits=*/ptrSize * 8, /*alignInBits=*/0, /*offset=*/0,
        /*optional<address space>=*/std::nullopt,
        /*flags=*/aiir::LLVM::DIFlags::Zero, /*extra data=*/nullptr);
  } else if (auto refTy = aiir::dyn_cast_if_present<fir::ReferenceType>(Ty)) {
    auto elTy = refTy.getEleTy();
    return convertPointerLikeType(elTy, fileAttr, scope, declOp,
                                  /*genAllocated=*/false,
                                  /*genAssociated=*/false);
  } else if (auto vecTy = aiir::dyn_cast_if_present<fir::VectorType>(Ty)) {
    return convertVectorType(vecTy, fileAttr, scope, declOp);
  } else if (aiir::isa<aiir::IndexType>(Ty)) {
    unsigned bitWidth = llvmTypeConverter.getIndexTypeBitwidth();
    return genBasicType(context, getBasicTypeName(context, "integer", bitWidth),
                        bitWidth, llvm::dwarf::DW_ATE_signed);
  } else if (auto boxTy = aiir::dyn_cast_if_present<fir::BaseBoxType>(Ty)) {
    auto elTy = boxTy.getEleTy();
    if (auto seqTy = aiir::dyn_cast_if_present<fir::SequenceType>(elTy))
      return convertBoxedSequenceType(seqTy, fileAttr, scope, declOp, false,
                                      false);
    if (auto heapTy = aiir::dyn_cast_if_present<fir::HeapType>(elTy))
      return convertPointerLikeType(heapTy.getElementType(), fileAttr, scope,
                                    declOp, /*genAllocated=*/true,
                                    /*genAssociated=*/false);
    if (auto ptrTy = aiir::dyn_cast_if_present<fir::PointerType>(elTy))
      return convertPointerLikeType(ptrTy.getElementType(), fileAttr, scope,
                                    declOp, /*genAllocated=*/false,
                                    /*genAssociated=*/true);
    return convertPointerLikeType(elTy, fileAttr, scope, declOp,
                                  /*genAllocated=*/false,
                                  /*genAssociated=*/false);
  } else {
    // FIXME: These types are currently unhandled. We are generating a
    // placeholder type to allow us to test supported bits.
    return genPlaceholderType(context);
  }
}

} // namespace fir
