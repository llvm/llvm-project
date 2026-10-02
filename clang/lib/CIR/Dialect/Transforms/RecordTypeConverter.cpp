//===- RecordTypeConverter.cpp - Record-rebuilding type converter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RecordTypeConverter.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Threading.h"

#include <mutex>
#include <shared_mutex>

using namespace cir;

RecordRewritingTypeConverter::RecordRewritingTypeConverter(
    mlir::MLIRContext &context)
    : context(context) {
  addConversion([&](mlir::Type type) -> mlir::Type { return type; });
  // This is necessary in order to convert CIR pointer types that are pointing
  // to CIR types that are being converted.
  addConversion([&](cir::PointerType type) -> mlir::Type {
    mlir::Type loweredPointeeType = convertType(type.getPointee());
    if (!loweredPointeeType)
      return {};
    return cir::PointerType::get(type.getContext(), loweredPointeeType,
                                 type.getAddrSpace());
  });
  addConversion([&](cir::ArrayType type) -> mlir::Type {
    mlir::Type loweredElementType = convertType(type.getElementType());
    if (!loweredElementType)
      return {};
    return cir::ArrayType::get(loweredElementType, type.getSize());
  });
  // This is necessary in order to convert CIR function types that have
  // argument or return types that use CIR types that are being converted.
  addConversion([&](cir::FuncType type) -> mlir::Type {
    llvm::SmallVector<mlir::Type> loweredInputTypes;
    loweredInputTypes.reserve(type.getNumInputs());
    if (mlir::failed(convertTypes(type.getInputs(), loweredInputTypes)))
      return {};

    mlir::Type loweredReturnType = convertType(type.getReturnType());
    if (!loweredReturnType)
      return {};

    return cir::FuncType::get(loweredInputTypes, loweredReturnType,
                              /*isVarArg=*/type.getVarArg());
  });
  addConversion([&](cir::StructType type) -> mlir::Type {
    return convertRecordType(type);
  });
  addConversion([&](cir::UnionType type) -> mlir::Type {
    return convertRecordType(type);
  });
}

void RecordRewritingTypeConverter::restoreRecordTypeNames() {
  std::unique_lock<decltype(recordTypeMutex)> lock(recordTypeMutex);

  for (auto rt : convertedRecordTypes)
    rt.removeABIConversionNamePrefix();
}

// This provides a stack for the RecordTypes being processed on the current
// thread, which lets us solve recursive conversions. This implementation is
// cribbed from the LLVMTypeConverter which solves a similar but not identical
// problem.
llvm::SmallVector<cir::RecordType> &
RecordRewritingTypeConverter::getCurrentThreadRecursiveStack() {
  {
    // Most of the time, the entry already exists in the map.
    std::shared_lock<decltype(callStackMutex)> lock(callStackMutex,
                                                    std::defer_lock);
    if (context.isMultithreadingEnabled())
      lock.lock();
    auto recursiveStack = conversionCallStack.find(llvm::get_threadid());
    if (recursiveStack != conversionCallStack.end())
      return *recursiveStack->second;
  }

  // First time this thread gets here, we have to get an exclusive access to
  // insert in the map
  std::unique_lock<decltype(callStackMutex)> lock(callStackMutex);
  auto recursiveStackInserted = conversionCallStack.insert(
      std::make_pair(llvm::get_threadid(),
                     std::make_unique<llvm::SmallVector<cir::RecordType>>()));
  return *recursiveStackInserted.first->second;
}

void RecordRewritingTypeConverter::addConvertedRecordType(cir::RecordType rt) {
  std::unique_lock<decltype(recordTypeMutex)> lock(recordTypeMutex);
  convertedRecordTypes.push_back(rt);
}

llvm::SmallVector<mlir::Type>
RecordRewritingTypeConverter::convertRecordMemberTypes(cir::RecordType type) {
  llvm::SmallVector<mlir::Type> loweredMemberTypes;
  loweredMemberTypes.reserve(type.getNumElements());

  if (mlir::failed(convertTypes(type.getMembers(), loweredMemberTypes)))
    return {};

  return loweredMemberTypes;
}

cir::RecordType
RecordRewritingTypeConverter::convertRecordType(cir::RecordType type) {
  if (!shouldConvertRecord(type))
    return type;

  // Unnamed record types can't be referred to recursively, so we can just
  // convert this one. It also doesn't have uniqueness problems, so we can
  // just do a conversion on it.
  if (!type.getName()) {
    llvm::SmallVector<mlir::Type> converted = convertRecordMemberTypes(type);
    assert(converted.size() == type.getNumElements() &&
           "member conversion must be one type in, one type out for the "
           "kinds to carry over by index");
    if (auto u = mlir::dyn_cast<cir::UnionType>(type)) {
      mlir::Type loweredPadding;
      if (mlir::Type pad = u.getPadding())
        loweredPadding = convertType(pad);
      return cir::UnionType::get(type.getContext(), converted, type.getPacked(),
                                 loweredPadding, u.getMemberKinds());
    }
    auto s = mlir::cast<cir::StructType>(type);
    return cir::StructType::get(type.getContext(), converted, type.getPacked(),
                                s.getIsClass(), s.getMemberKinds());
  }

  assert(!type.isIncomplete() || type.getMembers().empty());

  // If the type has already been converted, we can just return, since there
  // is nothing to do. Also, if it is incomplete, it can't have invalid
  // members! So we can skip transforming it.
  if (type.isIncomplete() || type.isABIConvertedRecord())
    return type;

  llvm::SmallVectorImpl<cir::RecordType> &recursiveStack =
      getCurrentThreadRecursiveStack();

  cir::RecordType convertedType;
  if (mlir::isa<cir::UnionType>(type))
    convertedType =
        cir::UnionType::get(type.getContext(), type.getABIConvertedName());
  else
    convertedType =
        cir::StructType::get(type.getContext(), type.getABIConvertedName(),
                             mlir::cast<cir::StructType>(type).getIsClass());

  // This type has already been converted, just return it.
  if (convertedType.isComplete())
    return convertedType;

  // We put the existing 'type' into the vector if we're in the process of
  // converting it (and pop it when we're done).  To prevent recursion,
  // just return the 'incomplete' version, and the 'top level' version of this
  // call will call 'complete' on it.
  if (llvm::is_contained(recursiveStack, type))
    return convertedType;

  recursiveStack.push_back(type);
  llvm::scope_exit popConvertingType(
      [&recursiveStack]() { recursiveStack.pop_back(); });

  llvm::SmallVector<mlir::Type> convertedMembers =
      convertRecordMemberTypes(type);
  assert(convertedMembers.size() == type.getNumElements() &&
         "member conversion must be one type in, one type out for the kinds "
         "to carry over by index");

  mlir::Type loweredPadding;
  if (auto u = mlir::dyn_cast<cir::UnionType>(type))
    if (mlir::Type pad = u.getPadding())
      loweredPadding = convertType(pad);
  convertedType.complete(convertedMembers, type.getPacked(), loweredPadding,
                         type.getMemberKinds());
  addConvertedRecordType(convertedType);
  return convertedType;
}
