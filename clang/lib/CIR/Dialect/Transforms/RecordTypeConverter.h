//===- RecordTypeConverter.h - Record-rebuilding type converter -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A type converter base for CIR-to-CIR passes that rewrite types nested in
// records.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_RECORDTYPECONVERTER_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_RECORDTYPECONVERTER_H

#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/RWMutex.h"

#include <memory>

namespace cir {

/// A type converter that converts the types nested in pointers, arrays,
/// functions and records.
///
/// Identified records can't be modified in place, so each one is rebuilt under
/// a temporary name (see RecordType::getABIConvertedName). The conversion keeps
/// a stack of the records being converted, which resolves recursive records.
/// Once the conversion is done, restoreRecordTypeNames() gives the rebuilt
/// records their original names back.
///
/// Subclasses add conversions for the types they rewrite. The type converter
/// tries the most recently added conversion first, so a subclass can also
/// replace one of the conversions registered here.
class RecordRewritingTypeConverter : public mlir::TypeConverter {
public:
  explicit RecordRewritingTypeConverter(mlir::MLIRContext &context);
  virtual ~RecordRewritingTypeConverter() = default;

  /// Remove the temporary name of every record rebuilt by this converter.
  void restoreRecordTypeNames();

protected:
  /// Whether \p type has to be rebuilt. Rebuilding every record is always
  /// correct, but makes every operation that uses one illegal.
  virtual bool shouldConvertRecord(cir::RecordType type) { return true; }

private:
  cir::RecordType convertRecordType(cir::RecordType type);
  llvm::SmallVector<mlir::Type> convertRecordMemberTypes(cir::RecordType type);
  llvm::SmallVector<cir::RecordType> &getCurrentThreadRecursiveStack();
  void addConvertedRecordType(cir::RecordType rt);

  mlir::MLIRContext &context;

  // Recursive structure detection.
  // We store one entry per thread here, and rely on locking. This works the
  // same way as the LLVM-IR lowering does it, which has a similar problem.
  llvm::DenseMap<uint64_t, std::unique_ptr<llvm::SmallVector<cir::RecordType>>>
      conversionCallStack;
  llvm::sys::SmartRWMutex<true> callStackMutex;

  // In order to let us 'change the names' back after the fact, we collect them
  // along the way. They should only be added/accessed via the thread-safe
  // functions.
  llvm::SmallVector<cir::RecordType> convertedRecordTypes;
  llvm::sys::SmartRWMutex<true> recordTypeMutex;
};

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_RECORDTYPECONVERTER_H
