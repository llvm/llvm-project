//===-- Optimizer/Support/KindMapping.h -- support kind mapping -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_SUPPORT_KINDMAPPING_H
#define OPTIMIZER_SUPPORT_KINDMAPPING_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Type.h"

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace fir {

/// The kind mapping is an encoded string that informs FIR how the Fortran KIND
/// values from the front-end should be converted to LLVM IR types.  This
/// encoding allows the mapping from front-end KIND values to backend LLVM IR
/// types to be customized by the front-end.
///
/// The provided string uses the following syntax.
///
///   intrinsic-key `:` kind-value (`,` intrinsic-key `:` kind-value)*
///
/// intrinsic-key is a single character for the intrinsic type.
///   'i' : INTEGER   (size in bits)
///   'l' : LOGICAL   (size in bits)
///   'a' : CHARACTER (size in bits)
///   'r' : REAL    (encoding value)
///   'c' : COMPLEX (encoding value)
///
/// kind-value is either an unsigned integer (for 'i', 'l', and 'a') or one of
/// 'Half', 'Float', 'Double', 'X86_FP80', or 'FP128' (for 'r' and 'c').
///
/// If LLVM adds support for new floating-point types, the final list should be
/// extended.
class KindMapping {
public:
  using KindTy = unsigned;
  using Bitsize = unsigned;
  using LLVMTypeID = llvm::Type::TypeID;
  using MatchResult = mlir::ParseResult;

  explicit KindMapping(mlir::MLIRContext *context);
  explicit KindMapping(mlir::MLIRContext *context, llvm::StringRef map);

  /// Get the size in bits of !fir.char<kind>
  Bitsize getCharacterBitsize(KindTy kind) const;

  /// Get the size in bits of !fir.int<kind>
  Bitsize getIntegerBitsize(KindTy kind) const;

  /// Get the size in bits of !fir.logical<kind>
  Bitsize getLogicalBitsize(KindTy kind) const;

  /// Get the size in bits of !fir.real<kind>
  Bitsize getRealBitsize(KindTy kind) const;

  /// Get the LLVM Type::TypeID of !fir.real<kind>
  LLVMTypeID getRealTypeID(KindTy kind) const;

  /// Get the LLVM Type::TypeID of !fir.complex<kind>
  LLVMTypeID getComplexTypeID(KindTy kind) const;

  mlir::MLIRContext *getContext() const { return context; }

  /// Get the float semantics of !fir.real<kind>
  const llvm::fltSemantics &getFloatSemantics(KindTy kind) const;

private:
  MatchResult badMapString(const llvm::Twine &ptr);
  MatchResult parse(llvm::StringRef kindMap);

  mlir::MLIRContext *context;
  llvm::DenseMap<std::pair<char, KindTy>, Bitsize> intMap;
  llvm::DenseMap<std::pair<char, KindTy>, LLVMTypeID> floatMap;
};

} // namespace fir

#endif // OPTIMIZER_SUPPORT_KINDMAPPING_H
