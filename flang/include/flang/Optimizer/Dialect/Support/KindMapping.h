//===-- Optimizer/Support/KindMapping.h -- support kind mapping -*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_SUPPORT_KINDMAPPING_H
#define FORTRAN_OPTIMIZER_SUPPORT_KINDMAPPING_H

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
/// 'Half', 'BFloat', 'Float', 'Double', 'X86_FP80', or 'FP128' (for 'r' and
/// 'c').
///
/// If LLVM adds support for new floating-point types, the final list should be
/// extended.
class KindMapping {
public:
  using KindTy = unsigned;
  using Bitsize = unsigned;
  using LLVMTypeID = llvm::Type::TypeID;
  using MatchResult = mlir::ParseResult;

  /// KindMapping constructor with both the kind map and default kinds read from
  /// command-line options.
  explicit KindMapping(mlir::MLIRContext *context);
  /// KindMapping constructor taking a `defs` argument to specify the default
  /// kinds for intrinsic types. To set the default kinds, an ArrayRef of 6
  /// KindTy must be passed. The kinds must be the given in the following order:
  /// CHARACTER, COMPLEX, DOUBLE PRECISION, INTEGER, LOGICAL, and REAL.  The
  /// kind map is read from command-line options, if given.
  explicit KindMapping(mlir::MLIRContext *context, llvm::ArrayRef<KindTy> defs);
  /// KindMapping constructor taking an optional `defs` argument to specify the
  /// default kinds for intrinsic types. To set the default kinds, an ArrayRef
  /// of 6 KindTy must be passed. The kinds must be the given in the following
  /// order: CHARACTER, COMPLEX, DOUBLE PRECISION, INTEGER, LOGICAL, and REAL.
  explicit KindMapping(mlir::MLIRContext *context, llvm::StringRef map,
                       llvm::ArrayRef<KindTy> defs = std::nullopt);
  explicit KindMapping(mlir::MLIRContext *context, llvm::StringRef map,
                       llvm::StringRef defs)
      : KindMapping{context, map, toDefaultKinds(defs)} {}

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

  /// Get the default kind map as a string.
  static constexpr const char *getDefaultMap() { return ""; }

  /// Convert the current kind map to a string.
  std::string mapToString() const;

  //===--------------------------------------------------------------------===//
  // Default kinds of intrinsic types
  //===--------------------------------------------------------------------===//

  KindTy defaultCharacterKind() const;
  KindTy defaultComplexKind() const;
  KindTy defaultDoubleKind() const;
  KindTy defaultIntegerKind() const;
  KindTy defaultLogicalKind() const;
  KindTy defaultRealKind() const;

  /// Get the default kinds as a string.
  static constexpr const char *getDefaultKinds() { return "a1c4d8i4l4r4"; }

  /// Convert the current default kinds to a string.
  std::string defaultsToString() const;

  /// Translate a default kinds string into a default kind vector. This vector
  /// can be passed to the KindMapping ctor.
  static std::vector<KindTy> toDefaultKinds(llvm::StringRef defs);

private:
  MatchResult badMapString(const llvm::Twine &ptr);
  MatchResult parse(llvm::StringRef kindMap);
  mlir::LogicalResult setDefaultKinds(llvm::ArrayRef<KindTy> defs);

  mlir::MLIRContext *context;
  llvm::DenseMap<std::pair<char, KindTy>, Bitsize> intMap;
  llvm::DenseMap<std::pair<char, KindTy>, LLVMTypeID> floatMap;
  llvm::DenseMap<char, KindTy> defaultMap;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_KINDMAPPING_H
