//===-- Character.h -- lowering of characters -------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_BUILDER_CHARACTER_H
#define FORTRAN_OPTIMIZER_BUILDER_CHARACTER_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace fir::factory {

/// Helper to facilitate lowering of CHARACTER in FIR.
class CharacterExprHelper {
public:
  /// Constructor.
  explicit CharacterExprHelper(FirOpBuilder &builder, mlir::Location loc)
      : builder{builder}, loc{loc} {}
  CharacterExprHelper(const CharacterExprHelper &) = delete;

  /// Copy the \p count first characters of \p src into \p dest.
  /// \p count can have any integer type.
  void createCopy(const fir::CharBoxValue &dest, const fir::CharBoxValue &src,
                  mlir::Value count);

  /// Set characters of \p str at position [\p lower, \p upper) to blanks.
  /// \p lower and \upper bounds are zero based.
  /// If \p upper <= \p lower, no padding is done.
  /// \p upper and \p lower can have any integer type.
  void createPadding(const fir::CharBoxValue &str, mlir::Value lower,
                     mlir::Value upper);

  /// Create str(lb:ub), lower bounds must always be specified, upper
  /// bound is optional.
  fir::CharBoxValue createSubstring(const fir::CharBoxValue &str,
                                    llvm::ArrayRef<mlir::Value> bounds);

  /// Return blank character of given \p type !fir.char<kind>
  mlir::Value createBlankConstant(fir::CharacterType type);

  /// Lower \p lhs = \p rhs where \p lhs and \p rhs are scalar characters.
  /// It handles cases where \p lhs and \p rhs may overlap.
  void createAssign(const fir::ExtendedValue &lhs,
                    const fir::ExtendedValue &rhs);

  /// Create lhs // rhs in temp obtained with fir.alloca
  fir::CharBoxValue createConcatenate(const fir::CharBoxValue &lhs,
                                      const fir::CharBoxValue &rhs);

  /// LEN_TRIM intrinsic.
  mlir::Value createLenTrim(const fir::CharBoxValue &str);

  /// Embox \p addr and \p len and return fir.boxchar.
  /// Take care of type conversions before emboxing.
  /// \p len is converted to the integer type for character lengths if needed.
  mlir::Value createEmboxChar(mlir::Value addr, mlir::Value len);
  /// Create a fir.boxchar for \p str. If \p str is not in memory, a temp is
  /// allocated to create the fir.boxchar.
  mlir::Value createEmbox(const fir::CharBoxValue &str);
  /// Embox a string array. Note that the size/shape of the array is not
  /// retrievable from the resulting mlir::Value.
  mlir::Value createEmbox(const fir::CharArrayBoxValue &str);

  /// Convert character array to a scalar by reducing the extents into the
  /// length. Will fail if call on non reference like base.
  fir::CharBoxValue toScalarCharacter(const fir::CharArrayBoxValue &);

  /// Unbox \p boxchar into (fir.ref<fir.char<kind>>, character length type).
  std::pair<mlir::Value, mlir::Value> createUnboxChar(mlir::Value boxChar);

  /// Allocate a temp of fir::CharacterType type and length len.
  /// Returns related fir.ref<fir.array<? x fir.char<kind>>>.
  fir::CharBoxValue createCharacterTemp(mlir::Type type, mlir::Value len);

  /// Allocate a temp of compile time constant length.
  /// Returns related fir.ref<fir.array<len x fir.char<kind>>>.
  fir::CharBoxValue createCharacterTemp(mlir::Type type, int len);

  /// Create a temporary with the same kind, length, and value as source.
  fir::CharBoxValue createTempFrom(const fir::ExtendedValue &source);

  /// Return true if \p type is a character literal type (is
  /// `fir.array<len x fir.char<kind>>`).;
  static bool isCharacterLiteral(mlir::Type type);

  /// Return true if \p type is one of the following type
  /// - fir.boxchar<kind>
  /// - fir.ref<fir.char<kind,len>>
  /// - fir.char<kind,len>
  static bool isCharacterScalar(mlir::Type type);

  /// Does this extended value base type is fir.char<kind,len>
  /// where len is not the unknown extent ?
  static bool hasConstantLengthInType(const fir::ExtendedValue &);

  /// Extract the kind of a character type
  static fir::KindTy getCharacterKind(mlir::Type type);

  /// Extract the kind of a character or array of character type.
  static fir::KindTy getCharacterOrSequenceKind(mlir::Type type);

  /// Determine the base character type
  static fir::CharacterType getCharacterType(mlir::Type type);
  static fir::CharacterType getCharacterType(const fir::CharBoxValue &box);
  static fir::CharacterType getCharacterType(mlir::Value str);

  /// Create an extended value from a value of type:
  /// - fir.boxchar<kind>
  /// - fir.ref<fir.char<kind,len>>
  /// - fir.char<kind,len>
  /// or the array versions:
  /// - fir.ref<fir.array<n x...x fir.char<kind,len>>>
  /// - fir.array<n x...x fir.char<kind,len>>
  ///
  /// Does the heavy lifting of converting the value \p character (along with an
  /// optional \p len value) to an extended value. If \p len is null, a length
  /// value is extracted from \p character (or its type). This will produce an
  /// error if it's not possible. The returned value is a CharBoxValue if \p
  /// character is a scalar, otherwise it is a CharArrayBoxValue.
  fir::ExtendedValue toExtendedValue(mlir::Value character,
                                     mlir::Value len = {});

  /// Is `type` a sequence (array) of CHARACTER type? Return true for any of the
  /// following cases:
  ///   - !fir.array<dim x ... x !fir.char<kind, len>>
  ///   - !fir.ref<T>  where T is either of the first case
  ///   - !fir.box<T>  where T is either of the first case
  ///
  /// In certain contexts, Fortran allows an array of CHARACTERs to be treated
  /// as if it were one longer CHARACTER scalar, each element append to the
  /// previous.
  static bool isArray(mlir::Type type);

  /// Temporary helper to help migrating towards properties of
  /// ExtendedValue containing characters.
  /// Mainly, this ensure that characters are always CharArrayBoxValue,
  /// CharBoxValue, or BoxValue and that the base address is not a boxchar.
  /// Return the argument if this is not a character.
  /// TODO: Create and propagate ExtendedValue according to properties listed
  /// above instead of fixing it when needed.
  fir::ExtendedValue cleanUpCharacterExtendedValue(const fir::ExtendedValue &);

  /// Create fir.char<kind> singleton from \p code integer value.
  mlir::Value createSingletonFromCode(mlir::Value code, int kind);
  /// Returns integer value held in a character singleton.
  mlir::Value extractCodeFromSingleton(mlir::Value singleton);

  /// Create a value for the length of a character based on its memory reference
  /// that may be a boxchar, box or !fir.[ptr|ref|heap]<fir.char<kind, len>>. If
  /// the memref is a simple address and the length is not constant in type, the
  /// returned length will be empty.
  mlir::Value getLength(mlir::Value memref);

  /// Compute length given a fir.box describing a character entity.
  /// It adjusts the length from the number of bytes per the descriptor
  /// to the number of characters per the Fortran KIND.
  mlir::Value readLengthFromBox(mlir::Value box);

private:
  /// FIXME: the implementation also needs a clean-up now that
  /// CharBoxValue are better propagated.
  fir::CharBoxValue materializeValue(mlir::Value str);
  mlir::Value getCharBoxBuffer(const fir::CharBoxValue &box);
  mlir::Value createElementAddr(mlir::Value buffer, mlir::Value index);
  mlir::Value createLoadCharAt(mlir::Value buff, mlir::Value index);
  void createStoreCharAt(mlir::Value str, mlir::Value index, mlir::Value c);
  void createLengthOneAssign(const fir::CharBoxValue &lhs,
                             const fir::CharBoxValue &rhs);
  void createAssign(const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs);
  mlir::Value createBlankConstantCode(fir::CharacterType type);

  FirOpBuilder &builder;
  mlir::Location loc;
};

// FIXME: Move these to Optimizer
mlir::FuncOp getLlvmMemcpy(FirOpBuilder &builder);
mlir::FuncOp getLlvmMemmove(FirOpBuilder &builder);
mlir::FuncOp getLlvmMemset(FirOpBuilder &builder);
mlir::FuncOp getRealloc(FirOpBuilder &builder);

//===----------------------------------------------------------------------===//
// Tools to work with Character dummy procedures
//===----------------------------------------------------------------------===//

/// Create a tuple<function type, length type> type to pass character functions
/// as arguments along their length. The function type set in the tuple is the
/// one provided by \p funcPointerType.
mlir::Type getCharacterProcedureTupleType(mlir::Type funcPointerType);

/// Is this tuple type holding a character function and its result length ?
bool isCharacterProcedureTuple(mlir::Type type);

/// Is \p tuple a value holding a character function address and its result
/// length ?
inline bool isCharacterProcedureTuple(mlir::Value tuple) {
  return isCharacterProcedureTuple(tuple.getType());
}

/// Create a tuple<addr, len> given \p addr and \p len as well as the tuple
/// type \p argTy. \p addr must be any function address, and \p len must be
/// any integer. Converts will be inserted if needed if \addr and \p len
/// types are not the same as the one inside the tuple type \p tupleType.
mlir::Value createCharacterProcedureTuple(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          mlir::Type tupleType,
                                          mlir::Value addr, mlir::Value len);

/// Given a tuple containing a character function address and its result length,
/// extract the tuple into a pair of value <function address, result length>.
std::pair<mlir::Value, mlir::Value>
extractCharacterProcedureTuple(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value tuple);

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_CHARACTER_H
