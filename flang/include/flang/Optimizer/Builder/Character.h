//===-- Character.h -- lowering of characters -------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_BUILDER_CHARACTER_H
#define FORTRAN_OPTIMIZER_BUILDER_CHARACTER_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"

namespace fir {
class FirOpBuilder;
}

namespace fir::factory {

/// Helper to facilitate lowering of CHARACTER in FIR.
class CharacterExprHelper {
public:
  /// Constructor.
  explicit CharacterExprHelper(FirOpBuilder &builder, aiir::Location loc)
      : builder{builder}, loc{loc} {}
  CharacterExprHelper(const CharacterExprHelper &) = delete;

  /// Copy the \p count first characters of \p src into \p dest.
  /// \p count can have any integer type.
  void createCopy(const fir::CharBoxValue &dest, const fir::CharBoxValue &src,
                  aiir::Value count);

  /// Set characters of \p str at position [\p lower, \p upper) to blanks.
  /// \p lower and \upper bounds are zero based.
  /// If \p upper <= \p lower, no padding is done.
  /// \p upper and \p lower can have any integer type.
  void createPadding(const fir::CharBoxValue &str, aiir::Value lower,
                     aiir::Value upper);

  /// Create str(lb:ub), lower bounds must always be specified, upper
  /// bound is optional.
  fir::CharBoxValue createSubstring(const fir::CharBoxValue &str,
                                    llvm::ArrayRef<aiir::Value> bounds);

  /// Compute substring base address given the raw address (not fir.boxchar) of
  /// a scalar string, a substring / lower bound, and the substring type.
  aiir::Value genSubstringBase(aiir::Value stringRawAddr,
                               aiir::Value lowerBound,
                               aiir::Type substringAddrType,
                               aiir::Value one = {});

  /// Return blank character of given \p type !fir.char<kind>
  aiir::Value createBlankConstant(fir::CharacterType type);

  /// Lower \p lhs = \p rhs where \p lhs and \p rhs are scalar characters.
  /// It handles cases where \p lhs and \p rhs may overlap.
  void createAssign(const fir::ExtendedValue &lhs,
                    const fir::ExtendedValue &rhs);

  /// Create lhs // rhs in temp obtained with fir.alloca
  fir::CharBoxValue createConcatenate(const fir::CharBoxValue &lhs,
                                      const fir::CharBoxValue &rhs);

  /// Create {max,min}(lhs,rhs) in temp obtained with fir.alloca
  fir::CharBoxValue
  createCharExtremum(bool predIsMin, llvm::ArrayRef<fir::CharBoxValue> opCBVs);

  /// LEN_TRIM intrinsic.
  aiir::Value createLenTrim(const fir::CharBoxValue &str);

  /// Embox \p addr and \p len and return fir.boxchar.
  /// Take care of type conversions before emboxing.
  /// \p len is converted to the integer type for character lengths if needed.
  aiir::Value createEmboxChar(aiir::Value addr, aiir::Value len);
  /// Create a fir.boxchar for \p str. If \p str is not in memory, a temp is
  /// allocated to create the fir.boxchar.
  aiir::Value createEmbox(const fir::CharBoxValue &str);
  /// Embox a string array. Note that the size/shape of the array is not
  /// retrievable from the resulting aiir::Value.
  aiir::Value createEmbox(const fir::CharArrayBoxValue &str);

  /// Convert character array to a scalar by reducing the extents into the
  /// length. Will fail if call on non reference like base.
  fir::CharBoxValue toScalarCharacter(const fir::CharArrayBoxValue &);

  /// Unbox \p boxchar into (fir.ref<fir.char<kind>>, character length type).
  std::pair<aiir::Value, aiir::Value> createUnboxChar(aiir::Value boxChar);

  /// Allocate a temp of fir::CharacterType type and length len.
  /// Returns related fir.ref<fir.array<? x fir.char<kind>>>.
  fir::CharBoxValue createCharacterTemp(aiir::Type type, aiir::Value len);

  /// Allocate a temp of compile time constant length.
  /// Returns related fir.ref<fir.array<len x fir.char<kind>>>.
  fir::CharBoxValue createCharacterTemp(aiir::Type type, int len);

  /// Create a temporary with the same kind, length, and value as source.
  fir::CharBoxValue createTempFrom(const fir::ExtendedValue &source);

  /// Return true if \p type is a character literal type (is
  /// `fir.array<len x fir.char<kind>>`).;
  static bool isCharacterLiteral(aiir::Type type);

  /// Return true if \p type is one of the following type
  /// - fir.boxchar<kind>
  /// - fir.ref<fir.char<kind,len>>
  /// - fir.char<kind,len>
  static bool isCharacterScalar(aiir::Type type);

  /// Does this extended value base type is fir.char<kind,len>
  /// where len is not the unknown extent ?
  static bool hasConstantLengthInType(const fir::ExtendedValue &);

  /// Extract the kind of a character type
  static fir::KindTy getCharacterKind(aiir::Type type);

  /// Extract the kind of a character or array of character type.
  static fir::KindTy getCharacterOrSequenceKind(aiir::Type type);

  // TODO: Do we really need all these flavors of unwrapping to get the fir.char
  // type? Or can we merge these? It would be better to merge them and eliminate
  // the confusion.

  /// Determine the inner character type. Unwraps references, boxes, and
  /// sequences to find the !fir.char element type.
  static fir::CharacterType getCharType(aiir::Type type);

  /// Get fir.char<kind> type with the same kind as inside str.
  static fir::CharacterType getCharacterType(aiir::Type type);
  static fir::CharacterType getCharacterType(const fir::CharBoxValue &box);
  static fir::CharacterType getCharacterType(aiir::Value str);

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
  fir::ExtendedValue toExtendedValue(aiir::Value character,
                                     aiir::Value len = {});

  /// Is `type` a sequence (array) of CHARACTER type? Return true for any of the
  /// following cases:
  ///   - !fir.array<dim x ... x !fir.char<kind, len>>
  ///   - !fir.ref<T>  where T is either of the first case
  ///   - !fir.box<T>  where T is either of the first case
  ///
  /// In certain contexts, Fortran allows an array of CHARACTERs to be treated
  /// as if it were one longer CHARACTER scalar, each element append to the
  /// previous.
  static bool isArray(aiir::Type type);

  /// Temporary helper to help migrating towards properties of
  /// ExtendedValue containing characters.
  /// Mainly, this ensure that characters are always CharArrayBoxValue,
  /// CharBoxValue, or BoxValue and that the base address is not a boxchar.
  /// Return the argument if this is not a character.
  /// TODO: Create and propagate ExtendedValue according to properties listed
  /// above instead of fixing it when needed.
  fir::ExtendedValue cleanUpCharacterExtendedValue(const fir::ExtendedValue &);

  /// Create fir.char<kind> singleton from \p code integer value.
  aiir::Value createSingletonFromCode(aiir::Value code, int kind);
  /// Returns integer value held in a character singleton.
  aiir::Value extractCodeFromSingleton(aiir::Value singleton);

  /// Create a value for the length of a character based on its memory reference
  /// that may be a boxchar, box or !fir.[ptr|ref|heap]<fir.char<kind, len>>. If
  /// the memref is a simple address and the length is not constant in type, the
  /// returned length will be empty.
  aiir::Value getLength(aiir::Value memref);

  /// Compute length given a fir.box describing a character entity.
  /// It adjusts the length from the number of bytes per the descriptor
  /// to the number of characters per the Fortran KIND.
  aiir::Value readLengthFromBox(aiir::Value box);

  /// Same as readLengthFromBox but the CharacterType is provided.
  aiir::Value readLengthFromBox(aiir::Value box, fir::CharacterType charTy);

private:
  /// FIXME: the implementation also needs a clean-up now that
  /// CharBoxValue are better propagated.
  fir::CharBoxValue materializeValue(aiir::Value str);
  aiir::Value getCharBoxBuffer(const fir::CharBoxValue &box);
  aiir::Value createElementAddr(aiir::Value buffer, aiir::Value index);
  aiir::Value createLoadCharAt(aiir::Value buff, aiir::Value index);
  void createStoreCharAt(aiir::Value str, aiir::Value index, aiir::Value c);
  void createLengthOneAssign(const fir::CharBoxValue &lhs,
                             const fir::CharBoxValue &rhs);
  void createAssign(const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs);
  aiir::Value createBlankConstantCode(fir::CharacterType type);

private:
  FirOpBuilder &builder;
  aiir::Location loc;
};

//===----------------------------------------------------------------------===//
// Tools to work with Character dummy procedures
//===----------------------------------------------------------------------===//

/// Create a tuple<function type, length type> type to pass character functions
/// as arguments along their length. The function type set in the tuple is the
/// one provided by \p funcPointerType.
aiir::Type getCharacterProcedureTupleType(aiir::Type funcPointerType);

/// Create a tuple<addr, len> given \p addr and \p len as well as the tuple
/// type \p argTy. \p addr must be any function address, and \p len may be any
/// integer or nullptr. Converts will be inserted if needed if \addr and \p len
/// types are not the same as the one inside the tuple type \p tupleType.
aiir::Value createCharacterProcedureTuple(fir::FirOpBuilder &builder,
                                          aiir::Location loc,
                                          aiir::Type tupleType,
                                          aiir::Value addr, aiir::Value len);

/// Given a tuple containing a character function address and its result length,
/// extract the tuple into a pair of value <function address, result length>.
/// If openBoxProc is true, the function address is extracted from the
/// fir.boxproc, otherwise, the returned function address is the fir.boxproc.
std::pair<aiir::Value, aiir::Value>
extractCharacterProcedureTuple(fir::FirOpBuilder &builder, aiir::Location loc,
                               aiir::Value tuple, bool openBoxProc = true);

fir::CharBoxValue convertCharacterKind(fir::FirOpBuilder &builder,
                                       aiir::Location loc,
                                       fir::CharBoxValue srcBoxChar,
                                       int toKind);

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_CHARACTER_H
