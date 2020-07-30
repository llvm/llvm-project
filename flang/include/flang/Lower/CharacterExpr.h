//===-- Lower/CharacterExpr.h -- lowering of characters ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CHARACTEREXPR_H
#define FORTRAN_LOWER_CHARACTEREXPR_H

#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"

namespace Fortran::lower {

/// Helper to facilitate lowering of CHARACTER in FIR.
class CharacterExprHelper {
public:
  /// Constructor.
  explicit CharacterExprHelper(FirOpBuilder &builder, mlir::Location loc)
      : builder{builder}, loc{loc} {}
  CharacterExprHelper(const CharacterExprHelper &) = delete;

  /// Unless otherwise stated, all mlir::Value inputs of these pseudo-fir ops
  /// must be of type:
  /// - fir.boxchar<kind> (dynamic length character),
  /// - fir.ref<fir.array<len x fir.char<kind>>> (character with compile time
  ///      constant length),
  /// - fir.array<len x fir.char<kind>> (compile time constant character)

  /// Copy the \p count first characters of \p src into \p dest.
  /// \p count can have any integer type.
  void createCopy(mlir::Value dest, mlir::Value src, mlir::Value count);

  /// Set characters of \p str at position [\p lower, \p upper) to blanks.
  /// \p lower and \upper bounds are zero based.
  /// If \p upper <= \p lower, no padding is done.
  /// \p upper and \p lower can have any integer type.
  void createPadding(mlir::Value str, mlir::Value lower, mlir::Value upper);

  /// Create str(lb:ub), lower bounds must always be specified, upper
  /// bound is optional.
  mlir::Value createSubstring(mlir::Value str,
                              llvm::ArrayRef<mlir::Value> bounds);

  /// Return blank character of given \p type !fir.char<kind>
  mlir::Value createBlankConstant(fir::CharacterType type);

  /// Lower \p lhs = \p rhs where \p lhs and \p rhs are scalar characters.
  /// It handles cases where \p lhs and \p rhs may overlap.
  void createAssign(mlir::Value lhs, mlir::Value rhs);

  /// Lower an assignment where the buffer and LEN parameter are known and do
  /// not need to be unboxed.
  void createAssign(mlir::Value lptr, mlir::Value llen, mlir::Value rptr,
                    mlir::Value rlen);

  /// Create lhs // rhs in temp obtained with fir.alloca
  mlir::Value createConcatenate(mlir::Value lhs, mlir::Value rhs);

  /// LEN_TRIM intrinsic.
  mlir::Value createLenTrim(mlir::Value str);

  /// Embox \p addr and \p len and return fir.boxchar.
  /// Take care of type conversions before emboxing.
  /// \p len is converted to the integer type for character lengths if needed.
  mlir::Value createEmboxChar(mlir::Value addr, mlir::Value len);

  /// Unbox \p boxchar into (fir.ref<fir.char<kind>>, getLengthType()).
  std::pair<mlir::Value, mlir::Value> createUnboxChar(mlir::Value boxChar);

  /// Allocate a temp of fir::CharacterType type and length len.
  /// Returns related fir.ref<fir.char<kind>>.
  mlir::Value createCharacterTemp(mlir::Type type, mlir::Value len);

  /// Allocate a temp of compile time constant length.
  /// Returns related fir.ref<fir.array<len x fir.char<kind>>>.
  mlir::Value createCharacterTemp(mlir::Type type, int len) {
    return createTemp(type, len);
  }

  /// Return buffer/length pair of character str, if str is a constant,
  /// it is allocated into a temp, otherwise, its memory reference is
  /// returned as the buffer.
  /// The buffer type of str is of type:
  ///   - fir.ref<fir.array<len x fir.char<kind>>> if str has compile time
  ///      constant length.
  ///   - fir.ref<fir.char<kind>> if str has dynamic length.
  std::pair<mlir::Value, mlir::Value> materializeCharacter(mlir::Value str);

  /// Return true if \p type is a character literal type (is
  /// fir.array<len x fir.char<kind>>).;
  static bool isCharacterLiteral(mlir::Type type);

  /// Return true if \p type is one of the following type
  /// - fir.boxchar<kind>
  /// - fir.ref<fir.array<len x fir.char<kind>>>
  /// - fir.array<len x fir.char<kind>>
  static bool isCharacter(mlir::Type type);

  /// Extract the kind of a character type
  static int getCharacterKind(mlir::Type type);

  /// Return the integer type that must be used to manipulate
  /// Character lengths. TODO: move this to FirOpBuilder?
  mlir::Type getLengthType() { return builder.getIndexType(); }

  /// Create an extended value from:
  /// - fir.boxchar<kind>
  /// - fir.ref<fir.array<len x fir.char<kind>>>
  /// - fir.array<len x fir.char<kind>>
  /// - fir.char<kind>
  /// - fir.ref<char<kind>>
  /// If the no length is passed, it is attempted to be extracted from \p
  /// character (or its type). This will crash if this is not possible.
  /// The returned value is a CharBoxValue if \p character is a scalar,
  /// otherwise it is a CharArrayBoxValue.
  fir::ExtendedValue toExtendedValue(mlir::Value character,
                                     mlir::Value len = {});

private:
  fir::CharBoxValue materializeValue(const fir::CharBoxValue &str);
  fir::CharBoxValue toDataLengthPair(mlir::Value character);
  mlir::Type getReferenceType(const fir::CharBoxValue &c) const;
  mlir::Value createEmbox(const fir::CharBoxValue &str);
  mlir::Value createLoadCharAt(const fir::CharBoxValue &str, mlir::Value index);
  void createStoreCharAt(const fir::CharBoxValue &str, mlir::Value index,
                         mlir::Value c);
  void createCopy(const fir::CharBoxValue &dest, const fir::CharBoxValue &src,
                  mlir::Value count);
  void createPadding(const fir::CharBoxValue &str, mlir::Value lower,
                     mlir::Value upper);
  fir::CharBoxValue createTemp(mlir::Type type, mlir::Value len);
  void createLengthOneAssign(const fir::CharBoxValue &lhs,
                             const fir::CharBoxValue &rhs);
  void createAssign(const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs);
  fir::CharBoxValue createConcatenate(const fir::CharBoxValue &lhs,
                                      const fir::CharBoxValue &rhs);
  fir::CharBoxValue createSubstring(const fir::CharBoxValue &str,
                                    llvm::ArrayRef<mlir::Value> bounds);
  mlir::Value createLenTrim(const fir::CharBoxValue &str);
  mlir::Value createTemp(mlir::Type type, int len);
  mlir::Value createBlankConstantCode(fir::CharacterType type);

private:
  FirOpBuilder &builder;
  mlir::Location loc;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CHARACTEREXPR_H
