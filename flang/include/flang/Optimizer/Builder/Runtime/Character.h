//===-- Character.h -- generate calls to character runtime API --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CHARACTER_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CHARACTER_H

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"

namespace fir {
class ExtendedValue;
class FirOpBuilder;
} // namespace fir

namespace fir::runtime {

/// Generate a call to the `ADJUSTL` runtime.
/// This calls the simple runtime entry point that then calls into the more
/// complex runtime cases handling left or right adjustments.
///
/// \p resultBox must be an unallocated allocatable used for the temporary
/// result. \p StringBox must be a `fir.box` describing the `ADJUSTL` string
/// argument. Note that the \p genAdjust() helper is called to do the majority
/// of the lowering work.
void genAdjustL(fir::FirOpBuilder &builder, aiir::Location loc,
                aiir::Value resultBox, aiir::Value stringBox);

/// Generate a call to the `ADJUSTR` runtime.
/// This calls the simple runtime entry point that then calls into the more
/// complex runtime cases handling left or right adjustments.
///
/// \p resultBox must be an unallocated allocatable used for the temporary
/// result.  \p StringBox must be a fir.box describing the adjustr string
/// argument. Note that the \p genAdjust() helper is called to do the majority
/// of the lowering work.
void genAdjustR(fir::FirOpBuilder &builder, aiir::Location loc,
                aiir::Value resultBox, aiir::Value stringBox);

/// Generate call to a character comparison for two ssa-values of type
/// `boxchar`.
aiir::Value genCharCompare(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::arith::CmpIPredicate cmp,
                           const fir::ExtendedValue &lhs,
                           const fir::ExtendedValue &rhs);

/// Generate call to a character comparison op for two unboxed variables. There
/// are 4 arguments, 2 for the lhs and 2 for the rhs. Each CHARACTER must pass a
/// reference to its buffer (`ref<char<K>>`) and its LEN type parameter (some
/// integral type).
aiir::Value genCharCompare(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::arith::CmpIPredicate cmp, aiir::Value lhsBuff,
                           aiir::Value lhsLen, aiir::Value rhsBuff,
                           aiir::Value rhsLen);

/// Generate call to F_C_STRING intrinsic runtime routine
/// This appends a null character to a Fortran character string to create
/// a C-compatible null-terminated string.
///
/// \p resultBox must be an unallocated allocatable used for the temporary
/// result. \p stringBox must be a fir.box describing the F_C_STRING string
/// argument. \p asis must be a boxed logical value (fir.box<i1>) or an
/// AbsentOp: if true, trailing blanks are kept; if false or absent (default),
/// trailing blanks are trimmed before appending the null.
/// The runtime will always allocate the resultBox.
void genFCString(fir::FirOpBuilder &builder, aiir::Location loc,
                 aiir::Value resultBox, aiir::Value stringBox,
                 aiir::Value asis);

/// Generate call to INDEX runtime.
/// This calls the simple runtime entry points based on the KIND of the string.
/// No descriptors are used.
aiir::Value genIndex(fir::FirOpBuilder &builder, aiir::Location loc, int kind,
                     aiir::Value stringBase, aiir::Value stringLen,
                     aiir::Value substringBase, aiir::Value substringLen,
                     aiir::Value back);

/// Generate call to INDEX runtime.
/// This calls the simple runtime entry points based on the KIND of the string.
/// A version of interface taking a `boxchar` for string and substring.
/// Uses no-descriptors flow.
aiir::Value genIndex(fir::FirOpBuilder &builder, aiir::Location loc,
                     const fir::ExtendedValue &str,
                     const fir::ExtendedValue &substr, aiir::Value back);

/// Generate call to INDEX runtime.
/// This calls the descriptor based runtime call implementation for the index
/// intrinsic.
void genIndexDescriptor(fir::FirOpBuilder &builder, aiir::Location loc,
                        aiir::Value resultBox, aiir::Value stringBox,
                        aiir::Value substringBox, aiir::Value backOpt,
                        aiir::Value kind);

/// Generate call to repeat runtime.
///   \p resultBox must be an unallocated allocatable used for the temporary
///   result. \p stringBox must be a fir.box describing repeat string argument.
///   \p ncopies must be a value representing the number of copies.
/// The runtime will always allocate the resultBox.
void genRepeat(fir::FirOpBuilder &builder, aiir::Location loc,
               aiir::Value resultBox, aiir::Value stringBox,
               aiir::Value ncopies);

/// Generate call to trim runtime.
///   \p resultBox must be an unallocated allocatable used for the temporary
///   result. \p stringBox must be a fir.box describing trim string argument.
/// The runtime will always allocate the resultBox.
void genTrim(fir::FirOpBuilder &builder, aiir::Location loc,
             aiir::Value resultBox, aiir::Value stringBox);

/// Generate call to scan runtime.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsics.
void genScanDescriptor(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value resultBox, aiir::Value stringBox,
                       aiir::Value setBox, aiir::Value backBox,
                       aiir::Value kind);

/// Generate call to the scan runtime routine that is specialized on
/// \param kind.
/// The \param kind represents the kind of the elements in the strings.
aiir::Value genScan(fir::FirOpBuilder &builder, aiir::Location loc, int kind,
                    aiir::Value stringBase, aiir::Value stringLen,
                    aiir::Value setBase, aiir::Value setLen, aiir::Value back);

/// Generate call to verify runtime.
/// This calls the descriptor based runtime call implementation of the scan
/// intrinsics.
void genVerifyDescriptor(fir::FirOpBuilder &builder, aiir::Location loc,
                         aiir::Value resultBox, aiir::Value stringBox,
                         aiir::Value setBox, aiir::Value backBox,
                         aiir::Value kind);

/// Generate call to the verify runtime routine that is specialized on
/// \param kind.
/// The \param kind represents the kind of the elements in the strings.
aiir::Value genVerify(fir::FirOpBuilder &builder, aiir::Location loc, int kind,
                      aiir::Value stringBase, aiir::Value stringLen,
                      aiir::Value setBase, aiir::Value setLen,
                      aiir::Value back);

/// Generate call to the SPLIT runtime routine that is specialized on
/// \param kind.
/// The \param kind represents the kind of the elements in the strings.
/// Updates \p pos to the next separator position.
aiir::Value genSplit(fir::FirOpBuilder &builder, aiir::Location loc, int kind,
                     aiir::Value stringBase, aiir::Value stringLen,
                     aiir::Value setBase, aiir::Value setLen, aiir::Value pos,
                     aiir::Value back);

/// Generate call to TOKENIZE runtime (Form 1).
/// Splits \p stringBox into tokens based on separator characters in \p setBox.
/// \p tokensBox must be an unallocated allocatable array that receives the
/// token substrings. \p separatorBox is optional and receives separator chars.
void genTokenize(fir::FirOpBuilder &builder, aiir::Location loc,
                 aiir::Value tokensBox, aiir::Value separatorBox,
                 aiir::Value stringBox, aiir::Value setBox);

/// Generate call to TOKENIZE runtime (Form 2).
/// Returns token positions rather than substrings.
/// \p firstBox and \p lastBox must be unallocated allocatable integer arrays
/// that receive the starting and ending positions of each token.
void genTokenizePositions(fir::FirOpBuilder &builder, aiir::Location loc,
                          aiir::Value firstBox, aiir::Value lastBox,
                          aiir::Value stringBox, aiir::Value setBox);

} // namespace fir::runtime

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CHARACTER_H
