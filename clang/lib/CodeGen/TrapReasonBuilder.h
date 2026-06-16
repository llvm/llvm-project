//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of TrapReasonBuilder and related classes.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_CODEGEN_TRAP_REASON_BUILDER_H
#define LLVM_CLANG_CODEGEN_TRAP_REASON_BUILDER_H
#include "clang/Basic/Diagnostic.h"

namespace clang {
namespace CodeGen {

/// Helper class for \class TrapReasonBuilder. \class TrapReason stores the
/// "trap reason" built by \class TrapReasonBuilder. This consists of
/// a trap message and trap category.
///
/// It is intended that this object be allocated on the stack.
class TrapReason {
public:
  TrapReason() = default;
  /// \return The trap message. Note the lifetime of the underlying storage for
  /// the returned StringRef lives in this class which means the returned
  /// StringRef should not be used after this class is destroyed.
  StringRef getMessage() const { return Message; }

  /// \return the trap category (e.g. "Undefined Behavior Sanitizer")
  StringRef getCategory() const { return Category; }

  bool isEmpty() const {
    // Note both Message and Category are checked because it is legitimate for
    // the Message to be empty but for the Category to be non-empty when the
    // trap category is known but the specific reason is not available during
    // codegen.
    return Message.size() == 0 && Category.size() == 0;
  }

private:
  llvm::SmallString<64> Message;
  // The Category doesn't need its own storage because the StringRef points
  // to a global constant string.
  StringRef Category;

  // Only this class can set the private fields.
  friend class TrapReasonBuilder;
};

/// Class to make it convenient to initialize TrapReason objects which can be
/// used to attach the "trap reason" to trap instructions.
///
/// Although this class inherits from \class DiagnosticBuilder it has slightly
/// different semantics.
///
/// * This class should only be used with trap diagnostics (declared in
/// `DiagnosticTrapKinds.td`).
/// * The `TrapReasonBuilder` does not emit diagnostics to the normal
///   diagnostics consumers on destruction like normal Diagnostic builders.
///   Instead on destruction it assigns to the TrapReason object passed into
///   the constructor.
///
/// Given that this class inherits from `DiagnosticBuilder` it inherits all of
/// its abilities to format diagnostic messages and consume various types in
/// class (e.g. Type, Exprs, etc.). This makes it particularly suited to
/// printing types and expressions from the AST while codegen-ing runtime
/// checks.
///
///
/// Example use via the `CodeGenModule::BuildTrapReason` helper.
///
/// \code
/// {
///   TrapReason TR;
///   CGM.BuildTrapReason(diag::trap_diagnostic, TR) << 0 << SomeExpr;
///   consume(&TR);
/// }
/// \endcode
///
///
class TrapReasonBuilder : public DiagnosticBuilder {
public:
  TrapReasonBuilder(DiagnosticsEngine *DiagObj, unsigned DiagID,
                    TrapReason &TR);
  ~TrapReasonBuilder();

  // Prevent accidentally copying or assigning
  TrapReasonBuilder &operator=(const TrapReasonBuilder &) = delete;
  TrapReasonBuilder &operator=(const TrapReasonBuilder &&) = delete;
  TrapReasonBuilder(const TrapReasonBuilder &) = delete;
  TrapReasonBuilder(const TrapReasonBuilder &&) = delete;

private:
  /// \return Format the trap message into `Storage`.
  void getMessage(SmallVectorImpl<char> &Storage);

  /// \return Return the trap category. These are the `CategoryName` property
  /// of `trap` diagnostics declared in `DiagnosticTrapKinds.td`.
  StringRef getCategory();

private:
  TrapReason &TR;
};

} // namespace CodeGen
} // namespace clang

#endif
