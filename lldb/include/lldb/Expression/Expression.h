//===-- Expression.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_EXPRESSION_EXPRESSION_H
#define LLDB_EXPRESSION_EXPRESSION_H

#include <map>
#include <string>
#include <vector>

#include "llvm/Support/FormatProviders.h"

#include "lldb/Expression/ExpressionTypeSystemHelper.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

/// \class Expression Expression.h "lldb/Expression/Expression.h" Encapsulates
/// a single expression for use in lldb
///
/// LLDB uses expressions for various purposes, notably to call functions
/// and as a backend for the expr command.  Expression encapsulates the
/// objects needed to parse and interpret or JIT an expression.  It uses the
/// expression parser appropriate to the language of the expression to produce
/// LLVM IR from the expression.
class Expression {
public:
  enum ResultType { eResultTypeAny, eResultTypeId };

  Expression(Target &target);

  Expression(ExecutionContextScope &exe_scope);

  /// Destructor
  virtual ~Expression() = default;

  /// Return the string that the parser should parse.  Must be a full
  /// translation unit.
  virtual const char *Text() = 0;

  /// Return the function name that should be used for executing the
  /// expression.  Text() should contain the definition of this function.
  virtual const char *FunctionName() = 0;

  /// Return the language that should be used when parsing.
  virtual SourceLanguage Language() const { return {}; }

  /// Return the Materializer that the parser should use when registering
  /// external values.
  virtual Materializer *GetMaterializer() { return nullptr; }

  /// Return the desired result type of the function, or eResultTypeAny if
  /// indifferent.
  virtual ResultType DesiredResultType() const { return eResultTypeAny; }

  /// Flags

  /// Return true if validation code should be inserted into the expression.
  virtual bool NeedsValidation() = 0;

  /// Return true if external variables in the expression should be resolved.
  virtual bool NeedsVariableResolution() = 0;

  virtual EvaluateExpressionOptions *GetOptions() { return nullptr; };

  /// Return the address of the function's JIT-compiled code, or
  /// LLDB_INVALID_ADDRESS if the function is not JIT compiled
  lldb::addr_t StartAddress() { return m_jit_start_addr; }

  /// Called to notify the expression that it is about to be executed.
  virtual void WillStartExecuting() {}

  /// Called to notify the expression that its execution has finished.
  virtual void DidFinishExecuting() {}

  virtual ExpressionTypeSystemHelper *GetTypeSystemHelper() { return nullptr; }

  // LLVM RTTI support
  virtual bool isA(const void *ClassID) const = 0;

protected:
  lldb::TargetWP m_target_wp; /// Expression's always have to have a target...
  lldb::ProcessWP m_jit_process_wp; /// An expression might have a process, but
                                    /// it doesn't need to (e.g. calculator
                                    /// mode.)
  lldb::addr_t m_jit_start_addr; ///< The address of the JITted function within
                                 ///the JIT allocation.  LLDB_INVALID_ADDRESS if
                                 ///invalid.
  lldb::addr_t m_jit_end_addr;   ///< The address of the JITted function within
                                 ///the JIT allocation.  LLDB_INVALID_ADDRESS if
                                 ///invalid.
};

/// Holds parsed information about a function call label that
/// LLDB attaches as an AsmLabel to function AST nodes it parses
/// from debug-info.
///
/// The format being:
///
///   <prefix>:<discriminator>:<module uid>:<symbol uid>:<name>
///
/// The label string needs to stay valid for the entire lifetime
/// of this object.
struct FunctionCallLabel {
  /// Arbitrary string which language plugins can interpret for their
  /// own needs.
  llvm::StringRef discriminator;

  /// Unique identifier of the lldb_private::Module
  /// which contains the symbol identified by \c symbol_id.
  lldb::user_id_t module_id;

  /// Unique identifier of the function symbol on which to
  /// perform the function call. For example, for DWARF this would
  /// be the DIE UID.
  lldb::user_id_t symbol_id;

  /// Name to use when searching for the function symbol in
  /// \c module_id. For most function calls this will be a
  /// mangled name. In cases where a mangled name can't be used,
  /// this will be the function name.
  ///
  /// NOTE: kept as last element so we don't have to worry about
  /// ':' in the mangled name when parsing the label.
  llvm::StringRef lookup_name;

  /// Decodes the specified function \c label into a \c FunctionCallLabel.
  static llvm::Expected<FunctionCallLabel> fromString(llvm::StringRef label);

  /// Encode this FunctionCallLabel into its string representation.
  ///
  /// The representation roundtrips through \c fromString:
  /// \code{.cpp}
  /// llvm::StringRef encoded = "$__lldb_func:blah:0x0:0x0:_Z3foov";
  /// FunctionCallLabel label = *fromString(label);
  ///
  /// assert (label.toString() == encoded);
  /// assert (*fromString(label.toString()) == label);
  /// \endcode
  std::string toString() const;
};

/// LLDB attaches this prefix to mangled names of functions that get called
/// from JITted expressions.
inline constexpr llvm::StringRef FunctionCallLabelPrefix = "$__lldb_func";

} // namespace lldb_private

namespace llvm {
template <> struct format_provider<lldb_private::FunctionCallLabel> {
  static void format(const lldb_private::FunctionCallLabel &label,
                     raw_ostream &OS, StringRef Style);
};
} // namespace llvm

#endif // LLDB_EXPRESSION_EXPRESSION_H
