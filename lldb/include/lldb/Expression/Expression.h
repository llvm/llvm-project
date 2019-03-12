//===-- Expression.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Expression_h_
#define liblldb_Expression_h_

#include <map>
#include <string>
#include <vector>


#include "lldb/Expression/ExpressionTypeSystemHelper.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/SmallVector.h"

namespace lldb_private {

class RecordingMemoryManager;

//----------------------------------------------------------------------
/// \class Expression Expression.h "lldb/Expression/Expression.h" Encapsulates
/// a single expression for use in lldb
///
/// LLDB uses expressions for various purposes, notably to call functions
/// and as a backend for the expr command.  Expression encapsulates the
/// objects needed to parse and interpret or JIT an expression.  It uses the
/// expression parser appropriate to the language of the expression to produce
/// LLVM IR from the expression.
//----------------------------------------------------------------------
class Expression {
public:
  enum ResultType { eResultTypeAny, eResultTypeId };

  Expression(Target &target);

  Expression(ExecutionContextScope &exe_scope);

  //------------------------------------------------------------------
  /// Destructor
  //------------------------------------------------------------------
  virtual ~Expression() {}

  //------------------------------------------------------------------
  /// Return the string that the parser should parse.  Must be a full
  /// translation unit.
  //------------------------------------------------------------------
  virtual const char *Text() = 0;

  //------------------------------------------------------------------
  /// Return the function name that should be used for executing the
  /// expression.  Text() should contain the definition of this function.
  //------------------------------------------------------------------
  virtual const char *FunctionName() = 0;

  //------------------------------------------------------------------
  /// Return the language that should be used when parsing.  To use the
  /// default, return eLanguageTypeUnknown.
  //------------------------------------------------------------------
  virtual lldb::LanguageType Language() { return lldb::eLanguageTypeUnknown; }

  //------------------------------------------------------------------
  /// Return the object that the parser should use when registering external
  /// values (assuming it doesn't use a ClangExpressionDeclMap).  May be
  /// NULL if there is a ClangExpressionDeclMap or everything should be
  /// self-contained.
  //------------------------------------------------------------------
  virtual Materializer *GetMaterializer() { return NULL; }

  //------------------------------------------------------------------
  /// Return the desired result type of the function, or eResultTypeAny if
  /// indifferent.
  //------------------------------------------------------------------
  virtual ResultType DesiredResultType() { return eResultTypeAny; }

  //------------------------------------------------------------------
  /// Flags
  //------------------------------------------------------------------

  //------------------------------------------------------------------
  /// Return true if validation code should be inserted into the expression.
  //------------------------------------------------------------------
  virtual bool NeedsValidation() = 0;

  //------------------------------------------------------------------
  /// Return true if external variables in the expression should be resolved.
  //------------------------------------------------------------------
  virtual bool NeedsVariableResolution() = 0;

  virtual EvaluateExpressionOptions *GetOptions() { return nullptr; };

  //------------------------------------------------------------------
  /// Return the address of the function's JIT-compiled code, or
  /// LLDB_INVALID_ADDRESS if the function is not JIT compiled
  //------------------------------------------------------------------
  lldb::addr_t StartAddress() { return m_jit_start_addr; }

  //------------------------------------------------------------------
  /// Called to notify the expression that it is about to be executed.
  //------------------------------------------------------------------
  virtual void WillStartExecuting() {}

  //------------------------------------------------------------------
  /// Called to notify the expression that its execution has finished.
  //------------------------------------------------------------------
  virtual void DidFinishExecuting() {}

  struct SwiftGenericInfo {
    struct Binding {
      const char *name;
      CompilerType type;
    };
    llvm::SmallVector<Binding, 3> class_bindings;
  };

  const SwiftGenericInfo &GetSwiftGenericInfo() { return m_swift_generic_info; }

  virtual ExpressionTypeSystemHelper *GetTypeSystemHelper() { return nullptr; }

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
  SwiftGenericInfo m_swift_generic_info;
};

} // namespace lldb_private

#endif // liblldb_Expression_h_
