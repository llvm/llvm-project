//===-- DILEval.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILEVAL_H
#define LLDB_VALUEOBJECT_DILEVAL_H

#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILParser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <vector>

namespace lldb_private::dil {

/// Given the name of an identifier (variable name, member name, type name,
/// etc.), find the ValueObject for that name (if it exists), excluding global
/// variables, and create and return an IdentifierInfo object containing all
/// the relevant information about that object (for DIL parsing and
/// evaluating).
lldb::ValueObjectSP LookupIdentifier(llvm::StringRef name_ref,
                                     std::shared_ptr<StackFrame> frame_sp,
                                     lldb::DynamicValueType use_dynamic,
                                     CompilerType *scope_ptr = nullptr);

/// Given the name of an identifier, check to see if it matches the name of a
/// global variable. If so, find the ValueObject for that global variable, and
/// create and return an IdentifierInfo object containing all the relevant
/// informatin about it.
lldb::ValueObjectSP LookupGlobalIdentifier(llvm::StringRef name_ref,
                                           std::shared_ptr<StackFrame> frame_sp,
                                           lldb::TargetSP target_sp,
                                           lldb::DynamicValueType use_dynamic,
                                           CompilerType *scope_ptr = nullptr);

class FlowAnalysis {
public:
  FlowAnalysis(bool address_of_is_pending)
      : m_address_of_is_pending(address_of_is_pending) {}

  bool AddressOfIsPending() const { return m_address_of_is_pending; }
  void DiscardAddressOf() { m_address_of_is_pending = false; }

private:
  bool m_address_of_is_pending;
};

class Interpreter : Visitor {
public:
  Interpreter(lldb::TargetSP target, llvm::StringRef expr,
              lldb::DynamicValueType use_dynamic,
              std::shared_ptr<StackFrame> frame_sp);

  llvm::Expected<lldb::ValueObjectSP> Evaluate(const ASTNode *node);

private:
  llvm::Expected<lldb::ValueObjectSP>
  EvaluateNode(const ASTNode *node, FlowAnalysis *flow = nullptr);

  llvm::Expected<lldb::ValueObjectSP>
  Visit(const IdentifierNode *node) override;
  llvm::Expected<lldb::ValueObjectSP> Visit(const UnaryOpNode *node) override;

  lldb::ValueObjectSP EvaluateDereference(lldb::ValueObjectSP rhs);

  FlowAnalysis *flow_analysis() { return m_flow_analysis_chain.back(); }

  // Used by the interpreter to create objects, perform casts, etc.
  lldb::TargetSP m_target;
  llvm::StringRef m_expr;
  // Flow analysis chain represents the expression evaluation flow for the
  // current code branch. Each node in the chain corresponds to an AST node,
  // describing the semantics of the evaluation for it. Currently, flow analysis
  // propagates the information about the pending address-of operator, so that
  // combination of address-of and a subsequent dereference can be eliminated.
  // End of the chain (i.e. `back()`) contains the flow analysis instance for
  // the current node. It may be `nullptr` if no relevant information is
  // available, the caller/user is supposed to check.
  std::vector<FlowAnalysis *> m_flow_analysis_chain;
  lldb::ValueObjectSP m_scope;
  lldb::DynamicValueType m_default_dynamic;
  std::shared_ptr<StackFrame> m_exe_ctx_scope;
};

} // namespace lldb_private::dil

#endif // LLDB_VALUEOBJECT_DILEVAL_H
