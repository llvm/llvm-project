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

class Interpreter : Visitor {
public:
  Interpreter(lldb::TargetSP target, llvm::StringRef expr,
              lldb::DynamicValueType use_dynamic,
              std::shared_ptr<StackFrame> frame_sp);

  llvm::Expected<lldb::ValueObjectSP> Evaluate(const ASTNode *node);

private:
  llvm::Expected<lldb::ValueObjectSP>
  Visit(const IdentifierNode *node) override;

  // Used by the interpreter to create objects, perform casts, etc.
  lldb::TargetSP m_target;
  llvm::StringRef m_expr;
  lldb::ValueObjectSP m_scope;
  lldb::DynamicValueType m_default_dynamic;
  std::shared_ptr<StackFrame> m_exe_ctx_scope;
};

} // namespace lldb_private::dil

#endif // LLDB_VALUEOBJECT_DILEVAL_H
