//===-- DILEval.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_VALUEOBJECT_DILEVAL_H_
#define LLDB_VALUEOBJECT_DILEVAL_H_

#include <memory>
#include <vector>

#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILParser.h"

namespace lldb_private {

namespace dil {

class DILInterpreter : Visitor {
public:
  DILInterpreter(lldb::TargetSP target, std::shared_ptr<std::string> sm);
  DILInterpreter(lldb::TargetSP target, std::shared_ptr<std::string> sm,
                 lldb::ValueObjectSP scope);
  DILInterpreter(lldb::TargetSP target, std::shared_ptr<std::string> sm,
                 lldb::DynamicValueType use_dynamic);

  lldb::ValueObjectSP DILEval(const DILASTNode *tree, lldb::TargetSP target_sp,
                              Status &error);

private:
  lldb::ValueObjectSP DILEvalNode(const DILASTNode *node);

  bool Success() { return m_error.Success(); }

  void SetError(ErrorCode error_code, std::string error, uint32_t loc);

  void Visit(const ErrorNode *node) override;
  void Visit(const IdentifierNode *node) override;

private:
  // Used by the interpreter to create objects, perform casts, etc.
  lldb::TargetSP m_target;

  std::shared_ptr<std::string> m_sm;

  lldb::ValueObjectSP m_result;

  lldb::ValueObjectSP m_scope;

  lldb::DynamicValueType m_default_dynamic;

  Status m_error;
};

} // namespace dil

} // namespace lldb_private

#endif // LLDB_VALUEOBJECT_DILEVAL_H_
