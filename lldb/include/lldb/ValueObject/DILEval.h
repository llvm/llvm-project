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
#include <memory>
#include <vector>

namespace lldb_private::dil {

/// Class used to store & manipulate information about identifiers.
class IdentifierInfo {
public:
  enum class Kind {
    eValue,
    eContextArg,
  };

  static std::unique_ptr<IdentifierInfo> FromValue(ValueObject &valobj) {
    CompilerType type;
    type = valobj.GetCompilerType();
    return std::unique_ptr<IdentifierInfo>(
        new IdentifierInfo(Kind::eValue, type, valobj.GetSP(), {}));
  }

  static std::unique_ptr<IdentifierInfo> FromContextArg(CompilerType type) {
    lldb::ValueObjectSP empty_value;
    return std::unique_ptr<IdentifierInfo>(
        new IdentifierInfo(Kind::eContextArg, type, empty_value, {}));
  }

  Kind GetKind() const { return m_kind; }
  lldb::ValueObjectSP GetValue() const { return m_value; }

  CompilerType GetType() { return m_type; }
  bool IsValid() const { return m_type.IsValid(); }

  IdentifierInfo(Kind kind, CompilerType type, lldb::ValueObjectSP value,
                 std::vector<uint32_t> path)
      : m_kind(kind), m_type(type), m_value(std::move(value)) {}

private:
  Kind m_kind;
  CompilerType m_type;
  lldb::ValueObjectSP m_value;
};

/// Given the name of an identifier (variable name, member name, type name,
/// etc.), find the ValueObject for that name (if it exists) and create and
/// return an IdentifierInfo object containing all the relevant information
/// about that object (for DIL parsing and evaluating).
std::unique_ptr<IdentifierInfo>
LookupIdentifier(llvm::StringRef name_ref,
                 std::shared_ptr<ExecutionContextScope> ctx_scope,
                 lldb::TargetSP target_sp, lldb::DynamicValueType use_dynamic,
                 CompilerType *scope_ptr = nullptr);

class Interpreter : Visitor {
public:
  Interpreter(lldb::TargetSP target, llvm::StringRef expr,
              lldb::DynamicValueType use_dynamic,
              std::shared_ptr<ExecutionContextScope> exe_ctx_scope);

  llvm::Expected<lldb::ValueObjectSP> DILEvalNode(const ASTNode *node);

private:
  llvm::Expected<lldb::ValueObjectSP>
  Visit(const IdentifierNode *node) override;

private:
  // Used by the interpreter to create objects, perform casts, etc.
  lldb::TargetSP m_target;

  llvm::StringRef m_expr;

  lldb::ValueObjectSP m_scope;

  lldb::DynamicValueType m_default_dynamic;

  std::shared_ptr<ExecutionContextScope> m_exe_ctx_scope;
};

} // namespace lldb_private::dil

#endif // LLDB_VALUEOBJECT_DILEVAL_H
