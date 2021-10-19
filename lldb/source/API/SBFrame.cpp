//===-- SBFrame.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <set>
#include <string>

#include "lldb/API/SBFrame.h"

#include "lldb/lldb-types.h"

#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Expression/UserExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Target/StackID.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Stream.h"

#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBExpressionOptions.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBVariablesOptions.h"

#include "llvm/Support/PrettyStackTrace.h"

using namespace lldb;
using namespace lldb_private;

SBFrame::SBFrame() : m_opaque_sp(new ExecutionContextRef()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBFrame);
}

SBFrame::SBFrame(const StackFrameSP &lldb_object_sp)
    : m_opaque_sp(new ExecutionContextRef(lldb_object_sp)) {
  LLDB_RECORD_CONSTRUCTOR(SBFrame, (const lldb::StackFrameSP &),
                          lldb_object_sp);
}

SBFrame::SBFrame(const SBFrame &rhs) : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR(SBFrame, (const lldb::SBFrame &), rhs);

  m_opaque_sp = clone(rhs.m_opaque_sp);
}

SBFrame::~SBFrame() = default;

const SBFrame &SBFrame::operator=(const SBFrame &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBFrame &,
                     SBFrame, operator=,(const lldb::SBFrame &), rhs);

  if (this != &rhs)
    m_opaque_sp = clone(rhs.m_opaque_sp);
  return LLDB_RECORD_RESULT(*this);
}

StackFrameSP SBFrame::GetFrameSP() const {
  return (m_opaque_sp ? m_opaque_sp->GetFrameSP() : StackFrameSP());
}

void SBFrame::SetFrameSP(const StackFrameSP &lldb_object_sp) {
  return m_opaque_sp->SetFrameSP(lldb_object_sp);
}

bool SBFrame::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFrame, IsValid);
  return this->operator bool();
}
SBFrame::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFrame, operator bool);

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock()))
      return GetFrameSP().get() != nullptr;
  }

  // Without a target & process we can't have a valid stack frame.
  return false;
}

SBSymbolContext SBFrame::GetSymbolContext(uint32_t resolve_scope) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBSymbolContext, SBFrame, GetSymbolContext,
                           (uint32_t), resolve_scope);

  SBSymbolContext sb_sym_ctx;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);
  SymbolContextItem scope = static_cast<SymbolContextItem>(resolve_scope);
  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame)
        sb_sym_ctx.SetSymbolContext(&frame->GetSymbolContext(scope));
    }
  }

  return LLDB_RECORD_RESULT(sb_sym_ctx);
}

SBModule SBFrame::GetModule() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBModule, SBFrame, GetModule);

  SBModule sb_module;
  ModuleSP module_sp;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        module_sp = frame->GetSymbolContext(eSymbolContextModule).module_sp;
        sb_module.SetSP(module_sp);
      }
    }
  }

  return LLDB_RECORD_RESULT(sb_module);
}

SBCompileUnit SBFrame::GetCompileUnit() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBCompileUnit, SBFrame,
                                   GetCompileUnit);

  SBCompileUnit sb_comp_unit;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        sb_comp_unit.reset(
            frame->GetSymbolContext(eSymbolContextCompUnit).comp_unit);
      }
    }
  }

  return LLDB_RECORD_RESULT(sb_comp_unit);
}

SBFunction SBFrame::GetFunction() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBFunction, SBFrame, GetFunction);

  SBFunction sb_function;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        sb_function.reset(
            frame->GetSymbolContext(eSymbolContextFunction).function);
      }
    }
  }

  return LLDB_RECORD_RESULT(sb_function);
}

SBSymbol SBFrame::GetSymbol() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBSymbol, SBFrame, GetSymbol);

  SBSymbol sb_symbol;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        sb_symbol.reset(frame->GetSymbolContext(eSymbolContextSymbol).symbol);
      }
    }
  }

  return LLDB_RECORD_RESULT(sb_symbol);
}

SBBlock SBFrame::GetBlock() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBBlock, SBFrame, GetBlock);

  SBBlock sb_block;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame)
        sb_block.SetPtr(frame->GetSymbolContext(eSymbolContextBlock).block);
    }
  }
  return LLDB_RECORD_RESULT(sb_block);
}

SBBlock SBFrame::GetFrameBlock() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBBlock, SBFrame, GetFrameBlock);

  SBBlock sb_block;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame)
        sb_block.SetPtr(frame->GetFrameBlock());
    }
  }
  return LLDB_RECORD_RESULT(sb_block);
}

SBLineEntry SBFrame::GetLineEntry() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBLineEntry, SBFrame, GetLineEntry);

  SBLineEntry sb_line_entry;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        sb_line_entry.SetLineEntry(
            frame->GetSymbolContext(eSymbolContextLineEntry).line_entry);
      }
    }
  }
  return LLDB_RECORD_RESULT(sb_line_entry);
}

uint32_t SBFrame::GetFrameID() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBFrame, GetFrameID);

  uint32_t frame_idx = UINT32_MAX;

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (frame)
    frame_idx = frame->GetFrameIndex();

  return frame_idx;
}

lldb::addr_t SBFrame::GetCFA() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::addr_t, SBFrame, GetCFA);

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (frame)
    return frame->GetStackID().GetCallFrameAddress();
  return LLDB_INVALID_ADDRESS;
}

addr_t SBFrame::GetPC() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::addr_t, SBFrame, GetPC);

  addr_t addr = LLDB_INVALID_ADDRESS;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        addr = frame->GetFrameCodeAddress().GetOpcodeLoadAddress(
            target, AddressClass::eCode);
      }
    }
  }

  return addr;
}

bool SBFrame::SetPC(addr_t new_pc) {
  LLDB_RECORD_METHOD(bool, SBFrame, SetPC, (lldb::addr_t), new_pc);

  bool ret_val = false;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      if (StackFrame *frame = exe_ctx.GetFramePtr()) {
        if (RegisterContextSP reg_ctx_sp = frame->GetRegisterContext()) {
          ret_val = reg_ctx_sp->SetPC(new_pc);
        }
      }
    }
  }

  return ret_val;
}

addr_t SBFrame::GetSP() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::addr_t, SBFrame, GetSP);

  addr_t addr = LLDB_INVALID_ADDRESS;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      if (StackFrame *frame = exe_ctx.GetFramePtr()) {
        if (RegisterContextSP reg_ctx_sp = frame->GetRegisterContext()) {
          addr = reg_ctx_sp->GetSP();
        }
      }
    }
  }

  return addr;
}

addr_t SBFrame::GetFP() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::addr_t, SBFrame, GetFP);

  addr_t addr = LLDB_INVALID_ADDRESS;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      if (StackFrame *frame = exe_ctx.GetFramePtr()) {
        if (RegisterContextSP reg_ctx_sp = frame->GetRegisterContext()) {
          addr = reg_ctx_sp->GetFP();
        }
      }
    }
  }

  return addr;
}

SBAddress SBFrame::GetPCAddress() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBAddress, SBFrame, GetPCAddress);

  SBAddress sb_addr;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame)
        sb_addr.SetAddress(frame->GetFrameCodeAddress());
    }
  }
  return LLDB_RECORD_RESULT(sb_addr);
}

void SBFrame::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBFrame, Clear);

  m_opaque_sp->Clear();
}

lldb::SBValue SBFrame::GetValueForVariablePath(const char *var_path) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, GetValueForVariablePath,
                     (const char *), var_path);

  SBValue sb_value;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  Target *target = exe_ctx.GetTargetPtr();
  if (frame && target) {
    lldb::DynamicValueType use_dynamic =
        frame->CalculateTarget()->GetPreferDynamicValue();
    sb_value = GetValueForVariablePath(var_path, use_dynamic);
  }
  return LLDB_RECORD_RESULT(sb_value);
}

lldb::SBValue SBFrame::GetValueForVariablePath(const char *var_path,
                                               DynamicValueType use_dynamic) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, GetValueForVariablePath,
                     (const char *, lldb::DynamicValueType), var_path,
                     use_dynamic);

  SBValue sb_value;
  if (var_path == nullptr || var_path[0] == '\0') {
    return LLDB_RECORD_RESULT(sb_value);
  }

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        VariableSP var_sp;
        Status error;
        ValueObjectSP value_sp(frame->GetValueForVariableExpressionPath(
            var_path, eNoDynamicValues,
            StackFrame::eExpressionPathOptionCheckPtrVsMember |
                StackFrame::eExpressionPathOptionsAllowDirectIVarAccess,
            var_sp, error));
        sb_value.SetSP(value_sp, use_dynamic);
      }
    }
  }
  return LLDB_RECORD_RESULT(sb_value);
}

SBValue SBFrame::FindVariable(const char *name) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, FindVariable, (const char *),
                     name);

  SBValue value;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  Target *target = exe_ctx.GetTargetPtr();
  if (frame && target) {
    lldb::DynamicValueType use_dynamic =
        frame->CalculateTarget()->GetPreferDynamicValue();
    value = FindVariable(name, use_dynamic);
  }
  return LLDB_RECORD_RESULT(value);
}

SBValue SBFrame::FindVariable(const char *name,
                              lldb::DynamicValueType use_dynamic) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, FindVariable,
                     (const char *, lldb::DynamicValueType), name, use_dynamic);

  VariableSP var_sp;
  SBValue sb_value;

  if (name == nullptr || name[0] == '\0') {
    return LLDB_RECORD_RESULT(sb_value);
  }

  ValueObjectSP value_sp;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        value_sp = frame->FindVariable(ConstString(name));

        if (value_sp)
          sb_value.SetSP(value_sp, use_dynamic);
      }
    }
  }

  return LLDB_RECORD_RESULT(sb_value);
}

SBValue SBFrame::FindValue(const char *name, ValueType value_type) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, FindValue,
                     (const char *, lldb::ValueType), name, value_type);

  SBValue value;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  Target *target = exe_ctx.GetTargetPtr();
  if (frame && target) {
    lldb::DynamicValueType use_dynamic =
        frame->CalculateTarget()->GetPreferDynamicValue();
    value = FindValue(name, value_type, use_dynamic);
  }
  return LLDB_RECORD_RESULT(value);
}

SBValue SBFrame::FindValue(const char *name, ValueType value_type,
                           lldb::DynamicValueType use_dynamic) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, FindValue,
                     (const char *, lldb::ValueType, lldb::DynamicValueType),
                     name, value_type, use_dynamic);

  SBValue sb_value;

  if (name == nullptr || name[0] == '\0') {
    return LLDB_RECORD_RESULT(sb_value);
  }

  ValueObjectSP value_sp;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        VariableList variable_list;

        switch (value_type) {
        case eValueTypeVariableGlobal:      // global variable
        case eValueTypeVariableStatic:      // static variable
        case eValueTypeVariableArgument:    // function argument variables
        case eValueTypeVariableLocal:       // function local variables
        case eValueTypeVariableThreadLocal: // thread local variables
        {
          SymbolContext sc(frame->GetSymbolContext(eSymbolContextBlock));

          const bool can_create = true;
          const bool get_parent_variables = true;
          const bool stop_if_block_is_inlined_function = true;

          if (sc.block)
            sc.block->AppendVariables(
                can_create, get_parent_variables,
                stop_if_block_is_inlined_function,
                [frame](Variable *v) { return v->IsInScope(frame); },
                &variable_list);
          if (value_type == eValueTypeVariableGlobal) {
            const bool get_file_globals = true;
            VariableList *frame_vars = frame->GetVariableList(get_file_globals);
            if (frame_vars)
              frame_vars->AppendVariablesIfUnique(variable_list);
          }
          ConstString const_name(name);
          VariableSP variable_sp(
              variable_list.FindVariable(const_name, value_type));
          if (variable_sp) {
            value_sp = frame->GetValueObjectForFrameVariable(variable_sp,
                                                             eNoDynamicValues);
            sb_value.SetSP(value_sp, use_dynamic);
          }
        } break;

        case eValueTypeRegister: // stack frame register value
        {
          RegisterContextSP reg_ctx(frame->GetRegisterContext());
          if (reg_ctx) {
            if (const RegisterInfo *reg_info =
                    reg_ctx->GetRegisterInfoByName(name)) {
              value_sp = ValueObjectRegister::Create(frame, reg_ctx, reg_info);
              sb_value.SetSP(value_sp);
            }
          }
        } break;

        case eValueTypeRegisterSet: // A collection of stack frame register
                                    // values
        {
          RegisterContextSP reg_ctx(frame->GetRegisterContext());
          if (reg_ctx) {
            const uint32_t num_sets = reg_ctx->GetRegisterSetCount();
            for (uint32_t set_idx = 0; set_idx < num_sets; ++set_idx) {
              const RegisterSet *reg_set = reg_ctx->GetRegisterSet(set_idx);
              if (reg_set &&
                  ((reg_set->name && strcasecmp(reg_set->name, name) == 0) ||
                   (reg_set->short_name &&
                    strcasecmp(reg_set->short_name, name) == 0))) {
                value_sp =
                    ValueObjectRegisterSet::Create(frame, reg_ctx, set_idx);
                sb_value.SetSP(value_sp);
                break;
              }
            }
          }
        } break;

        case eValueTypeConstResult: // constant result variables
        {
          ConstString const_name(name);
          ExpressionVariableSP expr_var_sp(
              target->GetPersistentVariable(const_name));
          if (expr_var_sp) {
            value_sp = expr_var_sp->GetValueObject();
            sb_value.SetSP(value_sp, use_dynamic);
          }
        } break;

        default:
          break;
        }
      }
    }
  }

  return LLDB_RECORD_RESULT(sb_value);
}

bool SBFrame::IsEqual(const SBFrame &that) const {
  LLDB_RECORD_METHOD_CONST(bool, SBFrame, IsEqual, (const lldb::SBFrame &),
                           that);

  lldb::StackFrameSP this_sp = GetFrameSP();
  lldb::StackFrameSP that_sp = that.GetFrameSP();
  return (this_sp && that_sp && this_sp->GetStackID() == that_sp->GetStackID());
}

bool SBFrame::operator==(const SBFrame &rhs) const {
  LLDB_RECORD_METHOD_CONST(bool, SBFrame, operator==,(const lldb::SBFrame &),
                           rhs);

  return IsEqual(rhs);
}

bool SBFrame::operator!=(const SBFrame &rhs) const {
  LLDB_RECORD_METHOD_CONST(bool, SBFrame, operator!=,(const lldb::SBFrame &),
                           rhs);

  return !IsEqual(rhs);
}

SBThread SBFrame::GetThread() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBThread, SBFrame, GetThread);

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  ThreadSP thread_sp(exe_ctx.GetThreadSP());
  SBThread sb_thread(thread_sp);

  return LLDB_RECORD_RESULT(sb_thread);
}

const char *SBFrame::Disassemble() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBFrame, Disassemble);

  const char *disassembly = nullptr;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        disassembly = frame->Disassemble();
      }
    }
  }

  return disassembly;
}

SBValueList SBFrame::GetVariables(bool arguments, bool locals, bool statics,
                                  bool in_scope_only) {
  LLDB_RECORD_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                     (bool, bool, bool, bool), arguments, locals, statics,
                     in_scope_only);

  SBValueList value_list;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  Target *target = exe_ctx.GetTargetPtr();
  if (frame && target) {
    lldb::DynamicValueType use_dynamic =
        frame->CalculateTarget()->GetPreferDynamicValue();
    const bool include_runtime_support_values =
        target ? target->GetDisplayRuntimeSupportValues() : false;

    SBVariablesOptions options;
    options.SetIncludeArguments(arguments);
    options.SetIncludeLocals(locals);
    options.SetIncludeStatics(statics);
    options.SetInScopeOnly(in_scope_only);
    options.SetIncludeRuntimeSupportValues(include_runtime_support_values);
    options.SetUseDynamic(use_dynamic);

    value_list = GetVariables(options);
  }
  return LLDB_RECORD_RESULT(value_list);
}

lldb::SBValueList SBFrame::GetVariables(bool arguments, bool locals,
                                        bool statics, bool in_scope_only,
                                        lldb::DynamicValueType use_dynamic) {
  LLDB_RECORD_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                     (bool, bool, bool, bool, lldb::DynamicValueType),
                     arguments, locals, statics, in_scope_only, use_dynamic);

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  Target *target = exe_ctx.GetTargetPtr();
  const bool include_runtime_support_values =
      target ? target->GetDisplayRuntimeSupportValues() : false;
  SBVariablesOptions options;
  options.SetIncludeArguments(arguments);
  options.SetIncludeLocals(locals);
  options.SetIncludeStatics(statics);
  options.SetInScopeOnly(in_scope_only);
  options.SetIncludeRuntimeSupportValues(include_runtime_support_values);
  options.SetUseDynamic(use_dynamic);
  return LLDB_RECORD_RESULT(GetVariables(options));
}

SBValueList SBFrame::GetVariables(const lldb::SBVariablesOptions &options) {
  LLDB_RECORD_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                     (const lldb::SBVariablesOptions &), options);

  SBValueList value_list;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();

  const bool statics = options.GetIncludeStatics();
  const bool arguments = options.GetIncludeArguments();
  const bool recognized_arguments =
        options.GetIncludeRecognizedArguments(SBTarget(exe_ctx.GetTargetSP()));
  const bool locals = options.GetIncludeLocals();
  const bool in_scope_only = options.GetInScopeOnly();
  const bool include_runtime_support_values =
      options.GetIncludeRuntimeSupportValues();
  const lldb::DynamicValueType use_dynamic = options.GetUseDynamic();


  std::set<VariableSP> variable_set;
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        VariableList *variable_list = nullptr;
        variable_list = frame->GetVariableList(true);
        if (variable_list) {
          const size_t num_variables = variable_list->GetSize();
          if (num_variables) {
            for (const VariableSP &variable_sp : *variable_list) {
              if (variable_sp) {
                bool add_variable = false;
                switch (variable_sp->GetScope()) {
                case eValueTypeVariableGlobal:
                case eValueTypeVariableStatic:
                case eValueTypeVariableThreadLocal:
                  add_variable = statics;
                  break;

                case eValueTypeVariableArgument:
                  add_variable = arguments;
                  break;

                case eValueTypeVariableLocal:
                  add_variable = locals;
                  break;

                default:
                  break;
                }
                if (add_variable) {
                  // Only add variables once so we don't end up with duplicates
                  if (variable_set.find(variable_sp) == variable_set.end())
                    variable_set.insert(variable_sp);
                  else
                    continue;

                  if (in_scope_only && !variable_sp->IsInScope(frame))
                    continue;

                  ValueObjectSP valobj_sp(frame->GetValueObjectForFrameVariable(
                      variable_sp, eNoDynamicValues));

                  if (!include_runtime_support_values && valobj_sp != nullptr &&
                      valobj_sp->IsRuntimeSupportValue())
                    continue;

                  SBValue value_sb;
                  value_sb.SetSP(valobj_sp, use_dynamic);
                  value_list.Append(value_sb);
                }
              }
            }
          }
        }
        if (recognized_arguments) {
          auto recognized_frame = frame->GetRecognizedFrame();
          if (recognized_frame) {
            ValueObjectListSP recognized_arg_list =
                recognized_frame->GetRecognizedArguments();
            if (recognized_arg_list) {
              for (auto &rec_value_sp : recognized_arg_list->GetObjects()) {
                SBValue value_sb;
                value_sb.SetSP(rec_value_sp, use_dynamic);
                value_list.Append(value_sb);
              }
            }
          }
        }
      }
    }
  }

  return LLDB_RECORD_RESULT(value_list);
}

SBValueList SBFrame::GetRegisters() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBValueList, SBFrame, GetRegisters);

  SBValueList value_list;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        RegisterContextSP reg_ctx(frame->GetRegisterContext());
        if (reg_ctx) {
          const uint32_t num_sets = reg_ctx->GetRegisterSetCount();
          for (uint32_t set_idx = 0; set_idx < num_sets; ++set_idx) {
            value_list.Append(
                ValueObjectRegisterSet::Create(frame, reg_ctx, set_idx));
          }
        }
      }
    }
  }

  return LLDB_RECORD_RESULT(value_list);
}

SBValue SBFrame::FindRegister(const char *name) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, FindRegister, (const char *),
                     name);

  SBValue result;
  ValueObjectSP value_sp;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        RegisterContextSP reg_ctx(frame->GetRegisterContext());
        if (reg_ctx) {
          if (const RegisterInfo *reg_info =
                  reg_ctx->GetRegisterInfoByName(name)) {
            value_sp = ValueObjectRegister::Create(frame, reg_ctx, reg_info);
            result.SetSP(value_sp);
          }
        }
      }
    }
  }

  return LLDB_RECORD_RESULT(result);
}

bool SBFrame::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBFrame, GetDescription, (lldb::SBStream &),
                     description);

  Stream &strm = description.ref();

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        frame->DumpUsingSettingsFormat(&strm);
      }
    }

  } else
    strm.PutCString("No value");

  return true;
}

SBValue SBFrame::EvaluateExpression(const char *expr) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, EvaluateExpression, (const char *),
                     expr);

  SBValue result;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  Target *target = exe_ctx.GetTargetPtr();
  if (frame && target) {
    SBExpressionOptions options;
    lldb::DynamicValueType fetch_dynamic_value =
        frame->CalculateTarget()->GetPreferDynamicValue();
    options.SetFetchDynamicValue(fetch_dynamic_value);
    options.SetUnwindOnError(true);
    options.SetIgnoreBreakpoints(true);
    if (target->GetLanguage() != eLanguageTypeUnknown)
      options.SetLanguage(target->GetLanguage());
    else
      options.SetLanguage(frame->GetLanguage());
    return LLDB_RECORD_RESULT(EvaluateExpression(expr, options));
  }
  return LLDB_RECORD_RESULT(result);
}

SBValue
SBFrame::EvaluateExpression(const char *expr,
                            lldb::DynamicValueType fetch_dynamic_value) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                     (const char *, lldb::DynamicValueType), expr,
                     fetch_dynamic_value);

  SBExpressionOptions options;
  options.SetFetchDynamicValue(fetch_dynamic_value);
  options.SetUnwindOnError(true);
  options.SetIgnoreBreakpoints(true);
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  Target *target = exe_ctx.GetTargetPtr();
  if (target && target->GetLanguage() != eLanguageTypeUnknown)
    options.SetLanguage(target->GetLanguage());
  else if (frame)
    options.SetLanguage(frame->GetLanguage());
  return LLDB_RECORD_RESULT(EvaluateExpression(expr, options));
}

SBValue SBFrame::EvaluateExpression(const char *expr,
                                    lldb::DynamicValueType fetch_dynamic_value,
                                    bool unwind_on_error) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                     (const char *, lldb::DynamicValueType, bool), expr,
                     fetch_dynamic_value, unwind_on_error);

  SBExpressionOptions options;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  options.SetFetchDynamicValue(fetch_dynamic_value);
  options.SetUnwindOnError(unwind_on_error);
  options.SetIgnoreBreakpoints(true);
  StackFrame *frame = exe_ctx.GetFramePtr();
  Target *target = exe_ctx.GetTargetPtr();
  if (target && target->GetLanguage() != eLanguageTypeUnknown)
    options.SetLanguage(target->GetLanguage());
  else if (frame)
    options.SetLanguage(frame->GetLanguage());
  return LLDB_RECORD_RESULT(EvaluateExpression(expr, options));
}

lldb::SBValue SBFrame::EvaluateExpression(const char *expr,
                                          const SBExpressionOptions &options) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                     (const char *, const lldb::SBExpressionOptions &), expr,
                     options);

  Log *expr_log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  SBValue expr_result;

  if (expr == nullptr || expr[0] == '\0') {
    return LLDB_RECORD_RESULT(expr_result);
  }

  ValueObjectSP expr_value_sp;

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);


  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();

  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        std::unique_ptr<llvm::PrettyStackTraceFormat> stack_trace;
        if (target->GetDisplayExpressionsInCrashlogs()) {
          StreamString frame_description;
          frame->DumpUsingSettingsFormat(&frame_description);
          stack_trace = std::make_unique<llvm::PrettyStackTraceFormat>(
              "SBFrame::EvaluateExpression (expr = \"%s\", fetch_dynamic_value "
              "= %u) %s",
              expr, options.GetFetchDynamicValue(),
              frame_description.GetData());
        }

        target->EvaluateExpression(expr, frame, expr_value_sp, options.ref());
        expr_result.SetSP(expr_value_sp, options.GetFetchDynamicValue());
      }
    }
  }

  LLDB_LOGF(expr_log,
            "** [SBFrame::EvaluateExpression] Expression result is "
            "%s, summary %s **",
            expr_result.GetValue(), expr_result.GetSummary());

  return LLDB_RECORD_RESULT(expr_result);
}

bool SBFrame::IsInlined() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBFrame, IsInlined);

  return static_cast<const SBFrame *>(this)->IsInlined();
}

bool SBFrame::IsInlined() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFrame, IsInlined);

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {

        Block *block = frame->GetSymbolContext(eSymbolContextBlock).block;
        if (block)
          return block->GetContainingInlinedBlock() != nullptr;
      }
    }
  }
  return false;
}

bool SBFrame::IsArtificial() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBFrame, IsArtificial);

  return static_cast<const SBFrame *>(this)->IsArtificial();
}

bool SBFrame::IsArtificial() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFrame, IsArtificial);

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (frame)
    return frame->IsArtificial();

  return false;
}

const char *SBFrame::GetFunctionName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBFrame, GetFunctionName);

  return static_cast<const SBFrame *>(this)->GetFunctionName();
}

lldb::LanguageType SBFrame::GuessLanguage() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::LanguageType, SBFrame, GuessLanguage);

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        return frame->GuessLanguage();
      }
    }
  }
  return eLanguageTypeUnknown;
}

const char *SBFrame::GetFunctionName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBFrame, GetFunctionName);

  const char *name = nullptr;
  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        SymbolContext sc(frame->GetSymbolContext(eSymbolContextFunction |
                                                 eSymbolContextBlock |
                                                 eSymbolContextSymbol));
        if (sc.block) {
          Block *inlined_block = sc.block->GetContainingInlinedBlock();
          if (inlined_block) {
            const InlineFunctionInfo *inlined_info =
                inlined_block->GetInlinedFunctionInfo();
            name = inlined_info->GetName().AsCString();
          }
        }

        if (name == nullptr) {
          if (sc.function)
            name = sc.function->GetName().GetCString();
        }

        if (name == nullptr) {
          if (sc.symbol)
            name = sc.symbol->GetName().GetCString();
        }
      }
    }
  }
  return name;
}

const char *SBFrame::GetDisplayFunctionName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBFrame, GetDisplayFunctionName);

  const char *name = nullptr;

  std::unique_lock<std::recursive_mutex> lock;
  ExecutionContext exe_ctx(m_opaque_sp.get(), lock);

  StackFrame *frame = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  Process *process = exe_ctx.GetProcessPtr();
  if (target && process) {
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process->GetRunLock())) {
      frame = exe_ctx.GetFramePtr();
      if (frame) {
        SymbolContext sc(frame->GetSymbolContext(eSymbolContextFunction |
                                                 eSymbolContextBlock |
                                                 eSymbolContextSymbol));
        if (sc.block) {
          Block *inlined_block = sc.block->GetContainingInlinedBlock();
          if (inlined_block) {
            const InlineFunctionInfo *inlined_info =
                inlined_block->GetInlinedFunctionInfo();
            name = inlined_info->GetDisplayName().AsCString();
          }
        }

        if (name == nullptr) {
          if (sc.function)
            name = sc.function->GetDisplayName().GetCString();
        }

        if (name == nullptr) {
          if (sc.symbol)
            name = sc.symbol->GetDisplayName().GetCString();
        }
      }
    }
  }
  return name;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBFrame>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBFrame, ());
  LLDB_REGISTER_CONSTRUCTOR(SBFrame, (const lldb::StackFrameSP &));
  LLDB_REGISTER_CONSTRUCTOR(SBFrame, (const lldb::SBFrame &));
  LLDB_REGISTER_METHOD(const lldb::SBFrame &,
                       SBFrame, operator=,(const lldb::SBFrame &));
  LLDB_REGISTER_METHOD_CONST(bool, SBFrame, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBFrame, operator bool, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBSymbolContext, SBFrame, GetSymbolContext,
                             (uint32_t));
  LLDB_REGISTER_METHOD_CONST(lldb::SBModule, SBFrame, GetModule, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBCompileUnit, SBFrame, GetCompileUnit,
                             ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBFunction, SBFrame, GetFunction, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBSymbol, SBFrame, GetSymbol, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBBlock, SBFrame, GetBlock, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBBlock, SBFrame, GetFrameBlock, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBLineEntry, SBFrame, GetLineEntry, ());
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBFrame, GetFrameID, ());
  LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBFrame, GetCFA, ());
  LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBFrame, GetPC, ());
  LLDB_REGISTER_METHOD(bool, SBFrame, SetPC, (lldb::addr_t));
  LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBFrame, GetSP, ());
  LLDB_REGISTER_METHOD_CONST(lldb::addr_t, SBFrame, GetFP, ());
  LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBFrame, GetPCAddress, ());
  LLDB_REGISTER_METHOD(void, SBFrame, Clear, ());
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, GetValueForVariablePath,
                       (const char *));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, GetValueForVariablePath,
                       (const char *, lldb::DynamicValueType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, FindVariable, (const char *));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, FindVariable,
                       (const char *, lldb::DynamicValueType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, FindValue,
                       (const char *, lldb::ValueType));
  LLDB_REGISTER_METHOD(
      lldb::SBValue, SBFrame, FindValue,
      (const char *, lldb::ValueType, lldb::DynamicValueType));
  LLDB_REGISTER_METHOD_CONST(bool, SBFrame, IsEqual, (const lldb::SBFrame &));
  LLDB_REGISTER_METHOD_CONST(bool,
                             SBFrame, operator==,(const lldb::SBFrame &));
  LLDB_REGISTER_METHOD_CONST(bool,
                             SBFrame, operator!=,(const lldb::SBFrame &));
  LLDB_REGISTER_METHOD_CONST(lldb::SBThread, SBFrame, GetThread, ());
  LLDB_REGISTER_METHOD_CONST(const char *, SBFrame, Disassemble, ());
  LLDB_REGISTER_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                       (bool, bool, bool, bool));
  LLDB_REGISTER_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                       (bool, bool, bool, bool, lldb::DynamicValueType));
  LLDB_REGISTER_METHOD(lldb::SBValueList, SBFrame, GetVariables,
                       (const lldb::SBVariablesOptions &));
  LLDB_REGISTER_METHOD(lldb::SBValueList, SBFrame, GetRegisters, ());
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, FindRegister, (const char *));
  LLDB_REGISTER_METHOD(bool, SBFrame, GetDescription, (lldb::SBStream &));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                       (const char *));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                       (const char *, lldb::DynamicValueType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                       (const char *, lldb::DynamicValueType, bool));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBFrame, EvaluateExpression,
                       (const char *, const lldb::SBExpressionOptions &));
  LLDB_REGISTER_METHOD(bool, SBFrame, IsInlined, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBFrame, IsInlined, ());
  LLDB_REGISTER_METHOD(bool, SBFrame, IsArtificial, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBFrame, IsArtificial, ());
  LLDB_REGISTER_METHOD(const char *, SBFrame, GetFunctionName, ());
  LLDB_REGISTER_METHOD_CONST(lldb::LanguageType, SBFrame, GuessLanguage, ());
  LLDB_REGISTER_METHOD_CONST(const char *, SBFrame, GetFunctionName, ());
  LLDB_REGISTER_METHOD(const char *, SBFrame, GetDisplayFunctionName, ());
}

}
}
