//===-- SBFunction.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFunction.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

SBFunction::SBFunction() : m_opaque_ptr(nullptr) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBFunction);
}

SBFunction::SBFunction(lldb_private::Function *lldb_object_ptr)
    : m_opaque_ptr(lldb_object_ptr) {}

SBFunction::SBFunction(const lldb::SBFunction &rhs)
    : m_opaque_ptr(rhs.m_opaque_ptr) {
  LLDB_RECORD_CONSTRUCTOR(SBFunction, (const lldb::SBFunction &), rhs);
}

const SBFunction &SBFunction::operator=(const SBFunction &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBFunction &,
                     SBFunction, operator=,(const lldb::SBFunction &), rhs);

  m_opaque_ptr = rhs.m_opaque_ptr;
  return LLDB_RECORD_RESULT(*this);
}

SBFunction::~SBFunction() { m_opaque_ptr = nullptr; }

bool SBFunction::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFunction, IsValid);
  return this->operator bool();
}
SBFunction::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBFunction, operator bool);

  return m_opaque_ptr != nullptr;
}

const char *SBFunction::GetName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBFunction, GetName);

  const char *cstr = nullptr;
  if (m_opaque_ptr)
    cstr = m_opaque_ptr->GetName().AsCString();

  return cstr;
}

const char *SBFunction::GetDisplayName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBFunction, GetDisplayName);

  const char *cstr = nullptr;
  if (m_opaque_ptr)
    cstr = m_opaque_ptr->GetMangled().GetDisplayDemangledName().AsCString();

  return cstr;
}

const char *SBFunction::GetMangledName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBFunction, GetMangledName);

  const char *cstr = nullptr;
  if (m_opaque_ptr)
    cstr = m_opaque_ptr->GetMangled().GetMangledName().AsCString();
  return cstr;
}

bool SBFunction::operator==(const SBFunction &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBFunction, operator==,(const lldb::SBFunction &), rhs);

  return m_opaque_ptr == rhs.m_opaque_ptr;
}

bool SBFunction::operator!=(const SBFunction &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBFunction, operator!=,(const lldb::SBFunction &), rhs);

  return m_opaque_ptr != rhs.m_opaque_ptr;
}

bool SBFunction::GetDescription(SBStream &s) {
  LLDB_RECORD_METHOD(bool, SBFunction, GetDescription, (lldb::SBStream &), s);

  if (m_opaque_ptr) {
    s.Printf("SBFunction: id = 0x%8.8" PRIx64 ", name = %s",
             m_opaque_ptr->GetID(), m_opaque_ptr->GetName().AsCString());
    Type *func_type = m_opaque_ptr->GetType();
    if (func_type)
      s.Printf(", type = %s", func_type->GetName().AsCString());
    return true;
  }
  s.Printf("No value");
  return false;
}

SBInstructionList SBFunction::GetInstructions(SBTarget target) {
  LLDB_RECORD_METHOD(lldb::SBInstructionList, SBFunction, GetInstructions,
                     (lldb::SBTarget), target);

  return LLDB_RECORD_RESULT(GetInstructions(target, nullptr));
}

SBInstructionList SBFunction::GetInstructions(SBTarget target,
                                              const char *flavor) {
  LLDB_RECORD_METHOD(lldb::SBInstructionList, SBFunction, GetInstructions,
                     (lldb::SBTarget, const char *), target, flavor);

  SBInstructionList sb_instructions;
  if (m_opaque_ptr) {
    TargetSP target_sp(target.GetSP());
    std::unique_lock<std::recursive_mutex> lock;
    ModuleSP module_sp(
        m_opaque_ptr->GetAddressRange().GetBaseAddress().GetModule());
    if (target_sp && module_sp) {
      lock = std::unique_lock<std::recursive_mutex>(target_sp->GetAPIMutex());
      const bool prefer_file_cache = false;
      sb_instructions.SetDisassembler(Disassembler::DisassembleRange(
          module_sp->GetArchitecture(), nullptr, flavor, *target_sp,
          m_opaque_ptr->GetAddressRange(), prefer_file_cache));
    }
  }
  return LLDB_RECORD_RESULT(sb_instructions);
}

lldb_private::Function *SBFunction::get() { return m_opaque_ptr; }

void SBFunction::reset(lldb_private::Function *lldb_object_ptr) {
  m_opaque_ptr = lldb_object_ptr;
}

SBAddress SBFunction::GetStartAddress() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBAddress, SBFunction, GetStartAddress);

  SBAddress addr;
  if (m_opaque_ptr)
    addr.SetAddress(m_opaque_ptr->GetAddressRange().GetBaseAddress());
  return LLDB_RECORD_RESULT(addr);
}

SBAddress SBFunction::GetEndAddress() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBAddress, SBFunction, GetEndAddress);

  SBAddress addr;
  if (m_opaque_ptr) {
    addr_t byte_size = m_opaque_ptr->GetAddressRange().GetByteSize();
    if (byte_size > 0) {
      addr.SetAddress(m_opaque_ptr->GetAddressRange().GetBaseAddress());
      addr->Slide(byte_size);
    }
  }
  return LLDB_RECORD_RESULT(addr);
}

const char *SBFunction::GetArgumentName(uint32_t arg_idx) {
  LLDB_RECORD_METHOD(const char *, SBFunction, GetArgumentName, (uint32_t),
                     arg_idx);

  if (m_opaque_ptr) {
    Block &block = m_opaque_ptr->GetBlock(true);
    VariableListSP variable_list_sp = block.GetBlockVariableList(true);
    if (variable_list_sp) {
      VariableList arguments;
      variable_list_sp->AppendVariablesWithScope(eValueTypeVariableArgument,
                                                 arguments, true);
      lldb::VariableSP variable_sp = arguments.GetVariableAtIndex(arg_idx);
      if (variable_sp)
        return variable_sp->GetName().GetCString();
    }
  }
  return nullptr;
}

uint32_t SBFunction::GetPrologueByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBFunction, GetPrologueByteSize);

  if (m_opaque_ptr)
    return m_opaque_ptr->GetPrologueByteSize();
  return 0;
}

SBType SBFunction::GetType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBFunction, GetType);

  SBType sb_type;
  if (m_opaque_ptr) {
    Type *function_type = m_opaque_ptr->GetType();
    if (function_type)
      sb_type.ref().SetType(function_type->shared_from_this());
  }
  return LLDB_RECORD_RESULT(sb_type);
}

SBBlock SBFunction::GetBlock() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBBlock, SBFunction, GetBlock);

  SBBlock sb_block;
  if (m_opaque_ptr)
    sb_block.SetPtr(&m_opaque_ptr->GetBlock(true));
  return LLDB_RECORD_RESULT(sb_block);
}

lldb::LanguageType SBFunction::GetLanguage() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::LanguageType, SBFunction, GetLanguage);

  if (m_opaque_ptr) {
    if (m_opaque_ptr->GetCompileUnit())
      return m_opaque_ptr->GetCompileUnit()->GetLanguage();
  }
  return lldb::eLanguageTypeUnknown;
}

bool SBFunction::GetIsOptimized() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBFunction, GetIsOptimized);

  if (m_opaque_ptr) {
    if (m_opaque_ptr->GetCompileUnit())
      return m_opaque_ptr->GetCompileUnit()->GetIsOptimized();
  }
  return false;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBFunction>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBFunction, ());
  LLDB_REGISTER_CONSTRUCTOR(SBFunction, (const lldb::SBFunction &));
  LLDB_REGISTER_METHOD(const lldb::SBFunction &,
                       SBFunction, operator=,(const lldb::SBFunction &));
  LLDB_REGISTER_METHOD_CONST(bool, SBFunction, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBFunction, operator bool, ());
  LLDB_REGISTER_METHOD_CONST(const char *, SBFunction, GetName, ());
  LLDB_REGISTER_METHOD_CONST(const char *, SBFunction, GetDisplayName, ());
  LLDB_REGISTER_METHOD_CONST(const char *, SBFunction, GetMangledName, ());
  LLDB_REGISTER_METHOD_CONST(
      bool, SBFunction, operator==,(const lldb::SBFunction &));
  LLDB_REGISTER_METHOD_CONST(
      bool, SBFunction, operator!=,(const lldb::SBFunction &));
  LLDB_REGISTER_METHOD(bool, SBFunction, GetDescription, (lldb::SBStream &));
  LLDB_REGISTER_METHOD(lldb::SBInstructionList, SBFunction, GetInstructions,
                       (lldb::SBTarget));
  LLDB_REGISTER_METHOD(lldb::SBInstructionList, SBFunction, GetInstructions,
                       (lldb::SBTarget, const char *));
  LLDB_REGISTER_METHOD(lldb::SBAddress, SBFunction, GetStartAddress, ());
  LLDB_REGISTER_METHOD(lldb::SBAddress, SBFunction, GetEndAddress, ());
  LLDB_REGISTER_METHOD(const char *, SBFunction, GetArgumentName, (uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBFunction, GetPrologueByteSize, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBFunction, GetType, ());
  LLDB_REGISTER_METHOD(lldb::SBBlock, SBFunction, GetBlock, ());
  LLDB_REGISTER_METHOD(lldb::LanguageType, SBFunction, GetLanguage, ());
  LLDB_REGISTER_METHOD(bool, SBFunction, GetIsOptimized, ());
}

}
}
