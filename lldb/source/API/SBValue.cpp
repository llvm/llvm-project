//===-- SBValue.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBValue.h"
#include "SBReproducerPrivate.h"

#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"

#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Declaration.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Stream.h"

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBExpressionOptions.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

class ValueImpl {
public:
  ValueImpl() = default;

  ValueImpl(lldb::ValueObjectSP in_valobj_sp,
            lldb::DynamicValueType use_dynamic, bool use_synthetic,
            const char *name = nullptr)
      : m_valobj_sp(), m_use_dynamic(use_dynamic),
        m_use_synthetic(use_synthetic), m_name(name) {
    if (in_valobj_sp) {
      if ((m_valobj_sp = in_valobj_sp->GetQualifiedRepresentationIfAvailable(
               lldb::eNoDynamicValues, false))) {
        if (!m_name.IsEmpty())
          m_valobj_sp->SetName(m_name);
      }
    }
  }

  ValueImpl(const ValueImpl &rhs)
      : m_valobj_sp(rhs.m_valobj_sp), m_use_dynamic(rhs.m_use_dynamic),
        m_use_synthetic(rhs.m_use_synthetic), m_name(rhs.m_name) {}

  ValueImpl &operator=(const ValueImpl &rhs) {
    if (this != &rhs) {
      m_valobj_sp = rhs.m_valobj_sp;
      m_use_dynamic = rhs.m_use_dynamic;
      m_use_synthetic = rhs.m_use_synthetic;
      m_name = rhs.m_name;
    }
    return *this;
  }

  bool IsValid() {
    if (m_valobj_sp.get() == nullptr)
      return false;
    else {
      // FIXME: This check is necessary but not sufficient.  We for sure don't
      // want to touch SBValues whose owning
      // targets have gone away.  This check is a little weak in that it
      // enforces that restriction when you call IsValid, but since IsValid
      // doesn't lock the target, you have no guarantee that the SBValue won't
      // go invalid after you call this... Also, an SBValue could depend on
      // data from one of the modules in the target, and those could go away
      // independently of the target, for instance if a module is unloaded.
      // But right now, neither SBValues nor ValueObjects know which modules
      // they depend on.  So I have no good way to make that check without
      // tracking that in all the ValueObject subclasses.
      TargetSP target_sp = m_valobj_sp->GetTargetSP();
      return target_sp && target_sp->IsValid();
    }
  }

  lldb::ValueObjectSP GetRootSP() { return m_valobj_sp; }

  lldb::ValueObjectSP GetSP(Process::StopLocker &stop_locker,
                            std::unique_lock<std::recursive_mutex> &lock,
                            Status &error) {
    if (!m_valobj_sp) {
      error.SetErrorString("invalid value object");
      return m_valobj_sp;
    }

    lldb::ValueObjectSP value_sp = m_valobj_sp;

    Target *target = value_sp->GetTargetSP().get();
    if (!target)
      return ValueObjectSP();

    lock = std::unique_lock<std::recursive_mutex>(target->GetAPIMutex());

    ProcessSP process_sp(value_sp->GetProcessSP());
    if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock())) {
      // We don't allow people to play around with ValueObject if the process
      // is running. If you want to look at values, pause the process, then
      // look.
      error.SetErrorString("process must be stopped.");
      return ValueObjectSP();
    }

    if (m_use_dynamic != eNoDynamicValues) {
      ValueObjectSP dynamic_sp = value_sp->GetDynamicValue(m_use_dynamic);
      if (dynamic_sp)
        value_sp = dynamic_sp;
    }

    if (m_use_synthetic) {
      ValueObjectSP synthetic_sp = value_sp->GetSyntheticValue();
      if (synthetic_sp)
        value_sp = synthetic_sp;
    }

    if (!value_sp)
      error.SetErrorString("invalid value object");
    if (!m_name.IsEmpty())
      value_sp->SetName(m_name);

    return value_sp;
  }

  void SetUseDynamic(lldb::DynamicValueType use_dynamic) {
    m_use_dynamic = use_dynamic;
  }

  void SetUseSynthetic(bool use_synthetic) { m_use_synthetic = use_synthetic; }

  lldb::DynamicValueType GetUseDynamic() { return m_use_dynamic; }

  bool GetUseSynthetic() { return m_use_synthetic; }

  // All the derived values that we would make from the m_valobj_sp will share
  // the ExecutionContext with m_valobj_sp, so we don't need to do the
  // calculations in GetSP to return the Target, Process, Thread or Frame.  It
  // is convenient to provide simple accessors for these, which I do here.
  TargetSP GetTargetSP() {
    if (m_valobj_sp)
      return m_valobj_sp->GetTargetSP();
    else
      return TargetSP();
  }

  ProcessSP GetProcessSP() {
    if (m_valobj_sp)
      return m_valobj_sp->GetProcessSP();
    else
      return ProcessSP();
  }

  ThreadSP GetThreadSP() {
    if (m_valobj_sp)
      return m_valobj_sp->GetThreadSP();
    else
      return ThreadSP();
  }

  StackFrameSP GetFrameSP() {
    if (m_valobj_sp)
      return m_valobj_sp->GetFrameSP();
    else
      return StackFrameSP();
  }

private:
  lldb::ValueObjectSP m_valobj_sp;
  lldb::DynamicValueType m_use_dynamic;
  bool m_use_synthetic;
  ConstString m_name;
};

class ValueLocker {
public:
  ValueLocker() = default;

  ValueObjectSP GetLockedSP(ValueImpl &in_value) {
    return in_value.GetSP(m_stop_locker, m_lock, m_lock_error);
  }

  Status &GetError() { return m_lock_error; }

private:
  Process::StopLocker m_stop_locker;
  std::unique_lock<std::recursive_mutex> m_lock;
  Status m_lock_error;
};

SBValue::SBValue() : m_opaque_sp() { LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBValue); }

SBValue::SBValue(const lldb::ValueObjectSP &value_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBValue, (const lldb::ValueObjectSP &), value_sp);

  SetSP(value_sp);
}

SBValue::SBValue(const SBValue &rhs) {
  LLDB_RECORD_CONSTRUCTOR(SBValue, (const lldb::SBValue &), rhs);

  SetSP(rhs.m_opaque_sp);
}

SBValue &SBValue::operator=(const SBValue &rhs) {
  LLDB_RECORD_METHOD(lldb::SBValue &,
                     SBValue, operator=,(const lldb::SBValue &), rhs);

  if (this != &rhs) {
    SetSP(rhs.m_opaque_sp);
  }
  return LLDB_RECORD_RESULT(*this);
}

SBValue::~SBValue() = default;

bool SBValue::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, IsValid);
  return this->operator bool();
}
SBValue::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBValue, operator bool);

  // If this function ever changes to anything that does more than just check
  // if the opaque shared pointer is non NULL, then we need to update all "if
  // (m_opaque_sp)" code in this file.
  return m_opaque_sp.get() != nullptr && m_opaque_sp->IsValid() &&
         m_opaque_sp->GetRootSP().get() != nullptr;
}

void SBValue::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBValue, Clear);

  m_opaque_sp.reset();
}

SBError SBValue::GetError() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBError, SBValue, GetError);

  SBError sb_error;

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    sb_error.SetError(value_sp->GetError());
  else
    sb_error.SetErrorStringWithFormat("error: %s",
                                      locker.GetError().AsCString());

  return LLDB_RECORD_RESULT(sb_error);
}

user_id_t SBValue::GetID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::user_id_t, SBValue, GetID);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    return value_sp->GetID();
  return LLDB_INVALID_UID;
}

const char *SBValue::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBValue, GetName);

  const char *name = nullptr;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    name = value_sp->GetName().GetCString();

  return name;
}

const char *SBValue::GetTypeName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBValue, GetTypeName);

  const char *name = nullptr;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    name = value_sp->GetQualifiedTypeName().GetCString();
  }

  return name;
}

const char *SBValue::GetDisplayTypeName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBValue, GetDisplayTypeName);

  const char *name = nullptr;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    name = value_sp->GetDisplayTypeName().GetCString();
  }

  return name;
}

size_t SBValue::GetByteSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBValue, GetByteSize);

  size_t result = 0;

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    result = value_sp->GetByteSize();
  }

  return result;
}

bool SBValue::IsInScope() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, IsInScope);

  bool result = false;

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    result = value_sp->IsInScope();
  }

  return result;
}

const char *SBValue::GetValue() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBValue, GetValue);

  const char *cstr = nullptr;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    cstr = value_sp->GetValueAsCString();
  }

  return cstr;
}

ValueType SBValue::GetValueType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::ValueType, SBValue, GetValueType);

  ValueType result = eValueTypeInvalid;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    result = value_sp->GetValueType();

  return result;
}

const char *SBValue::GetObjectDescription() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBValue, GetObjectDescription);

  const char *cstr = nullptr;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    cstr = value_sp->GetObjectDescription();
  }

  return cstr;
}

SBType SBValue::GetType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBValue, GetType);

  SBType sb_type;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  TypeImplSP type_sp;
  if (value_sp) {
    type_sp = std::make_shared<TypeImpl>(value_sp->GetTypeImpl());
    sb_type.SetSP(type_sp);
  }

  return LLDB_RECORD_RESULT(sb_type);
}

bool SBValue::GetValueDidChange() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, GetValueDidChange);

  bool result = false;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    if (value_sp->UpdateValueIfNeeded(false))
      result = value_sp->GetValueDidChange();
  }

  return result;
}

const char *SBValue::GetSummary() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBValue, GetSummary);

  const char *cstr = nullptr;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    cstr = value_sp->GetSummaryAsCString();
  }

  return cstr;
}

const char *SBValue::GetSummary(lldb::SBStream &stream,
                                lldb::SBTypeSummaryOptions &options) {
  LLDB_RECORD_METHOD(const char *, SBValue, GetSummary,
                     (lldb::SBStream &, lldb::SBTypeSummaryOptions &), stream,
                     options);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    std::string buffer;
    if (value_sp->GetSummaryAsCString(buffer, options.ref()) && !buffer.empty())
      stream.Printf("%s", buffer.c_str());
  }
  const char *cstr = stream.GetData();
  return cstr;
}

const char *SBValue::GetLocation() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBValue, GetLocation);

  const char *cstr = nullptr;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    cstr = value_sp->GetLocationAsCString();
  }
  return cstr;
}

// Deprecated - use the one that takes an lldb::SBError
bool SBValue::SetValueFromCString(const char *value_str) {
  LLDB_RECORD_METHOD(bool, SBValue, SetValueFromCString, (const char *),
                     value_str);

  lldb::SBError dummy;
  return SetValueFromCString(value_str, dummy);
}

bool SBValue::SetValueFromCString(const char *value_str, lldb::SBError &error) {
  LLDB_RECORD_METHOD(bool, SBValue, SetValueFromCString,
                     (const char *, lldb::SBError &), value_str, error);

  bool success = false;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    success = value_sp->SetValueFromCString(value_str, error.ref());
  } else
    error.SetErrorStringWithFormat("Could not get value: %s",
                                   locker.GetError().AsCString());

  return success;
}

lldb::SBTypeFormat SBValue::GetTypeFormat() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBTypeFormat, SBValue, GetTypeFormat);

  lldb::SBTypeFormat format;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    if (value_sp->UpdateValueIfNeeded(true)) {
      lldb::TypeFormatImplSP format_sp = value_sp->GetValueFormat();
      if (format_sp)
        format.SetSP(format_sp);
    }
  }
  return LLDB_RECORD_RESULT(format);
}

lldb::SBTypeSummary SBValue::GetTypeSummary() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBTypeSummary, SBValue, GetTypeSummary);

  lldb::SBTypeSummary summary;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    if (value_sp->UpdateValueIfNeeded(true)) {
      lldb::TypeSummaryImplSP summary_sp = value_sp->GetSummaryFormat();
      if (summary_sp)
        summary.SetSP(summary_sp);
    }
  }
  return LLDB_RECORD_RESULT(summary);
}

lldb::SBTypeFilter SBValue::GetTypeFilter() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBTypeFilter, SBValue, GetTypeFilter);

  lldb::SBTypeFilter filter;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    if (value_sp->UpdateValueIfNeeded(true)) {
      lldb::SyntheticChildrenSP synthetic_sp = value_sp->GetSyntheticChildren();

      if (synthetic_sp && !synthetic_sp->IsScripted()) {
        TypeFilterImplSP filter_sp =
            std::static_pointer_cast<TypeFilterImpl>(synthetic_sp);
        filter.SetSP(filter_sp);
      }
    }
  }
  return LLDB_RECORD_RESULT(filter);
}

lldb::SBTypeSynthetic SBValue::GetTypeSynthetic() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBTypeSynthetic, SBValue, GetTypeSynthetic);

  lldb::SBTypeSynthetic synthetic;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    if (value_sp->UpdateValueIfNeeded(true)) {
      lldb::SyntheticChildrenSP children_sp = value_sp->GetSyntheticChildren();

      if (children_sp && children_sp->IsScripted()) {
        ScriptedSyntheticChildrenSP synth_sp =
            std::static_pointer_cast<ScriptedSyntheticChildren>(children_sp);
        synthetic.SetSP(synth_sp);
      }
    }
  }
  return LLDB_RECORD_RESULT(synthetic);
}

lldb::SBValue SBValue::CreateChildAtOffset(const char *name, uint32_t offset,
                                           SBType type) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, CreateChildAtOffset,
                     (const char *, uint32_t, lldb::SBType), name, offset,
                     type);

  lldb::SBValue sb_value;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  lldb::ValueObjectSP new_value_sp;
  if (value_sp) {
    TypeImplSP type_sp(type.GetSP());
    if (type.IsValid()) {
      sb_value.SetSP(value_sp->GetSyntheticChildAtOffset(
                         offset, type_sp->GetCompilerType(false), true),
                     GetPreferDynamicValue(), GetPreferSyntheticValue(), name);
    }
  }
  return LLDB_RECORD_RESULT(sb_value);
}

lldb::SBValue SBValue::Cast(SBType type) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, Cast, (lldb::SBType), type);

  lldb::SBValue sb_value;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  TypeImplSP type_sp(type.GetSP());
  if (value_sp && type_sp)
    sb_value.SetSP(value_sp->Cast(type_sp->GetCompilerType(false)),
                   GetPreferDynamicValue(), GetPreferSyntheticValue());
  return LLDB_RECORD_RESULT(sb_value);
}

lldb::SBValue SBValue::CreateValueFromExpression(const char *name,
                                                 const char *expression) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, CreateValueFromExpression,
                     (const char *, const char *), name, expression);

  SBExpressionOptions options;
  options.ref().SetKeepInMemory(true);
  return LLDB_RECORD_RESULT(
      CreateValueFromExpression(name, expression, options));
}

lldb::SBValue SBValue::CreateValueFromExpression(const char *name,
                                                 const char *expression,
                                                 SBExpressionOptions &options) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, CreateValueFromExpression,
                     (const char *, const char *, lldb::SBExpressionOptions &),
                     name, expression, options);

  lldb::SBValue sb_value;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  lldb::ValueObjectSP new_value_sp;
  if (value_sp) {
    ExecutionContext exe_ctx(value_sp->GetExecutionContextRef());
    new_value_sp = ValueObject::CreateValueObjectFromExpression(
        name, expression, exe_ctx, options.ref());
    if (new_value_sp)
      new_value_sp->SetName(ConstString(name));
  }
  sb_value.SetSP(new_value_sp);
  return LLDB_RECORD_RESULT(sb_value);
}

lldb::SBValue SBValue::CreateValueFromAddress(const char *name,
                                              lldb::addr_t address,
                                              SBType sb_type) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, CreateValueFromAddress,
                     (const char *, lldb::addr_t, lldb::SBType), name, address,
                     sb_type);

  lldb::SBValue sb_value;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  lldb::ValueObjectSP new_value_sp;
  lldb::TypeImplSP type_impl_sp(sb_type.GetSP());
  if (value_sp && type_impl_sp) {
    CompilerType ast_type(type_impl_sp->GetCompilerType(true));
    ExecutionContext exe_ctx(value_sp->GetExecutionContextRef());
    new_value_sp = ValueObject::CreateValueObjectFromAddress(name, address,
                                                             exe_ctx, ast_type);
  }
  sb_value.SetSP(new_value_sp);
  return LLDB_RECORD_RESULT(sb_value);
}

lldb::SBValue SBValue::CreateValueFromData(const char *name, SBData data,
                                           SBType sb_type) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, CreateValueFromData,
                     (const char *, lldb::SBData, lldb::SBType), name, data,
                     sb_type);

  lldb::SBValue sb_value;
  lldb::ValueObjectSP new_value_sp;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  lldb::TypeImplSP type_impl_sp(sb_type.GetSP());
  if (value_sp && type_impl_sp) {
    ExecutionContext exe_ctx(value_sp->GetExecutionContextRef());
    new_value_sp = ValueObject::CreateValueObjectFromData(
        name, **data, exe_ctx, type_impl_sp->GetCompilerType(true));
    new_value_sp->SetAddressTypeOfChildren(eAddressTypeLoad);
  }
  sb_value.SetSP(new_value_sp);
  return LLDB_RECORD_RESULT(sb_value);
}

SBValue SBValue::GetChildAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, GetChildAtIndex, (uint32_t), idx);

  const bool can_create_synthetic = false;
  lldb::DynamicValueType use_dynamic = eNoDynamicValues;
  TargetSP target_sp;
  if (m_opaque_sp)
    target_sp = m_opaque_sp->GetTargetSP();

  if (target_sp)
    use_dynamic = target_sp->GetPreferDynamicValue();

  return LLDB_RECORD_RESULT(
      GetChildAtIndex(idx, use_dynamic, can_create_synthetic));
}

SBValue SBValue::GetChildAtIndex(uint32_t idx,
                                 lldb::DynamicValueType use_dynamic,
                                 bool can_create_synthetic) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, GetChildAtIndex,
                     (uint32_t, lldb::DynamicValueType, bool), idx, use_dynamic,
                     can_create_synthetic);

  lldb::ValueObjectSP child_sp;

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    const bool can_create = true;
    child_sp = value_sp->GetChildAtIndex(idx, can_create);
    if (can_create_synthetic && !child_sp) {
      child_sp = value_sp->GetSyntheticArrayMember(idx, can_create);
    }
  }

  SBValue sb_value;
  sb_value.SetSP(child_sp, use_dynamic, GetPreferSyntheticValue());

  return LLDB_RECORD_RESULT(sb_value);
}

uint32_t SBValue::GetIndexOfChildWithName(const char *name) {
  LLDB_RECORD_METHOD(uint32_t, SBValue, GetIndexOfChildWithName, (const char *),
                     name);

  uint32_t idx = UINT32_MAX;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    idx = value_sp->GetIndexOfChildWithName(ConstString(name));
  }
  return idx;
}

SBValue SBValue::GetChildMemberWithName(const char *name) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, GetChildMemberWithName,
                     (const char *), name);

  lldb::DynamicValueType use_dynamic_value = eNoDynamicValues;
  TargetSP target_sp;
  if (m_opaque_sp)
    target_sp = m_opaque_sp->GetTargetSP();

  if (target_sp)
    use_dynamic_value = target_sp->GetPreferDynamicValue();
  return LLDB_RECORD_RESULT(GetChildMemberWithName(name, use_dynamic_value));
}

SBValue
SBValue::GetChildMemberWithName(const char *name,
                                lldb::DynamicValueType use_dynamic_value) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, GetChildMemberWithName,
                     (const char *, lldb::DynamicValueType), name,
                     use_dynamic_value);

  lldb::ValueObjectSP child_sp;
  const ConstString str_name(name);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    child_sp = value_sp->GetChildMemberWithName(str_name, true);
  }

  SBValue sb_value;
  sb_value.SetSP(child_sp, use_dynamic_value, GetPreferSyntheticValue());

  return LLDB_RECORD_RESULT(sb_value);
}

lldb::SBValue SBValue::GetDynamicValue(lldb::DynamicValueType use_dynamic) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, GetDynamicValue,
                     (lldb::DynamicValueType), use_dynamic);

  SBValue value_sb;
  if (IsValid()) {
    ValueImplSP proxy_sp(new ValueImpl(m_opaque_sp->GetRootSP(), use_dynamic,
                                       m_opaque_sp->GetUseSynthetic()));
    value_sb.SetSP(proxy_sp);
  }
  return LLDB_RECORD_RESULT(value_sb);
}

lldb::SBValue SBValue::GetStaticValue() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBValue, SBValue, GetStaticValue);

  SBValue value_sb;
  if (IsValid()) {
    ValueImplSP proxy_sp(new ValueImpl(m_opaque_sp->GetRootSP(),
                                       eNoDynamicValues,
                                       m_opaque_sp->GetUseSynthetic()));
    value_sb.SetSP(proxy_sp);
  }
  return LLDB_RECORD_RESULT(value_sb);
}

lldb::SBValue SBValue::GetNonSyntheticValue() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBValue, SBValue, GetNonSyntheticValue);

  SBValue value_sb;
  if (IsValid()) {
    ValueImplSP proxy_sp(new ValueImpl(m_opaque_sp->GetRootSP(),
                                       m_opaque_sp->GetUseDynamic(), false));
    value_sb.SetSP(proxy_sp);
  }
  return LLDB_RECORD_RESULT(value_sb);
}

lldb::DynamicValueType SBValue::GetPreferDynamicValue() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::DynamicValueType, SBValue,
                             GetPreferDynamicValue);

  if (!IsValid())
    return eNoDynamicValues;
  return m_opaque_sp->GetUseDynamic();
}

void SBValue::SetPreferDynamicValue(lldb::DynamicValueType use_dynamic) {
  LLDB_RECORD_METHOD(void, SBValue, SetPreferDynamicValue,
                     (lldb::DynamicValueType), use_dynamic);

  if (IsValid())
    return m_opaque_sp->SetUseDynamic(use_dynamic);
}

bool SBValue::GetPreferSyntheticValue() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, GetPreferSyntheticValue);

  if (!IsValid())
    return false;
  return m_opaque_sp->GetUseSynthetic();
}

void SBValue::SetPreferSyntheticValue(bool use_synthetic) {
  LLDB_RECORD_METHOD(void, SBValue, SetPreferSyntheticValue, (bool),
                     use_synthetic);

  if (IsValid())
    return m_opaque_sp->SetUseSynthetic(use_synthetic);
}

bool SBValue::IsDynamic() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, IsDynamic);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    return value_sp->IsDynamic();
  return false;
}

bool SBValue::IsSynthetic() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, IsSynthetic);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    return value_sp->IsSynthetic();
  return false;
}

bool SBValue::IsSyntheticChildrenGenerated() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, IsSyntheticChildrenGenerated);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    return value_sp->IsSyntheticChildrenGenerated();
  return false;
}

void SBValue::SetSyntheticChildrenGenerated(bool is) {
  LLDB_RECORD_METHOD(void, SBValue, SetSyntheticChildrenGenerated, (bool), is);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    return value_sp->SetSyntheticChildrenGenerated(is);
}

lldb::SBValue SBValue::GetValueForExpressionPath(const char *expr_path) {
  LLDB_RECORD_METHOD(lldb::SBValue, SBValue, GetValueForExpressionPath,
                     (const char *), expr_path);

  lldb::ValueObjectSP child_sp;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    // using default values for all the fancy options, just do it if you can
    child_sp = value_sp->GetValueForExpressionPath(expr_path);
  }

  SBValue sb_value;
  sb_value.SetSP(child_sp, GetPreferDynamicValue(), GetPreferSyntheticValue());

  return LLDB_RECORD_RESULT(sb_value);
}

int64_t SBValue::GetValueAsSigned(SBError &error, int64_t fail_value) {
  LLDB_RECORD_METHOD(int64_t, SBValue, GetValueAsSigned,
                     (lldb::SBError &, int64_t), error, fail_value);

  error.Clear();
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    bool success = true;
    uint64_t ret_val = fail_value;
    ret_val = value_sp->GetValueAsSigned(fail_value, &success);
    if (!success)
      error.SetErrorString("could not resolve value");
    return ret_val;
  } else
    error.SetErrorStringWithFormat("could not get SBValue: %s",
                                   locker.GetError().AsCString());

  return fail_value;
}

uint64_t SBValue::GetValueAsUnsigned(SBError &error, uint64_t fail_value) {
  LLDB_RECORD_METHOD(uint64_t, SBValue, GetValueAsUnsigned,
                     (lldb::SBError &, uint64_t), error, fail_value);

  error.Clear();
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    bool success = true;
    uint64_t ret_val = fail_value;
    ret_val = value_sp->GetValueAsUnsigned(fail_value, &success);
    if (!success)
      error.SetErrorString("could not resolve value");
    return ret_val;
  } else
    error.SetErrorStringWithFormat("could not get SBValue: %s",
                                   locker.GetError().AsCString());

  return fail_value;
}

int64_t SBValue::GetValueAsSigned(int64_t fail_value) {
  LLDB_RECORD_METHOD(int64_t, SBValue, GetValueAsSigned, (int64_t), fail_value);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    return value_sp->GetValueAsSigned(fail_value);
  }
  return fail_value;
}

uint64_t SBValue::GetValueAsUnsigned(uint64_t fail_value) {
  LLDB_RECORD_METHOD(uint64_t, SBValue, GetValueAsUnsigned, (uint64_t),
                     fail_value);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    return value_sp->GetValueAsUnsigned(fail_value);
  }
  return fail_value;
}

bool SBValue::MightHaveChildren() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, MightHaveChildren);

  bool has_children = false;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    has_children = value_sp->MightHaveChildren();

  return has_children;
}

bool SBValue::IsRuntimeSupportValue() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, IsRuntimeSupportValue);

  bool is_support = false;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    is_support = value_sp->IsRuntimeSupportValue();

  return is_support;
}

uint32_t SBValue::GetNumChildren() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBValue, GetNumChildren);

  return GetNumChildren(UINT32_MAX);
}

uint32_t SBValue::GetNumChildren(uint32_t max) {
  LLDB_RECORD_METHOD(uint32_t, SBValue, GetNumChildren, (uint32_t), max);

  uint32_t num_children = 0;

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    num_children = value_sp->GetNumChildren(max);

  return num_children;
}

SBValue SBValue::Dereference() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBValue, SBValue, Dereference);

  SBValue sb_value;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    Status error;
    sb_value = value_sp->Dereference(error);
  }

  return LLDB_RECORD_RESULT(sb_value);
}

// Deprecated - please use GetType().IsPointerType() instead.
bool SBValue::TypeIsPointerType() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBValue, TypeIsPointerType);

  return GetType().IsPointerType();
}

void *SBValue::GetOpaqueType() {
  LLDB_RECORD_METHOD_NO_ARGS(void *, SBValue, GetOpaqueType);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    return value_sp->GetCompilerType().GetOpaqueQualType();
  return nullptr;
}

lldb::SBTarget SBValue::GetTarget() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBTarget, SBValue, GetTarget);

  SBTarget sb_target;
  TargetSP target_sp;
  if (m_opaque_sp) {
    target_sp = m_opaque_sp->GetTargetSP();
    sb_target.SetSP(target_sp);
  }

  return LLDB_RECORD_RESULT(sb_target);
}

lldb::SBProcess SBValue::GetProcess() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBProcess, SBValue, GetProcess);

  SBProcess sb_process;
  ProcessSP process_sp;
  if (m_opaque_sp) {
    process_sp = m_opaque_sp->GetProcessSP();
    sb_process.SetSP(process_sp);
  }

  return LLDB_RECORD_RESULT(sb_process);
}

lldb::SBThread SBValue::GetThread() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBThread, SBValue, GetThread);

  SBThread sb_thread;
  ThreadSP thread_sp;
  if (m_opaque_sp) {
    thread_sp = m_opaque_sp->GetThreadSP();
    sb_thread.SetThread(thread_sp);
  }

  return LLDB_RECORD_RESULT(sb_thread);
}

lldb::SBFrame SBValue::GetFrame() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBFrame, SBValue, GetFrame);

  SBFrame sb_frame;
  StackFrameSP frame_sp;
  if (m_opaque_sp) {
    frame_sp = m_opaque_sp->GetFrameSP();
    sb_frame.SetFrameSP(frame_sp);
  }

  return LLDB_RECORD_RESULT(sb_frame);
}

lldb::ValueObjectSP SBValue::GetSP(ValueLocker &locker) const {
  if (!m_opaque_sp || !m_opaque_sp->IsValid()) {
    locker.GetError().SetErrorString("No value");
    return ValueObjectSP();
  }
  return locker.GetLockedSP(*m_opaque_sp.get());
}

lldb::ValueObjectSP SBValue::GetSP() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::ValueObjectSP, SBValue, GetSP);

  ValueLocker locker;
  return LLDB_RECORD_RESULT(GetSP(locker));
}

void SBValue::SetSP(ValueImplSP impl_sp) { m_opaque_sp = impl_sp; }

void SBValue::SetSP(const lldb::ValueObjectSP &sp) {
  if (sp) {
    lldb::TargetSP target_sp(sp->GetTargetSP());
    if (target_sp) {
      lldb::DynamicValueType use_dynamic = target_sp->GetPreferDynamicValue();
      bool use_synthetic =
          target_sp->TargetProperties::GetEnableSyntheticValue();
      m_opaque_sp = ValueImplSP(new ValueImpl(sp, use_dynamic, use_synthetic));
    } else
      m_opaque_sp = ValueImplSP(new ValueImpl(sp, eNoDynamicValues, true));
  } else
    m_opaque_sp = ValueImplSP(new ValueImpl(sp, eNoDynamicValues, false));
}

void SBValue::SetSP(const lldb::ValueObjectSP &sp,
                    lldb::DynamicValueType use_dynamic) {
  if (sp) {
    lldb::TargetSP target_sp(sp->GetTargetSP());
    if (target_sp) {
      bool use_synthetic =
          target_sp->TargetProperties::GetEnableSyntheticValue();
      SetSP(sp, use_dynamic, use_synthetic);
    } else
      SetSP(sp, use_dynamic, true);
  } else
    SetSP(sp, use_dynamic, false);
}

void SBValue::SetSP(const lldb::ValueObjectSP &sp, bool use_synthetic) {
  if (sp) {
    lldb::TargetSP target_sp(sp->GetTargetSP());
    if (target_sp) {
      lldb::DynamicValueType use_dynamic = target_sp->GetPreferDynamicValue();
      SetSP(sp, use_dynamic, use_synthetic);
    } else
      SetSP(sp, eNoDynamicValues, use_synthetic);
  } else
    SetSP(sp, eNoDynamicValues, use_synthetic);
}

void SBValue::SetSP(const lldb::ValueObjectSP &sp,
                    lldb::DynamicValueType use_dynamic, bool use_synthetic) {
  m_opaque_sp = ValueImplSP(new ValueImpl(sp, use_dynamic, use_synthetic));
}

void SBValue::SetSP(const lldb::ValueObjectSP &sp,
                    lldb::DynamicValueType use_dynamic, bool use_synthetic,
                    const char *name) {
  m_opaque_sp =
      ValueImplSP(new ValueImpl(sp, use_dynamic, use_synthetic, name));
}

bool SBValue::GetExpressionPath(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBValue, GetExpressionPath, (lldb::SBStream &),
                     description);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    value_sp->GetExpressionPath(description.ref());
    return true;
  }
  return false;
}

bool SBValue::GetExpressionPath(SBStream &description,
                                bool qualify_cxx_base_classes) {
  LLDB_RECORD_METHOD(bool, SBValue, GetExpressionPath, (lldb::SBStream &, bool),
                     description, qualify_cxx_base_classes);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    value_sp->GetExpressionPath(description.ref());
    return true;
  }
  return false;
}

lldb::SBValue SBValue::EvaluateExpression(const char *expr) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBValue, SBValue, EvaluateExpression,
                           (const char *), expr);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (!value_sp)
    return LLDB_RECORD_RESULT(SBValue());

  lldb::TargetSP target_sp = value_sp->GetTargetSP();
  if (!target_sp)
    return LLDB_RECORD_RESULT(SBValue());

  lldb::SBExpressionOptions options;
  options.SetFetchDynamicValue(target_sp->GetPreferDynamicValue());
  options.SetUnwindOnError(true);
  options.SetIgnoreBreakpoints(true);

  return LLDB_RECORD_RESULT(EvaluateExpression(expr, options, nullptr));
}

lldb::SBValue
SBValue::EvaluateExpression(const char *expr,
                            const SBExpressionOptions &options) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBValue, SBValue, EvaluateExpression,
                           (const char *, const lldb::SBExpressionOptions &),
                           expr, options);

  return LLDB_RECORD_RESULT(EvaluateExpression(expr, options, nullptr));
}

lldb::SBValue SBValue::EvaluateExpression(const char *expr,
                                          const SBExpressionOptions &options,
                                          const char *name) const {
  LLDB_RECORD_METHOD_CONST(
      lldb::SBValue, SBValue, EvaluateExpression,
      (const char *, const lldb::SBExpressionOptions &, const char *), expr,
      options, name);


  if (!expr || expr[0] == '\0') {
    return LLDB_RECORD_RESULT(SBValue());
  }


  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (!value_sp) {
    return LLDB_RECORD_RESULT(SBValue());
  }

  lldb::TargetSP target_sp = value_sp->GetTargetSP();
  if (!target_sp) {
    return LLDB_RECORD_RESULT(SBValue());
  }

  std::lock_guard<std::recursive_mutex> guard(target_sp->GetAPIMutex());
  ExecutionContext exe_ctx(target_sp.get());

  StackFrame *frame = exe_ctx.GetFramePtr();
  if (!frame) {
    return LLDB_RECORD_RESULT(SBValue());
  }

  ValueObjectSP res_val_sp;
  target_sp->EvaluateExpression(expr, frame, res_val_sp, options.ref(), nullptr,
                                value_sp.get());

  if (name)
    res_val_sp->SetName(ConstString(name));

  SBValue result;
  result.SetSP(res_val_sp, options.GetFetchDynamicValue());
  return LLDB_RECORD_RESULT(result);
}

bool SBValue::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBValue, GetDescription, (lldb::SBStream &),
                     description);

  Stream &strm = description.ref();

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    value_sp->Dump(strm);
  else
    strm.PutCString("No value");

  return true;
}

lldb::Format SBValue::GetFormat() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::Format, SBValue, GetFormat);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    return value_sp->GetFormat();
  return eFormatDefault;
}

void SBValue::SetFormat(lldb::Format format) {
  LLDB_RECORD_METHOD(void, SBValue, SetFormat, (lldb::Format), format);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp)
    value_sp->SetFormat(format);
}

lldb::SBValue SBValue::AddressOf() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBValue, SBValue, AddressOf);

  SBValue sb_value;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    Status error;
    sb_value.SetSP(value_sp->AddressOf(error), GetPreferDynamicValue(),
                   GetPreferSyntheticValue());
  }

  return LLDB_RECORD_RESULT(sb_value);
}

lldb::addr_t SBValue::GetLoadAddress() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::addr_t, SBValue, GetLoadAddress);

  lldb::addr_t value = LLDB_INVALID_ADDRESS;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    TargetSP target_sp(value_sp->GetTargetSP());
    if (target_sp) {
      const bool scalar_is_load_address = true;
      AddressType addr_type;
      value = value_sp->GetAddressOf(scalar_is_load_address, &addr_type);
      if (addr_type == eAddressTypeFile) {
        ModuleSP module_sp(value_sp->GetModule());
        if (!module_sp)
          value = LLDB_INVALID_ADDRESS;
        else {
          Address addr;
          module_sp->ResolveFileAddress(value, addr);
          value = addr.GetLoadAddress(target_sp.get());
        }
      } else if (addr_type == eAddressTypeHost ||
                 addr_type == eAddressTypeInvalid)
        value = LLDB_INVALID_ADDRESS;
    }
  }

  return value;
}

lldb::SBAddress SBValue::GetAddress() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBAddress, SBValue, GetAddress);

  Address addr;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    TargetSP target_sp(value_sp->GetTargetSP());
    if (target_sp) {
      lldb::addr_t value = LLDB_INVALID_ADDRESS;
      const bool scalar_is_load_address = true;
      AddressType addr_type;
      value = value_sp->GetAddressOf(scalar_is_load_address, &addr_type);
      if (addr_type == eAddressTypeFile) {
        ModuleSP module_sp(value_sp->GetModule());
        if (module_sp)
          module_sp->ResolveFileAddress(value, addr);
      } else if (addr_type == eAddressTypeLoad) {
        // no need to check the return value on this.. if it can actually do
        // the resolve addr will be in the form (section,offset), otherwise it
        // will simply be returned as (NULL, value)
        addr.SetLoadAddress(value, target_sp.get());
      }
    }
  }

  return LLDB_RECORD_RESULT(SBAddress(new Address(addr)));
}

lldb::SBData SBValue::GetPointeeData(uint32_t item_idx, uint32_t item_count) {
  LLDB_RECORD_METHOD(lldb::SBData, SBValue, GetPointeeData,
                     (uint32_t, uint32_t), item_idx, item_count);

  lldb::SBData sb_data;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    TargetSP target_sp(value_sp->GetTargetSP());
    if (target_sp) {
      DataExtractorSP data_sp(new DataExtractor());
      value_sp->GetPointeeData(*data_sp, item_idx, item_count);
      if (data_sp->GetByteSize() > 0)
        *sb_data = data_sp;
    }
  }

  return LLDB_RECORD_RESULT(sb_data);
}

lldb::SBData SBValue::GetData() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBData, SBValue, GetData);

  lldb::SBData sb_data;
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  if (value_sp) {
    DataExtractorSP data_sp(new DataExtractor());
    Status error;
    value_sp->GetData(*data_sp, error);
    if (error.Success())
      *sb_data = data_sp;
  }

  return LLDB_RECORD_RESULT(sb_data);
}

bool SBValue::SetData(lldb::SBData &data, SBError &error) {
  LLDB_RECORD_METHOD(bool, SBValue, SetData, (lldb::SBData &, lldb::SBError &),
                     data, error);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  bool ret = true;

  if (value_sp) {
    DataExtractor *data_extractor = data.get();

    if (!data_extractor) {
      error.SetErrorString("No data to set");
      ret = false;
    } else {
      Status set_error;

      value_sp->SetData(*data_extractor, set_error);

      if (!set_error.Success()) {
        error.SetErrorStringWithFormat("Couldn't set data: %s",
                                       set_error.AsCString());
        ret = false;
      }
    }
  } else {
    error.SetErrorStringWithFormat(
        "Couldn't set data: could not get SBValue: %s",
        locker.GetError().AsCString());
    ret = false;
  }

  return ret;
}

lldb::SBDeclaration SBValue::GetDeclaration() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBDeclaration, SBValue, GetDeclaration);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  SBDeclaration decl_sb;
  if (value_sp) {
    Declaration decl;
    if (value_sp->GetDeclaration(decl))
      decl_sb.SetDeclaration(decl);
  }
  return LLDB_RECORD_RESULT(decl_sb);
}

lldb::SBWatchpoint SBValue::Watch(bool resolve_location, bool read, bool write,
                                  SBError &error) {
  LLDB_RECORD_METHOD(lldb::SBWatchpoint, SBValue, Watch,
                     (bool, bool, bool, lldb::SBError &), resolve_location,
                     read, write, error);

  SBWatchpoint sb_watchpoint;

  // If the SBValue is not valid, there's no point in even trying to watch it.
  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  TargetSP target_sp(GetTarget().GetSP());
  if (value_sp && target_sp) {
    // Read and Write cannot both be false.
    if (!read && !write)
      return LLDB_RECORD_RESULT(sb_watchpoint);

    // If the value is not in scope, don't try and watch and invalid value
    if (!IsInScope())
      return LLDB_RECORD_RESULT(sb_watchpoint);

    addr_t addr = GetLoadAddress();
    if (addr == LLDB_INVALID_ADDRESS)
      return LLDB_RECORD_RESULT(sb_watchpoint);
    size_t byte_size = GetByteSize();
    if (byte_size == 0)
      return LLDB_RECORD_RESULT(sb_watchpoint);

    uint32_t watch_type = 0;
    if (read)
      watch_type |= LLDB_WATCH_TYPE_READ;
    if (write)
      watch_type |= LLDB_WATCH_TYPE_WRITE;

    Status rc;
    CompilerType type(value_sp->GetCompilerType());
    WatchpointSP watchpoint_sp =
        target_sp->CreateWatchpoint(addr, byte_size, &type, watch_type, rc);
    error.SetError(rc);

    if (watchpoint_sp) {
      sb_watchpoint.SetSP(watchpoint_sp);
      Declaration decl;
      if (value_sp->GetDeclaration(decl)) {
        if (decl.GetFile()) {
          StreamString ss;
          // True to show fullpath for declaration file.
          decl.DumpStopContext(&ss, true);
          watchpoint_sp->SetDeclInfo(std::string(ss.GetString()));
        }
      }
    }
  } else if (target_sp) {
    error.SetErrorStringWithFormat("could not get SBValue: %s",
                                   locker.GetError().AsCString());
  } else {
    error.SetErrorString("could not set watchpoint, a target is required");
  }

  return LLDB_RECORD_RESULT(sb_watchpoint);
}

// FIXME: Remove this method impl (as well as the decl in .h) once it is no
// longer needed.
// Backward compatibility fix in the interim.
lldb::SBWatchpoint SBValue::Watch(bool resolve_location, bool read,
                                  bool write) {
  LLDB_RECORD_METHOD(lldb::SBWatchpoint, SBValue, Watch, (bool, bool, bool),
                     resolve_location, read, write);

  SBError error;
  return LLDB_RECORD_RESULT(Watch(resolve_location, read, write, error));
}

lldb::SBWatchpoint SBValue::WatchPointee(bool resolve_location, bool read,
                                         bool write, SBError &error) {
  LLDB_RECORD_METHOD(lldb::SBWatchpoint, SBValue, WatchPointee,
                     (bool, bool, bool, lldb::SBError &), resolve_location,
                     read, write, error);

  SBWatchpoint sb_watchpoint;
  if (IsInScope() && GetType().IsPointerType())
    sb_watchpoint = Dereference().Watch(resolve_location, read, write, error);
  return LLDB_RECORD_RESULT(sb_watchpoint);
}

lldb::SBValue SBValue::Persist() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBValue, SBValue, Persist);

  ValueLocker locker;
  lldb::ValueObjectSP value_sp(GetSP(locker));
  SBValue persisted_sb;
  if (value_sp) {
    persisted_sb.SetSP(value_sp->Persist());
  }
  return LLDB_RECORD_RESULT(persisted_sb);
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBValue>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBValue, ());
  LLDB_REGISTER_CONSTRUCTOR(SBValue, (const lldb::ValueObjectSP &));
  LLDB_REGISTER_CONSTRUCTOR(SBValue, (const lldb::SBValue &));
  LLDB_REGISTER_METHOD(lldb::SBValue &,
                       SBValue, operator=,(const lldb::SBValue &));
  LLDB_REGISTER_METHOD(bool, SBValue, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBValue, operator bool, ());
  LLDB_REGISTER_METHOD(void, SBValue, Clear, ());
  LLDB_REGISTER_METHOD(lldb::SBError, SBValue, GetError, ());
  LLDB_REGISTER_METHOD(lldb::user_id_t, SBValue, GetID, ());
  LLDB_REGISTER_METHOD(const char *, SBValue, GetName, ());
  LLDB_REGISTER_METHOD(const char *, SBValue, GetTypeName, ());
  LLDB_REGISTER_METHOD(const char *, SBValue, GetDisplayTypeName, ());
  LLDB_REGISTER_METHOD(size_t, SBValue, GetByteSize, ());
  LLDB_REGISTER_METHOD(bool, SBValue, IsInScope, ());
  LLDB_REGISTER_METHOD(const char *, SBValue, GetValue, ());
  LLDB_REGISTER_METHOD(lldb::ValueType, SBValue, GetValueType, ());
  LLDB_REGISTER_METHOD(const char *, SBValue, GetObjectDescription, ());
  LLDB_REGISTER_METHOD(lldb::SBType, SBValue, GetType, ());
  LLDB_REGISTER_METHOD(bool, SBValue, GetValueDidChange, ());
  LLDB_REGISTER_METHOD(const char *, SBValue, GetSummary, ());
  LLDB_REGISTER_METHOD(const char *, SBValue, GetSummary,
                       (lldb::SBStream &, lldb::SBTypeSummaryOptions &));
  LLDB_REGISTER_METHOD(const char *, SBValue, GetLocation, ());
  LLDB_REGISTER_METHOD(bool, SBValue, SetValueFromCString, (const char *));
  LLDB_REGISTER_METHOD(bool, SBValue, SetValueFromCString,
                       (const char *, lldb::SBError &));
  LLDB_REGISTER_METHOD(lldb::SBTypeFormat, SBValue, GetTypeFormat, ());
  LLDB_REGISTER_METHOD(lldb::SBTypeSummary, SBValue, GetTypeSummary, ());
  LLDB_REGISTER_METHOD(lldb::SBTypeFilter, SBValue, GetTypeFilter, ());
  LLDB_REGISTER_METHOD(lldb::SBTypeSynthetic, SBValue, GetTypeSynthetic, ());
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, CreateChildAtOffset,
                       (const char *, uint32_t, lldb::SBType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, Cast, (lldb::SBType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, CreateValueFromExpression,
                       (const char *, const char *));
  LLDB_REGISTER_METHOD(
      lldb::SBValue, SBValue, CreateValueFromExpression,
      (const char *, const char *, lldb::SBExpressionOptions &));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, CreateValueFromAddress,
                       (const char *, lldb::addr_t, lldb::SBType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, CreateValueFromData,
                       (const char *, lldb::SBData, lldb::SBType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetChildAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetChildAtIndex,
                       (uint32_t, lldb::DynamicValueType, bool));
  LLDB_REGISTER_METHOD(uint32_t, SBValue, GetIndexOfChildWithName,
                       (const char *));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetChildMemberWithName,
                       (const char *));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetChildMemberWithName,
                       (const char *, lldb::DynamicValueType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetDynamicValue,
                       (lldb::DynamicValueType));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetStaticValue, ());
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetNonSyntheticValue, ());
  LLDB_REGISTER_METHOD(lldb::DynamicValueType, SBValue, GetPreferDynamicValue,
                       ());
  LLDB_REGISTER_METHOD(void, SBValue, SetPreferDynamicValue,
                       (lldb::DynamicValueType));
  LLDB_REGISTER_METHOD(bool, SBValue, GetPreferSyntheticValue, ());
  LLDB_REGISTER_METHOD(void, SBValue, SetPreferSyntheticValue, (bool));
  LLDB_REGISTER_METHOD(bool, SBValue, IsDynamic, ());
  LLDB_REGISTER_METHOD(bool, SBValue, IsSynthetic, ());
  LLDB_REGISTER_METHOD(bool, SBValue, IsSyntheticChildrenGenerated, ());
  LLDB_REGISTER_METHOD(void, SBValue, SetSyntheticChildrenGenerated, (bool));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, GetValueForExpressionPath,
                       (const char *));
  LLDB_REGISTER_METHOD(int64_t, SBValue, GetValueAsSigned,
                       (lldb::SBError &, int64_t));
  LLDB_REGISTER_METHOD(uint64_t, SBValue, GetValueAsUnsigned,
                       (lldb::SBError &, uint64_t));
  LLDB_REGISTER_METHOD(int64_t, SBValue, GetValueAsSigned, (int64_t));
  LLDB_REGISTER_METHOD(uint64_t, SBValue, GetValueAsUnsigned, (uint64_t));
  LLDB_REGISTER_METHOD(bool, SBValue, MightHaveChildren, ());
  LLDB_REGISTER_METHOD(bool, SBValue, IsRuntimeSupportValue, ());
  LLDB_REGISTER_METHOD(uint32_t, SBValue, GetNumChildren, ());
  LLDB_REGISTER_METHOD(uint32_t, SBValue, GetNumChildren, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, Dereference, ());
  LLDB_REGISTER_METHOD(bool, SBValue, TypeIsPointerType, ());
  LLDB_REGISTER_METHOD(void *, SBValue, GetOpaqueType, ());
  LLDB_REGISTER_METHOD(lldb::SBTarget, SBValue, GetTarget, ());
  LLDB_REGISTER_METHOD(lldb::SBProcess, SBValue, GetProcess, ());
  LLDB_REGISTER_METHOD(lldb::SBThread, SBValue, GetThread, ());
  LLDB_REGISTER_METHOD(lldb::SBFrame, SBValue, GetFrame, ());
  LLDB_REGISTER_METHOD_CONST(lldb::ValueObjectSP, SBValue, GetSP, ());
  LLDB_REGISTER_METHOD(bool, SBValue, GetExpressionPath, (lldb::SBStream &));
  LLDB_REGISTER_METHOD(bool, SBValue, GetExpressionPath,
                       (lldb::SBStream &, bool));
  LLDB_REGISTER_METHOD_CONST(lldb::SBValue, SBValue, EvaluateExpression,
                             (const char *));
  LLDB_REGISTER_METHOD_CONST(
      lldb::SBValue, SBValue, EvaluateExpression,
      (const char *, const lldb::SBExpressionOptions &));
  LLDB_REGISTER_METHOD_CONST(
      lldb::SBValue, SBValue, EvaluateExpression,
      (const char *, const lldb::SBExpressionOptions &, const char *));
  LLDB_REGISTER_METHOD(bool, SBValue, GetDescription, (lldb::SBStream &));
  LLDB_REGISTER_METHOD(lldb::Format, SBValue, GetFormat, ());
  LLDB_REGISTER_METHOD(void, SBValue, SetFormat, (lldb::Format));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, AddressOf, ());
  LLDB_REGISTER_METHOD(lldb::addr_t, SBValue, GetLoadAddress, ());
  LLDB_REGISTER_METHOD(lldb::SBAddress, SBValue, GetAddress, ());
  LLDB_REGISTER_METHOD(lldb::SBData, SBValue, GetPointeeData,
                       (uint32_t, uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBData, SBValue, GetData, ());
  LLDB_REGISTER_METHOD(bool, SBValue, SetData,
                       (lldb::SBData &, lldb::SBError &));
  LLDB_REGISTER_METHOD(lldb::SBDeclaration, SBValue, GetDeclaration, ());
  LLDB_REGISTER_METHOD(lldb::SBWatchpoint, SBValue, Watch,
                       (bool, bool, bool, lldb::SBError &));
  LLDB_REGISTER_METHOD(lldb::SBWatchpoint, SBValue, Watch,
                       (bool, bool, bool));
  LLDB_REGISTER_METHOD(lldb::SBWatchpoint, SBValue, WatchPointee,
                       (bool, bool, bool, lldb::SBError &));
  LLDB_REGISTER_METHOD(lldb::SBValue, SBValue, Persist, ());
}

}
}
