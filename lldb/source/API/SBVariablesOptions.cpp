//===-- SBVariablesOptions.cpp --------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBVariablesOptions.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBTarget.h"
#include "lldb/Target/Target.h"

#include "lldb/lldb-private.h"

using namespace lldb;
using namespace lldb_private;

class VariablesOptionsImpl {
public:
  VariablesOptionsImpl()
      : m_include_arguments(false), m_include_locals(false),
        m_include_statics(false), m_in_scope_only(false),
        m_include_runtime_support_values(false),
        m_include_recognized_arguments(eLazyBoolCalculate),
        m_use_dynamic(lldb::eNoDynamicValues) {}

  VariablesOptionsImpl(const VariablesOptionsImpl &) = default;

  ~VariablesOptionsImpl() = default;

  VariablesOptionsImpl &operator=(const VariablesOptionsImpl &) = default;

  bool GetIncludeArguments() const { return m_include_arguments; }

  void SetIncludeArguments(bool b) { m_include_arguments = b; }

  bool GetIncludeRecognizedArguments(const lldb::TargetSP &target_sp) const {
    if (m_include_recognized_arguments != eLazyBoolCalculate)
        return m_include_recognized_arguments;
    return target_sp ? target_sp->GetDisplayRecognizedArguments() : false;
  }

  void SetIncludeRecognizedArguments(bool b) {
    m_include_recognized_arguments = b ? eLazyBoolYes : eLazyBoolNo;
  }

  bool GetIncludeLocals() const { return m_include_locals; }

  void SetIncludeLocals(bool b) { m_include_locals = b; }

  bool GetIncludeStatics() const { return m_include_statics; }

  void SetIncludeStatics(bool b) { m_include_statics = b; }

  bool GetInScopeOnly() const { return m_in_scope_only; }

  void SetInScopeOnly(bool b) { m_in_scope_only = b; }

  bool GetIncludeRuntimeSupportValues() const {
    return m_include_runtime_support_values;
  }

  void SetIncludeRuntimeSupportValues(bool b) {
    m_include_runtime_support_values = b;
  }

  lldb::DynamicValueType GetUseDynamic() const { return m_use_dynamic; }

  void SetUseDynamic(lldb::DynamicValueType d) { m_use_dynamic = d; }

private:
  bool m_include_arguments : 1;
  bool m_include_locals : 1;
  bool m_include_statics : 1;
  bool m_in_scope_only : 1;
  bool m_include_runtime_support_values : 1;
  LazyBool m_include_recognized_arguments; // can be overridden with a setting
  lldb::DynamicValueType m_use_dynamic;
};

SBVariablesOptions::SBVariablesOptions()
    : m_opaque_up(new VariablesOptionsImpl()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBVariablesOptions);
}

SBVariablesOptions::SBVariablesOptions(const SBVariablesOptions &options)
    : m_opaque_up(new VariablesOptionsImpl(options.ref())) {
  LLDB_RECORD_CONSTRUCTOR(SBVariablesOptions,
                          (const lldb::SBVariablesOptions &), options);
}

SBVariablesOptions &SBVariablesOptions::
operator=(const SBVariablesOptions &options) {
  LLDB_RECORD_METHOD(
      lldb::SBVariablesOptions &,
      SBVariablesOptions, operator=,(const lldb::SBVariablesOptions &),
      options);

  m_opaque_up.reset(new VariablesOptionsImpl(options.ref()));
  return *this;
}

SBVariablesOptions::~SBVariablesOptions() = default;

bool SBVariablesOptions::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBVariablesOptions, IsValid);

  return m_opaque_up != nullptr;
}

bool SBVariablesOptions::GetIncludeArguments() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBVariablesOptions,
                                   GetIncludeArguments);

  return m_opaque_up->GetIncludeArguments();
}

void SBVariablesOptions::SetIncludeArguments(bool arguments) {
  LLDB_RECORD_METHOD(void, SBVariablesOptions, SetIncludeArguments, (bool),
                     arguments);

  m_opaque_up->SetIncludeArguments(arguments);
}

bool SBVariablesOptions::GetIncludeRecognizedArguments(
    const lldb::SBTarget &target) const {
  LLDB_RECORD_METHOD_CONST(bool, SBVariablesOptions,
                           GetIncludeRecognizedArguments,
                           (const lldb::SBTarget &), target);

  return m_opaque_up->GetIncludeRecognizedArguments(target.GetSP());
}

void SBVariablesOptions::SetIncludeRecognizedArguments(bool arguments) {
  LLDB_RECORD_METHOD(void, SBVariablesOptions, SetIncludeRecognizedArguments,
                     (bool), arguments);

  m_opaque_up->SetIncludeRecognizedArguments(arguments);
}

bool SBVariablesOptions::GetIncludeLocals() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBVariablesOptions, GetIncludeLocals);

  return m_opaque_up->GetIncludeLocals();
}

void SBVariablesOptions::SetIncludeLocals(bool locals) {
  LLDB_RECORD_METHOD(void, SBVariablesOptions, SetIncludeLocals, (bool),
                     locals);

  m_opaque_up->SetIncludeLocals(locals);
}

bool SBVariablesOptions::GetIncludeStatics() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBVariablesOptions, GetIncludeStatics);

  return m_opaque_up->GetIncludeStatics();
}

void SBVariablesOptions::SetIncludeStatics(bool statics) {
  LLDB_RECORD_METHOD(void, SBVariablesOptions, SetIncludeStatics, (bool),
                     statics);

  m_opaque_up->SetIncludeStatics(statics);
}

bool SBVariablesOptions::GetInScopeOnly() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBVariablesOptions, GetInScopeOnly);

  return m_opaque_up->GetInScopeOnly();
}

void SBVariablesOptions::SetInScopeOnly(bool in_scope_only) {
  LLDB_RECORD_METHOD(void, SBVariablesOptions, SetInScopeOnly, (bool),
                     in_scope_only);

  m_opaque_up->SetInScopeOnly(in_scope_only);
}

bool SBVariablesOptions::GetIncludeRuntimeSupportValues() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBVariablesOptions,
                                   GetIncludeRuntimeSupportValues);

  return m_opaque_up->GetIncludeRuntimeSupportValues();
}

void SBVariablesOptions::SetIncludeRuntimeSupportValues(
    bool runtime_support_values) {
  LLDB_RECORD_METHOD(void, SBVariablesOptions, SetIncludeRuntimeSupportValues,
                     (bool), runtime_support_values);

  m_opaque_up->SetIncludeRuntimeSupportValues(runtime_support_values);
}

lldb::DynamicValueType SBVariablesOptions::GetUseDynamic() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::DynamicValueType, SBVariablesOptions,
                                   GetUseDynamic);

  return m_opaque_up->GetUseDynamic();
}

void SBVariablesOptions::SetUseDynamic(lldb::DynamicValueType dynamic) {
  LLDB_RECORD_METHOD(void, SBVariablesOptions, SetUseDynamic,
                     (lldb::DynamicValueType), dynamic);

  m_opaque_up->SetUseDynamic(dynamic);
}

VariablesOptionsImpl *SBVariablesOptions::operator->() {
  return m_opaque_up.operator->();
}

const VariablesOptionsImpl *SBVariablesOptions::operator->() const {
  return m_opaque_up.operator->();
}

VariablesOptionsImpl *SBVariablesOptions::get() { return m_opaque_up.get(); }

VariablesOptionsImpl &SBVariablesOptions::ref() { return *m_opaque_up; }

const VariablesOptionsImpl &SBVariablesOptions::ref() const {
  return *m_opaque_up;
}

SBVariablesOptions::SBVariablesOptions(VariablesOptionsImpl *lldb_object_ptr)
    : m_opaque_up(std::move(lldb_object_ptr)) {}

void SBVariablesOptions::SetOptions(VariablesOptionsImpl *lldb_object_ptr) {
  m_opaque_up.reset(std::move(lldb_object_ptr));
}
