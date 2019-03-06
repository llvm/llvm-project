//===-- SBExpressionOptions.cpp ---------------------------------------------*-
//C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBExpressionOptions.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

SBExpressionOptions::SBExpressionOptions()
    : m_opaque_up(new EvaluateExpressionOptions()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBExpressionOptions);
}

SBExpressionOptions::SBExpressionOptions(const SBExpressionOptions &rhs)
    : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBExpressionOptions,
                          (const lldb::SBExpressionOptions &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

const SBExpressionOptions &SBExpressionOptions::
operator=(const SBExpressionOptions &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBExpressionOptions &,
      SBExpressionOptions, operator=,(const lldb::SBExpressionOptions &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

SBExpressionOptions::~SBExpressionOptions() {}

bool SBExpressionOptions::GetCoerceResultToId() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBExpressionOptions,
                                   GetCoerceResultToId);

  return m_opaque_up->DoesCoerceToId();
}

void SBExpressionOptions::SetCoerceResultToId(bool coerce) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetCoerceResultToId, (bool),
                     coerce);

  m_opaque_up->SetCoerceToId(coerce);
}

bool SBExpressionOptions::GetUnwindOnError() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBExpressionOptions, GetUnwindOnError);

  return m_opaque_up->DoesUnwindOnError();
}

void SBExpressionOptions::SetUnwindOnError(bool unwind) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetUnwindOnError, (bool),
                     unwind);

  m_opaque_up->SetUnwindOnError(unwind);
}

bool SBExpressionOptions::GetIgnoreBreakpoints() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBExpressionOptions,
                                   GetIgnoreBreakpoints);

  return m_opaque_up->DoesIgnoreBreakpoints();
}

void SBExpressionOptions::SetIgnoreBreakpoints(bool ignore) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetIgnoreBreakpoints, (bool),
                     ignore);

  m_opaque_up->SetIgnoreBreakpoints(ignore);
}

lldb::DynamicValueType SBExpressionOptions::GetFetchDynamicValue() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::DynamicValueType, SBExpressionOptions,
                                   GetFetchDynamicValue);

  return m_opaque_up->GetUseDynamic();
}

void SBExpressionOptions::SetFetchDynamicValue(lldb::DynamicValueType dynamic) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetFetchDynamicValue,
                     (lldb::DynamicValueType), dynamic);

  m_opaque_up->SetUseDynamic(dynamic);
}

uint32_t SBExpressionOptions::GetTimeoutInMicroSeconds() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBExpressionOptions,
                                   GetTimeoutInMicroSeconds);

  return m_opaque_up->GetTimeout() ? m_opaque_up->GetTimeout()->count() : 0;
}

void SBExpressionOptions::SetTimeoutInMicroSeconds(uint32_t timeout) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetTimeoutInMicroSeconds,
                     (uint32_t), timeout);

  m_opaque_up->SetTimeout(timeout == 0 ? Timeout<std::micro>(llvm::None)
                                       : std::chrono::microseconds(timeout));
}

uint32_t SBExpressionOptions::GetOneThreadTimeoutInMicroSeconds() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBExpressionOptions,
                                   GetOneThreadTimeoutInMicroSeconds);

  return m_opaque_up->GetOneThreadTimeout()
             ? m_opaque_up->GetOneThreadTimeout()->count()
             : 0;
}

void SBExpressionOptions::SetOneThreadTimeoutInMicroSeconds(uint32_t timeout) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions,
                     SetOneThreadTimeoutInMicroSeconds, (uint32_t), timeout);

  m_opaque_up->SetOneThreadTimeout(timeout == 0
                                       ? Timeout<std::micro>(llvm::None)
                                       : std::chrono::microseconds(timeout));
}

bool SBExpressionOptions::GetTryAllThreads() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBExpressionOptions, GetTryAllThreads);

  return m_opaque_up->GetTryAllThreads();
}

void SBExpressionOptions::SetTryAllThreads(bool run_others) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetTryAllThreads, (bool),
                     run_others);

  m_opaque_up->SetTryAllThreads(run_others);
}

bool SBExpressionOptions::GetStopOthers() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBExpressionOptions, GetStopOthers);

  return m_opaque_up->GetStopOthers();
}

void SBExpressionOptions::SetStopOthers(bool run_others) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetStopOthers, (bool),
                     run_others);

  m_opaque_up->SetStopOthers(run_others);
}

bool SBExpressionOptions::GetTrapExceptions() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBExpressionOptions,
                                   GetTrapExceptions);

  return m_opaque_up->GetTrapExceptions();
}

void SBExpressionOptions::SetTrapExceptions(bool trap_exceptions) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetTrapExceptions, (bool),
                     trap_exceptions);

  m_opaque_up->SetTrapExceptions(trap_exceptions);
}

void SBExpressionOptions::SetLanguage(lldb::LanguageType language) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetLanguage,
                     (lldb::LanguageType), language);

  m_opaque_up->SetLanguage(language);
}

void SBExpressionOptions::SetCancelCallback(
    lldb::ExpressionCancelCallback callback, void *baton) {
  m_opaque_up->SetCancelCallback(callback, baton);
}

bool SBExpressionOptions::GetPlaygroundTransformEnabled() const {
  return m_opaque_up->GetPlaygroundTransformEnabled();
}

void SBExpressionOptions::SetPlaygroundTransformEnabled(
    bool enable_playground_transform) {
  m_opaque_up->SetPlaygroundTransformEnabled(enable_playground_transform);
}

bool SBExpressionOptions::GetREPLMode() const {
  return m_opaque_up->GetREPLEnabled();
}

void SBExpressionOptions::SetREPLMode(bool enable_repl_mode) {
  m_opaque_up->SetREPLEnabled(enable_repl_mode);
  // Don't trap exceptions in the REPL.  That would make it impossible
  // to call any swift code whose underlying framework used exceptions.
  if (enable_repl_mode == true)
    SetTrapExceptions(false);
}

bool SBExpressionOptions::GetGenerateDebugInfo() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBExpressionOptions, GetGenerateDebugInfo);

  return m_opaque_up->GetGenerateDebugInfo();
}

void SBExpressionOptions::SetGenerateDebugInfo(bool b) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetGenerateDebugInfo, (bool),
                     b);

  return m_opaque_up->SetGenerateDebugInfo(b);
}

bool SBExpressionOptions::GetSuppressPersistentResult() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBExpressionOptions,
                             GetSuppressPersistentResult);

  return m_opaque_up->GetResultIsInternal();
}

void SBExpressionOptions::SetSuppressPersistentResult(bool b) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetSuppressPersistentResult,
                     (bool), b);

  return m_opaque_up->SetResultIsInternal(b);
}

const char *SBExpressionOptions::GetPrefix() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBExpressionOptions,
                                   GetPrefix);

  return m_opaque_up->GetPrefix();
}

void SBExpressionOptions::SetPrefix(const char *prefix) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetPrefix, (const char *),
                     prefix);

  return m_opaque_up->SetPrefix(prefix);
}

bool SBExpressionOptions::GetAutoApplyFixIts() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBExpressionOptions, GetAutoApplyFixIts);

  return m_opaque_up->GetAutoApplyFixIts();
}

void SBExpressionOptions::SetAutoApplyFixIts(bool b) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetAutoApplyFixIts, (bool), b);

  return m_opaque_up->SetAutoApplyFixIts(b);
}

bool SBExpressionOptions::GetTopLevel() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBExpressionOptions, GetTopLevel);

  return m_opaque_up->GetExecutionPolicy() == eExecutionPolicyTopLevel;
}

void SBExpressionOptions::SetTopLevel(bool b) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetTopLevel, (bool), b);

  m_opaque_up->SetExecutionPolicy(b ? eExecutionPolicyTopLevel
                                    : m_opaque_up->default_execution_policy);
}

bool SBExpressionOptions::GetAllowJIT() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBExpressionOptions, GetAllowJIT);

  return m_opaque_up->GetExecutionPolicy() != eExecutionPolicyNever;
}

void SBExpressionOptions::SetAllowJIT(bool allow) {
  LLDB_RECORD_METHOD(void, SBExpressionOptions, SetAllowJIT, (bool), allow);

  m_opaque_up->SetExecutionPolicy(allow ? m_opaque_up->default_execution_policy
                                        : eExecutionPolicyNever);
}

EvaluateExpressionOptions *SBExpressionOptions::get() const {
  return m_opaque_up.get();
}

EvaluateExpressionOptions &SBExpressionOptions::ref() const {
  return *(m_opaque_up.get());
}
