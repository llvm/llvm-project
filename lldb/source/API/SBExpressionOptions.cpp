//===-- SBExpressionOptions.cpp ---------------------------------------------*-
//C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBExpressionOptions.h"
#include "lldb/API/SBStream.h"

#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

SBExpressionOptions::SBExpressionOptions()
    : m_opaque_up(new EvaluateExpressionOptions()) {}

SBExpressionOptions::SBExpressionOptions(const SBExpressionOptions &rhs) {
  m_opaque_up.reset(new EvaluateExpressionOptions());
  *(m_opaque_up.get()) = rhs.ref();
}

const SBExpressionOptions &SBExpressionOptions::
operator=(const SBExpressionOptions &rhs) {
  if (this != &rhs) {
    this->ref() = rhs.ref();
  }
  return *this;
}

SBExpressionOptions::~SBExpressionOptions() {}

bool SBExpressionOptions::GetCoerceResultToId() const {
  return m_opaque_up->DoesCoerceToId();
}

void SBExpressionOptions::SetCoerceResultToId(bool coerce) {
  m_opaque_up->SetCoerceToId(coerce);
}

bool SBExpressionOptions::GetUnwindOnError() const {
  return m_opaque_up->DoesUnwindOnError();
}

void SBExpressionOptions::SetUnwindOnError(bool unwind) {
  m_opaque_up->SetUnwindOnError(unwind);
}

bool SBExpressionOptions::GetIgnoreBreakpoints() const {
  return m_opaque_up->DoesIgnoreBreakpoints();
}

void SBExpressionOptions::SetIgnoreBreakpoints(bool ignore) {
  m_opaque_up->SetIgnoreBreakpoints(ignore);
}

lldb::DynamicValueType SBExpressionOptions::GetFetchDynamicValue() const {
  return m_opaque_up->GetUseDynamic();
}

void SBExpressionOptions::SetFetchDynamicValue(lldb::DynamicValueType dynamic) {
  m_opaque_up->SetUseDynamic(dynamic);
}

uint32_t SBExpressionOptions::GetTimeoutInMicroSeconds() const {
  return m_opaque_up->GetTimeout() ? m_opaque_up->GetTimeout()->count() : 0;
}

void SBExpressionOptions::SetTimeoutInMicroSeconds(uint32_t timeout) {
  m_opaque_up->SetTimeout(timeout == 0 ? Timeout<std::micro>(llvm::None)
                                       : std::chrono::microseconds(timeout));
}

uint32_t SBExpressionOptions::GetOneThreadTimeoutInMicroSeconds() const {
  return m_opaque_up->GetOneThreadTimeout()
             ? m_opaque_up->GetOneThreadTimeout()->count()
             : 0;
}

void SBExpressionOptions::SetOneThreadTimeoutInMicroSeconds(uint32_t timeout) {
  m_opaque_up->SetOneThreadTimeout(timeout == 0
                                       ? Timeout<std::micro>(llvm::None)
                                       : std::chrono::microseconds(timeout));
}

bool SBExpressionOptions::GetTryAllThreads() const {
  return m_opaque_up->GetTryAllThreads();
}

void SBExpressionOptions::SetTryAllThreads(bool run_others) {
  m_opaque_up->SetTryAllThreads(run_others);
}

bool SBExpressionOptions::GetStopOthers() const {
  return m_opaque_up->GetStopOthers();
}

void SBExpressionOptions::SetStopOthers(bool run_others) {
  m_opaque_up->SetStopOthers(run_others);
}

bool SBExpressionOptions::GetTrapExceptions() const {
  return m_opaque_up->GetTrapExceptions();
}

void SBExpressionOptions::SetTrapExceptions(bool trap_exceptions) {
  m_opaque_up->SetTrapExceptions(trap_exceptions);
}

void SBExpressionOptions::SetLanguage(lldb::LanguageType language) {
  m_opaque_up->SetLanguage(language);
}

void SBExpressionOptions::SetCancelCallback(
    lldb::ExpressionCancelCallback callback, void *baton) {
  m_opaque_up->SetCancelCallback(callback, baton);
}

bool SBExpressionOptions::GetGenerateDebugInfo() {
  return m_opaque_up->GetGenerateDebugInfo();
}

void SBExpressionOptions::SetGenerateDebugInfo(bool b) {
  return m_opaque_up->SetGenerateDebugInfo(b);
}

bool SBExpressionOptions::GetSuppressPersistentResult() {
  return m_opaque_up->GetResultIsInternal();
}

void SBExpressionOptions::SetSuppressPersistentResult(bool b) {
  return m_opaque_up->SetResultIsInternal(b);
}

const char *SBExpressionOptions::GetPrefix() const {
  return m_opaque_up->GetPrefix();
}

void SBExpressionOptions::SetPrefix(const char *prefix) {
  return m_opaque_up->SetPrefix(prefix);
}

bool SBExpressionOptions::GetAutoApplyFixIts() {
  return m_opaque_up->GetAutoApplyFixIts();
}

void SBExpressionOptions::SetAutoApplyFixIts(bool b) {
  return m_opaque_up->SetAutoApplyFixIts(b);
}

bool SBExpressionOptions::GetTopLevel() {
  return m_opaque_up->GetExecutionPolicy() == eExecutionPolicyTopLevel;
}

void SBExpressionOptions::SetTopLevel(bool b) {
  m_opaque_up->SetExecutionPolicy(b ? eExecutionPolicyTopLevel
                                    : m_opaque_up->default_execution_policy);
}

bool SBExpressionOptions::GetAllowJIT() {
  return m_opaque_up->GetExecutionPolicy() != eExecutionPolicyNever;
}

void SBExpressionOptions::SetAllowJIT(bool allow) {
  m_opaque_up->SetExecutionPolicy(allow ? m_opaque_up->default_execution_policy
                                        : eExecutionPolicyNever);
}

EvaluateExpressionOptions *SBExpressionOptions::get() const {
  return m_opaque_up.get();
}

EvaluateExpressionOptions &SBExpressionOptions::ref() const {
  return *(m_opaque_up.get());
}
