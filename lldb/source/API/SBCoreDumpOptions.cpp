//===-- SBCoreDumpOptions.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBCoreDumpOptions.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/CoreDumpOptions.h"
#include "lldb/Utility/Instrumentation.h"

#include "Utils.h"

using namespace lldb;

SBCoreDumpOptions::SBCoreDumpOptions() {
  m_opaque_up = std::make_unique<lldb_private::CoreDumpOptions>();
}

SBCoreDumpOptions::SBCoreDumpOptions(const SBCoreDumpOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

const SBCoreDumpOptions &
SBCoreDumpOptions::operator=(const SBCoreDumpOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

void SBCoreDumpOptions::SetCoreDumpPluginName(const char *name) {
  m_opaque_up->SetCoreDumpPluginName(name);
}

void SBCoreDumpOptions::SetCoreDumpStyle(lldb::SaveCoreStyle style) {
  m_opaque_up->SetCoreDumpStyle(style);
}

void SBCoreDumpOptions::SetOutputFile(lldb::SBFileSpec &file_spec) {
  m_opaque_up->SetOutputFile(file_spec.ref());
}

const char *
SBCoreDumpOptions::GetCoreDumpPluginName() const {
  const auto name = m_opaque_up->GetCoreDumpPluginName();
  if (!name)
    return nullptr;
  return lldb_private::ConstString(name.value()).GetCString();
}

SBFileSpec SBCoreDumpOptions::GetOutputFile() const {
  const auto file_spec = m_opaque_up->GetOutputFile();
  if (file_spec)
    return SBFileSpec(file_spec.value());
  return SBFileSpec();
}

lldb::SaveCoreStyle
SBCoreDumpOptions::GetCoreDumpStyle() const {
  return m_opaque_up->GetCoreDumpStyle();
}

lldb_private::CoreDumpOptions &SBCoreDumpOptions::Ref() const {
  return *m_opaque_up.get();
}

void SBCoreDumpOptions::Clear() {
  m_opaque_up->Clear();
}
