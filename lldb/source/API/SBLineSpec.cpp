//===-- SBLineSpec.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBLineSpec.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Instrumentation.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

// Mirrors the internal `lldb_private::SourceLocationSpec`.
struct SBLineSpec::Impl {
  FileSpec file_spec;
  uint32_t line = LLDB_INVALID_LINE_NUMBER;
  uint32_t column = LLDB_INVALID_COLUMN_NUMBER;
  bool check_inlines = true;
};

SBLineSpec::SBLineSpec() : m_opaque_up(std::make_unique<Impl>()) {
  LLDB_INSTRUMENT_VA(this);
}

SBLineSpec::SBLineSpec(const SBLineSpec &rhs)
    : m_opaque_up(std::make_unique<Impl>(*rhs.m_opaque_up)) {
  LLDB_INSTRUMENT_VA(this, rhs);
}

SBLineSpec::SBLineSpec(const SBFileSpec &file_spec, uint32_t line,
                       uint32_t column)
    : m_opaque_up(std::make_unique<Impl>()) {
  LLDB_INSTRUMENT_VA(this, file_spec, line, column);

  SetFileSpec(file_spec);
  SetLine(line);
  SetColumn(column);
}

SBLineSpec::~SBLineSpec() = default;

SBLineSpec &SBLineSpec::operator=(const SBLineSpec &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    *m_opaque_up = *rhs.m_opaque_up;
  return *this;
}

SBLineSpec::operator bool() const {
  LLDB_INSTRUMENT_VA(this);

  return static_cast<bool>(m_opaque_up->file_spec) &&
         m_opaque_up->line != LLDB_INVALID_LINE_NUMBER &&
         m_opaque_up->line != 0;
}

bool SBLineSpec::IsValid() const {
  LLDB_INSTRUMENT_VA(this);
  return this->operator bool();
}

SBFileSpec SBLineSpec::GetFileSpec() const {
  LLDB_INSTRUMENT_VA(this);

  SBFileSpec sb_file_spec;
  sb_file_spec.SetFileSpec(m_opaque_up->file_spec);
  return sb_file_spec;
}

uint32_t SBLineSpec::GetLine() const {
  LLDB_INSTRUMENT_VA(this);
  return m_opaque_up->line;
}

uint32_t SBLineSpec::GetColumn() const {
  LLDB_INSTRUMENT_VA(this);
  return m_opaque_up->column;
}

bool SBLineSpec::GetCheckInlines() const {
  LLDB_INSTRUMENT_VA(this);
  return m_opaque_up->check_inlines;
}

void SBLineSpec::SetFileSpec(SBFileSpec file_spec) {
  LLDB_INSTRUMENT_VA(this, file_spec);

  if (file_spec.IsValid())
    m_opaque_up->file_spec = file_spec.ref();
  else
    m_opaque_up->file_spec = FileSpec();
}

void SBLineSpec::SetLine(uint32_t line) {
  LLDB_INSTRUMENT_VA(this, line);
  m_opaque_up->line = line;
}

void SBLineSpec::SetColumn(uint32_t column) {
  LLDB_INSTRUMENT_VA(this, column);
  m_opaque_up->column = column;
}

void SBLineSpec::SetCheckInlines(bool value) {
  LLDB_INSTRUMENT_VA(this, value);
  m_opaque_up->check_inlines = value;
}