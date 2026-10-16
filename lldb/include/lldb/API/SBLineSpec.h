//===-- SBLineSpec.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBLINESPEC_H
#define LLDB_API_SBLINESPEC_H

#include "lldb/API/SBDefines.h"
#include "lldb/lldb-defines.h"

namespace lldb {

/// A search specification for a source location: Containing the file spec with
/// line and column information plus flags controlling how the search behaves.
/// Passed to APIs like SBTarget::FindSymbolContexts and
/// SBModule::FindSymbolContexts.
class LLDB_API SBLineSpec {
public:
  SBLineSpec();

  SBLineSpec(const lldb::SBLineSpec &rhs);

  /// Constructor.
  ///
  /// Takes a \a file_spec with a \a line number and a \a column number.
  ///
  /// \param file_spec
  ///     The full or partial path to a file.
  ///
  /// \param line
  ///     The line number in the source file.
  ///
  ///  \param column
  ///     The column number in the line of the source file.
  explicit SBLineSpec(const lldb::SBFileSpec &file_spec,
                      uint32_t line = LLDB_INVALID_LINE_NUMBER,
                      uint32_t column = LLDB_INVALID_COLUMN_NUMBER);

  ~SBLineSpec();

  lldb::SBLineSpec &operator=(const lldb::SBLineSpec &rhs);

  explicit operator bool() const;

  bool IsValid() const;

  lldb::SBFileSpec GetFileSpec() const;

  uint32_t GetLine() const;

  uint32_t GetColumn() const;

  /// Sets whether to look for a match in inlined declaration.
  /// Defaults to true.
  bool GetCheckInlines() const;

  void SetFileSpec(lldb::SBFileSpec file_spec);

  void SetLine(uint32_t line);

  void SetColumn(uint32_t column);

  /// Sets whether to look for a match in inlined declaration.
  void SetCheckInlines(bool value);

private:
  friend class SBModule;
  friend class SBTarget;

  struct Impl;
  std::unique_ptr<Impl> m_opaque_up;
};

} // namespace lldb

#endif // LLDB_API_SBLINESPEC_H
