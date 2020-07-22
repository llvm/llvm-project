//===-- runtime/format.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "format-implementation.h"

namespace Fortran::runtime::io {

DataEdit DefaultFormatControlCallbacks::GetNextDataEdit(int) {
  Crash("DefaultFormatControlCallbacks::GetNextDataEdit() called for "
        "non-formatted I/O statement");
  return {};
}
bool DefaultFormatControlCallbacks::Emit(
    const char *, std::size_t, std::size_t) {
  Crash("DefaultFormatControlCallbacks::Emit(char) called for non-output I/O "
        "statement");
  return {};
}
bool DefaultFormatControlCallbacks::Emit(const char16_t *, std::size_t) {
  Crash("DefaultFormatControlCallbacks::Emit(char16_t) called for non-output "
        "I/O statement");
  return {};
}
bool DefaultFormatControlCallbacks::Emit(const char32_t *, std::size_t) {
  Crash("DefaultFormatControlCallbacks::Emit(char32_t) called for non-output "
        "I/O statement");
  return {};
}
std::optional<char32_t> DefaultFormatControlCallbacks::GetCurrentChar() {
  Crash("DefaultFormatControlCallbacks::GetCurrentChar() called for non-input "
        "I/O "
        "statement");
  return {};
}
bool DefaultFormatControlCallbacks::AdvanceRecord(int) {
  Crash("DefaultFormatControlCallbacks::AdvanceRecord() called unexpectedly");
  return {};
}
void DefaultFormatControlCallbacks::BackspaceRecord() {
  Crash("DefaultFormatControlCallbacks::BackspaceRecord() called unexpectedly");
}
void DefaultFormatControlCallbacks::HandleAbsolutePosition(std::int64_t) {
  Crash("DefaultFormatControlCallbacks::HandleAbsolutePosition() called for "
        "non-formatted I/O statement");
}
void DefaultFormatControlCallbacks::HandleRelativePosition(std::int64_t) {
  Crash("DefaultFormatControlCallbacks::HandleRelativePosition() called for "
        "non-formatted I/O statement");
}

template class FormatControl<
    InternalFormattedIoStatementState<Direction::Output>>;
template class FormatControl<
    InternalFormattedIoStatementState<Direction::Input>>;
template class FormatControl<
    ExternalFormattedIoStatementState<Direction::Output>>;
template class FormatControl<
    ExternalFormattedIoStatementState<Direction::Input>>;
} // namespace Fortran::runtime::io
