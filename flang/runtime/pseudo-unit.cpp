//===-- runtime/pseudo-unit.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implemenation of ExternalFileUnit and PseudoOpenFile for
// RT_USE_PSEUDO_FILE_UNIT=1.
//
//===----------------------------------------------------------------------===//

#include "io-error.h"
#include "tools.h"
#include "unit.h"

// NOTE: the header files above may define OpenMP declare target
// variables, so they have to be included unconditionally
// so that the offload entries are consistent between host and device.
#if defined(RT_USE_PSEUDO_FILE_UNIT)
#include <cstdio>

namespace Fortran::runtime::io {

void FlushOutputOnCrash(const Terminator &) {}

ExternalFileUnit *ExternalFileUnit::LookUp(int) {
  Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

ExternalFileUnit *ExternalFileUnit::LookUpOrCreate(
    int, const Terminator &, bool &) {
  Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

ExternalFileUnit *ExternalFileUnit::LookUpOrCreateAnonymous(int unit,
    Direction direction, Fortran::common::optional<bool>,
    IoErrorHandler &handler) {
  if (direction != Direction::Output) {
    handler.Crash("ExternalFileUnit only supports output IO");
  }
  return New<ExternalFileUnit>{handler}(unit).release();
}

ExternalFileUnit *ExternalFileUnit::LookUp(const char *, std::size_t) {
  Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

ExternalFileUnit &ExternalFileUnit::CreateNew(int, const Terminator &) {
  Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

ExternalFileUnit *ExternalFileUnit::LookUpForClose(int) {
  Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

ExternalFileUnit &ExternalFileUnit::NewUnit(const Terminator &, bool) {
  Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

bool ExternalFileUnit::OpenUnit(Fortran::common::optional<OpenStatus> status,
    Fortran::common::optional<Action>, Position, OwningPtr<char> &&,
    std::size_t, Convert, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

void ExternalFileUnit::OpenAnonymousUnit(Fortran::common::optional<OpenStatus>,
    Fortran::common::optional<Action>, Position, Convert convert,
    IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

void ExternalFileUnit::CloseUnit(CloseStatus, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

void ExternalFileUnit::DestroyClosed() {
  Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

Iostat ExternalFileUnit::SetDirection(Direction direction) {
  if (direction != Direction::Output) {
    return IostatReadFromWriteOnly;
  }
  direction_ = direction;
  return IostatOk;
}

void ExternalFileUnit::CloseAll(IoErrorHandler &) {}

void ExternalFileUnit::FlushAll(IoErrorHandler &) {}

int ExternalFileUnit::GetAsynchronousId(IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

bool ExternalFileUnit::Wait(int) {
  Terminator{__FILE__, __LINE__}.Crash("unsupported");
}

void PseudoOpenFile::set_mayAsynchronous(bool yes) {
  if (yes) {
    Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
  }
}

Fortran::common::optional<PseudoOpenFile::FileOffset>
PseudoOpenFile::knownSize() const {
  Terminator{__FILE__, __LINE__}.Crash("unsupported");
}

void PseudoOpenFile::Open(OpenStatus, Fortran::common::optional<Action>,
    Position, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

void PseudoOpenFile::Close(CloseStatus, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

std::size_t PseudoOpenFile::Read(
    FileOffset, char *, std::size_t, std::size_t, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

std::size_t PseudoOpenFile::Write(FileOffset at, const char *buffer,
    std::size_t bytes, IoErrorHandler &handler) {
  if (at) {
    handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
  }
  // TODO: use persistent string buffer that can be reallocated
  // as needed, and only freed at destruction of *this.
  auto string{SizedNew<char>{handler}(bytes + 1)};
  std::memcpy(string.get(), buffer, bytes);
  string.get()[bytes] = '\0';
  std::printf("%s", string.get());
  return bytes;
}

void PseudoOpenFile::Truncate(FileOffset, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

int PseudoOpenFile::ReadAsynchronously(
    FileOffset, char *, std::size_t, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

int PseudoOpenFile::WriteAsynchronously(
    FileOffset, const char *, std::size_t, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

void PseudoOpenFile::Wait(int, IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

void PseudoOpenFile::WaitAll(IoErrorHandler &handler) {
  handler.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

Position PseudoOpenFile::InquirePosition() const {
  Terminator{__FILE__, __LINE__}.Crash("%s: unsupported", RT_PRETTY_FUNCTION);
}

} // namespace Fortran::runtime::io

#endif // defined(RT_USE_PSEUDO_FILE_UNIT)
