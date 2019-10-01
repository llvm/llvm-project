//===-- SBStream.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBStream.h"

#include "SBReproducerPrivate.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

SBStream::SBStream() : m_opaque_up(new StreamString()), m_is_file(false) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBStream);
}

SBStream::SBStream(SBStream &&rhs)
    : m_opaque_up(std::move(rhs.m_opaque_up)), m_is_file(rhs.m_is_file) {}

SBStream::~SBStream() {}

bool SBStream::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBStream, IsValid);
  return this->operator bool();
}
SBStream::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBStream, operator bool);

  return (m_opaque_up != nullptr);
}

// If this stream is not redirected to a file, it will maintain a local cache
// for the stream data which can be accessed using this accessor.
const char *SBStream::GetData() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBStream, GetData);

  if (m_is_file || m_opaque_up == nullptr)
    return nullptr;

  return static_cast<StreamString *>(m_opaque_up.get())->GetData();
}

// If this stream is not redirected to a file, it will maintain a local cache
// for the stream output whose length can be accessed using this accessor.
size_t SBStream::GetSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBStream, GetSize);

  if (m_is_file || m_opaque_up == nullptr)
    return 0;

  return static_cast<StreamString *>(m_opaque_up.get())->GetSize();
}

void SBStream::Printf(const char *format, ...) {
  if (!format)
    return;
  va_list args;
  va_start(args, format);
  ref().PrintfVarArg(format, args);
  va_end(args);
}

void SBStream::RedirectToFile(const char *path, bool append) {
  LLDB_RECORD_METHOD(void, SBStream, RedirectToFile, (const char *, bool), path,
                     append);

  if (path == nullptr)
    return;

  std::string local_data;
  if (m_opaque_up) {
    // See if we have any locally backed data. If so, copy it so we can then
    // redirect it to the file so we don't lose the data
    if (!m_is_file)
      local_data = static_cast<StreamString *>(m_opaque_up.get())->GetString();
  }
  StreamFile *stream_file = new StreamFile;
  uint32_t open_options = File::eOpenOptionWrite | File::eOpenOptionCanCreate;
  if (append)
    open_options |= File::eOpenOptionAppend;
  else
    open_options |= File::eOpenOptionTruncate;

  FileSystem::Instance().Open(stream_file->GetFile(), FileSpec(path),
                              open_options);
  m_opaque_up.reset(stream_file);

  if (m_opaque_up) {
    m_is_file = true;

    // If we had any data locally in our StreamString, then pass that along to
    // the to new file we are redirecting to.
    if (!local_data.empty())
      m_opaque_up->Write(&local_data[0], local_data.size());
  } else
    m_is_file = false;
}

void SBStream::RedirectToFileHandle(FILE *fh, bool transfer_fh_ownership) {
  LLDB_RECORD_METHOD(void, SBStream, RedirectToFileHandle, (FILE *, bool), fh,
                     transfer_fh_ownership);

  if (fh == nullptr)
    return;

  std::string local_data;
  if (m_opaque_up) {
    // See if we have any locally backed data. If so, copy it so we can then
    // redirect it to the file so we don't lose the data
    if (!m_is_file)
      local_data = static_cast<StreamString *>(m_opaque_up.get())->GetString();
  }
  m_opaque_up.reset(new StreamFile(fh, transfer_fh_ownership));

  if (m_opaque_up) {
    m_is_file = true;

    // If we had any data locally in our StreamString, then pass that along to
    // the to new file we are redirecting to.
    if (!local_data.empty())
      m_opaque_up->Write(&local_data[0], local_data.size());
  } else
    m_is_file = false;
}

void SBStream::RedirectToFileDescriptor(int fd, bool transfer_fh_ownership) {
  LLDB_RECORD_METHOD(void, SBStream, RedirectToFileDescriptor, (int, bool), fd,
                     transfer_fh_ownership);

  std::string local_data;
  if (m_opaque_up) {
    // See if we have any locally backed data. If so, copy it so we can then
    // redirect it to the file so we don't lose the data
    if (!m_is_file)
      local_data = static_cast<StreamString *>(m_opaque_up.get())->GetString();
  }

  m_opaque_up.reset(new StreamFile(::fdopen(fd, "w"), transfer_fh_ownership));
  if (m_opaque_up) {
    m_is_file = true;

    // If we had any data locally in our StreamString, then pass that along to
    // the to new file we are redirecting to.
    if (!local_data.empty())
      m_opaque_up->Write(&local_data[0], local_data.size());
  } else
    m_is_file = false;
}

lldb_private::Stream *SBStream::operator->() { return m_opaque_up.get(); }

lldb_private::Stream *SBStream::get() { return m_opaque_up.get(); }

lldb_private::Stream &SBStream::ref() {
  if (m_opaque_up == nullptr)
    m_opaque_up.reset(new StreamString());
  return *m_opaque_up;
}

void SBStream::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBStream, Clear);

  if (m_opaque_up) {
    // See if we have any locally backed data. If so, copy it so we can then
    // redirect it to the file so we don't lose the data
    if (m_is_file)
      m_opaque_up.reset();
    else
      static_cast<StreamString *>(m_opaque_up.get())->Clear();
  }
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBStream>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBStream, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBStream, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBStream, operator bool, ());
  LLDB_REGISTER_METHOD(const char *, SBStream, GetData, ());
  LLDB_REGISTER_METHOD(size_t, SBStream, GetSize, ());
  LLDB_REGISTER_METHOD(void, SBStream, RedirectToFile, (const char *, bool));
  LLDB_REGISTER_METHOD(void, SBStream, RedirectToFileHandle, (FILE *, bool));
  LLDB_REGISTER_METHOD(void, SBStream, RedirectToFileDescriptor, (int, bool));
  LLDB_REGISTER_METHOD(void, SBStream, Clear, ());
}

}
}
