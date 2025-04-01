//===-- StreamFile.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_STREAMFILE_H
#define LLDB_HOST_STREAMFILE_H

#include "lldb/Host/File.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>

namespace lldb_private {

class StreamFile : public Stream {
public:
  StreamFile(uint32_t flags, uint32_t addr_size, lldb::ByteOrder byte_order);

  StreamFile(int fd, bool transfer_ownership);

  StreamFile(const char *path, File::OpenOptions options,
             uint32_t permissions = lldb::eFilePermissionsFileDefault);

  StreamFile(FILE *fh, bool transfer_ownership);

  StreamFile(std::shared_ptr<File> file) : m_file_sp(file) { assert(file); };

  ~StreamFile() override;

  File &GetFile() { return *m_file_sp; }

  const File &GetFile() const { return *m_file_sp; }

  std::shared_ptr<File> GetFileSP() { return m_file_sp; }

  void Flush() override;

protected:
  // Classes that inherit from StreamFile can see and modify these
  std::shared_ptr<File> m_file_sp; // never NULL
  size_t WriteImpl(const void *s, size_t length) override;

private:
  StreamFile(const StreamFile &) = delete;
  const StreamFile &operator=(const StreamFile &) = delete;
};

class LockableStreamFile;
class LockedStreamFile : public StreamFile {
public:
  ~LockedStreamFile() { Flush(); }

  LockedStreamFile(LockedStreamFile &&other)
      : StreamFile(other.m_file_sp), m_lock(std::move(other.m_lock)) {}

private:
  LockedStreamFile(std::shared_ptr<File> file, std::recursive_mutex &mutex)
      : StreamFile(file), m_lock(mutex) {}

  friend class LockableStreamFile;

  std::unique_lock<std::recursive_mutex> m_lock;
};

class LockableStreamFile {
public:
  using Mutex = std::recursive_mutex;

  LockableStreamFile(std::shared_ptr<StreamFile> stream_file_sp, Mutex &mutex)
      : m_file_sp(stream_file_sp->GetFileSP()), m_mutex(mutex) {}
  LockableStreamFile(StreamFile &stream_file, Mutex &mutex)
      : m_file_sp(stream_file.GetFileSP()), m_mutex(mutex) {}
  LockableStreamFile(FILE *fh, bool transfer_ownership, Mutex &mutex)
      : m_file_sp(std::make_shared<NativeFile>(fh, transfer_ownership)),
        m_mutex(mutex) {}
  LockableStreamFile(std::shared_ptr<File> file_sp, Mutex &mutex)
      : m_file_sp(file_sp), m_mutex(mutex) {}

  LockedStreamFile Lock() { return LockedStreamFile(m_file_sp, m_mutex); }

  /// Unsafe accessors to get the underlying File without a lock. Exists for
  /// legacy reasons.
  /// @{
  File &GetUnlockedFile() { return *m_file_sp; }
  std::shared_ptr<File> GetUnlockedFileSP() { return m_file_sp; }
  /// @}

protected:
  std::shared_ptr<File> m_file_sp;
  Mutex &m_mutex;

private:
  LockableStreamFile(const LockableStreamFile &) = delete;
  const LockableStreamFile &operator=(const LockableStreamFile &) = delete;
};

} // namespace lldb_private

#endif // LLDB_HOST_STREAMFILE_H
