//===-- runtime/file.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "file.h"
#include "magic-numbers.h"
#include "memory.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#ifdef _WIN32
#define NOMINMAX
#include <io.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace Fortran::runtime::io {

void OpenFile::set_path(OwningPtr<char> &&path, std::size_t bytes) {
  path_ = std::move(path);
  pathLength_ = bytes;
}

static int openfile_mkstemp(IoErrorHandler &handler) {
#ifdef _WIN32
  const unsigned int uUnique{0};
  // GetTempFileNameA needs a directory name < MAX_PATH-14 characters in length.
  // https://docs.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-gettempfilenamea
  char tempDirName[MAX_PATH - 14];
  char tempFileName[MAX_PATH];
  unsigned long nBufferLength{sizeof(tempDirName)};
  nBufferLength = ::GetTempPathA(nBufferLength, tempDirName);
  if (nBufferLength > sizeof(tempDirName) || nBufferLength == 0) {
    return -1;
  }
  if (::GetTempFileNameA(tempDirName, "Fortran", uUnique, tempFileName) == 0) {
    return -1;
  }
  int fd{::_open(tempFileName, _O_CREAT | _O_TEMPORARY, _S_IREAD | _S_IWRITE)};
#else
  char path[]{"/tmp/Fortran-Scratch-XXXXXX"};
  int fd{::mkstemp(path)};
#endif
  if (fd < 0) {
    handler.SignalErrno();
  }
#ifndef _WIN32
  ::unlink(path);
#endif
  return fd;
}

void OpenFile::Open(OpenStatus status, std::optional<Action> action,
    Position position, IoErrorHandler &handler) {
  if (fd_ >= 0 &&
      (status == OpenStatus::Old || status == OpenStatus::Unknown)) {
    return;
  }
  if (fd_ >= 0) {
    if (fd_ <= 2) {
      // don't actually close a standard file descriptor, we might need it
    } else {
      if (::close(fd_) != 0) {
        handler.SignalErrno();
      }
    }
    fd_ = -1;
  }
  if (status == OpenStatus::Scratch) {
    if (path_.get()) {
      handler.SignalError("FILE= must not appear with STATUS='SCRATCH'");
      path_.reset();
    }
    if (!action) {
      action = Action::ReadWrite;
    }
    fd_ = openfile_mkstemp(handler);
  } else {
    if (!path_.get()) {
      handler.SignalError("FILE= is required");
      return;
    }
    int flags{0};
    if (status != OpenStatus::Old) {
      flags |= O_CREAT;
    }
    if (status == OpenStatus::New) {
      flags |= O_EXCL;
    } else if (status == OpenStatus::Replace) {
      flags |= O_TRUNC;
    }
    if (!action) {
      // Try to open read/write, back off to read-only on failure
      fd_ = ::open(path_.get(), flags | O_RDWR, 0600);
      if (fd_ >= 0) {
        action = Action::ReadWrite;
      } else {
        action = Action::Read;
      }
    }
    if (fd_ < 0) {
      switch (*action) {
      case Action::Read:
        flags |= O_RDONLY;
        break;
      case Action::Write:
        flags |= O_WRONLY;
        break;
      case Action::ReadWrite:
        flags |= O_RDWR;
        break;
      }
      fd_ = ::open(path_.get(), flags, 0600);
      if (fd_ < 0) {
        handler.SignalErrno();
      }
    }
  }
  RUNTIME_CHECK(handler, action.has_value());
  pending_.reset();
  if (position == Position::Append && !RawSeekToEnd()) {
    handler.SignalErrno();
  }
  isTerminal_ = ::isatty(fd_) == 1;
  mayRead_ = *action != Action::Write;
  mayWrite_ = *action != Action::Read;
  if (status == OpenStatus::Old || status == OpenStatus::Unknown) {
    knownSize_.reset();
#ifndef _WIN32
    struct stat buf;
    if (::fstat(fd_, &buf) == 0) {
      mayPosition_ = S_ISREG(buf.st_mode);
      knownSize_ = buf.st_size;
    }
#else // TODO: _WIN32
    mayPosition_ = true;
#endif
  } else {
    knownSize_ = 0;
    mayPosition_ = true;
  }
}

void OpenFile::Predefine(int fd) {
  fd_ = fd;
  path_.reset();
  pathLength_ = 0;
  position_ = 0;
  knownSize_.reset();
  nextId_ = 0;
  pending_.reset();
  mayRead_ = fd == 0;
  mayWrite_ = fd != 0;
  mayPosition_ = false;
}

void OpenFile::Close(CloseStatus status, IoErrorHandler &handler) {
  CheckOpen(handler);
  pending_.reset();
  knownSize_.reset();
  switch (status) {
  case CloseStatus::Keep:
    break;
  case CloseStatus::Delete:
    if (path_.get()) {
      ::unlink(path_.get());
    }
    break;
  }
  path_.reset();
  if (fd_ >= 0) {
    if (::close(fd_) != 0) {
      handler.SignalErrno();
    }
    fd_ = -1;
  }
}

std::size_t OpenFile::Read(FileOffset at, char *buffer, std::size_t minBytes,
    std::size_t maxBytes, IoErrorHandler &handler) {
  if (maxBytes == 0) {
    return 0;
  }
  CheckOpen(handler);
  if (!Seek(at, handler)) {
    return 0;
  }
  minBytes = std::min(minBytes, maxBytes);
  std::size_t got{0};
  while (got < minBytes) {
    auto chunk{::read(fd_, buffer + got, maxBytes - got)};
    if (chunk == 0) {
      handler.SignalEnd();
      break;
    } else if (chunk < 0) {
      auto err{errno};
      if (err != EAGAIN && err != EWOULDBLOCK && err != EINTR) {
        handler.SignalError(err);
        break;
      }
    } else {
      position_ += chunk;
      got += chunk;
    }
  }
  return got;
}

std::size_t OpenFile::Write(FileOffset at, const char *buffer,
    std::size_t bytes, IoErrorHandler &handler) {
  if (bytes == 0) {
    return 0;
  }
  CheckOpen(handler);
  if (!Seek(at, handler)) {
    return 0;
  }
  std::size_t put{0};
  while (put < bytes) {
    auto chunk{::write(fd_, buffer + put, bytes - put)};
    if (chunk >= 0) {
      position_ += chunk;
      put += chunk;
    } else {
      auto err{errno};
      if (err != EAGAIN && err != EWOULDBLOCK && err != EINTR) {
        handler.SignalError(err);
        break;
      }
    }
  }
  if (knownSize_ && position_ > *knownSize_) {
    knownSize_ = position_;
  }
  return put;
}

inline static int openfile_ftruncate(int fd, OpenFile::FileOffset at) {
#ifdef _WIN32
  return !::_chsize(fd, at);
#else
  return ::ftruncate(fd, at);
#endif
}

void OpenFile::Truncate(FileOffset at, IoErrorHandler &handler) {
  CheckOpen(handler);
  if (!knownSize_ || *knownSize_ != at) {
    if (openfile_ftruncate(fd_, at) != 0) {
      handler.SignalErrno();
    }
    knownSize_ = at;
  }
}

// The operation is performed immediately; the results are saved
// to be claimed by a later WAIT statement.
// TODO: True asynchronicity
int OpenFile::ReadAsynchronously(
    FileOffset at, char *buffer, std::size_t bytes, IoErrorHandler &handler) {
  CheckOpen(handler);
  int iostat{0};
  for (std::size_t got{0}; got < bytes;) {
#if _XOPEN_SOURCE >= 500 || _POSIX_C_SOURCE >= 200809L
    auto chunk{::pread(fd_, buffer + got, bytes - got, at)};
#else
    auto chunk{Seek(at, handler) ? ::read(fd_, buffer + got, bytes - got) : -1};
#endif
    if (chunk == 0) {
      iostat = FORTRAN_RUNTIME_IOSTAT_END;
      break;
    }
    if (chunk < 0) {
      auto err{errno};
      if (err != EAGAIN && err != EWOULDBLOCK && err != EINTR) {
        iostat = err;
        break;
      }
    } else {
      at += chunk;
      got += chunk;
    }
  }
  return PendingResult(handler, iostat);
}

// TODO: True asynchronicity
int OpenFile::WriteAsynchronously(FileOffset at, const char *buffer,
    std::size_t bytes, IoErrorHandler &handler) {
  CheckOpen(handler);
  int iostat{0};
  for (std::size_t put{0}; put < bytes;) {
#if _XOPEN_SOURCE >= 500 || _POSIX_C_SOURCE >= 200809L
    auto chunk{::pwrite(fd_, buffer + put, bytes - put, at)};
#else
    auto chunk{
        Seek(at, handler) ? ::write(fd_, buffer + put, bytes - put) : -1};
#endif
    if (chunk >= 0) {
      at += chunk;
      put += chunk;
    } else {
      auto err{errno};
      if (err != EAGAIN && err != EWOULDBLOCK && err != EINTR) {
        iostat = err;
        break;
      }
    }
  }
  return PendingResult(handler, iostat);
}

void OpenFile::Wait(int id, IoErrorHandler &handler) {
  std::optional<int> ioStat;
  Pending *prev{nullptr};
  for (Pending *p{pending_.get()}; p; p = (prev = p)->next.get()) {
    if (p->id == id) {
      ioStat = p->ioStat;
      if (prev) {
        prev->next.reset(p->next.release());
      } else {
        pending_.reset(p->next.release());
      }
      break;
    }
  }
  if (ioStat) {
    handler.SignalError(*ioStat);
  }
}

void OpenFile::WaitAll(IoErrorHandler &handler) {
  while (true) {
    int ioStat;
    if (pending_) {
      ioStat = pending_->ioStat;
      pending_.reset(pending_->next.release());
    } else {
      return;
    }
    handler.SignalError(ioStat);
  }
}

void OpenFile::CheckOpen(const Terminator &terminator) {
  RUNTIME_CHECK(terminator, fd_ >= 0);
}

bool OpenFile::Seek(FileOffset at, IoErrorHandler &handler) {
  if (at == position_) {
    return true;
  } else if (RawSeek(at)) {
    position_ = at;
    return true;
  } else {
    handler.SignalErrno();
    return false;
  }
}

bool OpenFile::RawSeek(FileOffset at) {
#ifdef _LARGEFILE64_SOURCE
  return ::lseek64(fd_, at, SEEK_SET) == at;
#else
  return ::lseek(fd_, at, SEEK_SET) == at;
#endif
}

bool OpenFile::RawSeekToEnd() {
#ifdef _LARGEFILE64_SOURCE
  std::int64_t at{::lseek64(fd_, 0, SEEK_END)};
#else
  std::int64_t at{::lseek(fd_, 0, SEEK_END)};
#endif
  if (at >= 0) {
    knownSize_ = at;
    return true;
  } else {
    return false;
  }
}

int OpenFile::PendingResult(const Terminator &terminator, int iostat) {
  int id{nextId_++};
  pending_ = New<Pending>{terminator}(id, iostat, std::move(pending_));
  return id;
}

bool IsATerminal(int fd) { return ::isatty(fd); }

bool IsExtant(const char *path) { return ::access(path, F_OK) == 0; }
bool MayRead(const char *path) { return ::access(path, R_OK) == 0; }
bool MayWrite(const char *path) { return ::access(path, W_OK) == 0; }
bool MayReadAndWrite(const char *path) {
  return ::access(path, R_OK | W_OK) == 0;
}
} // namespace Fortran::runtime::io
