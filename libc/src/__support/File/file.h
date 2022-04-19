//===--- A platform independent file data structure -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_FILE_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_FILE_H

#include "src/__support/threads/mutex.h"

#include <stddef.h>
#include <stdint.h>

namespace __llvm_libc {

// This a generic base class to encapsulate a platform independent file data
// structure. Platform specific specializations should create a subclass as
// suitable for their platform.
class File {
public:
  static constexpr size_t DEFAULT_BUFFER_SIZE = 1024;

  using LockFunc = void(File *);
  using UnlockFunc = void(File *);

  using WriteFunc = size_t(File *, const void *, size_t);
  using ReadFunc = size_t(File *, void *, size_t);
  using SeekFunc = int(File *, long, int);
  using CloseFunc = int(File *);
  using FlushFunc = int(File *);

  using ModeFlags = uint32_t;

  // The three different types of flags below are to be used with '|' operator.
  // Their values correspond to mutually exclusive bits in a 32-bit unsigned
  // integer value. A flag set can include both READ and WRITE if the file
  // is opened in update mode (ie. if the file was opened with a '+' the mode
  // string.)
  enum class OpenMode : ModeFlags {
    READ = 0x1,
    WRITE = 0x2,
    APPEND = 0x4,
    PLUS = 0x8,
  };

  // Denotes a file opened in binary mode (which is specified by including
  // the 'b' character in teh mode string.)
  enum class ContentType : ModeFlags {
    BINARY = 0x10,
  };

  // Denotes a file to be created for writing.
  enum class CreateType : ModeFlags {
    EXCLUSIVE = 0x100,
  };

private:
  enum class FileOp : uint8_t { NONE, READ, WRITE, SEEK };

  // Platfrom specific functions which create new file objects should initialize
  // these fields suitably via the constructor. Typically, they should be simple
  // syscall wrappers for the corresponding functionality.
  WriteFunc *platform_write;
  ReadFunc *platform_read;
  SeekFunc *platform_seek;
  CloseFunc *platform_close;
  FlushFunc *platform_flush;

  Mutex mutex;

  void *buf;      // Pointer to the stream buffer for buffered streams
  size_t bufsize; // Size of the buffer pointed to by |buf|.

  // Buffering mode to used to buffer.
  int bufmode;

  // If own_buf is true, the |buf| is owned by the stream and will be
  // free-ed when close method is called on the stream.
  bool own_buf;

  // The mode in which the file was opened.
  ModeFlags mode;

  // Current read or write pointer.
  size_t pos;

  // Represents the previous operation that was performed.
  FileOp prev_op;

  // When the buffer is used as a read buffer, read_limit is the upper limit
  // of the index to which the buffer can be read until.
  size_t read_limit;

  bool eof;
  bool err;

protected:
  bool write_allowed() const {
    return mode & (static_cast<ModeFlags>(OpenMode::WRITE) |
                   static_cast<ModeFlags>(OpenMode::APPEND) |
                   static_cast<ModeFlags>(OpenMode::PLUS));
  }

  bool read_allowed() const {
    return mode & (static_cast<ModeFlags>(OpenMode::READ) |
                   static_cast<ModeFlags>(OpenMode::PLUS));
  }

public:
  // We want this constructor to be constexpr so that global file objects
  // like stdout do not require invocation of the constructor which can
  // potentially lead to static initialization order fiasco.
  constexpr File(WriteFunc *wf, ReadFunc *rf, SeekFunc *sf, CloseFunc *cf,
                 FlushFunc *ff, void *buffer, size_t buffer_size,
                 int buffer_mode, bool owned, ModeFlags modeflags)
      : platform_write(wf), platform_read(rf), platform_seek(sf),
        platform_close(cf), platform_flush(ff), mutex(false, false, false),
        buf(buffer), bufsize(buffer_size), bufmode(buffer_mode), own_buf(owned),
        mode(modeflags), pos(0), prev_op(FileOp::NONE), read_limit(0),
        eof(false), err(false) {}

  // This function helps initialize the various fields of the File data
  // structure after a allocating memory for it via a call to malloc.
  static void init(File *f, WriteFunc *wf, ReadFunc *rf, SeekFunc *sf,
                   CloseFunc *cf, FlushFunc *ff, void *buffer,
                   size_t buffer_size, int buffer_mode, bool owned,
                   ModeFlags modeflags) {
    Mutex::init(&f->mutex, false, false, false);
    f->platform_write = wf;
    f->platform_read = rf;
    f->platform_seek = sf;
    f->platform_close = cf;
    f->platform_flush = ff;
    f->buf = reinterpret_cast<uint8_t *>(buffer);
    f->bufsize = buffer_size;
    f->bufmode = buffer_mode;
    f->own_buf = owned;
    f->mode = modeflags;

    f->prev_op = FileOp::NONE;
    f->read_limit = f->pos = 0;
    f->eof = f->err = false;
  }

  // Buffered write of |len| bytes from |data| without the file lock.
  size_t write_unlocked(const void *data, size_t len);

  // Buffered write of |len| bytes from |data| under the file lock.
  size_t write(const void *data, size_t len) {
    lock();
    size_t ret = write_unlocked(data, len);
    unlock();
    return ret;
  }

  // Buffered read of |len| bytes into |data| without the file lock.
  size_t read_unlocked(void *data, size_t len);

  // Buffered read of |len| bytes into |data| under the file lock.
  size_t read(void *data, size_t len) {
    lock();
    size_t ret = read_unlocked(data, len);
    unlock();
    return ret;
  }

  int seek(long offset, int whence);

  // If buffer has data written to it, flush it out. Does nothing if the
  // buffer is currently being used as a read buffer.
  int flush();

  // Sets the internal buffer to |buffer| with buffering mode |mode|.
  // |size| is the size of |buffer|. This new |buffer| is owned by the
  // stream only if |owned| is true.
  void set_buffer(void *buffer, size_t size, bool owned);

  // Closes the file stream and frees up all resources owned by it.
  int close();

  void lock() { mutex.lock(); }
  void unlock() { mutex.unlock(); }

  bool error() const { return err; }
  void clearerr() { err = false; }
  bool iseof() const { return eof; }

  // Returns an bit map of flags corresponding to enumerations of
  // OpenMode, ContentType and CreateType.
  static ModeFlags mode_flags(const char *mode);
};

// This is a convenience RAII class to lock and unlock file objects.
class FileLock {
  File *file;

public:
  explicit FileLock(File *f) : file(f) { file->lock(); }

  ~FileLock() { file->unlock(); }

  FileLock(const FileLock &) = delete;
  FileLock(FileLock &&) = delete;
};

// The implementaiton of this function is provided by the platfrom_file
// library.
File *openfile(const char *path, const char *mode);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_FILE_H
