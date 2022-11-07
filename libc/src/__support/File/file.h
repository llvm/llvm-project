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

  // For files which are readable, we should be able to support one ungetc
  // operation even if |buf| is nullptr. So, in the constructor of File, we
  // set |buf| to point to this buffer character.
  char ungetc_buf;

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

  // This is a convenience RAII class to lock and unlock file objects.
  class FileLock {
    File *file;

  public:
    explicit FileLock(File *f) : file(f) { file->lock(); }

    ~FileLock() { file->unlock(); }

    FileLock(const FileLock &) = delete;
    FileLock(FileLock &&) = delete;
  };

protected:
  constexpr bool write_allowed() const {
    return mode & (static_cast<ModeFlags>(OpenMode::WRITE) |
                   static_cast<ModeFlags>(OpenMode::APPEND) |
                   static_cast<ModeFlags>(OpenMode::PLUS));
  }

  constexpr bool read_allowed() const {
    return mode & (static_cast<ModeFlags>(OpenMode::READ) |
                   static_cast<ModeFlags>(OpenMode::PLUS));
  }

public:
  // We want this constructor to be constexpr so that global file objects
  // like stdout do not require invocation of the constructor which can
  // potentially lead to static initialization order fiasco. Consequently,
  // we will assume that the |buffer| and |buffer_size| argument are
  // meaningful - that is, |buffer| is nullptr if and only if |buffer_size|
  // is zero. This way, we will not have to employ the semantics of
  // the set_buffer method and allocate a buffer.
  constexpr File(WriteFunc *wf, ReadFunc *rf, SeekFunc *sf, CloseFunc *cf,
                 FlushFunc *ff, void *buffer, size_t buffer_size,
                 int buffer_mode, bool owned, ModeFlags modeflags)
      : platform_write(wf), platform_read(rf), platform_seek(sf),
        platform_close(cf), platform_flush(ff), mutex(false, false, false),
        ungetc_buf(0), buf(buffer), bufsize(buffer_size), bufmode(buffer_mode),
        own_buf(owned), mode(modeflags), pos(0), prev_op(FileOp::NONE),
        read_limit(0), eof(false), err(false) {
    adjust_buf();
  }

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

    f->adjust_buf();
  }

  // Buffered write of |len| bytes from |data| without the file lock.
  size_t write_unlocked(const void *data, size_t len);

  // Buffered write of |len| bytes from |data| under the file lock.
  size_t write(const void *data, size_t len) {
    FileLock l(this);
    return write_unlocked(data, len);
  }

  // Buffered read of |len| bytes into |data| without the file lock.
  size_t read_unlocked(void *data, size_t len);

  // Buffered read of |len| bytes into |data| under the file lock.
  size_t read(void *data, size_t len) {
    FileLock l(this);
    return read_unlocked(data, len);
  }

  int seek(long offset, int whence);

  // If buffer has data written to it, flush it out. Does nothing if the
  // buffer is currently being used as a read buffer.
  int flush() {
    FileLock lock(this);
    return flush_unlocked();
  }

  int flush_unlocked();

  // Returns EOF on error and keeps the file unchanged.
  int ungetc_unlocked(int c);

  int ungetc(int c) {
    FileLock lock(this);
    return ungetc_unlocked(c);
  }

  // Sets the internal buffer to |buffer| with buffering mode |mode|.
  // |size| is the size of |buffer|. If |size| is non-zero, but |buffer|
  // is nullptr, then a buffer owned by this file will be allocated.
  // Else, |buffer| will not be owned by this file.
  //
  // Will return zero on success, or an error value on failure. Will fail
  // if:
  //   1. |buffer| is not a nullptr but |size| is zero.
  //   2. |buffer_mode| is not one of _IOLBF, IOFBF or _IONBF.
  // In both the above cases, error returned in EINVAL.
  int set_buffer(void *buffer, size_t size, int buffer_mode);

  // Closes the file stream and frees up all resources owned by it.
  int close();

  void lock() { mutex.lock(); }
  void unlock() { mutex.unlock(); }

  bool error_unlocked() const { return err; }

  bool error() {
    FileLock l(this);
    return error_unlocked();
  }

  void clearerr_unlocked() { err = false; }

  void clearerr() {
    FileLock l(this);
    clearerr_unlocked();
  }

  bool iseof_unlocked() { return eof; }

  bool iseof() {
    FileLock l(this);
    return iseof_unlocked();
  }

  // Returns an bit map of flags corresponding to enumerations of
  // OpenMode, ContentType and CreateType.
  static ModeFlags mode_flags(const char *mode);

private:
  size_t write_unlocked_lbf(const uint8_t *data, size_t len);
  size_t write_unlocked_fbf(const uint8_t *data, size_t len);
  size_t write_unlocked_nbf(const uint8_t *data, size_t len);

  constexpr void adjust_buf() {
    if (read_allowed() && (buf == nullptr || bufsize == 0)) {
      // We should allow atleast one ungetc operation.
      // This might give an impression that a buffer will be used even when
      // the user does not want a buffer. But, that will not be the case.
      // For reading, the buffering does not come into play. For writing, let
      // us take up the three different kinds of buffering separately:
      // 1. If user wants _IOFBF but gives a zero buffer, buffering still
      //    happens in the OS layer until the user flushes. So, from the user's
      //    point of view, this single byte buffer does not affect their
      //    experience.
      // 2. If user wants _IOLBF but gives a zero buffer, the reasoning is
      //    very similar to the _IOFBF case.
      // 3. If user wants _IONBF, then the buffer is ignored for writing.
      // So, all of the above cases, having a single ungetc buffer does not
      // affect the behavior experienced by the user.
      buf = &ungetc_buf;
      bufsize = 1;
      own_buf = false; // We shouldn't call free on |buf| when closing the file.
    }
  }
};

// The implementaiton of this function is provided by the platfrom_file
// library.
File *openfile(const char *path, const char *mode);

extern File *stdin;
extern File *stdout;
extern File *stderr;

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_FILE_H
