//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Platform independent file data structure.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_FILE_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_FILE_H

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/off_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/wchar_t.h"
#include "hdr/types/wint_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/File/file_io_result.h"
#include "src/__support/File/file_mode.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/wchar/mbstate.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

namespace internal {

template <typename FileT>
LIBC_INLINE FileIOResult default_write(FileT *, const void *, size_t size) {
  return size;
}

template <typename FileT>
LIBC_INLINE FileIOResult default_read(FileT *, void *, size_t) {
  return 0;
}

template <typename FileT>
LIBC_INLINE ErrorOr<off_t> default_seek(FileT *, off_t, int) {
  return Error(ESPIPE);
}

template <typename FileT> LIBC_INLINE int default_close(FileT *) { return 0; }

} // namespace internal

template <typename FileT> struct FileOps {
  using WriteFunc = FileIOResult(FileT *, const void *, size_t);
  using ReadFunc = FileIOResult(FileT *, void *, size_t);
  // The SeekFunc is expected to return the current offset of the external
  // file position indicator.
  using SeekFunc = ErrorOr<off_t>(FileT *, off_t, int);
  using CloseFunc = int(FileT *);
  // Optional hook invoked when reopening a CustomFile onto an OS file
  // descriptor to tear down subclass-specific state without deleting |FileT|.
  using DetachFunc = void(FileT *);

  WriteFunc *write;
  ReadFunc *read;
  SeekFunc *seek;
  CloseFunc *close;
  DetachFunc *detach;

  LIBC_INLINE constexpr FileOps(WriteFunc *w, ReadFunc *r, SeekFunc *s,
                                CloseFunc *c, DetachFunc *d = nullptr)
      : write(w != nullptr ? w : &internal::default_write<FileT>),
        read(r != nullptr ? r : &internal::default_read<FileT>),
        seek(s != nullptr ? s : &internal::default_seek<FileT>),
        close(c != nullptr ? c : &internal::default_close<FileT>), detach(d) {}
};

struct CookieOps {
  using ReadFunc = ssize_t(void *, char *, size_t);
  using WriteFunc = ssize_t(void *, const char *, size_t);
  using SeekFunc = int(void *, off_t *, int);
  using CloseFunc = int(void *);

  ReadFunc *read = nullptr;
  WriteFunc *write = nullptr;
  SeekFunc *seek = nullptr;
  CloseFunc *close = nullptr;
};

// This is a generic base class template to encapsulate a platform independent
// file data structure. Platform specific specializations provide a static
// Platform policy and optionally subclass FileImpl as suitable for their
// platform.
template <typename Platform> class FileImpl {
public:
  using PlatformType = Platform;
  using FileBase = FileImpl;
  using Ops = FileOps<FileImpl>;

  enum class DispatchKind : uint8_t {
    STATIC = 0,
    CUSTOM = 1,
    COOKIE = 2,
  };

  static void add_file(FileImpl *f);
  static void remove_file(FileImpl *f);
  static FileImpl *get_first_file();
  static void lock_list();
  static void unlock_list();

  static FileImpl *list_all;
  static Mutex list_lock;

  LIBC_INLINE FileImpl *get_next() const { return next; }

  static constexpr size_t DEFAULT_BUFFER_SIZE = 1024;

  enum class Orientation : uint8_t { UNORIENTED, BYTE, WIDE };

  using LockFunc = void(FileImpl *);
  using UnlockFunc = void(FileImpl *);

  using WriteFunc = typename Ops::WriteFunc;
  using ReadFunc = typename Ops::ReadFunc;
  using SeekFunc = typename Ops::SeekFunc;
  using CloseFunc = typename Ops::CloseFunc;

  // This is a convenience RAII class to lock and unlock file objects.
  class FileLock {
    FileImpl *file;

  public:
    LIBC_INLINE explicit FileLock(FileImpl *f) : file(f) { file->lock(); }

    LIBC_INLINE ~FileLock() { file->unlock(); }

    LIBC_INLINE FileLock(const FileLock &) = delete;
    LIBC_INLINE FileLock(FileLock &&) = delete;
  };

private:
  enum class FileOp : uint8_t { NONE, READ, WRITE, SEEK };

  uint8_t *buf;   // Pointer to the stream buffer for buffered streams
  size_t bufsize; // Size of the buffer pointed to by |buf|.

  // Current read or write pointer.
  size_t pos;

  // When the buffer is used as a read buffer, read_limit is the upper limit
  // of the index to which the buffer can be read until.
  size_t read_limit;

  FileImpl *prev;
  FileImpl *next;

  internal::mbstate mbstate;

  // Used to handle the File's mode
  FileMode mode;

  // For files which are readable, we should be able to support one ungetc
  // operation even if |buf| is nullptr. So, in the constructor of FileImpl, we
  // set |buf| to point to this buffer character. It needs to be at least 4
  // bytes so we can store a widechar.
  uint8_t ungetc_buf[4];

  Mutex mutex;

  // Buffering mode to used to buffer.
  int8_t bufmode;

  // Represents the previous operation that was performed.
  FileOp prev_op : 2;
  Orientation orientation : 2;

  // If own_buf is true, the |buf| is owned by the stream and will be
  // free-ed when close method is called on the stream.
  bool own_buf : 1;
  bool eof : 1;
  bool err : 1;
  DispatchKind kind : 2;

protected:
  LIBC_INLINE constexpr bool write_allowed() const {
    return mode.write_allowed();
  }

  LIBC_INLINE constexpr bool read_allowed() const {
    return mode.read_allowed();
  }

  LIBC_INLINE void reset_stream_state_unlocked(FileMode new_mode) {
    mode = new_mode;
    pos = 0;
    prev_op = FileOp::NONE;
    read_limit = 0;
    eof = false;
    err = false;
    orientation = Orientation::UNORIENTED;
    mbstate = internal::mbstate();
    adjust_buf();
  }

  LIBC_INLINE constexpr FileImpl(DispatchKind dispatch_kind, uint8_t *buffer,
                                 size_t buffer_size, int buffer_mode,
                                 bool owned, FileMode mode)
      : buf(buffer), bufsize(buffer_size), pos(0), read_limit(0), prev(nullptr),
        next(nullptr), mbstate(), mode(mode), ungetc_buf{},
        mutex(/*timed=*/false, /*recursive=*/true,
              /*robust=*/false, /*pshared=*/false),
        bufmode(static_cast<int8_t>(buffer_mode)), prev_op(FileOp::NONE),
        orientation(Orientation::UNORIENTED), own_buf(owned), eof(false),
        err(false), kind(dispatch_kind) {
    adjust_buf();
  }

  // We want this constructor to be constexpr so that global file objects
  // like stdout do not require invocation of the constructor which can
  // potentially lead to static initialization order fiasco. Consequently,
  // we will assume that the |buffer| and |buffer_size| argument are
  // meaningful - that is, |buffer| is nullptr if and only if |buffer_size|
  // is zero. This way, we will not have to employ the semantics of
  // the set_buffer method and allocate a buffer.
  LIBC_INLINE constexpr FileImpl(uint8_t *buffer, size_t buffer_size,
                                 int buffer_mode, bool owned, FileMode mode)
      : FileImpl(DispatchKind::STATIC, buffer, buffer_size, buffer_mode, owned,
                 mode) {}

public:
  LIBC_INLINE constexpr DispatchKind get_dispatch_kind() const { return kind; }
  LIBC_INLINE constexpr bool is_dynamic() const {
    return kind != DispatchKind::STATIC;
  }

  // Buffered write of |len| bytes from |data| without the file lock.
  FileIOResult write_unlocked(const void *data, size_t len);

  // Buffered write of |len| bytes from |data| under the file lock.
  LIBC_INLINE FileIOResult write(const void *data, size_t len) {
    FileLock l(this);
    return write_unlocked(data, len);
  }

  // Buffered read of |len| bytes into |data| without the file lock.
  FileIOResult read_unlocked(void *data, size_t len);

  // Buffered read of |len| bytes into |data| under the file lock.
  LIBC_INLINE FileIOResult read(void *data, size_t len) {
    FileLock l(this);
    return read_unlocked(data, len);
  }

  ErrorOr<int> seek_unlocked(off_t offset, int whence);

  LIBC_INLINE ErrorOr<int> seek(off_t offset, int whence) {
    FileLock lock(this);
    return seek_unlocked(offset, whence);
  }

  ErrorOr<off_t> tell_unlocked();

  LIBC_INLINE ErrorOr<off_t> tell() {
    FileLock lock(this);
    return tell_unlocked();
  }

  // An internal representation of the stream's position and parse state.
  struct Position {
    off_t offset;
    internal::mbstate state;
  };

  ErrorOr<Position> get_pos_unlocked();

  LIBC_INLINE ErrorOr<Position> get_pos() {
    FileLock lock(this);
    return get_pos_unlocked();
  }

  ErrorOr<int> set_pos_unlocked(const Position &fpos);

  LIBC_INLINE ErrorOr<int> set_pos(const Position &fpos) {
    FileLock lock(this);
    return set_pos_unlocked(fpos);
  }

  // If buffer has data written to it, flush it out. Does nothing if the
  // buffer is currently being used as a read buffer.
  LIBC_INLINE int flush() {
    FileLock lock(this);
    return flush_unlocked();
  }

  int flush_unlocked();

  // Returns EOF on error and keeps the file unchanged.
  int ungetc_unlocked(int c);

  LIBC_INLINE int ungetc(int c) {
    FileLock lock(this);
    return ungetc_unlocked(c);
  }

  FileIOResult write_unlocked(const wchar_t *ws, size_t len);

  LIBC_INLINE FileIOResult write(const wchar_t *ws, size_t len) {
    FileLock l(this);
    return write_unlocked(ws, len);
  }

  FileIOResult read_unlocked(wchar_t *ws, size_t len);

  LIBC_INLINE FileIOResult read(wchar_t *ws, size_t len) {
    FileLock l(this);
    return read_unlocked(ws, len);
  }

  ErrorOr<wint_t> ungetwc_unlocked(wint_t wc);

  LIBC_INLINE ErrorOr<wint_t> ungetwc(wint_t wc) {
    FileLock lock(this);
    return ungetwc_unlocked(wc);
  }

  // Closes the file stream and frees up all resources owned by it.
  int close();

  // Sets the internal buffer to |buffer| with buffering mode |mode|.
  // |size| is the size of |buffer|. If |size| is non-zero, but |buffer|
  // is nullptr, then a buffer owned by this file will be allocated.
  // Else, |buffer| will not be owned by this file.
  //
  // Will return zero on success, or an error value on failure. Will fail
  // if:
  //   1. |buffer| is not a nullptr but |size| is zero.
  //   2. |buffer_mode| is not one of _IOLBF, IOFBF or _IONBF.
  //   3. If an allocation was required but the allocation failed.
  // For cases 1 and 2, the error returned in EINVAL. For case 3, error returned
  // is ENOMEM.
  int set_buffer(void *buffer, size_t size, int buffer_mode);

  LIBC_INLINE void lock() { mutex.lock(); }
  LIBC_INLINE void unlock() { mutex.unlock(); }

  LIBC_INLINE bool error_unlocked() const { return err; }

  LIBC_INLINE void set_error_unlocked() { err = true; }

  LIBC_INLINE bool error() {
    FileLock l(this);
    return error_unlocked();
  }

  // TODO: https://github.com/llvm/llvm-project/issues/172302
  // MacOS defines clearerr_unlocked as a macro. While pre-processing, the
  // identifier below is substituted for the definition in the SDK, which leads
  // to compile time errors due to ill-formed statements. This is a workaround
  // for the pre-processor.
#pragma push_macro("clearerr_unlocked")
#undef clearerr_unlocked
  LIBC_INLINE void clearerr_unlocked() { err = false; }
#pragma pop_macro("clearerr_unlocked")

  LIBC_INLINE void clearerr() {
    FileLock l(this);
    clearerr_unlocked();
  }

  LIBC_INLINE bool iseof_unlocked() { return eof; }

  LIBC_INLINE bool iseof() {
    FileLock l(this);
    return iseof_unlocked();
  }

  LIBC_INLINE Orientation get_orientation_unlocked() const {
    return orientation;
  }

  LIBC_INLINE Orientation get_orientation() {
    FileLock l(this);
    return get_orientation_unlocked();
  }

  LIBC_INLINE Orientation try_set_orientation_unlocked(Orientation o) {
    if (orientation == Orientation::UNORIENTED)
      orientation = o;
    return orientation;
  }

  LIBC_INLINE Orientation try_set_orientation(Orientation o) {
    FileLock l(this);
    return try_set_orientation_unlocked(o);
  }

private:
  LIBC_NOINLINE static FileIOResult
  platform_write_dynamic(FileImpl *f, const void *data, size_t len);
  LIBC_NOINLINE static FileIOResult
  platform_read_dynamic(FileImpl *f, void *data, size_t len);
  LIBC_NOINLINE static ErrorOr<off_t>
  platform_seek_dynamic(FileImpl *f, off_t offset, int whence);
  LIBC_NOINLINE static int platform_close_dynamic(FileImpl *f);

  LIBC_INLINE static FileIOResult platform_write(FileImpl *f, const void *data,
                                                 size_t len) {
    if (LIBC_LIKELY(f->kind == DispatchKind::STATIC))
      return Platform::write(f, data, len);
    return platform_write_dynamic(f, data, len);
  }

  LIBC_INLINE static FileIOResult platform_read(FileImpl *f, void *data,
                                                size_t len) {
    if (LIBC_LIKELY(f->kind == DispatchKind::STATIC))
      return Platform::read(f, data, len);
    return platform_read_dynamic(f, data, len);
  }

  LIBC_INLINE static ErrorOr<off_t> platform_seek(FileImpl *f, off_t offset,
                                                  int whence) {
    if (LIBC_LIKELY(f->kind == DispatchKind::STATIC))
      return Platform::seek(f, offset, whence);
    return platform_seek_dynamic(f, offset, whence);
  }

  LIBC_INLINE static int platform_close(FileImpl *f) {
    if (LIBC_LIKELY(f->kind == DispatchKind::STATIC))
      return Platform::close(f);
    return platform_close_dynamic(f);
  }

  FileIOResult write_unlocked_impl(const void *data, size_t len);
  FileIOResult read_unlocked_impl(void *data, size_t len);

  FileIOResult write_unlocked_lbf(const uint8_t *data, size_t len);
  FileIOResult write_unlocked_fbf(const uint8_t *data, size_t len);
  FileIOResult write_unlocked_nbf(const uint8_t *data, size_t len);

  FileIOResult read_unlocked_fbf(uint8_t *data, size_t len);
  FileIOResult read_unlocked_nbf(uint8_t *data, size_t len);
  FileIOResult read_from_platform(uint8_t *dst, size_t min_bytes,
                                  size_t max_bytes);
  size_t copy_data_from_buf(uint8_t *data, size_t len);

  LIBC_INLINE constexpr void adjust_buf() {
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
      buf = ungetc_buf;
      bufsize = sizeof(ungetc_buf);
      own_buf = false; // We shouldn't call free on |buf| when closing the file.
    }
  }
};

struct FilePlatform;

template <typename BaseFile> class CustomFileImpl : public BaseFile {
  static_assert(
      cpp::is_same_v<BaseFile, typename BaseFile::PlatformType::StreamBase>,
      "BaseFile must be Platform::StreamBase");

public:
  using Ops = typename BaseFile::Ops;

private:
  Ops ops;
  friend typename BaseFile::FileBase;

  LIBC_INLINE FileIOResult custom_write(const void *data, size_t len) {
    return ops.write(this, data, len);
  }
  LIBC_INLINE FileIOResult custom_read(void *data, size_t len) {
    return ops.read(this, data, len);
  }
  LIBC_INLINE ErrorOr<off_t> custom_seek(off_t offset, int whence) {
    return ops.seek(this, offset, whence);
  }
  LIBC_INLINE int custom_close() { return ops.close(this); }

public:
  template <typename... Args>
  LIBC_INLINE constexpr CustomFileImpl(const Ops &fops, uint8_t *buffer,
                                       size_t buffer_size, int buffer_mode,
                                       bool owned, FileMode mode,
                                       Args... base_args)
      : BaseFile(BaseFile::DispatchKind::CUSTOM, buffer, buffer_size,
                 buffer_mode, owned, mode, base_args...),
        ops(fops) {}

  // Tears down subclass-specific backing state (if any) and rebinds the I/O
  // hooks while preserving |ops.close| so |close()| still deletes the exact
  // derived dynamic type.
  LIBC_INLINE void rebind_ops(typename Ops::WriteFunc *w,
                              typename Ops::ReadFunc *r,
                              typename Ops::SeekFunc *s) {
    using FileBaseT = typename BaseFile::FileBase;
    if (ops.detach != nullptr) {
      ops.detach(this);
      ops.detach = nullptr;
    }
    ops.write = w != nullptr ? w : &internal::default_write<FileBaseT>;
    ops.read = r != nullptr ? r : &internal::default_read<FileBaseT>;
    ops.seek = s != nullptr ? s : &internal::default_seek<FileBaseT>;
  }
};

template <typename BaseFile> class CookieFileImpl final : public BaseFile {
  static_assert(
      cpp::is_same_v<BaseFile, typename BaseFile::PlatformType::StreamBase>,
      "BaseFile must be Platform::StreamBase");

  void *cookie;
  CookieOps ops;
  friend typename BaseFile::FileBase;

  LIBC_INLINE FileIOResult cookie_write(const void *data, size_t size) {
    if (ops.write == nullptr)
      return size;
    ssize_t result =
        ops.write(cookie, reinterpret_cast<const char *>(data), size);
    if (result < 0 || static_cast<size_t>(result) > size)
      return {0, EIO};
    size_t written = static_cast<size_t>(result);
    return {written, written < size ? EIO : 0};
  }

  LIBC_INLINE FileIOResult cookie_read(void *data, size_t size) {
    if (ops.read == nullptr)
      return 0;
    ssize_t result = ops.read(cookie, reinterpret_cast<char *>(data), size);
    if (result < 0 || static_cast<size_t>(result) > size)
      return {0, EIO};
    return static_cast<size_t>(result);
  }

  LIBC_INLINE ErrorOr<off_t> cookie_seek(off_t offset, int whence) {
    if (ops.seek == nullptr)
      return Error(ESPIPE);
    if (ops.seek(cookie, &offset, whence) == 0)
      return offset;
    return Error(EIO);
  }

  LIBC_INLINE int cookie_close() {
    BaseFile::FileBase::remove_file(this);
    int retval = ops.close != nullptr ? ops.close(cookie) : 0;
    delete this;
    return retval == 0 ? 0 : EIO;
  }

public:
  // CookieFileImpl instances must be heap-allocated because close() deletes
  // |this| after invoking |ops.close(cookie)|.
  template <typename... Args>
  LIBC_INLINE constexpr CookieFileImpl(void *c, CookieOps cops, uint8_t *buffer,
                                       size_t buffer_size, int buffer_mode,
                                       bool owned, FileMode mode,
                                       Args... base_args)
      : BaseFile(BaseFile::DispatchKind::COOKIE, buffer, buffer_size,
                 buffer_mode, owned, mode, base_args...),
        cookie(c), ops(cops) {}
};

using File = FileImpl<FilePlatform>;

extern template class FileImpl<FilePlatform>;

// The implementation of this function is provided by the platform_file
// library.
ErrorOr<File *> openfile(const char *path, const char *mode);
// Reopens a file stream.
// Note: On failure, `reopenfile` will place the file stream in an invalid state
// (closing the underlying file descriptor) but will not deallocate the `File`
// object itself, ensuring static streams like stdin/stdout/stderr are
// preserved.
int reopenfile(File *f, const char *path, const char *mode);
// Expected to be implemented by the platform file, will be called after
// locking.
int reopenfile_unlocked(File *f, const char *path, const char *mode);

// The platform_file library should implement it if it relevant for that
// platform.
ErrorOr<int> get_fileno(File *f);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_FILE_H
