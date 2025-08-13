//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_FD
#define _LIBCPP_STACKTRACE_FD

#include <__config>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <string_view>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

/* Wraps a C file descriptor, converting to/from `int`s.
 * Not copyable; should only be moved, so that it only has one "owner". */
struct _LIBCPP_HIDE_FROM_ABI fd {
  int fd_{-1}; // invalid iff negative

  fd() = default;
  ~fd() { close(); }

  // To / from plain old ints
  fd(int x) : fd_(x) {}
  operator int() const { return fd_; } // implicit

  // No copying (other than contorting to/from ints)
  fd(fd const&)            = delete;
  fd& operator=(fd const&) = delete;

  // Moving is ok, and the moved-from object is invalidated (gets fd of -1)
  fd(fd&& rhs) { std::swap(fd_, rhs.fd_); }
  fd& operator=(fd&& rhs) { return (std::addressof(rhs) == this) ? *this : *new (this) fd(std::move(rhs)); }

  bool valid() const { return fd_ >= 0; }

  void close() {
    if (fd_ != -1) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  /** Open `/dev/null` for reading and writing */
  static fd& null_fd() {
    static fd ret{::open("/dev/null", O_RDWR)};
    return ret;
  }

  static fd open_ro(std::string_view path) {
    fd ret = {::open(path.data(), O_RDONLY)};
    return ret;
  }

  /** Create pipe pair via `pipe`, assign into these two destination `fd`s */
  static int pipe_pair(fd& read_fd, fd& write_fd) {
    int fd_ints[2];
    if (::pipe(fd_ints) == -1) {
      return errno;
    }
    read_fd  = fd_ints[0];
    write_fd = fd_ints[1];
    return 0;
  }

  struct _LIBCPP_HIDE_FROM_ABI streambuf;
  struct _LIBCPP_HIDE_FROM_ABI istream;
  struct _LIBCPP_HIDE_FROM_ABI mmap;
};

/** Wraps a readable fd using the `streambuf` interface. */
struct _LIBCPP_HIDE_FROM_ABI fd::streambuf final : std::streambuf {
  fd& fd_;
  char* buf_;
  size_t size_;

  streambuf(fd& fd, char* buf, size_t size) : fd_(fd), buf_(buf), size_(size) {}
  virtual ~streambuf() = default;

  int underflow() override {
    int count = ::read(fd_, buf_, size_);
    if (count <= 0) {
      // error or EOF: return eof to stop
      return traits_type::eof();
    }
    auto ret = int(*buf_);
    setg(buf_, buf_, buf_ + count);
    return ret;
  }
};

/** Wraps an `FDInStreamBuffer` in an `istream` */
struct _LIBCPP_HIDE_FROM_ABI fd::istream final : std::istream {
  fd::streambuf& buf_;
  virtual ~istream() = default;
  explicit istream(fd::streambuf& buf) : std::istream(nullptr), buf_(buf) { rdbuf(&buf_); }
};

/** Read-only memory mapping.  Requires an `fd`, or a path to open an `fd` out of.  Takes ownership and destruction duty
 * of the fd. */
struct _LIBCPP_HIDE_FROM_ABI fd::mmap final {
  fd fd_{};
  size_t size_{0};
  std::byte const* addr_{nullptr};

  explicit mmap(std::string_view path) : mmap(fd::open_ro(path)) {}

  explicit mmap(fd&& fd) : fd_(std::move(fd)) {
    if (fd_) {
      if ((size_ = ::lseek(fd_, 0, SEEK_END))) {
        addr_ = (std::byte const*)::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
      }
    }
  }

  operator bool() const { return addr_; }

  ~mmap() {
    if (addr_) {
      ::munmap(const_cast<void*>((void const*)addr_), size_);
    }
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_FD
