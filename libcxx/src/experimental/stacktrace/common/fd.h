//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_FD_H
#define _LIBCPP_STACKTRACE_FD_H

#include <__config>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

/** Encapsulates a plain old file descriptor `int`.  Avoids copies in order to
force some component to "own" this, although it's freely convertible back to
integer form.  Default-constructed, closed, and moved-out-of instances will have
the invalid fd `-1`. */
class fd {
  int fd_{-1};

public:
  fd() : fd(-1) {}
  fd(int fdint) : fd_(fdint) {}

  fd(fd const&)            = delete;
  fd& operator=(fd const&) = delete;

  fd(fd&& rhs) {
    if (&rhs != this) {
      std::exchange(fd_, rhs.fd_);
    }
  }

  fd& operator=(fd&& rhs) { return *new (this) fd(std::move(rhs)); }

  ~fd() { close(); }

  /** Returns true IFF fd is above zero */
  bool valid() const { return fd_ > 0; }

  /* implicit */ operator int() const {
    auto ret = fd_;
    assert(ret > 0);
    return ret;
  }

  void close() {
    int fd_old = -1;
    std::exchange(fd_old, fd_);
    if (fd_old != -1) {
      ::close(fd_old);
    }
  }

  static fd& null_fd() {
    static fd ret = {::open("/dev/null", O_RDWR)};
    return ret;
  }

  static fd open(std::string_view path) {
    fd ret = {::open(path.data(), O_RDONLY)};
    return ret;
  }
};

/** Wraps a readable fd using the `streambuf` interface.  I/O errors arising
from reading the provided fd will result in a `Failed` being thrown. */
struct fd_streambuf final : std::streambuf {
  fd& fd_;
  char* buf_;
  size_t size_;
  fd_streambuf(fd& fd, char* buf, size_t size) : fd_(fd), buf_(buf), size_(size) {}
  virtual ~fd_streambuf();
  int underflow() override;
};

/** Wraps an `FDInStreamBuffer` in an `istream` */
struct fd_istream final : std::istream {
  fd_streambuf& buf_;
  virtual ~fd_istream();
  explicit fd_istream(fd_streambuf& buf) : std::istream(nullptr), buf_(buf) { rdbuf(&buf_); }
};

struct fd_mmap final {
  fd fd_{};
  size_t size_{0};
  std::byte const* addr_{nullptr};

  explicit fd_mmap(std::string_view path) : fd_mmap(fd::open(path)) {}

  explicit fd_mmap(fd&& fd) : fd_(std::move(fd)) {
    if (fd_) {
      if ((size_ = ::lseek(fd, 0, SEEK_END))) {
        addr_ = (std::byte const*)::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
      }
    }
  }

  operator bool() const { return addr_; }

  ~fd_mmap() {
    if (addr_) {
      ::munmap(const_cast<void*>((void const*)addr_), size_);
    }
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_FD_H
