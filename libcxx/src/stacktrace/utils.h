//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_DEBUG_H
#define _LIBCPP_STACKTRACE_DEBUG_H

#include <__config>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct failed : std::runtime_error {
  virtual ~failed() = default;
  int errno_{0};
  failed() : std::runtime_error({}) {}
  failed(char const* msg, int err) : std::runtime_error(msg), errno_(err) {}
};

/** Debug-message output stream.  If `LIBCXX_STACKTRACE_DEBUG` is defined in the environment
or as a macro with exactly the string `1` then this is enabled (prints to `std::cerr`);
otherwise its does nothing by returning a dummy stream. */
struct _LIBCPP_HIDE_FROM_ABI debug : std::ostream {
  _LIBCPP_HIDE_FROM_ABI virtual ~debug() = default;

  _LIBCPP_HIDE_FROM_ABI static bool enabled() {
#if defined(LIBCXX_STACKTRACE_DEBUG) && LIBCXX_STACKTRACE_DEBUG == 1
    return true;
#else
    static bool ret = [] {
      auto const* val = getenv("LIBCXX_STACKTRACE_DEBUG");
      return val && !strncmp(val, "1", 1);
    }();
    return ret;
#endif
  }

  /** No-op output stream. */
  struct _LIBCPP_HIDE_FROM_ABI dummy_ostream final : std::ostream {
    _LIBCPP_HIDE_FROM_ABI virtual ~dummy_ostream() = default;
    friend std::ostream& operator<<(dummy_ostream& bogus, auto const&) { return bogus; }
  };

  friend std::ostream& operator<<(debug& dp, auto const& val) {
    static dummy_ostream kdummy;
    if (!enabled()) {
      return kdummy;
    }
    std::cerr << val;
    return std::cerr;
  }
};

/** Encapsulates a plain old file descriptor `int`.  Avoids copies in order to
force some component to "own" this, although it's freely convertible back to
integer form.  Default-constructed, closed, and moved-out-of instances will have
the invalid fd `-1`. */
class _LIBCPP_HIDE_FROM_ABI fd {
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
struct _LIBCPP_HIDE_FROM_ABI fd_streambuf final : std::streambuf {
  fd& fd_;
  char* buf_;
  size_t size_;
  _LIBCPP_HIDE_FROM_ABI fd_streambuf(fd& fd, char* buf, size_t size) : fd_(fd), buf_(buf), size_(size) {}
  _LIBCPP_HIDE_FROM_ABI virtual ~fd_streambuf() = default;
  _LIBCPP_HIDE_FROM_ABI int underflow() override;
};

/** Wraps an `FDInStreamBuffer` in an `istream` */
struct fd_istream final : std::istream {
  fd_streambuf& buf_;
  _LIBCPP_HIDE_FROM_ABI virtual ~fd_istream() = default;
  _LIBCPP_HIDE_FROM_ABI explicit fd_istream(fd_streambuf& buf) : std::istream(nullptr), buf_(buf) { rdbuf(&buf_); }
};

struct fd_mmap final {
  fd fd_{};
  size_t size_{0};
  std::byte const* addr_{nullptr};

  _LIBCPP_HIDE_FROM_ABI explicit fd_mmap(std::string_view path) : fd_mmap(fd::open(path)) {}

  _LIBCPP_HIDE_FROM_ABI explicit fd_mmap(fd&& fd) : fd_(std::move(fd)) {
    if (fd_) {
      if ((size_ = ::lseek(fd, 0, SEEK_END))) {
        addr_ = (std::byte const*)::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
      }
    }
  }

  _LIBCPP_HIDE_FROM_ABI operator bool() const { return addr_; }

  _LIBCPP_HIDE_FROM_ABI ~fd_mmap() {
    if (addr_) {
      ::munmap(const_cast<void*>((void const*)addr_), size_);
    }
  }
};

constexpr unsigned k_max_images = 256;

struct _LIBCPP_HIDE_FROM_ABI image {
  uintptr_t loaded_at_{};
  intptr_t slide_{};
  std::string_view name_{};
  bool is_main_prog_{};

  _LIBCPP_HIDE_FROM_ABI bool operator<(image const& rhs) const { return loaded_at_ < rhs.loaded_at_; }
  _LIBCPP_HIDE_FROM_ABI operator bool() const { return !name_.empty(); }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_DEBUG_H
