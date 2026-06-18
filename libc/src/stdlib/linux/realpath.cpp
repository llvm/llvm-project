//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of POSIX realpath.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/realpath.h"
#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace {

// Separator character for POSIX paths.
constexpr char PATH_SEP = '/';

// Separator for POSIX paths, as a `cpp::string_view`.
constexpr cpp::string_view PATH_SEP_STRING = "/";

// Dummy struct to represent success in `ErrorOr` when no value is needed.
struct Ok {};

// Whether a path is absolute.
bool is_absolute(cpp::string_view path) { return path.starts_with(PATH_SEP); }

// Container for a fully resolved, canonical path.
//
// The contained path is always in its canonical form. It is:
// - Absolute
// - Symlink-free
// - Without a trailing separator
// - Devoid of path traversals like "." or ".."
class ResolvedPath {
public:
  ResolvedPath() { set_to_root(); }

  void set_to_root() {
    buf_[0] = PATH_SEP;
    size_ = 1;
  }

  bool is_root() const { return view() == PATH_SEP_STRING; }

  ErrorOr<Ok> set_to_cwd() { return Error(ENOSYS); }

  void set_to_parent() {
    size_t sep_index = view().find_last_of(PATH_SEP);

    // Ensure we maintain the root separator.
    size_ = sep_index == 0 ? 1 : sep_index;
  }

  // Adds a single component to the end of this path.
  ErrorOr<Ok> push_component(cpp::string_view component) {
    if (!is_root()) {
      if (ErrorOr<Ok> res = push_raw(PATH_SEP_STRING); !res)
        return res;
    }

    return push_raw(component);
  }

  cpp::string_view view() const { return cpp::string_view(buf_, size_); }

private:
  ErrorOr<Ok> push_raw(cpp::string_view value) {
    if (value.size() > sizeof(buf_) - size_)
      return Error(ENAMETOOLONG);

    inline_memcpy(buf_ + size_, value.data(), value.size());
    size_ += value.size();
    return Ok{};
  }

  // Current size of the path stored in `buf_`.
  size_t size_;

  // `PATH_MAX` includes a null-terminator in its count,
  // so use `PATH_MAX - 1` here as `ResolvedPath` is not null-terminated.
  char buf_[PATH_MAX - 1];
};

// A view over path components yet to be processed by realpath.
//
// When `realpath("./a/../b")` is called, the input path can be viewed as
// a stack of components, where components closest to the root are at the top.
// For example:
//
//   ```
//   PendingPath p("./a/..");
//   assert(p.advance_component() == ".");
//   assert(p.advance_component() == "a");
//   assert(p.advance_component() == "..");
//   assert(p.empty());
//   ```
class PendingPath {
public:
  explicit PendingPath(cpp::string_view path) : view_(path) {}

  // Whether all path components have been consumed.
  bool empty() const { return view_.empty(); }

  // Takes the next path component,
  // starting with the component closest to the root.
  cpp::string_view advance_component() {
    const cpp::string_view path = view_;

    const size_t component_start = path.find_first_not_of(PATH_SEP);
    if (component_start == cpp::string_view::npos) {
      view_ = "";
      return "";
    }

    const size_t component_end =
        path.find_first_of(PATH_SEP, /* From = */ component_start);
    if (component_end == cpp::string_view::npos) {
      view_ = "";
      return path.substr(component_start);
    }

    view_ = view_.substr(component_end);
    return path.substr(component_start, component_end - component_start);
  }

private:
  cpp::string_view view_;
};

ErrorOr<char *> copy_or_allocate_cstr(char *dst, cpp::string_view src) {
  if (dst == nullptr) {
    AllocChecker ac;
    // `dst` is safe to return and let the caller `free()`, since AllocChecker
    // delegates to malloc, and this value is trivially destructible.
    dst = new (ac) char[src.size() + 1];
    if (!ac)
      return Error(ENOMEM);
  }
  inline_memcpy(dst, src.data(), src.size());
  dst[src.size()] = '\0';
  return dst;
}

ErrorOr<Ok> resolve_path(PendingPath &pending_path,
                         ResolvedPath &resolved_path) {
  while (!pending_path.empty()) {
    cpp::string_view component = pending_path.advance_component();
    if (component.empty() || component == ".")
      continue;

    if (component == "..") {
      resolved_path.set_to_parent();
      continue;
    }

    if (ErrorOr<Ok> res = resolved_path.push_component(component); !res)
      return res;
  }

  return Ok{};
}

ErrorOr<char *> realpath_impl(const char *__restrict path_cstr,
                              char *__restrict resolved_path_buf) {
  if (path_cstr == nullptr)
    return Error(EINVAL);

  cpp::string_view path(path_cstr);
  if (path.size() == 0)
    return Error(ENOENT);

  if (path.size() >= PATH_MAX)
    return Error(ENAMETOOLONG);

  PendingPath pending_path(path);

  ResolvedPath resolved_path;
  if (!is_absolute(path)) {
    if (ErrorOr<Ok> res = resolved_path.set_to_cwd(); !res)
      return Error(res.error());
  }

  if (ErrorOr<Ok> res = resolve_path(pending_path, resolved_path); !res)
    return Error(res.error());

  return copy_or_allocate_cstr(resolved_path_buf, resolved_path.view());
}

} // namespace

LLVM_LIBC_FUNCTION(char *, realpath,
                   (const char *__restrict path,
                    char *__restrict resolved_path)) {
  ErrorOr<char *> res = realpath_impl(path, resolved_path);
  if (!res) {
    libc_errno = res.error();
    return nullptr;
  }
  return *res;
}

} // namespace LIBC_NAMESPACE_DECL
