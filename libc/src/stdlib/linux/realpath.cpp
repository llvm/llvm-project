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
#include "hdr/fcntl_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/stat/kernel_statx_types.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getcwd.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/statx.h"
#include "src/__support/OSUtil/path.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace {

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

  void set_to_root() { path_ = path::SEPARATOR; }

  cpp::optional<Error> set_to_cwd() {
    char buf[PATH_MAX];
    ErrorOr<ssize_t> ret = linux_syscalls::getcwd(buf, PATH_MAX);
    if (!ret) {
      if (ret.error() == ERANGE)
        return Error(ENAMETOOLONG);
      return Error(ret.error());
    }

    if (*ret <= 0)
      return Error(EIO);

    path_ = cpp::string_view(buf, *ret - 1);
    return cpp::nullopt;
  }

  // Removes the trailing path component.
  void set_to_parent() {
    size_t sep_index = cpp::string_view(path_).find_last_of(path::SEPARATOR);

    // Never move past the root separator. For example,
    // ensures that set_to_parent on "/hello" only resizes to "/".
    path_.resize(sep_index >= 1 ? sep_index : 1);
  }

  // Adds a single component to the end of this path.
  cpp::optional<Error> push_component(cpp::string_view component) {
    if (!path::is_root(path_)) {
      if (cpp::optional<Error> err = push_raw(path::SEPARATOR); err)
        return err;
    }

    return push_raw(component);
  }

  // Releases ownership of the underlying C-string and resets this path.
  //
  // Must be free'd by the caller.
  char *release() { return path_.release_c_str(); }

  const char *c_str() const { return path_.c_str(); }

  // Copies the content of this path to `dst`.
  void copy_to(char *dst) {
    inline_memcpy(dst, path_.c_str(), path_.size() + 1);
  }

private:
  cpp::optional<Error> push_raw(cpp::string_view value) {
    // -1 because PATH_MAX includes a null-terminator.
    size_t remaining_bytes = (PATH_MAX - 1) - path_.size();
    if (value.size() > remaining_bytes)
      return Error(ENAMETOOLONG);

    path_ += value;
    return cpp::nullopt;
  }

  cpp::optional<Error> push_raw(char c) {
    return push_raw(cpp::string_view(&c, 1));
  }

  cpp::string path_;
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

    const size_t component_start = path.find_first_not_of(path::SEPARATOR);
    if (component_start == cpp::string_view::npos) {
      view_ = "";
      return "";
    }

    const size_t component_end =
        path.find_first_of(path::SEPARATOR, /* From = */ component_start);
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

ErrorOr<mode_t> read_file_type(const char *path) {
  internal::kernel_statx_buf buf;
  ErrorOr<int> ret =
      linux_syscalls::statx(AT_FDCWD, path, AT_SYMLINK_NOFOLLOW,
                            internal::KERNEL_STATX_TYPE_MASK, &buf);
  if (!ret)
    return Error(ret.error());

  if (!(buf.stx_mask & internal::KERNEL_STATX_TYPE_MASK))
    return Error(EIO);

  return static_cast<mode_t>(buf.stx_mode);
}

cpp::optional<Error> resolve_path(PendingPath &pending_path,
                                  ResolvedPath &resolved_path) {
  while (!pending_path.empty()) {
    cpp::string_view component = pending_path.advance_component();
    if (component.empty() || component == path::CURRENT_DIR_COMPONENT)
      continue;

    if (component == path::PARENT_DIR_COMPONENT) {
      resolved_path.set_to_parent();
      continue;
    }

    if (cpp::optional<Error> err = resolved_path.push_component(component); err)
      return err;

    ErrorOr<mode_t> mode = read_file_type(resolved_path.c_str());
    if (!mode)
      return Error(mode.error());

    // TODO: Resolve symbolic links.
    if (S_ISLNK(*mode))
      return Error(ENOSYS);

    // If the path is not a directory, but there is more to resolve, then error.
    // For example, realpath("/path/to/file.txt/") should give ENOTDIR.
    if (!S_ISDIR(*mode) && !pending_path.empty())
      return Error(ENOTDIR);
  }

  return cpp::nullopt;
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
  if (!path::is_absolute(path)) {
    if (cpp::optional<Error> err = resolved_path.set_to_cwd(); err)
      return *err;
  }

  if (cpp::optional<Error> err = resolve_path(pending_path, resolved_path); err)
    return *err;

  if (resolved_path_buf != nullptr) {
    resolved_path.copy_to(resolved_path_buf);
    return resolved_path_buf;
  }
  return resolved_path.release();
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
