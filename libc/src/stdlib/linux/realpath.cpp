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
#include "src/__support/OSUtil/linux/syscall_wrappers/readlink.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/statx.h"
#include "src/__support/OSUtil/path.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace {

#ifdef SYMLOOP_MAX
constexpr size_t MAX_SYMLINK_FOLLOWS = SYMLOOP_MAX;
#else
// Maximum number of symlinks that may be followed during path resolution.
// This is a large, arbitrary value consistent with other libc implementations.
// Must be at least _POSIX_SYMLOOP_MAX (8). Ideally should be read from sysconf,
// so that it follows limits set by the system libc in overlay mode.
constexpr size_t MAX_SYMLINK_FOLLOWS = 40;
#endif

// Container for a fully resolved, canonical path.
//
// The contained path is always in its canonical form. It is:
// - Absolute
// - Without a trailing separator
// - Devoid of path traversals like "." or ".."
class ResolvedPath {
public:
  ResolvedPath() { set_to_root(); }

  void set_to_root() { path = path::SEPARATOR; }

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

    path = cpp::string_view(buf, *ret - 1);
    return cpp::nullopt;
  }

  // Removes the trailing path component.
  void set_to_parent() {
    size_t sep_index = cpp::string_view(path).find_last_of(path::SEPARATOR);

    // Never move past the root separator. For example,
    // ensures that set_to_parent on "/hello" only resizes to "/".
    path.resize(sep_index >= 1 ? sep_index : 1);
  }

  // Adds a single component to the end of this path.
  cpp::optional<Error> push_component(cpp::string_view component) {
    if (!path::is_root(path)) {
      if (cpp::optional<Error> err = push_raw(path::SEPARATOR); err)
        return err;
    }

    return push_raw(component);
  }

  // Releases ownership of the underlying C-string and resets this path.
  //
  // Must be free'd by the caller.
  char *release() { return path.release_c_str(); }

  const char *c_str() const { return path.c_str(); }

  // Copies the content of this path to `dst`.
  void copy_to(char *dst) { inline_memcpy(dst, path.c_str(), path.size() + 1); }

private:
  cpp::optional<Error> push_raw(cpp::string_view value) {
    // -1 because PATH_MAX includes a null-terminator.
    size_t remaining_bytes = (PATH_MAX - 1) - path.size();
    if (value.size() > remaining_bytes)
      return Error(ENAMETOOLONG);

    path += value;
    return cpp::nullopt;
  }

  cpp::optional<Error> push_raw(char c) {
    return push_raw(cpp::string_view(&c, 1));
  }

  cpp::string path;
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
//
//   p.prepend("b/c");
//   assert(p.advance_component() == "b");
//   assert(p.advance_component() == "c");
//   assert(p.advance_component() == "..");
//   assert(p.empty());
//   ```
class PendingPath {
public:
  PendingPath() {}

  // Whether all path components have been consumed.
  bool empty() const { return consumed_size >= path.size(); }

  // Takes the next path component,
  // starting with the component closest to the root.
  cpp::string_view advance_component() {
    cpp::string_view view = path;

    const size_t component_start =
        view.find_first_not_of(path::SEPARATOR, /* From= */ consumed_size);
    if (component_start == cpp::string_view::npos) {
      consumed_size = path.size();
      return "";
    }

    size_t component_end =
        view.find_first_of(path::SEPARATOR, /* From = */ component_start);
    if (component_end == cpp::string_view::npos)
      component_end = path.size();

    consumed_size = component_end;
    return view.substr(component_start, component_end - component_start);
  }

  // Prepends other_path to this path.
  cpp::optional<Error> prepend(cpp::string_view other_path) {
    size_t new_size = other_path.size() + path.size() - consumed_size;
    if (new_size >= PATH_MAX)
      return Error(ENAMETOOLONG);

    if (other_path.size() <= consumed_size) {
      // If the string to prepend fits in the unused prefix of path,
      // just slot it in directly and decrease consumed_size.
      size_t start = consumed_size - other_path.size();
      path.replace(start, other_path.size(), other_path);
      consumed_size = start;
    } else {
      path.replace(0, consumed_size, other_path);
      consumed_size = 0;
    }
    return cpp::nullopt;
  }

private:
  // The pending path. The unprocessed section is [consumed_size, path.size()).
  cpp::string path;

  // The prefix of path that has already been advanced through.
  size_t consumed_size = 0;
};

// Reads the target of a symlink. Returns a view into buf.
ErrorOr<cpp::string_view> readlink(const char *path, char (&buf)[PATH_MAX]) {
  ErrorOr<ssize_t> bytes_written =
      linux_syscalls::readlink(path, buf, sizeof(buf));
  if (!bytes_written)
    return Error(bytes_written.error());
  if (*bytes_written <= 0)
    return Error(EIO); // Should not be possible, but check to guard underflow

  cpp::string_view target(buf, static_cast<size_t>(*bytes_written));
  if (target.size() >= sizeof(buf))
    return Error(ENAMETOOLONG);

  return target;
}

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
  size_t symlinks_followed = 0;

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

    if (S_ISLNK(*mode)) {
      if (symlinks_followed >= MAX_SYMLINK_FOLLOWS)
        return Error(ELOOP);
      symlinks_followed += 1;

      char buf[PATH_MAX];
      ErrorOr<cpp::string_view> target = readlink(resolved_path.c_str(), buf);
      if (!target)
        return Error(target.error());

      if (cpp::optional<Error> err = pending_path.prepend(*target); err)
        return err;

      if (path::is_absolute(*target))
        resolved_path.set_to_root();

      // Since the last component of resolved_path was a link, remove it.
      resolved_path.set_to_parent();
      continue;
    }

    // If the path is not a directory, but there are more directory traversals,
    // then error. e.g. realpath("/path/to/file.txt/") should give ENOTDIR.
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

  PendingPath pending_path;
  if (cpp::optional<Error> err = pending_path.prepend(path); err)
    return Error(*err);

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
