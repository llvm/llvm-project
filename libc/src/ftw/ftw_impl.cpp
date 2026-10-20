//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal recursive file tree walk implementation.
///
//===----------------------------------------------------------------------===//

#include "src/ftw/ftw_impl.h"

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/ftw_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/__ftw_func_t.h"
#include "hdr/types/dev_t.h"
#include "hdr/types/ino_t.h"
#include "hdr/types/struct_dirent.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/expected.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/dir.h"
#include "src/__support/File/scoped_dir.h"
#include "src/__support/OSUtil/linux/stat/stat_via_statx.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

namespace {

struct Node {
  ino_t ino;
  dev_t dev;
  const Node *parent;
};

struct PathBuffer {
  char *data = nullptr;
  size_t len = 0;
  size_t capacity = 0;

  LIBC_INLINE ~PathBuffer() { operator delete(data); }

  bool ensure_capacity(size_t required) {
    if (required <= capacity)
      return true;
    size_t new_cap = capacity == 0 ? 256 : capacity * 2;
    if (new_cap < required)
      new_cap = required;
    AllocChecker ac;
    char *new_data = new (ac) char[new_cap];
    if (!ac)
      return false;
    if (len > 0)
      inline_memcpy(new_data, data, len);
    new_data[len] = '\0';
    operator delete(data);
    data = new_data;
    capacity = new_cap;
    return true;
  }

  bool init(cpp::string_view sv) {
    while (sv.size() > 1 && sv.back() == '/')
      sv.remove_suffix(1);
    if (!ensure_capacity(sv.size() + 1))
      return false;
    inline_memcpy(data, sv.data(), sv.size());
    len = sv.size();
    data[len] = '\0';
    return true;
  }

  bool push_child(cpp::string_view child_name) {
    bool need_slash = (len > 0 && data[len - 1] != '/');
    size_t add_len = (need_slash ? 1 : 0) + child_name.size();
    if (!ensure_capacity(len + add_len + 1))
      return false;
    if (need_slash)
      data[len++] = '/';
    inline_memcpy(data + len, child_name.data(), child_name.size());
    len += child_name.size();
    data[len] = '\0';
    return true;
  }

  void pop_to(size_t prev_len) {
    len = prev_len;
    data[len] = '\0';
  }
};

cpp::expected<int, int> walk_tree(PathBuffer &path_buf, __ftw_func_t fn,
                                  int fd_limit, int level,
                                  const Node *history) {
  if (fd_limit <= 0)
    return 0;

  struct stat stat_buf = {};
  int type_flag = FTW_NS;
  ErrorOr<void> stat_res =
      internal::stat_via_statx(AT_FDCWD, path_buf.data, 0, &stat_buf);
  if (stat_res.has_value()) {
    if (S_ISDIR(stat_buf.st_mode))
      type_flag = FTW_D;
    else if (S_ISLNK(stat_buf.st_mode))
      type_flag = FTW_SL;
    else
      type_flag = FTW_F;
  } else {
    int err = stat_res.error();
    if (err == EACCES && level > 0) {
      type_flag = FTW_NS;
    } else if (err == ENOENT || err == ENOTDIR) {
      ErrorOr<void> lstat_res = internal::stat_via_statx(
          AT_FDCWD, path_buf.data, AT_SYMLINK_NOFOLLOW, &stat_buf);
      if (lstat_res.has_value() && S_ISLNK(stat_buf.st_mode))
        type_flag = FTW_SL;
      else
        return cpp::unexpected(err);
    } else {
      return cpp::unexpected(err);
    }
  }

  if (type_flag != FTW_D)
    return fn(path_buf.data, &stat_buf, type_flag);

  for (const Node *curr = history; curr != nullptr; curr = curr->parent) {
    if (curr->ino == stat_buf.st_ino && curr->dev == stat_buf.st_dev)
      return fn(path_buf.data, &stat_buf, FTW_D);
  }

  ErrorOr<Dir *> dir_or_err = Dir::open(path_buf.data);
  ScopedDir open_dir;
  if (!dir_or_err.has_value()) {
    if (dir_or_err.error() != EACCES)
      return cpp::unexpected(dir_or_err.error());
    type_flag = FTW_DNR;
  } else {
    open_dir = ScopedDir(dir_or_err.value());
  }

  int res = fn(path_buf.data, &stat_buf, type_flag);
  if (res != 0 || !open_dir)
    return res;

  Node current_node = {stat_buf.st_ino, stat_buf.st_dev, history};
  while (true) {
    ErrorOr<struct dirent *> entry_or_err = open_dir->read();
    if (!entry_or_err.has_value())
      return cpp::unexpected(entry_or_err.error());

    struct dirent *entry = entry_or_err.value();
    if (entry == nullptr)
      break;

    cpp::string_view name(entry->d_name);
    if (name == "." || name == "..")
      continue;

    size_t saved_len = path_buf.len;
    if (!path_buf.push_child(name))
      return cpp::unexpected(ENOMEM);

    auto child_res =
        walk_tree(path_buf, fn, fd_limit - 1, level + 1, &current_node);
    path_buf.pop_to(saved_len);

    if (!child_res.has_value())
      return child_res;
    if (child_res.value() != 0)
      return child_res.value();
  }

  int close_ret = open_dir.close();
  if (close_ret != 0)
    return cpp::unexpected(close_ret);
  return 0;
}

} // namespace

cpp::expected<int, int> ftw_impl(const char *dir_path, __ftw_func_t fn,
                                 int fd_limit) {
  if (fd_limit <= 0)
    return cpp::unexpected(EINVAL);
  if (dir_path[0] == '\0')
    return cpp::unexpected(ENOENT);

  PathBuffer path_buf;
  if (!path_buf.init(dir_path))
    return cpp::unexpected(ENOMEM);

  return walk_tree(path_buf, fn, fd_limit, 0, nullptr);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
