//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal recursive file tree walk implementation for ftw and nftw.
///
//===----------------------------------------------------------------------===//

#include "src/ftw/ftw_impl.h"

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/ftw_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/__ftw_func_t.h"
#include "hdr/types/__nftw_func_t.h"
#include "hdr/types/dev_t.h"
#include "hdr/types/ino_t.h"
#include "hdr/types/struct_FTW.h"
#include "hdr/types/struct_dirent.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/expected.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/dir.h"
#include "src/__support/File/scoped_dir.h"
#include "src/__support/OSUtil/linux/stat/stat_via_statx.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fchdir.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
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

struct FchdirGuard {
  int saved_fd;
  bool should_close;

  LIBC_INLINE FchdirGuard(int fd, bool close_on_exit)
      : saved_fd(fd), should_close(close_on_exit) {}

  LIBC_INLINE ~FchdirGuard() {
    if (saved_fd >= 0) {
      linux_syscalls::fchdir(saved_fd);
      if (should_close)
        linux_syscalls::close(saved_fd);
    }
  }
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

struct CallbackAdapter {
  __ftw_func_t ftw_fn;
  __nftw_func_t nftw_fn;
  bool is_nftw;

  LIBC_INLINE int invoke(const char *path, const struct stat *sb, int type_flag,
                         struct FTW *ftw_buf) const {
    if (is_nftw)
      return nftw_fn(path, sb, type_flag, ftw_buf);
    return ftw_fn(path, sb, type_flag);
  }
};

cpp::expected<int, int> walk_tree(PathBuffer &path_buf,
                                  const CallbackAdapter &cb, int fd_limit,
                                  int flags, int level, dev_t &start_device,
                                  int parent_fd, const Node *history) {
  if (fd_limit <= 0)
    return 0;

  cpp::string_view path_view(path_buf.data, path_buf.len);
  size_t slash_pos = path_view.find_last_of('/');
  int base = 0;
  if (slash_pos != cpp::string_view::npos && path_buf.len > 1)
    base = static_cast<int>(slash_pos + 1);
  struct FTW ftw_buf = {base, level};

  const char *os_path = path_buf.data;
  if ((flags & FTW_CHDIR) && level > 0)
    os_path = path_buf.data + ftw_buf.base;

  struct stat stat_buf = {};
  int type_flag = FTW_NS;
  int stat_flags = (flags & FTW_PHYS) ? AT_SYMLINK_NOFOLLOW : 0;
  ErrorOr<void> stat_res =
      internal::stat_via_statx(AT_FDCWD, os_path, stat_flags, &stat_buf);
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
    } else if (!(flags & FTW_PHYS) && (err == ENOENT || err == ENOTDIR)) {
      ErrorOr<void> lstat_res = internal::stat_via_statx(
          AT_FDCWD, os_path, AT_SYMLINK_NOFOLLOW, &stat_buf);
      if (lstat_res.has_value() && S_ISLNK(stat_buf.st_mode))
        type_flag = cb.is_nftw ? FTW_SLN : FTW_SL;
      else
        return cpp::unexpected(err);
    } else {
      return cpp::unexpected(err);
    }
  }

  if (type_flag != FTW_NS) {
    if (level == 0)
      start_device = stat_buf.st_dev;
    else if ((flags & FTW_MOUNT) && stat_buf.st_dev != start_device)
      return 0;
  }

  auto normalize_ret = [&](int res) -> int {
    if (!(flags & FTW_ACTIONRETVAL))
      return res;
    if (res == FTW_CONTINUE || res == FTW_SKIP_SUBTREE)
      return 0;
    if (res == FTW_SKIP_SIBLINGS)
      return level == 0 ? 0 : FTW_SKIP_SIBLINGS;
    return res;
  };

  if (type_flag != FTW_D)
    return normalize_ret(
        cb.invoke(path_buf.data, &stat_buf, type_flag, &ftw_buf));

  for (const Node *curr = history; curr != nullptr; curr = curr->parent) {
    if (curr->ino == stat_buf.st_ino && curr->dev == stat_buf.st_dev) {
      if (!(flags & FTW_DEPTH))
        return normalize_ret(
            cb.invoke(path_buf.data, &stat_buf, FTW_D, &ftw_buf));
      return 0;
    }
  }

  ErrorOr<Dir *> dir_or_err = Dir::open(os_path);
  ScopedDir open_dir;
  if (!dir_or_err.has_value()) {
    if (dir_or_err.error() != EACCES)
      return cpp::unexpected(dir_or_err.error());
    type_flag = FTW_DNR;
  } else {
    open_dir = ScopedDir(dir_or_err.value());
  }

  bool skip_siblings = false;
  if (!(flags & FTW_DEPTH)) {
    int res = cb.invoke(path_buf.data, &stat_buf, type_flag, &ftw_buf);
    if (flags & FTW_ACTIONRETVAL) {
      if (res == FTW_SKIP_SUBTREE)
        return 0;
      if (res == FTW_SKIP_SIBLINGS)
        skip_siblings = true;
      else if (res != FTW_CONTINUE)
        return res;
    } else if (res != 0) {
      return res;
    }
  }

  if (open_dir) {
    int cur_dir_fd = open_dir->getfd();
    if (flags & FTW_CHDIR) {
      ErrorOr<int> chdir_res = linux_syscalls::fchdir(cur_dir_fd);
      if (!chdir_res.has_value())
        return cpp::unexpected(chdir_res.error());
    }
    FchdirGuard level_cwd_guard((flags & FTW_CHDIR) ? parent_fd : -1, false);

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

      auto child_res = walk_tree(path_buf, cb, fd_limit - 1, flags, level + 1,
                                 start_device, cur_dir_fd, &current_node);
      path_buf.pop_to(saved_len);

      if (!child_res.has_value())
        return child_res;
      if ((flags & FTW_ACTIONRETVAL) && child_res.value() == FTW_SKIP_SIBLINGS)
        break;
      if (child_res.value() != 0)
        return child_res.value();
    }

    int close_ret = open_dir.close();
    if (close_ret != 0)
      return cpp::unexpected(close_ret);
  }

  if (flags & FTW_DEPTH) {
    int post_flag = (type_flag == FTW_DNR) ? FTW_DNR : FTW_DP;
    return normalize_ret(
        cb.invoke(path_buf.data, &stat_buf, post_flag, &ftw_buf));
  }

  return skip_siblings ? (level == 0 ? 0 : FTW_SKIP_SIBLINGS) : 0;
}

cpp::expected<int, int> run_ftw(const char *dir_path, const CallbackAdapter &cb,
                                int fd_limit, int flags) {
  if (fd_limit <= 0)
    return cpp::unexpected(EINVAL);
  if (dir_path[0] == '\0')
    return cpp::unexpected(ENOENT);

  PathBuffer path_buf;
  if (!path_buf.init(dir_path))
    return cpp::unexpected(ENOMEM);

  int start_fd = -1;
  if (flags & FTW_CHDIR) {
    ErrorOr<int> fd_or_err =
        linux_syscalls::open(".", O_RDONLY | O_DIRECTORY, 0);
    if (!fd_or_err.has_value())
      return cpp::unexpected(fd_or_err.error());
    start_fd = fd_or_err.value();
  }
  FchdirGuard start_cwd_guard(start_fd, true);

  dev_t start_device = 0;
  return walk_tree(path_buf, cb, fd_limit, flags, 0, start_device, start_fd,
                   nullptr);
}

} // namespace

cpp::expected<int, int> ftw_impl(const char *dir_path, __ftw_func_t fn,
                                 int fd_limit) {
  CallbackAdapter cb = {fn, nullptr, false};
  return run_ftw(dir_path, cb, fd_limit, 0);
}

cpp::expected<int, int> nftw_impl(const char *dir_path, __nftw_func_t fn,
                                  int fd_limit, int flags) {
  CallbackAdapter cb = {nullptr, fn, true};
  return run_ftw(dir_path, cb, fd_limit, flags);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL
