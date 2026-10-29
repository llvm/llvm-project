//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of the POSIX nftw function.
///
//===----------------------------------------------------------------------===//

#include "src/ftw/nftw.h"

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/ftw_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/dev_t.h"
#include "hdr/types/ino_t.h"
#include "hdr/types/struct_FTW.h"
#include "hdr/types/struct_dirent.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/dir.h"
#include "src/__support/OSUtil/linux/stat/stat_via_statx.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/chdir.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fchdir.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/macros/optimization.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace {

using NftwCallback = int (*)(const char *, const struct stat *, int,
                             struct FTW *);

constexpr size_t INITIAL_PATH_CAPACITY = 256;
constexpr size_t INITIAL_NAMES_CAPACITY = 256;
constexpr size_t INITIAL_VISITED_CAPACITY = 32;
constexpr size_t MAX_PATH_BUF_CAPACITY =
    static_cast<size_t>(cpp::numeric_limits<int>::max());
constexpr int VALID_FLAGS =
    FTW_PHYS | FTW_MOUNT | FTW_CHDIR | FTW_DEPTH | FTW_ACTIONRETVAL;

struct Node {
  ino_t ino;
  dev_t dev;
  const Node *parent;
};

struct ScopedDir {
  Dir *dir = nullptr;
  explicit ScopedDir(Dir *d) : dir(d) {}
  ScopedDir(const ScopedDir &) = delete;
  ScopedDir &operator=(const ScopedDir &) = delete;
  LIBC_INLINE ~ScopedDir() noexcept {
    if (dir != nullptr)
      static_cast<void>(dir->close());
  }
  Dir *operator->() const { return dir; }
  int close() {
    if (dir == nullptr)
      return 0;
    Dir *d = dir;
    dir = nullptr;
    return d->close();
  }
};

struct VisitedDirs {
  struct Entry {
    dev_t dev;
    ino_t ino;
  };
  Entry *data = nullptr;
  size_t count = 0;
  size_t capacity = 0;

  VisitedDirs() = default;
  VisitedDirs(const VisitedDirs &) = delete;
  VisitedDirs &operator=(const VisitedDirs &) = delete;
  LIBC_INLINE ~VisitedDirs() noexcept { delete[] data; }

  bool contains(dev_t dev, ino_t ino) const {
    for (size_t i = 0; i < count; ++i)
      if (data[i].dev == dev && data[i].ino == ino)
        return true;
    return false;
  }

  bool insert(dev_t dev, ino_t ino) {
    if (count == capacity) {
      constexpr size_t MAX_CAP =
          cpp::numeric_limits<size_t>::max() / sizeof(Entry);
      if (LIBC_UNLIKELY(capacity > MAX_CAP / 2))
        return false;
      size_t new_cap = capacity == 0 ? INITIAL_VISITED_CAPACITY : capacity * 2;
      AllocChecker ac;
      Entry *new_data = new (ac) Entry[new_cap];
      if (LIBC_UNLIKELY(!ac))
        return false;
      if (count > 0)
        inline_memcpy(new_data, data, count * sizeof(Entry));
      delete[] data;
      data = new_data;
      capacity = new_cap;
    }
    data[count++] = {dev, ino};
    return true;
  }
};

struct PathBuffer {
  char *data = nullptr;
  size_t len = 0;
  size_t capacity = 0;

  PathBuffer() = default;
  PathBuffer(const PathBuffer &) = delete;
  PathBuffer &operator=(const PathBuffer &) = delete;
  LIBC_INLINE ~PathBuffer() noexcept { delete[] data; }

  bool ensure_capacity(size_t required) {
    if (required <= capacity)
      return true;
    if (LIBC_UNLIKELY(required > MAX_PATH_BUF_CAPACITY))
      return false;
    size_t new_cap = capacity == 0 ? INITIAL_PATH_CAPACITY
                                   : (capacity > MAX_PATH_BUF_CAPACITY / 2
                                          ? MAX_PATH_BUF_CAPACITY
                                          : capacity * 2);
    if (new_cap < required)
      new_cap = required;
    AllocChecker ac;
    char *new_data = new (ac) char[new_cap];
    if (LIBC_UNLIKELY(!ac))
      return false;
    if (len > 0)
      inline_memcpy(new_data, data, len);
    new_data[len] = '\0';
    delete[] data;
    data = new_data;
    capacity = new_cap;
    return true;
  }

  bool init(cpp::string_view sv) {
    while (sv.size() > 1 && sv.back() == '/')
      sv.remove_suffix(1);
    if (LIBC_UNLIKELY(sv.size() >= MAX_PATH_BUF_CAPACITY ||
                      !ensure_capacity(sv.size() + 1)))
      return false;
    inline_memcpy(data, sv.data(), sv.size());
    len = sv.size();
    data[len] = '\0';
    return true;
  }

  bool push_child(cpp::string_view child_name) {
    bool need_slash = (len > 0 && data[len - 1] != '/');
    size_t slash_len = need_slash ? 1 : 0;
    if (LIBC_UNLIKELY(child_name.size() >=
                      MAX_PATH_BUF_CAPACITY - len - slash_len))
      return false;
    size_t add_len = slash_len + child_name.size();
    if (LIBC_UNLIKELY(!ensure_capacity(len + add_len + 1)))
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

// Contiguous buffer of null-terminated entry names used when available_fds == 1
// requires closing the current directory before recursing into children.
struct DirEntryNames {
  char *buf = nullptr;
  size_t used = 0;
  size_t cap = 0;

  DirEntryNames() = default;
  DirEntryNames(const DirEntryNames &) = delete;
  DirEntryNames &operator=(const DirEntryNames &) = delete;
  LIBC_INLINE ~DirEntryNames() noexcept { delete[] buf; }

  bool append(cpp::string_view name) {
    constexpr size_t MAX_SIZE = cpp::numeric_limits<size_t>::max();
    if (LIBC_UNLIKELY(name.size() >= MAX_SIZE - used))
      return false;
    size_t need = used + name.size() + 1;
    if (LIBC_UNLIKELY(need > cap)) {
      size_t new_cap = cap == 0 ? INITIAL_NAMES_CAPACITY
                                : (cap > MAX_SIZE / 2 ? MAX_SIZE : cap * 2);
      if (new_cap < need)
        new_cap = need;
      AllocChecker ac;
      char *new_buf = new (ac) char[new_cap];
      if (LIBC_UNLIKELY(!ac))
        return false;
      if (used > 0)
        inline_memcpy(new_buf, buf, used);
      delete[] buf;
      buf = new_buf;
      cap = new_cap;
    }
    inline_memcpy(buf + used, name.data(), name.size());
    used += name.size();
    buf[used++] = '\0';
    return true;
  }
};

struct OriginCwd {
  int fd = -1;
  bool dirty = false;

  OriginCwd() = default;
  OriginCwd(const OriginCwd &) = delete;
  OriginCwd &operator=(const OriginCwd &) = delete;

  LIBC_INLINE ~OriginCwd() noexcept {
    if (dirty && fd >= 0)
      linux_syscalls::fchdir(fd);
    if (fd >= 0)
      linux_syscalls::close(fd);
  }

  ErrorOr<int> capture(int fd_limit, int &available_fds) {
    ErrorOr<int> fd_or_err =
        linux_syscalls::open(".", O_PATH | O_DIRECTORY | O_CLOEXEC, 0);
    if (!fd_or_err.has_value())
      return Error(fd_or_err.error());
    fd = fd_or_err.value();
    available_fds = (fd_limit > 1) ? (fd_limit - 1) : 1;
    return 0;
  }

  ErrorOr<int> restore() {
    if (fd < 0)
      return 0;
    ErrorOr<int> res = linux_syscalls::fchdir(fd);
    if (res.has_value())
      dirty = false;
    return res;
  }

  ErrorOr<int> chdir_to_walk_path(const char *walk_path) {
    if (walk_path[0] != '/') {
      ErrorOr<int> res = restore();
      if (!res.has_value())
        return res;
    }
    ErrorOr<int> res = linux_syscalls::chdir(walk_path);
    if (res.has_value())
      dirty = true;
    return res;
  }
};

struct WalkContext {
  const char *raw_root_path;
  PathBuffer path_buf;
  NftwCallback fn;
  int flags;
  dev_t start_device = 0;
  OriginCwd origin_cwd;
  VisitedDirs visited_dirs;

  ErrorOr<int> chdir_to_parent_of(int base) {
    if (base == 0)
      return origin_cwd.restore();
    if (base == 1) {
      ErrorOr<int> res = linux_syscalls::chdir("/");
      if (res.has_value())
        origin_cwd.dirty = true;
      return res;
    }
    char saved = path_buf.data[base - 1];
    path_buf.data[base - 1] = '\0';
    ErrorOr<int> res = origin_cwd.chdir_to_walk_path(path_buf.data);
    path_buf.data[base - 1] = saved;
    return res;
  }
};

ErrorOr<int> walk_tree(WalkContext &ctx, int available_fds, int base, int level,
                       int parent_fd, const Node *history) {
  struct FTW ftw_buf = {base, level};

  // Preserve the caller's raw dir_path at level 0 so trailing slashes on files
  // or symlinks are resolved by the kernel according to POSIX rules; at
  // level > 0, resolve single-component child names against parent_fd (or
  // AT_FDCWD when FTW_CHDIR is active) whenever available.
  int lookup_dfd = (level > 0 && parent_fd >= 0) ? parent_fd : AT_FDCWD;
  const char *lookup_path = (level == 0)
                                ? ctx.raw_root_path
                                : ((parent_fd >= 0 || (ctx.flags & FTW_CHDIR))
                                       ? ctx.path_buf.data + ftw_buf.base
                                       : ctx.path_buf.data);

  struct stat stat_buf = {};
  int type_flag = FTW_NS;
  int stat_flags = (ctx.flags & FTW_PHYS) ? AT_SYMLINK_NOFOLLOW : 0;
  ErrorOr<void> stat_res =
      internal::stat_via_statx(lookup_dfd, lookup_path, stat_flags, &stat_buf);
  if (stat_res.has_value()) {
    if (S_ISDIR(stat_buf.st_mode))
      type_flag = FTW_D;
    else if (S_ISLNK(stat_buf.st_mode))
      type_flag = FTW_SL;
    else
      type_flag = FTW_F;
  } else {
    int err = stat_res.error();
    if (!(ctx.flags & FTW_PHYS) && (err == ENOENT || err == ENOTDIR)) {
      ErrorOr<void> lstat_res = internal::stat_via_statx(
          lookup_dfd, lookup_path, AT_SYMLINK_NOFOLLOW, &stat_buf);
      if (lstat_res.has_value() && S_ISLNK(stat_buf.st_mode))
        type_flag = FTW_SLN;
      else if (level > 0 && err == ENOENT)
        type_flag = FTW_NS;
      else
        return Error(err);
    } else if (level > 0 && (err == EACCES || err == ENOENT)) {
      type_flag = FTW_NS;
    } else {
      return Error(err);
    }
  }

  if (type_flag != FTW_NS) {
    if (level == 0)
      ctx.start_device = stat_buf.st_dev;
    else if ((ctx.flags & FTW_MOUNT) && stat_buf.st_dev != ctx.start_device)
      return 0;
  }

  auto normalize_ret = [&](int res) -> int {
    if (!(ctx.flags & FTW_ACTIONRETVAL))
      return res;
    if (res == FTW_CONTINUE || res == FTW_SKIP_SUBTREE)
      return 0;
    if (res == FTW_SKIP_SIBLINGS)
      return level == 0 ? 0 : FTW_SKIP_SIBLINGS;
    return res;
  };

  if (type_flag != FTW_D) {
    if ((ctx.flags & FTW_CHDIR) && level == 0 && ftw_buf.base > 0) {
      ErrorOr<int> cwd_res = ctx.chdir_to_parent_of(ftw_buf.base);
      if (!cwd_res.has_value())
        return Error(cwd_res.error());
    }
    return normalize_ret(
        ctx.fn(ctx.path_buf.data, &stat_buf, type_flag, &ftw_buf));
  }

  // Always check active ancestors (preventing both symlink and bind-mount loops
  // even under FTW_PHYS), and in logical mode skip any directory already
  // visited so no directory is reported twice per POSIX.1-2024.
  for (const Node *curr = history; curr != nullptr; curr = curr->parent) {
    if (curr->ino == stat_buf.st_ino && curr->dev == stat_buf.st_dev)
      return 0;
  }
  if (!(ctx.flags & FTW_PHYS)) {
    if (ctx.visited_dirs.contains(stat_buf.st_dev, stat_buf.st_ino))
      return 0;
    if (LIBC_UNLIKELY(
            !ctx.visited_dirs.insert(stat_buf.st_dev, stat_buf.st_ino)))
      return Error(ENOMEM);
  }

  int open_flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
  if (ctx.flags & FTW_PHYS)
    open_flags |= O_NOFOLLOW;
  ErrorOr<int> dir_fd_or_err =
      linux_syscalls::openat(lookup_dfd, lookup_path, open_flags, 0);
  if (!dir_fd_or_err.has_value()) {
    int err = dir_fd_or_err.error();
    if (level > 0 &&
        (err == EACCES || err == ENOENT || err == ENOTDIR || err == ELOOP)) {
      int fail_type = (err == EACCES) ? FTW_DNR : FTW_NS;
      return normalize_ret(
          ctx.fn(ctx.path_buf.data, &stat_buf, fail_type, &ftw_buf));
    }
    return Error(err);
  }
  int cur_dir_fd = dir_fd_or_err.value();

  // Verify the opened directory matches the stat_buf inode to prevent TOCTOU
  // directory-replacement races before reading or fchdir'ing into it.
  struct stat opened_stat = {};
  ErrorOr<void> verify_res =
      internal::stat_via_statx(cur_dir_fd, "", AT_EMPTY_PATH, &opened_stat);
  if (!verify_res.has_value() || opened_stat.st_dev != stat_buf.st_dev ||
      opened_stat.st_ino != stat_buf.st_ino) {
    linux_syscalls::close(cur_dir_fd);
    if (level == 0)
      return Error(verify_res.has_value() ? ENOENT : verify_res.error());
    return normalize_ret(
        ctx.fn(ctx.path_buf.data, &stat_buf, FTW_NS, &ftw_buf));
  }

  ErrorOr<Dir *> dir_or_err = Dir::fdopen(cur_dir_fd);
  if (!dir_or_err.has_value()) {
    linux_syscalls::close(cur_dir_fd);
    return Error(dir_or_err.error());
  }
  ScopedDir open_dir(dir_or_err.value());

  if (!(ctx.flags & FTW_DEPTH)) {
    if ((ctx.flags & FTW_CHDIR) && level == 0 && ftw_buf.base > 0) {
      ErrorOr<int> cwd_res = ctx.chdir_to_parent_of(ftw_buf.base);
      if (!cwd_res.has_value())
        return Error(cwd_res.error());
    }
    int res = ctx.fn(ctx.path_buf.data, &stat_buf, FTW_D, &ftw_buf);
    if (res != 0)
      return normalize_ret(res);
  }

  if (ctx.flags & FTW_CHDIR) {
    ErrorOr<int> chdir_res = linux_syscalls::fchdir(cur_dir_fd);
    if (!chdir_res.has_value())
      return Error(chdir_res.error());
    ctx.origin_cwd.dirty = true;
  }

  Node current_node = {stat_buf.st_ino, stat_buf.st_dev, history};
  bool stop_siblings = false;
  auto process_child = [&](cpp::string_view name, int child_fds,
                           int child_parent_fd) -> ErrorOr<int> {
    size_t saved_len = ctx.path_buf.len;
    if (LIBC_UNLIKELY(!ctx.path_buf.push_child(name)))
      return Error(ENOMEM);
    int child_base = static_cast<int>(ctx.path_buf.len - name.size());
    ErrorOr<int> child_res = walk_tree(ctx, child_fds, child_base, level + 1,
                                       child_parent_fd, &current_node);
    ctx.path_buf.pop_to(saved_len);
    if (!child_res.has_value())
      return child_res;
    int val = child_res.value();
    if ((ctx.flags & FTW_ACTIONRETVAL) && val == FTW_SKIP_SIBLINGS) {
      stop_siblings = true;
      return 0;
    }
    return val;
  };

  if (available_fds > 1) {
    while (!stop_siblings) {
      ErrorOr<struct dirent *> entry_or_err = open_dir->read();
      if (!entry_or_err.has_value())
        return Error(entry_or_err.error());
      struct dirent *entry = entry_or_err.value();
      if (entry == nullptr)
        break;
      cpp::string_view name(entry->d_name);
      if (name == "." || name == "..")
        continue;
      ErrorOr<int> res = process_child(name, available_fds - 1, cur_dir_fd);
      if (!res.has_value() || res.value() != 0)
        return res;
    }
    if (int close_ret = open_dir.close(); close_ret != 0)
      return Error(close_ret);
  } else {
    DirEntryNames names;
    while (true) {
      ErrorOr<struct dirent *> entry_or_err = open_dir->read();
      if (!entry_or_err.has_value())
        return Error(entry_or_err.error());
      struct dirent *entry = entry_or_err.value();
      if (entry == nullptr)
        break;
      cpp::string_view name(entry->d_name);
      if (name == "." || name == "..")
        continue;
      if (LIBC_UNLIKELY(!names.append(name)))
        return Error(ENOMEM);
    }
    if (int close_ret = open_dir.close(); close_ret != 0)
      return Error(close_ret);

    for (size_t offset = 0; offset < names.used && !stop_siblings;) {
      cpp::string_view name(names.buf + offset);
      offset += name.size() + 1;
      ErrorOr<int> res = process_child(name, 1, -1);
      if (!res.has_value() || res.value() != 0)
        return res;
    }
  }

  if (ctx.flags & FTW_CHDIR) {
    ErrorOr<int> cwd_res = (level > 0 && parent_fd >= 0)
                               ? linux_syscalls::fchdir(parent_fd)
                               : ((level > 0 || (ctx.flags & FTW_DEPTH))
                                      ? ctx.chdir_to_parent_of(ftw_buf.base)
                                      : ErrorOr<int>(0));
    if (!cwd_res.has_value())
      return Error(cwd_res.error());
  }

  if (ctx.flags & FTW_DEPTH)
    return normalize_ret(
        ctx.fn(ctx.path_buf.data, &stat_buf, FTW_DP, &ftw_buf));

  return 0;
}

ErrorOr<int> run_nftw(const char *dir_path, NftwCallback fn, int fd_limit,
                      int flags) {
  if (fd_limit <= 0 || (flags & ~VALID_FLAGS) != 0)
    return Error(EINVAL);
  if (dir_path[0] == '\0')
    return Error(ENOENT);

  WalkContext ctx{dir_path, {}, fn, flags, 0, {}, {}};
  if (LIBC_UNLIKELY(!ctx.path_buf.init(dir_path)))
    return Error(ENOMEM);

  cpp::string_view root_view(ctx.path_buf.data, ctx.path_buf.len);
  size_t slash_pos = root_view.find_last_of('/');
  int root_base = 0;
  if (slash_pos != cpp::string_view::npos && ctx.path_buf.len > 1)
    root_base = static_cast<int>(slash_pos + 1);

  int available_fds = fd_limit;
  if (flags & FTW_CHDIR) {
    ErrorOr<int> cwd_res = ctx.origin_cwd.capture(fd_limit, available_fds);
    if (!cwd_res.has_value())
      return Error(cwd_res.error());
  }

  ErrorOr<int> walk_res =
      walk_tree(ctx, available_fds, root_base, 0, -1, nullptr);
  if ((flags & FTW_CHDIR) && ctx.origin_cwd.dirty) {
    ErrorOr<int> restore_res = ctx.origin_cwd.restore();
    if (!restore_res.has_value() && walk_res.has_value())
      return Error(restore_res.error());
  }
  return walk_res;
}

} // namespace

LLVM_LIBC_FUNCTION(int, nftw,
                   (const char *dir_path,
                    int (*fn)(const char *, const struct stat *, int,
                              struct FTW *),
                    int fd_limit, int flags)) {
  LIBC_CRASH_ON_NULLPTR(dir_path);
  LIBC_CRASH_ON_NULLPTR(fn);
  ErrorOr<int> result = run_nftw(dir_path, fn, fd_limit, flags);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
