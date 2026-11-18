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
#include "hdr/limits_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/dev_t.h"
#include "hdr/types/ino_t.h"
#include "hdr/types/off_t.h"
#include "hdr/types/struct_FTW.h"
#include "hdr/types/struct_dirent.h"
#include "hdr/types/struct_stat.h"
#include "hdr/unistd_macros.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/unique_ptr.h"
#include "src/__support/File/dir.h"
#include "src/__support/OSUtil/linux/stat/stat_via_statx.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/chdir.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fchdir.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/lseek.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/openat.h"
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

constexpr size_t INITIAL_FRAME_CAPACITY = 16;
constexpr size_t INITIAL_VISITED_CAPACITY = 32;

// Bound path storage to PATH_MAX when defined, falling back to a fixed 16 KiB
// cap on POSIX systems where PATH_MAX is omitted as indeterminate.
#ifdef PATH_MAX
constexpr size_t MAX_PATH_LEN = PATH_MAX;
#else
constexpr size_t MAX_PATH_LEN = 16384;
#endif

constexpr int VALID_FLAGS =
    FTW_PHYS | FTW_MOUNT | FTW_CHDIR | FTW_DEPTH | FTW_ACTIONRETVAL;

template <typename T, size_t InitialCap> struct CheckedBuffer {
  cpp::unique_ptr<T[]> data;
  size_t size = 0;
  size_t capacity = 0;

  bool reserve(size_t required) {
    constexpr size_t MAX_CAP = cpp::numeric_limits<size_t>::max() / sizeof(T);
    if (required <= capacity)
      return true;
    if (LIBC_UNLIKELY(required > MAX_CAP))
      return false;
    size_t new_cap = capacity == 0
                         ? InitialCap
                         : (capacity > MAX_CAP / 2 ? MAX_CAP : capacity * 2);
    if (new_cap < required)
      new_cap = required;
    AllocChecker ac;
    T *new_data = new (ac) T[new_cap];
    if (LIBC_UNLIKELY(!ac))
      return false;
    if (size > 0)
      inline_memcpy(new_data, data.get(), size * sizeof(T));
    data.reset(new_data);
    capacity = new_cap;
    return true;
  }
};

struct DirId {
  dev_t dev;
  ino_t ino;

  LIBC_INLINE bool operator==(const DirId &other) const {
    return dev == other.dev && ino == other.ino;
  }
};

struct VisitedDirs {
  CheckedBuffer<DirId, INITIAL_VISITED_CAPACITY> entries;

  bool contains(DirId id) const {
    for (size_t i = 0; i < entries.size; ++i)
      if (entries.data[i] == id)
        return true;
    return false;
  }

  bool insert(DirId id) {
    if (LIBC_UNLIKELY(!entries.reserve(entries.size + 1)))
      return false;
    entries.data[entries.size++] = id;
    return true;
  }
};

struct PathBuffer {
  char buf[MAX_PATH_LEN];
  size_t size = 0;

  LIBC_INLINE char *c_str() { return buf; }
  LIBC_INLINE size_t len() const { return size; }

  bool init(cpp::string_view sv) {
    if (LIBC_UNLIKELY(sv.size() >= MAX_PATH_LEN))
      return false;
    while (sv.size() > 1 && sv.back() == '/')
      sv.remove_suffix(1);
    inline_memcpy(buf, sv.data(), sv.size());
    size = sv.size();
    buf[size] = '\0';
    return true;
  }

  bool push_child(cpp::string_view child_name) {
    bool need_slash = (size > 0 && buf[size - 1] != '/');
    size_t slash_len = need_slash ? 1 : 0;
    if (LIBC_UNLIKELY(child_name.size() >= MAX_PATH_LEN - size - slash_len))
      return false;
    if (need_slash)
      buf[size++] = '/';
    inline_memcpy(buf + size, child_name.data(), child_name.size());
    size += child_name.size();
    buf[size] = '\0';
    return true;
  }

  void pop_to(size_t prev_len) {
    size = prev_len;
    buf[prev_len] = '\0';
  }
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
  LIBC_INLINE Dir *operator->() const { return dir; }
  LIBC_INLINE Dir *release() {
    Dir *d = dir;
    dir = nullptr;
    return d;
  }
  int close() {
    if (dir == nullptr)
      return 0;
    Dir *d = dir;
    dir = nullptr;
    return d->close();
  }
};

struct DirFrame {
  Dir *dir;
  off_t resume_offset;
  bool has_resume_offset;
  DirId id;
  size_t path_len;
  int base;
  int level;
  struct stat stat_buf;
};

struct FrameStack {
  CheckedBuffer<DirFrame, INITIAL_FRAME_CAPACITY> frames;
  int open_dirs = 0;

  FrameStack() = default;
  FrameStack(const FrameStack &) = delete;
  FrameStack &operator=(const FrameStack &) = delete;

  LIBC_INLINE ~FrameStack() noexcept {
    for (size_t i = frames.size; i > 0; --i) {
      if (frames.data[i - 1].dir != nullptr)
        static_cast<void>(frames.data[i - 1].dir->close());
    }
  }

  LIBC_INLINE bool empty() const { return frames.size == 0; }
  LIBC_INLINE size_t size() const { return frames.size; }
  LIBC_INLINE DirFrame &back() { return frames.data[frames.size - 1]; }
  LIBC_INLINE DirFrame &operator[](size_t idx) { return frames.data[idx]; }

  bool has_ancestor(DirId id) const {
    for (size_t i = 0; i < frames.size; ++i)
      if (frames.data[i].id == id)
        return true;
    return false;
  }

  bool push(const DirFrame &frame) {
    if (LIBC_UNLIKELY(!frames.reserve(frames.size + 1)))
      return false;
    frames.data[frames.size++] = frame;
    if (frame.dir != nullptr)
      ++open_dirs;
    return true;
  }

  ErrorOr<int> evict_oldest_open() {
    for (size_t i = 0; i < frames.size; ++i) {
      if (frames.data[i].dir != nullptr) {
        int close_err = frames.data[i].dir->close();
        if (close_err != 0)
          return Error(close_err);
        frames.data[i].dir = nullptr;
        --open_dirs;
        return 0;
      }
    }
    return 0;
  }

  ErrorOr<int> close_and_pop() {
    DirFrame &top = back();
    if (top.dir != nullptr) {
      int close_err = top.dir->close();
      top.dir = nullptr;
      --open_dirs;
      if (close_err != 0) {
        --frames.size;
        return Error(close_err);
      }
    }
    --frames.size;
    return 0;
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

  LIBC_INLINE int root_dfd() const { return fd >= 0 ? fd : AT_FDCWD; }

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
  int available_fds;
  dev_t start_device = 0;
  OriginCwd origin_cwd;
  VisitedDirs visited_dirs;

  WalkContext(const char *root, NftwCallback cb, int fl, int fds)
      : raw_root_path(root), fn(cb), flags(fl), available_fds(fds) {}

  ErrorOr<int> chdir_to_parent_of(int base) {
    if (base == 0)
      return origin_cwd.restore();
    if (base == 1) {
      ErrorOr<int> res = linux_syscalls::chdir("/");
      if (res.has_value())
        origin_cwd.dirty = true;
      return res;
    }
    char saved = path_buf.c_str()[base - 1];
    path_buf.c_str()[base - 1] = '\0';
    ErrorOr<int> res = origin_cwd.chdir_to_walk_path(path_buf.c_str());
    path_buf.c_str()[base - 1] = saved;
    return res;
  }
};

enum class WalkAction { Continue, SkipSubtree, SkipSiblings, Stop };

struct CallbackOutcome {
  WalkAction action;
  int stop_value;
};

CallbackOutcome evaluate_callback(int flags, int res) {
  if (!(flags & FTW_ACTIONRETVAL))
    return res == 0 ? CallbackOutcome{WalkAction::Continue, 0}
                    : CallbackOutcome{WalkAction::Stop, res};
  switch (res) {
  case FTW_CONTINUE:
    return {WalkAction::Continue, 0};
  case FTW_SKIP_SUBTREE:
    return {WalkAction::SkipSubtree, 0};
  case FTW_SKIP_SIBLINGS:
    return {WalkAction::SkipSiblings, 0};
  default:
    return {WalkAction::Stop, res};
  }
}

ErrorOr<int> classify_entry(WalkContext &ctx, int lookup_dfd,
                            const char *lookup_path, int level,
                            struct stat &stat_buf) {
  int stat_flags = (ctx.flags & FTW_PHYS) ? AT_SYMLINK_NOFOLLOW : 0;
  ErrorOr<void> stat_res =
      internal::stat_via_statx(lookup_dfd, lookup_path, stat_flags, &stat_buf);
  if (stat_res.has_value()) {
    if (S_ISDIR(stat_buf.st_mode))
      return FTW_D;
    if (S_ISLNK(stat_buf.st_mode))
      return FTW_SL;
    return FTW_F;
  }

  int err = stat_res.error();
  if (!(ctx.flags & FTW_PHYS) && (err == ENOENT || err == ENOTDIR)) {
    ErrorOr<void> lstat_res = internal::stat_via_statx(
        lookup_dfd, lookup_path, AT_SYMLINK_NOFOLLOW, &stat_buf);
    if (lstat_res.has_value() && S_ISLNK(stat_buf.st_mode))
      return FTW_SLN;
    if (level > 0 && err == ENOENT)
      return FTW_NS;
    return Error(err);
  }
  if (level > 0 && (err == EACCES || err == ENOENT))
    return FTW_NS;
  return Error(err);
}

struct OpenDirResult {
  Dir *dir;
  int report_flag;
};

ErrorOr<OpenDirResult> open_verified_dir(int lookup_dfd,
                                         const char *lookup_path, int flags,
                                         int level, DirId expected_id) {
  int open_flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
  if (flags & FTW_PHYS)
    open_flags |= O_NOFOLLOW;

  ErrorOr<int> fd_or_err =
      linux_syscalls::openat(lookup_dfd, lookup_path, open_flags, 0);
  if (!fd_or_err.has_value()) {
    int err = fd_or_err.error();
    if (level > 0 &&
        (err == EACCES || err == ENOENT || err == ENOTDIR || err == ELOOP))
      return OpenDirResult{nullptr, (err == EACCES) ? FTW_DNR : FTW_NS};
    return Error(err);
  }
  int fd = fd_or_err.value();

  // Verify the opened directory matches expected_id to prevent TOCTOU
  // directory-replacement races before reading or fchdir'ing into it.
  struct stat opened_stat = {};
  ErrorOr<void> verify_res =
      internal::stat_via_statx(fd, "", AT_EMPTY_PATH, &opened_stat);
  if (!verify_res.has_value() ||
      !(DirId{opened_stat.st_dev, opened_stat.st_ino} == expected_id)) {
    linux_syscalls::close(fd);
    if (level == 0)
      return Error(verify_res.has_value() ? ENOENT : verify_res.error());
    return OpenDirResult{nullptr, FTW_NS};
  }

  ErrorOr<Dir *> dir_or_err = Dir::fdopen(fd);
  if (!dir_or_err.has_value()) {
    linux_syscalls::close(fd);
    return Error(dir_or_err.error());
  }
  return OpenDirResult{dir_or_err.value(), 0};
}

ErrorOr<int> reopen_frame_dir(WalkContext &ctx, FrameStack &frames,
                              size_t frame_idx) {
  if (frames.open_dirs >= ctx.available_fds) {
    ErrorOr<int> evict_res = frames.evict_oldest_open();
    if (!evict_res.has_value())
      return evict_res;
  }

  DirFrame &frame = frames[frame_idx];
  int open_flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
  if (ctx.flags & FTW_PHYS)
    open_flags |= O_NOFOLLOW;

  ErrorOr<int> fd_or_err(0);
  if (frame_idx == 0) {
    fd_or_err = linux_syscalls::openat(ctx.origin_cwd.root_dfd(),
                                       ctx.raw_root_path, open_flags, 0);
  } else {
    char saved = ctx.path_buf.c_str()[frame.path_len];
    ctx.path_buf.c_str()[frame.path_len] = '\0';
    fd_or_err = linux_syscalls::openat(ctx.origin_cwd.root_dfd(),
                                       ctx.path_buf.c_str(), open_flags, 0);
    ctx.path_buf.c_str()[frame.path_len] = saved;
  }
  if (!fd_or_err.has_value())
    return Error(fd_or_err.error());
  int fd = fd_or_err.value();

  struct stat reopened_stat = {};
  ErrorOr<void> verify_res =
      internal::stat_via_statx(fd, "", AT_EMPTY_PATH, &reopened_stat);
  if (!verify_res.has_value() ||
      !(DirId{reopened_stat.st_dev, reopened_stat.st_ino} == frame.id)) {
    linux_syscalls::close(fd);
    return Error(verify_res.has_value() ? ENOENT : verify_res.error());
  }

  if (frame.has_resume_offset) {
    ErrorOr<off_t> seek_res =
        linux_syscalls::lseek(fd, frame.resume_offset, SEEK_SET);
    if (!seek_res.has_value()) {
      linux_syscalls::close(fd);
      return Error(seek_res.error());
    }
  }

  ErrorOr<Dir *> dir_or_err = Dir::fdopen(fd);
  if (!dir_or_err.has_value()) {
    linux_syscalls::close(fd);
    return Error(dir_or_err.error());
  }
  frame.dir = dir_or_err.value();
  ++frames.open_dirs;
  return 0;
}

ErrorOr<CallbackOutcome> visit_entry(WalkContext &ctx, FrameStack &frames,
                                     int base, int level) {
  struct FTW ftw_buf = {base, level};

  // Preserve the caller's raw dir_path at level 0 so trailing slashes on files
  // or symlinks are resolved by the kernel according to POSIX rules; at
  // level > 0, parent's Dir is guaranteed open here, so resolve the child's
  // single-component name directly against parent_fd.
  int lookup_dfd = (level == 0) ? AT_FDCWD : frames.back().dir->getfd();
  const char *lookup_path =
      (level == 0) ? ctx.raw_root_path : (ctx.path_buf.c_str() + base);

  struct stat stat_buf = {};
  ErrorOr<int> type_or_err =
      classify_entry(ctx, lookup_dfd, lookup_path, level, stat_buf);
  if (!type_or_err.has_value())
    return Error(type_or_err.error());
  int type_flag = type_or_err.value();

  if (type_flag != FTW_NS) {
    if (level == 0)
      ctx.start_device = stat_buf.st_dev;
    else if ((ctx.flags & FTW_MOUNT) && stat_buf.st_dev != ctx.start_device)
      return CallbackOutcome{WalkAction::Continue, 0};
  }

  if (type_flag != FTW_D) {
    if ((ctx.flags & FTW_CHDIR) && level == 0 && base > 0) {
      ErrorOr<int> cwd_res = ctx.chdir_to_parent_of(base);
      if (!cwd_res.has_value())
        return Error(cwd_res.error());
    }
    return evaluate_callback(ctx.flags, ctx.fn(ctx.path_buf.c_str(), &stat_buf,
                                               type_flag, &ftw_buf));
  }

  // Always check active ancestors (preventing both symlink and bind-mount loops
  // even under FTW_PHYS), and in logical mode skip any directory already
  // visited so no directory is reported twice per POSIX.1-2024.
  DirId dir_id{stat_buf.st_dev, stat_buf.st_ino};
  if (frames.has_ancestor(dir_id))
    return CallbackOutcome{WalkAction::Continue, 0};
  if (!(ctx.flags & FTW_PHYS)) {
    if (ctx.visited_dirs.contains(dir_id))
      return CallbackOutcome{WalkAction::Continue, 0};
    if (LIBC_UNLIKELY(!ctx.visited_dirs.insert(dir_id)))
      return Error(ENOMEM);
  }
  if (LIBC_UNLIKELY(level == cpp::numeric_limits<int>::max()))
    return Error(EOVERFLOW);

  if (frames.open_dirs >= ctx.available_fds) {
    ErrorOr<int> evict_res = frames.evict_oldest_open();
    if (!evict_res.has_value())
      return Error(evict_res.error());
  }

  int open_dfd = AT_FDCWD;
  const char *open_path = ctx.raw_root_path;
  if (level > 0) {
    if (frames.back().dir != nullptr) {
      open_dfd = frames.back().dir->getfd();
      open_path = ctx.path_buf.c_str() + base;
    } else if (ctx.flags & FTW_CHDIR) {
      open_path = ctx.path_buf.c_str() + base;
    } else {
      open_path = ctx.path_buf.c_str();
    }
  }

  ErrorOr<OpenDirResult> open_res =
      open_verified_dir(open_dfd, open_path, ctx.flags, level, dir_id);
  if (!open_res.has_value())
    return Error(open_res.error());
  if (open_res->dir == nullptr) {
    return evaluate_callback(ctx.flags,
                             ctx.fn(ctx.path_buf.c_str(), &stat_buf,
                                    open_res->report_flag, &ftw_buf));
  }
  ScopedDir open_dir(open_res->dir);

  if (!(ctx.flags & FTW_DEPTH)) {
    if ((ctx.flags & FTW_CHDIR) && level == 0 && base > 0) {
      ErrorOr<int> cwd_res = ctx.chdir_to_parent_of(base);
      if (!cwd_res.has_value())
        return Error(cwd_res.error());
    }
    CallbackOutcome pre_outcome = evaluate_callback(
        ctx.flags, ctx.fn(ctx.path_buf.c_str(), &stat_buf, FTW_D, &ftw_buf));
    if (pre_outcome.action != WalkAction::Continue) {
      int close_err = open_dir.close();
      if (close_err != 0 && pre_outcome.action != WalkAction::Stop)
        return Error(close_err);
      return pre_outcome;
    }
  }

  if (ctx.flags & FTW_CHDIR) {
    ErrorOr<int> chdir_res = linux_syscalls::fchdir(open_dir->getfd());
    if (!chdir_res.has_value())
      return Error(chdir_res.error());
    ctx.origin_cwd.dirty = true;
  }

  DirFrame frame{open_dir.release(), 0,    false, dir_id,
                 ctx.path_buf.len(), base, level, stat_buf};
  if (LIBC_UNLIKELY(!frames.push(frame))) {
    static_cast<void>(frame.dir->close());
    return Error(ENOMEM);
  }
  return CallbackOutcome{WalkAction::Continue, 0};
}

ErrorOr<CallbackOutcome> finish_directory(WalkContext &ctx,
                                          FrameStack &frames) {
  int base = frames.back().base;
  int level = frames.back().level;
  struct stat stat_buf = frames.back().stat_buf;

  ErrorOr<int> pop_res = frames.close_and_pop();
  if (!pop_res.has_value())
    return Error(pop_res.error());

  if (ctx.flags & FTW_CHDIR) {
    if (!frames.empty()) {
      if (frames.back().dir == nullptr) {
        ErrorOr<int> reopen_res =
            reopen_frame_dir(ctx, frames, frames.size() - 1);
        if (!reopen_res.has_value())
          return Error(reopen_res.error());
      }
      ErrorOr<int> chdir_res =
          linux_syscalls::fchdir(frames.back().dir->getfd());
      if (!chdir_res.has_value())
        return Error(chdir_res.error());
    } else if (ctx.flags & FTW_DEPTH) {
      ErrorOr<int> cwd_res = ctx.chdir_to_parent_of(base);
      if (!cwd_res.has_value())
        return Error(cwd_res.error());
    }
  }

  CallbackOutcome outcome{WalkAction::Continue, 0};
  if (ctx.flags & FTW_DEPTH) {
    struct FTW ftw_buf = {base, level};
    outcome = evaluate_callback(
        ctx.flags, ctx.fn(ctx.path_buf.c_str(), &stat_buf, FTW_DP, &ftw_buf));
  }

  if (!frames.empty())
    ctx.path_buf.pop_to(frames.back().path_len);
  return outcome;
}

ErrorOr<int> walk_tree_iterative(WalkContext &ctx, int root_base) {
  FrameStack frames;
  ErrorOr<CallbackOutcome> root_res = visit_entry(ctx, frames, root_base, 0);
  if (!root_res.has_value())
    return Error(root_res.error());
  if (root_res->action == WalkAction::Stop)
    return root_res->stop_value;

  while (!frames.empty()) {
    if (frames.back().dir == nullptr) {
      ErrorOr<int> reopen_res =
          reopen_frame_dir(ctx, frames, frames.size() - 1);
      if (!reopen_res.has_value())
        return reopen_res;
    }

    DirFrame &top = frames.back();
    ErrorOr<struct dirent *> entry_or_err = top.dir->read();
    if (!entry_or_err.has_value())
      return Error(entry_or_err.error());
    struct dirent *entry = entry_or_err.value();

    bool finish_cur_dir = (entry == nullptr);
    if (!finish_cur_dir) {
      top.resume_offset = entry->d_off;
      top.has_resume_offset = true;
      cpp::string_view name(entry->d_name);
      if (name == "." || name == "..")
        continue;

      size_t parent_len = top.path_len;
      size_t depth_before = frames.size();
      int child_level = top.level + 1;
      if (LIBC_UNLIKELY(!ctx.path_buf.push_child(name)))
        return Error(ENAMETOOLONG);
      int child_base = static_cast<int>(ctx.path_buf.len() - name.size());

      ErrorOr<CallbackOutcome> child_res =
          visit_entry(ctx, frames, child_base, child_level);
      if (!child_res.has_value())
        return Error(child_res.error());

      if (frames.size() == depth_before)
        ctx.path_buf.pop_to(parent_len);

      if (child_res->action == WalkAction::Stop)
        return child_res->stop_value;
      if (child_res->action == WalkAction::SkipSiblings)
        finish_cur_dir = true;
    }

    if (finish_cur_dir) {
      while (true) {
        ErrorOr<CallbackOutcome> fin_res = finish_directory(ctx, frames);
        if (!fin_res.has_value())
          return Error(fin_res.error());
        if (fin_res->action == WalkAction::Stop)
          return fin_res->stop_value;
        if (fin_res->action == WalkAction::SkipSiblings && !frames.empty())
          continue;
        break;
      }
    }
  }

  return 0;
}

ErrorOr<int> run_nftw(const char *dir_path, NftwCallback fn, int fd_limit,
                      int flags) {
  if (fd_limit <= 0 || (flags & ~VALID_FLAGS) != 0)
    return Error(EINVAL);
  if (dir_path[0] == '\0')
    return Error(ENOENT);

  WalkContext ctx(dir_path, fn, flags, fd_limit);
  if (LIBC_UNLIKELY(!ctx.path_buf.init(dir_path)))
    return Error(ENAMETOOLONG);

  cpp::string_view root_view(ctx.path_buf.c_str(), ctx.path_buf.len());
  size_t slash_pos = root_view.find_last_of('/');
  int root_base = 0;
  if (slash_pos != cpp::string_view::npos && ctx.path_buf.len() > 1)
    root_base = static_cast<int>(slash_pos + 1);

  if (flags & FTW_CHDIR) {
    ErrorOr<int> cwd_res = ctx.origin_cwd.capture(fd_limit, ctx.available_fds);
    if (!cwd_res.has_value())
      return Error(cwd_res.error());
  }

  ErrorOr<int> walk_res = walk_tree_iterative(ctx, root_base);
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
