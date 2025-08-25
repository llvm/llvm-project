//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-create-symlinks
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// recursive_directory_iterator

#include <filesystem>
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;

#if defined(_WIN32)
static void set_last_write_time_in_iteration(const fs::path& dir) {
  // Windows can postpone updating last write time for file especially for
  // directory because last write time of directory depends of its childs.
  // See
  // https://learn.microsoft.com/en-us/windows/win32/sysinfo/file-times
  // To force updating file entries calls "last_write_time" with own value.
  const fs::recursive_directory_iterator end_it{};

  std::error_code ec;
  fs::recursive_directory_iterator it(dir, ec);
  assert(!ec);

  fs::file_time_type now_time = fs::file_time_type::clock::now();
  for (; it != end_it; ++it) {
    const fs::path entry = *it;
    fs::last_write_time(entry, now_time, ec);
    assert(!ec);
  }

  assert(it == end_it);
}

struct directory_entry_and_values {
  fs::directory_entry entry;

  fs::file_status symlink_status;
  fs::file_status status;
  std::uintmax_t file_size;
  fs::file_time_type last_write_time;
};

std::vector<directory_entry_and_values>
get_directory_entries_for(const fs::path& dir, const std::set<fs::path>& dir_contents) {
  const fs::recursive_directory_iterator end_it{};

  std::error_code ec;
  fs::recursive_directory_iterator it(dir, ec);
  assert(!ec);

  std::vector<directory_entry_and_values> dir_entries;
  std::set<fs::path> unseen_entries = dir_contents;
  while (!unseen_entries.empty()) {
    assert(it != end_it);
    const fs::directory_entry& entry = *it;

    assert(unseen_entries.erase(entry.path()) == 1);

    dir_entries.push_back(directory_entry_and_values{
        .entry           = entry,
        .symlink_status  = entry.symlink_status(),
        .status          = entry.status(),
        .file_size       = entry.is_regular_file() ? entry.file_size() : 0,
        .last_write_time = entry.last_write_time()});

    fs::recursive_directory_iterator& it_ref = it.increment(ec);
    assert(!ec);
    assert(&it_ref == &it);
  }
  return dir_entries;
}
#endif // _WIN32

// Checks that the directory_entry properties will be the same before and after
// calling "refresh" in case of iteration.
// In case of Windows expects that directory_entry caches the properties during
// iteration.
static void test_cache_and_refresh_in_iteration() {
  static_test_env static_env;
  const fs::path test_dir = static_env.Dir;
#if defined(_WIN32)
  set_last_write_time_in_iteration(test_dir);
#endif
  const std::set<fs::path> dir_contents(static_env.RecDirIterationList.begin(), static_env.RecDirIterationList.end());
  const fs::recursive_directory_iterator end_it{};

  std::error_code ec;
  fs::recursive_directory_iterator it(test_dir, ec);
  assert(!ec);

  std::set<fs::path> unseen_entries = dir_contents;
  while (!unseen_entries.empty()) {
    assert(it != end_it);
    const fs::directory_entry& entry = *it;

    assert(unseen_entries.erase(entry.path()) == 1);

    fs::file_status symlink_status     = entry.symlink_status();
    fs::file_status status             = entry.status();
    std::uintmax_t file_size           = entry.is_regular_file() ? entry.file_size() : 0;
    fs::file_time_type last_write_time = entry.last_write_time();

    fs::directory_entry mutable_entry = *it;
    mutable_entry.refresh();
    fs::file_status upd_symlink_status     = mutable_entry.symlink_status();
    fs::file_status upd_status             = mutable_entry.status();
    std::uintmax_t upd_file_size           = mutable_entry.is_regular_file() ? mutable_entry.file_size() : 0;
    fs::file_time_type upd_last_write_time = mutable_entry.last_write_time();
    assert(upd_symlink_status.type() == symlink_status.type() &&
           upd_symlink_status.permissions() == symlink_status.permissions());
    assert(upd_status.type() == status.type() && upd_status.permissions() == status.permissions());
    assert(upd_file_size == file_size);
    assert(upd_last_write_time == last_write_time);

    fs::recursive_directory_iterator& it_ref = it.increment(ec);
    assert(!ec);
    assert(&it_ref == &it);
  }
}

#if defined(_WIN32)
// In case of Windows expects that the directory_entry caches the properties
// during iteration and the properties don't change after deleting folders
// and files.
static void test_cached_values_in_iteration() {
  std::vector<directory_entry_and_values> dir_entries;
  {
    static_test_env static_env;
    const fs::path testDir = static_env.Dir;
    set_last_write_time_in_iteration(testDir);
    const std::set<fs::path> dir_contents(static_env.RecDirIterationList.begin(), static_env.RecDirIterationList.end());
    dir_entries = get_directory_entries_for(testDir, dir_contents);
  }
  // Testing folder should be deleted after destroying static_test_env.

  for (const auto& dir_entry : dir_entries) {
    // During iteration Windows provides information only about symlink itself
    // not about file/folder which symlink points to.
    if (dir_entry.entry.is_symlink()) {
      // Check that symlink is not using cached value about existing file.
      assert(!dir_entry.entry.exists());
    } else {
      // Check that entry uses cached value about existing file.
      assert(dir_entry.entry.exists());
    }
    fs::file_status symlink_status = dir_entry.entry.symlink_status();
    assert(dir_entry.symlink_status.type() == symlink_status.type() &&
           dir_entry.symlink_status.permissions() == symlink_status.permissions());

    if (!dir_entry.entry.is_symlink()) {
      fs::file_status status = dir_entry.entry.status();
      assert(dir_entry.status.type() == status.type() && dir_entry.status.permissions() == status.permissions());

      std::uintmax_t file_size = dir_entry.entry.is_regular_file() ? dir_entry.entry.file_size() : 0;
      assert(dir_entry.file_size == file_size);

      fs::file_time_type last_write_time = dir_entry.entry.last_write_time();
      assert(dir_entry.last_write_time == last_write_time);
    }
  }
}
#endif // _WIN32

int main(int, char**) {
  test_cache_and_refresh_in_iteration();
#if defined(_WIN32)
  test_cached_values_in_iteration();
#endif

  return 0;
}
