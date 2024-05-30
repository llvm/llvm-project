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
using namespace fs;

#if defined(_WIN32)
static void set_last_write_time_in_iteration(const fs::path& dir)
{
    // Windows can postpone updating last write time for file especially for
    // directory because last write time of directory depends of its childs.
    // See
    // https://learn.microsoft.com/en-us/windows/win32/sysinfo/file-times
    // To force updating file entries calls "last_write_time" with own value.
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(dir, ec);
    assert(!ec);

    file_time_type now_time = file_time_type::clock::now();
    for ( ; it != endIt; ++it) {
        const path entry = *it;
        last_write_time(entry, now_time, ec);
        assert(!ec);
    }

    assert(it == endIt);
}
#endif 

static void test_cache_and_refresh_in_iteration()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
#if defined(_WIN32)    
    set_last_write_time_in_iteration(testDir);
#endif 
    const std::set<path> dir_contents(static_env.RecDirIterationList.begin(),
                                      static_env.RecDirIterationList.end());
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    assert(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        assert(it != endIt);
        const directory_entry& entry = *it;

        assert(unseen_entries.erase(entry.path()) == 1);

        file_status symlink_status     = entry.symlink_status();
        file_status status             = entry.status();
        std::uintmax_t file_size       = entry.is_regular_file() ? entry.file_size() : 0;
        file_time_type last_write_time = entry.last_write_time();

        directory_entry mutable_entry = *it;
        mutable_entry.refresh();
        file_status upd_symlink_status     = mutable_entry.symlink_status();
        file_status upd_status             = mutable_entry.status();
        std::uintmax_t upd_file_size       = mutable_entry.is_regular_file() ? mutable_entry.file_size() : 0;
        file_time_type upd_last_write_time = mutable_entry.last_write_time();
        assert(upd_symlink_status == symlink_status);
        assert(upd_status == status);
        assert(upd_file_size == file_size);
        assert(upd_last_write_time == last_write_time);

        recursive_directory_iterator& it_ref = it.increment(ec);
        assert(!ec);
        assert(&it_ref == &it);
    }
}

int main(int, char**) {
    test_cache_and_refresh_in_iteration();

    return 0;
}
