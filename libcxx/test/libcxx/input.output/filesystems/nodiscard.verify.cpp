//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// UNSUPPORTED: no-filesystem

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <filesystem>

// Check that functions are marked [[nodiscard]]

#include <filesystem>
#include <string>
#include <string_view>

#include "test_macros.h"

void test() {
  {
    const auto op = std::filesystem::copy_options::none;

    op & op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    op | op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    op ^ op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ~op;     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    const std::filesystem::directory_entry de;
    std::error_code ec;

    de.path(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    de.exists();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.exists(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    de.is_block_file();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_block_file(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_character_file();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_character_file(ec);

    de.is_directory();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_directory(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    de.is_fifo();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_fifo(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    de.is_other();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_other(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_regular_file();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_regular_file(ec);

    de.is_socket();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_socket(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    de.is_symlink();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.is_symlink(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    de.file_size();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.file_size(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.hard_link_count();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.hard_link_count(ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.last_write_time();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.last_write_time(ec);

    de.status();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.status(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    de.symlink_status();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    de.symlink_status(ec); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    const std::filesystem::directory_iterator di;

    *di; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::begin(di);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::end(di);
  }

  {
    const auto op = std::filesystem::directory_options::none;

    op & op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    op | op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    op ^ op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ~op;     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    const std::filesystem::file_status fs;

    fs.type();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    fs.permissions(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    std::error_code ec;
    const std::filesystem::filesystem_error fs("zmt", ec);

    fs.path1(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    fs.path2(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    fs.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    const std::filesystem::path p;
    std::error_code ec;
    const std::filesystem::file_status fs;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::absolute(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::absolute(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::canonical(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::canonical(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::current_path();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::current_path(ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::equivalent(p, p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::equivalent(p, p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::status_known(fs);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::exists(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::exists(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::exists(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::file_size(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::file_size(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::hard_link_count(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::hard_link_count(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_block_file(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_block_file(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_block_file(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_character_file(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_character_file(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_character_file(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_directory(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_directory(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_directory(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_empty(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_empty(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_fifo(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_fifo(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_fifo(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_regular_file(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_regular_file(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_regular_file(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_symlink(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_symlink(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_symlink(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_other(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_other(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_other(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_socket(fs);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_socket(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::is_socket(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::last_write_time(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::last_write_time(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::proximate(p, p, ec);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::proximate(p, ec);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::proximate(p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::read_symlink(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::read_symlink(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::relative(p, p, ec);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::relative(p, ec);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::relative(p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::space(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::space(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::status(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::status(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::symlink_status(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::symlink_status(p, ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::temp_directory_path();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::temp_directory_path(ec);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::weakly_canonical(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::weakly_canonical(p, ec);
  }

  {
    std::filesystem::path::iterator it;

    *it; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    std::filesystem::path p;
    const std::string src;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.native();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.c_str();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.string();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.u8string();
#if !defined(TEST_HAS_NO_LOCALIZATION)
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.string<char>();

#  if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.wstring();
#  endif
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.u16string();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.u32string();
#endif // !defined(TEST_HAS_NO_LOCALIZATION)

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.generic_string();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.generic_u8string();
#if !defined(TEST_HAS_NO_LOCALIZATION)
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.generic_string<char>();

#  if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.generic_wstring();
#  endif
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.generic_u16string();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.generic_u32string();
#endif // !defined(TEST_HAS_NO_LOCALIZATION)

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.compare(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.compare(std::string{});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.compare(std::string_view{});
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.compare("");

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.root_name();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.root_directory();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.root_path();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.relative_path();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.parent_path();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.filename();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.stem();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.empty();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.has_root_name();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.has_root_directory();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.has_root_path();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.has_relative_path();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.has_parent_path();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.has_filename();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.has_stem();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.has_extension();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.is_absolute();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.is_relative();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.lexically_normal();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.lexically_relative(p);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.lexically_proximate(p);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    p.end();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::hash_value(p);

    std::hash<std::filesystem::path> hash;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    hash(p);
  }

  {
    const auto op = std::filesystem::perm_options::add;

    op & op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    op | op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    op ^ op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ~op;     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    const auto op = std::filesystem::perms::all;

    op & op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    op | op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    op ^ op; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    ~op;     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

  {
    const std::filesystem::recursive_directory_iterator it;

    *it; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    it.options(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.depth();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.recursion_pending();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::begin(it);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::end(it);
  }

  {
    const std::string str;

#if !defined(TEST_HAS_NO_LOCALIZATION)
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::u8path(str.begin(), str.end());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::u8path(str.begin());
#endif
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::filesystem::u8path(str);
  }
}
