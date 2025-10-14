//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-localization
// UNSUPPORTED: no-threads
// UNSUPPORTED: no-filesystem

// <filesystem>

// Test for a time-of-check to time-of-use issue with std::filesystem::remove_all.
//
// Scenario:
// The attacker wants to get directory contents deleted, to which he does not have access.
// He has a way to get a privileged binary call `std::filesystem::remove_all()` on a
// directory he controls, e.g. in his home directory.
//
// The POC sets up the `attack_dest/attack_file` which the attacker wants to have deleted.
// The attacker repeatedly creates a directory and replaces it with a symlink from
// `victim_del` to `attack_dest` while the victim code calls `std::filesystem::remove_all()`
// on `victim_del`. After a few seconds the attack has succeeded and
// `attack_dest/attack_file` is deleted.
//
// This is taken from https://github.com/rust-lang/wg-security-response/blob/master/patches/CVE-2022-21658/0002-Fix-CVE-2022-21658-for-UNIX-like.patch

// This test requires a dylib containing the fix shipped in https://reviews.llvm.org/D118134 (4f67a909902d).
// We use UNSUPPORTED instead of XFAIL because the test might not fail reliably.
// UNSUPPORTED: using-built-library-before-llvm-14

// Windows doesn't support the necessary APIs to mitigate this issue.
// XFAIL: target={{.+}}-windows-{{.+}}

#include <cstdio>
#include <filesystem>
#include <system_error>
#include <thread>

#include <filesystem>
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;

int main(int, char**) {
  scoped_test_env env;
  fs::path const tmpdir = env.create_dir("mydir");
  fs::path const victim_del_path = tmpdir / "victim_del";
  fs::path const attack_dest_dir = env.create_dir(tmpdir / "attack_dest");
  fs::path const attack_dest_file = env.create_file(attack_dest_dir / "attack_file", 42);

  // victim just continuously removes `victim_del`
  bool stop = false;
  std::thread t{[&]() {
    while (!stop) {
        std::error_code ec;
        fs::remove_all(victim_del_path, ec); // ignore any error
    }
  }};

  // attacker (could of course be in a separate process)
  auto start_time = std::chrono::system_clock::now();
  auto elapsed_since = [](std::chrono::system_clock::time_point const& time_point) {
      return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - time_point);
  };
  bool attack_succeeded = false;
  while (elapsed_since(start_time) < std::chrono::seconds(5)) {
    if (!fs::exists(attack_dest_file)) {
      std::printf("Victim deleted symlinked file outside of victim_del. Attack succeeded in %lld seconds.\n",
                  elapsed_since(start_time).count());
      attack_succeeded = true;
      break;
    }
    std::error_code ec;
    fs::create_directory(victim_del_path, ec);
    if (ec) {
      continue;
    }

    fs::remove(victim_del_path);
    fs::create_directory_symlink(attack_dest_dir, victim_del_path);
  }
  stop = true;
  t.join();

  return attack_succeeded ? 1 : 0;
}
