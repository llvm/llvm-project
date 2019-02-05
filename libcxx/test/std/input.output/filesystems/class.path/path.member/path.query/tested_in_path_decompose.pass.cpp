//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>

// class path

//-------------------------------
// 8.4.10 path query [path.query]
//-------------------------------
// bool empty() const noexcept;
// bool has_root_path() const;
// bool has_root_name() const;
// bool has_root_directory() const;
// bool has_relative_path() const;
// bool has_parent_path() const;
// bool has_filename() const;
// bool has_stem() const;
// bool has_extension() const;
// bool is_absolute() const;
// bool is_relative() const;

// tested in path.decompose
int main(int, char**) {
  return 0;
}
