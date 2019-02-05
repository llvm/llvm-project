//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// XFAIL: availability=macosx10.13
// XFAIL: availability=macosx10.12
// XFAIL: availability=macosx10.11
// XFAIL: availability=macosx10.10
// XFAIL: availability=macosx10.9
// XFAIL: availability=macosx10.7
// XFAIL: availability=macosx10.8

// <optional>

// class bad_optional_access is default constructible

#include <optional>
#include <type_traits>

int main(int, char**)
{
    using std::bad_optional_access;
    bad_optional_access ex;

  return 0;
}
