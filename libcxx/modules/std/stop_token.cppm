// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#if __has_include(<stop_token>) // D145183 contains a patch for this header
#  error "include this header unconditionally and uncomment the exported symbols"
#  include <stop_token>
#endif

export module std:stop_token;
export namespace std {
#if 0
  // [stoptoken], class stop_­token
  using std::stop_token;

  // [stopsource], class stop_­source
  using std::stop_source;

  // no-shared-stop-state indicator
  using std::nostopstate;
  using std::nostopstate_t;

  // [stopcallback], class template stop_­callback
  using std::stop_callback;
#endif
} // namespace std
