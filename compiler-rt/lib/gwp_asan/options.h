//===-- options.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_OPTIONS_H_
#define GWP_ASAN_OPTIONS_H_

namespace gwp_asan {
namespace options {
// The function pointer type for printf(). Follows the standard format from the
// sanitizers library. If the supported allocator exposes printing via a
// different function signature, please provide a wrapper which has this
// printf() signature, and pass the wrapper instead.
typedef void (*Printf_t)(const char *Format, ...);

struct Options {
  Printf_t Printf = nullptr;

  // Read the options from the included definitions file.
#define GWP_ASAN_OPTION(Type, Name, DefaultValue, Description)                 \
  Type Name = DefaultValue;
#include "gwp_asan/options.inc"
#undef GWP_ASAN_OPTION

  void setDefaults() {
#define GWP_ASAN_OPTION(Type, Name, DefaultValue, Description)                 \
  Name = DefaultValue;
#include "gwp_asan/options.inc"
#undef GWP_ASAN_OPTION

    Printf = nullptr;
  }
};
} // namespace options
} // namespace gwp_asan

#endif // GWP_ASAN_OPTIONS_H_
