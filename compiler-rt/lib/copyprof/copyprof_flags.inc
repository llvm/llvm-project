//===-- copyprof_flags.inc --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the list of CopyProf runtime flags, their default values,
/// and descriptions for flag registration and parsing.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_FLAG
#  error "Define COPYPROF_FLAG prior to including this file!"
#endif

// COPYPROF_FLAG(Type, Name, DefaultValue, Description)
// See COMMON_FLAG in sanitizer_flags.inc for more details.

COPYPROF_FLAG(int, obj_size_threshold, 16,
              "Only report copies of objects whose transitive size in bytes is "
              "strictly larger than the threshold.")
COPYPROF_FLAG(
    bool, must_allocate, true,
    "Only report copies of objects that allocated memory when copied.")
COPYPROF_FLAG(
    const char*, report_output_file, nullptr,
    "If set, write CopyProf reports to this file instead of stdout. The "
    "filename is made unique by appending a suffix to it. This avoids "
    "collisions when an application starts multiple copies of itself.")
