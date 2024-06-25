//===-- include/flang/Common/windows-include.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper around windows.h that works around the name conflicts.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_WINDOWS_INCLUDE_H_
#define FORTRAN_COMMON_WINDOWS_INCLUDE_H_

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>

#endif // _WIN32

#endif // FORTRAN_COMMON_WINDOWS_INCLUDE_H_
