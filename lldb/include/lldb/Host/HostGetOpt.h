//===-- HostGetOpt.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#if !defined(_MSC_VER) && !defined(__NetBSD__)

#ifdef _WIN32
#define _BSD_SOURCE // Required so that getopt.h defines optreset
#endif

#include <getopt.h>
#include <unistd.h>

#else

#include <lldb/Host/common/GetOptInc.h>

#endif
