//===- lib/Common/Version.cpp - LLD Version Number ---------------*- C++-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines several version-related utility functions for LLD.
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Version.h"

#ifdef HAVE_VCS_VERSION_INC
#include "VCSVersion.inc"
#endif

// Returns a version string, e.g.:
// lld 9.0.0 (https://github.com/llvm/llvm-project.git 9efdd7ac5e914d3c9fa1ef)
std::string lld::getLLDVersion() {
#ifdef LLD_VENDOR
#define LLD_VENDOR_DISPLAY LLD_VENDOR " "
#else
#define LLD_VENDOR_DISPLAY
#endif
#if defined(LLD_REPOSITORY) && defined(LLD_REVISION)
  return LLD_VENDOR_DISPLAY "LLD " LLD_VERSION_STRING " (" LLD_REPOSITORY
                            " " LLD_REVISION ")";
#else
  return LLD_VENDOR_DISPLAY "LLD " LLD_VERSION_STRING;
#endif
#undef LLD_VENDOR_DISPLAY
}
