//===- lib/Config/Version.cpp - LLD Version Number ---------------*- C++-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines several version-related utility functions for LLD.
//
//===----------------------------------------------------------------------===//

#include "lld/Config/Version.h"

using namespace llvm;

#define JOIN2(X) #X
#define JOIN(X, Y) JOIN2(X.Y)

// A string that describes the lld version number, e.g., "1.0".
#define VERSION JOIN(LLD_VERSION_MAJOR, LLD_VERSION_MINOR)

// A string that describes SVN repository, e.g.,
// " (https://llvm.org/svn/llvm-project/lld/trunk 284614)".
#if defined(LLD_REPOSITORY_STRING) && defined(LLD_REVISION_STRING)
#define REPO " (" LLD_REPOSITORY_STRING " " LLD_REVISION_STRING ")"
#elif defined(LLD_REPOSITORY_STRING)
#define REPO " (" LLD_REPOSITORY_STRING ")"
#elif defined(LLD_REVISION_STRING)
#define REPO " (" LLD_REVISION_STRING ")"
#else
#define REPO ""
#endif

// Returns a version string, e.g.,
// "LLD 4.0 (https://llvm.org/svn/llvm-project/lld/trunk 284614)".
StringRef lld::getLLDVersion() { return "LLD " VERSION REPO; }
