// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/gnumode-non-benign -verify %s

// expected-no-diagnostics

// This test ensures that submodules have the same GNUMode language option
// setting as the main clang invocation.
// Note that we set GNUMode = 0 with -std=c++11 for this file.

// This module fails to compile with GNUMode = 1.
#include "module.h"
