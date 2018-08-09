// RUN: not %clang_cc1 -fsyntax-only -triple i686-win32 %s 2>&1 | FileCheck %s

#include "Inputs\success.h"

// CHECK: error: 'Inputs\success.h' file not found
// CHECK: #include "Inputs\success.h"
// CHECK:          ^

// expected to fail on windows as the inclusion would succeed and the
// compilation will fail due to the '#error success'.
// XFAIL: windows-msvc

// This test may or may not fail since 'Inputs\success.h' is passed
// to Win32 APIs on Windows.
// REQUIRES: disabled
