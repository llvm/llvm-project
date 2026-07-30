// PR15642
// RUN: %clang -M -MG -fno-ms-compatibility %s | FileCheck -strict-whitespace %s --check-prefix=CHECK --check-prefix=SEP2F
// RUN: %clang -M -MG -fms-compatibility %s | FileCheck -strict-whitespace %s --check-prefix=CHECK --check-prefix=SEP5C
// RUN: %clang -M -MG -MV %s | FileCheck -strict-whitespace %s --check-prefix=NMAKE

// CHECK: \ \ \ \ .h
// CHECK: $$$$.h
// CHECK: \#\#.h
// NMAKE: "    .h"
// NMAKE: "$$.h"
// NMAKE: "##.h"
// NMAKE-NOT: "
// NMAKE: normal.h
// NMAKE-NOT: "

#include "    .h"
#include "$$.h"
#include "##.h"
#include "normal.h"

// Backslash followed by # or space should escape both characters, because
// that's what GNU Make wants.  GCC does the right thing with space, but not
// #, so Clang does too. (There should be 3 backslashes before the #.)
// SEP2F: a{{[/\\]}}b{{[/\\]}}\#c{{/|\\\\}}\ d.h
// With -fms-compatibility, Backslashes in #include are treated as path separators.
// Backslashes are given in the emission for special characters, like 0x20 or 0x23.
// SEP5C: a{{[/\\]}}b{{[/\\]}}\#c{{/|\\\\}}\ d.h
// These combinations are just another case for NMAKE.
// NMAKE: "a{{[/\\]}}b{{[/\\]}}#c{{[/\\]}} d.h"

#include "a\b\#c\ d.h"
