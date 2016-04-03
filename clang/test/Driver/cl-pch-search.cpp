// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// REQUIRES: x86-registered-target
// Check that pchfile.h next to to pchfile.cc is found correctly.
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /c /Fo%t.obj /Fp%t.pch -- %S/Inputs/pchfile.cpp 

// Check that i_group flags other than -include aren't skipped (e.g. -isystem).
#include "header0.h"
// RUN: %clang_cl -Werror -isystem%S/Inputs /Yupchfile.h /FIpchfile.h /c /Fo%t.obj /Fp%t.pch -- %s
