// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// REQUIRES: x86-registered-target
// Check that pchfile.h next to pchfile.cc is found correctly.
// RUN: %clang_cl -Werror -Wno-microsoft-include --target=x86_64-windows /Ycpchfile.h /FIpchfile.h /c /Fo%t.obj /Fp%t.pch -- %S/Inputs/pchfile.cpp
