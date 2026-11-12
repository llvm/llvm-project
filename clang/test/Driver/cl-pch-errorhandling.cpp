// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// /Yc but pch generation fails => main file not compiled
// This is a separate file since executing this failure path requires
// code generation, which makes this test require an x86 backend.
// REQUIRES: x86-registered-target

// RUN: not %clang_cl -Werror --target=x86_64 /Yc%S/Inputs/pchfile.h /FI%S/Inputs/pchfile.h /Fp%t.pch /c -DERR_HEADER -- %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: nope1
// CHECK-NOT: nope2

#error nope2
