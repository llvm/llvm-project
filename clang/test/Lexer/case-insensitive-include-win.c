// Most Microsoft-specific testing should go in case-insensitive-include-ms.c
// This file should only include code that really needs a Windows host OS to
// run.

// Note: We must use the real path here, because the logic to detect case
// mismatches must resolve the real path to figure out the original casing.
// If we use %t and we are on a substitute drive S: mapping to C:\subst,
// then we will compare "S:\test.dir\FOO.h" to "C:\subst\test.dir\foo.h"
// and avoid emitting the diagnostic because the structure is different.

// REQUIRES: system-windows
// RUN: mkdir -p %t.dir
// RUN: touch %t.dir/foo.h
// RUN: not %clang_cl /FI \\?\%{t:real}.dir\FOO.h /WX -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: non-portable path to file '"\\?\
