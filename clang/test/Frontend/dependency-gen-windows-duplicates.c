// REQUIRES: system-windows

// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/subdir
// RUN: echo > %t.dir/subdir/x.h
// RUN: cp %s %t.dir/test.c
// RUN: cd %t.dir

// RUN: %clang -MD -MF - %t.dir/test.c -fsyntax-only -I %t.dir/subdir | FileCheck %s
// CHECK: test.o:
// CHECK-NEXT: \test.c
// CHECK-NEXT: \subdir\x.h
// File x.h must appear only once (case insensitive check).
// CHECK-NOT: {{\\|/}}{{x|X}}.{{h|H}}

// Include x.h several times, with different casing and separators.
// Since all paths are passed to clang as absolute, all dependencies are absolute paths.
// We expect the output dependencies to contain only one line for file x.h

// Test case sensitivity.
#include "SubDir/X.h"
#include "subdir/x.h"

// Test separator sensitivity:
// clang internally concatenates x.h using the Windows native separator.
#include <x.h>

