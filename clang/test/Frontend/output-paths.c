// RUN: not %clang_cc1 -emit-llvm -o %t.doesnotexist/somename %s 2> %t
// RUN: FileCheck -check-prefix=OUTPUTFAIL -DMSG=%errc_ENOENT -input-file=%t %s

// OUTPUTFAIL: error: unable to open output file '{{.*}}doesnotexist{{.}}somename': '[[MSG]]'

// Check that -working-directory is respected when diagnosing output failures.
//
// RUN: rm -rf %t.d && mkdir -p %t.d/%basename_t-inner.d
// RUN: %clang_cc1 -working-directory %t.d -E -o %basename_t-inner.d/somename %s -verify
// expected-no-diagnostics

// RUN: %clang_cc1 -working-directory %t.d -E %s -o - | FileCheck %s
// CHECK: # 1
