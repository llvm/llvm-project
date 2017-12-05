// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: not env CINDEXTEST_INVOCATION_EMISSION_PATH=%t c-index-test -test-load-source all %s
// RUN: cat %t/libclang-* | FileCheck %s

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env LIBCLANG_DISABLE_CRASH_RECOVERY=1 CINDEXTEST_INVOCATION_EMISSION_PATH=%t not --crash c-index-test -test-load-source all %s
// RUN: cat %t/libclang-* | FileCheck %s

// Verify that the file is removed for successful operation:
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env CINDEXTEST_INVOCATION_EMISSION_PATH=%t c-index-test -test-load-source all %s -DAVOID_CRASH
// RUN: ls %t | count 0

#ifndef AVOID_CRASH
#  pragma clang __debug parser_crash
#endif

// CHECK: {"toolchain":"{{.*}}","libclang.operation":"parse","libclang.opts":1,"args":["clang","-fno-spell-checking","{{.*}}record-parsing-invocation.c","-Xclang","-detailed-preprocessing-record","-fallow-editor-placeholders"]}
