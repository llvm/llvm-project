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

// Make sure we record the unsaved file hash.
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env CINDEXTEST_INVOCATION_EMISSION_PATH=%t not c-index-test -test-load-source all "-remap-file=%s,%S/Inputs/record-parsing-invocation-remap.c" %s
// RUN: cat %t/libclang-* | FileCheck --check-prefix=CHECK-UNSAVED %s

#ifndef AVOID_CRASH
#  pragma clang __debug parser_crash
#endif

// CHECK: {"toolchain":"{{.*}}","libclang.operation":"parse","libclang.opts":1,"args":["clang","-fno-spell-checking","{{.*}}record-parsing-invocation.c","-Xclang","-detailed-preprocessing-record","-fallow-editor-placeholders"]}
// CHECK-UNSAVED: {"toolchain":"{{.*}}","libclang.operation":"parse","libclang.opts":1,"args":["clang","-fno-spell-checking","{{.*}}record-parsing-invocation.c","-Xclang","-detailed-preprocessing-record","-fallow-editor-placeholders"],"unsaved_file_hashes":[{"name":"{{.*}}record-parsing-invocation.c","md5":"aee23773de90e665992b48209351d70e"}]}
