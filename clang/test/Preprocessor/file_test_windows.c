// REQUIRES: system-windows
// RUN: %clang -E -ffile-prefix-map=%p=A:\UNLIKELY_PATH\empty -c -o - %s | FileCheck %s
// RUN: %clang -E -fmacro-prefix-map=%p=A:\UNLIKELY_PATH\empty -c -o - %s | FileCheck %s
// RUN: %clang -E -fmacro-prefix-map=%p=A:\UNLIKELY_PATH=empty -c -o - %s | FileCheck %s -check-prefix CHECK-EVIL
// RUN: %clang -E -fmacro-prefix-map=%p/iNPUTS\=A:\UNLIKELY_PATH_INC\ -fmacro-prefix-map=%p/=A:\UNLIKELY_PATH_BASE\ -c -o - %s | FileCheck %s -check-prefix CHECK-CASE
// RUN: %clang -E -fmacro-prefix-map=%p\= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE

filename: __FILE__
#include "Inputs/include-file-test/file_test.h"

// CHECK: filename: "A:\\UNLIKELY_PATH\\empty\\file_test_windows.c"
// CHECK: filename: "A:\\UNLIKELY_PATH\\empty/Inputs/include-file-test/file_test.h"
// CHECK: basefile: "A:\\UNLIKELY_PATH\\empty\\file_test_windows.c"
// CHECK-NOT: filename:

// CHECK-EVIL: filename: "A:\\UNLIKELY_PATH=empty\\file_test_windows.c"
// CHECK-EVIL: filename: "A:\\UNLIKELY_PATH=empty/Inputs/include-file-test/file_test.h"
// CHECK-EVIL: basefile: "A:\\UNLIKELY_PATH=empty\\file_test_windows.c"
// CHECK-EVIL-NOT: filename:

// CHECK-CASE: filename: "A:\\UNLIKELY_PATH_BASE\\file_test_windows.c"
// CHECK-CASE: filename: "A:\\UNLIKELY_PATH_INC\\include-file-test/file_test.h"
// CHECK-CASE: basefile: "A:\\UNLIKELY_PATH_BASE\\file_test_windows.c"
// CHECK-CASE-NOT: filename:

// CHECK-REMOVE: filename: "file_test_windows.c"
// CHECK-REMOVE: filename: "Inputs/include-file-test/file_test.h"
// CHECK-REMOVE: basefile: "file_test_windows.c"
// CHECK-REMOVE-NOT: filename:
