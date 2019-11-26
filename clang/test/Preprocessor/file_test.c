// RUN: %clang -E -ffile-prefix-map=%p=/UNLIKELY_PATH/empty -c -o - %s | FileCheck %s
// RUN: %clang -E -fmacro-prefix-map=%p=/UNLIKELY_PATH/empty -c -o - %s | FileCheck %s
// RUN: %clang -E -fmacro-prefix-map=%p=/UNLIKELY_PATH=empty -c -o - %s | FileCheck %s -check-prefix CHECK-EVIL
// RUN: %clang -E -fmacro-prefix-map=%p/= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE

filename: __FILE__
#include "file_test.h"

// CHECK: filename: "/UNLIKELY_PATH/empty{{[/\\]}}file_test.c"
// CHECK: filename: "/UNLIKELY_PATH/empty{{[/\\]}}file_test.h"
// CHECK: basefile: "/UNLIKELY_PATH/empty{{[/\\]}}file_test.c"
// CHECK-NOT: filename:

// CHECK-EVIL: filename: "/UNLIKELY_PATH=empty{{[/\\]}}file_test.c"
// CHECK-EVIL: filename: "/UNLIKELY_PATH=empty{{[/\\]}}file_test.h"
// CHECK-EVIL: basefile: "/UNLIKELY_PATH=empty{{[/\\]}}file_test.c"
// CHECK-EVIL-NOT: filename:

// CHECK-REMOVE: filename: "file_test.c"
// CHECK-REMOVE: filename: "file_test.h"
// CHECK-REMOVE: basefile: "file_test.c"
// CHECK-REMOVE-NOT: filename:
