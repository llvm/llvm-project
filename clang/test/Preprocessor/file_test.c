// UNSUPPORTED: system-windows
// RUN: %clang -E -ffile-prefix-map=%p=/UNLIKELY_PATH/empty -c -o - %s | FileCheck %s
// RUN: %clang -E -fmacro-prefix-map=%p=/UNLIKELY_PATH/empty -c -o - %s | FileCheck %s
// RUN: %clang -E -fmacro-prefix-map=%p=/UNLIKELY_PATH=empty -c -o - %s | FileCheck %s -check-prefix CHECK-EVIL
// RUN: %clang -E -fmacro-prefix-map=%p/= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE
// RUN: %clang -E -fno-file-reproducible -fmacro-prefix-map=%p/= -target x86_64-pc-win32 -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE
// RUN: %clang -E -fno-file-reproducible -ffile-prefix-map=%p/= -target x86_64-pc-win32 -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE
// RUN: %clang -E -fmacro-prefix-map=%p/= -target x86_64-pc-win32 -c -o - %s | FileCheck %s --check-prefix CHECK-WINDOWS
// RUN: %clang -E -ffile-prefix-map=%p/= -target x86_64-pc-win32 -c -o - %s | FileCheck %s --check-prefix CHECK-WINDOWS
// RUN: %clang -E -ffile-reproducible -target x86_64-pc-win32 -c -o - %s | FileCheck %s --check-prefix CHECK-WINDOWS-REPRODUCIBLE
// RUN: %clang -E -ffile-reproducible -target x86_64-pc-linux-gnu -c -o - %s | FileCheck %s --check-prefix CHECK-LINUX-REPRODUCIBLE

/// Clang-format wants to do something unreasonable to this file.
// clang-format off

filename: __FILE__

/// This line tests that the __FILE__ in included header is canonicalized
/// (has "./" removed).
#include "./Inputs/./include-file-test/file_test.h"

// CHECK: filename: "/UNLIKELY_PATH/empty/file_test.c"
// CHECK: filename: "/UNLIKELY_PATH/empty/Inputs/include-file-test/file_test.h"
// CHECK: basefile: "/UNLIKELY_PATH/empty/file_test.c"
// CHECK-NOT: filename:

// CHECK-EVIL: filename: "/UNLIKELY_PATH=empty/file_test.c"
// CHECK-EVIL: filename: "/UNLIKELY_PATH=empty/Inputs/include-file-test/file_test.h"
// CHECK-EVIL: basefile: "/UNLIKELY_PATH=empty/file_test.c"
// CHECK-EVIL-NOT: filename:

// CHECK-REMOVE: filename: "file_test.c"
// CHECK-REMOVE: filename: "{{.*}}Inputs/{{.*}}include-file-test/file_test.h"
// CHECK-REMOVE: basefile: "file_test.c"
// CHECK-REMOVE-NOT: filename:

// CHECK-WINDOWS: filename: "file_test.c"
// CHECK-WINDOWS: filename: "Inputs\\include-file-test\\file_test.h"
// CHECK-WINDOWS: basefile: "file_test.c"
// CHECK-WINDOWS-NOT: filename:

// CHECK-WINDOWS-REPRODUCIBLE: filename: "{{.*}}\\file_test.c"
// CHECK-WINDOWS-REPRODUCIBLE: filename: "{{.*}}\\Inputs\\include-file-test\\file_test.h"
// CHECK-WINDOWS-REPRODUCIBLE: basefile: "{{.*}}\\file_test.c"
// CHECK-WINDOWS-REPRODUCIBLE-NOT: filename:

// CHECK-LINUX-REPRODUCIBLE: filename: "{{.*}}/file_test.c"
// CHECK-LINUX-REPRODUCIBLE: filename: "{{.*}}/Inputs/include-file-test/file_test.h"
// CHECK-LINUX-REPRODUCIBLE: basefile: "{{.*}}/file_test.c"
// CHECK-LINUX-REPRODUCIBLE-NOT: filename:
