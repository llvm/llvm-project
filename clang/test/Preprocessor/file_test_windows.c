// REQUIRES: system-windows
// RUN: %clang -E -fno-file-reproducible -target x86_64-pc-win32 -ffile-prefix-map=%p=A:\UNLIKELY_PATH\empty -c -o - %s | FileCheck %s
// RUN: %clang -E -fno-file-reproducible -target x86_64-pc-win32 -fmacro-prefix-map=%p=A:\UNLIKELY_PATH\empty -c -o - %s | FileCheck %s
// RUN: %clang -E -fno-file-reproducible -target x86_64-pc-win32 -fmacro-prefix-map=%p=A:\UNLIKELY_PATH=empty -c -o - %s | FileCheck %s -check-prefix CHECK-EVIL
// RUN: %clang -E -fno-file-reproducible -target x86_64-pc-win32 -fmacro-prefix-map=%p/iNPUTS\=A:\UNLIKELY_PATH_INC\ -fmacro-prefix-map=%p/=A:\UNLIKELY_PATH_BASE\ -c -o - %s | FileCheck %s -check-prefix CHECK-CASE
// RUN: %clang -E -fno-file-reproducible -target x86_64-pc-win32 -fmacro-prefix-map=%p\= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE

// RUN: %clang -E -target x86_64-pc-win32 -ffile-prefix-map=%p=A:\UNLIKELY_PATH\empty -c -o - %s | FileCheck %s --check-prefix CHECK-REPRODUCIBLE
// RUN: %clang -E -target x86_64-pc-win32 -fmacro-prefix-map=%p=A:\UNLIKELY_PATH\empty -c -o - %s | FileCheck %s --check-prefix CHECK-REPRODUCIBLE
// RUN: %clang -E -target x86_64-pc-win32 -fmacro-prefix-map=%p=A:\UNLIKELY_PATH=empty -c -o - %s | FileCheck %s -check-prefix CHECK-EVIL-REPRODUCIBLE
// RUN: %clang -E -target x86_64-pc-win32 -fmacro-prefix-map=%p/iNPUTS\=A:\UNLIKELY_PATH_INC\ -fmacro-prefix-map=%p/=A:\UNLIKELY_PATH_BASE\ -c -o - %s | FileCheck %s -check-prefix CHECK-CASE-REPRODUCIBLE
// RUN: %clang -E -target x86_64-pc-win32 -fmacro-prefix-map=%p\= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE-REPRODUCIBLE

// RUN: %clang -E -target x86_64-pc-linux-gnu -fmacro-prefix-map=%p\= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE
// RUN: %clang -E -target x86_64-pc-linux-gnu -ffile-prefix-map=%p\= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE

// Clang defaults to forward slashes for the non-prefix portion of the path even if the build environment is Windows.
// RUN: %clang -E -fno-file-reproducible -target x86_64-pc-linux-gnu -fmacro-prefix-map=%p\= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE
// RUN: %clang -E -fno-file-reproducible -target x86_64-pc-linux-gnu -ffile-prefix-map=%p\= -c -o - %s | FileCheck %s --check-prefix CHECK-REMOVE

// RUN: %clang -E -ffile-reproducible -target x86_64-pc-win32 -c -o - %s | FileCheck %s --check-prefix CHECK-WINDOWS-FULL
// RUN: %clang -E -ffile-reproducible -target x86_64-pc-linux-gnu -c -o - %s | FileCheck %s --check-prefix CHECK-LINUX-FULL

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

// CHECK-REPRODUCIBLE: filename: "A:\\UNLIKELY_PATH\\empty\\file_test_windows.c"
// CHECK-REPRODUCIBLE: filename: "A:\\UNLIKELY_PATH\\empty\\Inputs\\include-file-test\\file_test.h"
// CHECK-REPRODUCIBLE: basefile: "A:\\UNLIKELY_PATH\\empty\\file_test_windows.c"
// CHECK-REPRODUCIBLE-NOT: filename:

// CHECK-EVIL-REPRODUCIBLE: filename: "A:\\UNLIKELY_PATH=empty\\file_test_windows.c"
// CHECK-EVIL-REPRODUCIBLE: filename: "A:\\UNLIKELY_PATH=empty\\Inputs\\include-file-test\\file_test.h"
// CHECK-EVIL-REPRODUCIBLE: basefile: "A:\\UNLIKELY_PATH=empty\\file_test_windows.c"
// CHECK-EVIL-REPRODUCIBLE-NOT: filename:

// CHECK-CASE-REPRODUCIBLE: filename: "A:\\UNLIKELY_PATH_BASE\\file_test_windows.c"
// CHECK-CASE-REPRODUCIBLE: filename: "A:\\UNLIKELY_PATH_INC\\include-file-test\\file_test.h"
// CHECK-CASE-REPRODUCIBLE: basefile: "A:\\UNLIKELY_PATH_BASE\\file_test_windows.c"
// CHECK-CASE-REPRODUCIBLE-NOT: filename:

// CHECK-REMOVE-REPRODUCIBLE: filename: "file_test_windows.c"
// CHECK-REMOVE-REPRODUCIBLE: filename: "Inputs\\include-file-test\\file_test.h"
// CHECK-REMOVE-REPRODUCIBLE: basefile: "file_test_windows.c"
// CHECK-REMOVE-REPRODUCIBLE-NOT: filename:

// CHECK-WINDOWS-FULL: filename: "{{[^/]*}}file_test_windows.c"
// CHECK-WINDOWS-FULL: filename: "{{[^/]*}}Inputs\\include-file-test\\file_test.h"
// CHECK-WINDOWS-FULL: basefile: "{{[^/]*}}file_test_windows.c"
// CHECK-WINDOWS-FULL-NOT: filename:

// Clang does not modify the prefix for POSIX style, so it may have backslashes.
// CHECK-LINUX-FULL: filename: "{{.*}}file_test_windows.c"
// CHECK-LINUX-FULL: filename: "{{.*}}Inputs/include-file-test/file_test.h"
// CHECK-LINUX-FULL: basefile: "{{.*}}file_test_windows.c"
// CHECK-LINUX-FULL-NOT: filename:
