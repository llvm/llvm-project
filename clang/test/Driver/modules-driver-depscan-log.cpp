// Check that -fdepscan-log-path enables dependency scanning logging.

// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang -c -std=c++23 -fmodules-driver -fdepscan-log-path=%t/scan.log \
// RUN:   %t/A.cppm
// RUN: FileCheck %s --input-file %t/scan.log

// CHECK: logging_start
// CHECK: starting scanning command:
// CHECK: logging_end

//--- A.cppm
export module A;
export int a() { return 0; }
