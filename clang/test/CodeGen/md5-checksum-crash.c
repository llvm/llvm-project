// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -dwarf-version=5 %s -emit-llvm -o- | FileCheck %s
// RUN: %clang_cc1 -triple %ms_abi_triple -gcodeview -debug-info-kind=limited %s -emit-llvm -o- | FileCheck %s

// This had been crashing, no MD5 checksum for string.h.
// Now if there are #line directives, don't bother with checksums
// as a preprocessed file won't properly reflect the original source.
#define __NTH fct
void fn1() {}
# 7 "/usr/include/string.h"
void __NTH() {}
// Verify no checksum attributes on these files.
// CHECK-DAG: DIFile(filename: "{{.*}}.c", directory: "{{[^"]*}}")
// CHECK-DAG: DIFile(filename: "{{.*}}string.h", directory: "{{[^"]*}}")
