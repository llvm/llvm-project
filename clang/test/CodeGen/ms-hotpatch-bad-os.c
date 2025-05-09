// This verifies that enabling Windows hotpatching when targeting a non-Windows OS causes an error.
//
// RUN: not %clang -c --target=x86_64-unknown-linux-elf -fms-hotpatch -fms-hotpatch-functions-list=foo -fms-hotpatch-functions-file=this_will_never_be_accessed -o%t.o %s 2>&1 | FileCheck %s
// CHECK: hotpatch functions file (-fms-hotpatch-functions-file) is only supported on Windows targets
// CHECK: hotpatch functions list (-fms-hotpatch-functions-list) is only supported on Windows targets

void foo();
