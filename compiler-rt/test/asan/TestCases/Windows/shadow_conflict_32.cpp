// UNSUPPORTED: target={{.*-windows-gnu}}

// Load this DLL at the default 32-bit ASan shadow base, and test how we dump
// the process memory layout.
// REQUIRES: asan-32-bits
//
// RUN: %clang_cl_asan -DBUILD_DLL -LD %s -Fe%t_dll.dll -link -base:0x30000000 -fixed -dynamicbase:no
// RUN: %clang_cl_asan %s -Fe%t.exe -link %t_dll.lib
// RUN: not %run %t.exe 2>&1 | FileCheck %s

#ifndef BUILD_DLL
#include <stdio.h>

extern "C" __declspec(dllimport) int test_function();

int main() {
  fprintf(stderr, "should have failed to initialize, DLL got loaded near 0x%p\n",
          (void *)&test_function);
}

#else
extern "C" __declspec(dllexport) int test_function() { return 0; }
#endif

// CHECK: =={{[0-9:]+}}==Shadow memory range interleaves with an existing memory mapping. ASan cannot proceed correctly. ABORTING.
// CHECK: =={{[0-9:]+}}==ASan shadow was supposed to be located in the [0x2fff0000-0x3fffffff] range.
// CHECK: =={{[0-9:]+}}==Dumping process modules

// CHECK-DAG: {{0x30000000-0x300.....}} {{.*}}\shadow_conflict_32.cpp.tmp_dll.dll
// CHECK-DAG: {{0x........-0x........}} {{.*}}\shadow_conflict_32.cpp.tmp.exe
// CHECK-DAG: {{0x........-0x........}} {{.*}}\ntdll.dll
