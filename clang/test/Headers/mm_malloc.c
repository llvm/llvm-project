
// RUN: %clang_cc1 -internal-isystem %S/Inputs/include %s -emit-llvm -O1 -triple x86_64-linux-gnu -o - | FileCheck %s
#include <mm_malloc.h>

_Bool align_test(void) {
// CHECK-LABEL: @align_test(
// CHECK:    ret i1 true
     void *p = _mm_malloc(1024, 16);
    _Bool ret = ((__UINTPTR_TYPE__)p % 16) == 0;
    _mm_free(p);
    return ret;
}
