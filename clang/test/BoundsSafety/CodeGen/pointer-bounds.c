

// RUN: %clang_cc1 %s -O2 -fbounds-safety -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O2 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm -o - | FileCheck %s

#include <ptrcheck.h>

int garray[10];
int *const __bidi_indexable lower = &garray[0];
int *const __bidi_indexable mid = &garray[5];
int *const __bidi_indexable upper = &garray[10];

#define ASSERT(X) if (!(X)) __builtin_trap();

int testGarray() {
    ASSERT(__ptr_lower_bound(lower) == lower);
    ASSERT(__ptr_lower_bound(mid) == lower);
    ASSERT(__ptr_lower_bound(upper) == lower);

    ASSERT(__ptr_upper_bound(lower) == upper);
    ASSERT(__ptr_upper_bound(mid) == upper);
    ASSERT(__ptr_upper_bound(upper) == upper);
    return 0;
}
// CHECK-LABEL: @testGarray
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i32 0

int testArray() {
    int array[10];
    int *lower = &array[0];
    int *mid = &array[5];
    int *upper = &array[10];
    ASSERT(__ptr_lower_bound(lower) == lower);
    ASSERT(__ptr_lower_bound(mid) == lower);
    ASSERT(__ptr_lower_bound(upper) == lower);

    ASSERT(__ptr_upper_bound(lower) == upper);
    ASSERT(__ptr_upper_bound(mid) == upper);
    ASSERT(__ptr_upper_bound(upper) == upper);
    return 0;
}
// CHECK-LABEL: @testArray
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i32 0

int testStaticArray() {
    static int array[10];
    static int *const __bidi_indexable lower = &array[0];
    static int *const __bidi_indexable mid = &array[5];
    static int *const __bidi_indexable upper = &array[10];
    ASSERT(__ptr_lower_bound(lower) == lower);
    ASSERT(__ptr_lower_bound(mid) == lower);
    ASSERT(__ptr_lower_bound(upper) == lower);

    ASSERT(__ptr_upper_bound(lower) == upper);
    ASSERT(__ptr_upper_bound(mid) == upper);
    ASSERT(__ptr_upper_bound(upper) == upper);
    return 0;
}
// CHECK-LABEL: @testStaticArray
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i32 0
