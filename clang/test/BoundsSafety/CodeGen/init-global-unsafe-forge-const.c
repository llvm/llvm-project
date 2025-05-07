

// RUN: %clang_cc1 -triple arm64-apple-iphoneos -O0 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple  arm64-apple-iphoneos -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

struct s { int arr[10]; char *cp; };

void *g1 = __unsafe_forge_single(void *, 1);
struct s *g2 = __unsafe_forge_single(struct s*, 2);
void *__bidi_indexable g3 = __unsafe_forge_bidi_indexable(void *, 3, 10);
struct s *__bidi_indexable g4 = __unsafe_forge_bidi_indexable(void *, 4, sizeof(struct s));

// CHECK: @g1 = global ptr inttoptr (i64 1 to ptr), align 8
// CHECK: @g2 = global ptr inttoptr (i64 2 to ptr), align 8
// CHECK: @g3 = global %"__bounds_safety::wide_ptr.bidi_indexable" { ptr inttoptr (i64 3 to ptr), ptr inttoptr (i64 13 to ptr), ptr inttoptr (i64 3 to ptr) }, align 8
// CHECK: @g4 = global %"__bounds_safety::wide_ptr.bidi_indexable{{.*}}" { ptr inttoptr (i64 4 to ptr), ptr inttoptr (i64 52 to ptr), ptr inttoptr (i64 4 to ptr) }, align 8
