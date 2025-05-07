

// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -fsyntax-only -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsyntax-only -ast-dump %s | FileCheck %s
#include <ptrcheck.h>

int * _Nullable __single global_var1;
// CHECK: global_var1 'int *__single _Nullable':'int *__single'

int * __single _Nullable global_var2;
// CHECK: global_var2 'int *__single _Nullable':'int *__single'

void f(void) {
  int * _Nullable __bidi_indexable local_var1;
  // CHECK: local_var1 'int *__bidi_indexable _Nullable':'int *__bidi_indexable'

  int * __bidi_indexable _Nullable local_var2;
  // CHECK: local_var2 'int *__bidi_indexable _Nullable':'int *__bidi_indexable'

  int (* _Nullable __bidi_indexable local_var3)[4];

  int (* __bidi_indexable _Nullable local_var4)[4];

  int (* _Nullable __indexable local_var5)[4];

  int (* __indexable _Nullable local_var6)[4];
}

struct S {
  int * __single _Nullable member_var1;
  // CHECK: member_var1 'int *__single _Nullable':'int *__single'

  int * _Nullable __single member_var2;
  // CHECK: member_var2 'int *__single _Nullable':'int *__single'
};

union U {
  int * __single _Nullable union_var1;
  // CHECK: union_var1 'int *__single _Nullable':'int *__single'

  int * _Nullable __single union_var2;
  // CHECK: union_var2 'int *__single _Nullable':'int *__single'
};

void foo1(int * _Nonnull __single __counted_by(n) ptr1, unsigned n);
// CHECK: ptr1 'int *__single __counted_by(n) _Nonnull':'int *__single'

void foo2(int * __single __counted_by(n) _Nonnull ptr2, unsigned n);
// CHECK: ptr2 'int *__single __counted_by(n) _Nonnull':'int *__single'
