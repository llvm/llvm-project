

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

// Deliberately use `__indexable` as an attribute
// because nothing else in this program uses it.
// That means when check for a cast to this pointer
// attribute in `call_has_no_prototype_k_and_r` that
// we can be sure it is being done because of the attribute
// in this function declaration and nothing else.
int *has_no_prototype_k_and_r(a)
int *__indexable a;
{
  return a;
}

void call_has_no_prototype_k_and_r(int *__counted_by(count) p, int count) {
  int *a = p;
  has_no_prototype_k_and_r(a);
}

// CHECK-LABEL:|-FunctionDecl {{.*}} has_no_prototype_k_and_r 'int *__single(int *__indexable)'
// CHECK-NEXT:| |-ParmVarDecl {{.*}} a 'int *__indexable'

// This behavior is a little surprising. `has_no_prototype_k_and_r` does not
// have a prototype so one might expect the default argument promotion to apply.
// However, that's not what clang does because in `Sema::BuildResolvedCallExpr`
// although it figures out there's no prototype it tries looking for it
// elsewhere and thinks it finds one so it doesn't do default argument promotion.
// While this might be a violation of the C standard it means we end up casting
// to the actual types on the function which means we use the correct -fbounds-safety
// attributes. So this likely bug in Clang is actually good for BoundsSafety
// because it means there won't be an ABI mismatch.

// CHECK-LABEL:`-FunctionDecl {{.*}} call_has_no_prototype_k_and_r 'void (int *__single __counted_by(count), int)'
// CHECK:    `-CallExpr {{.*}} 'int *__single'
// CHECK-NEXT:      |-ImplicitCastExpr {{.*}} 'int *__single(*__single)()' <FunctionToPointerDecay>
// CHECK-NEXT:      | `-DeclRefExpr {{.*}} 'int *__single()' Function {{.*}} 'has_no_prototype_k_and_r' 'int *__single(int *__indexable)'
// CHECK-NEXT:      `-ImplicitCastExpr {{.*}} 'int *__indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:          `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'a' 'int *__bidi_indexable'
