// RUN: clang-refactor local-rename -selection=test:%s -new-name=bar %s -- | grep -v CHECK | FileCheck %s

struct Struct {
  int /*range f=*/field;
};

struct Struct2 {
  Struct /*range array=*/array[4][2];
};

void foo() {
  (void)__builtin_offsetof(Struct, /*range f=*/field);
  (void)__builtin_offsetof(Struct2, /*range array=*/array[1][0]./*range f=*/field);
}

#define OFFSET_OF_(X, Y) __builtin_offsetof(X, Y)

class SubclassOffsetof : public Struct {
  void foo() {
    (void)OFFSET_OF_(SubclassOffsetof, field);
  }
};

// CHECK: 2 'array' results:
// CHECK: Struct /*range array=*/bar[4][2];
// CHECK: __builtin_offsetof(Struct2, /*range array=*/bar[1][0]./*range f=*/field);

// CHECK: 3 'f' results:
// CHECK: int /*range f=*/bar;
// CHECK: __builtin_offsetof(Struct, /*range f=*/bar);
// CHECK-NEXT: __builtin_offsetof(Struct2, /*range array=*/array[1][0]./*range f=*/bar);
// CHECK: OFFSET_OF_(SubclassOffsetof, bar);
