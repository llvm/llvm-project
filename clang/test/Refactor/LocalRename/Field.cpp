// RUN: clang-refactor local-rename -selection=test:%s -new-name=Bar %s -- | grep -v CHECK | FileCheck %s

class Baz {
  int /*range=*/Foo;
  // CHECK: int /*range=*/Bar;
public:
  Baz();
};

Baz::Baz() : /*range=*/Foo(0) {}
// CHECK: Baz::Baz() : /*range=*/Bar(0) {}
