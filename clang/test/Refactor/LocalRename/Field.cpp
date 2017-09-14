// RUN: clang-refactor local-rename -selection=test:%s -no-dbs %s | FileCheck %s

class Baz {
  int /*range=*/Foo; // CHECK: int /*range=*/Bar;
public:
  Baz();
};

Baz::Baz() : /*range=*/Foo(0) {}  // CHECK: Baz::Baz() : /*range=*/Bar(0) {};
