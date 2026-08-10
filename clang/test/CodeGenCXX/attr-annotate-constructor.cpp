// RUN: %clang %s -S -emit-llvm -target x86_64-unknown-linux -o -

// Test annotation attributes on constructors do not crash.

class Foo {
public:
  [[clang::annotate("test")]] Foo() {}
};

Foo foo;
