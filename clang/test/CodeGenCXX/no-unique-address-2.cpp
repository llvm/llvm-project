// RUN: %clang_cc1 -std=c++2a %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s

struct TriviallyCopyable {};

struct NonTriviallyCopyable {
  NonTriviallyCopyable() = default;
  NonTriviallyCopyable(const NonTriviallyCopyable&) = default;
  NonTriviallyCopyable(NonTriviallyCopyable &&) {}
};

struct Foo {
  int i;
  [[no_unique_address]] TriviallyCopyable m;
  [[no_unique_address]] NonTriviallyCopyable n;
};

void call() {
  Foo foo;
  Foo foo2(static_cast<Foo&&>(foo));
}

// The memcpy call should copy exact 4 bytes for member 'int i'
// CHECK: define {{.*}} void @_ZN3FooC2EOS_
// CHECK:  call void @llvm.memcpy.p0.p0.i64(ptr {{.+}}, ptr {{.+}}, i64 4, i1 false)
// CHECK:  call void @_ZN20NonTriviallyCopyableC2EOS_
