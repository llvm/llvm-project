// RUN: %clangxx -target x86_64-unknown-unknown -g %s -emit-llvm -S -o - | FileCheck %s
// RUN: %clangxx -target x86_64-unknown-unknown -g -fno-elide-constructors %s -emit-llvm -S -o - | FileCheck %s -check-prefix=NOELIDE
struct Foo {
  Foo() = default;
  Foo(Foo &&other) { x = other.x; }
  int x;
};
void some_function(int);
Foo getFoo() {
  Foo foo;
  foo.x = 41;
  some_function(foo.x);
  return foo;
}

int main() {
  Foo bar = getFoo();
  return bar.x;
}

// Check that NRVO variables are stored as a pointer with deref if they are
// stored in the return register.

// CHECK: %result.ptr = alloca i8*, align 8
// CHECK: call void @llvm.dbg.declare(metadata i8** %result.ptr,
// CHECK-SAME: metadata !DIExpression(DW_OP_deref)
// NOELIDE: call void @llvm.dbg.declare(metadata %struct.Foo* %foo,
// NOELIDE-SAME:                        metadata !DIExpression()
