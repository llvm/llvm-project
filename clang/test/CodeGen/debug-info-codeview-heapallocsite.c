// RUN: %clang_cc1 -triple x86_64-windows-msvc -debug-info-kind=limited -gcodeview -fdeclspec -S -emit-llvm < %s | FileCheck %s

struct Foo {
  int x;
};

__declspec(allocator) void *alloc_void();
__declspec(allocator) struct Foo *alloc_foo();

void call_alloc_void() {
  struct Foo *p = (struct Foo*)alloc_void();
}

void call_alloc_foo() {
  struct Foo *p = alloc_foo();
}

// CHECK-LABEL: define {{.*}}void @call_alloc_void
// CHECK: call i8* {{.*}}@alloc_void{{.*}} !heapallocsite [[DBG1:!.*]]

// CHECK-LABEL: define {{.*}}void @call_alloc_foo
// CHECK: call %struct.Foo* {{.*}}@alloc_foo{{.*}} !heapallocsite [[DBG2:!.*]]

// CHECK: [[DBG1]] = !{}
// CHECK: [[DBG2]] = distinct !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:                                 name: "Foo"

