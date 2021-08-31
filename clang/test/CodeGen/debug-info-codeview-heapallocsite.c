// RUN: %clang_cc1 -triple x86_64-windows-msvc -debug-info-kind=limited -gcodeview -fdeclspec -S -emit-llvm %s -o - | FileCheck %s

struct Foo;
struct Bar;

__declspec(allocator) void *alloc_void();
__declspec(allocator) struct Foo *alloc_foo();

void call_alloc() {
  struct Foo *p = alloc_void();
  struct Foo *w = alloc_foo();
  struct Foo *q = (struct Foo*)alloc_void();
  struct Foo *r = (struct Foo*)(struct Bar*)alloc_void();
}

// CHECK-LABEL: define {{.*}}void @call_alloc
// CHECK: call i8* {{.*}}@alloc_void{{.*}} !heapallocsite [[DBG1:!.*]]
// CHECK: call %struct.Foo* {{.*}}@alloc_foo{{.*}} !heapallocsite [[DBG2:!.*]]
// CHECK: call i8* {{.*}}@alloc_void{{.*}} !heapallocsite [[DBG2]]
// CHECK: call i8* {{.*}}@alloc_void{{.*}} !heapallocsite [[DBG3:!.*]]

// CHECK: [[DBG2]] = !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:                                 name: "Foo"
// CHECK: [[DBG3]] = !DICompositeType(tag: DW_TAG_structure_type,
// CHECK-SAME:                                 name: "Bar"
// CHECK: [[DBG1]] = !{}
