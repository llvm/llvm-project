// RUN: %clang_cc1 -fsanitize=alloc-token -triple x86_64-apple-macosx10.15.0 -std=c++20 -fblocks -fobjc-arc -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

@class ObjCClass;

struct StructWithBlock {
  void (^b)(void);
};

// CHECK-LABEL: define {{.*}}ptr @_Z22test_struct_with_blockv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 8){{.*}} !alloc_token [[META_STRUCTWITHBLOCK:![0-9]+]]
StructWithBlock *test_struct_with_block() {
  return new StructWithBlock;
}

struct StructWithObjCPtr {
  id obj;
  Class cls;
  ObjCClass *ptr;
};

// CHECK-LABEL: define {{.*}}ptr @_Z25test_struct_with_objc_ptrv(
// CHECK: call noalias noundef nonnull ptr @_Znwm(i64 noundef 24){{.*}} !alloc_token [[META_STRUCTWITHOBJCPTR:![0-9]+]]
StructWithObjCPtr *test_struct_with_objc_ptr() {
  return new StructWithObjCPtr;
}

// CHECK: [[META_STRUCTWITHBLOCK]] = !{!"StructWithBlock", i1 true}
// CHECK: [[META_STRUCTWITHOBJCPTR]] = !{!"StructWithObjCPtr", i1 true}
