// RUN: %clang_cc1 -fno-rtti -emit-llvm %s -o - -mconstructor-aliases -triple=i386-pc-win32 | FileCheck %s

struct Left {
  virtual void left();
};

struct Right {
  virtual void right();
};

struct ChildNoOverride : Left, Right {
};

struct ChildOverride : Left, Right {
  virtual void left();
  virtual void right();
};

extern "C" void foo(void *);

void call_left_no_override(ChildNoOverride *child) {
// CHECK-LABEL: define dso_local void @"?call_left_no_override
// CHECK: %[[CHILD:.*]] = load ptr

  child->left();
// Only need to cast 'this' to Left*.
// CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[CHILD]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load ptr, ptr %[[VFUN]]
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](ptr {{[^,]*}} %[[CHILD]])
// CHECK: ret
}

void ChildOverride::left() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"?left@ChildOverride@@UAEXXZ"
// CHECK-SAME: (ptr {{[^,]*}} %[[THIS:.*]])
//
// No need to adjust 'this' as the ChildOverride's layout begins with Left.
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 4
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 4

  foo(this);
// CHECK: %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: call void @foo(ptr noundef %[[THIS]])
// CHECK: ret
}

void call_left_override(ChildOverride *child) {
// CHECK-LABEL: define dso_local void @"?call_left_override
// CHECK: %[[CHILD:.*]] = load ptr

  child->left();
// CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[CHILD]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load ptr, ptr %[[VFUN]]
//
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](ptr {{[^,]*}} %[[CHILD]])
// CHECK: ret
}

void call_right_no_override(ChildNoOverride *child) {
// CHECK-LABEL: define dso_local void @"?call_right_no_override
// CHECK: %[[CHILD:.*]] = load ptr

  child->right();
// When calling a right base's virtual method, one needs to adjust 'this' at
// the caller site.
//
// CHECK: %[[RIGHT_i8:.*]] = getelementptr inbounds i8, ptr %[[CHILD]], i32 4
//
// CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[RIGHT_i8]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load ptr, ptr %[[VFUN]]
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](ptr {{[^,]*}} %[[RIGHT_i8]])
// CHECK: ret
}

void ChildOverride::right() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"?right@ChildOverride@@UAEXXZ"(ptr
//
// ChildOverride::right gets 'this' cast to Right* in ECX (i.e. this+4) so we
// need to adjust 'this' before use.
//
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 4
// CHECK: store ptr %[[ECX:.*]], ptr %[[THIS_ADDR]], align 4
// CHECK: %[[THIS_RELOAD:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[THIS_ADJUSTED:.*]] = getelementptr inbounds i8, ptr %[[THIS_RELOAD]], i32 -4

  foo(this);
// CHECK: call void @foo(ptr noundef %[[THIS_ADJUSTED]])
// CHECK: ret
}

void call_right_override(ChildOverride *child) {
// CHECK-LABEL: define dso_local void @"?call_right_override
// CHECK: %[[CHILD:.*]] = load ptr

  child->right();
// When calling a right child's virtual method, one needs to adjust 'this' at
// the caller site.
//
// CHECK: %[[RIGHT:.*]] = getelementptr inbounds i8, ptr %[[CHILD]], i32 4
//
// CHECK: %[[VFPTR_i8:.*]] = getelementptr inbounds i8, ptr %[[CHILD]], i32 4
// CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[VFPTR_i8]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load ptr, ptr %[[VFUN]]
//
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](ptr noundef %[[RIGHT]])
// CHECK: ret
}

struct GrandchildOverride : ChildOverride {
  virtual void right();
};

void GrandchildOverride::right() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"?right@GrandchildOverride@@UAEXXZ"(ptr
//
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 4
// CHECK: store ptr %[[ECX:.*]], ptr %[[THIS_ADDR]], align 4
// CHECK: %[[THIS_RELOAD:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: %[[THIS_ADJUSTED:.*]] = getelementptr inbounds i8, ptr %[[THIS_RELOAD]], i32 -4

  foo(this);
// CHECK: call void @foo(ptr noundef %[[THIS_ADJUSTED]])
// CHECK: ret
}

void call_grandchild_right(GrandchildOverride *obj) {
  // Just make sure we don't crash.
  obj->right();
}

void emit_ctors() {
  Left l;
  // CHECK-LABEL: define {{.*}} @"??0Left@@QAE@XZ"
  // CHECK-NOT: getelementptr
  // CHECK:   store ptr @"??_7Left@@6B@"
  // CHECK: ret

  Right r;
  // CHECK-LABEL: define {{.*}} @"??0Right@@QAE@XZ"
  // CHECK-NOT: getelementptr
  // CHECK:   store ptr @"??_7Right@@6B@"
  // CHECK: ret

  ChildOverride co;
  // CHECK-LABEL: define {{.*}} @"??0ChildOverride@@QAE@XZ"
  // CHECK:   %[[THIS:.*]] = load ptr, ptr
  // CHECK:   store ptr @"??_7ChildOverride@@6BLeft@@@", ptr %[[THIS]]
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 4
  // CHECK:   store ptr @"??_7ChildOverride@@6BRight@@@", ptr %[[VFPTR_i8]]
  // CHECK: ret

  GrandchildOverride gc;
  // CHECK-LABEL: define {{.*}} @"??0GrandchildOverride@@QAE@XZ"
  // CHECK:   %[[THIS:.*]] = load ptr, ptr
  // CHECK:   store ptr @"??_7GrandchildOverride@@6BLeft@@@", ptr %[[THIS]]
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8, ptr %[[THIS]], i32 4
  // CHECK:   store ptr @"??_7GrandchildOverride@@6BRight@@@", ptr %[[VFPTR_i8]]
  // CHECK: ret
}

struct LeftWithNonVirtualDtor {
  virtual void left();
  ~LeftWithNonVirtualDtor();
};

struct AsymmetricChild : LeftWithNonVirtualDtor, Right {
  virtual ~AsymmetricChild();
};

void call_asymmetric_child_complete_dtor() {
  // CHECK-LABEL: define dso_local void @"?call_asymmetric_child_complete_dtor@@YAXXZ"
  AsymmetricChild obj;
  // CHECK: call x86_thiscallcc noundef ptr @"??0AsymmetricChild@@QAE@XZ"(ptr {{[^,]*}} %[[OBJ:.*]])
  // CHECK-NOT: getelementptr
  // CHECK: call x86_thiscallcc void @"??1AsymmetricChild@@UAE@XZ"(ptr {{[^,]*}} %[[OBJ]])
  // CHECK: ret
}
