// RUN: %clang_cc1 -fno-rtti -emit-llvm %s -o - -mconstructor-aliases -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm %s -o - -mconstructor-aliases -triple=i386-pc-win32 -fno-delete-null-pointer-checks | FileCheck %s

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
  // CHECK: %[[CHILD:.*]] = load ptr
  child->left();
}

void ChildOverride::left() {}

void call_right_no_override(ChildNoOverride *child) {
  child->right();
  // When calling a right base's virtual method, one needs to adjust `this` at the caller site.
  //
  // CHECK: %[[RIGHT_i8:.*]] = getelementptr inbounds i8, ptr %[[CHILD]], i32 4
  //
  // CHECK: %[[VFTABLE:.*]] = load ptr, ptr %[[RIGHT_i8]]
  // CHECK: %[[VFUN:.*]] = getelementptr inbounds ptr, ptr %[[VFTABLE]], i64 0
}

void ChildOverride::right() {
  foo(this);
}

void call_right_override(ChildOverride *child) {
  child->right();
  // Ensure that `nonnull` and `dereferenceable(N)` are not emitted whether or not null is valid
  //
  // CHECK: %[[RIGHT:.*]] = getelementptr inbounds i8, ptr %[[CHILD]], i32 4
  // CHECK: %[[VFUN_VALUE:.*]] = load ptr, ptr %[[VFUN]]
  // CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](ptr noundef %[[RIGHT]])
}
