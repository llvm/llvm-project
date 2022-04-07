// RUN: %clang_cc1 -no-opaque-pointers -fno-rtti -emit-llvm %s -o - -mconstructor-aliases -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fno-rtti -emit-llvm %s -o - -mconstructor-aliases -triple=i386-pc-win32 -fno-delete-null-pointer-checks | FileCheck %s

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
  // CHECK: %[[CHILD:.*]] = load %struct.ChildNoOverride
  child->left();
}

void ChildOverride::left() {}

void call_right_no_override(ChildNoOverride *child) {
  child->right();
  // When calling a right base's virtual method, one needs to adjust `this` at the caller site.
  //
  // CHECK: %[[CHILD_i8:.*]] = bitcast %struct.ChildNoOverride* %[[CHILD]] to i8*
  // CHECK: %[[RIGHT_i8:.*]] = getelementptr inbounds i8, i8* %[[CHILD_i8]], i32 4
  // CHECK: %[[RIGHT:.*]] = bitcast i8* %[[RIGHT_i8]] to %struct.Right*
  //
  // CHECK: %[[VFPTR:.*]] = bitcast %struct.Right* %[[RIGHT]] to void (%struct.Right*)***
  // CHECK: %[[VFTABLE:.*]] = load void (%struct.Right*)**, void (%struct.Right*)*** %[[VFPTR]]
  // CHECK: %[[VFUN:.*]] = getelementptr inbounds void (%struct.Right*)*, void (%struct.Right*)** %[[VFTABLE]], i64 0
}

void ChildOverride::right() {
  foo(this);
}

void call_right_override(ChildOverride *child) {
  child->right();
  // Ensure that `nonnull` and `dereferenceable(N)` are not emitted whether or not null is valid
  //
  // CHECK: %[[RIGHT:.*]] = getelementptr inbounds i8, i8* %[[CHILD_i8]], i32 4
  // CHECK: %[[VFUN_VALUE:.*]] = load void (i8*)*, void (i8*)** %[[VFUN]]
  // CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](i8* noundef %[[RIGHT]])
}
