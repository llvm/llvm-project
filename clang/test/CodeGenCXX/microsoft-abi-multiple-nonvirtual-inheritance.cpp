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
// CHECK-LABEL: define dso_local void @"\01?call_left_no_override
// CHECK: %[[CHILD:.*]] = load %struct.ChildNoOverride

  child->left();
// Only need to cast 'this' to Left*.
// CHECK: %[[LEFT:.*]] = bitcast %struct.ChildNoOverride* %[[CHILD]] to %struct.Left*
// CHECK: %[[VFPTR:.*]] = bitcast %struct.Left* %[[LEFT]] to void (%struct.Left*)***
// CHECK: %[[VFTABLE:.*]] = load void (%struct.Left*)**, void (%struct.Left*)*** %[[VFPTR]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds void (%struct.Left*)*, void (%struct.Left*)** %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load void (%struct.Left*)*, void (%struct.Left*)** %[[VFUN]]
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](%struct.Left* %[[LEFT]])
// CHECK: ret
}

void ChildOverride::left() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"\01?left@ChildOverride@@UAEXXZ"
// CHECK-SAME: (%struct.ChildOverride* %[[THIS:.*]])
//
// No need to adjust 'this' as the ChildOverride's layout begins with Left.
// CHECK: %[[THIS_ADDR:.*]] = alloca %struct.ChildOverride*, align 4
// CHECK: store %struct.ChildOverride* %[[THIS]], %struct.ChildOverride** %[[THIS_ADDR]], align 4

  foo(this);
// CHECK: %[[THIS:.*]] = load %struct.ChildOverride*, %struct.ChildOverride** %[[THIS_ADDR]]
// CHECK: %[[THIS_i8:.*]] = bitcast %struct.ChildOverride* %[[THIS]] to i8*
// CHECK: call void @foo(i8* %[[THIS_i8]])
// CHECK: ret
}

void call_left_override(ChildOverride *child) {
// CHECK-LABEL: define dso_local void @"\01?call_left_override
// CHECK: %[[CHILD:.*]] = load %struct.ChildOverride

  child->left();
// CHECK: %[[VFPTR:.*]] = bitcast %struct.ChildOverride* %[[CHILD]] to void (%struct.ChildOverride*)***
// CHECK: %[[VFTABLE:.*]] = load void (%struct.ChildOverride*)**, void (%struct.ChildOverride*)*** %[[VFPTR]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds void (%struct.ChildOverride*)*, void (%struct.ChildOverride*)** %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load void (%struct.ChildOverride*)*, void (%struct.ChildOverride*)** %[[VFUN]]
//
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](%struct.ChildOverride* %[[CHILD]])
// CHECK: ret
}

void call_right_no_override(ChildNoOverride *child) {
// CHECK-LABEL: define dso_local void @"\01?call_right_no_override
// CHECK: %[[CHILD:.*]] = load %struct.ChildNoOverride

  child->right();
// When calling a right base's virtual method, one needs to adjust 'this' at
// the caller site.
//
// CHECK: %[[CHILD_i8:.*]] = bitcast %struct.ChildNoOverride* %[[CHILD]] to i8*
// CHECK: %[[RIGHT_i8:.*]] = getelementptr inbounds i8, i8* %[[CHILD_i8]], i32 4
// CHECK: %[[RIGHT:.*]] = bitcast i8* %[[RIGHT_i8]] to %struct.Right*
//
// CHECK: %[[VFPTR:.*]] = bitcast %struct.Right* %[[RIGHT]] to void (%struct.Right*)***
// CHECK: %[[VFTABLE:.*]] = load void (%struct.Right*)**, void (%struct.Right*)*** %[[VFPTR]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds void (%struct.Right*)*, void (%struct.Right*)** %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load void (%struct.Right*)*, void (%struct.Right*)** %[[VFUN]]
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](%struct.Right* %[[RIGHT]])
// CHECK: ret
}

void ChildOverride::right() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"\01?right@ChildOverride@@UAEXXZ"(i8*
//
// ChildOverride::right gets 'this' cast to Right* in ECX (i.e. this+4) so we
// need to adjust 'this' before use.
//
// CHECK: %[[THIS_STORE:.*]] = alloca %struct.ChildOverride*, align 4
// CHECK: %[[THIS_ADDR:.*]] = alloca %struct.ChildOverride*, align 4
// CHECK: %[[COERCE_VAL:.*]] = bitcast i8* %[[ECX:.*]] to %struct.ChildOverride*
// CHECK: store %struct.ChildOverride* %[[COERCE_VAL]], %struct.ChildOverride** %[[THIS_STORE]], align 4
// CHECK: %[[THIS_INIT:.*]] = load %struct.ChildOverride*, %struct.ChildOverride** %[[THIS_STORE]], align 4
// CHECK: store %struct.ChildOverride* %[[THIS_INIT]], %struct.ChildOverride** %[[THIS_ADDR]], align 4
// CHECK: %[[THIS_RELOAD:.*]] = load %struct.ChildOverride*, %struct.ChildOverride** %[[THIS_ADDR]]
// CHECK: %[[THIS_i8:.*]] = bitcast %struct.ChildOverride* %[[THIS_RELOAD]] to i8*
// CHECK: %[[THIS_ADJUSTED:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 -4
// CHECK: %[[THIS:.*]] = bitcast i8* %[[THIS_ADJUSTED]] to %struct.ChildOverride*

  foo(this);
// CHECK: %[[THIS_PARAM:.*]] = bitcast %struct.ChildOverride* %[[THIS]] to i8*
// CHECK: call void @foo(i8* %[[THIS_PARAM]])
// CHECK: ret
}

void call_right_override(ChildOverride *child) {
// CHECK-LABEL: define dso_local void @"\01?call_right_override
// CHECK: %[[CHILD:.*]] = load %struct.ChildOverride

  child->right();
// When calling a right child's virtual method, one needs to adjust 'this' at
// the caller site.
//
// CHECK: %[[CHILD_i8:.*]] = bitcast %struct.ChildOverride* %[[CHILD]] to i8*
// CHECK: %[[RIGHT:.*]] = getelementptr inbounds i8, i8* %[[CHILD_i8]], i32 4
//
// CHECK: %[[CHILD_i8:.*]] = bitcast %struct.ChildOverride* %[[CHILD]] to i8*
// CHECK: %[[VFPTR_i8:.*]] = getelementptr inbounds i8, i8* %[[CHILD_i8]], i32 4
// CHECK: %[[VFPTR:.*]] = bitcast i8* %[[VFPTR_i8]] to void (i8*)***
// CHECK: %[[VFTABLE:.*]] = load void (i8*)**, void (i8*)*** %[[VFPTR]]
// CHECK: %[[VFUN:.*]] = getelementptr inbounds void (i8*)*, void (i8*)** %[[VFTABLE]], i64 0
// CHECK: %[[VFUN_VALUE:.*]] = load void (i8*)*, void (i8*)** %[[VFUN]]
//
// CHECK: call x86_thiscallcc void %[[VFUN_VALUE]](i8* %[[RIGHT]])
// CHECK: ret
}

struct GrandchildOverride : ChildOverride {
  virtual void right();
};

void GrandchildOverride::right() {
// CHECK-LABEL: define dso_local x86_thiscallcc void @"\01?right@GrandchildOverride@@UAEXXZ"(i8*
//
// CHECK: %[[THIS_STORE:.*]] = alloca %struct.GrandchildOverride*, align 4
// CHECK: %[[THIS_ADDR:.*]] = alloca %struct.GrandchildOverride*, align 4
// CHECK: %[[COERCE_VAL:.*]] = bitcast i8* %[[ECX:.*]] to %struct.GrandchildOverride*
// CHECK: store %struct.GrandchildOverride* %[[COERCE_VAL]], %struct.GrandchildOverride** %[[THIS_STORE]], align 4
// CHECK: %[[THIS_INIT:.*]] = load %struct.GrandchildOverride*, %struct.GrandchildOverride** %[[THIS_STORE]], align 4
// CHECK: store %struct.GrandchildOverride* %[[THIS_INIT]], %struct.GrandchildOverride** %[[THIS_ADDR]], align 4
// CHECK: %[[THIS_RELOAD:.*]] = load %struct.GrandchildOverride*, %struct.GrandchildOverride** %[[THIS_ADDR]]
// CHECK: %[[THIS_i8:.*]] = bitcast %struct.GrandchildOverride* %[[THIS_RELOAD]] to i8*
// CHECK: %[[THIS_ADJUSTED:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 -4
// CHECK: %[[THIS:.*]] = bitcast i8* %[[THIS_ADJUSTED]] to %struct.GrandchildOverride*

  foo(this);
// CHECK: %[[THIS_PARAM:.*]] = bitcast %struct.GrandchildOverride* %[[THIS]] to i8*
// CHECK: call void @foo(i8* %[[THIS_PARAM]])
// CHECK: ret
}

void call_grandchild_right(GrandchildOverride *obj) {
  // Just make sure we don't crash.
  obj->right();
}

void emit_ctors() {
  Left l;
  // CHECK-LABEL: define {{.*}} @"\01??0Left@@QAE@XZ"
  // CHECK-NOT: getelementptr
  // CHECK:   store i32 (...)** bitcast ({ [1 x i8*] }* @"\01??_7Left@@6B@" to i32 (...)**)
  // CHECK: ret

  Right r;
  // CHECK-LABEL: define {{.*}} @"\01??0Right@@QAE@XZ"
  // CHECK-NOT: getelementptr
  // CHECK:   store i32 (...)** bitcast ({ [1 x i8*] }* @"\01??_7Right@@6B@" to i32 (...)**)
  // CHECK: ret

  ChildOverride co;
  // CHECK-LABEL: define {{.*}} @"\01??0ChildOverride@@QAE@XZ"
  // CHECK:   %[[THIS:.*]] = load %struct.ChildOverride*, %struct.ChildOverride**
  // CHECK:   %[[VFPTR:.*]] = bitcast %struct.ChildOverride* %[[THIS]] to i32 (...)***
  // CHECK:   store i32 (...)** bitcast ({ [1 x i8*] }* @"\01??_7ChildOverride@@6BLeft@@@" to i32 (...)**), i32 (...)*** %[[VFPTR]]
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.ChildOverride* %[[THIS]] to i8*
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 4
  // CHECK:   %[[VFPTR:.*]] = bitcast i8* %[[VFPTR_i8]] to i32 (...)***
  // CHECK:   store i32 (...)** bitcast ({ [1 x i8*] }* @"\01??_7ChildOverride@@6BRight@@@" to i32 (...)**), i32 (...)*** %[[VFPTR]]
  // CHECK: ret

  GrandchildOverride gc;
  // CHECK-LABEL: define {{.*}} @"\01??0GrandchildOverride@@QAE@XZ"
  // CHECK:   %[[THIS:.*]] = load %struct.GrandchildOverride*, %struct.GrandchildOverride**
  // CHECK:   %[[VFPTR:.*]] = bitcast %struct.GrandchildOverride* %[[THIS]] to i32 (...)***
  // CHECK:   store i32 (...)** bitcast ({ [1 x i8*] }* @"\01??_7GrandchildOverride@@6BLeft@@@" to i32 (...)**), i32 (...)*** %[[VFPTR]]
  // CHECK:   %[[THIS_i8:.*]] = bitcast %struct.GrandchildOverride* %[[THIS]] to i8*
  // CHECK:   %[[VFPTR_i8:.*]] = getelementptr inbounds i8, i8* %[[THIS_i8]], i32 4
  // CHECK:   %[[VFPTR:.*]] = bitcast i8* %[[VFPTR_i8]] to i32 (...)***
  // CHECK:   store i32 (...)** bitcast ({ [1 x i8*] }* @"\01??_7GrandchildOverride@@6BRight@@@" to i32 (...)**), i32 (...)*** %[[VFPTR]]
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
  // CHECK-LABEL: define dso_local void @"\01?call_asymmetric_child_complete_dtor@@YAXXZ"
  AsymmetricChild obj;
  // CHECK: call x86_thiscallcc %struct.AsymmetricChild* @"\01??0AsymmetricChild@@QAE@XZ"(%struct.AsymmetricChild* %[[OBJ:.*]])
  // CHECK-NOT: getelementptr
  // CHECK: call x86_thiscallcc void @"\01??1AsymmetricChild@@UAE@XZ"(%struct.AsymmetricChild* %[[OBJ]])
  // CHECK: ret
}
